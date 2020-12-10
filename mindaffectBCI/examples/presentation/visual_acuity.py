#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pyglet
import mindaffectBCI.examples.presentation.selectionMatrix as selectionMatrix
from mindaffectBCI.noisetag import Noisetag
from mindaffectBCI.decoder.utils import search_directories_for_file
import os

class VisualAcuityScreen(selectionMatrix.SelectionGridScreen):
    """variant of SelectionGridScreen which is optimized for visual acuity testing -- bascially it's **really** small
    """
    def __init__(self,*args, **kwargs):
        if 'gridfraction' in kwargs:
            self.gridfraction = kwargs['gridfraction']
            kwargs.pop('gridfraction',None) # remove from args
        super().__init__(*args,**kwargs)

    def init_symbols(self, symbols, x=0, y=0, w=None, h=None, bgFraction=0, font_size=32, gridfraction=.1):
        if w is None:
            w, h=self.window.get_size()
        # compute central bounding box
        sw = w*gridfraction # bb-width
        sh = h*gridfraction # bb-height
        sx = x + w/2 - sw/2 # left
        sy = y + h/2 - sh/2 # top

        # make the matrix
        selectionMatrix.SelectionGridScreen.init_symbols(self,symbols, x=sx, y=sy, w=sw, h=sh, 
                                                         bgFraction=bgFraction, font_size=0)
        
        # add a fixation point in the middle of the grid
        self.fixation = self.init_label("+",x=sx,y=sy,w=sw,h=sh,font_size=30)
        self.fixation.color = (255,0,0,255)


def run(symbols, host:str='-', optosensor:bool=True, bgFraction=.1, gridfraction=.1, stimfile:str=None, fullscreen=False, windowed=False, fullscreen_stimulus=True, **kwargs):
    """ run the selection Matrix with default settings

    Args:
        nCal (int, optional): number of calibration trials. Defaults to 10.
        nPred (int, optional): number of prediction trials at a time. Defaults to 10.
        simple_calibration (bool, optional): flag if we show only a single target during calibration, Defaults to False.
        stimFile ([type], optional): the stimulus file to use for the codes. Defaults to None.
        framesperbit (int, optional): number of video frames per stimulus codebit. Defaults to 1.
        fullscreen (bool, optional): flag if should runn full-screen. Defaults to False.
        fullscreen_stimulus (bool, optional): flag if should run the stimulus (i.e. flicker) in fullscreen mode. Defaults to True.
        simple_calibration (bool, optional): flag if we only show the *target* during calibration.  Defaults to False
        calibration_trialduration (float, optional): flicker duration for the calibration trials. Defaults to 4.2.
        prediction_trialduration (float, optional): flicker duration for the prediction trials.  Defaults to 10.
        calibration_args (dict, optional): additional keyword arguments to pass to `noisetag.startCalibration`. Defaults to None.
        prediction_args (dict, optional): additional keyword arguments to pass to `noisetag.startPrediction`. Defaults to None.
        gridfraction (float,optional): fraction of the symbols area of the screen to use for the stimuli, you should set this such that the stimuli span about 7-deg of visual angle for your participant.  Defaults to .1
    """
    if stimfile is None:
        stimfile = 'mgold_61_6521_psk_60hz.txt'
    if fullscreen is None and windowed is not None:
        fullscreen = not windowed
    if windowed == True or fullscreen == True:
        fullscreen_stimulus = False
    nt=Noisetag(stimFile=stimfile,clientid='Presentation:selectionMatrix')
    if host is not None and not host in ('','-'):
        nt.connect(host, queryifhostnotfound=False)

    # init the graphics system
    window = selectionMatrix.initPyglet(fullscreen=fullscreen)

    # make the screen manager object which manages the app state
    ss = selectionMatrix.ExptScreenManager(window, nt, symbols, 
                        fullscreen_stimulus=fullscreen_stimulus, 
                        optosensor=optosensor,  
                        bgFraction=bgFraction, **kwargs)
    # override the selection grid with the tictactoe one
    ss.selectionGrid = VisualAcuityScreen(window=window, symbols=symbols, noisetag=nt, optosensor=optosensor, bgFraction=bgFraction, gridfraction=gridfraction)
    ss.calibrationSentence = 'Look at the *RED* cross'
    ss.calibrationInstruct = "Calibration\n\nThe next stage is CALIBRATION\nlook at the *RED* +\n try not to move your eyes\n ignore the flashing green cue\n\nkey to continue"
    ss.cuedpredictionInstruct = "Testing\n\nFocus on the *RED* +\n try not to move your eyes.\nignore the flashing green\n\nkey to continue"
    ss.predictionSentence = "Testing\n\nFocus on the *RED* +"

    # run the app
    selectionMatrix.run_screen(ss)


if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    setattr(args,'calibration_symbols',[["."]])
    setattr(args,'symbols','visual_acuity.txt')
    #setattr(args,'stimfile','visual_acuity.txt')
    setattr(args,'framesperbit',1)
    setattr(args,'bgFraction',0)
    setattr(args,'simple_calibration',False)
    run(**vars(args))
