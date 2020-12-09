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


def run(symbols, ncal:int=10, npred:int=10, calibration_trialduration=4.2,  prediction_trialduration=20, stimfile=None, selectionThreshold:float=.1,
        framesperbit:int=1, optosensor:bool=True, fullscreen:bool=False, windowed:bool=None, 
        fullscreen_stimulus:bool=True, simple_calibration=False, host=None, calibration_symbols=None, bgFraction=.1,
        calibration_args:dict=None, prediction_args:dict=None, extra_symbols=None): 
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
    ss = selectionMatrix.ExptScreenManager(window, nt, symbols, nCal=ncal, nPred=npred, framesperbit=framesperbit, 
                        fullscreen_stimulus=fullscreen_stimulus, selectionThreshold=selectionThreshold, 
                        optosensor=optosensor, simple_calibration=simple_calibration, calibration_symbols=calibration_symbols, 
                        bgFraction=bgFraction, 
                        calibration_args=calibration_args, calibration_trialduration=calibration_trialduration, 
                        prediction_args=prediction_args, prediction_trialduration=prediction_trialduration)

    # override the selection grid with the tictactoe one
    ss.selectionGrid = VisualAcuityScreen(window=window, symbols=symbols, noisetag=nt, optosensor=optosensor, bgFraction=bgFraction)
    ss.calibrationSentence = 'Look at the *RED* cross'
    ss.calibrationInstruct = "Calibration\n\nThe next stage is CALIBRATION\nlook at the *RED* +\n try not to move your eyes\n ignore the flashing green cue\n\nkey to continue"

    # run the app
    selectionMatrix.run_screen(ss)


if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    setattr(args,'symbols','visual_acuity.txt')
    #setattr(args,'stimfile','visual_acuity.txt')
    setattr(args,'framesperbit',1)
    setattr(args,'bgFraction',0)
    setattr(args,'simple_calibration',False)
    run(**vars(args))
