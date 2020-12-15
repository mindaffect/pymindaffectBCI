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

class ImageFlashScreen(selectionMatrix.SelectionGridScreen):
    """variant of SelectionGridScreen which changes the background image on 'flash' rather than luminosity
    """    
    def init_target(self, symb, x, y, w, h, font_size:int=None):
        # make list of sprites for the different state-dependent images
        symbs = symb.split("|")
        sprite = [self.init_sprite(symb,x,y,w,h)[0] for symb in symbs]

        # get the label
        symb = symbs[0] if len(symbs)>1 else symb
        label= self.init_label(symb,x,y,w,h,font_size)

        return sprite, label

    def update_object_state(self, idx:int, state):
        if self.objects[idx]:
            # turn all sprites off, non-colored
            for sprite in self.objects[idx]:
                sprite.visible = False
            img_idx = state if len(self.objects[idx])>1 and state<len(self.objects[idx]) else 0
            self.objects[idx][img_idx].visible = True
            self.objects[idx][img_idx].color = self.state2color[state]
        if self.labels[idx]:
            self.labels[idx].color=(255,255,255,255) # reset labels


def run(symbols, host:str='-', optosensor:bool=True, bgFraction:float=.1, gridfraction:float=1, stimfile:str=None, fullscreen=False, windowed=False, fullscreen_stimulus=True, transform:str='color', fixation:bool=False, **kwargs):
    """
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
    # override the selection grid with the image_flash one
    ss.selectionGrid = ImageFlashScreen(window=window, symbols=symbols, noisetag=nt, optosensor=optosensor)

    # run the app
    selectionMatrix.run_screen(ss)


if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    setattr(args,'symbols','rc5x5_faces.txt')
    setattr(args,'stimfile','rc5x5.txt')
    setattr(args,'framesperbit',4)
    run(**vars(args))
