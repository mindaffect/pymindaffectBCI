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
    def init_target(self, symb, x, y, w, h):
        # make list of sprites for the different state-dependent images
        symbs = symb.split("|")
        sprite = [self.init_sprite(symb,x,y,w,h) for symb in symbs]

        # get the label
        symb = symbs[0] if len(symbs)>1 else symb
        label= self.init_label(symb,x,y,w,h)

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
                        optosensor=optosensor, simple_calibration=True, calibration_symbols=calibration_symbols, 
                        bgFraction=bgFraction, 
                        calibration_args=calibration_args, calibration_trialduration=calibration_trialduration, 
                        prediction_args=prediction_args, prediction_trialduration=prediction_trialduration)

    # override the selection grid with the tictactoe one
    ss.selectionGrid = ImageFlashScreen(window=window, symbols=symbols, noisetag=nt, optosensor=optosensor)

    # run the app
    selectionMatrix.run_screen(ss)


if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    setattr(args,'symbols','visual_acuity.txt')
    #setattr(args,'stimfile','visual_acuity.txt')
    setattr(args,'framesperbit',4)
    run(**vars(args))
