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
import os

class VisualAcuityScreen(selectionMatrix.SelectionGridScreen):
    """variant of SelectionGridScreen which is optimized for visual acuity testing -- bascially it's **really** small
    """
    def __init__(self,*args, **kwargs):
        self.gridfraction=1
        self.transform = 'color'
        self.fixation = False
        # remove the args we set directly
        for arg in ('gridfraction', 'transform', 'fixation'):
            if arg in kwargs:
                setattr(self,arg,kwargs[arg])
                kwargs.pop(arg,None) # remove from args
        if not self.transform == 'color':
            # BODGE: tweak the background color so white font shows....
            self.state2color[0]=(128,128,128)
        # pass the rest on
        super().__init__(*args,**kwargs)

    def init_symbols(self, symbols, x=0, y=0, w=None, h=None, bgFraction:float=0, font_size:int=None):
        if w is None:
            w, h=self.window.get_size()
        # compute central bounding box
        sw = w*self.gridfraction # bb-width
        sh = h*self.gridfraction # bb-height
        sx = x + w/2 - sw/2 # left
        sy = y + h/2 - sh/2 # top

        # make the matrix
        selectionMatrix.SelectionGridScreen.init_symbols(self,symbols, x=sx, y=sy, w=sw, h=sh, 
                                                         bgFraction=bgFraction, font_size=0 if self.fixation else font_size)
        
        # add a fixation point in the middle of the grid
        if self.fixation:
            self.fixation = self.init_label("+",x=sx,y=sy,w=sw,h=sh,font_size=font_size)
            self.fixation.color = (255,0,0,255)

    # mappings for stimulus-state -> stimulus state
    state2scale = { 0:1, 1:.7, 2:1.5, 3:1.5 } # bigger for cue, smaller for stim
    state2rotation = { 0:0, 1:90, 2:45, 3:45 } # slant for cue, flip for stim

    # override the object drawing code for different types of stimulus change
    def update_object_state(self, idx:int, state):
        """update the idx'th object to stimulus state state

            N.B. override this class to implement different state dependent stimulus changes, e.g. size changes, background images, rotations
        Args:
            idx (int): index of the object to update
            state (int or float): the new desired object state
        """        
        if self.objects[idx]:
            if self.transform == 'color':
                if isinstance(state,int): # integer state, map to color lookup table
                    self.objects[idx].color = self.state2color[state]
                elif isinstance(state,float): # float state, map to intensity
                    self.objects[idx].color = tuple(int(c*state) for c in self.state2color[1])
            elif self.transform == 'scale':
                self.objects[idx].color = self.state2color[0]
                if isinstance(state,int): # integer state, map to color lookup table
                    self.objects[idx].scale = self.state2scale[state]
                elif isinstance(state,float): # float state, map to intensity
                    self.objects[idx].scale = .5+state if state < 1.0 else state/128
            elif self.transform == 'rotation':
                self.objects[idx].color = self.state2color[0]
                if isinstance(state,int): # integer state, map to color lookup table
                    self.objects[idx].rotation = self.state2rotation[state]
                elif isinstance(state,float): # float state, map to intensity
                    self.objects[idx].rotation = state*90

        if self.labels[idx]:
            self.labels[idx].color=(255,255,255,255) # reset labels


def run(symbols, host:str='-', optosensor:bool=True, bgFraction:float=.1, gridfraction:float=1, 
        fullscreen=False, windowed=False, fullscreen_stimulus=True, 
        transform:str='color', fixation:bool=False, **kwargs):
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
    if fullscreen is None and windowed is not None:
        fullscreen = not windowed
    if windowed == True or fullscreen == True:
        fullscreen_stimulus = False
    nt=Noisetag(clientid='Presentation:selectionMatrix')
    if host is not None and not host in ('','-'):
        nt.connect(host, queryifhostnotfound=False)

    # init the graphics system
    window = selectionMatrix.initPyglet(fullscreen=fullscreen)

    # make the screen manager object which manages the app state
    ss = selectionMatrix.ExptScreenManager(window, nt, symbols, 
                        fullscreen_stimulus=fullscreen_stimulus, 
                        optosensor=optosensor,  
                        bgFraction=bgFraction, **kwargs)
    # override the selection grid with the visual acuity screen one
    ss.selectionGrid = VisualAcuityScreen(window=window, symbols=symbols, noisetag=nt, optosensor=optosensor, bgFraction=bgFraction, gridfraction=gridfraction, transform=transform, fixation=fixation)
    if fixation:
        ss.calibrationSentence = 'Look at the *RED* cross'
        ss.calibrationInstruct = "Calibration\n\nThe next stage is CALIBRATION\nlook at the *RED* +\n try not to move your eyes\n ignore the flashing green cue\n\nkey to continue"
        ss.cuedpredictionInstruct = "Testing\n\nFocus on the *RED* +\n try not to move your eyes.\nignore the flashing green\n\nkey to continue"
        ss.predictionSentence = "Testing\n\nFocus on the *RED* +"

    # run the app
    selectionMatrix.run_screen(ss)


if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    #setattr(args,'calibration_symbols',[["."]])
    #setattr(args,'symbols',[["."]])
    #setattr(args,'symbols','visual_acuity.txt')
    setattr(args,'symbols','symbols.txt')
    setattr(args,'calibration_stimseq','rc5x5.txt')
    setattr(args,'stimfile','level11_cont.txt')
    setattr(args,'framesperbit',1)
    setattr(args,'bgFraction',.2)
    setattr(args,'simple_calibration',False)
    setattr(args,'calibration_screen','mindaffectBCI.examples.presentation.visual_acuity.VisualAcuityScreen')
    setattr(args,'calscreen_args',dict(transform='rotation',fixation=False))
    selectionMatrix.run(**vars(args))
