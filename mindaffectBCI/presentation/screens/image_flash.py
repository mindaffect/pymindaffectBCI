#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jadref@gmail.com>
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

from tkinter.constants import SE
import pyglet
import mindaffectBCI.presentation.selectionMatrix as selectionMatrix
from mindaffectBCI.noisetag import Noisetag
from mindaffectBCI.decoder.utils import search_directories_for_file
from mindaffectBCI.presentation.screens.visual_stimuli import Checkerboard, CheckerboardSegment, Rectangle, TriangleStrip
import os
import numpy as np
import math
import random as rnd

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class ImageFlashScreen(SelectionGridScreen):
    def __init__(self, window, noisetag, symbols=None, scale_to_fit:bool=True, color_labels:bool=True, **kwargs):
        """variant of SelectionGridScreen which changes the background image on 'flash' rather than luminosity

        Args:
            window (_type_): the window the show the display in 
            noisetag (_type_): the noisetag object to communicate with the EEG/BCI system
            symbols (list-of-lists, optional): the grid layout, as a list of lists for rows/cols. Defaults to None.
            scale_to_fit (bool,optional): scale the image to fix the size of the target.  Defaults to True.
            color_labels (bool, optional): if true, then color the label as well as the background on state change.
            ischeckerboard (bool, optional): _description_. Defaults to False.
        """        
        self.scale_to_fit = scale_to_fit
        self.color_labels = color_labels
        super().__init__(window,noisetag,symbols=symbols,**kwargs)

    def init_target(self, symb, x, y, w, h, i, j, font_size:int=None):
        """create a display sector target (i.e. pie wedge)

        Args:
            symb (str): the filename of the image to show as the background for this target
            x,y,w,h (float): bounding box for the target
            i,j (float): position of the target in the symbols grid, i.e. row + col
            font_size (int, optional): size of the font to use for the sector label. Defaults to None.

        Returns:
            _type_: _description_
        """
        # make list of sprites for the different state-dependent images
        symbs = symb.split("|")
        sprite = [self.init_sprite(symb,x,y,w,h,scale_to_fit=self.scale_to_fit)[0] for symb in symbs]

        # get the label
        symb = symbs[0] if len(symbs)>1 else symb
        label= self.init_label(symb,x,y,w,h,font_size)

        return sprite, label

    def update_object_state(self, idx:int, state):
        """ update the state of an stimulus object given it's index and the new state

        Here, the state is used to select which background image to show for this target.

        Args:
            idx (int): idx of the stimulus object to update in the symbols set
            state (int): the new state for this stimulus object
        """
        if self.objects[idx]:
            # turn all sprites off, non-colored
            for sprite in self.objects[idx]:
                sprite.visible = False
            img_idx = state if len(self.objects[idx])>1 and state<len(self.objects[idx]) else 0
            self.objects[idx][img_idx].visible = True
            #self.objects[idx][img_idx].color = self.state2color[state]

        if self.labels[idx]:
            col = self.state2color.get(state,(255,255,255,255)) if self.color_labels else (255,255,255,255)
            self.labels[idx].color=  col if len(col)==4 else col+(255,) #(255,255,255,255) # reset labels


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class ImageStackScreen(SelectionGridScreen):
    def __init__(self, window, noisetag, symbols=None, scale_to_fit:bool=True, color_labels:bool=True, **kwargs):
        """variant of SelectionGridScreen which has a 'stack' of images *on top of each other* rather than a 'grid' of options.
        i.e. in this screen all elements in the selection take the full grid display width.

        Args:
            window (_type_): the window the show the display in 
            noisetag (_type_): the noisetag object to communicate with the EEG/BCI system
            symbols (list-of-lists, optional): the grid layout, as a list of lists for rows/cols. Defaults to None.
            scale_to_fit (bool,optional): scale the image to fix the size of the target.  Defaults to True.
            color_labels (bool, optional): if true, then color the label as well as the background on state change.
            ischeckerboard (bool, optional): _description_. Defaults to False.
        """        
        self.scale_to_fit, self.color_labels = scale_to_fit, color_labels
        super().__init__(window,noisetag,symbols=symbols,**kwargs)

    def init_symbols(self, symbols, x, y, w, h, bgFraction:float=.1, font_size:int=None):
        """setup the display for the given symbols set

        Args:
            symbols ([type]): the symbols to show, inside the given display box
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            bgFraction (float, optional): fraction of empty space between objects. Defaults to .1.
            font_size (int, optional): label font size. Defaults to None.
        """        
        # now create the display objects
        idx=-1
        for i in range(len(symbols)): # rows	 
            for j in range(len(symbols[i])): # cols
                if symbols[i][j] is None or symbols[i][j]=="": continue
                idx = idx+1
                symb = symbols[i][j]
                self.objects[idx], self.labels[idx] = self.init_target(symb, 
                                                       x, y, 
                                                       w, h,
                                                       i, j,
                                                       font_size=font_size)


    def init_target(self, symb, x, y, w, h, i, j, font_size:int=None):
        """create a display sector target
        In this case, we make a stack of images from the symbols as a file-name all with the full size of the grid region

        Args:
            symb (str): the filename of the image(s) to show as the background for this target.  Multiple images are given separated with '|' characters, e.g. 'duck.png|cat.png'.  When multiple images are given then the 1 image will be shown in state '1', and the 2nd in state '2' etc.
            x,y,w,h (float): bounding box for the target
            i,j (float): position of the target in the symbols grid, i.e. row + col
            font_size (int, optional): size of the font to use for the sector label. Defaults to None.

        Returns:
            _type_: _description_
        """
        symbs = symb.split("|")
        sprite = [self.init_sprite(symb,x,y,w,h,scale_to_fit=self.scale_to_fit)[0] for symb in symbs]

        # get the label
        symb = symbs[0] if len(symbs)>1 else symb
        label= self.init_label(symb,x,y,w,h,font_size)

        return sprite, label

    def update_object_state(self, idx:int, state):
        """ update the state of an stimulus object given it's index and the new state

        Here, the state is used to select which background image to show for this target.

        Note: this varient has been modifed to *not* update the 'artificial deficit' entries..

        Args:
            idx (int): idx of the stimulus object to update in the symbols set
            state (int): the new state for this stimulus object
        """
        if self.objects[idx]:
            # turn all sprites off, non-colored
            for sprite in self.objects[idx]:
                sprite.visible = False
            if idx == self.artificial_deficit_id and state==1:
                # TODO[X]: use a randomly choosen state!
                state = 0 # np.random.randint(0,1)  # don't flicker it!
            img_idx = min(state+1,len(self.objects[idx])-1) if len(self.objects[idx])>1 else 0
            self.objects[idx][img_idx].visible = True
            #self.objects[idx][img_idx].color = self.state2color[state]
        if self.labels[idx]:
            col = self.state2color.get(state,(255,255,255,255)) if self.color_labels else (255,255,255,255)
            self.labels[idx].color =  col if len(col)==4 else col+(255,) #(255,255,255,255) # reset labels


    # def draw(self, t):
    #     # modify the signal injection to only apply when an object state changes
    #     # record the old stimulus state, only inject if it's different
    #     if self.stimulus_weight is not None and not self.stimulus_state == self.prev_stimulus_state :
    #         if self.prev_stimulus_state:
    #             # inject only on rising edge for given output
    #             self.injectSignal = sum(max(s-ps,0)*w for ps,s,w in zip(self.prev_stimulus_state, self.stimulus_state, self.stimulus_weight))
    #         else:
    #             self.injectSignal = sum(max(s,0)*w for s,w in zip(self.stimulus_state, self.stimulus_weight))
    #         #print("{} ->\n{}\nInject: {}".format(self.prev_stimulus_state, self.stimulus_state, self.injectSignal))
    #     else:
    #         self.injectSignal =None 
    #     self.prev_stimulus_state = self.stimulus_state
    #     super().draw(t)



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class ImageStackScreenMovingFixation(ImageStackScreen):
    def __init__(self, window, noisetag, symbols=None, artificial_deficit_id=None, scale_to_fit:bool=True, color_labels:bool=True, stimulus_weight=None, **kwargs):
        self.scale_to_fit, self.color_labels, self.artificial_deficit_id, self.stimulus_weight, self.prev_stimulus_state = \
            (scale_to_fit, color_labels, artificial_deficit_id, stimulus_weight, None)
        """variant of the ImageStackScreen which in addition moves the fixation point around during display

        Args:
            window (_type_): the window the show the display in 
            noisetag (_type_): the noisetag object to communicate with the EEG/BCI system
            symbols (list-of-lists, optional): the grid layout, as a list of lists for rows/cols. Defaults to None.
            scale_to_fit (bool,optional): scale the image to fix the size of the target.  Defaults to True.
            color_labels (bool, optional): if true, then color the label as well as the background on state change.
            ischeckerboard (bool, optional): _description_. Defaults to False.
        """        
        super().__init__(window,noisetag,symbols=symbols,**kwargs)
        self.vectorX = 0
        self.vectorY = 0
        self.finalX=0
        self.finalY=0

        def do_draw(self, speed = 0.1):
            stimulus_state, target_idx, objIDs, sendEvents=self.noisetag.getStimulusState()
            winw, winh=self.window.get_size()
            if self.finalX == 0 or int(self.fixation_obj.x) == int(self.finalX):
                self.finalX = winw/2 + rnd.randint(-5,5)
                self.vectorX = (np.sign(int(self.finalX) - int(self.fixation_obj.x))) * speed

            if self.finalY == 0 or int(self.fixation_obj.y) == int(self.finalY):
                self.finalY = 0.9*winh/2 +rnd.randint(-5,5)
                self.vectorY = (np.sign(int(self.finalY) - int(self.fixation_obj.y))) * speed

        # just draw the batch/background
            self.batch.draw()
            if self.logo: self.logo.draw()
            if self.fixation_obj: 
                self.fixation_obj.x +=self.vectorX
                self.fixation_obj.y += self.vectorY
                self.fixation_obj.draw()
            self.frameend=self.getTimeStamp()

            print("final X, final Y , X , Y, vectorX, vectorY",int(self.finalX), int(self.finalY), int(self.fixation_obj.x), int(self.fixation_obj.y),self.vectorX,self.vectorY)
               #self.finalX = rnd.randint(-1,1)
             # add the frame rate info
             # TODO[]: limit update rate..
            if self.framerate_display:
                from mindaffectBCI.presentation.selectionMatrix import flipstats
                flipstats.update_statistics()
                self.set_framerate("{:4.1f} +/-{:4.1f}ms".format(flipstats.median,flipstats.sigma))                



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    #setattr(args,'calibration_screen','mindaffectBCI.examples.presentation.image_flash.ImageStackScreen')
    #setattr(args,'symbols','mfvaflash.txt')
    setattr(args,'symbols',[['1','2','3','4','5','6','7','8','9','10','11','12'],
                            ['1','2','3','4','5','6','7','8','9','10','11','12']])
    setattr(args,'stimfile','mgold_65_6532.txt')
    setattr(args,'framesperbit',1)
    setattr(args,'calibration_screen','mindaffectBCI.examples.presentation.image_flash.SelectionWheelScreen')
    setattr(args,'calibration_args',dict(startframe='random',permute=True))
    setattr(args,'prediction_args',dict(startframe='random',permute=True)) # rand start point, and shuffle each repeat

    # 300s prediction data = 5min
    setattr(args,'prediction_trialduration',10)
    setattr(args,'npred',30)

    wght = np.exp( -.5* (np.arange(60)-30)**2 / 15) # gaussian blob
    wght = 1-wght # make it a blind-spot
    wght = wght * 50 / np.sum(wght)
    setattr(args,'calibration_screen_args', dict(ring_radii=[0,.1,.2,.3], fixation=True, target_only=True, ischeckerboard=True))
    #setattr(args,'calibration_screen_args', dict(stimulus_weight=wght, font_size=0, fixation="dartboardflash/cross.png", artificial_deficit_id=5, target_only=True))

    selectionMatrix.run(**vars(args))
