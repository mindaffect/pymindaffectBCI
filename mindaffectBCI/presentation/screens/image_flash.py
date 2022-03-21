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


class CheckerboardGridScreen(selectionMatrix.SelectionGridScreen):
    """variant of SelectionGridScreen which uses a checkerboard behind each target
    """  
    def __init__(self, window, noisetag, symbols=None, nchecks:int=3, **kwargs):
        self.nchecks = nchecks if hasattr(nchecks,'__iter__') else (nchecks,nchecks)
        super().__init__(window,noisetag,symbols=symbols,**kwargs)

    # mapping from bci-stimulus-states to display color
    state2color={0:(5, 5, 5, 0),     # off=invisible
                 1:(160, 160, 160),  # on=white
                 2:(96, 96, 96),     # invert
                 254:(0,255,0),      # cue=green
                 255:(0,0,255),      # feedback=blue
                 None:(100,0,0)}     # red(ish)=task     
    def init_target(self, symb, x, y, w, h, i, j, font_size:int=None):
        # make the background
        bg = Checkerboard(x,y,w,h,nx=self.nchecks[0],ny=self.nchecks[1],batch=self.batch,group=self.background)
        # make the label
        label= self.init_label(symb,x,y,w,h,font_size)
        return bg, label


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



class CheckerboardStackScreen(CheckerboardGridScreen):
    """variant of SelectionGridScreen uses a stack of checkerboards
    """  
    def __init__(self, window, noisetag, symbols=None, artificial_deficit_id=None, scale_to_fit:bool=True, color_labels:bool=True, stimulus_weight=None, **kwargs):
        self.scale_to_fit, self.color_labels, self.artificial_deficit_id, self.stimulus_weight, self.prev_stimulus_state = \
            (scale_to_fit, color_labels, artificial_deficit_id, stimulus_weight, None)
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
        x =  x + bgFraction*w/2
        y =  y + bgFraction*h/2
        w =  w - bgFraction*w
        h =  h - bgFraction*h
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
        if isinstance(symb,str): symb=symb.split("|")
        lab=symb.pop(0)
        nx, ny = (symb[0],1) if len(symb)==1 else symb[:2] 
        # make the background
        bg = Checkerboard(x,y,w,h,nx=int(nx),ny=int(ny),batch=self.batch,group=self.background)
        # make the label
        label= self.init_label(lab,x,y,w,h,font_size)
        return bg, label



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


class SelectionWheelScreen(selectionMatrix.SelectionGridScreen):
    """variant of SelectionGridScreen lays the grid out on a series of concentric rings
    """  

    def __init__(self, window, noisetag, symbols=None, ring_radii=None, ischeckerboard:bool=False, **kwargs):
        self.ring_radii, self.ischeckerboard = ring_radii, ischeckerboard
        super().__init__(window,noisetag,symbols=symbols,**kwargs)

    # mapping from bci-stimulus-states to display color
    state2color={0:(5, 5, 5, 0),     # off=invisible
                 1:(160, 160, 160),  # on=white
                 2:(96, 96, 96),     # invert
                 254:(0,255,0),      # cue=green
                 255:(0,0,255),      # feedback=blue
                 None:(100,0,0)}     # red(ish)=task     
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

        # TODO: Include a background fraction
        cx, cy = ( (x+w)/2, (y+h)/2 )
        nrings = len(symbols)

        # equally spaced rings
        if self.ring_radii is None:
            ringradius = min(w,h)/(nrings+1)/2
            ring_radii = [(i+1)*ringradius for i in range(nrings+1)]
        else:
            # use the given ring radii
            ring_radii = [ r*min(w,h)/2 for r in self.ring_radii ]


        # now create the display objects
        idx=-1
        for i in range(len(symbols)): # radius
            dr = ring_radii[i+1]-ring_radii[i]
            radius = ring_radii[i]
            dtheta = math.pi*2 / max(1,len(symbols[i]))
            for j in range(len(symbols[i])): # theta
                if symbols[i][j] is None or symbols[i][j]=="": continue
                theta = j * dtheta
                idx = idx+1
                symb = symbols[i][j]
                self.objects[idx], self.labels[idx] = self.init_target(symb, 
                                                       cx, cy, 
                                                       theta + bgFraction * dtheta/2, 
                                                       radius + bgFraction * dr/2, 
                                                       dtheta - bgFraction * dtheta/2, 
                                                       dr - bgFraction * dr/2,
                                                       font_size=font_size)

    def init_target(self, symb, cx, cy, theta, radius, dtheta, dr, font_size:int=None):
        # make the background
        if self.ischeckerboard:
            nx, ny = (3,3) if self.ischeckerboard == True else (self.ischeckerboard,self.ischeckerboard)
            seg = CheckerboardSegment(cx,cy,theta,radius,dtheta,dr,nx=nx,ny=ny,batch=self.batch,group=self.foreground)
        else:
            seg = PieSegment(cx,cy,theta,radius,dtheta,dr,batch=self.batch,group=self.foreground)

        # make the label
        centx, centy = polar2cart(cx,cy,theta+dtheta/2, radius+dr/2)
        lb, tr = (polar2cart(cx,cy,theta,radius), polar2cart(cx,cy,theta+dtheta,radius+dr))
        w = max(abs(lb[0]-tr[0]), abs(lb[1]-tr[1]))
        label= self.init_label(symb,centx-w/2,centy-w/2,w,w,font_size)

        return seg, label

    # def draw(self,t):
    #     super().draw(t)
        # self.window.clear()
        # for o in self.objects:
        #     o.draw()
        #     pass


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class ImageFlashScreen(selectionMatrix.SelectionGridScreen):
    """variant of SelectionGridScreen which changes the background image on 'flash' rather than luminosity
    """  

    def __init__(self, window, noisetag, symbols=None, scale_to_fit:bool=True, color_labels:bool=True, **kwargs):
        self.scale_to_fit = scale_to_fit
        self.color_labels = color_labels
        super().__init__(window,noisetag,symbols=symbols,**kwargs)

    def init_target(self, symb, x, y, w, h, i, j, font_size:int=None):
        # make list of sprites for the different state-dependent images
        symbs = symb.split("|")
        sprite = [self.init_sprite(symb,x,y,w,h,scale_to_fit=self.scale_to_fit)[0] for symb in symbs]

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
            #self.objects[idx][img_idx].color = self.state2color[state]
        if self.labels[idx]:
            col = self.state2color.get(state,(255,255,255,255)) if self.color_labels else (255,255,255,255)
            self.labels[idx].color=  col if len(col)==4 else col+(255,) #(255,255,255,255) # reset labels


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class ImageStackScreen(selectionMatrix.SelectionGridScreen):
    """variant of SelectionGridScreen which changes the background image on 'flash' rather than luminosity
    """  
    def __init__(self, window, noisetag, symbols=None, artificial_deficit_id=None, scale_to_fit:bool=True, color_labels:bool=True, stimulus_weight=None, **kwargs):
        self.scale_to_fit, self.color_labels, self.artificial_deficit_id, self.stimulus_weight, self.prev_stimulus_state = \
            (scale_to_fit, color_labels, artificial_deficit_id, stimulus_weight, None)
        super().__init__(window,noisetag,symbols=symbols,**kwargs)
        self.vectorX = 0
        self.vectorY = 0
        self.finalX=0
        self.finalY=0

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
        # make list of sprites for the different state-dependent images
        symbs = symb.split("|")
        sprite = [self.init_sprite(symb,x,y,w,h,scale_to_fit=self.scale_to_fit)[0] for symb in symbs]

        # get the label
        symb = symbs[0] if len(symbs)>1 else symb
        label= self.init_label(symb,x,y,w,h,font_size)

        return sprite, label

    def update_object_state(self, idx:int, state):
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


class ImageStackScreenMovingFixation(ImageStackScreen):
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
