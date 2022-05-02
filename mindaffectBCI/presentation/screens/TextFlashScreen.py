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

import pyglet
from mindaffectBCI.presentation.screens.SelectionGridScreen import SelectionGridScreen
from mindaffectBCI.noisetag import Noisetag
from mindaffectBCI.decoder.utils import search_directories_for_file
import os

class TextFlashScreen(SelectionGridScreen):
    """variant of SelectionGridScreen which changes the text on 'flash' rather than luminosity
    """  
    def __init__(self, window, noisetag, symbols, scale_to_fit:bool=True, color_labels:bool=True, **kwargs):
        self.scale_to_fit = scale_to_fit
        self.color_labels = color_labels
        super().__init__(window,noisetag,symbols=symbols,**kwargs)

    def init_target(self, symb, x, y, w, h, i, j, font_size:int=None):
        # make list of strings for the different state-dependent strings
        symbs = symb.split("|")
        sprite = [self.init_label(symb,x,y,w,h,font_size) for symb in symbs]

        # get the label
        symb = symbs[0] if len(symbs)>1 else symb
        label= self.init_label(symb,x,y,w,h,1)

        return sprite, label

    def update_object_state(self, idx:int, state):
        if self.objects[idx]:
            # make text black when "off", white otherwise.. 
            for sprite in self.objects[idx]:
                sprite.color = (0,0,0,0)
            img_idx = state if len(self.objects[idx])>1 and state<len(self.objects[idx]) else 0
            self.objects[idx][img_idx].color = (255,255,255,255)
            #self.objects[idx][img_idx].color = self.state2color[state]
        if self.labels[idx]:
            col = self.state2color.get(state,(255,255,255,255)) if self.color_labels else (255,255,255,255)
            col = tuple(col)
            self.labels[idx].color=  col if len(col)==4 else col+(255,) #(255,255,255,255) # reset labels

if __name__ == "__main__":
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    window = initPyglet(width=640, height=480)
    nt = Noisetag(stimSeq='level4_gold.txt', utopiaController=None)

    symbols=[['o1.e0|o1.e1|o1.e2|o1.e3|o1.e4','o2.e0|o2.e1|o2.e2|o2.e3|o2.e4']]
    screen = TextFlashScreen(window, noisetag=nt, symbols=symbols)

    nt.startFlicker(framesperbit=6, numframes=60*60)

    run_screen(window, screen)
