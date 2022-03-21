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
from mindaffectBCI.presentation.screens.CheckerboardGridScreen import CheckerboardGridScreen
from mindaffectBCI.presentation.screens.visual_stimuli import Checkerboard

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class CheckerboardStackScreen(CheckerboardGridScreen):
    """variant of SelectionGridScreen uses a stack of checkerboards placed on top of each other
    """  
    def __init__(self, window, noisetag, symbols=None, artificial_deficit_id=None, scale_to_fit:bool=True, color_labels:bool=True, stimulus_weight=None, **kwargs):
        self.scale_to_fit, self.color_labels, self.artificial_deficit_id, self.stimulus_weight, self.prev_stimulus_state = \
            (scale_to_fit, color_labels, artificial_deficit_id, stimulus_weight, None)
        super().__init__(window,noisetag,symbols=symbols,**kwargs)

    def init_symbols(self, symbols, x, y, w, h, bgFraction:float=.1, font_size:int=None):
        """setup the display for the given symbols set

        Args:
            symbols (list-of-list-of-str): the set of symbols to show, inside the given display box
            x,y,w,h (float): bounding box for the whole grid of symbols
            font_size (int, optional): fontsize for the text. Defaults to None.
            bgFraction (float, optional): fraction of empty space between objects. Defaults to .1.
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
        """setup a checkerboard background target

        Args:
            symb (_type_): the symbol to display, e.g. the letter/image/symbol
            x,y,w,h (float): bounding box for the target
            i,j (float): position of the target in the symbols grid, i.e. row + col
            font_size (int, optional): fontsize for the text. Defaults to None.

        Returns:
            background,foreground: visual objects for the background (check) and foreground (letter)
        """
        if isinstance(symb,str): symb=symb.split("|")
        lab=symb.pop(0)
        nx, ny = (symb[0],1) if len(symb)==1 else symb[:2] 
        # make the background
        bg = Checkerboard(x,y,w,h,nx=int(nx),ny=int(ny),batch=self.batch,group=self.background)
        # make the label
        label= self.init_label(lab,x,y,w,h,font_size)
        return bg, label

