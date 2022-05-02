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
from mindaffectBCI.presentation.screens.SelectionGridScreen import SelectionGridScreen
from mindaffectBCI.presentation.screens.visual_stimuli import Checkerboard

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class CheckerboardGridScreen(SelectionGridScreen):
    def __init__(self, window, noisetag, symbols=None, nchecks:int=3, **kwargs):
        """variant of SelectionGridScreen which uses a checkerboard behind each target

        Args:
            window (_type_): the window the show the display in 
            noisetag (_type_): the noisetag object to communicate with the EEG/BCI system
            symbols (_type_, optional): the grid layout, as a list of lists for rows/cols. Defaults to None.
            nchecks (int, optional): number of checks in the checkerboard behind each symbol.  for example if nchecks=3 we have a 3x3 checkerboard behind each character. Defaults to 3.
        """
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
        """setup a checkerboard background target

        Args:
            symb (_type_): the symbol to display, e.g. the letter/image/symbol
            x,y,w,h (float): bounding box for the target
            i,j (float): position of the target in the symbols grid, i.e. row + col
            font_size (int, optional): fontsize for the text. Defaults to None.

        Returns:
            background,foreground: visual objects for the background (check) and foreground (letter)
        """
        # make the background
        bg = Checkerboard(x,y,w,h,nx=self.nchecks[0],ny=self.nchecks[1],batch=self.batch,group=self.background)
        # make the label
        label= self.init_label(symb,x,y,w,h,font_size)
        return bg, label


if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    # make a noisetag object without the connection to the hub for testing
    nt = Noisetag(stimSeq='mgold_65_6532.txt', utopiaController=None)
    window = initPyglet(width=640, height=480) # small testing window
    screen = CheckerboardGridScreen(window, nt, symbols=[['1','2'],['3','4']])
    # start the stimulus sequence playing, 10s with a 10x slowdown
    nt.startFlicker(numframes=60*10, framesperbit=10)
    # run the screen with the flicker
    run_screen(window, screen)