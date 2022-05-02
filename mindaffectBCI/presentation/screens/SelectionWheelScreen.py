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
import math
from mindaffectBCI.presentation.screens.SelectionGridScreen import SelectionGridScreen
from mindaffectBCI.presentation.screens.visual_stimuli import CheckerboardSegment, PieSegment, polar2cart

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class SelectionWheelScreen(SelectionGridScreen):

    def __init__(self, window, noisetag, symbols=None, ring_radii=None, ischeckerboard:bool=False, theta0:float=0, **kwargs):
        """ variant of SelectionGridScreen lays the grid out on a series of concentric rings

        Args:
            window (_type_): the window the show the display in
            noisetag (_type_): the noisetag object to communicate with the EEG/BCI system
            symbols (list-of-lists, optional): the grid layout, as a list of lists for rows/cols. Defaults to None.
            ring_radii (list-of-float, optional): the radii of the rings, where each ring gets one row of stimuli starting from the inner-most ring.  N.B. ring_radii should be number of rows+1. Defaults to None.
            ischeckerboard (bool or int, optional): _description_. number of checkers. Three checkers if true. 
                Defaults to False.
        """
        self.ring_radii, self.ischeckerboard, self.theta0 = ring_radii, ischeckerboard, theta0
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
            x,y,w,h (float): bounding box for the target
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
                theta = j * dtheta + self.theta0
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
        """create a display sector target (i.e. pie wedge)

        Args:
            symb (str): the symbol to show in this sector
            cx,cy (float): the *center* of the sector in pixels on screen
            theta (float): the starting rotational angle of the sector in radians.  Where 0 is vertically up.
            radius (foat): the radius of the inner side of the sector, where r=0 is the center of the grid.
            dtheta (_type_): the rotational width of the sector, in radians
            dr (_type_): the radial width of the sector, in pixels
            font_size (int, optional): size of the font to use for the sector label. Defaults to None.

        Returns:
            _type_: _description_
        """
        # make the background
        if self.ischeckerboard:
            nx, ny = (3,3) if self.ischeckerboard == True else (self.ischeckerboard,self.ischeckerboard)
            seg = CheckerboardSegment(cx,cy,theta,radius,dtheta,dr,nx=nx,ny=ny,batch=self.batch,group=self.foreground_group)
        else:
            seg = PieSegment(cx,cy,theta,radius,dtheta,dr,batch=self.batch,group=self.foreground_group)

        # make the label
        centx, centy = polar2cart(cx,cy,theta+dtheta/2, radius+dr/2)
        lb, tr = (polar2cart(cx,cy,theta,radius), polar2cart(cx,cy,theta+dtheta,radius+dr))
        w = max(abs(lb[0]-tr[0]), abs(lb[1]-tr[1]))
        label= self.init_label(symb,centx-w/2,centy-w/2,w,w,font_size)

        return seg, label



if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    # make a noisetag object without the connection to the hub for testing
    nt = Noisetag(stimSeq='mgold_65_6532.txt', utopiaController=None)
    window = initPyglet(width=640, height=480)
    screen = SelectionWheelScreen(window, nt, symbols=[['1','2','3','4','5'],['6','7','8','9','10']], ischeckerboard=4, theta0=.8, bgFraction=0)
    # wait for a connection to the BCI
    nt.startFlicker()
    # run the screen with the flicker
    run_screen(window, screen)