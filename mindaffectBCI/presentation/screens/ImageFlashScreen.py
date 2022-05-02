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
            col = tuple(col)
            self.labels[idx].color=  col if len(col)==4 else col+(255,) #(255,255,255,255) # reset labels


if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    nt = Noisetag(stimSeq='level4_gold_soa12.txt', utopiaController=None)
    window = initPyglet(width=640, height=480)
    screen = ImageFlashScreen(window, nt, symbols=[["faces/0.jpg|faces/1.jpg|faces/2.jpg|faces/3.jpg","houses/House_005.jpg|houses/House_0011.jpg|houses/House_012.jpg|houses/House_13.jpg"],["objects/0.JPG|objects/1.jpg|objects/2.jpg|objects/6.jpg","animals/0.jpg|animals/1.jpg|animals/3.jpg|animals/5.jpg"]])
    nt.startFlicker(framesperbit=30)
    run_screen(window, screen)

