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
class ImageStackScreen(SelectionGridScreen):
    def __init__(self, window, noisetag, symbols=None, scale_to_fit:bool=True, artifical_deficit_id:int=None, color_labels:bool=True, **kwargs):
        """variant of SelectionGridScreen which has a 'stack' of images *on top of each other* rather than a 'grid' of options.
        i.e. in this screen all elements in the selection take the full grid display width.

        Args:
            window (_type_): the window the show the display in 
            noisetag (_type_): the noisetag object to communicate with the EEG/BCI system
            symbols (list-of-lists, optional): the grid layout, as a list of lists for rows/cols. Defaults to None.
            scale_to_fit (bool,optional): scale the image to fix the size of the target.  Defaults to True.
            color_labels (bool, optional): if true, then color the label as well as the background on state change.
        """        
        self.scale_to_fit, self.color_labels, self.artifical_deficit_id = scale_to_fit, color_labels, artifical_deficit_id
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
            if idx == self.artifical_deficit_id and state==1:
                # TODO[X]: use a randomly choosen state!
                state = 0 # np.random.randint(0,1)  # don't flicker it!
            img_idx = min(state+1,len(self.objects[idx])-1) if len(self.objects[idx])>1 else 0
            self.objects[idx][img_idx].visible = True
            #self.objects[idx][img_idx].color = self.state2color[state]
        if self.labels[idx]:
            col = self.state2color.get(state,(255,255,255,255)) if self.color_labels else (255,255,255,255)
            col = tuple(col)
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
                self.window.flipstats.update_statistics()
                self.set_framerate("{:4.1f} +/-{:4.1f}ms".format(self.window.flipstats.median,self.window.flipstats.sigma))                



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


if __name__ == "__main__":
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    nt = Noisetag(stimSeq='level4_gold_soa12.txt', utopiaController=None)
    window = initPyglet(width=640, height=480)
    screen = ImageStackScreen(window, nt, symbols=[["faces/0.jpg|faces/1.jpg|faces/2.jpg|faces/3.jpg","houses/House_005.jpg|houses/House_0011.jpg|houses/House_012.jpg|houses/House_13.jpg"],["objects/0.JPG|objects/1.jpg|objects/2.jpg|objects/6.jpg","animals/0.jpg|animals/1.jpg|animals/3.jpg|animals/5.jpg"]])
    nt.startFlicker(framesperbit=30)
    run_screen(window, screen)
