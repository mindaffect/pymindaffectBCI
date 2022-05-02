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
from mindaffectBCI.presentation.screens.basic_screens import WaitScreen
from mindaffectBCI.examples.presentation.snakegame import SnakeGame

class SnakeGameScreen(WaitScreen):
    def __init__(self, window, symbols, noisetag, duration:float=None, waitKey:bool=False, logo:str=None, framespermove:int=60, target_only:bool=False, clearScreen:bool=True, **kwargs):
        super().__init__(window, duration, waitKey, logo)
        self.window=window
        self.noisetag = noisetag
        self.liveSelections = None
        self.show_newtarget = None
        self.clearScreen = clearScreen
        self.framespermove = framespermove
        self.target_only = target_only
        self.reset()

    def set_grid(self,**kwargs):
        self.objIDs = [50,51,52,53] # [L,R,U,D]
        self.noisetag.setActiveObjIDs(self.objIDs) # register these objIDs

    def set_sentence(self, **kwargs):
        pass

    def setliveSelections(self, value):
        if self.liveSelections is None :
            self.noisetag.addSelectionHandler(self.doSelection)
        self.liveSelections = value

    def doSelection(self,objID):
        if objID in self.objIDs:
                print("doSelection: {}".format(objID))
                symbIdx = self.objIDs.index(objID)
                # move the snake in the desired direction
                self.snake.move(symbIdx)

    def setshowNewTarget(self, value):
        if self.show_newtarget is None:
            self.noisetag.addNewTargetHandler(self.doNewTarget)
        self.show_newtarget=value

    def doNewTarget(self):
        print("Got new target")

    def reset(self):
        self.nframe = 0
        self.snakegame = SnakeGame(self.window,None)

    def draw(self, t):
        """draw the letter-grid with given stimulus state for each object.
        Note: To maximise timing accuracy we send the info on the grid-stimulus state
        at the start of the *next* frame, as this happens as soon as possible after
        the screen 'flip'. """
        if not self.isRunning:
            self.nframe = 0
            self.isRunning=True
        self.framestart=self.noisetag.getTimeStamp()
        winflip = self.window.lastfliptime
        if winflip > self.framestart or winflip < self.frameend:
            print("Error: frameend={} winflip={} framestart={}".format(self.frameend,winflip,self.framestart))
        self.nframe = self.nframe+1
        if self.sendEvents:
            self.noisetag.sendStimulusState(timestamp=winflip)

        # get the current stimulus state to show
        try:
            self.noisetag.updateStimulusState()
            stimulus_state, target_idx, objIDs, sendEvents=self.noisetag.getStimulusState()
            target_state = stimulus_state[target_idx] if target_idx>=0 else -1
            if target_idx >= 0 : self.last_target_idx = target_idx
        except StopIteration:
            self.isDone=True
            return

        # turn all off if no stim-state
        if stimulus_state is None:
            stimulus_state = [0]*len(self.objIDs)

        # insert the flicker state into the game grid...
        # draw the white background onto the surface
        if self.clearScreen:
            self.window.clear()
        # update the state
        for idx in range(min(len(self.objIDs), len(stimulus_state))):
            # set background color based on the stimulus state (if set)
            try:
                ssi = stimulus_state[idx]
                if self.target_only and not target_idx == idx :
                    ssi = 0
                self.update_object_state(idx,ssi)
            except KeyError:
                pass

        # update the game state
        self.nframe = self.nframe + 1
        if self.nframe % self.framespermove == 0 :
            # game state tick!
            self.snake.run_rules()
            self.snake.move()

        # call the game draw functions
        self.snake.draw()
        self.snake.draw_score()


    state2color={0:(5, 5, 5),       # off=grey
                 1:(255, 255, 255), # on=white
                 2:(0, 255, 0),     # cue=green
                 3:(0, 0, 255)}     # feedback=blue
    def update_object_state(self,idx,state):
        # get the snake head coord
        x,y = self.snake.body[-1]
        if idx == 0: # L  # match the command order [LRUD]
            x = x-1
        elif idx == 1: # R
            x = x+1
        elif idx == 2: # U
            y = y+1
        elif idx == 3: # D
            y = y-1
        # set the cell state, occupied and given color
        self.snake.cells[x][y] = (1, self.state2color[state])


if __name__=="__main__":
    args = selectionMatrix.parse_args()
    setattr(args,'symbols',None)
    setattr(args,'calibration_screen','mindaffectBCI.examples.presentation.snake_game.SnakeGameScreen')
    selectionMatrix.run(**vars(args))
