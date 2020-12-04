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

from build.lib.mindaffectBCI.examples.presentation.selectionMatrix import initPyglet
import mindaffectBCI.examples.presentation.selectionMatrix as selectionMatrix
from mindaffectBCI.noisetag import Noisetag
import random
import copy

tictactoe_symbols=[['+','+','+'],['+','+','+'],['+','+','+']]

class TictactoeScreen(selectionMatrix.SelectionGridScreen):
    """class implementing as simple tac-tac-toe BCI game

    TODO[]: win check logic
    TODO[]: don't flicker filled in cells
    """
    def __init__(self, symbols=tictactoe_symbols, **kwargs):
        symbols = copy.deepcopy(symbols)
        super().__init__(symbols=symbols, **kwargs)

    def doMove(self, player_move_idx:int):
        if player_move_idx is not None:
            self.setLabel(player_move_idx, "X")
            self.setObj(player_move_idx, None)
        computer_move_idx = self.get_computer_move()
        if computer_move_idx is not None:
            self.setLabel(computer_move_idx, "O")
            self.setObj(computer_move_idx, None)

    def get_computer_move(self):
        """get the computer move from the current game state -- just a random free square
        """
        freeidx = [ i for i in range(len(self.labels)) if self.labels[i].text=="+" ]
        if len(freeidx)==0:
            return None
        random.shuffle(freeidx) # permute free list
        return freeidx[0] # return random element

    def doSelection(self, objID):
        if self.liveSelections == True:
            if objID in self.objIDs:
                print("doSelection: {}".format(objID))
                idx = self.objIDs.index(objID)
                sel = self.getLabel(idx)
                sel = sel.text if sel is not None else ''
                if self.show_correct and self.last_target_idx>=0:
                    sel += "*" if idx==self.last_target_idx else "_"
                self.doMove(idx)

def run(symbols=None, ncal:int=10, npred:int=10, calibration_trialduration=4.2,  prediction_trialduration=20, stimfile=None, selectionThreshold:float=.1,
        framesperbit:int=1, optosensor:bool=True, fullscreen:bool=False, windowed:bool=None, 
        fullscreen_stimulus:bool=True, simple_calibration=False, host=None, calibration_symbols=None, bgFraction=.1,
        calibration_args:dict=None, prediction_args:dict=None): 
    if stimfile is None:
        stimfile = 'mgold_61_6521_psk_60hz.txt'
    if fullscreen is None and windowed is not None:
        fullscreen = not windowed
    if windowed == True or fullscreen == True:
        fullscreen_stimulus = False
    nt=Noisetag(stimFile=stimfile,clientid='Presentation:selectionMatrix')
    if host is not None and not host in ('','-'):
        nt.connect(host, queryifhostnotfound=False)

    # init the graphics system
    window = initPyglet(fullscreen=fullscreen)

    # make the screen manager object which manages the app state
    ss = selectionMatrix.ExptScreenManager(window, nt, tictactoe_symbols, nCal=ncal, nPred=npred, framesperbit=framesperbit, 
                        fullscreen_stimulus=fullscreen_stimulus, selectionThreshold=selectionThreshold, 
                        optosensor=optosensor, simple_calibration=True, calibration_symbols=calibration_symbols, 
                        bgFraction=bgFraction, 
                        calibration_args=calibration_args, calibration_trialduration=calibration_trialduration, 
                        prediction_args=prediction_args, prediction_trialduration=prediction_trialduration)

    # override the selection grid with the tictactoe one
    ss.selectionGrid = TictactoeScreen(window=window, symbols=tictactoe_symbols, noisetag=nt, optosensor=optosensor)

    # run the app
    selectionMatrix.run_screen(ss)


if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    run(**vars(args))
