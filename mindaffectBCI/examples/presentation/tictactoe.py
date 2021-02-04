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

import mindaffectBCI.examples.presentation.selectionMatrix as selectionMatrix
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

if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    setattr(args,'symbols',tictactoe_symbols)
    setattr(args,'calibration_screen','mindaffectBCI.examples.presentation.tictactoe.TictactoeScreen')
    selectionMatrix.run(**vars(args))
