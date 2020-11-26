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

from mindaffectBCI.noisetag import Noisetag, sumstats
from psychopy import visual, core

nt = None
opto = None
squares = None
window = None

# dictionary mapping from stimulus-state to colors
state2color={0:(-.8,-.8,-.8), # off=grey
             1:(1,1,1),    # on=white
             2:(-1,1,-1),    # cue=green
             3:(-1,-1,1)}    # feedback=blue
def draw(time):
    '''draw the display with colors from noisetag'''
    global lastfliptime, nt, opto, squares
    # send info on the *previous* stimulus state, with the recorded vsync time (if available)
    fliptime = lastfliptime if lastfliptime is not None else nt.getTimeStamp()
    nt.sendStimulusState(timestamp=fliptime)
    # update and get the new stimulus state to display
    try : 
        nt.updateStimulusState()
        stimulus_state,target_idx,objIDs,sendEvents=nt.getStimulusState()
        target_state = stimulus_state[target_idx] if target_idx>=0 else -1
    except StopIteration :
        exit() # terminate app when noisetag is done
        return

    # draw the display with the instructed colors
    if stimulus_state : 
        for i,s in enumerate(squares):
            s.fillColor = state2color[stimulus_state[i]]

    # some textual logging of what's happening
    if target_state is not None and target_state>=0:
        opto.fillColor = state2color[1 if target_state==1 else 0]
        print("*" if target_state>0 else '.',end='',flush=True)
    else:
        print('.',end='',flush=True)

# WARNING: Hacking ahead!!!
# record the flip time
lastfliptime=0

# define a trival selection handler
def selectionHandler(objID):
    print("Selected: %d"%(objID))    

def init(nCal=5,nPred=10,duration=4,framesperbit=1,cuedprediction=True, fullscreen=False):
    global window, lastfliptime, nt, opto, squares
    # Initialize the noise-tagging connection
    nt = Noisetag()
    nt.connect(timeout_ms=5000)
    nt.addSelectionHandler(selectionHandler)
    # tell the noisetag framework to run a full : calibrate->prediction sequence
    nt.setnumActiveObjIDs(2)
    nt.startExpt(nCal=nCal,nPred=nPred,duration=duration,framesperbit=framesperbit,cuedprediction=cuedprediction)

    # Initialize the drawing window
    # make a default window, with fixed size for simplicty, and vsync for timing
    if fullscreen:
        window = visual.Window(fullscr=True,color=(-1,-1,-1))
    else:
        window = visual.Window(size=(640,480),color=(-1,-1,-1))

    # grid
    squares = [visual.Rect(window,pos=(-.5,0),size=(.4,.4),autoDraw=True),
            visual.Rect(window,pos=(.5,0),size=(.4,.4),autoDraw=True)]
    # opto-sensor
    opto = visual.Rect(window,pos=(-1,1),size=(.4,.4),autoDraw=True)

def run(ncal=5,npred=10,duration=4,framesperbit=1,cuedprediction=True,fullscreen=False,**kwargs):
    """run the psychopy based presentation 

    Args:
        ncal (int, optional): number of calibration trials to use. Defaults to 10.
        npred (int, optional): number of prediction trials to use. Defaults to 10.
    """
    global window    
    init(ncal,npred,duration,framesperbit,cuedprediction,fullscreen=fullscreen)
    # run the main loop
    clock = core.Clock()
    while True:
        draw(clock.getTime())
        ffliptime = window.flip() # flip-time in seconds...
        lastfliptime = nt.getTimeStamp()
        #lastfliptime = ffliptime/1000.0 # don't use, need to fix hearbeats also!

if __name__=="__main__":
    run()