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

# Set up imports and paths
import time
from mindaffectBCI.noisetag import NoiseTag
from gpiozero import LED 

#---------------------------------------------------------------------
def draw():
    """draw the LED state"""
    #print("Background state"+str(backgroundstate))
    # get the updated stimulus state info
    global nt, leds, framerate
    nt.updateStimulusState()
    stimulus_state,target_state,objIDs,sendEvents=nt.getStimulusState()

    # BODGE: sleep to limit the stimulus update rate
    time.sleep(1/framerate)
    # update the state of each LED to match the stimulusstate
    for idx in range(len(fp.objects)): 
        # get the background state of this cell
        bs = stimulusstate[idx] if stimulus_state else None
        if not bs is None and bs>0 :
            leds[idx].on()
        else :
            leds[idx].off()
    # send info on updated display state
    nt.sendStimulusState()

def selectionHandler(objID):
    print("Selection: objID=%d"%(objID))

#------------------------------------------------------------------------
# Initialization : display
def init():
    framerate=1/60

    numleds=2
    leds=[]
    objIDs=[]
    for i in range(len(leds)):
        leds.append(LED(leds[i]))
        objIDs.append(i+1)

    nt=Noisetag()
    nt.connect()
    nt.startExpt(objIDs,nCal=10,nPred=10,
                cueduration=4,duration=10,feedbackduration=4)
    # register function to call if selection is made
    nt.addSelectionHandler(selectionHandler)

if __name__=="__main__":
    init()
    while True :
        draw()
