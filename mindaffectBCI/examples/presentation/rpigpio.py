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
from mindaffectBCI.noisetag import Noisetag
from gpiozero import LED 


nt=None
leds=[]
objIDs=[]
framerate = 60

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
    for i,led in enumerate(leds): 
        # get the background state of this cell
        bs = stimulus_state[i] if stimulus_state else None
        if not bs is None and bs>0 :
            led.on()
        else :
            led.off()

    if target_state is not None and target_state>=0:
        print("*" if target_state==1 else '.', end='', flush=True)
    # send info on updated display state
    nt.sendStimulusState()

def selectionHandler(objID):
    print("Selection: objID=%d"%(objID))

#------------------------------------------------------------------------
# Initialization : display
def init(numleds=2, led2gpiopin=(2,3,4)):
    global nt, objIDs, leds
    
    if led2gpiopin is None:
        led2gpiopin = list(range(numleds))
    leds=[]
    objIDs=[]
    for i in range(numleds):
        leds.append(LED(led2gpiopin[i]))
        objIDs.append(i+1)

    nt=Noisetag()
    nt.connect()
    nt.setActiveObjIDs(objIDs)
    nt.startExpt(nCal=10,nPred=10,
                cueduration=4,duration=10,feedbackduration=4, framesperbit=4)
    # register function to call if selection is made
    nt.addSelectionHandler(selectionHandler)

if __name__=="__main__":
    framerate = 60
    init(numleds=2, led2gpiopin=(2,3,4))
    while True :
        draw()
