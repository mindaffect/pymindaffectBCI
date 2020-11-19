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
try:
    from gpiozero import LED 
except:
    print("Making a mock LED class for testing")
    class led:
        def __init__(self,id=-1): self.id=id
        def on(self): print('{}*'.format(self.id),end='')
        def off(self): print('{}.'.format(self.id),end='')
    LEDS= [led(i) for i in range(20)]
    def LED(i:int): return LEDS[i]

nt=None
leds=[]
objIDs=[]
frameduration_s = 1/60

#---------------------------------------------------------------------
def draw():
    """draw the GPIO output with the flicker state as obtained from the noisetag object"""
    #print("Background state"+str(backgroundstate))
    # get the updated stimulus state info
    global nt, leds, frameduration_s
    nt.updateStimulusState()
    stimulus_state,target_idx,objIDs,sendEvents=nt.getStimulusState()
    target_state = stimulus_state[target_idx] if target_idx>=0 else -1

    # BODGE: sleep to limit the stimulus update rate
    time.sleep(frameduration_s)
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
    """function called back when a BCI selection is made which prints something
    Modify this function if you would like something more exciting to happen on BCI selection

    Args:
        objID (int): the ID number of the selected object
    """    
    print("Selection: objID=%d"%(objID))

#------------------------------------------------------------------------
# Initialization : display
def init(framerate_hz=15, numleds=2, led2gpiopin=(2,3,4), nCal=10, nPred=10):
    """setup the pi for GPIO based presentation

    Args:
        framerate_hz (float, optional): framerate for the flicker. Defaults to 15.
        numleds (int, optional): number of leds to flicker. Defaults to 2.
        led2gpiopin (tuple, optional): the LED index to GPIO pin mapping to use. Defaults to (2,3,4).
        nCal (int, optional): number of calibration trials to use. Defaults to 10.
        nPred (int, optional): number of prediction trials to use. Defaults to 10.
    """    
    global nt, objIDs, leds, frameduration_s
    
    frameduration_s = 1.0/framerate_hz
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
    nt.startExpt(nCal=nCal,nPred=nPred,
                cueduration=4,duration=10,feedbackduration=4)
    # register function to call if selection is made
    nt.addSelectionHandler(selectionHandler)


def run(framerate_hz=15, numleds=1, led2gpiopin=(2,3,4), ncal=10, npred=10, **kwargs):
    """run the pi GPIO based presentation 

    Args:
        framerate_hz (float, optional): framerate for the flicker. Defaults to 15.
        numleds (int, optional): number of leds to flicker. Defaults to 2.
        led2gpiopin (tuple, optional): the LED index to GPIO pin mapping to use. Defaults to (2,3,4).
        ncal (int, optional): number of calibration trials to use. Defaults to 10.
        npred (int, optional): number of prediction trials to use. Defaults to 10.
    """
    if kwargs is not None:
        print("Warning additional args ignored: {}".format(kwargs))    
    init(framerate_hz=framerate_hz, numleds=numleds, led2gpiopin=led2gpiopin, nCal=ncal, nPred=npred)
    while True :
        draw()

if __name__=="__main__":
    run(framerate_hz = 15, numleds=1, led2gpiopin=(2,3,4), ncal=10, npred=100)
