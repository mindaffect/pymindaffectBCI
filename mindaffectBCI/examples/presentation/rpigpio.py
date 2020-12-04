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
optoled=None
leds=[]
objIDs=[]
isi = 1/60

#---------------------------------------------------------------------
def draw():
    """draw the GPIO output with the flicker state as obtained from the noisetag object"""
    #print("Background state"+str(backgroundstate))
    # get the updated stimulus state info
    global nt, leds, optoled, isi
    nt.updateStimulusState()
    stimulus_state,target_idx,objIDs,sendEvents=nt.getStimulusState()
    target_state = stimulus_state[target_idx] if target_idx>=0 else -1

    # BODGE: sleep to limit the stimulus update rate
    time.sleep(isi)
    # update the state of each LED to match the stimulusstate
    for i,led in enumerate(leds): 
        # get the background state of this cell
        bs = stimulus_state[i] if stimulus_state else None
        led.on() if not bs is None and bs>0 else led.off()

    if target_state is not None and target_state>=0:
        print("t*" if target_state>0 else 't.', end='', flush=True)
        if optoled:
            optoled.on() if target_state>0 else optoled.off()
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
def init(framerate_hz=15, numleds=2, led2gpiopin=(2,3,4), nCal=10, nPred=10, cuedprediction=True, 
         duration=4, cueduration=2, feedbackduration=4, opto:bool=True, **kwargs):
    """setup the pi for GPIO based presentation

    Args:
        framerate_hz (float, optional): framerate for the flicker. Defaults to 15.
        numleds (int, optional): number of leds to flicker. Defaults to 2.
        led2gpiopin (tuple, optional): the LED index to GPIO pin mapping to use. Defaults to (2,3,4).
        nCal (int, optional): number of calibration trials to use. Defaults to 10.
        nPred (int, optional): number of prediction trials to use. Defaults to 10.
        cuedprediction (bool, optional): use cues in prediction phase. Defaults to True.
        duration (int, optional): flicker duration. Defaults to 4.
        cueduration (int, optional): target cue duration. Defaults to 2.
        feedbackduration (int, optional): target feedback duration. Defaults to 4.
        opto (bool,optional): flag if use the 1st led as 'opto' trigger, so it always shows the cued target. Defaults to True.
    """
    global nt, objIDs, leds, optoled, isi

    if kwargs is not None:
        print("Warning additional args ignored: {}".format(kwargs))     
    
    if framerate_hz is not None:
        isi = 1/framerate_hz

    if led2gpiopin is None:
        led2gpiopin = list(range(numleds))

    gpioi = 0
    if opto: # 1st led is opto
        optoled = LED(led2gpiopin[gpioi])
        gpioi = gpioi+1

    leds=[]
    for i in range(numleds):
        try:
            leds.append(LED(led2gpiopin[gpioi]))
            objIDs.append(i+1)
            gpioi=gpioi+1
        except:
            print("Error adding extra leds -- did you specify enough in led2gpiopin")
            raise

    nt=Noisetag()
    nt.connect()
    nt.setActiveObjIDs(objIDs)
    nt.set_isi(1/framerate_hz)  # N.B. set before start expt. etc, to get seconds->frames conversion
    nt.startExpt(nCal=nCal,nPred=nPred, cuedprediction=cuedprediction,
                cueframes=cueduration//isi,
                numframes=duration//isi,
                feedbackframes=feedbackduration//isi,
                interphaseframes=40//isi)
    # register function to call if selection is made
    nt.addSelectionHandler(selectionHandler)

def run(framerate_hz=15, numleds=1, led2gpiopin=(2,3,4,5,6,7,8), ncal=10, npred=10, **kwargs):
    """run the pi GPIO based presentation 

    Args:
        framerate_hz (float, optional): framerate for the flicker. Defaults to 15.
        numleds (int, optional): number of leds to flicker. Defaults to 2.
        led2gpiopin (tuple, optional): the LED index to GPIO pin mapping to use. Defaults to (2,3,4).
        ncal (int, optional): number of calibration trials to use. Defaults to 10.
        npred (int, optional): number of prediction trials to use. Defaults to 10.
    """   
    init(framerate_hz=framerate_hz, numleds=numleds, led2gpiopin=led2gpiopin, nCal=ncal, nPred=npred, **kwargs)
    while True :
        draw()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncal',type=int, help='number calibration trials', default=10)
    parser.add_argument('--npred',type=int, help='number prediction trials', default=10)
    parser.add_argument('--framerate_hz',type=float, help='flicker rate', default=15)
    parser.add_argument('--numleds',type=int, help='number of flickering leds', default=2)
    parser.add_argument('--duration',type=float, help='duration in seconds of trial flickering', default=5)
    parser.add_argument('--cueduration',type=float, help='duration in seconds of trial cue', default=2)
    parser.add_argument('--feedbackduration',type=float, help='duration in seconds of trial feedback', default=2)
    parser.add_argument('--cuedprediction',type=bool, help='use cued or un-cued prediction', default=True)
    parser.add_argument('--opto',type=bool, help='use the 1st led as an "opto" led, i.e. always tracks the cued target.', default=False)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    run(**vars(args))