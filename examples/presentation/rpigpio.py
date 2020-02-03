#!/usr/bin/env python3
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
    nt.startExpt(objIDs,nCal=10,nPred=10,
                cueduration=4,duration=10,feedbackduration=4)
    # register function to call if selection is made
    nt.addSelectionHandler(selectionHandler)

if __name__=="__main__":
    init()
    while True :
        draw()
