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

import time
import random
from .stimseq import StimSeq
from .utopiaController import UtopiaController, getTimeStamp
import os
scriptpath=os.path.dirname(os.path.realpath(__file__))
stimFile=os.path.join(scriptpath,'mgold_61_6521_psk_60hz.txt')
objIDs=list(range(1,10))
isi=1/60
MAXOBJID=64

class FSM:
    ''' simple finite state machine, using a generator-like pattern'''
    def next(self,t):
        '''update the current state, return the new state, raise StopIteration exception when done'''
        return t
    def get(self):
        '''get the current state, for this set of active objects'''
        return (None,-1,None,False)# BODGE: stimState,tgtState,objIDs,sendEvent

class GSM(FSM):
    '''Generalized state machine with stack of states'''
    def __init__(self): self.stack=[]
    def clear(self):    self.stack=[]
    def push(self,s):   self.stack.append(s); return self.stack
    def pop(self):      return self.stack.pop()
    def next(self,t):
        '''get the next stimulus state to shown'''
        while self.stack :
            try : 
                self.stack[-1].next(t)
                return
            except StopIteration :
                # end of this fsm, unwind the fsm stack
                self.pop()
                # for pretty printing
                print()
        raise StopIteration()
    def get(self):
        if self.stack :
            return self.stack[-1].get()
        else :
            return None

class WaitFor(FSM):
    '''wait for given number of frames to pass'''
    def __init__(self,numframes):
        self.numframes=numframes
        self.nframe=0
        print("waitFor: %g"%(self.numframes))
    def next(self,t):
        self.nframe=self.nframe+1
        if self.nframe>self.numframes :
            raise StopIteration()
    def get(self):
        return (None,-1,None,False) # BODGE: stimState,tgtState,objIDs,sendEvent
    
class Flicker(FSM):
    ''' do a normal flicker sequence'''
    def __init__(self,stimSeq=None,
                 numframes=4*isi,tgtidx=-1,
                 sendEvents=True):
        self.stimSeq=stimSeq
        self.numframes=numframes
        self.nframe=0
        self.tgtidx=tgtidx
        self.tgtstate=-1
        self.sendEvents=sendEvents

        # ensure right length
        self.ss=None
        #print("stimSeq:")
        #for i in range(len(self.objIDs)):
        #    print(["objID %d = %s"%(i,"".join([ '*' if self.stimSeq[t][i]==1 else '.' for t in range(len(self.stimSeq))]))])
        print('flicker: %d frames, tgt %d'%(self.numframes,tgtidx if tgtidx >=0 else -1))
        
    def next(self,t):
        self.nframe=self.nframe+1
        if self.nframe>self.numframes:
            raise StopIteration()

    def update_ss(self):
        # extract the current frames stimulus state, loop if past end
        self.ss       = self.stimSeq[self.nframe % len(self.stimSeq)]
        # extract the current target state, for these objects
        self.tgtstate = self.ss[self.tgtidx] if self.tgtidx>=0 else -1
        
    def get(self):
        # update the curent stimulus state info
        self.update_ss()        
        # default to the stored set active objIDs
        #if objIDs is None : objIDs=self.objIDs
        # get the state info for the given set of objectIDs
        #ss =[self.ss[i] for i in objIDs-1]
        return (self.ss,self.tgtstate,objIDs,self.sendEvents)

    
class FlickerWithSelection(Flicker):
    ''' do a normal flicker sequence, with early stopping selection'''
    def __init__(self,stimSeq=None,numframes=4*isi,tgtidx=-1,
                 utopiaController=None,
                 sendEvents=True):
        super().__init__(stimSeq,numframes,tgtidx,sendEvents)
        self.utopiaController = utopiaController
        if self.utopiaController is None : raise ValueError("must have utopiaController")
        # ensure old predictions are gone..
        self.utopiaController.getNewMessages() # ensure all old predictions processed
        self.utopiaController.clearLastPrediction() # reset to no waiting predictions
        print(' with selection')
        
    def next(self,t):
        super().next(t)
        # check for selection and stop if found
        objId,selected=self.utopiaController.getLastSelection()
        if selected :
            if self.sendEvents :
                # send event to say selection has occured
                self.utopiaController.selection(objId)
            raise StopIteration()

def mkBlinkingSequence(numframes,tgtidx,tgtState=1):
    blinkSeq=[[0 for i in range(MAXOBJID)] for i in range(numframes)]
    for i in range(int(numframes/2)) : blinkSeq[i][tgtidx]=tgtState # tgt on
    return blinkSeq

class HighlightObject(Flicker):
    '''Highlight a single object for a number of frames'''
    def __init__(self,numframes=isi*2,tgtidx=-1,tgtState=2,
                 sendEvents=False,numblinkframes=int(.5/isi)):
        #self.objIDs=objIDs if hasattr(objIDs, "__len__") else list(range(1,objIDs+1))
        if numblinkframes>0 and tgtidx>=0 : # blinking cue
            stimSeq = mkBlinkingSequence(int(numblinkframes),tgtidx,tgtState)
        else :
            stimSeq = [[0]*MAXOBJID]
            if tgtidx>=0:
                stimSeq[0][tgtidx]=tgtState
        super().__init__(stimSeq,numframes,tgtidx,sendEvents=sendEvents)
        print('highlight: tgtidx=%d nframes=%d'%(tgtidx if tgtidx>=0 else -1,numframes))

class SingleTrial(FSM):
    ''' do a complete single trial with: cue->wait->flicker->feedback '''
    def __init__(self,stimSeq,tgtidx,
                 utopiaController,stimulusStateStack,
                 numframes=None,
                 selectionThreshold=None,
                 duration=4,cueduration=1,feedbackduration=1,waitduration=1,
                 cueframes=None,feedbackframes=None,waitframes=None):
        self.tgtidx=tgtidx
        self.stimSeq=stimSeq
        self.utopiaController = utopiaController
        self.stimulusStateStack=stimulusStateStack
        self.numframes=numframes if numframes else duration/isi
        self.cueframes=cueframes if cueframes else cueduration/isi
        self.feedbackframes=feedbackframes if feedbackframes else feedbackduration/isi
        self.waitframes=waitframes if waitframes else waitduration/isi
        self.selectionThreshold=selectionThreshold
        self.stage=0
        print("tgtidx=%d"%(self.tgtidx if self.tgtidx>=0 else -1))
        
    def next(self,t):
        if self.stage==0 : # trial-start + cue
            # tell decoder to start trial
            self.utopiaController.newTarget()
            # tell decoder to clear predictions if needed
            if self.selectionThreshold is not None:
                self.utopiaController.selectionThreshold=self.selectionThreshold
                self.utopiaController.clearLastPrediction() # reset to no waiting predictions

            if self.tgtidx>=0 : # blinking: only if target is set
                self.stimulusStateStack.push(
                    HighlightObject(self.cueframes,
                                    self.tgtidx,
                                    sendEvents=False))
            else : # skip cue+wait
                self.stage=1
                
        elif self.stage==1 : # wait
            print('1.wait')
            self.stimulusStateStack.push(
                HighlightObject(self.waitframes,-1,
                                sendEvents=False))
            
        elif self.stage==2 : # stim
            print('2.stim, tgt:%d'%(self.tgtidx if self.tgtidx>=0 else -1))
            #objIDs=8,stimSeq=None,numframes=None,tgtidx=None,duration=4
            if self.selectionThreshold is not None: # early stop if thres set
                self.stimulusStateStack.push(
                    FlickerWithSelection(self.stimSeq,
                                         self.numframes,
                                         self.tgtidx,
                                         self.utopiaController,
                                         sendEvents=True))
            else: # no selection based stopping
                self.stimulusStateStack.push(
                    Flicker(self.stimSeq,
                            self.numframes,
                            self.tgtidx,
                            sendEvents=True))
                
        elif self.stage==3 : # wait/feedback
            if self.selectionThreshold is None:
                print('3.wait')
                self.stimulusStateStack.push(
                    HighlightObject(self.waitframes,-1))
            else :
                print('3.feedback') # solid on 
                predObjId,selected=self.utopiaController.getLastSelection()
                print(' pred:%d sel:%d'%(predObjId if predObjId else -1,selected))
                if selected :
                    #tgtidx = self.objIDs.index(predObjId) if predObjId in self.objIDs else -1
                    tgtidx = predObjId-1
                    self.stimulusStateStack.push(
                        HighlightObject(self.feedbackframes,
                                        tgtidx,
                                        tgtState=3,
                                        sendEvents=False,
                                        numblinkframes=0))
                    
        else :
            raise StopIteration
        self.stage=self.stage+1
        
class CalibrationPhase(FSM):
    ''' do a complete calibration phase with nTrials x CalibrationTrial '''
    def __init__(self,objIDs=8,stimSeq=None,nTrials=10,
                 utopiaController=None,stimulusStateStack=None,
                 *args,**kwargs):
        self.objIDs=objIDs if hasattr(objIDs, "__len__") else list(range(1,objIDs+1)) 
        self.stimSeq=stimSeq
        self.nTrials=nTrials
        self.utopiaController=utopiaController
        self.isRunning=False
        self.args=args
        self.kwargs=kwargs
        if self.utopiaController is None : raise ValueError
        self.stimulusStateStack=stimulusStateStack
        if self.stimulusStateStack is None : raise ValueError
        self.trli=0
    def next(self,t):
        if not self.isRunning :
            # tell decoder to start cal
            self.utopiaController.modeChange("Calibration.supervised")
            self.isRunning=True
        if self.trli<self.nTrials:
            # TODO []: should choose from set active objIDs?
            self.tgtidx = random.randint(0,len(self.objIDs)-1)
            print("Start Cal: %d/%d tgtidx=%d"%(self.trli,self.nTrials,self.tgtidx))
            self.stimulusStateStack.push(
                SingleTrial(self.stimSeq,
                            self.tgtidx,
                            self.utopiaController,
                            self.stimulusStateStack,
                            *self.args,**self.kwargs)) # pass through other arguments?
        else:
            self.utopiaController.modeChange("idle")
            raise StopIteration()
        self.trli=self.trli+1
            
class PredictionPhase(FSM):
    '''do complete prediction phase with nTrials x trials with early-stopping feedback'''
    def __init__(self,objIDs,stimSeq=None,nTrials=10,
                 utopiaController=None,stimulusStateStack=None,
                 selectionThreshold=.1,cuedprediction=False,*args,**kwargs):
        self.objIDs=objIDs if hasattr(objIDs, "__len__") else list(range(1,objIDs+1)) 
        self.stimSeq=stimSeq
        self.nTrials=nTrials
        self.selectionThreshold=selectionThreshold
        self.cuedprediction=cuedprediction
        self.args=args
        self.kwargs=kwargs
        self.tgti=0
        self.isRunning=False
        self.utopiaController=utopiaController
        if self.utopiaController is None : raise ValueError
        self.stimulusStateStack=stimulusStateStack
        if self.stimulusStateStack is None : raise ValueError
        # tell decoder to start cal
    def next(self,t):
        if not self.isRunning :
            self.utopiaController.modeChange("Prediction.static")
            self.isRunning=True
        if self.tgti<self.nTrials:
            if self.cuedprediction :
                self.tgtidx = random.randint(0,len(self.objIDs)-1)
            else:
                self.tgtidx = -1
            print("Start Pred: %d/%d"%(self.tgti,self.nTrials))
            self.stimulusStateStack.push(
                SingleTrial(self.stimSeq,
                            self.tgtidx,
                            self.utopiaController,
                            self.stimulusStateStack,
                            selectionThreshold=self.selectionThreshold,
                            *self.args,**self.kwargs))
        else:
            self.utopiaController.modeChange("idle")
            raise StopIteration()
        self.tgti=self.tgti+1
        
class Experiment(FSM):
    '''do a complete experiment, with calibration -> prediction'''
    def __init__(self,objIDs,stimSeq=None,nCal=10,nPred=20,
                 selectionThreshold=.1,cuedprediction=False,
                 utopiaController=None,stimulusStateStack=None,
                 duration=4,calduration=4,predduration=10,
                 *args,**kwargs):
        self.objIDs=objIDs if hasattr(objIDs, "__len__") else list(range(1,objIDs+1)) 
        self.stimSeq=stimSeq
        self.nCal=nCal
        self.nPred=nPred
        self.calduration=calduration
        self.predduration=predduration
        self.selectionThreshold=selectionThreshold
        self.cuedprediction=cuedprediction
        self.utopiaController=utopiaController
        if self.utopiaController is None : raise ValueError
        self.stimulusStateStack=stimulusStateStack
        if self.stimulusStateStack is None : raise ValueError
        self.args=args
        self.kwargs=kwargs
        self.stage=0
    def next(self,t):
        if self.stage==0:
            self.stimulusStateStack.push(WaitFor(2/isi))
        
        elif self.stage==1:
            self.stimulusStateStack.push(
                CalibrationPhase(self.objIDs,self.stimSeq,self.nCal,
                                 self.utopiaController,
                                 self.stimulusStateStack,
                                 *self.args,duration=self.calduration,**self.kwargs))
        
        elif self.stage==2:
            self.stimulusStateStack.push(WaitFor(10/isi))
        
        elif self.stage==3:
            self.stimulusStateStack.push(
                PredictionPhase(self.objIDs,self.stimSeq,self.nPred,
                                self.utopiaController,
                                self.stimulusStateStack,
                                self.selectionThreshold,
                                self.cuedprediction,
                                *self.args,duration=self.predduration,**self.kwargs))
        
        else:
            raise StopIteration()    
        self.stage=self.stage+1

# use global for the noisecode + utopiaController so can easily
# share over instantiations of the NoiseTag framework.
uc=None
        
class Noisetag:
    '''noisetag abstraction layer to handle *both* the sequencing of the stimulus
    flicker, *and* the communications with the Mindaffect decoder.  Clients can
    use this class to implement BCI control by:
     0) setting the flicker sequence to use (method: startFlicker, startFlickerWithSelection, startCalibration, startPrediction
     1) getting the current stimulus state (method: getStimulusState), and using that to draw the display
     2) telling Noisetag when *exactly* the stimulus update took place (method: sendStimulusState)
     3) getting the predictions/selections from noisetag and acting on them. (method: getLastPrediction() or getLastSelection())
     '''
    def __init__(self,stimFile=stimFile,utopiaController=None,stimulusStateMachineStack=None):
        # global flicker stimulus sequence
        noisecode = StimSeq.fromFile(stimFile)
        noisecode.convertstimSeq2int()
        noisecode.setStimRate(2)
        self.noisecode=noisecode

        # utopiaController
        self.utopiaController = utopiaController
    
        # stimulus state-machine stack
        # Individual stimulus-state-machines track progress in a single
        # stimulus state playback function.
        # Stack allows sequencing of sets of playback functions in loops
        if stimulusStateMachineStack is None :
            stimulusStateMachineStack=GSM()
        self.stimulusStateMachineStack=stimulusStateMachineStack
        self.laststate=(None,None,None,None)
        self.objIDs=None


    def connect(self,host=None,port=-1,queryifhostnotfound=True):
        if self.utopiaController is None :
            # use the global controller if none given
            global uc
            if uc is None :
                # auto-connect the global controller if none given
                uc = UtopiaController()
            self.utopiaController=uc
        if self.utopiaController.isConnected() :
            return True
        self.utopiaController.autoconnect(host,port,
                                          queryifhostnotfound=queryifhostnotfound)
        return self.utopiaController.isConnected()
    
    def isConnected(self):
        if self.utopiaController :
            return self.utopiaController.isConnected()
        return False
    def gethostport(self):
        if self.utopiaController :
            return self.utopiaController.gethostport()
        return None
        
    # stimulus sequence methods via the stimulus state machine stack
    # returns if sequence is still running
    def updateStimulusState(self,t=None):
        self.stimulusStateMachineStack.next(t)

    def getStimulusState(self,objIDs):
        # update set active objects if a set is given
        setActiveObjIDs(objIDs)
        return self.getStimulusState()

    def getStimulusState(self):
        # get the complete stimulus state (for MAXOBIDS objects)
        stimState,tgtstate,objIDs,sendEvents = self.stimulusStateMachineStack.get()
        # subset to the active set, matching objIDs to allobjIDs
        # N.B. objID-1 to map from objID->stimStateIndex
        if stimState is not None :
            stimState = [ stimState[i-1] for i in self.objIDs ]
        self.laststate = (stimState,tgtstate,self.objIDs,sendEvents)
        return self.laststate
    
    def setActiveObjIDs(self,objIDs):
        '''update the set of active objects we send info to decoder about'''
        self.objIDs=objIDs
        return self.objIDs
    
    def setnumActiveObjIDs(self,nobj):
        '''update to say number active objects'''
        objIDs=list(range(1,nobj+1))
        return self.setActiveObjIDs(objIDs)
    
    # decoder interaction methods via. utopia controller
    def sendStimulusState(self,timestamp=None):
        stimState,targetState,objIDs,sendEvent=self.laststate
        # send info about the stimulus displayed
        if sendEvent and stimState is not None :
            #print((stimState,targetState))
            self.utopiaController.sendStimulusEvent(stimState,
                                                    timestamp,
                                                    targetState,
                                                    objIDs)

    def getLastPrediction(self):
        if self.utopiaController :
            return self.utopiaController.getLastPrediction()
        return None
    def getLastSignalQuality(self):
        if self.utopiaController :
            return self.utopiaController.getLastSignalQuality()
        return None
    def getLastSelection(self):
        if self.utopiaController :
            return self.utopiaController.getLastSelection()
        return None
    def addMessageHandler(self,cb):
        if self.utopiaController :
            self.utopiaController.addMessageHandler(cb)
    def addPredictionHandler(self,cb):
        if self.utopiaController :
            self.utopiaController.addPredictionHandler(cb)
    def addSelectionHandler(self,cb):
        if self.utopiaController :
            self.utopiaController.addSelectionHandler(cb)
        
    def getTimeStamp(self,t0=0):
        return getTimeStamp(t0)
    def log(self,msg):
        if self.utopiaController:
            self.utopiaController.log(msg)
    def modeChange(self,newmode):
        '''manually change the decoder mode'''
        if self.utopiaController:
            self.utopiaController.modeChange(newmode)

    # methods to define what (meta) stimulus sequence we will play
    def startExpt(self,nCal=1,nPred=20,selnThreshold=.1,
                  cuedprediction=False,
                  *args,**kwargs):
        if  self.stimulusStateMachineStack.stack :
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()
        self.stimulusStateMachineStack.push(
            Experiment(self.objIDs,self.noisecode.stimSeq,
                       nCal,nPred,
                       selnThreshold,cuedprediction,
                       self.utopiaController,
                       self.stimulusStateMachineStack,
                       *args,**kwargs))
        
    def startCalibration(self,nTrials=10,stimSeq=None,
                         *args,**kwargs):
        if  self.stimulusStateMachineStack.stack :
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()
        
        self.stimulusStateMachineStack.push(
            CalibrationPhase(self.objIDs,
                             self.noisecode.stimSeq,
                             nTrials,
                             self.utopiaController,
                             self.stimulusStateMachineStack,
                             *args,**kwargs))
        
    def startPrediction(self,nTrials=10,stimSeq=None,
                        *args,**kwargs):
        if  self.stimulusStateMachineStack.stack :
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()
        self.stimulusStateMachineStack.push(
            PredictionPhase(self.objIDs,
                            self.noisecode.stimSeq,
                            nTrials,
                            self.utopiaController,
                            self.stimulusStateMachineStack,
                            *args,**kwargs))
        
    def startSingleTrial(self,numframes=100,tgtidx=-1,*args,**kwargs):
        if  self.stimulusStateMachineStack.stack :
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()
        self.stimulusStateMachineStack.push(
            SingleTrial(self.noisecode.stimSeq,
                        tgtidx,
                        self.utopiaController,
                        self.stimulusStateMachineStack,
                        numframes,
                        *args,**kwargs))

    def startFlicker(self,numframes=100,tgtidx=-1,*args,**kwargs):
        if  self.stimulusStateMachineStack.stack :
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()
        self.stimulusStateMachineStack.push(
            Flicker(self.noisecode.stimSeq,
                    numframes,
                    tgtidx,
                    *args,**kwargs))
        
    def startFlickerWithSelection(self,numframes=100,
                                  tgtidx=-1,*args,**kwargs):
        if  self.stimulusStateMachineStack.stack :
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()
        self.stimulusStateMachineStack.push(
            FlickerWithSelection(self.noisecode.stimSeq,
                                 numframes,
                                 tgtidx,
                                 self.utopiaController,
                                 *args,**kwargs))


import math, statistics, numpy
class sumstats:
    '''Utility class to record summary stastics for, e.g. frame flip timing'''
    def __init__(self):
        self.buf=[0]*(70*10) # ring-buffer, 700 entries
        self.N=0
        self.sx=0
        self.mu=-1
        self.sx2=0
        self.sigma2=-1
        self.minx=0
        self.maxx=0
    def addpoint(self,x):
        self.buf[self.N%len(self.buf)]=x # ring-buffer
        self.N=self.N+1
        self.sx=self.sx+x
        self.sx2=self.sx2+x*x
        self.minx=x if x<self.minx else self.minx
        self.maxx=x if x>self.maxx else self.maxx
    def hist(self):
        buf = self.buf[:min(len(self.buf),self.N)]
        self.minx=min(buf)
        self.maxx=max(buf)
        bstart=self.minx + (self.maxx-self.minx)*0
        bw    =(self.maxx-self.minx)*(.6-0)
        bins=[bstart+bw*(x/15.0) for x in range(0,15)]
        [hist,bins]=numpy.histogram(buf,bins)
        pp= " ".join("%5.2f"%((bins[i]+bins[i+1])/2) for i in range(len(bins)-1))
        pp+="\n" + " ".join("%5d"%t for t in hist)
        return pp
    def __str__(self):
        buf = self.buf[:min(len(self.buf),self.N)]
        mu = statistics.mean(buf)
        med= statistics.median(buf)
        sigma=statistics.stdev(buf) if len(buf)>2 else -1
        return "%f,%f (%f,[%f,%f])"%(mu,med,sigma,min(self.buf),max(self.buf))

def doFrame(t,stimState,tgtState=-1,objIDs=None,utopiaController=None):
    if tgtState>=0 :
        print("*" if tgtState>0 else ".",end='',flush=True)
    else:
        print('_',end='',flush=True)
    
if __name__ == "__main__":
    # make the noisetag object to manage the tagging selections
    ntexpt = Noisetag()
    # set the subset of active objects being displayed
    ntexpt.setnumActiveObjIDs(10)
    # tell it to play a full experiment sequence
    ntexpt.startExpt()
    # mainloop
    nframe=0
    while True:
        try :
            # update the stimulus state w.r.t. current time
            ntexpt.updateStimulusState(nframe)
            # get the stimulus state info we whould display
            ss,ts,objIDs,sendEvent=ntexpt.getStimulusState()
            # update the display drawing
            doFrame(nframe,ss,ts,objIDs)            
            # send info about what we did to the decoder
            ntexpt.sendStimulusState()
            # simulate waiting for the flip
            time.sleep(isi)
        except StopIteration :
            # this event is raised when the stimulus sequence is finished
            break
        nframe=nframe+1
