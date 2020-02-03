import time
import random
from .stimseq import StimSeq
from .utopiaController import UtopiaController
import os
scriptpath=os.path.dirname(os.path.realpath(__file__))
stimFile=os.path.join(scriptpath,'../codebooks/mgold_61_6521_psk_60hz.txt')
objIDs=list(range(1,10))
isi=1/60

class FSM:
    ''' simple finite state machine, using a generator-like pattern'''
    def next(self,t):
        '''update the current state, return the new state, raise StopIteration exception when done'''
        return t
    def get(self):
        '''get the current state'''
        return (None,None,None,None)

class GSM(FSM):
    '''Generalized state machine with stack of states'''
    def __init__(self): self.stack=[]
    def push(self,s):   self.stack.append(s); return s
    def pop(self):      return self.stack.pop()
    def next(self,t):
        '''get the next stimulus state to shown'''
        while self.stack :
            try : 
                return self.stack[-1].next(t)
            except StopIteration as e :
                # end of this fsm, unwind the fsm stack
                self.pop()
                # for pretty printing
                print()
        raise StopIteration()
    def get(self):
        return self.stack[-1].get()

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
        return (None,None,None,False) # BODGE: stimState,tgtState,objIDs,sendEvent
    
class Flicker(FSM):
    ''' do a normal flicker sequence'''
    def __init__(self,objIDs=8,stimSeq=None,
                 numframes=4*isi,tgtidx=None,
                 sendEvents=True):
        self.stimSeq=stimSeq
        if not type(objIDs) is list :
            objIDs=list(range(1,objIDs+1))
        self.objIDs=objIDs
        self.numframes=numframes
        self.nframe=0
        self.tgtidx=tgtidx
        self.sendEvents=sendEvents

        # ensure right length
        self.ss=self.stimSeq[self.nframe][:len(self.objIDs)]
        print('flicker: %d frames, tgt %d'%(self.numframes,tgtidx if tgtidx is not None else -1))
        
    def next(self,t):
        self.nframe=self.nframe+1
        if self.nframe>self.numframes:
            raise StopIteration()
        # extract the current frames stimulus state, loop if past end
        self.ss=self.stimSeq[self.nframe % len(self.stimSeq)][:len(self.objIDs)]
        
    def get(self):
        # TODO: return a dictionary directly?
        # convert the stim-state into a stack
        tgtstate = self.ss[self.tgtidx] if self.tgtidx is not None else -1
        return (self.ss,tgtstate,self.objIDs,self.sendEvents)

    
class FlickerWithSelection(Flicker):
    ''' do a normal flicker sequence, with early stopping selection'''
    def __init__(self,objIDs=8,stimSeq=None,numframes=4*isi,tgtidx=None,
                 utopiaController=None,
                 sendEvents=True):
        super().__init__(objIDs,stimSeq,numframes,tgtidx,sendEvents)
        self.utopiaController = utopiaController
        if self.utopiaController is None : raise RunTimeException()
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
        
class HighlightObject(Flicker):
    '''Highlight a single object for a number of frames'''
    def __init__(self,objIDs=8,numframes=isi*2,tgtidx=None,tgtState=2,sendEvents=False):
        # make a target on stim sequence
        if not type(objIDs) is list :
            objIDs=list(range(1,objIDs+1))
        stimSeq = [[0]*len(objIDs)]
        if tgtidx is not None:
            stimSeq[0][tgtidx]=tgtState
        super().__init__(objIDs,stimSeq,numframes,tgtidx,sendEvents=sendEvents)
        print('highlight: tgt=%d nframes=%d'%(tgtidx if tgtidx is not None else -1,numframes))

class SingleTrial(FSM):
    ''' do a complete single trial with: cue->wait->flicker->feedback '''
    def __init__(self,objIDs,tgtidx,stimSeq,
                 utopiaController,stimulusStateStack,
                 numframes=None,selectionThreshold=None,
                 duration=4,cueduration=1,feedbackduration=1,waitduration=1,
                 cueframes=None,feedbackframes=None,waitframes=None):
        self.objIDs=objIDs
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
        print("Cal: tgt=%d"%(self.tgtidx if self.tgtidx is not None else -1))
        
    def next(self,t):
        if self.stage==0 : # trial-start + cue
            # tell decoder to start trial
            self.utopiaController.newTarget()
            # tell decoder to clear predictions if needed
            if self.selectionThreshold is not None:
                self.utopiaController.selectionThreshold=self.selectionThreshold
                self.utopiaController.getNewMessages() # ensure all old predictions processed
                self.utopiaController.clearLastPrediction() # reset to no waiting predictions

            print('0.cue')
            if self.tgtidx is not None : # only if target is set
                self.stimulusStateStack.push(
                    HighlightObject(self.objIDs,self.cueframes,self.tgtidx,
                                    sendEvents=False))
            else : # skip cue+wait
                self.stage=1
                
        elif self.stage==1 : # wait
            print('1.wait')
            self.stimulusStateStack.push(
                HighlightObject(self.objIDs,self.waitframes,None,
                                sendEvents=False))
            
        elif self.stage==2 : # stim
            print('2.stim, tgt:%d'%(self.tgtidx if self.tgtidx is not None else -1))
            #objIDs=8,stimSeq=None,numframes=None,tgtidx=None,duration=4
            if self.selectionThreshold is not None: # early stop if thres set
                self.stimulusStateStack.push(
                    FlickerWithSelection(self.objIDs,self.stimSeq,
                                         self.numframes,self.tgtidx,
                                         self.utopiaController,
                                         sendEvents=True))
            else: # no selection based stopping
                self.stimulusStateStack.push(
                    Flicker(self.objIDs,self.stimSeq,
                            self.numframes,self.tgtidx,
                            sendEvents=True))
                
        elif self.stage==3 : # wait/feedback
            if self.selectionThreshold is None:
                print('3.wait')
                self.stimulusStateStack.push(
                    HighlightObject(self.objIDs,self.waitframes,None))
            else :
                print('3.feedback')
                predObjId,selected=self.utopiaController.getLastSelection()
                print(' pred:%d sel:%d'%(predObjId if predObjId else -1,selected))
                if selected :
                    tgtidx = self.objIDs.index(predObjId) if predObjId in self.objIDs else -1
                    self.stimulusStateStack.push(
                        HighlightObject(self.objIDs,self.feedbackframes,tgtidx,
                                        tgtState=3,
                                        sendEvents=False))
                    
        else :
            raise StopIteration
        self.stage=self.stage+1
        
class CalibrationPhase(FSM):
    ''' do a complete calibration phase with nTrials x CalibrationTrial '''
    def __init__(self,objIDs=8,stimSeq=None,nTrials=10,
                 utopiaController=None,stimulusStateStack=None,
                 *args,**kwargs):
        self.objIDs = objIDs
        self.stimSeq=stimSeq
        self.nTrials=nTrials
        self.utopiaController=utopiaController
        self.isRunning=False
        self.args=args
        self.kwargs=kwargs
        if self.utopiaController is None : raise RunTimeException
        self.stimulusStateStack=stimulusStateStack
        if self.stimulusStateStack is None : raise RunTimeException
        self.trli=0
    def next(self,t):
        if not self.isRunning :
            # tell decoder to start cal
            self.utopiaController.modeChange("Calibration.supervised")
            self.isRunning=True
        if self.trli<self.nTrials:
            self.tgtidx = random.randint(0,self.nTrials-1)
            print("Start Cal: %d/%d tgt=%d"%(self.trli,self.nTrials,self.tgtidx))
            self.stimulusStateStack.push(
                SingleTrial(self.objIDs,self.tgtidx,self.stimSeq,
                            self.utopiaController,
                            self.stimulusStateStack,
                            *self.args,**self.kwargs)) # pass through other arguments?
        else:
            raise StopIteration()
        self.trli=self.trli+1
            
class PredictionPhase(FSM):
    '''do complete prediction phase with nTrials x trials with early-stopping feedback'''
    def __init__(self,objIDs,stimSeq=None,nTrials=10,
                 utopiaController=None,stimulusStateStack=None,
                 selectionThreshold=.1,*args,**kwargs):
        self.objIDs=objIDs
        self.stimSeq=stimSeq
        self.nTrials=nTrials
        self.selectionThreshold=selectionThreshold
        self.args=args
        self.kwargs=kwargs
        self.tgti=0
        self.isRunning=False
        self.utopiaController=utopiaController
        if self.utopiaController is None : raise RunTimeException
        self.stimulusStateStack=stimulusStateStack
        if self.stimulusStateStack is None : raise RunTimeException
        # tell decoder to start cal
    def next(self,t):
        if not self.isRunning :
            self.utopiaController.modeChange("Prediction.static")
            self.isRunning=True
        self.tgti=self.tgti+1
        if self.tgti<self.nTrials:
            print("Start Pred: %d/%d"%(self.tgti,self.nTrials))
            self.stimulusStateStack.push(
                SingleTrial(self.objIDs,None,self.stimSeq,
                                 self.utopiaController,
                                 self.stimulusStateStack,
                                 selectionThreshold=self.selectionThreshold,
                                 *self.args,**self.kwargs))
        else:
            raise StopIteration()
        
class Experiment(FSM):
    '''do a complete experiment, with calibration -> prediction'''
    def __init__(self,objIDs=8,stimSeq=None,nCal=10,nPred=20,
                 selectionThreshold=.1,
                 utopiaController=None,stimulusStateStack=None,*args,**kwargs):
        self.objIDs=objIDs
        self.stimSeq=stimSeq
        self.nCal=nCal
        self.nPred=nPred
        self.selectionThreshold=selectionThreshold
        self.utopiaController=utopiaController
        if self.utopiaController is None : raise RunTimeException
        self.stimulusStateStack=stimulusStateStack
        if self.stimulusStateStack is None : raise RunTimeException
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
                                 *self.args,**self.kwargs))
        
        elif self.stage==2:
            self.stimulusStateStack.push(WaitFor(2/isi))
        
        elif self.stage==3:
            self.stimulusStateStack.push(
                PredictionPhase(self.objIDs,self.stimSeq,self.nPred,
                                self.utopiaController,
                                self.stimulusStateStack,
                                selectionThreshold=self.selectionThreshold,
                                *self.args,**self.kwargs))
        
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
        if self.utopiaController is None :
            # use the global controller if none given
            global uc
            if uc is None :
                # auto-connect the global controller if none given
                uc = UtopiaController()
                uc.autoconnect()                
            self.utopiaController=uc
    
        # stimulus state-machine stack
        # Individual stimulus-state-machines track progress in a single
        # stimulus state playback function.
        # Stack allows sequencing of sets of playback functions in loops
        if stimulusStateMachineStack is None :
            stimulusStateMachineStack=GSM()
        self.stimulusStateMachineStack=stimulusStateMachineStack
        self.laststate=(None,None,None,None)

    # stimulus sequence methods via the stimulus state machine stack
    def updateStimulusState(self,t=None):
        self.stimulusStateMachineStack.next(t)
        self.laststate = self.stimulusStateMachineStack.get()
        #print(self.laststate)
        return self.laststate

    def getStimulusState(self):
        return self.laststate
    
    # decoder interaction methods via. utopia controller
    def sendStimulusState(self,stimState=None,tgtState=None,objIDs=None,sendEvent=True,timestamp=None):
        if stimState is None : # get from last-state
            stimState,targetState,objIDs,sendEvent=self.laststate
        # send info about the stimulus displayed
        if sendEvent and stimState is not None :
            #print((stimState,targetState))
            self.utopiaController.sendStimulusEvent(stimState,objIDs=objIDs,targetState=tgtState,timestamp=timestamp)

    def getLastPrediction(self):
        if self.utopiaController :
            return self.utopiaController.getLastPrediction()
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
        return self.utopiaController.getTimeStamp(t0)


    # methods to define what (meta) stimulus sequence we will play
    def startExpt(self,objIDs,nCal=1,nPred=20,selnThreshold=.1,
                  *args,**kwargs):
        self.objIDs=objIDs
        self.stimulusStateMachineStack.push(
            Experiment(objIDs,self.noisecode.stimSeq,
                       nCal,nPred,
                       selnThreshold,
                       self.utopiaController,
                       self.stimulusStateMachineStack,
                       *args,**kwargs))
        
    def startCalibration(self,objIDs,nTrials=10,stimSeq=None,
                         *args,**kwargs):
        self.objIDs=objIDs
        self.stimulusStateMachineStack.push(
            CalibrationPhase(objIDs,self.noisecode.stimSeq,
                             nTrials,
                             self.utopiaController,
                             self.stimulusStateMachineStack,
                             *args,**kwargs))
        
    def startPrediction(self,objIDs,nTrials=10,stimSeq=None,
                        *args,**kwargs):
        self.objIDs=objIDs
        self.stimulusStateMachineStack.push(
            PredictionPhase(objIDs,self.noisecode.stimSeq,
                            nTrials,
                            self.utopiaController,
                            self.stimulusStateMachineStack,
                            *args,**kwargs))
        
    def startFlicker(self,objIDs,numframes=100,tgtidx=None,*args,**kwargs):
        self.objIDs=objIDs
        self.stimulusStateMachineStack.push(
            Flicker(objIDs,self.noisecode.stimSeq,numframes,tgtidx,*args,**kwargs))
        
    def startFlickerWithSelection(self,objIDs,numframes=100,
                                  tgtidx=None,*args,**kwargs):
        self.objIDs=objIDs
        self.stimulusStateMachineStack.push(
            FlickerWithSelection(objIDs,self.noisecode.stimSeq,
                                 numframes,tgtidx,
                                 self.utopiaController,
                                 *args,**kwargs))

def doFrame(t,stimState,tgtState=None,objIDs=None,utopiaController=None):
    if tgtState is not None:
        print('%d(%d)'%(t,tgtState),end='',flush=True)
    else:
        print('%d'%(t),end='',flush=True)
    
if __name__ == "__main__":
    # make the noisetag object to manage the tagging selections
    ntexpt = Noisetag()
    # tell it to play a full experiment sequence
    ntexpt.startExpt(objIDs)
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
            print('.',end='',flush=True)
        except StopIteration :
            # this event is raised when the stimulus sequence is finished
            break
        nframe=nframe+1
