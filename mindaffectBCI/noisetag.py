#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V.
#  Author: Jason Farquhar <jadref@gmail.com>
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

#import mindaffectBCI
import time
import random
from mindaffectBCI.decoder.utils import search_directories_for_file, import_and_make_class
from mindaffectBCI.stimseq import StimSeq
from mindaffectBCI.utopiaController import UtopiaController, TimeStampClock
import os
scriptpath = os.path.dirname(os.path.realpath(__file__))
default_stimFile = os.path.join('mgold_61_6521_psk_60hz.txt')
objIDs = list(range(1, 10))
isi = 1/60
MAXOBJID = 64


class FSM:
    ''' simple finite state machine, using a generator-like pattern'''

    def next(self, t):
        '''update the current state, return the new state, raise StopIteration exception when done'''
        return t

    def get(self):
        """get the current state, for this set of active objects

        Returns:
            tuple : The current display state information as a 4-tuple with structure:
                (stimState, target_idx, objIDs, sendEvent)  
                stimState (list-int): the stimulus state for each object as an integer
                target_idx (int): the index into stimState of the cued target, or -1 if no target set
                objIDs (list-int): the objectIDs for each of the outputs
                sendEvent (bool): flag if we should send stimulus events in this state
        """
        return (None, -1, None, False)  # BODGE: stimState,tgtState,objIDs,sendEvent


class GSM(FSM):
    '''Generalized state machine with stack of states'''

    def __init__(self): self.stack = []

    def clear(self):
        """clear the state machine stack
        """
        self.stack = []

    def push(self, s):
        """add a new state machine to the stack

        Args:
            s (GSM): finite-state-machine to add

        Returns:
            list: current state machine stack 
        """
        self.stack.append(s)
        return self.stack

    def pop(self):
        """remove and return the currently active state machine

        Returns:
            GSM: the current state machine at the stop of the stack
        """
        return self.stack.pop()

    def next(self, t):
        """get the next stimulus state to shown

        Args:
            t (int): the current time

        Raises:
            StopIteration: when this state machine has run out of states
        """
        while self.stack:
            try:
                self.stack[-1].next(t)
                return
            except StopIteration:
                # end of this fsm, unwind the fsm stack
                self.pop()
                # for pretty printing
                print()
        raise StopIteration()

    def get(self):
        """return the current stimulus state

        Returns:
            StimulusState: the current stimulus state tuple (stimState,target_idx,objIDs,sendEvent)
        """
        if self.stack:
            return self.stack[-1].get()
        else:
            return None


class WaitFor(FSM):
    ''' state machine which waits for given number of frames to pass'''

    def __init__(self, numframes):
        """state machine which waits for given number of frames to pass

        Args:
            numframes (int): the number of frames to wait for
        """
        self.numframes = numframes
        self.nframe = 0
        print("waitFor: %g" % (self.numframes))

    def next(self, t):
        """stop after the desired number of frames has passed

        Args:
            t (int): current time stamp

        Raises:
            StopIteration: when desired number frames expired
        """
        self.nframe = self.nframe+1
        if self.nframe > self.numframes:
            raise StopIteration()

    def get(self):
        return (None, -1, None, False)  # BODGE: stimState,tgtState,objIDs,sendEvent


class Flicker(FSM):
    ''' do a normal flicker sequence'''

    # class variable to store frame counter information across calls
    lastframe: int = 0

    def __init__(self, stimSeq=None,
                 numframes: int = int(4/isi),
                 tgtidx: int = -1,
                 sendEvents: bool = True,
                 framesperbit: int = 1,
                 permute: bool = False,
                 startframe: int = 0):
        """Object to provide state information for a flicker sequence

        Args:
            stimSeq (list-of-lists, optional): (time,outputs) the stimulus sequence to use. Defaults to None.
            numframes (int, optional): number for frames to flicker for. Defaults to 4*isi.
            tgtidx (int, optional): the index into stimSeq of the target output, -1 for no target. Defaults to -1
            sendEvents (bool, optional): should we send stimulus events.  Defaults to True
            framesperbit (int, optional): number of video-frames, i.e. calls to next, per stimsequence bit.  Defaults to 1.
            permute (bool, optional): flag, should we permute the codebook to output mapping.  Defaults to False.
            startframe (int, optional): starting frame number in the stimulus sequence.  If set to 'random' then choose a random starting point.  If set to 'lastframe' continue from the last frame for which 'sendEvents' was true. Defaults to 0.
        """
        self.stimSeq, self.numframes, self.tgtidx, self.sendEvents, self.permute = (
            stimSeq, numframes, tgtidx, sendEvents, permute)
        if startframe == 'random':
            self.startframe = random.randint(0, len(self.stimSeq) if hasattr(self.stimSeq, '__len__') else 0)
            print('Start at frame: {}'.format(self.startframe))
        elif startframe == 'lastframe':
            self.startframe = Flicker.lastframe
            print('Start at frame : {}'.format(self.startframe))
        else:
            self.startframe = startframe
        self.nframe = 0
        self.framesperbit = framesperbit if framesperbit is not None else 1
        if self.permute == True:
            self.update_codebook_permutation()

        # ensure right length
        self.ss = None
        # print("stimSeq:")
        # for i in range(len(self.objIDs)):
        #    print(["objID %d = %s"%(i,"".join([ '*' if self.stimSeq[t][i]==1 else '.' for t in range(len(self.stimSeq))]))])
        print('flicker: %d frames, tgt %d' % (self.numframes, tgtidx if tgtidx >= 0 else -1))

    def update_codebook_permutation(self):
        """update the permutation randomly mapping between codebook rows and outputs
        """
        self.codebook_permutation = list(range(len(self.stimSeq[0])))  # linear index
        random.shuffle(self.codebook_permutation)  # N.B. in-place permutation

    def next(self, t):
        """move to the next state
        """

        self.nframe = self.nframe+1
        # do wrap-around specific processing
        if not hasattr(self.stimSeq, 'next'):
            if len(self.stimSeq) > 1 and self.nframe % (len(self.stimSeq) * self.framesperbit) == 0:
                if self.permute:
                    #print("Update Permutation: {} {}".format(self.nframe, len(self.stimSeq)))
                    self.update_codebook_permutation()
        if self.nframe > self.numframes:
            # record the current frame number for continuation later (if wanted)
            if self.sendEvents:
                Flicker.lastframe = self.startframe + self.nframe
            raise StopIteration()

    def update_ss(self):
        """update the information about the current stimulus state
        """
        # extract the current frames stimulus state, loop if past end
        if self.nframe >= self.numframes:
            # final frame is blank screen
            self.ss = [0 for i in range(len(self.ss))]
            return

        if hasattr(self.stimSeq, 'next'):
            self.ss = self.stimSeq.next()
        else:
            self.ss = self.stimSeq[(self.startframe+self.nframe)//self.framesperbit % len(self.stimSeq)]
        if self.permute:  # permute the codebook -> objID mapping if wanted
            self.ss = [self.ss[p] for p in self.codebook_permutation]

    def get(self):
        """ return the curent stimulus state info

        Returns:
            StimulusState: the current stimlus state tuple (stimState, target_idx, objIDs, sendEvent)
        """

        self.update_ss()
        # default to the stored set active objIDs
        #if objIDs is None : objIDs=self.objIDs
        # get the state info for the given set of objectIDs
        #ss =[self.ss[i] for i in objIDs-1]
        return (self.ss, self.tgtidx, None, self.sendEvents)


class FlickerWithSelection(Flicker):
    """ do a normal flicker sequence, with early stopping selection
    """

    def __init__(self,
                 stimSeq=None,
                 numframes=4*isi,
                 tgtidx=-1,
                 utopiaController=None,
                 framesperbit=1,
                 sendEvents=True, permute=False, startframe: int = 0):
        """Object to provide state information for a flicker sequence, with early-stopping on selection

        Args:
            stimSeq (list-of-lists, optional): (time,outputs) the stimulus sequence to use. Defaults to None.
            numframes (int, optional): number for frames to flicker for. Defaults to 4*isi.
            tgtidx (int, optional): the index in stimSeq of the cued target output. Defaults to -1.
            utopiaController (UtopiaController, optional): The utopiaController for interfacing to the mindaffectBCI. Defaults to None.
            framesperbit (int, optional): number of video frames per codebook bit. Defaults to 1.
            sendEvents (bool, optional): do we send stimulus events for this sequence. Defaults to True.
            permute (bool, optional): flag, should we permute the codebook to output mapping.  Defaults to False.
            startframe (int, optional): frame number to start stim-sequence at. Defaults to 0.
        Raises:
            ValueError: if invalid utopia controller
        """
        super().__init__(stimSeq, numframes, tgtidx, sendEvents, framesperbit, permute, startframe)
        self.utopiaController = utopiaController
        if self.utopiaController is None:
            raise ValueError("must have utopiaController")
        # ensure old predictions are gone..
        self.utopiaController.getNewMessages()  # ensure all old predictions processed
        self.utopiaController.clearLastPrediction()  # reset to no waiting predictions
        print(' with selection')

    def next(self, t):
        """get the next stimulus state in the sequence -- terminate if out of time, or BCI made a selection

        Args:
            t (int): current time

        Raises:
            StopIteration: stop if numframes exceeded or BCI made a selection
        """
        super().next(t)
        # check for selection and stop if found
        objId, selected = self.utopiaController.getLastSelection()
        if selected:
            if self.sendEvents:
                # send event to say selection has occured
                self.utopiaController.selection(objId)
            raise StopIteration()


def mkBlinkingSequence(numframes, tgtidx, tgtState=1):
    """make a simple on-off blinking sequence

    Args:
        numframes (int): number of frame to blink for
        tgtidx (int): the index of the output to blink
        tgtState (int, optional): the state to show for the blinking target. Defaults to 1.

    Returns:
        list-of-lists: (numframes,255) stimulus state for each output at each timepoint
    """

    blinkSeq = [[0 for i in range(MAXOBJID)] for i in range(numframes)]
    for i in range(int(numframes/2)):
        blinkSeq[i][tgtidx] = tgtState  # tgt on
    return blinkSeq


class HighlightObject(Flicker):
    """'Highlight a single object for a number of frames
    """

    def __init__(self, numframes=isi*2, tgtidx=-1, tgtState=254,
                 sendEvents=False, numblinkframes=int(.5/isi)):
        """Highlight a single object for a number of frames

        Args:
            numframes (int, optional): number of frames to highlight for. Defaults to isi*2.
            tgtidx (int, optional): the index of the target output. Defaults to -1.
            tgtState (int, optional): the state for the highlighted target. Defaults to 2.
            sendEvents (bool, optional): flag if we send stimulus info to the decoder. Defaults to False.
            numblinkframes ([type], optional): number of frames to blink on (and off). Defaults to int(.5/isi).
        """
        #self.objIDs=objIDs if hasattr(objIDs, "__len__") else list(range(1,objIDs+1))
        if numblinkframes > 0 and tgtidx >= 0:  # blinking cue
            stimSeq = mkBlinkingSequence(int(numblinkframes), tgtidx, tgtState)
        else:
            stimSeq = [[0]*MAXOBJID]
            if tgtidx >= 0:
                stimSeq[0][tgtidx] = tgtState
        super().__init__(stimSeq, numframes, tgtidx, sendEvents=sendEvents, permute=False)
        print('highlight: tgtidx=%d nframes=%d' % (tgtidx if tgtidx >= 0 else -1, numframes))


class SingleTrial(FSM):
    """do a complete single trial with: cue->wait->flicker->feedback
    """

    def __init__(self, stimSeq, tgtidx: int, utopiaController, stimulusStateMachineStack, numframes: int = None,
                 framesperbit: int = 1, selectionThreshold: float = None, cuestate: int = 254, feedbackstate: int = 255,
                 duration: float = 4, trialduration: float = None, cueduration: float = 1, feedbackduration: float = 1,
                 waitduration: float = 1, intertrialduration: float = 1, cueframes: int = None, feedbackframes: int = None,
                 waitframes: int = None, intertrialframes: int = None, permute: bool = False, startframe: int = 0,
                 sendEvents: bool = True):
        """do a complete single trial with: cue->wait->flicker->feedback

        Args:
            stimSeq (list-of-lists): (time,outputs)  the stimulus Sequence matrix to play
            tgtidx (int): the index in stimSeq of the cued target output.
            utopiaController (UtopiaController): the utopia controller for interaction with the decoder
            stimulusStateMachineStack (GSM): the stimulus state stack to which we add state machines
            numframes (int, optional): the number of frames to flicker for. Defaults to None.
            framesperbit (int, optional): the number of video frames per stimSeq time-points. Defaults to 1.
            selectionThreshold (float, optional): Target error probability for selection. Defaults to None.
            duration (int, optional): flicker duration in seconds. Defaults to 4.
            cueduration (int, optional): cue duration in seconds. Defaults to 1.
            feedbackduration (int, optional): feedback duration in seconds. Defaults to 1.
            waitduration (int, optional): wait duration in seconds. Defaults to 1.
            cueframes (int, optional): cue duration in frames. Defaults to None.
            feedbackframes (int, optional): feedback duration in frames. Defaults to None.
            waitframes (int, optional): wait duration in frames. Defaults to None.
            permute (bool, optional): if true then permute the order of objects before running the flicker sequence.  Default to False
            startframe (int, optional): set the frame in the sequence to start the flicker from.  if 'random' then pick a random starting point. If 'lastframe' then continue from the last played frame.  Defaults to 0.
            sendEvents (bool, optoinal): if true then send event info to the Hub, if false then do not send events.  For example used for for practice / debugging. Defaults to True.
        """
        self.stimSeq, self.tgtidx, self.utopiaController, self.stimulusStateMachineStack, self.selectionThreshold, self.permute, self.startframe, self.cuestate, self.feedbackstate, self.sendEvents = (
            stimSeq, tgtidx, utopiaController, stimulusStateMachineStack, selectionThreshold, permute, startframe, cuestate, feedbackstate, sendEvents)
        self.numframes = numframes if numframes else (duration/isi if duration is not None else trialduration/isi)
        self.cueframes = cueframes if cueframes else cueduration/isi
        self.framesperbit = framesperbit if framesperbit is not None else 1
        self.feedbackframes = feedbackframes if feedbackframes else feedbackduration/isi
        self.waitframes = waitframes if waitframes else waitduration/isi
        self.intertrialframes = intertrialframes if intertrialframes else intertrialduration/isi
        self.stage = 0
        self.stagestart = self.utopiaController.getTimeStamp()
        print("tgtidx=%d" % (self.tgtidx if self.tgtidx >= 0 else -1))

    def next(self, t):
        """get the next state in the sequence, moving through the single trail stages of cue->wait->flicker->feedback

        Args:
            t (int): current time

        Raises:
            StopIteration: when the whole sequence is complete
        """

        last_stage_dur = self.utopiaController.getTimeStamp()-self.stagestart
        if self.stage == 0:  # trial-start + cue
            # tell decoder to start trial
            if self.sendEvents:
                self.utopiaController.newTarget()
            # tell decoder to clear predictions if needed
            if self.selectionThreshold is not None:
                self.utopiaController.selectionThreshold = self.selectionThreshold
                self.utopiaController.clearLastPrediction()  # reset to no waiting predictions

            if self.tgtidx >= 0:  # blinking: only if target is set
                print('\n0.start/cue {}'.format(self.cueframes))
                self.stimulusStateMachineStack.push(
                    HighlightObject(self.cueframes,
                                    self.tgtidx,
                                    tgtState=self.cuestate,
                                    sendEvents=False))
            else:  # skip cue+wait
                self.stage = 1

        elif self.stage == 1:  # wait
            print('1.wait {}'.format(self.waitframes))
            self.stimulusStateMachineStack.push(
                HighlightObject(self.waitframes, -1,
                                sendEvents=False))

        elif self.stage == 2:  # stim
            print('2.stim, tgt:%d   %dframes' % (self.tgtidx if self.tgtidx >= 0 else -1, self.numframes))
            # objIDs=8,stimSeq=None,numframes=None,tgtidx=None,duration=4
            if self.selectionThreshold is not None:  # early stop if thres set
                self.stimulusStateMachineStack.push(
                    FlickerWithSelection(self.stimSeq,
                                         self.numframes,
                                         self.tgtidx,
                                         self.utopiaController,
                                         framesperbit=self.framesperbit,
                                         sendEvents=self.sendEvents, permute=self.permute,
                                         startframe=self.startframe))
            else:  # no selection based stopping
                self.stimulusStateMachineStack.push(
                    Flicker(self.stimSeq,
                            self.numframes,
                            self.tgtidx,
                            framesperbit=self.framesperbit,
                            sendEvents=self.sendEvents, startframe=self.startframe))

        elif self.stage == 3:  # wait/feedback
            if self.selectionThreshold is None:
                print('3.wait {}'.format(self.intertrialframes))
                self.stimulusStateMachineStack.push(
                    HighlightObject(self.intertrialframes, -1))
            else:
                print('3.feedback {}'.format(self.feedbackframes))  # solid on
                predObjId, selected = self.utopiaController.getLastSelection()
                print('%dms pred:%d sel:%d  %c' % (int(last_stage_dur),
                                                   predObjId-1 if predObjId else -1, selected,
                                                   "*" if predObjId and self.tgtidx == predObjId-1 else "x"))
                if selected:
                    #tgtidx = self.objIDs.index(predObjId) if predObjId in self.objIDs else -1
                    tgtidx = predObjId-1
                    self.stimulusStateMachineStack.push(
                        HighlightObject(self.feedbackframes,
                                        tgtidx,
                                        tgtState=self.feedbackstate,
                                        sendEvents=False,
                                        numblinkframes=0))
                else:  # no selection do the intertrial
                    print('3.wait {}'.format(self.intertrialframes))
                    self.stimulusStateMachineStack.push(
                        HighlightObject(self.intertrialframes, -1))

        else:
            raise StopIteration
        self.stagestart = self.utopiaController.getTimeStamp()
        self.stage = self.stage+1


class CalibrationPhase(FSM):
    """do a complete calibration phase with nTrials x CalibrationTrial

    Raises:
        ValueError: [description]
        ValueError: [description]
        StopIteration: [description]
    """

    def __init__(self, objIDs: int = 8, stimSeq=None, nTrials: int = 10,
                 utopiaController=None, stimulusStateMachineStack: GSM = None,
                 *args, **kwargs):
        """do a complete calibration phase with nTrials x CalibrationTrial

        Args:
            objIDs (int, optional): the number of output objects. Defaults to 8.
            stimSeq (list-of-lists, optional): (time,outputs) the stimulus sequence to play for the flicker. Defaults to None.
            nTrials (int | list, optional): the number of calibration trials  OR list of target IDs to give in order. Defaults to 10.
            utopiaController (UtopiaController, optional): the utopiaController for interfacing to the decoder. Defaults to None.
            stimulusStateMachineStack (GSM, optional): the state-machine-stack to add new machines to for playback. Defaults to None.

        Raises:
            ValueError: if utopiaController or stimulusStateMachine is not given
        """
        if objIDs is None:  objIDs=8
        self.objIDs = objIDs if hasattr(objIDs, "__len__") else list(range(1, objIDs+1))
        self.stimSeq = stimSeq
        self.trialSeq = [-1]*nTrials if not hasattr(nTrials, '__len__') else nTrials
        self.utopiaController = utopiaController
        self.isRunning = False
        self.args = args
        self.kwargs = kwargs
        if self.utopiaController is None:
            raise ValueError
        self.stimulusStateMachineStack = stimulusStateMachineStack
        if self.stimulusStateMachineStack is None:
            raise ValueError
        self.trli = 0
        self.tgtidx = -1

    def next(self, t):
        """get the next state in the sequence, moving through nTrials calibration trials where each trial has single trail stages of cue->wait->flicker

        Args:
            t (int): current time

        Raises:
            StopIteration: when the whole sequence is complete
        """
        if not self.isRunning:
            # tell decoder to start cal
            self.utopiaController.modeChange("Calibration.supervised")
            self.isRunning = True
        if self.trli < len(self.trialSeq):
            # TODO []: should choose from set active objIDs?
            tgtidx = self.trialSeq[self.trli]
            if tgtidx < 0:  # insert random target
                tgtidx = random.randint(0, len(self.objIDs)-1)
                if tgtidx == self.tgtidx:
                    tgtidx = (tgtidx+1) % len(self.objIDs)
                self.trialSeq[self.trli] = tgtidx
            self.tgtidx = tgtidx
            print("Start Cal: %d/%d tgtidx=%d" % (self.trli, len(self.trialSeq), self.tgtidx))
            self.stimulusStateMachineStack.push(
                SingleTrial(self.stimSeq,
                            self.tgtidx,
                            self.utopiaController,
                            self.stimulusStateMachineStack,
                            *self.args, **self.kwargs))  # pass through other arguments?
        else:
            self.utopiaController.modeChange("idle")
            raise StopIteration()
        self.trli = self.trli+1


class PredictionPhase(FSM):
    """do complete prediction phase with nTrials x trials with early-stopping feedback

    Args:
        FSM ([type]): [description]
    """

    def __init__(self, objIDs, stimSeq=None, nTrials=10,
                 utopiaController=None, stimulusStateMachineStack=None,
                 selectionThreshold=.1, cuedprediction=False, **kwargs):
        """do a complete calibration phase with nTrials x CalibrationTrial

        Args:
            objIDs (int, optional): the number of output objects. Defaults to 8.
            stimSeq (list-of-lists, optional): (time,outputs) the stimulus sequence to play for the flicker. Defaults to None.
            nTrials (int|list, optional): the number of calibration trials OR list of target IDs to give in order. Defaults to 10.
            utopiaController (UtopiaController, optional): the utopiaController for interfacing to the decoder. Defaults to None.
            stimulusStateMachineStack (GSM, optional): the state-machine-stack to add new machines to for playback. Defaults to None.
            selectionThreshold (float, optional): the Perr threshold for selection. Defaults to .1
            cuedprediction (bool, optional): flag if we do cueing before trial starts.  Default to False.

        Raises:
            ValueError: if utopiaController or stimulusStateMachine is not given
        """
        if objIDs is None:  objIDs=10
        self.objIDs = objIDs if hasattr(objIDs, "__iter__") else list(range(1, objIDs+1))
        self.stimSeq = stimSeq
        self.trialSeq = [-1]*nTrials if not hasattr(nTrials, '__iter__') else nTrials
        self.selectionThreshold = selectionThreshold
        self.cuedprediction = cuedprediction
        self.kwargs = kwargs
        self.tgti = 0
        self.tgtidx = -1
        self.isRunning = False
        self.utopiaController = utopiaController
        if self.utopiaController is None:
            raise ValueError
        self.stimulusStateMachineStack = stimulusStateMachineStack
        if self.stimulusStateMachineStack is None:
            raise ValueError
        # tell decoder to start cal

    def next(self, t):
        """get the next state in the sequence, moving through nTrials calibration trials where each trial has single trail stages of cue->wait->flicker->feedback

        Args:
            t (int): current time

        Raises:
            StopIteration: when the whole sequence is complete
        """
        if not self.isRunning:
            self.utopiaController.modeChange("Prediction.static")
            self.isRunning = True
        if self.tgti < len(self.trialSeq):
            self.tgtidx = self.trialSeq[self.tgti]
            if self.tgtidx < 0 and self.cuedprediction:
                tgtidx = random.randint(0, len(self.objIDs)-1)
                self.tgtidx = tgtidx if not tgtidx == self.tgtidx else (tgtidx+1) % len(self.objIDs)
                self.trialSeq[self.tgti] = self.tgtidx
            print("Start Pred: %d/%d" % (self.tgti, len(self.trialSeq)))
            self.stimulusStateMachineStack.push(
                SingleTrial(self.stimSeq,
                            self.tgtidx,
                            utopiaController=self.utopiaController,
                            stimulusStateMachineStack=self.stimulusStateMachineStack,
                            selectionThreshold=self.selectionThreshold,
                            **self.kwargs))
        else:
            self.utopiaController.modeChange("idle")
            raise StopIteration()
        self.tgti = self.tgti+1


class Experiment(FSM):
    """do a complete experiment, with calibration -> prediction
    """

    def __init__(self, objIDs, stimSeq=None, nCal=10, nPred=20,
                 selectionThreshold=.1, cuedprediction=False,
                 utopiaController=None, stimulusStateMachineStack=None,
                 numframes=4//isi, calframes=None, predframes=None,
                 interphaseframes=15//isi,
                 *args, **kwargs):
        """do a complete experiment, with calibration -> prediction

        Args:
            objIDs (int, optional): the number of output objects. Defaults to 8.
            stimSeq (list-of-lists, optional): (time,outputs) the stimulus sequence to play for the flicker. Defaults to None.
            nCal (int, optional): the number of calibration trials. Defaults to 10.
            nPred (int, optional): the number of prediction trials. Defaults to 20.
            utopiaController (UtopiaController, optional): the utopiaController for interfacing to the decoder. Defaults to None.
            stimulusStateMachineStack (GSM, optional): the state-machine-stack to add new machines to for playback. Defaults to None.
            selectionThreshold (float, optional): the Perr threshold for selection. Defaults to .1
            cuedprediction (bool, optional): flag if we do cueing before trial starts.  Default to False.
            selectionThreshold (float, optional): [description]. Defaults to .1.
            cuedprediction (bool, optional): [description]. Defaults to False.
            utopiaController ([type], optional): [description]. Defaults to None.
            stimulusStateMachineStack ([type], optional): [description]. Defaults to None.
            calduration (int, optional): the duration of the calibration flicker. Defaults to 4.
            predduration (int, optional): the duration of the prediction flicker. Defaults to 10.

        Raises:
            ValueError: if the utopiaController or stimulusStateMachineStack is not given
        """
        if objIDs is None:  objIDs=8
        self.objIDs = objIDs if hasattr(objIDs, "__len__") else list(range(1, objIDs+1))
        self.stimSeq = stimSeq
        self.nCal = nCal
        self.nPred = nPred
        self.calframes = calframes if calframes else numframes
        self.predframes = predframes if predframes else numframes
        self.interphaseframes = interphaseframes
        self.selectionThreshold = selectionThreshold
        self.cuedprediction = cuedprediction
        self.utopiaController = utopiaController
        if self.utopiaController is None:
            raise ValueError
        self.stimulusStateMachineStack = stimulusStateMachineStack
        if self.stimulusStateMachineStack is None:
            raise ValueError
        self.args = args
        self.kwargs = kwargs
        self.stage = 0

    def next(self, t):
        """get the next state in the sequence, moving through calibration->prediction phases where each trial has single trail stages of cue->wait->flicker->feedback

        Args:
            t (int): current time

        Raises:
            StopIteration: when the whole sequence is complete
        """
        if self.stage == 0:  # start
            self.stimulusStateMachineStack.push(WaitFor(2/isi))

        elif self.stage == 1:  # calibration
            self.stimulusStateMachineStack.push(
                CalibrationPhase(self.objIDs, self.stimSeq, self.nCal,
                                 self.utopiaController,
                                 self.stimulusStateMachineStack,
                                 *self.args, numframes=self.calframes, **self.kwargs))

        elif self.stage == 2:  # wait-for-classifier
            # TODO []: do a correct wait for classifier prediction message
            self.stimulusStateMachineStack.push(WaitFor(self.interphaseframes))

        elif self.stage == 3:  # prediction
            self.stimulusStateMachineStack.push(
                PredictionPhase(self.objIDs, self.stimSeq, self.nPred,
                                self.utopiaController,
                                self.stimulusStateMachineStack,
                                self.selectionThreshold,
                                self.cuedprediction,
                                *self.args, numframes=self.predframes, **self.kwargs))

        else:  # finish
            raise StopIteration()
        self.stage = self.stage+1


# use global for the noisecode + utopiaController so can easily
# share over instantiations of the NoiseTag framework.
uc = None


class Noisetag:
    def __init__(
            self, stimSeq: StimSeq = None, stimFile=None, utopiaController=None, stimulusStateMachineStack: GSM = None,
            clientid: str = None):
        """noisetag abstraction layer to handle *both* the sequencing of the stimulus flicker, *and* the communications with the Mindaffect decoder.

        noisetag abstraction layer to handle *both* the sequencing of the stimulus
        flicker, *and* the communications with the Mindaffect decoder.  Clients can
        use this class to implement BCI control by:
        0) setting the flicker sequence to use (method: startFlicker, startFlickerWithSelection, startCalibration, startPrediction, startExpt)
        1) getting the current stimulus state (method: getStimulusState), and using that to draw the display
        2) telling Noisetag when *exactly* the stimulus update took place (method: sendStimulusState)
        3) getting the predictions/selections from noisetag and acting on them. (method: getLastPrediction() or getLastSelection())

        Args:
            stimSeq (str file-like, optional): the file to load the stimulus sequence from. Defaults to None.
            utopiaController (UtopiaController, optional): the controller for interfacing to the decoder. Defaults to None.
            stimulusStateMachineStack (GSM, optional): The state machine stack to add stimulus state machines too. Defaults to None.
            clientid (str, optional): a label for this client for communications to the decoder. Defaults to None.
        """
        # global flicker stimulus sequence
        if stimSeq is None:
            stimSeq = stimFile
        self.set_stimSeq(stimSeq)

        # stimulus state-machine stack
        # Individual stimulus-state-machines track progress in a single
        # stimulus state playback function.
        # Stack allows sequencing of sets of playback functions in loops
        if stimulusStateMachineStack is None:
            stimulusStateMachineStack = GSM()
        self.stimulusStateMachineStack = stimulusStateMachineStack
        self.laststate = (None, None, None, None)
        self.objIDs = None
        self.clientid = clientid

        # utopiaController
        self.init_utopiaController(utopiaController)


    def set_stimSeq(self, stimseq: StimSeq = None):
        if stimseq is None:
            stimseq = default_stimFile
        if isinstance(stimseq, str):
            stimseq = StimSeq.fromFile(stimseq)
        elif hasattr(stimseq, '__iter__'):
            if len(stimseq) == 2 and isinstance(stimseq[0], str):
                stimseq = import_and_make_class(stimseq[0], **stimseq[1])
            else:
                stimseq = StimSeq(None, stimseq)
        # binarize the result
        if isinstance(stimseq, StimSeq):
            stimseq.convertstimSeq2int()
            stimseq = stimseq.stimSeq
        self.noisecode = stimseq

    def get_stimSeq(self):
        return self.noisecode

    def init_utopiaController(self, utopiaController):
        if utopiaController is None:
            # use the global controller if none given
            global uc
            if uc is None:
                # auto-connect the global controller if none given
                uc = UtopiaController(clientid=self.clientid)
            self.utopiaController = uc
        else:
            self.utopiaController = utopiaController

    def connect(self, host: str = None, port: int = -1, queryifhostnotfound: bool = True, timeout_ms: int = 5000):
        """connect to the utopia hub

        Args:
            host (str, optional): the hub hostname or ip address. Defaults to None.
            port (int, optional): the hub port. Defaults to -1.
            queryifhostnotfound (bool, optional): if auto-discovery fails do we query the user for the host IP. Defaults to True.
            timeout_ms (int, optional): timeout in milliseconds for host autodiscovery or connection. Defaults to 5000.

        Returns:
            bool: are we currently connected to the hub
        """
        if self.utopiaController.isConnected():
            return True
        self.utopiaController.autoconnect(host, port,
                                          queryifhostnotfound=queryifhostnotfound,
                                          localhostifhostnotfound=True,
                                          timeout_ms=timeout_ms)
        return self.utopiaController.isConnected()

    def isConnected(self):
        """query the hub connection status

        Returns:
            bool: are we connected to the hub
        """

        if self.utopiaController:
            return self.utopiaController.isConnected()
        return False

    def gethostport(self):
        """return the hostname:port we are currently connected to

        Returns:
            str: the hub host:port we are currently connected to
        """

        if self.utopiaController:
            return self.utopiaController.gethostport()
        return None

    def set_isi(self, new_isi: float):
        global isi
        isi = new_isi
        return isi

    def isRunning(self):
        """return if we are currently running a stimulus sequence

        Returns:
            bool: true if we are currently running a stimulus sequence
        """
        return self.stimulusStateMachineStack.stack


    # stimulus sequence methods via the stimulus state machine stack
    # returns if sequence is still running
    def updateStimulusState(self, t=None):
        """ update to the next stimulus state from the current sequence

        Args:
            t (int, optional): the current time. Defaults to None.
        """

        self.stimulusStateMachineStack.next(t)

    def getStimulusState(self, objIDs=None):
        """return the current stimlus state

        Args:
            objIDs (list-of-int, optional): the set of objectIDs to get the state information for. Defaults to None.

        Returns:
            StimulusState: the current stimulus state tuple (stimState,target_idx,objIDs,sendEvent)
        """

        # update set active objects if a set is given
        if objIDs is not None:
            self.setActiveObjIDs(objIDs)
        # get the complete stimulus state (for MAXOBIDS objects)
        stimState, tgtidx, objIDs, sendEvents = self.stimulusStateMachineStack.get()
        # subset to the active set, matching objIDs to allobjIDs
        # N.B. objID-1 to map from objID->stimStateIndex
        if stimState is not None and self.objIDs is not None:
            stimState = [stimState[i-1] for i in self.objIDs]
        self.laststate = (stimState, tgtidx, self.objIDs, sendEvents)
        return self.laststate

    def getNextStimulusState(self, objIDs=None):
        self.updateStimulusState()
        return self.getStimulusState(objIDs=objIDs)


    def setActiveObjIDs(self, objIDs):
        """update the set of active objects we send info to decoder about

        Args:
            objIDs (list-of-int): the set of objectIDs to register

        Returns:
            list-of-int: the set of active object IDs
        """

        self.objIDs = objIDs
        return self.objIDs

    def setnumActiveObjIDs(self, nobj: int):
        """update to say number active objects

        Args:
            nobj (int): the number of active objects to set

        Returns:
            list-of-int: the current set of active object IDs
        """
        objIDs = list(range(1, nobj+1))
        return self.setActiveObjIDs(objIDs)

    # decoder interaction methods via. utopia controller
    def encodeFloatState(self, state, mins: float = 0.0, maxs: float = 1.0, mini: int = 0, maxi: int = 255):
        """map from floating point stimulus state, to integer for sending to decoder

        Args:
            state (list-of-float): the floating point stimulus state
            mins (float, optional): min float state value. Defaults to 0.
            maxs (float, optional): max float state value. Defaults to 1.
            mini (int, optional): min integer state. Defaults to 0.
            maxi (int, optional): max integer state. Defaults to 255.
        """
        state = [int(mini + (s-mins)/maxs * maxi) for s in state]
        return state

    def sendStimulusState(self, timestamp=None, targetState=None, injectSignal=None):
        """send the current stimulus state information to the decoder

        Args:
            timestamp (int, optional): timestamp to use when sending the stimulus state information. Defaults to None.
            targetState (int, optional): state of the 'true' target.  If None, then computed from the state of the target idx.  Defaults to None.
            injectSignal (int|float, optional): amplitude of a signal to inject into the fake data stream for debugging.  If None then use the target stimulus state.  Defaults to None.
        """
        stimState, target_idx, objIDs, sendEvent = self.laststate
        # send info about the stimulus displayed
        if sendEvent and stimState is not None:
            if isinstance(stimState[0], float):
                stimState = self.encodeFloatState(stimState)
            if targetState is None:
                targetState = stimState[target_idx] if target_idx is not None and target_idx >= 0 else -1
            # print((stimState,targetState))
            # TODO[]: change to use the target_idx
            self.sendStimulusEvent(stimState,
                                   timestamp,
                                   targetState,
                                   objIDs,
                                   injectSignal)

    def sendStimulusEvent(self, stimState, timestamp=None, targetState=None, objIDs=None, injectSignal=None):
        """send the current stimulus state information to the decoder

        Args:
            stimulusState (list-of-int): the stimulus state of each object in objIDs
            timestamp (int, optional): timestamp for this stimulus change in milliseconds. Defaults to None.
            targetState ([type], optional): state of the current cued target. Defaults to None.
            objIDs (list-of-int, optional): the object Identifiers for the objects in stimulus state. Defaults to None.
            injectSignal (float, optional): the level of the injected signal to send (fakedata only). Defaults to None.
        """
        self.utopiaController.sendStimulusEvent(stimState,
                                                timestamp,
                                                targetState,
                                                objIDs,
                                                injectSignal)

    def getNewMessages(self):
        """get all new messages from the decoder

        Returns:
            list-of-UtopiaMessage: a list of all new UtopiaMessages recieved from the hub/decoder
        """
        if self.utopiaController:
            return self.utopiaController.msgs
        return []

    def getLastPrediction(self):
        """get the last prediction recieved from the hub/decoder

        Returns:
            PredictedTargetProb: the last recieved PredictedTargetProb message
        """

        if self.utopiaController:
            return self.utopiaController.getLastPrediction()
        return None

    def clearLastPrediction(self):
        """clear the information about the last recieved target prediction
        """
        if self.utopiaController:
            self.utopiaController.clearLastPrediction()

    def getLastPredictionDistribution(self):
        """get the last predicted target distribution recieved from the hub/decoder

        Returns:
            PredictedTargetDist: the last recieved PredictedTargetProb message
        """

        if self.utopiaController:
            return self.utopiaController.getLastPredictionDistribution()
        return None

    def clearLastPredictionDistribution(self):
        """clear the information about the last recieved target prediction
        """
        if self.utopiaController:
            self.utopiaController.clearLastPredictionDistribution()

    def getLastSignalQuality(self):
        """return the last signal quality message recieved from the hub/decoder

        Returns:
            ElectrodeQuality: the last ElectrodeQuality message from the hub/decoder
        """
        if self.utopiaController:
            return self.utopiaController.getLastSignalQuality()
        return None

    def clearLastSignalQuality(self):
        """clear the last signal quality message
        """
        if self.utopiaController:
            self.utopiaController.clearLastSignalQuality()

    def getLastSelection(self):
        """return the last selection message recieved from the hub/decoder

        Returns:
            Selection: the last Selection message recieved from the hub/decoder
        """
        if self.utopiaController:
            return self.utopiaController.getLastSelection()
        return None

    def clearLastSelection(self):
        """clear the last selelction message from the hub/decoder
        """
        if self.utopiaController:
            self.utopiaController.clearLastSelection()

    def addMessageHandler(self, cb):
        """add a handler which is called back when a new message is recieved

        Args:
            cb (function): the function to be called for each newly recieved message
        """
        if self.utopiaController:
            self.utopiaController.addMessageHandler(cb)

    def addPredictionHandler(self, cb):
        """add a handler which is called back when a Prediction is recieved from the decoder/hub

        Args:
            cb (function): the function to be called for each newly recieved Prediction
        """
        if self.utopiaController:
            self.utopiaController.addPredictionHandler(cb)

    def addPredictionDistributionHandler(self, cb):
        """add a handler which is called back when a Prediction is recieved from the decoder/hub

        Args:
            cb (function): the function to be called for each newly recieved Prediction
        """
        if self.utopiaController:
            self.utopiaController.addPredictionDistributionHandler(cb)

    def addSelectionHandler(self, cb):
        """add a handler which is called back when a Selection is recieved from the decoder/hub

        Args:
            cb (function): the function to be called for every newly recieved Selection
        """
        if self.utopiaController:
            self.utopiaController.addSelectionHandler(cb)

    def addNewTargetHandler(self, cb):
        """add a handler which is called back when a newtarget is recieved from the decoder/hub

        Args:
            cb (function): the function to be called for every newly recieved NewTarget
        """
        if self.utopiaController:
            self.utopiaController.addNewTargetHandler(cb)

    def setTimeStampClock(self, tsclock):
        """set the clock used by default to timestamp messages sent to the hub/decoder

        Args:
            tsclock (TimeStampClock): the time-stamp clock object to use
        """
        self.utopiaController.setTimeStampClock(tsclock)

    def getTimeStamp(self):
        """get the current time stamp

        Returns:
            int: the timestamp for the curren time in milliseconds
        """
        return self.utopiaController.getTimeStamp() if self.utopiaController is not None else -1

    def log(self, msg):
        """send a Log message to the decoder/hub

        Args:
            msg (str): the Log message to send
        """
        if self.utopiaController:
            self.utopiaController.log(msg)

    def modeChange(self, newmode):
        """manually change the decoder mode

        Args:
            newmode (str): the new mode string to send to the hub/decoder
        """
        if self.utopiaController:
            self.utopiaController.modeChange(newmode)

    def subscribe(self, msgs):
        """tell the hub we will subscribe to this set of message IDs

        Args:
            msgs (str): a list of messageIDs to subscribe to.  See mindaffectBCI.utopiaclient for the list of message types and IDs
        """
        if self.utopiaController:
            self.utopiaController.subscribe(msgs)

    def addSubscription(self, msgs):
        """add a set of messageIDs to our current set of subscribed message types.

        Args:
            msgs (str): a list of messageIDs to subscribe to.  See mindaffectBCI.utopiaclient for the list of message types and IDs
        """
        if self.utopiaController:
            self.utopiaController.addSubscription(msgs)

    def removeSubscription(self, msgs):
        """remove a set of messageIDs to our current set of subscribed message types.

        Args:
            msgs (str): a list of messageIDs to unsubscribe from.  See mindaffectBCI.utopiaclient for the list of message types and IDs
        """
        if self.utopiaController:
            self.utopiaController.removeSubscription(msgs)

    # methods to define what (meta) stimulus sequence we will play
    def startExpt(self, nCal=1, nPred=20, selectionThreshold=.1,
                  cuedprediction=False,  objIDs=None,
                  **kwargs):
        """Start the sequence for a full Calibration->Prediction experiment.

        Args:
            nCal (int, optional): the number of calibration trials. Defaults to 10.
            nPred (int, optional): the number of prediction trials. Defaults to 20.
            selectionThreshold (float, optional): the Perr threshold for selection. Defaults to .1
            cuedprediction (bool, optional): flag if we do cueing before trial starts.  Default to False.
            numframes (int, optional): the number of frames to flicker for. Defaults to None.
            framesperbit (int, optional): the number of video frames per stimSeq time-points. Defaults to 1.
            selectionThreshold (float, optional): Target error probability for selection. Defaults to None.
            duration (int, optional): flicker duration in seconds. Defaults to 4.
            cueduration (int, optional): cue duration in seconds. Defaults to 1.
            feedbackduration (int, optional): feedback duration in seconds. Defaults to 1.
            waitduration (int, optional): wait duration in seconds. Defaults to 1.
            cueframes (int, optional): cue duration in frames. Defaults to None.
            feedbackframes (int, optional): feedback duration in frames. Defaults to None.
            waitframes (int, optional): wait duration in frames. Defaults to None.
            permute (bool, optional): if true then permute the order of objects before running the flicker sequence.  Default to False
            startframe (int, optional): set the frame in the sequence to start the flicker from.  if 'random' then pick a random starting point. If 'lastframe' then continue from the last played frame.  Defaults to 0.
            sendEvents (bool, optoinal): if true then send event info to the Hub, if false then do not send events.  For example used for for practice / debugging. Defaults to True.
        """

        if self.stimulusStateMachineStack.stack:
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()

        if objIDs is not None:
            self.setActiveObjIDs(objIDs) if hasattr(objIDs,'__iter__') else self.setnumActiveObjIDs(objIDs)

        self.stimulusStateMachineStack.push(
            Experiment(objIDs=self.objIDs, stimSeq=self.noisecode,
                       nCal=nCal, nPred=nPred,
                       selectionThreshold=selectionThreshold, 
                       cuedprediction=cuedprediction,
                       utopiaController=self.utopiaController,
                       stimulusStateMachineStack=self.stimulusStateMachineStack,
                       **kwargs))

    def startCalibration(self, nTrials=10, stimSeq=None, objIDs=None,
                         **kwargs):
        """setup and run a complete calibration phase

        Args:
            nTrials (int, optional): number of calibration trials to run. Defaults to 10.
            stimSeq (list-of-lists-of-int, optional): the stimulus sequence to play for the flicker in the calibration phase. Defaults to None.
            numframes (int, optional): the number of frames to flicker for. Defaults to None.
            framesperbit (int, optional): the number of video frames per stimSeq time-points. Defaults to 1.
            selectionThreshold (float, optional): Target error probability for selection. Defaults to None.
            duration (int, optional): flicker duration in seconds. Defaults to 4.
            cueduration (int, optional): cue duration in seconds. Defaults to 1.
            feedbackduration (int, optional): feedback duration in seconds. Defaults to 1.
            waitduration (int, optional): wait duration in seconds. Defaults to 1.
            cueframes (int, optional): cue duration in frames. Defaults to None.
            feedbackframes (int, optional): feedback duration in frames. Defaults to None.
            waitframes (int, optional): wait duration in frames. Defaults to None.
            permute (bool, optional): if true then permute the order of objects before running the flicker sequence.  Default to False
            startframe (int, optional): set the frame in the sequence to start the flicker from.  if 'random' then pick a random starting point. If 'lastframe' then continue from the last played frame.  Defaults to 0.
            sendEvents (bool, optoinal): if true then send event info to the Hub, if false then do not send events.  For example used for for practice / debugging. Defaults to True.
        """
        if self.stimulusStateMachineStack.stack:
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()

        if stimSeq is not None:
            self.set_stimSeq(stimSeq)

        if objIDs is not None:
            self.setActiveObjIDs(objIDs) if hasattr(objIDs,'__iter__') else self.setnumActiveObjIDs(objIDs)

        self.stimulusStateMachineStack.push(
            CalibrationPhase(objIDs=self.objIDs,
                             stimSeq=self.noisecode,
                             nTrials=nTrials,
                             utopiaController=self.utopiaController,
                             stimulusStateMachineStack=self.stimulusStateMachineStack,
                             **kwargs))

    def startPrediction(self, nTrials:int=10, stimSeq=None, objIDs:list=None,
                        **kwargs):
        """setup and run a complete prediction phase

        Args:
            nTrials (int, optional): number of prediction trials to run. Defaults to 10.
            stimSeq (list-of-lists-of-int, optional): the stimulus sequence to play for the flicker in the calibration phase. Defaults to None.
            objIDs (list|int, optional): the set of objectIDs to generate stimulus states for in this sequence.  If None use the Noisetag objects list of objIDs.  Defaults to None.
            numframes (int, optional): the number of frames to flicker for. Defaults to None.
            framesperbit (int, optional): the number of video frames per stimSeq time-points. Defaults to 1.
            selectionThreshold (float, optional): Target error probability for selection. Defaults to None.
            duration (int, optional): flicker duration in seconds. Defaults to 4.
            cueduration (int, optional): cue duration in seconds. Defaults to 1.
            feedbackduration (int, optional): feedback duration in seconds. Defaults to 1.
            waitduration (int, optional): wait duration in seconds. Defaults to 1.
            cueframes (int, optional): cue duration in frames. Defaults to None.
            feedbackframes (int, optional): feedback duration in frames. Defaults to None.
            waitframes (int, optional): wait duration in frames. Defaults to None.
            permute (bool, optional): if true then permute the order of objects before running the flicker sequence.  Default to False
            startframe (int, optional): set the frame in the sequence to start the flicker from.  if 'random' then pick a random starting point. If 'lastframe' then continue from the last played frame.  Defaults to 0.
            sendEvents (bool, optoinal): if true then send event info to the Hub, if false then do not send events.  For example used for for practice / debugging. Defaults to True.
        """
        if self.stimulusStateMachineStack.stack:
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()

        if stimSeq is not None:
            self.set_stimSeq(stimSeq)

        if objIDs is not None:
            self.setActiveObjIDs(objIDs) if hasattr(objIDs,'__iter__') else self.setnumActiveObjIDs(objIDs)

        self.stimulusStateMachineStack.push(
            PredictionPhase(objIDs=self.objIDs,
                            stimSeq=self.noisecode,
                            nTrials=nTrials,
                            utopiaController=self.utopiaController,
                            stimulusStateMachineStack=self.stimulusStateMachineStack,
                            **kwargs))

    def startSingleTrial(self, numframes: int = 60*4, tgtidx: int = -1, stimSeq=None, objIDs:list=None, **kwargs):
        """setup and run a single flicker trial, with (optional)cue->wait->flicker->feedback

        Args:
            numframes (int, optional): the number of frames for the flicker. Defaults to 100.
            tgtidx (int, optional): the index in the stimSequence of the target, -1 if no cued target.  Default to -1
            numframes (int, optional): the number of frames to flicker for. Defaults to None.
            framesperbit (int, optional): the number of video frames per stimSeq time-points. Defaults to 1.
            selectionThreshold (float, optional): Target error probability for selection. Defaults to None.
            duration (int, optional): flicker duration in seconds. Defaults to 4.
            cueduration (int, optional): cue duration in seconds. Defaults to 1.
            feedbackduration (int, optional): feedback duration in seconds. Defaults to 1.
            waitduration (int, optional): wait duration in seconds. Defaults to 1.
            cueframes (int, optional): cue duration in frames. Defaults to None.
            feedbackframes (int, optional): feedback duration in frames. Defaults to None.
            waitframes (int, optional): wait duration in frames. Defaults to None.
            permute (bool, optional): if true then permute the order of objects before running the flicker sequence.  Default to False
            startframe (int, optional): set the frame in the sequence to start the flicker from.  if 'random' then pick a random starting point. If 'lastframe' then continue from the last played frame.  Defaults to 0.
            sendEvents (bool, optoinal): if true then send event info to the Hub, if false then do not send events.  For example used for for practice / debugging. Defaults to True.
        """
        if self.stimulusStateMachineStack.stack:
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()

        if stimSeq is not None:
            self.set_stimSeq(stimSeq)

        if objIDs is not None:
            self.setActiveObjIDs(objIDs) if hasattr(objIDs,'__iter__') else self.setnumActiveObjIDs(objIDs)

        self.stimulusStateMachineStack.push(
            SingleTrial(stimSeq=self.noisecode,
                        tgtidx=tgtidx,
                        utopiaController=self.utopiaController,
                        stimulusStateMachineStack=self.stimulusStateMachineStack,
                        numframes=numframes,
                        **kwargs))

    def startFlicker(self, numframes=60*4, tgtidx=-1, stimSeq=None, objIDs=None, *args, **kwargs):
        """setup and run the just the flicker

        Args:
            numframes (int, optional): the number of frames for the flicker. Defaults to 100.
            tgtidx (int, optional): the index in the stimSequence of the target, -1 if no cued target.  Default to -1
            numframes (int, optional): the number of frames to flicker for. Defaults to None.
            framesperbit (int, optional): the number of video frames per stimSeq time-points. Defaults to 1.
            selectionThreshold (float, optional): Target error probability for selection. Defaults to None.
            duration (int, optional): flicker duration in seconds. Defaults to 4.
            cueduration (int, optional): cue duration in seconds. Defaults to 1.
            feedbackduration (int, optional): feedback duration in seconds. Defaults to 1.
            waitduration (int, optional): wait duration in seconds. Defaults to 1.
            cueframes (int, optional): cue duration in frames. Defaults to None.
            feedbackframes (int, optional): feedback duration in frames. Defaults to None.
            waitframes (int, optional): wait duration in frames. Defaults to None.
            permute (bool, optional): if true then permute the order of objects before running the flicker sequence.  Default to False
            startframe (int, optional): set the frame in the sequence to start the flicker from.  if 'random' then pick a random starting point. If 'lastframe' then continue from the last played frame.  Defaults to 0.
            sendEvents (bool, optoinal): if true then send event info to the Hub, if false then do not send events.  For example used for for practice / debugging. Defaults to True.
        """
        if self.stimulusStateMachineStack.stack:
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()

        if stimSeq is not None:
            self.set_stimSeq(stimSeq)

        if objIDs is not None:
            self.setActiveObjIDs(objIDs) if hasattr(objIDs,'__iter__') else self.setnumActiveObjIDs(objIDs)


        self.stimulusStateMachineStack.push(
            Flicker(self.noisecode,
                    numframes,
                    tgtidx,
                    *args, **kwargs))

    def startFlickerWithSelection(self, numframes: int = 60*4,
                                  tgtidx: int = -1, stimSeq=None, objIDs=None, *args, **kwargs):
        """setup and run the just the flicker, with early stopping if the BCI selects an output

        Args:
            numframes (int, optional): the number of frames for the flicker. Defaults to 100.
            tgtidx (int, optional): the index in the stimSequence of the target, -1 if no cued target.  Default to -1
            numframes (int, optional): the number of frames to flicker for. Defaults to None.
            framesperbit (int, optional): the number of video frames per stimSeq time-points. Defaults to 1.
            selectionThreshold (float, optional): Target error probability for selection. Defaults to None.
            duration (int, optional): flicker duration in seconds. Defaults to 4.
            cueduration (int, optional): cue duration in seconds. Defaults to 1.
            feedbackduration (int, optional): feedback duration in seconds. Defaults to 1.
            waitduration (int, optional): wait duration in seconds. Defaults to 1.
            cueframes (int, optional): cue duration in frames. Defaults to None.
            feedbackframes (int, optional): feedback duration in frames. Defaults to None.
            waitframes (int, optional): wait duration in frames. Defaults to None.
            permute (bool, optional): if true then permute the order of objects before running the flicker sequence.  Default to False
            startframe (int, optional): set the frame in the sequence to start the flicker from.  if 'random' then pick a random starting point. If 'lastframe' then continue from the last played frame.  Defaults to 0.
            sendEvents (bool, optoinal): if true then send event info to the Hub, if false then do not send events.  For example used for for practice / debugging. Defaults to True.
        """
        if self.stimulusStateMachineStack.stack:
            print("Warning: replacing running sequence?")
            self.stimulusStateMachineStack.clear()

        if stimSeq is not None:
            self.set_stimSeq(stimSeq)

        if objIDs is not None:
            self.setActiveObjIDs(objIDs) if hasattr(objIDs,'__iter__') else self.setnumActiveObjIDs(objIDs)

        self.stimulusStateMachineStack.push(
            FlickerWithSelection(self.noisecode,
                                 numframes,
                                 tgtidx,
                                 self.utopiaController,
                                 *args, **kwargs))


class sumstats:
    '''Utility class to record summary stastics for, e.g. frame flip timing'''

    def __init__(self, bufsize: int = 60*2):
        self.buf = [0]*(bufsize)  # ring-buffer, 700 entries
        self.N = 0
        self.sx = 0
        self.mu = None
        self.median = None
        self.sx2 = 0
        self.sigma2 = None
        self.minx = 0
        self.maxx = 0

    def addpoint(self, x):
        """add a point to the running buffer of statistics

        Args:
            x (float): the point to add
        """
        self.buf[self.N % len(self.buf)] = x  # ring-buffer
        self.N = self.N+1
        self.sx = self.sx+x
        self.sx2 = self.sx2+x*x
        self.minx = x if x < self.minx else self.minx
        self.maxx = x if x > self.maxx else self.maxx

    def hist(self):
        """get the histogram of statistics

        Returns:
            str: string summary of the data histogram
        """
        try:
            import numpy
            buf = self.buf[:min(len(self.buf), self.N)]
            self.minx = min(buf)
            self.maxx = max(buf)
            bstart = self.minx + (self.maxx-self.minx)*0
            bw = (self.maxx-self.minx)*(.6-0)
            bins = [bstart+bw*(x/15.0) for x in range(0, 15)]
            [hist, bins] = numpy.histogram(buf, bins)
            pp = " ".join("%5.2f" % ((bins[i]+bins[i+1])/2) for i in range(len(bins)-1))
            pp += "\n" + " ".join("%5d" % t for t in hist)
        except:
            pp = ''
        return pp

    def update_statistics(self):
        """compute updated summary statistics, including: mu=average, med=median, sigma=std-dev, min=min, max=max
        """
        import statistics
        buf = self.buf[:min(len(self.buf), self.N)]
        self.mu = statistics.mean(buf) if len(buf) > 0 else -1
        self.median = statistics.median(buf) if len(buf) > 0 else -1
        self.sigma = statistics.stdev(buf) if len(buf) > 2 else -1
        self.min = min(buf) if len(buf) > 0 else -1
        self.max = max(buf) if len(buf) > 0 else -1

    def __str__(self):
        """string representation of the summary statistics

        Returns:
            str: string representation of the summary statistics
        """
        self.update_statistics()
        return "%f,%f (%f,[%f,%f])" % (self.mu, self.median, self.sigma, self.min, self.max)


def doFrame(t, stimState, tgt_idx=-1, objIDs=None, utopiaController=None):
    """utility function to print a summary of a single frame of a stimulus sequence

    Args:
        t (int): current time
        stimState (list-of-int): the current stimulus state for each output.
        tgt_idx (int, optional): the index of the cued target (if set), -1 if no target. Defaults to -1.
        objIDs (list-of-int, optional): the used object ids for each output. Defaults to None.
        utopiaController (UtopiaController, optional): the controller for communcation to the decoder. Defaults to None.
    """
    if tgt_idx >= 0:
        tgtState = stimState[tgt_idx]
        print("*" if tgtState > 0 else ".", end='', flush=True)
    else:
        print('_', end='', flush=True)


def run(symbols=None, ncal: int = 10, npred: int = 10, cuedprediction: bool = True, frameperbit: int = 1,
        calibration_trialduration: float = 4.2,  prediction_trialduration: float = 20, feedbackduration: float = 2,
        stimseq: str = None, selectionThreshold: float = .1, **kwargs):
    """run a noisetagging *fake-presentation* example

    Args:
        symbols ([type], optional): [description]. Defaults to None.
        ncal (int, optional): [description]. Defaults to 10.
        npred (int, optional): [description]. Defaults to 10.
        cuedprediction (bool, optional): [description]. Defaults to True.
        frameperbit (int, optional): [description]. Defaults to 1.
        calibration_trialduration (float, optional): [description]. Defaults to 4.2.
        prediction_trialduration (float, optional): [description]. Defaults to 20.
        feedbackduration (float, optional): [description]. Defaults to 2.
        stimseq (str, optional): [description]. Defaults to None.
        selectionThreshold (float, optional): [description]. Defaults to .1.
    """
    # make the noisetag object to manage the tagging selections
    nt = Noisetag(stimSeq=stimseq, clientid='fakepresentation')
    nt.connect()
    nsymb = sum([len(r) for r in symbols]) if symbols is not None else 10
    # set the subset of active objects being displayed
    # nt.setnumActiveObjIDs(nsymb)
    # tell it to play a full experiment sequence
    nt.startExpt(nCal=ncal, nPred=npred, cuedprediction=cuedprediction, framesperbit=frameperbit,
                 selectionThreshold=selectionThreshold, startframe='lastframe')
    # just do the prediction phase
    # nt.startPrediction(nTrials=npred, framesperbit=frameperbit,
    #              selectionThreshold=selectionThreshold, startframe='lastframe')
    # mainloop
    nframe = 0
    while True:
        try:
            # update the stimulus state w.r.t. current time
            nt.updateStimulusState(nframe)
            # get the stimulus state info we whould display
            ss, ts, objIDs, sendEvent = nt.getStimulusState()
            # update the display drawing
            doFrame(nframe, ss, ts, objIDs)
            # send info about what we did to the decoder
            nt.sendStimulusState()
            # simulate waiting for the flip
            time.sleep(isi)
        except StopIteration:
            # this event is raised when the stimulus sequence is finished
            break
        nframe = nframe+1


if __name__ == "__main__":
    run(ncal=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
