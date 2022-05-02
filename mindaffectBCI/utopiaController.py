
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

from mindaffectBCI.utopiaclient import *

class UtopiaController:
    """controller class to manage the interaction with the Mindaffect decoder,
    setting up the connection, sending and recieving messages, and firing message
    event handlers

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """    
    
    def __init__(self, clientid=None):
        self.client = UtopiaClient(clientid)
        self.msgs = []
        # callback list for new messages
        self.messageHandlers = []
        # call back list for new predictions
        self.lastPrediction = None
        self.predictionHandlers = []
        # call back list for new prediction distributions
        self.lastPredictionDistribution = None
        self.predictionDistributionHandlers = []
        # selection stuff
        self.selectionHandlers = []
        self.selectionThreshold = .1
        # signal quality stuff
        self.lastSignalQuality = None
        self.signalQualityHandlers = []
        # new target
        self.newTargetHandlers = []
        # our set message subscriptions
        # P=Target Prediction F=Target Dist S=selection N=new-target M=mode-change E=stimulus-event Q=signal-quality
        self.subscriptions = "PFSNMEQ"

    def addMessageHandler(self, cb):
        """[summary]

        Args:
            cb (function): [description]
        """        
        self.messageHandlers.append(cb)
    def addPredictionHandler(self, cb):
        """[summary]

        Args:
            cb (function): [description]
        """        
        self.predictionHandlers.append(cb)
    def addPredictionDistributionHandler(self, cb):
        """[summary]

        Args:
            cb (function): [description]
        """        
        self.predictionDistributionHandlers.append(cb)
    def addNewTargetHandler(self, cb):
        """[summary]

        Args:
            cb (function): [description]
        """        
        self.newTargetHandlers.append(cb)
    def addSelectionHandler(self, cb):
        """[summary]

        Args:
            cb (function): [description]
        """        
        self.selectionHandlers.append(cb)
    def addSignalQualityHandler(self, cb):
        """[summary]

        Args:
            cb (function): [description]
        """        
        self.signalQualityHandlers.append(cb)
    
    def setTimeStampClock(self,tsclock):
        """[summary]

        Args:
            tsclock ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return self.client.setTimeStampClock(tsclock)
    def getTimeStamp(self):
        '''get a (relative) wall-time stamp *in milliseconds*'''
        return self.client.getTimeStamp()
        
    def autoconnect(self, host=None, port=8400, timeout_ms=5000, 
                    localhostifhostnotfound=True, queryifhostnotfound=True, scanifhostnotfound=False):
        """[summary]

        Args:
            host ([type], optional): [description]. Defaults to None.
            port (int, optional): [description]. Defaults to 8400.
            timeout_ms (int, optional): [description]. Defaults to 5000.
            localhostifhostnotfound (bool, optional): [description]. Defaults to True.
            queryifhostnotfound (bool, optional): [description]. Defaults to True.
            scanifhostnotfound (bool, optional): [description]. Defaults to False.
        """                    
        try:
            self.client.autoconnect(host, port, timeout_ms=timeout_ms, localhostifhostnotfound=localhostifhostnotfound, queryifhostnotfound=queryifhostnotfound, scanifhostnotfound=scanifhostnotfound)
        except socket.error as ex:
            print("Socket error connecting: ")
            print(ex)
            pass

        if not self.client.isConnected:
            print("Warning:: couldnt connect to a utopia hub....")
        else:
            # set default subscriptions
            self.subscribe()        

    def isConnected(self):  return self.client.isConnected
    def gethostport(self):  return self.client.gethostport()
    def gethost(self): return self.client.gethost()

    def sendStimulusEvent(self, stimulusState, timestamp=None,
                          targetState=None, objIDs=None, injectSignal:float=None):
        """
        Send a message to the Utopia-HUB informing of the current stimulus state

        Args:
            stimulusState ([type]): [description]
            timestamp ([type], optional): [description]. Defaults to None.
            targetState ([type], optional): [description]. Defaults to None.
            objIDs ([type], optional): [description]. Defaults to None.
            injectSignal (int|callable,optional): inject a signal with this amplitude to fake data, should be 0-255 integer. 
                     If None then use the target state information. If callable, then injectSignal(targetState|stimulusState) to return the state to inject. Defautls to None.
        Returns:
            [type]: [description]
        """
        stimEvent = self.mkStimulusEvent(stimulusState, timestamp, targetState, objIDs)
        if self.isConnected(): self.client.sendMessage(stimEvent)
        # erp injection for debugging with fakedata
        if injectSignal is None:
            if targetState is not None and targetState >= 0:
                if 0 <= targetState and targetState <= 1:
                    # re-scale to 0-255
                    injectSignal = int(targetState*255)
                elif targetState > 3:
                    injectSignal = int(targetState)
            else: # simple super-position of stimulus activity
                injectSignal = int(sum(stimulusState))
        elif callable(injectSignal): # call function to get the signal to inject
            injectSignal = injectSignal(targetState if targetState>=0 else stimulusState)
        if injectSignal is not None:
            injectERP(int(injectSignal), self.gethost())
        return stimEvent


    def mkStimulusEvent(self, stimulusState, timestamp=None, 
                        targetState=None, objIDs=None):
        """
        make a valid stimulus event for the given stimulus state

        Args:
            stimulusState (list-of-int): the stimulus state of each object in objIDs
            timestamp (int, optional): timestamp for this stimulus change in milliseconds. Defaults to None.
            targetState ([type], optional): state of the current cued target. Defaults to None.
            objIDs (list-of-int, optional): the object Identifiers for the objects in stimulus state. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """                        
        if not hasattr(stimulusState,'__iter__'): stimulusState=[stimulusState]
        if timestamp is None:
            timestamp = self.getTimeStamp()
        if objIDs is None:
            objIDs = list(range(1, len(stimulusState)+1))
        if not hasattr(objIDs,'__iter__'): objIDs=[objIDs]
        if len(objIDs) != len(stimulusState):
            raise ValueError("ARGH! objIDs and stimulusState not same length!")
    
        # insert extra 0 object ID if targetState given
        if not targetState is None and targetState >= 0:
            # N.B. copy longer version to new variable, rather than modify in-place with append
            objIDs = objIDs+[0] 
            stimulusState = stimulusState+[targetState]
        
        return StimulusEvent(timestamp, objIDs, stimulusState)

    def modeChange(self, newmode):
        """[summary]

        Args:
            newmode ([type]): [description]
        """        
        if self.isConnected():
            self.client.sendMessage(
                ModeChange(self.getTimeStamp(), newmode))

    def subscribe(self, msgs=None):
        """[summary]

        Args:
            msgs ([type], optional): [description]. Defaults to None.
        """        
        # subscribe to PREDICTEDTARGETPROB, MODECHANGE, SELECTION and NEWTARGET, SIGNALQUALITY messages only
        if msgs:
            self.subscriptions = msgs
        if self.isConnected():
            print("NewSubscriptions: {}".format(self.subscriptions))
            self.client.sendMessage(
                Subscribe(self.getTimeStamp(), self.subscriptions))

    def addSubscription(self, msgs):
        """[summary]

        Args:
            msgs ([type]): [description]
        """        
        # N.B. we allow multiple subscribe to same message type so can remove without worrying about breaking
        # for another user
        for m in msgs:
            self.subscriptions += m
        self.subscribe()
    def removeSubscription(self, msgs):
        """[summary]

        Args:
            msgs ([type]): [description]
        """        
        # remove msgs from the list of subscribed messages
        for m in msgs:
            self.subscriptions = self.subscriptions.replace(m, "", 1) # remove single subscription
        self.subscribe()


    def log(self, msg):
        """[summary]

        Args:
            msg ([type]): [description]
        """        
        if self.isConnected():
            self.client.sendMessage(Log(self.getTimeStamp(), msg))

    def newTarget(self):
        """[summary]
        """        
        if self.isConnected():
            self.client.sendMessage(NewTarget(self.getTimeStamp()))
        for h in self.newTargetHandlers:
            h()         # do selection callbacks

    def selection(self, objID):
        """[summary]

        Args:
            objID ([type]): [description]
        """        
        if self.isConnected():
            self.client.sendMessage(Selection(self.getTimeStamp(), objID))
        for h in self.selectionHandlers:
            h(objID)         # do selection callbacks
            
    def getNewMessages(self, timeout_ms=0):
        """
        get new messages from the utopia-hub, and store the list of new

        Args:
            timeout_ms (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """        
        
        if not self.isConnected(): return None
        # get any messages with predictions
        self.msgs = self.client.getNewMessages(timeout_ms) if self.client else []
        # process these messages as needed & call-callbacks
        if len(self.msgs) > 0:
            for h in self.messageHandlers:
                h(self.msgs)
            for m in self.msgs:
   
                if m.msgID == PredictedTargetProb.msgID:
                    print("UtopiaController::getNewMessages::Pred:{}".format(m))
                    # record as last prediction
                    self.lastPrediction = m
                    # process new prediction callbacks
                    for h in self.predictionHandlers:
                        h(self.lastPrediction)

                elif m.msgID == PredictedTargetDist.msgID:
                    print("UtopiaController::getNewMessages::PredDist:{}".format(len(m.pTgt)))
                    # record as last prediction
                    self.lastPredictionDistribution = m
                    # process new prediction callbacks
                    for h in self.predictionDistributionHandlers:
                        h(self.lastPredictionDistribution)

                elif m.msgID == Selection.msgID:
                    # process selection callbacks
                    for h in self.selectionHandlers:
                        h(m.objID)

                elif m.msgID == NewTarget.msgID:
                    # process selection callbacks
                    for h in self.newTargetHandlers:
                        h()

                elif m.msgID == SignalQuality.msgID:
                    self.lastSignalQuality = m.signalQuality
                    # process selection callbacks
                    for h in self.signalQualityHandlers:
                        h(self.lastSignalQuality)

        return self.msgs

    def getLastPrediction(self):
        """
        check for new predictions from the utopia-decoder

        Returns:
            [type]: [description]
        """        
        
        # Q: should we do this here? or just return the lastPrediction?
        self.getNewMessages()
        # always return the last prediction, even if no new ones
        return self.lastPrediction
    
    def clearLastPrediction(self):
        '''clear the last predicted target'''
        self.lastPrediction = None
            
    def getLastSelection(self):
        """      
        check if any object prediction is high enough for it to be selected

        Returns:
            [type]: [description]
        """        

        self.getNewMessages()
        if self.lastPrediction is not None:
            if self.lastPrediction.Perr < self.selectionThreshold: # good enough to select?
                return (self.lastPrediction.Yest, True)

            else: # return predictedObjID but not-selected
                return (self.lastPrediction.Yest, False)
        return (None, False)

    def getLastSignalQuality(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
        self.getNewMessages()
        return self.lastSignalQuality
    
    def clearLastSignalQuality(self):
        self.lastSignalQuality = None

def injectERP(amp, host="localhost", port=8300):
    """
    Inject an erp into a simulated data-stream, sliently ignore if failed, e.g. because not simulated

    Args:
        amp (int|float): amplitude of the injected signal, in the range 0-256
        host (str, optional): [description]. Defaults to "localhost".
        port (int, optional): [description]. Defaults to 8300.
    """    
    
    import socket
    try:
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0).sendto(bytes([min(amp,255)]), (host, port))
    except ValueError: 
        print("Error: ERP amplitude should be between 0 and 256")
    except: # sliently igore any errors
        pass


def newMessageHandler(msgs):
    """[summary]

    Args:
        msgs ([type]): [description]
    """    
    for m in msgs:
        print(m)

if __name__ == "__main__":
    # simple message logging testcase
    uc = UtopiaController()
    uc.autoconnect()
    # add logging incomming message handler
    uc.addMessageHandler(newMessageHandler)
    while uc.isConnected():
        uc.getNewMessages(1000)
        #msgs = uc.getNewMessages(1000)
        # print the recieved messages
        #for m in msgs:
        #    print(m)
