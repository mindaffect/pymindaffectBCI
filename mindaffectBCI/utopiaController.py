
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

from .utopiaclient import *

class UtopiaController:
    '''controller class to manage the interaction with the Mindaffect decoder,
    setting up the connection, sending and recieving messages, and firing message
    event handlers'''
    def __init__(self):
        self.client=UtopiaClient()
        self.msgs=[]
        # callback list for new messages
        self.messageHandlers=[]
        # call back list for new predictions
        self.lastPrediction=None
        self.predictionHandlers=[]
        # selection stuff
        self.selectionHandlers=[]
        self.selectionThreshold=.1
        # signal quality stuff
        self.lastSignalQuality=None
        self.signalQualityHandlers=[]

    def addMessageHandler(self,cb):
        self.messageHandlers.append(cb)
    def addPredictionHandler(self,cb):
        self.predictionHandlers.append(cb)
    def addSelectionHandler(self,cb):
        self.selectionHandlers.append(cb)
    def addSignalQualityHandler(self,cb):
        self.signalQualityHandlers.append(cb)
        
    def getTimeStamp(self,t0=0):
        '''get a (relative) wall-time stamp *in milliseconds*'''
        return getTimeStamp(t0)    
        
    def autoconnect(self,host=None,port=8400,timeout_ms=5000,queryifhostnotfound=True): 
        try : 
            self.client.autoconnect(host,port,timeout_ms=timeout_ms)
        except socket.error as ex:
            pass

        # ask user for host
        if not self.client.isConnected and queryifhostnotfound :
            print("Could not auto-connect.  Trying manual")
            hostport=input("Enter the hostname/IP of the Utopia-HUB: ")
            try:
                self.client.autoconnect(hostport)
            except :
                print("Could not connect to %s. Run in disconnected!"%(hostport))

        if not self.client.isConnected :
            print("Warning:: couldnt connect to a utopia hub....")
        else :
            # subscribe to PREDICTEDTARGETPROB, MODECHANGE, SELECTION and NEWTARGET messages only
            self.subscribe()        

    def isConnected(self):  return self.client.isConnected
    def gethostport(self):  return self.client.gethostport()
    def subscribe(self):
        self.client.sendMessage(Subscribe(self.getTimeStamp(),"PMSN"))

                
    def sendStimulusEvent(self,stimulusState,timestamp=None,
                          targetState=None,objIDs=None):
        """Send a message to the Utopia-HUB informing of the current stimulus state"""
        stimEvent=self.mkStimulusEvent(stimulusState,timestamp,targetState,objIDs)
        if self.client : self.client.sendMessage(stimEvent)
        # erp injection for debugging with fakedata
        if targetState in (0,1) : injectERP(targetState)
        return stimEvent
        
    def mkStimulusEvent(self,stimulusState,timestamp=None,
                        targetState=None,objIDs=None):
        """make a valid stimulus event for the given stimulus state"""
        if timestamp is None:
            timestamp=self.getTimeStamp()
        if objIDs is None :
            objIDs = list(range(1,len(stimulusState)+1))
        elif len(objIDs)!=len(stimulusState):
            raise RunTimeException("ARGH! objIDs and stimulusState not same length!") 
    
        # insert extra 0 object ID if targetState given
        if not targetState is None and targetState>=0 :
            # N.B. copy longer version to new variable, rather than modify in-place with append
            objIDs = objIDs+[0] 
            stimulusState=stimulusState+[targetState]
        
        return StimulusEvent(timestamp,objIDs,stimulusState)

    def modeChange(self,newmode):
        if self.client :
            self.client.sendMessage(
                ModeChange(self.getTimeStamp(),newmode))

    def log(self,msg):
        if self.client :
            self.client.sendMessage(Log(self.getTimeStamp(),msg))

    def newTarget(self):
        if self.client :
            self.client.sendMessage(NewTarget(self.getTimeStamp()))

    def selection(self,objID):
        if self.client :
            self.client.sendMessage(Selection(self.getTimeStamp(),objID))
        for h in self.selectionHandlers :
            h(objID)         # do selection callbacks
            
    def getNewMessages(self,timeout_ms=0):
        '''get new messages from the utopia-hub, and store the list of new'''
        if not self.client : return None
        # get any messages with predictions
        self.msgs=self.client.getNewMessages(timeout_ms) if self.client else []
        # process these messages as needed & call-callbacks
        if len(self.msgs)>0 :
            for h in self.messageHandlers:
                h(self.msgs)
            newPrediction=None
            for m in self.msgs:
                if m.msgID==PredictedTargetProb.msgID:
                    newPrediction=m
                    # process new prediction callbacks
                    for h in self.predictionHandlers :
                        h(m)
                elif m.msgID==Selection.msgID:
                    # process selection callbacks
                    for h in self.selectionHandlers:
                        h(m.objID)

                elif m.msgID==SignalQuality.msgID:
                    self.lastSignalQuality=m.signalQuality
                    # process selection callbacks
                    for h in self.signalQualityHandlers:
                        h(self.lastSignalQuality)
                    
            if newPrediction and newPrediction.Yest>=0 :
                self.lastPrediction=newPrediction
        return self.msgs

    def getLastPrediction(self):
        '''check for new predictions from the utopia-decoder'''
        # Q: should we do this here? or just return the lastPrediction?
        self.getNewMessages()
        # always return the last prediction, even if no new ones
        return self.lastPrediction
    
    def clearLastPrediction(self):
        '''clear the last predicted target'''
        self.lastPrediction=None
            
    def getLastSelection(self):
        """check if any object prediction is high enough for it to be selected"""
        self.getNewMessages()
        if self.lastPrediction is not None :
            if self.lastPrediction.Perr<self.selectionThreshold : # good enough to select?
                return (self.lastPrediction.Yest,True)

            else: # return predictedObjID but not-selected
                return (self.lastPrediction.Yest,False)
        return (None,False)

    def getLastSignalQuality(self):
        self.getNewMessages()
        return self.lastSignalQuality

def injectERP(amp,host="localhost",port=8300):
    """Inject an erp into a simulated data-stream, sliently ignore if failed, e.g. because not simulated"""
    import socket
    try:
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0).sendto(bytes([amp]),(host,port))
    except: # sliently igore any errors
        pass


def newMessageHandler(msgs):
    for m in msgs:
        print(m)

if __name__=="__main__":
    # simple message logging testcase
    uc = UtopiaController()
    uc.autoconnect()
    # add logging incomming message handler
    uc.addMessageHandler(newMessageHandler)
    while uc.isConnected():
        uc.getNewMessages(1000)
        #msgs=uc.getNewMessages(1000)
        # print the recieved messages
        #for m in msgs:
        #    print(m)
