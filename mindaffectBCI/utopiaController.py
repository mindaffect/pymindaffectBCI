from .utopiaclient import *

class UtopiaController:
    '''controller class to manage the interaction with the Mindaffect decoder,
    setting up the connection, sending and recieving messages, and firing message
    event handlers'''
    def __init__(self):
        self.client=UtopiaClient()
        self.msgs=[]
        self.lastPrediction=None
        # callback list for new messages
        self.messageHandlers=[]
        # call back list for new predictions
        self.predictionHandlers=[]
        # selection stuff
        self.selectionHandlers=[]
        self.selectionThreshold=.1

    def addMessageHandler(self,cb):
        self.messageHandlers.append(cb)
    def addPredictionHandler(self,cb):
        self.predictionHandlers.append(cb)
    def addSelectionHandler(self,cb):
        self.selectionHandlers.append(cb)
        
    def getTimeStamp(self,t0=0):
        '''get a (relative) wall-time stamp *in milliseconds*'''
        if self.client:
            return self.client.getTimeStamp()-t0
        return time.perf_counter()*1000-t0    
        
    def autoconnect(self,host=None,port=8400,timeout_ms=5000): 
        try : 
            self.client.autoconnect(host,port,timeout_ms=timeout_ms)
        except :
            pass

        # ask user for host
        if not self.client.isConnected :
            print("Could not auto-connect.  Trying manual")
            hostport=input("Enter the hostname/IP of the Utopia-HUB: ")
            try:
                self.client.autoconnect(hostport)
            except :
                print("Could not connect to %s. Run in disconnected!"%(hostport))

            if not self.client.isConnected :
                print("Warning:: couldnt connect to a utopia hub....")
                self.client=None
            # subscribe to PREDICTEDTARGETPROB, MODECHANGE, SELECTION and NEWTARGET messages only
            if self.client:
                self.client.sendMessage(
                    Subscribe(self.getTimeStamp(),"PMSN"))

    def isConnected(self): return self.client.isConnected
                
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
        if not targetState is None :
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
                    
            if newPrediction:
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

def injectERP(amp=1,host="localhost",port=8300):
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
