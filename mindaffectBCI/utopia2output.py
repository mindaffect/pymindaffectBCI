#!/usr/bin/python3

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


from .utopiaclient import UtopiaClient,Subscribe,PredictedTargetProb,Selection
class Utopia2Output:    
    """Example class for a utopia OUTPUT module.  Connects to the utopia server
    and then either, depending on mode:
        a)  listens for output which exceed it's probability threshold before
    then printing them and using NEWTARGET to indicated the output has taken
    place
    """
    
    def __init__(self,outputPressThreshold=None,outputReleaseThreshold=None,objectID2Action=None):
        self.VERB=0 
        self.outputPressThreshold=outputPressThreshold
        if outputReleaseThreshold is not None :
            self.outputReleaseThreshold=outputPressThreshold*2
        else:
            self.outputReleaseThreshold=outputReleaseThreshold
        # this dictionay contains the functions to execute for the
        # selection IDs we are responsible for. 
        self.objectID2Action=objectID2Action 
        self.client = UtopiaClient()

    def connect(self,host=None,port=None,timeout_ms=30000):
        """[summary]

        Args:
            host ([type], optional): [description]. Defaults to None.
            port ([type], optional): [description]. Defaults to None.
            timeout_ms (int, optional): [description]. Defaults to 30000.
        """        
        print('Connecting to utopia on',host,":",port,",",timeout_ms)
        self.client.autoconnect(host,port,timeout_ms)
        self.client.initClockAlign()
        if self.outputPressThreshold is None :
            # selection mode => subscribe only to selection messages
            self.client.sendMessage(Subscribe(self.client.getTimeStamp(),"S"))
            print('In SelectionMode')
        else:
            # perr press/release mode => subscribe to probability output messages
            self.client.sendMessage(Subscribe(self.client.getTimeStamp(),"PS"))
            print('In PerrMode')

    def perrModeOutput(self,msgs):
        """
        process a perr-message generating appropriate output.  
           To avoid 'key-bounce' we use a press-release semantics, where the output is 'Pressed' 
           so the output is generated when the Perr < outputPressThreshold, then further
           output is inhibited until Perr > outputReleaseThreshold.

        Args:
            msgs ([type]): [description]
        """        
        
        for msg in msgs:
            if not msg.msgID==PredictedTargetProb.msgID: continue
            #print('OutputnMode:',msg)

            if msg.Perr < self.outputPressThreshold and not self.outputActivated :# low enough error, not already activated
                self.client.sendMessage(Selection(self.client.getTimeStamp(),msg.Yest)) # notify we've made selection
                self.outputActivated=True # update state to output activated
                self.doOutput(msg.objID) # call function to actually make the output happen
            elif msg.Perr > self.outputReleaseThreshold and self.outputActivated : # high-enough error, already activated
                self.outputActivated=False # relase output -> allow a new selection

    def selectionModeOutput(self,msgs):
        """
        Process selection message to generate output.  
        Basically generate output if the messages objectID is one of the ones
        we are tasked with generating output for

        Args:
            msgs ([type]): [description]
        """        
        
        for msg in msgs:
            if not msg.msgID==Selection.msgID: 
                continue
            print('SelnMode:',msg)
            self.doOutput(msg.objID) # call function to make output happen

    def run(self,timeout_ms=3000):
        """
        mainloop of utopia-to-output mapping
        runs an infinite loop, waiting for new messages from utopia, filtering out 
        those mesages which contain an output prediction (i.e. PREDICTEDTARGETPROB message)
        and if the output prediction is sufficiently confident forwarding this to the output
        device and sending a NEWTARGET to the recogniser to indicate the output was sent
        
        Args:
            timeout_ms (int, optional): [description]. Defaults to 3000.
        """        
        
        if not self.client.isConnected :
            self.connect()
        print("Waiting for messages")
        self.outputActivated=False
        while True:
            newmsgs = self.client.getNewMessages(timeout_ms)
            if not self.outputPressThreshold is None:
                # Perr output mode
                self.perrModeOutput(newmsgs)
            elif self.outputPressThreshold is None:
                # Perr output mode
                self.selectionModeOutput(newmsgs)
                
            print('.',end='')

            
    def doOutput(self,objID):
        """This function is run when objID has sufficiently low error to mean that 
        and output should be generated for this objID. 
        N.B. Override/Replace this function with your specific output method."""
        if self.objectID2Action is None : 
            print("Generated output for Target %d"%(objID))
        else :
            try :
                action = self.objectID2Action[int(objID)]
                action(objID)
            except KeyError:
                pass

''' simple driver for testing '''
if __name__ == "__main__":
    import sys
    host = None
    port = None
    
    if len(sys.argv)>1:
        hostname = sys.argv[1]
        tmp=[hostname.split(":")]
        if len(tmp)>1:
            hostname=tmp[0]
            port    =tmp[1]

    outputthreshold=None
    if len(sys.argv)>2:
        try:
            outputthreshold = float(sys.argv[2])
        except:
            outputthreshold = None
            print('Non-numerical outputthreshold, using selection mode')
            
    u2o = Utopia2Output(outputthreshold)
    u2o.connect(host,port)
    u2o.run()
