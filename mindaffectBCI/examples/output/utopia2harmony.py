#!/bin/env python3
#

# Copyright (c) 2019 MindAffect B.V. 
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


from utopiaclient import *
import subprocess
class Utopia2Harmony:    
    """Example class for a utopia OUTPUT module.  Connects to the utopia server
    and listens for output which exceed it's probability threshold before
    then printing them and using NEWTARGET to indicated the output has taken
    place
    """
    
    def __init__(self,outputPressThreshold=.1,outputReleaseThreshold=None,myObjectIDs=None):
        self.VERB=0
        self.outputPressThreshold=outputPressThreshold
        if outputReleaseThreshold is not None :
            self.outputReleaseThreshold=outputPressThreshold*2
        else:
            self.outputReleaseThreshold=outputReleaseThreshold
        self.myObjectIDs=myObjectIDs # this is the set of object IDs we generate output for
        self.client = UtopiaClient()

    def connect(self,host=UtopiaClient.DEFAULTHOST,port=UtopiaClient.DEFAULTPORT,timeout_ms=30000):
        self.client.autoconnect(host,port,timeout_ms)
        self.client.initClockAlign()

    def perrModeOutput(self,msg):
        """process a perr-message generating appropriate output.  
           To avoid 'key-bounce' we use a press-release semantics, where the output is 'Pressed' 
           so the output is generated when the Perr < outputPressThreshold, then further
           output is inhibited until Perr > outputReleaseThreshold."""
        if self.myObjectIDs is None or msg.Yest in self.myObjectIDs : # one of "our" outputs
            if msg.Perr < self.outputPressThreshold and not self.outputActivated : # low enough error, not already activated
                self.doOutputHarmony(msg.Yest) # call function to actually make the output happen
                # TODO [] : send a selection message rather than NewTarget to indicate *what* was selected
                self.client.sendMessage(NewTarget(self.client.getTimeStamp())) # notify we've made selection
                self.outputActivated=True # update state to output activated
            elif msg.Perr > self.outputReleaseThreshold and self.outputActivated : # high-enough error, already activated
                self.outputActivated=False # relase output -> allow a new selection

    def selectionModeOutput(self,msg):
        """ Process selection message to generate output.  
            Basically generate output if the messages objectID is one of the ones
            we are tasked with generating output for"""
        if self.myObjectIDs is None or msg.objID in self.myObjectIDs : # it's one of "our" outputs
            self.doOutputHarmony(msg.objID) # call function to make output happen

    def run(self,timeout_ms=3000):
        """mainloop of utopia-to-output mapping
        runs an infinite loop, waiting for new messages from utopia, filtering out 
        those mesages which contain an output prediction (i.e. PREDICTEDTARGETPROB message)
        and if the output prediction is sufficiently confident forwarding this to the output
        device and sending a NEWTARGET to the recogniser to indicate the output was sent
        """
        if not self.client.isConnected :
            self.connect()
        print("Waiting for messages")
        self.outputActivated=False
        while True:
            newmsgs = self.client.getNewMessages(timeout_ms)
            for msg in newmsgs:
                if self.VERB>1 : print("Got message " + str(msg) + " <- server")
                if msg.msgID == PredictedTargetProb.msgID and not self.outputPressThreshold is None :
                    # Perr message and in Perr based output mode
                    self.perrModeOutput(msg)
                    
                elif msg.msgID == Selection.msgID and self.outputPressThreshold is None :
                    # Selection message in in Selection based output mode
                    self.selectionModeOutput(msg)
                if self.VERB>0 : print('.')


            
    def doOutputHarmony(self,objID):
        """This function is run when objID has sufficiently low error to mean that 
        and output should be generated for this objID. 
        N.B. Override/Replace this function with your specific output method."""
        print("Generated output for Target %d"%(objID))
        #Set the ip address of the harmony hub
        ip_address = "192.168.1.31"
        # set the device id
        id_Telenet = "63961272"
        id_Apple_TV = "63944910"
        # check for selection messages and send the related infrared/bluetooth signal
        if objID ==96 : 	  
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Telenet, "--command", "PowerToggle"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==97 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Telenet, "--command", "VolumeUp"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==98 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Telenet, "--command", "VolumeDown"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==99 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Telenet, "--command", "ChannelUp"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==100 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Telenet, "--command", "ChannelDown"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==101 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Telenet, "--command", "Mute"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==104 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Apple_TV, "--command", "DirectionUp"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==105 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Apple_TV, "--command", "DirectionRight"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==106 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Apple_TV, "--command", "DirectionDown"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==107 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Apple_TV, "--command", "DirectionLeft"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==108 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Apple_TV, "--command", "select"]);
            print("Target %d is good enough!"%(objID))
        elif objID ==109 : 
            rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Apple_TV, "--command", "menu"]);
            print("Target %d is good enough!"%(objID))
        else :
           # rtn = subprocess.call(["aioharmony","--harmony_ip",ip_address,"send_command","--device_id", id_Apple_TV, "--command", "menu"]);
            print("an Unprogrammed button (ID = %d) is chosen!"%(objID))


''' simple driver for testing '''
if __name__ == "__main__":
    import sys
    host = UtopiaClient.DEFAULTHOST
    port = UtopiaClient.DEFAULTPORT
    
    if len(sys.argv)>1:
        hostname = sys.argv[1]
        tmp=[hostname.split(":")]
        if len(tmp)>1:
            hostname=tmp[0];
            port    =tmp[1]

    outputthreshold=.1
    if len(sys.argv)>2:
        try:
            outputthreshold = float(sys.argv[2])
        except:
            print('argument should be a valid floating point threshold'%sys.argv[2])
            
    u2harmony = Utopia2Harmony(None)
    u2harmony.connect(host,port)
    u2harmony.run()
