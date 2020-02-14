#!/bin/env python3

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
from utopia2output import *

class Utopia2Cybathalon(Utopia2Output):    
    """Example class for a utopia OUTPUT module.  Connects to the utopia server
       and listens for output which exceed it's probability threshold before
       then printing them and using NEWTARGET to indicated the output has taken
       place. """
    
    def __init__(self,outputPressThreshold=.1,outputReleaseThreshold=None,myObjectIDs=None):
        super().__init__(outputPressThreshold,outputReleaseThreshold,myObjectIDs)

        self.br_player=1

        # Command offsets, do not change.
        self.CMD_SPEED= 1
        self.CMD_JUMP = 2
        self.CMD_ROLL = 3
        self.CMD_RST  = 99
        
        # Command configuration
        self.CMDS      = [self.CMD_ROLL, self.CMD_RST, self.CMD_JUMP, self.CMD_SPEED]
        self.THRESHOLDS= [.1,        .1,       .1,     .1      ]

        #Connect to BrainRacers
        self.br_hostname=None
        self.br_port=None

    def connect(self,host=UtopiaClient.DEFAULTHOST,port=UtopiaClient.DEFAULTPORT,br_host='localhost',br_port=5555,timeout_ms=30000):
        super().connect(host,port,timeout_ms)
        self.connecttocybathalon(br_host,br_port)
                    
    def doOutput(self,objID):
        """This function is run when objID has sufficiently low error to mean that 
           and output should be generated for this objID. 
           N.B. Override/Replace this function with your specific output method."""
        if self.myObjectIDs : 
            command = self.myObjectIDs.index(objID)
        else:
            command = objID

        if command : 
            print("Send cmd " + str(command) )
            self.sendtocybathalon(command)

    def connecttocybathalon(self,br_host,br_port):
        '''setup the connection to the cybathalon game'''
        self.br_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM);
        self.br_hostname=br_host
        self.br_port    =br_port
        print("Connecting to BrainRacers on: %s:%d"%(self.br_hostname,self.br_port))

            
    def sendtocybathalon(self,command):
        '''send the given command to the cybathalon game'''
        cmd = (self.br_player * 10) + command # command is [player,command] as number
        data = struct.pack('B', cmd)
        
        self.br_socket.sendto(data, (self.br_hostname, self.br_port))

            
''' simple driver for testing '''
if __name__ == "__main__":
    import sys
    host = UtopiaClient.DEFAULTHOST
    port = UtopiaClient.DEFAULTPORT
    br_host='localhost'
    br_port=5555

    if len(sys.argv)>1:
        hostname = sys.argv[1]
        tmp=hostname.split(":")
        if len(tmp)>1:
            hostname=tmp[0]
            port    =int(tmp[1])
    print('utopia: %s:%d'%(hostname,port))
            
    outputthreshold=.1
    if len(sys.argv)>2:
        try:
            outputthreshold = float(sys.argv[2])
        except:
            outputthreshold = None
            print('Non-numerical outputthreshold, using selection mode')
            
    if len(sys.argv)>3:
        br_host = sys.argv[3]
        print('br_host:br_port %s:%d'%(br_host,br_port))
        tmp=br_host.split(":")
        if len(tmp)>1:
            br_host =tmp[0]
            br_port =int(tmp[1])
    print('Cybathalon %s:%d'%(br_host,br_port))

    u2c = Utopia2Cybathalon(outputthreshold)
    u2c.connect(host,port,br_host,br_port)

    u2c.sendtocybathalon(u2c.CMD_SPEED)

    u2c.run()
