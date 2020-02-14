#!/usr/bin/python3 
#
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

import struct
import time
import socket
import sys

class UtopiaMessage:
    """Class for a generic UtopiaMessage, i.e. the common structure of all messages """
    def __init__(self,msgID=0,msgName=None,version=0):
        self.msgID   = msgID
        self.msgName = msgName
        self.version = version

    def __str__(self):
        return '%i\n'%(self.msgID)

def getTimeStamp(t0=0):
    return time.perf_counter()*1000-t0

class RawMessage(UtopiaMessage) :
    msgName="RAW"
    """Class for a raw utopia message, i.e. decoded header but raw payload"""
    def __init__(self,msgID,version,payload):
        """Construct a raw message from given msgID and payload"""
        super().__init__(msgID,RawMessage.msgName,version)
        self.payload = payload
        if hasattr(self.payload,'encode'):
            self.payload = self.payload.encode()
        if not isinstance(self.payload,bytes):
            raise Exception("Illegal-value")
        
    @classmethod
    def fromUtopiaMessage(cls,msg):
        """Construct a raw-message wrapper from a normal payload message"""
        return cls(msg.msgID,msg.version,msg.serialize())

    def __str__(self):
        return '%c(%d) [%i]\n'%(chr(self.msgID),self.msgID,len(self.payload))
    
    def serialize(self):
        """convert the raw message to the string to make it network ready"""
        S= struct.pack("<BBH",self.msgID,self.version,len(self.payload))
        S = S + self.payload
        return S

    @classmethod
    def deserialize(cls,buf):
        """Read a raw message from a byte-buffer, return the read message and the number of bytes used from the bytebuffer, or None,0 if message is mal-formed"""
        bufsize = len(buf)
        if bufsize < 4:
            print("Buffer too short for header")
            return (None,0)
        (msgID,ver,msgsize) = struct.unpack('<BBH', buf[0:4])
        # read the rest of the message
        if msgsize > 0:
            if bufsize >= 4+msgsize :
                payload = buf[4:4+msgsize]
            else:
                print("Buffer too short for payload")
                return (None,0)
        else:
            payload = None
        msg=cls(msgID,ver,payload)
        return (msg,4+msgsize)        

    @classmethod
    def deserializeMany(cls,buf):
        """decode multiple RawMessages from the byte-buffer of data, return the length of data consumed."""
        msgs=[]
        nconsumed=0
        while nconsumed < len(buf):            
            (msg,msgconsumed)=RawMessage.deserialize(buf[nconsumed:])
            if  msg==None or msgconsumed == 0 :
                break # bug-out if invalid/incomplete message
            msgs.append(msg)
            nconsumed = nconsumed + msgconsumed
        return (msgs,nconsumed)
    
    
class Heartbeat(UtopiaMessage):    
    """ the HEARTBEAT utopia message class """

    # Static definitions of the class type constants
    msgID=ord('H')
    msgName="HEARTBEAT"
    
    def __init__(self, timestamp=None):
        super().__init__(Heartbeat.msgID,Heartbeat.msgName)
        self.timestamp=timestamp

    def serialize(self):
        """Returns the contents of this event as a string, ready to send over the network,
           or None in case of conversion problems.
        """
        S = struct.pack('<i', int(self.timestamp));
        return S

    def deserialize(buf):
        """Static method to create a HEARTBEAT class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 4:
            return (None,0)
        (timestamp,) = struct.unpack('<i',buf[0:4]);
        msg = Heartbeat(timestamp)
        return (msg,4)
    
    def __str__(self):
        return "%c(%d) %s %i"%(self.msgID,self.msgID,self.msgName,self.timestamp)


class StimulusEvent(UtopiaMessage):
    """ the STIMULUEVENT utopia message class """
    
    # Static definitions of the class type constants
    msgID=ord('E')
    msgName="STIMULUSEVENT"

    def __init__(self, timestamp=None, objIDs=None, objState=None):
        super().__init__(StimulusEvent.msgID,StimulusEvent.msgName)
        self.timestamp=timestamp
        self.objIDs=objIDs
        self.objState=objState

    def serialize(self):
        """Converts this message to a string representation to send over the network
        """
        S = struct.pack("<i", int(self.timestamp)) # timestamp
        S = S + struct.pack("<B",len(self.objIDs))  # nObj
        for objid,objstate in zip(self.objIDs,self.objState):
            S = S + struct.pack("<BB",int(objid),int(objstate)) # [objID,objState] pairs
        return S

    def deserialize(buf):
        """Static method to create a STIMULUSEVENT class from a **PAYLOAD** byte-stream, return created object and the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 5:
            return (None,0)
        (timestamp,nobj) = struct.unpack("<iB",buf[0:5])
        if bufsize < 5+nobj*2:
            return (None,0)
        objIDs=[]
        objState=[]
        for i in range(5,len(buf),2):
            (objid,objstate)=struct.unpack("<BB",buf[i:i+2])
            objIDs.append(objid)
            objState.append(objstate)
        msg=StimulusEvent(timestamp,objIDs,objState)
        return (msg, 5+nobj*2)

    def __str__(self):
        return "%s %i"%(self.msgName,self.timestamp) + "".join("(%i,%i)"%(x,y) for x,y in zip(self.objIDs,self.objState))

class DataPacket(UtopiaMessage):
    """ the DATAPACKET utopia message class """
    
    # Static definitions of the class type constants
    msgID=ord('D')
    msgName="DATAPACKET"

    def __init__(self, timestamp=None, samples=None):
        super().__init__(DataPacket.msgID,DataPacket.msgName)
        self.timestamp=timestamp
        self.samples=samples

    def serialize(self):
        """Converts this message to a string representation to send over the network
        """
        S = struct.pack("<i", int(self.timestamp)) # timestamp
        S = S + struct.pack("<i",len(self.samples))  # nsamp
        for tp in self.samples:
            S = S + struct.pack("<%df"%(len(tp)),*tp)
        return S

    def deserialize(buf):
        """Static method to create a STIMULUSEVENT class from a **PAYLOAD** byte-stream, return created object and the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 8:
            return (None,0)
        timestamp,nsamp = struct.unpack("<ii",buf[0:8])
        nch=int((bufsize-8)/(nsamp*4))
        samples=[]
        for t in range(nsamp):
            sampt=struct.unpack_from("<%df"%(nch),buf,8+t*4*nch)
            samples.append(sampt)
        msg=DataPacket(timestamp,samples)
        return (msg, 8+nsamp*nch*8)

    def __str__(self):
        ss="%c(%d) %s %i "%(self.msgID,self.msgID,self.msgName,self.timestamp)
        ss= ss+ "[%dx%d]"%(len(self.samples),len(self.samples[0]))
        for chs in self.samples:
            chstr = "".join(["%f,"%(c) for c in chs])
            ss = ss + "["+chstr+"]"
        return  ss


class DataHeader(UtopiaMessage):
    """ the DATAHEADER utopia message class """
    
    # Static definitions of the class type constants
    msgID=ord('A')
    msgName="DATAHEADER"

    def __init__(self, timestamp=None, fsample=None, nchannels=None, labels=None):
        super().__init__(DataHeader.msgID,DataHeader.msgName)
        self.timestamp=timestamp
        self.fsample=fsample
        self.nchannels=nchannels
        self.labels=labels

    def serialize(self):
        """Converts this message to a string representation to send over the network
        """
        S = struct.pack("<i", int(self.timestamp)) # timestamp
        S = S + struct.pack("<i",self.nchannels)
        S = S + struct.pack("<f",self.fsample)
        if self.labels : 
            # comma separated list of channel names
            S = S + bytes(",".join(self.labels),'utf-8')
        return S

    def deserialize(buf):
        """Static method to create a HEADER class from a **PAYLOAD** byte-stream, return created object and the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 12:
            return (None,0)
        timestamp,nchannels,fsample = struct.unpack("<iif",buf[0:8])
        labels = buf[4:].decode('utf-8')
        labels = split(",",labels)
        msg=DataHeader(timestamp,nchannels,fsample,labels)
        return (msg, bufsize)

    def __str__(self):
        ss="%c(%d) %s %i "%(self.msgID,self.msgID,self.msgName,self.timestamp)
        ss= ss+ "[%dx%d]"%(len(self.samples),len(self.samples[0]))
        for chs in self.samples:
            chstr = "".join(["%f,"%(c) for c in chs])
            ss = ss + "["+chstr+"]"
        return  ss

class NewTarget(UtopiaMessage):    
    """ the NEWTARGET utopia message class """

    # Static definitions of the class type constants
    msgID=ord('N')
    msgName="NEWTARGET"
    
    def __init__(self, timestamp=None):
        super().__init__(NewTarget.msgID,NewTarget.msgName)
        self.timestamp=timestamp

    def serialize(self):
        """Returns the contents of this event as a string, ready to send over the network,
           or None in case of conversion problems.
        """
        S = struct.pack('<i', int(self.timestamp));
        return S

    def deserialize(buf):
        """Static method to create a HEARTBEAT class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 4:
            return (None,0)
        (timestamp,) = struct.unpack('<i',buf[0:4]);
        msg = NewTarget(timestamp)
        return (msg,4)
    
    def __str__(self):
        return "%c(%d) %s %i"%(self.msgID,self.msgID,self.msgName,self.timestamp)

class Selection(UtopiaMessage):    
    """ the SELECTION utopia message class """

    # Static definitions of the class type constants
    msgID=ord('S')
    msgName="SELECTION"
    
    def __init__(self, timestamp=None, objID=None):
        super().__init__(Selection.msgID,Selection.msgName)
        self.timestamp=timestamp
        self.objID    =objID

    def serialize(self):
        """Returns the contents of this event as a string, ready to send over the network,
           or None in case of conversion problems.
        """
        S = struct.pack('<i', int(self.timestamp))
        S = S + struct.pack('<B', self.objID)
        return S

    def deserialize(buf):
        """Static method to create a SELECTION class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 5:
            return (None,0)
        (timestamp,objID) = struct.unpack('<iB',buf[0:5])
        msg = Selection(timestamp,objID)
        return (msg,5)
    
    def __str__(self):
        return "%c(%d) %s %i id:%i"%(self.msgID,self.msgID,self.msgName,self.timestamp,self.objID)
    
class Reset(UtopiaMessage):    
    """ the RESET utopia message class """

    # Static definitions of the class type constants
    msgID=ord('R')
    msgName="RESET"
    
    def __init__(self, timestamp=None):
        super().__init__(Reset.msgID,Reset.msgName)
        self.timestamp=timestamp

    def serialize(self):
        """Returns the contents of this event as a string, ready to send over the network,
           or None in case of conversion problems.
        """
        S = struct.pack('<i', int(self.timestamp));
        return S

    def deserialize(buf):
        """Static method to create a HEARTBEAT class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 4:
            return (None,0)
        (timestamp,) = struct.unpack('<i',buf[0:4]);
        msg = Reset(timestamp)
        return (msg,4)
    
    def __str__(self):
        return "%c(%d) %s %i"%(self.msgID,self.msgID,self.msgName,self.timestamp)

    

class ModeChange(UtopiaMessage):    
    """ the MODECHANGE utopia message class """

    # Static definitions of the class type constants
    msgID=ord('M')
    msgName="MODECHANGE"
    
    def __init__(self, timestamp=None, newmode=None):
        super().__init__(ModeChange.msgID,ModeChange.msgName)
        self.timestamp=timestamp
        self.newmode  =newmode

    def serialize(self):
        """Returns the contents of this event as a string, ready to send over the network,
           or None in case of conversion problems.
        """
        S = struct.pack('<i', int(self.timestamp))
        S = S + bytes(self.newmode,'utf-8')
        return S

    def deserialize(buf):
        """Static method to create a MODECHANGE class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 4:
            return (None,0)
        (timestamp,) = struct.unpack('<i',buf[0:4])
        newmode = buf[4:].decode('utf-8')
        msg = ModeChange(timestamp,newmode)
        return (msg,4)
    
    def __str__(self):
        return "%c(%d) %s %i %s"%(self.msgID,self.msgID,self.msgName,self.timestamp,self.newmode)

class Log(UtopiaMessage):    
    """ the LOG utopia message class """

    # Static definitions of the class type constants
    msgID=ord('L')
    msgName="LOG"
    
    def __init__(self, timestamp=None, logmsg=None):
        super().__init__(Log.msgID,Log.msgName)
        self.timestamp=timestamp
        self.logmsg  =logmsg

    def serialize(self):
        """Returns the contents of this event as a string, ready to send over the network,
           or None in case of conversion problems.
        """
        S = struct.pack('<i', int(self.timestamp))
        S = S + bytes(self.logmsg,'utf-8')
        return S

    def deserialize(buf):
        """Static method to create a MODECHANGE class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 4:
            return (None,0)
        (timestamp,) = struct.unpack('<i',buf[0:4])
        logmsg = buf[4:].decode('utf-8')
        msg = ModeChange(timestamp,logmsg)
        return (msg,4)
    
    def __str__(self):
        return "%c(%d) %s %i %s"%(self.msgID,self.msgID,self.msgName,self.timestamp,self.logmsg)


class PredictedTargetProb(UtopiaMessage):    
    """ the PREDICTEDTARGETPROB utopia message class """

    # Static definitions of the class type constants
    msgID=ord('P')
    msgName="PREDICTEDTARGETPROB"
    
    def __init__(self, timestamp=None, Yest=0, Perr=1.0):
        super().__init__(PredictedTargetProb.msgID,PredictedTargetProb.msgName)
        self.timestamp=timestamp
        self.Yest     =Yest
        self.Perr     =Perr

    def serialize(self):
        """Returns the contents of this event as a string, ready to send over the network,
           or None in case of conversion problems.
        """
        S = struct.pack('<ibf', int(self.timestamp),int(self.Yest),self.Perr)
        return S

    def deserialize(buf):
        """Static method to create a MODECHANGE class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 4:
            return (None,0)
        (timestamp,Yest,Perr) = struct.unpack('<ibf',buf)
        msg = PredictedTargetProb(timestamp,Yest,Perr)
        return (msg,4)
    
    def __str__(self):
        return "%c(%d) %s %i Yest=%d Perr=%f"%(self.msgID,self.msgID,self.msgName,self.timestamp,self.Yest,self.Perr)


class SignalQuality(UtopiaMessage):    
    """ the SIGNALQUALITY utopia message class """

    # Static definitions of the class type constants
    msgID=ord('Q')
    msgName="SIGNALQUALITY"
    
    def __init__(self, timestamp=None, signalQuality=None):
        super().__init__(SignalQuality.msgID,SignalQuality.msgName)
        self.timestamp=timestamp
        self.signalQuality=signalQuality

    def serialize(self):
        """Returns the contents of this event as a string, ready to send over the network,
           or None in case of conversion problems.
        """
        S = struct.pack('<i', int(self.timestamp))
        S = S + b''.join([ struct.pack('<f',q) for q in self.signalQuality ])
        return S

    def deserialize(buf):
        """Static method to create a SIGNALQUALIYT class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 4:
            return (None,0)
        (timestamp,) = struct.unpack('<i',buf[0:4])
        signalQuality = [ struct.unpack('<f',buf[i:i+4]) for i in range(4,bufsize,4) ]
        msg = SignalQuality(timestamp,signalQuality)
        return (msg,bufsize)
    
    def __str__(self):
        return "%c(%d) %s %i [%s]"%(self.msgID,self.msgID,self.msgName,self.timestamp,",".join(["%f"%q for q in self.signalQuality]))

class Subscribe(UtopiaMessage):    
    """ the SUBSCRIBE utopia message class """

    # Static definitions of the class type constants
    msgID=ord('B')
    msgName="SUBSCRIBE"
    
    def __init__(self, timestamp=None, messageIDs=None):
        super().__init__(Subscribe.msgID,Subscribe.msgName)
        self.timestamp=timestamp
        self.messageIDs=messageIDs

    def serialize(self):
        """Returns the contents of this event as a string, ready to send over the network,
           or None in case of conversion problems.
        """
        S = struct.pack('<i', int(self.timestamp))
        if type(self.messageIDs) is str :#str
            S = S + bytes(self.messageIDs,'utf-8')
        else:# int->byte
            S = S + b''.join([ struct.pack('<b',q) for q in self.messageIDs ])
        return S

    def deserialize(buf):
        """Static method to create a SIGNALQUALIYT class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf"""
        bufsize = len(buf)
        if bufsize < 4:
            return (None,0)
        (timestamp,) = struct.unpack('<i',buf[0:4])
        messageIDs = [ struct.unpack('<i',buf[i:i+4]) for i in range(4,bufsize,4) ]
        msg = Subscribe(timestamp,messageIDs)
        return (msg,bufsize)
    
    def __str__(self):
        return "%c(%d) %s %i [%s]"%(self.msgID,self.msgID,self.msgName,self.timestamp,",".join(["%i"%q for q in self.messageIDs]))

    
# Helper functions for dealing with raw-messages -> classes    
def decodeRawMessage(msg):
    """decode utopia RawMessages into actual message objects"""
    if msg.msgID==StimulusEvent.msgID:
        (decodedmsg,nconsumed)= StimulusEvent.deserialize(msg.payload)
    elif msg.msgID==Heartbeat.msgID:
        (decodedmsg,nconsumed) = Heartbeat.deserialize(msg.payload)
    elif msg.msgID==PredictedTargetProb.msgID:
        (decodedmsg,nconsumed) = PredictedTargetProb.deserialize(msg.payload)
    elif msg.msgID==Selection.msgID:
        (decodedmsg,nconsumed) = Selection.deserialize(msg.payload)
    elif msg.msgID==ModeChange.msgID:
        (decodedmsg,nconsumed) = ModeChange.deserialize(msg.payload)
    elif msg.msgID==NewTarget.msgID:
        (decodedmsg,nconsumed) = NewTarget.deserialize(msg.payload)
    elif msg.msgID==SignalQuality.msgID:
        (decodedmsg,nconsumed) = SignalQuality.deserialize(msg.payload)
    elif msg.msgID==Reset.msgID:
        (decodedmsg,nconsumed) = Reset.deserialize(msg.payload)
    elif msg.msgID==DataPacket.msgID:
        (decodedmsg,nconsumed) = DataPacket.deserialize(msg.payload)
    else:
        decodedmsg = msg
    return decodedmsg

def decodeRawMessages(msgs):
    """decode utopia RawMessages into actual message objects"""
    return [ decodeRawMessage(msg) for msg in msgs ]

def ssdpDiscover(servicetype=None,timeout=3,numretries=1):
    '''auto-discover the utopia-hub using ssdp discover messages'''
    ssdpgroup = ("239.255.255.250", 1900)
    msearchTemplate = "\r\n".join([
        'M-SEARCH * HTTP/1.1',
        'HOST: {0}:{1}',
        'MAN: "ssdp:discover"',
        'ST: {st}', 'MX: {mx}', '', ''])
    # make and send the discovery message
    service= servicetype if servicetype is not None else "ssdp:all"
    msearchMessage = msearchTemplate.format(*ssdpgroup, st=service, mx=timeout)
    if sys.version_info[0] == 3: msearchMessage = msearchMessage.encode("utf-8")
    try:
        # make the UDP socket to the multicast group with timeout
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP,1)
        sock.settimeout(timeout)
    except: # abort if cant setup socket
        return () 
    responses=[]
    for retry in range(numretries):
        print("Sending query message:\n%s"%(msearchMessage))
        try:
            sock.sendto(msearchMessage, ssdpgroup)
        except: # abort if send fails
            return ()
        # wait for responses and store the location info for the matching ones
        print("Waiting responses")
        try:
            rsp,addr=sock.recvfrom(8192)
            rsp=rsp.decode('utf-8')
            print("Got response from : %s\n%s\n"%(addr,rsp))
            # does the response contain servertype, if so then we match
            location=None
            if servicetype is None or servicetype in rsp :
                print("Response matches server type: %s"%(servicetype))
                # use the message source address as default location
                location=addr 
                # extract the location or IP from the message
                for line in rsp.split("\r\n"): # cut into lines
                    tmp=line.split(":",1) # cut line into key/val
                    # is this the key we care about -> then record the value
                    if len(tmp)>1 and tmp[0].lower()=="LOCATION".lower() :
                        location=tmp[1].strip()
                        # strip http:// xxxx /stuff
                        if location.startswith("http://"):
                            location=location[7:] # strip http
                        if '/' in location :
                            location=location[:location.index('/')] # strip desc.xml
                        print("Got location: %s"%(location))
                        break # done with this response
                # add to the list of possible servers
                print("Loc added to response list: %s"%(location))
                responses.append(location)
        except socket.timeout:
            print("Socket timeout")
        if len(responses)>0 : break
    return responses

class UtopiaClient:
    """Class for managing a client connection to a UtopiaServer."""

    UTOPIA_SSDP_SERVICE="utopia/1.1"
    DEFAULTHOST='localhost'
    DEFAULTPORT=8400    
    HEARTBEATINTERVAL_ms=1000
    HEARTBEATINTERVALUDP_ms=200
    
    def __init__(self):
        self.isConnected = False
        self.sock = []
        self.udpsock=None
        self.recvbuf = b''
        self.nextHeartbeatTime=self.getTimeStamp()
        self.nextHeartbeatTimeUDP=self.getTimeStamp()

    def getAbsTime(self):
        """Get the absolute time in seconds"""
        return getTimeStamp()
    def getTimeStamp(self):
        """Get the time-stamp for the current time"""
        return getTimeStamp()

    def connect(self, hostname=None, port=None):
        """connect([hostname, port]) -- make a connection, default host:port is localhost:1972"""
        if hostname is None:       hostname = UtopiaClient.DEFAULTHOST
        if port is None or port<0: port     = UtopiaClient.DEFAULTPORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((hostname, port))
        self.sock.setblocking(False)
        self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) # disable nagle
        self.isConnected = True
        # make udp socket also
        self.udpsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        # ensure tcp and udp have the same local port number...
        self.udpsock.bind(self.sock.getsockname())

    def autoconnect(self,hostname=None,port=None,timeout_ms=3000):
        if hostname is None :
            print('Trying to auto-discover the utopia-hub server');
            hosts=ssdpDiscover(servicetype=UtopiaClient.UTOPIA_SSDP_SERVICE,timeout=5,numretries=int(max(1,timeout_ms/5000)))
            print("Discovery returned %d utopia-hub servers"%len(hosts))
            if( len(hosts)>0 ):
                hostname=hosts[0].strip()
                print('Discovered utopia-hub on %s ...'%(hostname))

                if ":" in hostname :
                    hostname,port=hostname.split(":")
                    port=int(port)

        print("Trying to connect to: %s:%d"%(hostname,port))
        for i in range(int(timeout_ms/1000)):
            try:
                self.connect(hostname, port)
                print('Connected!',flush=True)
                break
            except socket.error as ex:
                print('Connection refused...  Waiting',flush=True)
                print(ex)
                time.sleep(1)
        if not self.isConnected:
            raise socket.error('Connection Refused!')

    def gethostport(self):
        if self.isConnected:
            hp=self.sock.getpeername()
            return ":".join(str(i) for i in hp)
        return None
        
    def disconnect(self):
        """disconnect() -- close a connection."""
        if self.isConnected:
            self.sock.close()
            self.sock = []
            self.isConnected = False

    def sendRaw(self, request):
        """Send all bytes of the string 'request' out to socket."""
        if not(self.isConnected):
            raise IOError('Not connected to utopia server')

        N = len(request);
        nw = self.sock.send(request)
        while nw<N:
            nw += self.sock.send(request[nw:])

    def sendRawUDP(self,request):
        self.udpsock.sendto(request,self.sock.getsockname())

    def sendMessage(self, msg):
        if not msg is RawMessage : # convert to raw for sending
            if msg.timestamp <= 0 : # insert valid time-stamp, N.B. all messages have timestamp!
                msg.timestamp = self.getTimeStamp()
            msg = RawMessage.fromUtopiaMessage(msg)
        self.sendRaw(msg.serialize()) # send the raw message directly
        self.sendHeartbeatIfTimeout()

    def sendHeartbeatIfTimeout(self):
        curtime = self.getTimeStamp()
        if curtime > self.nextHeartbeatTime :
           self.nextHeartbeatTime=curtime+self.HEARTBEATINTERVAL_ms
           self.sendRaw(RawMessage.fromUtopiaMessage(Heartbeat(curtime)).serialize())
        if curtime > self.nextHeartbeatTimeUDP :
           self.nextHeartbeatTimeUDP=curtime+self.HEARTBEATINTERVALUDP_ms
           self.sendRawUDP(RawMessage.fromUtopiaMessage(Heartbeat(curtime)).serialize())
        
    def sendMessages(self, msgs):
        """sends single or multiple utopia-messages to the utopia server
        """
        for msg in msgs:
            sendMessage(msg)
            
    def recvall(self, timeout_ms=0):
        """Read all the data from the socket or block for timeout_ms if nothing to do."""
        self.sock.setblocking(0)        
        data=[] 
        endtime=self.getAbsTime() + timeout_ms/1000
        while True: # guarantee at least once through recv
            try:
                part = self.sock.recv(1024) # non-blocking read what the socket has at this point
            except socket.error as ex:
                part =[]
            if len(part)>0 : # read data add to the store
                data += part
            elif len(data)>0 :
                break  # stop if had some data and read it all
            else: # no data yet
                if self.getAbsTime() > endtime :
                    break # quit if timeout 
                else: # poll again in 1ms
                    time.sleep(.001)  
        return bytes(data)
    
    def getNewMessages(self, timeout_ms=250):
        """Wait for new messages from the utopia-server (with optional timeout), decode them and return the list of new messages"""
        # get all the data in the socket
        buf = self.recvall(timeout_ms)
        # append to the receive buffer
        self.recvbuf = self.recvbuf + buf
        # decode all the new messages to RawMessage
        (newmessages, nconsumed) = RawMessage.deserializeMany(self.recvbuf)
        self.recvbuf = self.recvbuf[nconsumed:] # remove the consumed part of the buffer
        # decode the RawMessages to actual messages
        newmessages = decodeRawMessages(newmessages)
        self.sendHeartbeatIfTimeout()
        return newmessages

    def initClockAlign(self, delays_ms=[50]*10):
        """Send some initial heartbeat messages to seed the alignment of the server clock with
        our local clock"""
        self.sendMessage(Heartbeat(self.getTimeStamp()))
        for delay in delays_ms:
            time.sleep(delay/1000)
            self.sendMessage(Heartbeat(self.getTimeStamp()))
    
    def messagelogger(self,timeout_ms=1000):
        """Simple message logger, infinite loop waiting for and printing messages from the server"""
        while True:
            newmessages = client.getNewMessages(timeout_ms)
            print("%d new messages:"%len(newmessages))
            print("\n".join(str(msg) for msg in newmessages))


    def messagePingPong(self,timeout_ms=500):
        """Testing system sending 20 messages and printing any responses """
        # loop waiting for messages and printing them
        for i in range(20):
            self.sendMessage(StimulusEvent(i,[0,1,2,3],[i+1,i+2,i+3,i+4]))
            time.sleep(1)
            newmessages = self.getNewMessages(timeout_ms)
            print("%d) %d NewMessages\n"%(i,len(newmessages)))
            print("\n".join(str(msg) for msg in newmessages))

            
def testSerialization():
    rm=RawMessage(1,0,"payload")
    print("RawMessage: %s"%(rm))
    print("serialized: %s"%(rm.serialize()))
    print("deserialized : %s"%(RawMessage.deserialize(rm.serialize())[0]))
    print()

    hb=Heartbeat(10)
    print("Heartbeat: %s"%(hb))
    print("serialized: %s"%(hb.serialize()))
    print("deserialized : %s"%(Heartbeat.deserialize(hb.serialize())[0]))
    print()
    
    se=StimulusEvent(11,[0,1,2],[0,1,0])
    print("StimulusEvent: %s"%(se))
    print("serialized: %s"%(se.serialize()))
    print("deserialized : %s"%(StimulusEvent.deserialize(se.serialize())[0]))
    print()

    mc=ModeChange(12,"hello")
    print("ModeChange : %s"%(mc))
    print("serialized : %s"%(mc.serialize()))
    print("deserialized : %s"%(ModeChange.deserialize(mc.serialize())[0]))
    print()

    nt=NewTarget(13)
    print("NewTarget : %s"%(nt))
    print("serialized : %s"%(nt.serialize()))
    print("deserialized : %s"%(NewTarget.deserialize(nt.serialize())[0]))
    print()
  
    rs=Reset(14)
    print("Reset : %s"%(rs))
    print("serialized : %s"%(rs.serialize()))
    print("deserialized : %s"%(Reset.deserialize(rs.serialize())[0]))
    print()

    pt=PredictedTargetProb(15,10,.5)
    print("PredictedTargetProb : %s"%(pt))
    print("serialized          : %s"%(pt.serialize()))
    print("deserialized : %s"%(PredictedTargetProb.deserialize(pt.serialize())[0]))
    print()
    
    sq=SignalQuality(16,[.1,.2,.3,.4,.5])
    print("SignalQuality : %s"%(sq))
    print("serialized    : %s"%(sq.serialize()))
    print("deserialized : %s"%(SignalQuality.deserialize(sq.serialize())[0]))
    print()
    
    dp=DataPacket(16,[[.1,.2,.3],[.4,.5,.6]])
    print("DataPacket  : %s"%(dp))
    print("serialized  : %s"%(dp.serialize()))
    print("deserialized: %s"%(DataPacket.deserialize(dp.serialize())[0]))
    print()
    

    rhb=RawMessage.fromUtopiaMessage(hb)
    print("Raw(Heartbeat): %s"%(rhb))

    srhb=rhb.serialize()
    print("serialized    : %s"%(srhb))
    (dsrhb,nconsumed)=RawMessage.deserialize(srhb)
    print("Deserialized serialized Raw(Heartbeat) : %s"%(dsrhb))
    sedsrhb=decodeRawMessage(dsrhb)
    print("Decoded deserialized Raw(Heartbeat): %s"%(sedsrhb))

    rse=RawMessage.fromUtopiaMessage(se)
    print("Raw(StimulusEvent): %s"%(rse))
    srse=rse.serialize()
    print("serialized        : %s"%(srse))    
    #deserialize
    (dsrse,nconsumed)=RawMessage.deserialize(srse)
    print("Deserialized serialized Raw(StimulusEvent) : %s"%(dsrse))
    sedsrse=decodeRawMessage(dsrse)
    print("Decoded deserialized Raw(StimulusEvent): %s"%(sedsrse))

    #deserialize multiple
    (msgs,nconsumed)=RawMessage.deserializeMany(srse+srse+srhb)
    print("Deserialized Multiple : %s"%("\n".join([str(msg) for msg in msgs])))
    msgs=decodeRawMessages(msgs)
    print("Decoded Multiple : %s"%("\n".join([str(msg) for msg in msgs])))



def testSending():
    client = UtopiaClient()        
    client.autoconnect()

    # DataPacket
    client.sendMessage(DataPacket(client.getTimeStamp(),[[.1,.2,.3]]))

    # StimulusEvent
    time.sleep(5)

    
if __name__ == "__main__":

    #testCases()

    #hosts=ssdpDiscover(servicetype="ssdp:all",timeout=3)
    #print(hosts)


    # Just a small logging demo for testing purposes...

    hostname = None
    port = None
    
    if len(sys.argv)>1:
        hostname = sys.argv[1]
        tmp=[hostname.split(":")]
        if len(tmp)>1:
            hostname=tmp[0];
            port    =int(tmp[1])

    client = UtopiaClient()        
    client.autoconnect(hostname,port)
    client.messagelogger(30000)
    client.disconnect()
    
