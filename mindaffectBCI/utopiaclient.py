#!/usr/bin/env python3
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

""" This module contains the message classes for communication with the mindaffect BCI hub and a simple client for connecting to and managing this connection.
"""
import struct
import time
import socket
import sys

class UtopiaMessage:
    """
    Class for a generic UtopiaMessage, i.e. the common structure of all messages
    """    
    def __init__(self, msgID=0, msgName=None, version=0):
        """    Class for a generic UtopiaMessage, i.e. the common structure of all messages

        Args:
            msgID (int, optional): unique message ID. Defaults to 0.
            msgName (str, optional): unique message name. Defaults to None.
            version (int, optional): message version. Defaults to 0.
        """        
        self.msgID   = msgID
        self.msgName = msgName
        self.version = version

    def __str__(self):
        return '%i\n'%(self.msgID)

class TimeStampClock:
    """
    Base class for time-stamp sources.  Match this prototype to replace the default timestamp source

    """    
    
    def getTimeStamp(self):
        """
        get the time-stamp in milliseconds ! N.B. **must** fit in an int32!

        Returns:
            int: the timestamp
        """        
        return (int(time.perf_counter()*1000) % (1<<31))

class RawMessage(UtopiaMessage):
    """
    Class for a raw utopia message, i.e. decoded header but raw payload
    """    

    msgName="RAW"
    
    def __init__(self, msgID, version, payload):
        """for a raw utopia message, i.e. decoded header but raw payload

        Args:
            msgID (int, optional): unique message ID. Defaults to 0.
            version (int, optional): message version. Defaults to 0.
            payload (bytes): the raw payload

        Raises:
            Exception: if the payload is not bytes
        """        
        super().__init__(msgID, RawMessage.msgName, version)
        self.payload = payload
        if hasattr(self.payload, 'encode'):
            self.payload = self.payload.encode()
        if not isinstance(self.payload, bytes):
            raise Exception("Illegal-value")
        
    @classmethod
    def fromUtopiaMessage(cls, msg:UtopiaMessage):
        """
        Construct a raw-message wrapper from a normal payload message

        Args:
            msg (UtopiaMessage): the message to make to raw

        Returns:
            RawMessage: the raw version of this utopia message
        """    
        return cls(msg.msgID, msg.version, msg.serialize())

    def __str__(self):
        return '%c(%d) [%i]\n'%(chr(self.msgID), self.msgID, len(self.payload))
    
    def serialize(self):
        """
        convert the raw message to the string to make it network ready

        Returns:
            bytes: the byte serialized version of this raw message
        """        
        
        S= struct.pack("<BBH", self.msgID, self.version, len(self.payload))
        S = S + self.payload
        return S

    @classmethod
    def deserialize(cls, buf):
        """
        Read a raw message from a byte-buffer, return the read message and the number of bytes used from the bytebuffer, or None, 0 if message is mal-formed

        Args:
            buf (bytes): the byte buffer to read from

        Returns:
            RawMessage: the decoded raw message
        """        
        bufsize = len(buf)
        if bufsize < 4:
            print("Buffer too short for header")
            return (None, 0)
        (msgID, ver, msgsize) = struct.unpack('<BBH', buf[0:4])
        # read the rest of the message
        if msgsize > 0:
            if bufsize >= 4+msgsize:
                payload = buf[4:4+msgsize]
            else:
                print("Buffer too short for payload: id:{}, ver:{}, sz:{}".format(chr(msgID), ver, msgsize))
                return (None, 0)
        else:
            payload = None
        msg=cls(msgID, ver, payload)
        return (msg, 4+msgsize)        

    @classmethod
    def deserializeMany(cls, buf):
        """
        decode multiple RawMessages from the byte-buffer of data, return the length of data consumed.

        Args:
            buf (bytes): the byte buffer to read from

        Returns:
            list-of-RawMessage: the list of decoded raw messages
        """        
        msgs=[]
        nconsumed=0
        while nconsumed < len(buf):            
            (msg, msgconsumed)=RawMessage.deserialize(buf[nconsumed:])
            if  msg==None or msgconsumed == 0:
                break # bug-out if invalid/incomplete message
            msgs.append(msg)
            nconsumed = nconsumed + msgconsumed
        return (msgs, nconsumed)
    
    
class Heartbeat(UtopiaMessage):    
    """the HEARTBEAT utopia message class
    """    

    # Static definitions of the class type constants
    msgID=ord('H')
    msgName="HEARTBEAT"
    
    def __init__(self, timestamp:int=None, statemessage:str=None):
        """the HEARTBEAT utopia message class

        Args:
            timestamp (int, optional): time-stamp for this message. Defaults to None.
            statemessage (str, optional): addition state message string for this heartbeat. Defaults to None.
        """        
        super().__init__(Heartbeat.msgID, Heartbeat.msgName)
        self.timestamp=timestamp
        self.statemessage = statemessage
        if self.statemessage is not None: # v1 heartbeat with state message string
            self.version = 1

    def serialize(self):
        """
        Returns the contents of this event as a string, ready to send over the network, 
        or None in case of conversion problems.

        Returns:
            bytes: the serialized version of this message
        """        
        S = struct.pack('<i', int(self.timestamp))
        if self.statemessage is not None:
            S = S + bytes(self.statemessage, 'utf8')
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a HEARTBEAT class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            Heartbeat: the decoded message
        """        
        bufsize = len(buf)
        if bufsize < 4:
            return (None, 0)
        (timestamp, ) = struct.unpack('<i', buf[0:4])
        if bufsize>4:
            statemessage = buf[4:].decode('utf-8')
            return (Heartbeat(timestamp,statemessage), len(buf))
        else:
            return (Heartbeat(timestamp), 4)
    
    def __str__(self):
        return "%c(%d) %s %i"%(self.msgID, self.msgID, self.msgName, self.timestamp)


class StimulusEvent(UtopiaMessage):
    """the STIMULUSEVENT utopia message class  -- which is used to send information about the current stimulus state of this client to other clients
    """    
    
    # Static definitions of the class type constants
    msgID=ord('E')
    msgName="STIMULUSEVENT"

    def __init__(self, timestamp=None, objIDs=None, objState=None):
        """the STIMULUEVENT utopia message class

        Args:
            timestamp (int, optional): time-stamp for this message. Defaults to None.
            objIDs (list-of-int, optional): the unique object ids for these outputs. Defaults to None.
            objState (list-of-int, optional): the stimulus state of each output object. Defaults to None.
        """        
        super().__init__(StimulusEvent.msgID, StimulusEvent.msgName)
        self.timestamp=timestamp
        self.objIDs=objIDs
        self.objState=objState

    def serialize(self):
        """Converts this message to a string representation to send over the network

        Returns:
            bytes: the encoded message
        """                
        S = struct.pack("<i", int(self.timestamp)) # timestamp
        S = S + struct.pack("<B", len(self.objIDs))  # nObj
        for objid, objstate in zip(self.objIDs, self.objState):
            S = S + struct.pack("<BB", int(objid), int(objstate)) # [objID, objState] pairs
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a STIMULUSEVENT class from a **PAYLOAD** byte-stream, return created object and the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        bufsize = len(buf)
        if bufsize < 5:
            return (None, 0)
        (timestamp, nobj) = struct.unpack("<iB", buf[0:5])
        if bufsize < 5+nobj*2:
            return (None, 0)
        objIDs=[]
        objState=[]
        for i in range(5, len(buf), 2):
            (objid, objstate)=struct.unpack("<BB", buf[i:i+2])
            objIDs.append(objid)
            objState.append(objstate)
        msg=StimulusEvent(timestamp, objIDs, objState)
        return (msg, 5+nobj*2)

    def __str__(self):
        return "%s %i"%(self.msgName, self.timestamp) + "".join("(%i, %i)"%(x, y) for x, y in zip(self.objIDs, self.objState))

class PredictedTargetProb(UtopiaMessage):    
    """the PREDICTEDTARGETPROB utopia message class  -- which is used to communicate the most likely target and the estimate error rate for this prediction.
    """    
    

    # Static definitions of the class type constants
    msgID=ord('P')
    msgName="PREDICTEDTARGETPROB"
    
    def __init__(self, timestamp=None, Yest:int=-1, Perr:float=1.0):
        """the PREDICTEDTARGETPROB utopia message class

        Args:
            timestamp (int, optional): time-stamp for this message. Defaults to None.
            Yest (int, optional): the object ID for the most likely target. Defaults to -1.
            Perr (float, optional): the estimated probility that this predicted target is *wrong*. Defaults to 1.0.
        """        
        super().__init__(PredictedTargetProb.msgID, PredictedTargetProb.msgName)
        self.timestamp=timestamp
        self.Yest     =Yest
        self.Perr     =Perr

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack('<ibf', int(self.timestamp), int(self.Yest), self.Perr)
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a MODECHANGE class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        bufsize = len(buf)
        if bufsize < 4:
            return (None, 0)
        (timestamp, Yest, Perr) = struct.unpack('<ibf', buf[:(4+1+4)])
        msg = PredictedTargetProb(timestamp, Yest, Perr)
        return (msg, 4)
    
    def __str__(self):
        return "%c(%d) %s %i Yest=%d Perr=%f"%(self.msgID, self.msgID, self.msgName, self.timestamp, self.Yest, self.Perr)



class PredictedTargetDist(UtopiaMessage):
    """the PredictedTargetDist utopia message class -- which is used to send information about the current predicted target probabilites for the outputs
    """    
    
    # Static definitions of the class type constants
    msgID=ord('F')
    msgName="PREDICTEDTARGETDIST"

    def __init__(self, timestamp=None, objIDs=None, pTgt=None):
        """the PredictedTargetDist utopia message class

        Args:
            timestamp (int, optional): time-stamp for this message. Defaults to None.
            objIDs (list-of-int, optional): list of the objIDs the target probabilities refer to. Defaults to None.
            pTgt (list-of-float, optional): the target probabilities for the corrospending object IDs. Defaults to None.
        """        
        super().__init__(PredictedTargetDist.msgID, PredictedTargetDist.msgName)
        self.timestamp=timestamp
        self.objIDs=objIDs
        self.pTgt=pTgt

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack("<i", int(self.timestamp)) # timestamp
        S = S + struct.pack("<B", len(self.objIDs))  # nObj
        for objid, pTgt in zip(self.objIDs, self.pTgt):
            S = S + struct.pack("<Bf", int(objid), float(pTgt)) # [objID, pTgt] pairs
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a PREDICTEDTARGETDIST class from a **PAYLOAD** byte-stream, return created object and the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        
        bufsize = len(buf)
        if bufsize < 5:
            return (None, 0)
        (timestamp, nobj) = struct.unpack("<iB", buf[0:5])
        if bufsize < 5+nobj*(1+4):
            return (None, 0)
        objIDs=[]
        pTgt=[]
        for i in range(5, len(buf), (1+4)):
            (objid, ptgt)=struct.unpack("<Bf", buf[i:i+(1+4)])
            objIDs.append(objid)
            pTgt.append(ptgt)
        msg=PredictedTargetDist(timestamp, objIDs, pTgt)
        return (msg, 5+nobj*2)

    def __str__(self):
        return "%s %i"%(self.msgName, self.timestamp) + "".join("(%i, %f)"%(x, y) for x, y in zip(self.objIDs, self.pTgt))


class DataPacket(UtopiaMessage):
    """the DATAPACKET utopia message class  --- which is used to stream raw (time,channels) EEG data packets.
    """    
    
    # Static definitions of the class type constants
    msgID=ord('D')
    msgName="DATAPACKET"

    def __init__(self, timestamp=None, samples=None):
        """the DATAPACKET utopia message class  --- which is used to stream raw (time,channels) EEG data packets.

        Args:
            timestamp (int, optional): time-stamp for this message. Defaults to None.
            samples (list-of-lists-of-float, optional): (samples,channels) array of the raw EEG data. Defaults to None.
        """        
        super().__init__(DataPacket.msgID, DataPacket.msgName)
        self.timestamp=timestamp
        self.samples=samples

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack("<i", int(self.timestamp)) # timestamp
        S = S + struct.pack("<i", len(self.samples))  # nsamp
        for tp in self.samples:
            S = S + struct.pack("<%df"%(len(tp)), *tp)
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a STIMULUSEVENT class from a **PAYLOAD** byte-stream, return created object and the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        
        bufsize = len(buf)
        if bufsize < 8:
            return (None, 0)
        timestamp, nsamp = struct.unpack("<ii", buf[0:8])
        nch=int((bufsize-8)/(nsamp*4))
        samples=[]
        for t in range(nsamp):
            sampt=struct.unpack_from("<%df"%(nch), buf, 8+t*4*nch)
            samples.append(sampt)
        msg=DataPacket(timestamp, samples)
        return (msg, 8+nsamp*nch*8)

    def __str__(self):
        ss="%c(%d) %s %i "%(self.msgID, self.msgID, self.msgName, self.timestamp)
        ss= ss+ "[%dx%d]"%(len(self.samples), len(self.samples[0]))
        for chs in self.samples:
            chstr = "".join(["%f, "%(c) for c in chs])
            ss = ss + "["+chstr+"]"
        return  ss


class DataHeader(UtopiaMessage):
    """
    the DATAHEADER utopia message class  -- which is used to give general meta-information about the EEG stream
    """    
    
    # Static definitions of the class type constants
    msgID=ord('A')
    msgName="DATAHEADER"

    def __init__(self, timestamp=None, fsample=None, nchannels=None, labels=None):
        """the DATAHEADER utopia message class

        Args:
            timestamp (int, optional): the message timestamp in milliseconds. Defaults to None.
            fsample (float, optional): the nomional sampling rate of the EEG stream. Defaults to None.
            nchannels (int, optional): the number of channels in the data stream. Defaults to None.
            labels (list-of-str, optional): the textual names of the data channels. Defaults to None.
        """        
        super().__init__(DataHeader.msgID, DataHeader.msgName)
        self.timestamp=timestamp
        self.fsample=fsample
        self.nchannels=nchannels
        self.labels=labels

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack("<i", int(self.timestamp)) # timestamp
        S = S + struct.pack("<i", self.nchannels)
        S = S + struct.pack("<f", self.fsample)
        if self.labels: 
            # comma separated list of channel names
            S = S + bytes(", ".join(self.labels), 'utf-8')
        return S

    @staticmethod
    def deserialize(buf):
        """ 
        Static method to create a HEADER class from a **PAYLOAD** byte-stream, return created object and the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        

        bufsize = len(buf)
        if bufsize < 12:
            return (None, 0)
        timestamp, nchannels, fsample = struct.unpack("<iif", buf[0:8])
        labels = buf[4:].decode('utf-8')
        labels = labels.split(", ")
        msg=DataHeader(timestamp, nchannels, fsample, labels)
        return (msg, bufsize)

    def __str__(self):
        ss="%c(%d) %s %i "%(self.msgID, self.msgID, self.msgName, self.timestamp)
        ss= ss+ "%dch @ %gHz"%(self.nchannels, self.fsample)
        ss= ss+ ", ".join(self.labels)
        return  ss

class NewTarget(UtopiaMessage):  
    """ 
    the NEWTARGET utopia message class -- which is used to tell other clients that a new target selection round has begun.
    """      

    # Static definitions of the class type constants
    msgID=ord('N')
    msgName="NEWTARGET"
    
    def __init__(self, timestamp=None):
        """the NEWTARGET utopia message class

        Args:
            timestamp (int, optional): message timestamp in milliseconds. Defaults to None.
        """        

        super().__init__(NewTarget.msgID, NewTarget.msgName)
        self.timestamp=timestamp

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack('<i', int(self.timestamp))
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a HEARTBEAT class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        

        bufsize = len(buf)
        if bufsize < 4:
            return (None, 0)
        (timestamp, ) = struct.unpack('<i', buf[0:4])
        msg = NewTarget(timestamp)
        return (msg, 4)
    
    def __str__(self):
        return "%c(%d) %s %i"%(self.msgID, self.msgID, self.msgName, self.timestamp)

class Selection(UtopiaMessage):    
    """the SELECTION utopia message class  -- which is used to inform other clients that a particular output has been selected by **this** client.
    """    

    # Static definitions of the class type constants
    msgID=ord('S')
    msgName="SELECTION"
    
    def __init__(self, timestamp=None, objID=None):
        """ the SELECTION utopia message class

        Args:
            timestamp (int, optional): message timestamp in milliseconds. Defaults to None.
            objID (int, optional): the objID of the selected output. Defaults to None.
        """        
        super().__init__(Selection.msgID, Selection.msgName)
        self.timestamp=timestamp
        self.objID    =objID

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack('<i', int(self.timestamp))
        S = S + struct.pack('<B', self.objID)
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a SELECTION class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        bufsize = len(buf)
        if bufsize < 5:
            return (None, 0)
        (timestamp, objID) = struct.unpack('<iB', buf[0:5])
        msg = Selection(timestamp, objID)
        return (msg, 5)
    
    def __str__(self):
        return "%c(%d) %s %i id:%i"%(self.msgID, self.msgID, self.msgName, self.timestamp, self.objID)
    
class Reset(UtopiaMessage):    
    """the RESET utopia message class  -- which is used to tell the decoder to reset and clear it's model information.
    """    

    # Static definitions of the class type constants
    msgID=ord('R')
    msgName="RESET"
    
    def __init__(self, timestamp=None):
        """the RESET utopia message class 

        Args:
            timestamp (int, optional): message timestamp in milliseconds. Defaults to None.
        """        
        super().__init__(Reset.msgID, Reset.msgName)
        self.timestamp=timestamp

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack('<i', int(self.timestamp))
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a RESET class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        bufsize = len(buf)
        if bufsize < 4:
            return (None, 0)
        (timestamp, ) = struct.unpack('<i', buf[0:4])
        msg = Reset(timestamp)
        return (msg, 4)
    
    def __str__(self):
        return "%c(%d) %s %i"%(self.msgID, self.msgID, self.msgName, self.timestamp)

    

class ModeChange(UtopiaMessage):    
    """the MODECHANGE utopia message class -- which is used to tell the decoder to change system modes, for example to switch to calibration-mode.
    """    

    # Static definitions of the class type constants
    msgID=ord('M')
    msgName="MODECHANGE"
    
    def __init__(self, timestamp=None, newmode=None):
        """the MODECHANGE utopia message class

        Args:
            timestamp (int, optional): message timestamp in milliseconds. Defaults to None.
            newmode (str, optional): the desired new system mode. Defaults to None.
        """        
        super().__init__(ModeChange.msgID, ModeChange.msgName)
        self.timestamp=timestamp
        self.newmode  =newmode

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack('<i', int(self.timestamp))
        S = S + bytes(self.newmode, 'utf-8')
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a MODECHANGE class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        bufsize = len(buf)
        if bufsize < 4:
            return (None, 0)
        (timestamp, ) = struct.unpack('<i', buf[0:4])
        newmode = buf[4:].decode('utf-8')
        msg = ModeChange(timestamp, newmode)
        return (msg, 4)
    
    def __str__(self):
        return "%c(%d) %s %i %s"%(self.msgID, self.msgID, self.msgName, self.timestamp, self.newmode)

class Log(UtopiaMessage):    
    """the LOG utopia message class  -- which is used to send arbitary log messages to the hub
    """    

    # Static definitions of the class type constants
    msgID=ord('L')
    msgName="LOG"
    
    def __init__(self, timestamp:int=None, logmsg:str=None):
        """the LOG utopia message class

        Args:
            timestamp (int, optional): message timestamp in milliseconds. Defaults to None.
            logmsg (str, optional): the log message to send. Defaults to None.
        """        
        super().__init__(Log.msgID, Log.msgName)
        self.timestamp=timestamp
        self.logmsg  =logmsg

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack('<i', int(self.timestamp))
        S = S + bytes(self.logmsg, 'utf-8')
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a MODECHANGE class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        bufsize = len(buf)
        if bufsize < 4:
            return (None, 0)
        (timestamp, ) = struct.unpack('<i', buf[0:4])
        logmsg = buf[4:].decode('utf-8')
        msg = ModeChange(timestamp, logmsg)
        return (msg, 4)
    
    def __str__(self):
        return "%c(%d) %s %i %s"%(self.msgID, self.msgID, self.msgName, self.timestamp, self.logmsg)


class SignalQuality(UtopiaMessage):    
    """the SIGNALQUALITY utopia message class -- which is used to send information about the estimated signal to noise for each channel in the data stream
    """    

    # Static definitions of the class type constants
    msgID=ord('Q')
    msgName="SIGNALQUALITY"
    
    def __init__(self, timestamp:int=None, signalQuality=None):
        """the SIGNALQUALITY utopia message class

        Args:
            timestamp (int, optional): [description]. Defaults to None.
            signalQuality (list-of-float, optional): signal-quality estimate (noise-to-signal ratio) for each of the EEG channels. Defaults to None.
        """        
        super().__init__(SignalQuality.msgID, SignalQuality.msgName)
        self.timestamp=timestamp
        self.signalQuality=signalQuality

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack('<i', int(self.timestamp))
        S = S + b''.join([ struct.pack('<f', float(q)) for q in self.signalQuality ])
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a SIGNALQUALIYT class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        bufsize = len(buf)
        if bufsize < 4:
            return (None, 0)
        (timestamp, ) = struct.unpack('<i', buf[0:4])
        signalQuality = struct.unpack_from('<%df'%((bufsize-4)//4), buf, 4)
        #                  , buf[i:i+4])[0] for i in range(4, bufsize, 4) ]
        msg = SignalQuality(timestamp, signalQuality)
        return (msg, bufsize)
    
    def __str__(self):
        return "%c(%d) %s %i [%s]"%(self.msgID, self.msgID, self.msgName, self.timestamp, ", ".join(["%f"%q for q in self.signalQuality]))

class Subscribe(UtopiaMessage):    
    """the SUBSCRIBE utopia message class -- which is used to tell the hub which messages IDs to forward to this client.
    """    

    # Static definitions of the class type constants
    msgID=ord('B')
    msgName="SUBSCRIBE"
    
    def __init__(self, timestamp=None, messageIDs=None):
        """the SUBSCRIBE utopia message class 

        Args:
            timestamp ([type], optional): [description]. Defaults to None.
            messageIDs ([type], optional): [description]. Defaults to None.
        """        
        super().__init__(Subscribe.msgID, Subscribe.msgName)
        self.timestamp=timestamp
        self.messageIDs=messageIDs

    def serialize(self):
        """Returns the contents of this event as a byte-stream, ready to send over the network, 
           or None in case of conversion problems.

        Returns:
            bytes: the encoded message
        """
        S = struct.pack('<i', int(self.timestamp))
        if type(self.messageIDs) is str:#str
            S = S + bytes(self.messageIDs, 'utf-8')
        else:# int->byte
            S = S + b''.join([ struct.pack('<b', byte(q)) for q in self.messageIDs ])
        return S

    @staticmethod
    def deserialize(buf):
        """Static method to create a SIGNALQUALITY class from a **PAYLOAD** byte-stream, return the number of bytes consumed from buf

        Args:
            buf (bytes): the message buffer

        Returns:
            UtopiaMessage: the decoded message
        """        
        bufsize = len(buf)
        if bufsize < 4:
            return (None, 0)
        (timestamp, ) = struct.unpack('<i', buf[0:4])
        #messageIDs = [ struct.unpack('<i', buf[i:i+4]) for i in range(4, bufsize, 4) ]
        messageIDs = struct.unpack_from('<%di'%((bufsize-4)//4), buf, 4)
        msg = Subscribe(timestamp, messageIDs)
        return (msg, bufsize)
    
    def __str__(self):
        return "%c(%d) %s %i [%s]"%(self.msgID, self.msgID, self.msgName, self.timestamp, "{}".format(self.messageIDs))

    
# Helper functions for dealing with raw-messages -> classes    
def decodeRawMessage(msg):
    """decode utopia RawMessages into actual message objects

    Args:
        msg (RawMessage): the raw message whose payload should be decoded

    Returns:
        StimulusEvent|Heartbeat|....: the decoded specific message type
    """    
    if msg.msgID==StimulusEvent.msgID:
        decodedmsg, _= StimulusEvent.deserialize(msg.payload)
    elif msg.msgID==Heartbeat.msgID:
        decodedmsg, _ = Heartbeat.deserialize(msg.payload)
    elif msg.msgID==PredictedTargetProb.msgID:
        decodedmsg, _ = PredictedTargetProb.deserialize(msg.payload)
    elif msg.msgID==PredictedTargetDist.msgID:
        decodedmsg, _ = PredictedTargetDist.deserialize(msg.payload)
    elif msg.msgID==Selection.msgID:
        decodedmsg, _ = Selection.deserialize(msg.payload)
    elif msg.msgID==ModeChange.msgID:
        decodedmsg, _ = ModeChange.deserialize(msg.payload)
    elif msg.msgID==NewTarget.msgID:
        decodedmsg, _ = NewTarget.deserialize(msg.payload)
    elif msg.msgID==SignalQuality.msgID:
        decodedmsg, _ = SignalQuality.deserialize(msg.payload)
    elif msg.msgID==Reset.msgID:
        decodedmsg, _ = Reset.deserialize(msg.payload)
    elif msg.msgID==DataPacket.msgID:
        decodedmsg, _ = DataPacket.deserialize(msg.payload)
    elif msg.msgID==Subscribe.msgID:
        decodedmsg, _ = Subscribe.deserialize(msg.payload)
    else:
        decodedmsg = msg
    return decodedmsg

def decodeRawMessages(msgs):
    """decode utopia RawMessages into actual message objects

    Args:
        msgs (list-of-RawMessage): list of RawMessages to decode the payload of

    Returns:
        list-of-messages: list of the fully decoded messages
    """    
    return [ decodeRawMessage(msg) for msg in msgs ]

def ssdpDiscover(servicetype=None, timeout=3, numretries=1):
    """auto-discover the utopia-hub using ssdp discover messages

    Args:
        servicetype (str, optional): the SSDP service type to search for. Defaults to None.
        timeout (int, optional): timeout in seconds for the discovery. Defaults to 3.
        numretries (int, optional): number of times to retry discovery. Defaults to 1.

    Returns:
        list-of-str: list of the IP addresses of the discovered servers
    """    
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
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        sock.settimeout(timeout)
    except: # abort if cant setup socket
        return () 
    responses=[]
    for _ in range(numretries):
        print("Sending query message:\n%s"%(msearchMessage))
        try:
            sock.sendto(msearchMessage, ssdpgroup)
        except: # abort if send fails
            return ()
        # wait for responses and store the location info for the matching ones
        print("Waiting responses")
        try:
            rsp, addr=sock.recvfrom(8192)
            rsp=rsp.decode('utf-8')
            print("Got response from: %s\n%s\n"%(addr, rsp))
            # does the response contain servertype, if so then we match
            location=None
            if servicetype is None or servicetype in rsp:
                print("Response matches server type: %s"%(servicetype))
                # use the message source address as default location
                location=addr 
                # extract the location or IP from the message
                for line in rsp.split("\r\n"): # cut into lines
                    tmp=line.split(":", 1) # cut line into key/val
                    # is this the key we care about -> then record the value
                    if len(tmp)>1 and tmp[0].lower()=="LOCATION".lower():
                        location=tmp[1].strip()
                        # strip http:// xxxx /stuff
                        if location.startswith("http://"):
                            location=location[7:] # strip http
                        if '/' in location:
                            location=location[:location.index('/')] # strip desc.xml
                        print("Got location: %s"%(location))
                        break # done with this response
                # add to the list of possible servers
                print("Loc added to response list: %s"%(location))
                responses.append(location)
        except socket.timeout:
            print("Socket timeout")
        if len(responses)>0: break
    return responses

class UtopiaClient:
    """Class for managing a client connection to a UtopiaServer.
    """    

    UTOPIA_SSDP_SERVICE="utopia/1.1"
    DEFAULTHOST='localhost'
    DEFAULTPORT=8400    
    HEARTBEATINTERVAL_ms=1000
    HEARTBEATINTERVALUDP_ms=200
    MAXMESSAGESIZE = 1024 * 1024

    def __init__(self, clientstate=None):
        """Class for managing a client connection to a UtopiaServer

        Args:
            clientstate ([type], optional): [description]. Defaults to None.
        """   
        self.isConnected = False
        self.sock = []
        self.udpsock=None
        self.recvbuf = b''
        self.tsClock = TimeStampClock()
        self.sendHeartbeats = True
        self.nextHeartbeatTime = self.getTimeStamp()
        self.nextHeartbeatTimeUDP = self.getTimeStamp()
        self.ssdpDiscover = None
        self.clientstate = clientstate

    # time-stamp management
    def setTimeStampClock(self, tsClock):
        """set the clock to use for timestamping outgoing messages

        Args:
            tsClock (TimeStampClock): the timestamp clock to use

        Raises:
            ValueError: if the tsClock does not have a getTimeStamp method
        """    
        if not hasattr(tsClock,'getTimeStamp'):
            raise ValueError("Time Stamp clock must have getTimeStamp method")
        self.tsClock = tsClock

    def getTimeStamp(self):
        """Get the time-stamp for the current time

        Returns:
            int: the current timestamp
        """    
        return self.tsClock.getTimeStamp()

    def disableHeartbeats(self):
        '''' stop sending hearbeat messages. Use, e.g. when you want to use your own time-stamp clock. '''
        self.sendHeartbeats = False

    def enableHeartbeats(self):
        """ start sending heartbeat messages every few seconds
        """ 
        self.sendHeartbeats = True

    def connect(self, hostname=None, port=8400):
        """connect([hostname, port]) -- make a connection, default host:port is localhost:1972

        Args:
            hostname (str, optional): hostname or IP where the hub resides. Defaults to None.
            port (int, optional): port where the host is listening. Defaults to 8400.

        Returns:
            bool: the current connection status
        """        
        if hostname is None:       hostname = UtopiaClient.DEFAULTHOST
        if port is None or port<0: port     = UtopiaClient.DEFAULTPORT
        print("Trying to connect to: %s:%d"%(hostname, port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((hostname, port))
        self.sock.setblocking(False)
        self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1) # disable nagle
        self.isConnected = True
        # make udp socket also
        self.udpsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        # ensure tcp and udp have the same local port number...
        self.udpsock.bind(self.sock.getsockname())
        return self.isConnected

    def autoconnect(self, hostname=None, port=None, timeout_ms=5000, 
                queryifhostnotfound=False, localhostifhostnotfound=True, scanifhostnotfound=False):
        """connect to the hub/decoder, auto-discover if explict IP address is not given

        Args:
            hostname (str, optional): hostname or IP where the hub resides. Defaults to None.
            port (int, optional): port where the host is listening. Defaults to 8400.
            timeout_ms (int, optional): timeout in milliseconds for discovery. Defaults to 5000.
            queryifhostnotfound (bool, optional): query the user for autodiscover fails. Defaults to False.
            localhostifhostnotfound (bool, optional): try localhost if autodiscovery fails. Defaults to True.
            scanifhostnotfound (bool, optional): scan all local IP addresses if autodiscovery fails. Defaults to False.

        Raises:
            socket.error: if there is a socket error ;)
        """  
        if port is None: port = UtopiaClient.DEFAULTPORT
        if hostname == '-' : hostname = None
        if hostname is None:
            print('Trying to auto-discover the utopia-hub server')
            if True:
                if self.ssdpDiscover is None:
                    print('making discovery object')
                    from mindaffectBCI.ssdpDiscover import ssdpDiscover
                    # create the discovery object
                    self.ssdpDiscover=ssdpDiscover(UtopiaClient.UTOPIA_SSDP_SERVICE)
                hosts=self.ssdpDiscover.discover(timeout=timeout_ms/1000.0)
            else:
                hosts=ssdpDiscover(servicetype=UtopiaClient.UTOPIA_SSDP_SERVICE, timeout=5, numretries=int(max(1, timeout_ms/5000)))
            print("Discovery returned %d utopia-hub servers"%len(hosts))
            if( len(hosts)>0 ):
                hostname=hosts[0].strip()
                print('Discovered utopia-hub on %s ...'%(hostname))
            if hostname is None:
                print('Error:: couldnt autodiscover the decoder!')


        if hostname is not None:
            self.try_connect(hostname,port,timeout_ms)

        # Try different ways of getting the host info: localhost, query, scan
        if not self.isConnected and localhostifhostnotfound : 
            self.try_connect('localhost',port,timeout_ms)
        
        if not self.isConnected and queryifhostnotfound:
            # ask user for host
            print("Could not auto-connect.  Trying manual")
            hostname = input("Enter the hostname/IP of the Utopia-HUB: ")
            self.try_connect(hostname,port,timeout_ms)

        if not self.isConnected and scanifhostnotfound:
            print("Could not auto-discover.  Trying IP scan.")
            from mindaffectBCI.ssdpDiscover import ipscanDiscover
            hosts = ipscanDiscover(port)
            if  len(hosts)>0:
                hostname=hosts[0].strip()
                print('Discovered utopia-hub on %s ...'%(hostname))
                self.try_connect(hostname,port,timeout_ms)
            
        if not self.isConnected:            
            raise socket.error('Connection Refused!')

    def try_connect(self, hostname, port=None, timeout_ms=5000):
        """try to connect to the hub

        Args:
            hostname (str, optional): hostname or IP where the hub resides. Defaults to None.
            port (int, optional): port where the host is listening. Defaults to 8400.
            timeout_ms (int, optional): timeout in milliseconds for discovery. Defaults to 5000.
        """     
        if ":" in hostname:
            hostname, port=hostname.split(":")
            port=int(port)

        for i in range(max(1, int(timeout_ms/1000))):
            try:
                print("Tring to connect to: %s:%d"%(hostname, port))
                self.connect(hostname, port)
                print('Connected!', flush=True)
                break
            except socket.error as ex:
                print('Connection refused...  Waiting', flush=True)
                print(ex)
                time.sleep(1)


    def gethostport(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
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
        """Send all bytes of the string 'request' out to the connected hub/decoder

        Args:
            request (bytes): the raw stream to send

        Raises:
            IOError: if the send fails
        """    
        if not(self.isConnected):
            raise IOError('Not connected to utopia server')

        N = len(request)
        nw = self.sock.send(request)
        while nw<N:
            nw += self.sock.send(request[nw:])

    def sendRawUDP(self, request):
        """send a raw byte stream over a UDP socket

        Args:
            request (bytes): the raw stream to send
        """      
        self.udpsock.sendto(request, self.sock.getsockname())

    def sendMessage(self, msg:UtopiaMessage):
        """send a UtopiaMessage to the hub

        Args:
            msg (UtopiaMessage): the message to send
        """    

        if not msg is RawMessage: # convert to raw for sending
            if msg.timestamp is None: # insert valid time-stamp, N.B. all messages have timestamp!
                msg.timestamp = self.getTimeStamp()
            msg = RawMessage.fromUtopiaMessage(msg)
        self.sendRaw(msg.serialize()) # send the raw message directly
        self.sendHeartbeatIfTimeout()

    def sendHeartbeatIfTimeout(self, timestamp=None):
        """send a heartbeat message if the timeout since the last such message has passed.

        Args:
            timestamp (int, optional): The current timestamp. Defaults to None.
        """    
        if not self.sendHeartbeats:
            return
        if timestamp is None:
            timestamp = self.getTimeStamp()
        if timestamp > self.nextHeartbeatTime:
            self.nextHeartbeatTime=timestamp+self.HEARTBEATINTERVAL_ms
            self.sendRaw(RawMessage.fromUtopiaMessage(Heartbeat(timestamp,self.clientstate)).serialize())
        if timestamp > self.nextHeartbeatTimeUDP:
            self.nextHeartbeatTimeUDP=timestamp+self.HEARTBEATINTERVALUDP_ms
            self.sendRawUDP(RawMessage.fromUtopiaMessage(Heartbeat(timestamp,self.clientstate)).serialize())
        
    def sendMessages(self, msgs):
        """sends single or multiple utopia-messages to the utopia server

        Args:
            msgs (list-of-UtopiaMessage): list of messages to send
        """    
        for msg in msgs:
            self.sendMessage(msg)
            

    def recvall(self, timeout_ms=0):
        """Read all the data from the socket immeaditely or block for timeout_ms if nothing to do.

        Args:
            timeout_ms (int, optional): timeout in milliseconds for the read. Defaults to 0.

        Returns:
            bytes: the raw stream of bytes from the socket
        """
        if timeout_ms>0:
            self.sock.setblocking(1)
            self.sock.settimeout(timeout_ms/1000.0)
        else:
            self.sock.setblocking(0)
        data=[]
        try:
            data = self.sock.recv(self.MAXMESSAGESIZE)
        except socket.timeout:
            pass
        except socket.error as ex:
            if not ( ex.errno == 11 or ex.errno == 10035 or ex.errno==35 ): # 11 is raised when no-data to read
                print("Socket error" + str(ex) + "#" + str(ex.errno) ) 
        return bytes(data)
    
    def getNewMessages(self, timeout_ms=250):
        """Wait for new messages from the utopia-server (with optional timeout), decode them and return the list of new messages

        Args:
            timeout_ms (int, optional): timeout in milliseconds to wait for the new messages. Defaults to 250.

        Returns:
            list-of-UtopiaMessage: the list of recieved messages
        """   
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
        our local clock

        Args:
            delays_ms (list-of-int, optional): delay in milliseconds between sending the heartbeat messages. Defaults to [50]*10.
        """   
        if not self.sendHeartbeats:
            print("Warning: not sending heartbeats as they are disabled!")
            return
        self.sendMessage(Heartbeat(self.getTimeStamp(),self.clientstate))
        for delay in delays_ms:
            time.sleep(delay/1000)
            self.sendMessage(Heartbeat(self.getTimeStamp(),self.clientstate))
    
    def messagelogger(self, timeout_ms=1000):
        """Simple message logger, infinite loop waiting for and printing messages from the server

        Args:
            timeout_ms (int, optional): delay when waiting for messages. Defaults to 1000.
        """    
        client.sendMessage(Subscribe(None, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")) # subcribe to everything...
        while True:
            newmessages = client.getNewMessages(timeout_ms)
            print("%d new messages:"%len(newmessages))
            print("\n".join(str(msg) for msg in newmessages))


    def messagePingPong(self, timeout_ms=500):
        """Testing system sending 20 messages and printing any responses

        Args:
            timeout_ms (int, optional): delay when waiting for messages. Defaults to 1000.
        """     
        # loop waiting for messages and printing them
        for i in range(20):
            self.sendMessage(StimulusEvent(i, [0, 1, 2, 3], [i+1, i+2, i+3, i+4]))
            time.sleep(1)
            newmessages = self.getNewMessages(timeout_ms)
            print("%d) %d NewMessages\n"%(i, len(newmessages)))
            print("\n".join(str(msg) for msg in newmessages))

            
def testSerialization():
    """test object serialization and deserialization by encoded and decoding messages
    """    

    rm=RawMessage(1, 0, "payload")
    print("RawMessage: %s"%(rm))
    print("serialized: %s"%(rm.serialize()))
    print("deserialized: %s"%(RawMessage.deserialize(rm.serialize())[0]))
    print()

    hb=Heartbeat(10)
    print("Heartbeat: %s"%(hb))
    print("serialized: %s"%(hb.serialize()))
    print("deserialized: %s"%(Heartbeat.deserialize(hb.serialize())[0]))
    print()
    
    se=StimulusEvent(11, [0, 1, 2], [0, 1, 0])
    print("StimulusEvent: %s"%(se))
    print("serialized: %s"%(se.serialize()))
    print("deserialized: %s"%(StimulusEvent.deserialize(se.serialize())[0]))
    print()

    mc=ModeChange(12, "hello")
    print("ModeChange: %s"%(mc))
    print("serialized: %s"%(mc.serialize()))
    print("deserialized: %s"%(ModeChange.deserialize(mc.serialize())[0]))
    print()

    nt=NewTarget(13)
    print("NewTarget: %s"%(nt))
    print("serialized: %s"%(nt.serialize()))
    print("deserialized: %s"%(NewTarget.deserialize(nt.serialize())[0]))
    print()
  
    rs=Reset(14)
    print("Reset: %s"%(rs))
    print("serialized: %s"%(rs.serialize()))
    print("deserialized: %s"%(Reset.deserialize(rs.serialize())[0]))
    print()

    pt=PredictedTargetProb(15, 10, .5)
    print("PredictedTargetProb: %s"%(pt))
    print("serialized         : %s"%(pt.serialize()))
    print("deserialized: %s"%(PredictedTargetProb.deserialize(pt.serialize())[0]))
    print()

    pd=PredictedTargetDist(15, [1, 2, 3], [.5, .3, .2])
    print("PredictedTargetDist: %s"%(pd))
    print("serialized         : %s"%(pd.serialize()))
    print("deserialized: %s"%(PredictedTargetDist.deserialize(pd.serialize())[0]))
    print()
    
    sq=SignalQuality(16, [.1, .2, .3, .4, .5])
    print("SignalQuality: %s"%(sq))
    print("serialized   : %s"%(sq.serialize()))
    print("deserialized: %s"%(SignalQuality.deserialize(sq.serialize())[0]))
    print()
    
    dp=DataPacket(16, [[.1, .2, .3], [.4, .5, .6]])
    print("DataPacket : %s"%(dp))
    print("serialized : %s"%(dp.serialize()))
    print("deserialized: %s"%(DataPacket.deserialize(dp.serialize())[0]))
    print()
    

    rhb=RawMessage.fromUtopiaMessage(hb)
    print("Raw(Heartbeat): %s"%(rhb))

    srhb=rhb.serialize()
    print("serialized   : %s"%(srhb))
    dsrhb, =RawMessage.deserialize(srhb)
    print("Deserialized serialized Raw(Heartbeat): %s"%(dsrhb))
    sedsrhb=decodeRawMessage(dsrhb)
    print("Decoded deserialized Raw(Heartbeat): %s"%(sedsrhb))

    rse=RawMessage.fromUtopiaMessage(se)
    print("Raw(StimulusEvent): %s"%(rse))
    srse=rse.serialize()
    print("serialized       : %s"%(srse))    
    #deserialize
    (dsrse, nconsumed)=RawMessage.deserialize(srse)
    print("Deserialized serialized Raw(StimulusEvent): %s"%(dsrse))
    sedsrse=decodeRawMessage(dsrse)
    print("Decoded deserialized Raw(StimulusEvent): %s"%(sedsrse))

    #deserialize multiple
    (msgs, nconsumed)=RawMessage.deserializeMany(srse+srse+srhb)
    print("Deserialized Multiple: %s"%("\n".join([str(msg) for msg in msgs])))
    msgs=decodeRawMessages(msgs)
    print("Decoded Multiple: %s"%("\n".join([str(msg) for msg in msgs])))



def testSending():
    """test object sending by connecting and sending a set of test messages to the server
    """    

    client = UtopiaClient()        
    client.autoconnect()

    # DataPacket
    client.sendMessage(DataPacket(client.getTimeStamp(), [[.1, .2, .3]]))

    # StimulusEvent
    time.sleep(5)

    
if __name__ == "__main__":

    #testCases()

    #hosts=ssdpDiscover(servicetype="ssdp:all", timeout=3)
    #print(hosts)

    # Just a small logging demo for testing purposes...
    hostname = None
    port = None
    
    if len(sys.argv)>1:
        hostname = sys.argv[1]
        tmp=[hostname.split(":")]
        if len(tmp)>1:
            hostname=tmp[0]
            port    =int(tmp[1])

    client = UtopiaClient()        
    client.autoconnect(hostname, port)
    client.messagelogger(30000)
    client.disconnect()
