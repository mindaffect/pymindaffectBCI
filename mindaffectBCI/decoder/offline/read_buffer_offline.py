import struct
import numpy as np
import os
import array

typedict= {0:('char',1,'c'),
           1:('uint8',1,'B'),
           2:('uint16',2,'H'),
           3:('uint32',4,'I'),
           4:('unint64',8,'Q'),
           5:('int8',1,'b'),
           6:('int16',2,'h'),
           7:('int32',4,'i'),
           8:('int64',8,'l'),
           9:('float32',4,'f'),
           10:('float64',8,'d')}

class ftevent:
    def __init__(self,s,t,v,o=None,d=None):
        self.sample=s
        self.type=t
        self.value=v
        self.offset=o
        self.duration=d
    def __str__(self):
        return "{{ s:{} t:{} v:{} }}".format(self.sample,self.type,self.value)

class ftheader:
    def __init__(self,nch,nsamp,nevt,fs,data_type,labels=None):
        self.nch=nch
        self.nsamp=nsamp
        self.nevt=nevt
        self.fs=fs
        self.data_type=typedict[data_type] if isinstance(data_type,str) else data_type
        self.labels=labels

    
def read_buffer_offline_events(f):
    if isinstance(f,str):
        if os.path.isdir(f):
            f = os.path.join(f,'events')
        buf = open(f,'rb').read()
    else:
        buf = f

    devents=[]
    cursor=0
    while cursor<len(buf):
        t_type,t_len,v_type,v_len,sample,offset,duration,payloadsize = struct.unpack_from("IIIIiiiI",buf,cursor)
        cursor+=struct.calcsize("IIIIiiiI")
        payload = buf[cursor:cursor+payloadsize]
        cursor+=payloadsize

        # decode the payload
        tfmtstr = "{}{}".format(t_len,typedict[t_type][2])
        vfmtstr = "{}{}".format(v_len,typedict[v_type][2])
        if struct.calcsize(tfmtstr)+struct.calcsize(vfmtstr) > len(payload):
            print("Error: didn't understand event info, skipped")
            continue

        
        if typedict[t_type][2] == 'c':
            tval = payload[:struct.calcsize(tfmtstr)].decode('utf8')
        else:
            tval=struct.unpack_from(tfmtstr,payload,0)
        if typedict[v_type][2] == 'c':
            vval = payload[struct.calcsize(tfmtstr):].decode('utf8')
        else:
            vval=struct.unpack_from(vfmtstr,payload,struct.calcsize(tfmtstr))
        # make the event
        devents.append(ftevent(sample,tval,vval,offset,duration))
    return devents

def read_buffer_offline_header(f):
    if isinstance(f,str):
        if os.path.isdir(f):
            f = os.path.join(f,'header')
        buf = open(f,'rb').read()
    else:
        buf = f
    nch,nsamp,nevts,fs,data_type,bufsize=struct.unpack_from("IIIfII",buf,0)
    # TODO []: extract the channel labels
    labbuf = buf[struct.calcsize("IIIfII"):]
    # TODO []: read the channel names...
    return ftheader(nch,nsamp,nevts,fs,data_type,None)

def read_buffer_offline_data(f,hdr):
    fmtstr = hdr.data_type if isinstance(hdr.data_type,str) else typedict[hdr.data_type][2]
    if isinstance(f,str):
        if os.path.isdir(f):
            f = os.path.join(f,'samples')
        dat = np.fromfile(f,dtype=typedict[hdr.data_type][0],count=-1)
    else:
        dat = dat.frombuffer(buf,dtype=typedict[hdr.data_type][0],count=-1)
    dat = np.reshape(dat,(len(dat)//hdr.nch,hdr.nch))
    return dat

def testcase():
    from read_buffer_offline import read_buffer_offline_data, read_buffer_offline_events, read_buffer_offline_header
    datadir="../../resources/example_data/utopia_v1/s1/raw_buffer/0001"
    hdr = read_buffer_offline_header(datadir)
    print("{}".format(hdr))
    evts= read_buffer_offline_events(datadir)
    print("{} events".format(len(evts)))
    dat = read_buffer_offline_data(datadir,hdr)
    print("dat={}".format(dat.shape))    


if __name__=='__main__':
    testcase()
