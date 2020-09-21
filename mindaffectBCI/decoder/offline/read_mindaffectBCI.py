import os
import numpy as np
import re
from mindaffectBCI.utopiaclient import StimulusEvent, DataPacket, ModeChange, NewTarget, Selection

# named reg-exp to parse the different messages types log lines
recievedts_re = re.compile(r'\Wrts:(?P<sts>[-0-9]*)\W')
serverts_re = re.compile(r'.*sts:(?P<sts>[-0-9]*)\W')
clientts_re = re.compile(r'^.*\Wts:(?P<ts>[-0-9]*)\W')
clientip_re = re.compile(r'.*<-\W(?P<ip>[0-9.:]*)$')
stimevent_re = re.compile(r'^.*\Wts:(?P<ts>[-0-9]*)\W*v\[(?P<shape>[0-9x]*)\]:(?P<stimstate>.*) <-.*$')
datapacket_re = re.compile(r'^.*\Wts:(?P<ts>[-0-9]*)\W*v\[(?P<shape>[0-9x]*)\]:(?P<samples>.*) <-.*$')
modechange_re = re.compile(r'^.*\Wts:(?P<ts>[-0-9]*)\W.*mode:(?P<newmode>.*) <-.*$')

def read_StimulusEvent(line:str):
    ''' read a stimulus event message from a text-file save version '''
    # named reg-ex to extract the bits we need
    res = stimevent_re.match(line)
    if res is None:
        return None
    ts = int(res['ts'])
    # parse sample into numpy array
    shape = np.fromstring(res['shape'].replace('x',','),sep=',',dtype=int)
    shape = shape[::-1] # N.B. python order, fastest last..
    stiminfo = res['stimstate'].replace('{','').replace('}',',')
    stiminfo = np.fromstring(stiminfo, sep=',', dtype=int)
    objIDs = stiminfo[0::2]
    stimstate = stiminfo[1::2]
    #print("SE ts:{} objIDs:{} state:{}".format(ts,objIDs,stimstate))
    return StimulusEvent(ts,objIDs,stimstate)
    
def read_DataPacket(line:str ):
    # named reg-ex to extract the bits we need
    res = datapacket_re.match(line)
    if res is None:
        return None
    ts = int(res['ts'])
    # parse sample into numpy array
    shape = np.fromstring(res['shape'].replace('x',','),sep=',',dtype=int)
    shape = shape[::-1] # N.B. python order, fastest last..
    samples = np.fromstring(res['samples'].replace(']','').replace('[',''),sep=',',dtype=np.float32)
    samples = samples.reshape(shape)
    return DataPacket(ts,samples)
    
def read_ModeChange(line:str):
    res = modechange_re.match(line)
    if res is None:
        return None
    ts = int(res['ts'])
    newmode = res['newmode']
    return ModeChange(ts,newmode)

def read_NewTarget(line:str):
    ts = read_clientts(line)
    return NewTarget(ts)

def read_Selection(line:str):
    # TODO[]: read the actual selection info
    ts = read_clientts(line)
    return Selection(ts,-1)

def read_clientts(line:str):
    ts = clientts_re.match(line)
    ts = int(ts['ts']) if ts is not None else None
    return ts

def read_serverts(line:str):
    sts = serverts_re.match(line)
    sts = int(sts['sts']) if sts is not None else None
    return sts

def read_clientip(line:str):
    ip = clientip_re.match(line)
    ip = ip['ip'] if ip is not None else None
    return ip

def read_mindaffectBCI_message(line):
    if StimulusEvent.msgName in line:
        msg = read_StimulusEvent(line)
    elif DataPacket.msgName in line:
        msg = read_DataPacket(line)
    elif ModeChange.msgName in line:
        msg = read_ModeChange(line)
    elif NewTarget.msgName in line:
        msg = read_NewTarget(line)
    elif Selection.msgName in line:
        msg = read_Selection(line)
    else:
        msg = None
    # add the server time-stamp
    if msg :
        msg.sts = read_serverts(line) # server time-samp
        msg.clientip = read_clientip(line) # client ip-address
    return msg

def datapackets2array(msgs):
    data=[]
    from mindaffectBCI.decoder.UtopiaDataInterface import timestamp_interpolation, linear_trend_tracker
    tsfilt = timestamp_interpolation(sample2timestamp=linear_trend_tracker(halflife=500))
    for msg in msgs:
        samples = msg.samples
        ts   = msg.timestamp
        ts   = ts % (1<<24)
        samples_ts = tsfilt.transform(ts,len(samples))
        samples = np.append(samples, samples_ts[:,np.newaxis], -1).astype(samples.dtype)
        data.append(samples)
        #last_ts = ts
    # convert data into single np array
    data = np. concatenate(data,0)
    return data    

def rewrite_timestamps2servertimestamps(msgs):
    ''' rewrite message time-stamps  to best-fit server time-stamps '''
    # get the client-timestamp, server-timestamp pairs
    x = np.array([msg.timestamp for msg in msgs]) #  from: client timestamp
    y = np.array([msg.sts for msg in msgs]) # to: server timestamp

    # Warning: strip and -1....
    invalidts = np.logical_or(x==-1,y==-1)
    x = x[~invalidts]
    y = y[~invalidts]
    # add constant feature  for the intercept
    x = np.append(x[:,np.newaxis],np.ones((x.shape[0],1)),1)
    # LS  solve
    # TODO[X]: use a robust least squares which allows for outliers due to network delays
    # TODO[]: use a proper weighted least squares robust estimator, also for on-line
    y_fit = y.copy()
    for i in range(3):
        ab,res,_,_ = np.linalg.lstsq(x,y_fit,rcond=-1)
        y_est = x[:,0]*ab[0] + ab[1]
        err = y - y_est # server > true, clip positive errors
        scale = np.mean(np.abs(err))
        clipIdx = err > 3*scale
        #print("{} overestimates".format(np.sum(clipIdx)))
        y_fit[clipIdx] = y_est[clipIdx] + 3*scale
        clipIdx = err < -3*scale
        #print("{} underestimates".format(np.sum(clipIdx)))
        y_fit[clipIdx] = y_est[clipIdx] - 3*scale
    #print("ab={}".format(ab))
    # now rewrite the client time-stamps
    for m in msgs:
        m.rawtimestamp = m.timestamp
        m.timestamp = m.rawtimestamp*ab[0] + ab[1]

    return msgs

    
def read_mindaffectBCI_messages( fn:str, regress:bool=True ):
    ''' read the data from a text-file save version into a list of messages,
        WARNING: this reads the  messages as raw, and does *not* try to time-stamp clocks
                 w.r.t.  the message source.  To compare messages between clients you will
                 need to do this manually! '''
    fn = os.path.expanduser(fn)
    with open(fn,'r') as file:
        msgs=[]
        for line in file:
            msg = read_mindaffectBCI_message(line)
            if msg is not None:
                msgs.append(msg)

    # TODO [X]: intelligent time-stamp re-writer taking account of the client-ip
    if regress is None:
        # do nothing, leave client + server time-stamps in place
        pass
    elif regress==False: 
        # just use the server ts, as they have been put on the same time-line by the hub
        for m in msgs:
            m.timestamp = m.sts
    else:
        clientips = [ m.clientip for m in msgs ]
        for client in set(clientips):
            clientmsgs = [ c for c in msgs if c.clientip == client ]
            # only regress datapacket messages
            #if not any(isinstance(m,DataPacket) for m in clientmsgs):
            print('rewrite for client ip: {}'.format(client))
            rewrite_timestamps2servertimestamps(clientmsgs)
            #else:
            #    for m in msgs:
            #        m.timestamp = m.sts
        
    return msgs

def read_mindaffectBCI_data_messages( fn:str, regress=True ):
    ''' read the data from a text-file save version into a dataarray and message list '''
    rawmsgs = read_mindaffectBCI_messages(fn, regress)
    # split into datapacket messages and others
    data=[]
    msgs=[]
    for m in rawmsgs:
        #  WARNING BODGE: fit time-stamp in 24bits for float32 ring buffer
        #  Note: this leads to wrap-arroung in (1<<24)/1000/3600 = 4.6 hours
        #        but that shouldn't matter.....
        m.timestamp = m.timestamp % (1<<24)

        if isinstance(m,DataPacket):
            data.append(m)
        else:
            msgs.append(m)

    # convert the data messages into a single numpy array,
    # with (interpolated) time-stamps in the final 'channel'
    data = datapackets2array(data)
    
    return (data,msgs)


def testcase(fn=None):
    ''' testcase, load reference datafile '''
    if fn is None:
        fn = 'mindaffectBCI.txt'

    print("read messages from {}".format(fn))
    msgs = read_mindaffectBCI_messages(fn)
    for msg in msgs[:100]:
        print("{}".format(msg))

    print("read data messages")
    data,msgs = read_mindaffectBCI_data_messages(fn)
    print("Data({})={}".format(data.shape,data))
    for m in msgs[:100]:
        print("{}".format(m))

    
if __name__=="__main__":
    # default to last log file if not given
    import glob
    import os
    fileregexp = '../../../logs/mindaffectBCI*.txt'
    #fileregexp = '../../../../utopia/java/utopia2ft/UtopiaMessages*.log'
    files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),fileregexp)) # * means all if need specific format then *.csv
    fn = max(files, key=os.path.getctime)
    #if len(sys.argv) > 0:
    #    fn = sys.argv[1]
    testcase(fn)
