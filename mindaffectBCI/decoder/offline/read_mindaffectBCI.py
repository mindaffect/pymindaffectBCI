import os
import numpy as np
import re
from mindaffectBCI.utopiaclient import StimulusEvent, DataPacket, ModeChange, NewTarget, Selection, DataHeader
from mindaffectBCI.decoder.utils import unwrap

# named reg-exp to parse the different messages types log lines
recievedts_re = re.compile(r'.*rts:(?P<rts>[-0-9]*)\W')
serverts_re = re.compile(r'.*sts:(?P<sts>[-0-9]*)\W')
clientts_re = re.compile(r'^.*\Wts:(?P<ts>[-0-9]*)\W')
clientip_re = re.compile(r'.*<-\W(?P<ip>[0-9.:]*)$')
stimevent_re = re.compile(r'^.*\Wts:(?P<ts>[-0-9]*)\W*v\[(?P<shape>[0-9x]*)\]:(?P<stimstate>.*) <-.*$')
datapacket_re = re.compile(r'^.*\Wts:(?P<ts>[-0-9]*)\W*v\[(?P<shape>[0-9x]*)\]:(?P<samples>.*) <-.*$')
modechange_re = re.compile(r'^.*\Wts:(?P<ts>[-0-9]*)\W.*mode:(?P<newmode>.*) <-.*$')
dataheader_re = re.compile(r'^.*\Wts:(?P<ts>[-0-9]*)\W.*fs(?P<fs>[-0-9]*)\W.*ch\[(?P<nch>[-0-9]*)\]:(?P<labels>.*) <-.*$')

def read_StimulusEvent(line:str):
    ''' read a stimulus event message from a line of a mindaffectBCI offline save file '''
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
    stimstate = [ s if s>=0 else s+256 for s in stiminfo[1::2] ]  # signed->unsigned conversion 
    #print("SE ts:{} objIDs:{} state:{}".format(ts,objIDs,stimstate))
    return StimulusEvent(ts,objIDs,stimstate)
    
def read_DataPacket(line:str ):
    """read a data-packet line from a mindaffectBCI offline save file

    Args:
        line (str): the line to read

    Returns:
        DataPacket: a mindaffectBCI.utopiaclient.messages.DataPacket object containing (nsamp,d) EEG data
    """   
    # named reg-ex to extract the bits we need
    res = datapacket_re.match(line)
    if res is None:
        return None
    ts = int(res['ts'])
    # parse sample into numpy array
    shape = np.fromstring(res['shape'].replace('x',','),sep=',',dtype=int)
    shape = shape[::-1] # N.B. python order, fastest last..
    samples = np.fromstring(res['samples'].replace(']','').replace('[',''),sep=',',dtype=np.float32)
    samples = samples[:np.prod(shape)].reshape(shape) # guard too many samples?
    return DataPacket(ts,samples)

def read_DataHeader(line:str ):
    """read a data-header line from a mindaffectBCI offline save file

    Args:
        line (str): the line to read

    Returns:
        DataPacket: a mindaffectBCI.utopiaclient.messages.DataPacket object containing (nsamp,d) EEG data
    """   
    # named reg-ex to extract the bits we need
    res = dataheader_re.match(line)
    if res is None:
        return None
    ts = int(res['ts'])
    # parse sample into numpy array
    fs = float(res['fs'])
    nch = int(res['nch'])
    labels = [c.strip() for c in res['labels'].split(",")]
    return DataHeader(ts,fs,nch,labels)


def read_ModeChange(line:str):
    """read a mode-change line from a mindaffectBCI offline save file

    Args:
        line (str): the line to read

    Returns:
        ModeChange: a mode-change object, with the new mode information
    """
    res = modechange_re.match(line)
    if res is None:
        return None
    ts = int(res['ts'])
    newmode = res['newmode']
    return ModeChange(ts,newmode)

def read_NewTarget(line:str):
    """read a newtarget message line from a mindaffectBCI offline save file

    Args:
        line (str): the line to read

    Returns:
        NewTarget: a newtarget object
    """
    ts = read_clientts(line)
    return NewTarget(ts)

def read_Selection(line:str):
    """read a Selection message line from a mindaffectBCI offline save file

    Args:
        line (str): the line to read

    Returns:
        Selection: a selection object - Note: currently the selection information is *not* valid
    """
    # TODO[]: read the actual selection info
    match = read_clientts(line)
    return Selection(match,-1)

def read_clientts(line:str):
    """read the client-timestamp from a message line from a mindaffectBCI offline save file

    Note: the client-timestamp is the *raw* timestamp sent by the client, in the clients local clock.

    Args:
        line (str): the line to read

    Returns:
        clientts (int): the client timestamp
    """
    match = clientts_re.match(line)
    match = int(match['ts']) if match is not None else None
    return match

def read_serverts(line:str):
    """read the server-timestamp from a message line from a mindaffectBCI offline save file

    Note: the server-timestamp is the client-timestamp after mapping to the servers local clock.  Thus, server timestamps are directly comparable for all clients.

    Args:
        line (str): the line to read

    Returns:
        ts (int): the server timestamp
    """
    match = serverts_re.match(line)
    match = int(match['sts']) if match is not None else None
    return match

def read_recievedts(line:str):
    """read the recieved-timestamp from a message line from a mindaffectBCI offline save file

    Note: the recieved-timestamp is the time the client-message was recieved, measured on the servers clock.

    Args:
        line (str): the line to read

    Returns:
        ts (int): the timestamp
    """
    match = recievedts_re.match(line)
    match = int(match['rts']) if match is not None else None
    return match

def read_clientip(line:str):
    """read the client-ip-address from a message line from a mindaffectBCI offline save file

    Args:
        line (str): the line to read

    Returns:
        ip (str): the client ip-address
    """
    ip = clientip_re.match(line)
    ip = ip['ip'] if ip is not None else None
    return ip

def read_mindaffectBCI_message(line:str):
    """Read a mindaffectBCI message from a line of text

    Args:
        line (str): A line containing a mindaffectBCI message

    Returns:
        message_type: The decoded message as a message class.
    """    
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
    elif DataHeader.msgName in line:
        msg = read_DataHeader(line)
    else:
        msg = None
    # add the server time-stamp
    if msg :
        msg.sts = read_serverts(line) # server time-samp
        msg.rts = read_recievedts(line) # recieved time-stamp
        msg.clientip = read_clientip(line) # client ip-address
    return msg

def datapackets2array(msgs, sample2timestamp='lower_bound_tracker', timestamp_ch=None):
    """Convert a set of datapacket messages to a 2-d numpy array (with timestamp channel)

    Args:
        msgs ([type]): list of DataPacket messages
        sample2timestamp (str, optional): filtering function, to filter the data-packet time-stamps using the increasing sample counts. Defaults to 'lower_bound_tracker'.
        timestamp_ch (int, optional): If set, channel which contains the timestamp information. Defaults to None.

    Returns:
        X( (t,d) np.ndarray): the extracted samples in a numpy array
    """
    data=[]
    from mindaffectBCI.decoder.UtopiaDataInterface import timestamp_interpolation, linear_trend_tracker

    # extract the time-stamp channel and map to server time-stamps
    # and insert as the packet time-stamp
    if timestamp_ch is not None and timestamp_ch < len(msgs[0].samples[0]):
        # get the client-timestamp, server-timestamp pairs
        x = np.array([msg.samples[-1,timestamp_ch] for msg in msgs]) #  from: client timestamp, from ts-channel
        y = np.array([msg.rts for msg in msgs]) # to: server timestamp
        ab = robust_timestamp_regression(x,y)
        # now rewrite the client time-stamps
        for m in msgs:
            m.rawtimestamp = m.timestamp # save the old one
            m.timestamp = m.samples[-1,timestamp_ch]*ab[0] + ab[1]
            # wrap into the desired bit-size if wanted
            if hasattr(m,'unwrapped_timestamp'):
                m.unwrapped_timestamp = m.timestamp
                # BODGE: fixed wrapping size!
                m.timestamp = m.timestamp % (1<<24)
            
    tsfilt = timestamp_interpolation(sample2timestamp=sample2timestamp)
    for m in msgs:
        samples = m.samples
        ts   = m.timestamp
        # TODO[]: look at using the time-stamps non-filtered?
        samples_ts = tsfilt.transform(ts,len(samples))
        samples = np.append(samples, samples_ts[:,np.newaxis], -1).astype(samples.dtype)
        data.append(samples)
        #last_ts = ts
        
    # convert data into single np array
    data = np. concatenate(data,0)
    return data    


def robust_timestamp_regression(x,y):
    """Given 2 time-stamp streams, e.g. one from server, one from client, compute a robust, outlier resistant linear mapping from one to the other.

    Args:
        x (int/float, (ntimes,)): the source time-stamps
        y (int/float, (ntimes,)): the destination time-stamps

    Returns:
        ab: the gain and bias for the linear map such that:  y = ab[0]*x + ab[1]
    """    
    # Warning: strip and -1....
    invalidts = np.logical_or(x==-1,y==-1)
    x = x[~invalidts]
    x = unwrap(x)
    y = y[~invalidts]
    y = unwrap(y)
    # add constant feature  for the intercept
    x = np.append(x[:,np.newaxis],np.ones((x.shape[0],1)),1)
    # LS  solve
    # TODO[X]: use a proper weighted least squares robust estimator, also for on-line
    sqrt_w = np.ones(y.shape,dtype=x.dtype)
    for i in range(4):
        ab,res,_,_ = np.linalg.lstsq(x*sqrt_w[:,np.newaxis],y*sqrt_w,rcond=-1)
        y_est = x[:,0]*ab[0] + ab[1]
        err = y - y_est # server > true, clip positive errors
        scale = np.mean(np.abs(err[sqrt_w>0]))

        sqrt_w[:]=1
        clipIdx = err > 3*scale
        sqrt_w[clipIdx]=0
        #print("{} overestimates".format(np.sum(clipIdx)))
        clipIdx = err < -3*scale
        sqrt_w[clipIdx]=0
        #print("{} underestimates".format(np.sum(clipIdx)))
    return ab


def rewrite_timestamps2servertimestamps(msgs):
    ''' rewrite message client-timestamps  to best-fit server time-stamps '''
    # get the client-timestamp, server-timestamp pairs
    x = np.array([msg.timestamp for msg in msgs]) #  from: client timestamp
    y = np.array([msg.sts for msg in msgs]) # to: server timestamp
    ab = robust_timestamp_regression(x,y)

    #print("ab={}".format(ab))
    # now rewrite the client time-stamps
    for m in msgs:
        m.rawtimestamp = m.timestamp
        m.timestamp = m.rawtimestamp*ab[0] + ab[1]
    return msgs

    
def read_mindaffectBCI_messages( source, regress:bool=False ):
    """read all the messages from a mindaffetBCI offline save file

    Args:
        source ([str, stream]): the log file messages source, can be file-name, or IO-stream, or string
        regress (bool, optional): How should we regress the client-time stamps onto the server time-stamps.  If False then use the server-time-stamps, if None then leave the client-time-stamps, if True then use robust-regression to map from client to server time-stamps.
        Defaults to False.

    Returns:
        (list, messages): a list of all the decoded messages
    """
    if hasattr(source, 'readline'):
        stream = source
    elif isinstance(source,str):
        import glob
        source = max(glob.glob(os.path.expanduser(source)), key=os.path.getctime)
        if os.path.exists(source): # read from file
            source = os.path.expanduser(source)
            stream = open(source,'r')
        else: # assume it's already a string with the messages in
            import io
            stream = io.StringIO(fn)

    msgs=[]
    for line in stream:
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
            m.origal_timestamp = m.timestamp
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

def read_mindaffectBCI_data_messages( source, regress=False, timestamp_wrap_size=(1<<24), **kwargs ):
    """read an offline mindaffectBCI save file, and return raw-data (as a np.ndarray) and messages. 

    Args:
        source (str): the file name to load the data from
        regress (bool, optional): How to map from client-specific to a common time-stamp basis. Defaults to False.
        timestamp_wrap_size (tuple, optional): The bit-resolution of the time-stamps. Defaults to (1<<24).

    Returns:
        data (np.ndarray (nsamp,d) float): the time-stamped data stream
        messages (list messages): the (non-datapacket) messages in the file
    """
    rawmsgs = read_mindaffectBCI_messages(source, regress)
    # split into datapacket messages and others
    data=[]
    msgs=[]
    for m in rawmsgs:
        #  WARNING BODGE: fit time-stamp in 24bits for float32 ring buffer
        #  Note: this leads to wrap-arroung in (1<<24)/1000/3600 = 4.6 hours
        #        but that shouldn't matter.....
        if timestamp_wrap_size is not None:
            m.unwrapped_timestamp = m.timestamp
            m.timestamp = m.timestamp % timestamp_wrap_size

        if isinstance(m,DataPacket):
            data.append(m)
        else:
            msgs.append(m)

    # convert the data messages into a single numpy array,
    # with (interpolated) time-stamps in the final 'channel'
    data = datapackets2array(data, **kwargs)
    
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
    fileregexp = '../../../logs/mindaffectBCI*.txt'
    #fileregexp = '../../../../utopia/java/utopia2ft/UtopiaMessages*.log'
    #if len(sys.argv) > 0:
    #    fn = sys.argv[1]

    import glob
    import os
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    import os
    root = Tk()
    root.withdraw()
    savefile = askopenfilename(initialdir=os.getcwd(),
                                title='Chose mindaffectBCI save File',
                                filetypes=(('mindaffectBCI','mindaffectBCI*.txt'),('All','*.*')))
    testcase(savefile)
