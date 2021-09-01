import argparse
from pylsl import StreamInlet, resolve_stream, StreamInfo
from mindaffectBCI import utopiaclient 
from time import time, sleep
import traceback

PACKETRATE_HZ = 50
LOGINTERVAL_S = 3
t0=None
nextLogTime=None
def printLog(nSamp, nBlock):
    ''' textual logging of the data arrivals etc.'''
    global t0, nextLogTime
    t = time()
    if t0 is None: 
        t0 = t
        nextLogTime = t
    if t > nextLogTime:
        elapsed = time()-t0
        print("%d %d %f %f (samp,blk,s,hz)"%(nSamp, nBlock, elapsed, nSamp/elapsed), flush=True)
        nextLogTime = t +LOGINTERVAL_S


def parse_args():
    parser = argparse.ArgumentParser ()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument ('--host', type = str, help  = 'host name for the utopia hub', required = False, default = None)
    parser.add_argument ('--type', type = str, help  = 'stream type of the LSL resolver', required = False, default = 'EEG')
    args = parser.parse_args ()
    return args

inlet = None
client = None
def run (host=None, streamtype:str='EEG', channels:list=None, **kwargs):
    """[summary]

    Args:
        host ([type], optional): ip-address of the minaffectBCI Hub, auto-discover if None. Defaults to None.
        streamtype (str, optional): the type of stream to forward to the hub. Defaults to 'EEG'.
        channels (list, optional): list-of-int, channel indices to forward, list-of-str list of channel names to forward. Defaults to None.
    """    
    global inlet, client

    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', streamtype)

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    # get meta-info about this stream
    info = inlet.info()
    print("stream info: {}".format(info.as_xml()))
    fSample = info.nominal_srate()
    nch = info.channel_count()
    ch_names= [ "{}".format(i) for i in range(nch) ]
    try:
        ch = info.desc().child("channels").child("channel")
        for k in range(nch):
            ch_names[k] = ch.child_value("label")
            ch = ch.next_sibling()
    except:
        pass
    print("board with {} ch @ {} hz".format(nch, fSample))

    # get subset of channels to stream -- if wanted
    ch_idx = [True for _ in range(nch)]
    if channels is not None:
        ch_idx = [False for _ in range(nch)]
        if isinstance(channels,str):
            channels = [c.trim() for c in channels.split(',')]
        for c in channels:
            if isinstance(c,int):
                ch_idx[c]=True
            elif isinstance(c,str):
                try:
                    ch_idx[[n.lower() for n in ch_names].index(c.lower())]=True
                except ValueError:
                    pass

    nstream = sum(ch_idx)
    ch_stream = [ c for i,c in enumerate(ch_names) if ch_idx[i] ]

    print("Streaming {}ch = {}".format(nstream,ch_stream))

    # connect to the utopia client
    client = utopiaclient.UtopiaClient()
    client.disableHeartbeats() # disable heartbeats as we're a datapacket source
    client.autoconnect(host)

    # don't subscribe to anything
    client.sendMessage(utopiaclient.Subscribe(None, ""))
    print("Putting header.")
    client.sendMessage(utopiaclient.DataHeader(None, fSample, nstream, ch_stream))

    maxpacketsamples = int(32000 / 4 / nch)
    nSamp=0
    nBlock=0
    while True:

        # grab all the new data + time-stamp
        samples, timestamps = inlet.pull_chunk(timeout=1/PACKETRATE_HZ, max_samples=maxpacketsamples)
        if len(timestamps) == 0:
            sleep(.001)
            continue

        # forward the *EEG* data to the utopia client
        nSamp = nSamp +len(samples)

        # get the subset...
        if nstream < nch:
            tmp = samples
            samples = []
            for s in tmp:
                samples.append([c for i,c in enumerate(s) if ch_idx[i]])
            del tmp

        # fit time-stamp into 32-bit int (with potential wrap-around)
        ts = timestamps[-1]
        ts = (int(ts*1000))%(1<<31) 
        #ts = client.getTimeStamp()
        client.sendMessage(utopiaclient.DataPacket(ts, samples))
        nBlock = nBlock + 1

        # limit the packet sending rate..
        printLog(nSamp,nBlock)        

if __name__ == "__main__":
    print("To start a debug data stream use in console: python -m pylsl.examples.SendData")
    channels=['Cz' ,'C3']
    channels=[0,1,2,5,6]
    run(channels=channels)
