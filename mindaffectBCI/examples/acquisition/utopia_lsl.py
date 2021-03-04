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
def run (host=None, streamtype='EEG', **kwargs):
    global inlet, client

    # connect to the utopia client
    client = utopiaclient.UtopiaClient()
    client.disableHeartbeats() # disable heartbeats as we're a datapacket source
    client.autoconnect(host)

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
    labels=[]
    try:
        ch = info.desc().child("channels").child("channel")
        for k in range(nch):
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()
    except:
        pass
    print("board with {} ch @ {} hz".format(nch, fSample))

    # don't subscribe to anything
    client.sendMessage(utopiaclient.Subscribe(None, ""))
    print("Putting header.")
    client.sendMessage(utopiaclient.DataHeader(None, fSample, nch, labels))

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

        # fit time-stamp into 32-bit int (with potential wrap-around)
        ts = timestamps[-1]
        ts = (int(ts*1000))%(1<<31) 
        #ts = client.getTimeStamp()
        client.sendMessage(utopiaclient.DataPacket(ts, samples))
        nBlock = nBlock + 1

        # limit the packet sending rate..
        printLog(nSamp,nBlock)        

if __name__ == "__main__":
    run()