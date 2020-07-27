import argparse
import numpy as np
from mindaffectBCI import utopiaclient 
from time import time, sleep

LOGINTERVAL_S = 3
t0=None
nextLogTime=None
def printLog(nSamp, nBlock):
    ''' textual logging of the data arrivals etc.'''
    global t0, nextLogTime
    t = time()
    if t0 is None:
        t0 = t
    if nextLogTime is None:
        nextLogTime = t
    if t > nextLogTime:
        elapsed = time()-t0
        print("%d %d %f %f (samp,blk,s,hz)"%(nSamp, nBlock, elapsed, nSamp/elapsed), flush=True)
        nextLogTime = t +LOGINTERVAL_S


def parse_args():
    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--host', type=str, help='host name for the utopia hub', required=False, default=None)
    parser.add_argument('--nch', type=str, help='number of simulated channels', required=False, default=4)
    parser.add_argument('--fs', type=str, help='simulated channels sample rate', required=False, default=200)
    args = parser.parse_args()
    return args

client = None
def run(host=None, nch:int=4, fs:float=200, packet_size:int=10):
    """run a simple fake-data stream with gaussian noise channels

    Args:
        host ([str], optional): address for the utopia hub. Defaults to None.
        nch (int, optional): number of simulated channels. Defaults to 4.
        fs (int, optional): simulated data sample rate. Defaults to 200.
        packet_size (int, optional): number channels to put in each utopia-hub datapacket. Defaults to 10.
    """    
    global client
    # connect to the utopia client
    client = utopiaclient.UtopiaClient()
    client.autoconnect(host)
    # don't subscribe to anything
    client.sendMessage(utopiaclient.Subscribe(None, ""))
    print("Putting header.")
    client.sendMessage(utopiaclient.DataHeader(None, nch, fs, ""))

    nSamp = 0
    nPacket = 0
    t0 = client.getTimeStamp()
    packet_interval = packet_size * 1000 / fs
    while True:

        # limit the packet sending rate..
        sleep( (t0 + nPacket*packet_interval - client.getTimeStamp()) / 1000)
        # generate random data
        data = np.random.standard_normal((nch, packet_size))

        # forward to the utopia client
        nSamp = nSamp + data.shape[1]
        nPacket = nPacket + 1
        client.sendMessage(utopiaclient.DataPacket(client.getTimeStamp(), data))

        printLog(nSamp, nPacket)        

if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))