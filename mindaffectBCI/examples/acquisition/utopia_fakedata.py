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
def run(host=None, nch: int=4, fs: float=200, packet_size: int=10):
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
    print("Putting header. {} ch @ {} Hz".format(nch,fs))
    client.sendMessage(utopiaclient.DataHeader(None, nch, fs, ""))

    # setup the ERP injection trigger listener
    import socket
    import struct
    # Create a TCP/IP socket
    trigger_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    trigger_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    trigger_sock.bind(("0.0.0.0", 8300))
    trigger_sock.setblocking(False)  # set into non-blocking mode

    nSamp = 0
    nPacket = 0
    t0 = client.getTimeStamp()
    sample_interval = 1000 / fs
    data = np.zeros((packet_size, nch), dtype=np.float32)
    while True:

        # generate a packets worth of data in 'real-time' so attach triggers
        for si in range(packet_size):
            # limit the sample rate..
            sleep(max(0, (t0 + nSamp*sample_interval - client.getTimeStamp()) / 1000))
            # generate random data 
            data[si, :] = np.random.standard_normal((nch,))
            nSamp = nSamp + 1
            # check for trigger input
            try:
                trig, addr = trigger_sock.recvfrom(1024)
                # decode the payload
                trig = struct.unpack('f', trig[:4]) if len(trig) == 4 else int(trig[0]*5)
                # add to the raw data
                data[si, -1] = data[si, -1] + trig
                print('t', end='', flush=True)
            except socket.error as ex:
                pass

        # forward to the utopia client
        nPacket = nPacket + 1
        client.sendMessage(utopiaclient.DataPacket(client.getTimeStamp(), data))

        printLog(nSamp, nPacket)        


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))