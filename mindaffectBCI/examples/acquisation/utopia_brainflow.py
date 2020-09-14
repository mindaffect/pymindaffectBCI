import argparse
from brainflow.board_shim import BoardShim, BrainFlowInputParams
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
    parser.add_argument ('--timeout', type = int, help  = 'timeout for device discovery or connection', required = False, default = 0)
    parser.add_argument ('--ip-port', type = int, help  = 'ip port', required = False, default = 0)
    parser.add_argument ('--ip-protocol', type = int, help  = 'ip protocol, check IpProtocolType enum', required = False, default = 0)
    parser.add_argument ('--ip-address', type = str, help  = 'ip address', required = False, default = '')
    parser.add_argument ('--serial-port', type = str, help  = 'serial port', required = False, default = 'com3') # '')
    parser.add_argument ('--mac-address', type = str, help  = 'mac address', required = False, default = '')
    parser.add_argument ('--other-info', type = str, help  = 'other info', required = False, default = '')
    parser.add_argument ('--streamer-params', type = str, help  = 'streamer params', required = False, default = '')
    parser.add_argument ('--serial-number', type = str, help  = 'serial number', required = False, default = '')
    parser.add_argument ('--board-id', type = int, help  = 'board id, check docs to get a list of supported boards', default = 1 )#required = True)
    parser.add_argument ('--log', action = 'store_true')
    args = parser.parse_args ()
    return args

board = None
client = None
def run (host=None,board_id=1,ip_port=0,serial_port='',mac_address='',other_info='',
         serial_number='',ip_address='',ip_protocol=0,timeout=0,streamer_params='',log=0):
    global board, client

    # init the board params
    params = BrainFlowInputParams ()
    params.ip_port = ip_port
    params.serial_port = serial_port
    params.mac_address = mac_address
    params.other_info = other_info
    params.serial_number = serial_number
    params.ip_address = ip_address
    params.ip_protocol = ip_protocol
    params.timeout = timeout

    print('params= {}'.format(vars(params)))

    if (log):
        BoardShim.enable_dev_board_logger ()
    else:
        BoardShim.disable_board_logger ()

    board = BoardShim (board_id , params)
    board.prepare_session ()

    eeg_channels = BoardShim.get_eeg_channels (board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    fSample = BoardShim.get_sampling_rate(board_id)

    print("board with {} ch @ {} hz".format(len(eeg_channels), fSample))

    # connect to the utopia client
    client = utopiaclient.UtopiaClient()
    client.disableHeartbeats() # disable heartbeats as we're a datapacket source
    client.autoconnect(host)
    # don't subscribe to anything
    client.sendMessage(utopiaclient.Subscribe(None, ""))
    print("Putting header.")
    client.sendMessage(utopiaclient.DataHeader(None, len(eeg_channels), fSample, ""))

    sleep(1) # just in case
    board.start_stream (45000, streamer_params)
    # N.B. we force a sleep here to allow the board to startup correctly
    sleep(3)

    maxpacketsamples = int(16000 / 4 / len(eeg_channels))
    nSamp=0
    nBlock=0
    data=None
    ots = None
    while True:

        data = board.get_board_data () # (channels,samples) get all data and remove it from internal buffer
        if data.size == 0:
            sleep(.001)
            continue

        # BODGE: simulated boards seem to not work correctly without some blocking IO
        if board_id <= 0:
            print('.',end='',flush=True)
        #print("{} samp in {} uS - {}".format(data.shape[1],dts,data.shape[1]/dts))

        # extract the info we want
        eeg = data[eeg_channels,:] # (channels,samples) 
        timestamps = data[timestamp_channel,:] # (samples,)

        # forward the *EEG* data to the utopia client
        nSamp = nSamp + eeg.shape[1]
        # format for sending to MA
        eeg = eeg.T # MA uses (samples,channels)

        # TODO[]: send as smaller packets if too much data
        pktidx = list(range(0,eeg.shape[0],maxpacketsamples)) + [eeg.shape[0],]
        for i in range(len(pktidx)-1):
            d = eeg[pktidx[i]:pktidx[i+1],:]
            ts = timestamps[pktidx[i+1]-1]
            # ensure increasing time-stamps.... (brainflow bug?)
            ts = max(ots,ts) if ots is not None else ts
            ots = ts
            # fit time-stamp into 32-bit int (with potential wrap-around)
            ts = (int(ts*1000))%(1<<31) 
            #ts = client.getTimeStamp()
            client.sendMessage(utopiaclient.DataPacket(ts, d))
            nBlock = nBlock + 1

        # limit the packet sending rate..
        sleep(1/PACKETRATE_HZ)
        printLog(nSamp,nBlock)        

    board.stop_stream ()
    board.release_session ()

if __name__ == "__main__":
    #run(board_id=1,serial_port='com3')

    args=parse_args()    
    try:
        run(**vars(args))
    except:
        traceback.print_exc()
