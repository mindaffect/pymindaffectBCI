# Copyright (c) 2020 MindAffect B.V. 
#  Author: Khashayar Sharif <khash@mindaffect.nl>
#  Build on top of brainflow and MindAffect's original brainflow based EEG
#  driver written by Jason Farquhar. 
#  Extra features added by this code:
#  1.local openbci cyton timestamps are used instead of host timestamps.
#  2.The user has the option of choosing/changinh the sampling frequency of the eeg frontened.
#  3.The user has the option to set the board to bipolar mode to perform a triggercheck.
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

import argparse
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from mindaffectBCI import utopiaclient 
from time import time, sleep
import traceback
import numpy
from struct import *

PACKETRATE_HZ = 50
LOGINTERVAL_S = 3
t0=None
nextLogTime=None
SampFreq2WiFiCommands = {250:'~6', 500:'~5', 1000:'~4'}
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
    parser.add_argument ('--log', type=int, help = ' set the brainflow logging level', default=1)
    parser.add_argument ('--triggerCheck', type = int, help  = 'trigger check', required = False, default = 0)
    parser.add_argument ('--samplingFrequency', type = int, help  = 'sampling frequency', required = False, default = 0)
    args = parser.parse_args ()
    return args

board = None
client = None
def run (host:str=None,board_id:int=1,ip_port:int=0,serial_port:str='',mac_address:str='',other_info='',
         serial_number='',ip_address='',ip_protocol=0,timeout:float=0,streamer_params='',log:int=1,
         config_params:list=None, trigger_check:bool=0, samplingFrequency:float=0):
    """use the brainflow library to connect to a biosensing board and stream the data to mindaffectBCI

    Basically, this is a thin wrapper round the brainflow <https://brainflow.readthedocs.io> python library to forward to the mindaffectBCI utopia-hub

    Args:
        host (str, optional): hostname or IP for the utopiahub. Defaults to None.
        board_id (int, optional): brainflow board id, see <https://brainflow.readthedocs.io/en/stable/SupportedBoards.html> 0=cyton, 1=ganglyon. Defaults to 1.
        ip_port (int, optional): brainflow ip-port. Defaults to 0.
        serial_port (str, optional): brainflow serial-port for the board. Defaults to ''.
        mac_address (str, optional): brainflow mac-address. Defaults to ''.
        other_info (str, optional): other info to send to brainflow. Defaults to ''.
        serial_number (str, optional): board serial number. Optional. Defaults to ''.
        ip_address (str, optional): brainflow ip_address. Defaults to ''.
        ip_protocol (int, optional): brainflow ip protocol. Defaults to 0.
        timeout (float, optional): brainflow timeout. Defaults to 0.
        streamer_params (str, optional): brainflow streamer params. Defaults to ''.
        log (int, optional): brainflow log level. Defaults to 1.
        config_params (list, optional): additional configuration parameters to send to the board after startup. Defaults to None.
        trigger_check (bool, optional): flag to configure channel-8 on cyton as trigger input, e.g. for timing tests. Defaults to 0.
        samplingFrequency (float, optional): desired sampling rate to set the board to. Defaults to 0.
    """
    global board, client
    # log the config
    configmsg = "{}".format(dict(component=__file__, args=locals()))

    # init the board params
    params = BrainFlowInputParams ()
    params.serial_port = serial_port
    params.ip_port = ip_port
    params.mac_address = mac_address
    params.other_info = other_info
    params.serial_number = serial_number
    params.ip_address = ip_address
    params.ip_protocol = ip_protocol
    params.timeout = timeout

    print('board_id={} params= {}'.format(board_id, vars(params)))

    if (log):
        BoardShim.enable_dev_board_logger ()
    else:
        BoardShim.disable_board_logger ()

    board = BoardShim (board_id , params)
    board.prepare_session ()
    if samplingFrequency > 0 and board_id ==5 :
	    board.config_board (SampFreq2WiFiCommands[samplingFrequency])
    if trigger_check and board_id in (0,5): # cyton boards
        print('trigger is enabled, on cyton trigger channel: 8')
        board.config_board('x8020000X')
    sleep(1)
    if board_id in (0,5):
        print("Enabling AMP time-stamps on cyton")
        board.config_board ('<')

    # do any user-specified config
    if config_params is not None:
        if isinstance(config_params,str):
            config_params=[config_params]
        print('Setting board config:')
        for c in config_params:
            print(c)
            board.config_board(c)
            sleep(.1)
        print('done')

    sleep(1)
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
    # log the config
    client.sendMessage(utopiaclient.Log(None, configmsg))
    print("Putting header.")
    client.sendMessage(utopiaclient.DataHeader(None, len(eeg_channels), fSample, ""))

    sleep(1) # just in case
    board.start_stream (45000, streamer_params)
    # N.B. we force a sleep here to allow the board to startup correctly
    sleep(3)

    maxpacketsamples = int(32000 / 4 / len(eeg_channels))
    nSamp=0
    nBlock=0
    data=None
    ots = None
    while True:

        data = board.get_board_data () # (channels,samples) get all data and remove it from internal buffer
        stamps=[]
        if board_id==0 or board_id==5:
            stamps=[]
            for i in range(len(data[15])):
                array=numpy.array([data[15][i],data[16][i],data[17][i],data[18][i]] , dtype=numpy.uint8)
                stamps.append(unpack('>L',bytearray(array)))		
        if data.size == 0:
            sleep(.001)
            continue

        # BODGE: simulated boards seem to not work correctly without some blocking IO
        if board_id < 0:
            print('.',end='',flush=True)
        #print("{} samp in {} uS - {}".format(data.shape[1],dts,data.shape[1]/dts))

        # extract the info we want
        eeg = data[eeg_channels,:] # (channels,samples) 
        if board_id==0 or board_id==5:
            timestamps=numpy.array(stamps)
        else:
            timestamps = data[timestamp_channel,:] # (samples,)

        # forward the *EEG* data to the utopia client
        nSamp = nSamp + eeg.shape[1]
        # format for sending to MA
        eeg = eeg.T # MA uses (samples,channels)

        # TODO[x]: send as smaller packets if too much data
        if eeg.shape[0] < maxpacketsamples:
            # fit time-stamp into 32-bit int (with potential wrap-around)
            ts = timestamps[-1]
            if board_id !=0 and board_id !=5:
                ts = (int(ts*1000))%(1<<31) 
            #ts = client.getTimeStamp()
            client.sendMessage(utopiaclient.DataPacket(ts, eeg))
            nBlock = nBlock + 1
        else:
            pktidx = list(range(0,eeg.shape[0],maxpacketsamples)) + [eeg.shape[0],]
            for i in range(len(pktidx)-1):
                d = eeg[pktidx[i]:pktidx[i+1],:]
                ts = timestamps[pktidx[i+1]-1]
                # ensure increasing time-stamps.... (brainflow bug?)
                ts = max(ots,ts) if ots is not None else ts
                ots = ts
                # fit time-stamp into 32-bit int (with potential wrap-around)
                if board_id !=0 and board_id !=5:
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
    #run(board_id=1,serial_port='com3') # ganglion
    #run(board_id=0,serial_port='com4')

    args=parse_args()    
    try:
        run(**vars(args))
    except:
        traceback.print_exc()
