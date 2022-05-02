#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jadref@gmail.com>
# This file is part of pymindaffectBCI <https://github.com/mindaffect/pymindaffectBCI>.
#
# pymindaffectBCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pymindaffectBCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pymindaffectBCI.  If not, see <http://www.gnu.org/licenses/>

from urllib.request import DataHandler
from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_message
from mindaffectBCI.utopiaclient import DataPacket
from mindaffectBCI.decoder.utils import askloadsavefile
from time import sleep, perf_counter

class FileProxyHub:
    ''' Proxy UtopiaClient which gets messages from a saved log file '''
    def __init__(self, filename:str=None, speedup:float=None, use_server_ts:bool=True):
        import glob
        import os
        if filename == 'lastsavefile':
            # default to last log file if not given
            files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
            filename = max(files, key=os.path.getctime)
        elif filename == 'askloadsavefile' or filename is None:
            filename = askloadsavefile()
        else:
            files = glob.glob(os.path.expanduser(filename))
            filename = max(files, key=os.path.getctime)
        self.filename = filename
        print("Loading : {}\n".format(self.filename))
        self.speedup = speedup
        self.isConnected = True
        self.lasttimestamp = None
        self.lastrealtimestamp = self.getRealTimeStamp()
        self.use_server_ts = use_server_ts
        self.file = open(self.filename,'r')
        print('FileProxyHub.py:: playback file {} at {}x speedup'.format(self.filename,self.speedup))
    
    def getTimeStamp(self):
        """ get the time-stamp in milliscecond on the data clock

        Returns:
            [type]: [description]
        """        
        return self.lasttimestamp

    def getRealTimeStamp(self):
        """
        get the time-stamp in milliseconds on the real-time-clock

        Returns:
            int: the timestamp
        """        
        return (int(perf_counter()*1000) % (1<<31))

    
    def autoconnect(self, *args,**kwargs):
        """[summary]
        """        
        pass

    def sendMessage(self, msg):
        """[summary]

        Args:
            msg ([type]): [description]
        """        
        pass

    def getNewMessages(self, timeout_ms):
        """[summary]

        Args:
            timeout_ms ([type]): [description]

        Returns:
            [type]: [description]
        """        
        msgs = []
        for line in self.file:
            msg = read_mindaffectBCI_message(line)
            if msg is None:
                continue
            # re-write timestamp to server time stamp
            if self.use_server_ts:
                msg.timestamp = msg.sts
            # initialize the time-stamp tracking
            if self.lasttimestamp is None or self.lasttimestamp==0: 
                self.lasttimestamp = msg.timestamp
            # add to outgoing message queue
            msgs.append(msg)
            # check if should stop
            if self.lasttimestamp is not None and self.lasttimestamp+timeout_ms < msg.sts:
                break
        else:
            # mark as disconneted at EOF
            self.isConnected = False
        if self.speedup :
            ttg = timeout_ms - (self.getRealTimeStamp() - self.lastrealtimestamp)
            if ttg>0 :
                sleep(ttg/1000./self.speedup)
        # update the time-stamp cursor
        self.lasttimestamp = self.lasttimestamp + timeout_ms
        self.lastrealtimestamp = self.getRealTimeStamp()
        return msgs


def run(filename:str=None, speedup:float=1, timeout_ms:float=1000, host:str=None, only_data_messages:bool=True):
    """stream data from file to a live utopia hub

    Args:
        filename (str): save file name to playback
        speedup (float, optional): run this multiple of real-time. Defaults to 1.
        timeout_ms (float, optional): send data in blocks of this size. Defaults to 100.
        host (str, optional): utopia-hub name to connect to.  Auto-detect if None. Defaults to None.
    """
    from mindaffectBCI.utopiaclient import UtopiaClient, Subscribe, DataPacket, DataHeader
    acq = FileProxyHub(filename,speedup=speedup)

    # connect to the utopia client
    client = UtopiaClient()
    client.disableHeartbeats() # disable heartbeats as we're a datapacket source, with given time-stamps
    client.autoconnect(host)
    # don't subscribe to anything
    client.sendMessage(Subscribe(None, ""))

    while acq.isConnected and client.isConnected:
        msgs = acq.getNewMessages(timeout_ms)
        if only_data_messages: # filter only data messages are forwarded
            msgs = [m for m in msgs if m.msgID in (DataPacket.msgID, DataHeader.msgID)]
        client.sendMessages(msgs)


def testcase(filename, speedup=None, fs=200, fs_out=200, filterband=((45,65),(0,3),(25,-1)), order=4):
    """[summary]

    Args:
        filename ([type]): [description]
        fs (int, optional): [description]. Defaults to 200.
        fs_out (int, optional): [description]. Defaults to 200.
        filterband (tuple, optional): [description]. Defaults to ((45,65),(0,3),(25,-1)).
        order (int, optional): [description]. Defaults to 4.
    """    
    import numpy as np
    from mindaffectBCI.decoder.UtopiaDataInterface import timestamp_interpolation, linear_trend_tracker, butterfilt_and_downsample

    U = FileProxyHub(filename,speedup=speedup)
    tsfilt = timestamp_interpolation(fs=fs,sample2timestamp=linear_trend_tracker(500))

    if filterband is not None:
        ppfn = butterfilt_and_downsample(filterband=filterband, order=order, fs=fs, fs_out=fs_out)
    else:
        ppfn = None
    
    #ppfn = None

    nsamp=0
    t=None
    data=[]
    ts=[]
    while U.isConnected:
        msgs = U.getNewMessages(1000)
        if t is None: t = U.lasttimestamp
        print('{} s\r'.format((U.lasttimestamp-t)/1000),end='',flush=True)
        for m in msgs:
            if m.msgID == DataPacket.msgID:
                timestamp = m.timestamp % (1<<24)
                samples = m.samples
                sample_ts = tsfilt.transform(timestamp,len(samples))
                if ppfn: # apply pre-processor
                    samples, sample_ts = ppfn.transform(samples, sample_ts[:,np.newaxis])
                    sample_ts = sample_ts[:,0]
                if len(samples) > 0:
                    data.extend(samples)
                    ts.extend(sample_ts)
    data = np.array(data)
    ts = np.array(ts)
    data = np.append(data,ts[:,np.newaxis],-1)
    # dump as pickle
    import pickle
    if ppfn is None:
        pickle.dump(dict(data=data),open('raw_fph.pk','wb'))
    else:
        pickle.dump(dict(data=data),open('pp_fph.pk','wb'))


if __name__=="__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv)>1 else None

    testcase(filename)

