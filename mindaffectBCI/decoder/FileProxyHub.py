#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
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

from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_message
from mindaffectBCI.utopiaclient import DataPacket
from time import sleep

class FileProxyHub:
    ''' Proxy UtopiaClient which gets messages from a saved log file '''
    def __init__(self, filename:str=None, speedup:float=None, use_server_ts:bool=True):
        self.filename = filename
        import glob
        import os
        if self.filename is None or self.filename == '-':
            # default to last log file if not given
            files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
            self.filename = max(files, key=os.path.getctime)
        else:
            files = glob.glob(os.path.expanduser(filename))
            self.filename = max(files, key=os.path.getctime)
        print("Loading : {}\n".format(self.filename))
        self.speedup = speedup
        self.isConnected = True
        self.lasttimestamp = None
        self.use_server_ts = use_server_ts
        self.file = open(self.filename,'r')
    
    def getTimeStamp(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
        return self.lasttimestamp
    
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
            sleep(timeout_ms/1000./self.speedup)
        # update the time-stamp cursor
        self.lasttimestamp = self.lasttimestamp + timeout_ms
        return msgs

def testcase(filename, fs=200, fs_out=200, stopband=((45,65),(0,3),(25,-1)), order=4):
    """[summary]

    Args:
        filename ([type]): [description]
        fs (int, optional): [description]. Defaults to 200.
        fs_out (int, optional): [description]. Defaults to 200.
        stopband (tuple, optional): [description]. Defaults to ((45,65),(0,3),(25,-1)).
        order (int, optional): [description]. Defaults to 4.
    """    
    import numpy as np
    from mindaffectBCI.decoder.UtopiaDataInterface import timestamp_interpolation, linear_trend_tracker, butterfilt_and_downsample

    U = FileProxyHub(filename)
    tsfilt = timestamp_interpolation(fs=fs,sample2timestamp=linear_trend_tracker(500))

    if stopband is not None:
        ppfn = butterfilt_and_downsample(stopband=stopband, order=order, fs=fs, fs_out=fs_out)
    else:
        ppfn = None
    
    #ppfn = None

    nsamp=0
    t=0
    data=[]
    ts=[]
    while U.isConnected:
        msgs = U.getNewMessages(100)
        print('.',end='',flush=True)
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

    filename = "C:\\Users\\Developer\\Downloads\\mark\\mindaffectBCI_brainflow_200911_1229_90cal.txt"

    testcase(filename)

