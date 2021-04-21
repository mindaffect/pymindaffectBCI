#!/usr/bin/env python3
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

from mindaffectBCI.utopiaclient import UtopiaClient, Subscribe, StimulusEvent, NewTarget, Selection, DataPacket, UtopiaMessage, SignalQuality
from collections import deque
from mindaffectBCI.decoder.utils import RingBuffer, extract_ringbuffer_segment
from mindaffectBCI.decoder.lower_bound_tracker import lower_bound_tracker
from mindaffectBCI.decoder.linear_trend_tracker import linear_trend_tracker
from time import sleep
import numpy as np

class UtopiaDataInterface:
    """Adaptor class for interfacing between the decoder logic and the data source

    This class provides functionality to wrap a real time data and stimulus stream to make
    it easier to implement standard machine learning pipelines.  In particular it provides streamed
    pre-processing for both EEG and stimulus streams, and ring-buffers for the same with time-stamp based indexing.    
    """


    # TODO [X] : infer valid data time-stamps
    # TODO [X] : smooth and de-jitter the data time-stamps
    # TODO [] : expose a (potentially blocking) message generator interface
    # TODO [X] : ring-buffer for the stimulus-state also, so fast random access
    # TODO [X] : rate limit waiting to reduce computational load
    VERBOSITY = 1

    def __init__(self, datawindow_ms=60000, msgwindow_ms=60000,
                 data_preprocessor=None, stimulus_preprocessor=None, send_signalquality=True, 
                 timeout_ms=100, mintime_ms=50, fs=None, U=None, sample2timestamp='lower_bound_tracker',
                 clientid=None):
        # rate control
        self.timeout_ms = timeout_ms
        self.mintime_ms = mintime_ms # minimum time to spend in update => max processing rate
        # amout of data in the ring-buffer
        self.datawindow_ms = datawindow_ms
        self.msgwindow_ms = msgwindow_ms
        # connect to the mindaffectDecoder
        self.host = None
        self.port = -1
        self.U = UtopiaClient(clientid) if U is None else U
        self.t0 = self.getTimeStamp()
        # init the buffers

        # Messages
        self.msg_ringbuffer = deque()
        self.msg_timestamp = None # ts of most recent processed message

        # DataPackets
        self.data_ringbuffer = None # init later...
        self.data_timestamp = None # ts of last data packet seen
        self.sample2timestamp = sample2timestamp # sample tracker to de-jitter time-stamp information
        self.data_preprocessor = data_preprocessor # function to pre-process the incomming data

        # StimulusEvents
        self.stimulus_ringbuffer = None # init later...
        self.stimulus_timestamp = None # ts of most recent processed data
        self.stimulus_preprocessor = stimulus_preprocessor # function to pre-process the incomming data

        # Info about the data sample rate -- estimated from packet rates..
        self.raw_fs = fs
        self.fs = None
        self.newmsgs = [] # list new unprocssed messages since last update call

        # BODGE: running statistics for sig2noise estimation
        # TODO []: move into it's own Sig2Noise computation class
        self.send_signalquality = send_signalquality
        self.last_sigquality_ts = None
        self.last_log_ts = None
        self.send_sigquality_interval = 1000 # send signal qualities every 1000ms = 1Hz
        # noise2sig estimate halflife_ms, running-offset, de-trended power
        self.noise2sig_halflife_ms = (5000, 500) # 10s for offset, .5s for power
        # TODO [x]: move into a exp-move-ave power est class
        self.raw_power = None
        self.preproc_power = None

    def connect(self, host=None, port=-1, queryifhostnotfound=True):
        """[make a connection to the utopia host]

        Args:
            host ([type], optional): [description]. Defaults to None.
            port (int, optional): [description]. Defaults to -1.
            queryifhostnotfound (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """        
        
        if host:
            self.host = host
        if port > 0:
            self.port = port
        self.U.autoconnect(self.host, self.port, timeout_ms=5000, queryifhostnotfound=queryifhostnotfound)
        if self.U.isConnected:
            # subscribe to messages: data, stim, mode, selection
            self.U.sendMessage(Subscribe(None, "DEMSN"))
        return self.U.isConnected
    
    def isConnected(self):
        """[summary]

        Returns:
            [type]: [description]
        """        

        return self.U.isConnected if self.U is not None else False

    def getTimeStamp(self):
        """[summary]

        Returns:
            [type]: [description]
        """        

        return self.U.getTimeStamp()

    def sendMessage(self, msg: UtopiaMessage):
        """[send a UtopiaMessage to the utopia hub]

        Args:
            msg (UtopiaMessage): [description]
        """        

        self.U.sendMessage(msg)

    def getNewMessages(self, timeout_ms=0):
        """[get new messages from the UtopiaHub]

        Args:
            timeout_ms (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """        
        
        return self.U.getNewMessages(timeout_ms)

    def initDataRingBuffer(self):
        """[initialize the data ring buffer, by getting some seed messages and datapackets to get the data sizes etc.]

        Returns:
            [type]: [description]
        """        
        
        print("geting some initial data to setup the ring buffer")
        # get some initial data to get data shape and sample rate
        databuf = []
        nmsg = 0
        iter = 0
        data_start_ts = None
        data_ts = 0
        while data_start_ts is None or data_ts - data_start_ts < 3000:
            msgs = self.getNewMessages(100)
            for m in msgs:
                m = self.preprocess_message(m)
                if m.msgID == DataPacket.msgID: # data-packets are special
                    if len(m.samples) > 0:
                        databuf.append(m) # append raw data
                        if data_start_ts is None:
                            data_start_ts = m.timestamp
                        data_ts = m.timestamp
                    else:
                        print("Huh? got empty data packet: {}".format(m))
                else:
                    self.msg_ringbuffer.append(m)
                    self.msg_timestamp = m.timestamp
                    nmsg = nmsg+1
        nsamp = [len(m.samples) for m in databuf]
        data_ts = [ m.timestamp for m in databuf]
        if self.raw_fs is None:
            slopes = self.local_slope(nsamp,data_ts)
            self.raw_fs = np.median( slopes ) * 1000.0
        print('Estimated sample rate {} samp in {} s ={}'.format(sum(nsamp),(data_ts[-1]-data_ts[0])/1000.0,self.raw_fs))

        # init the pre-processor (if one)
        if self.data_preprocessor:
            self.data_preprocessor.fit(np.array(databuf[0].samples)[0:1,:], fs=self.raw_fs) # tell it the sample rate

        # apply the data packet pre-processing -- to get the info
        # on the data state after pre-processing
        tmpdatabuf = [self.processDataPacket(m) for m in databuf]
        # strip empty packets
        tmpdatabuf = [d for d in tmpdatabuf if d.shape[0]>0]
        # estimate the sample rate of the pre-processed data
        pp_nsamp = [m.shape[0] for m in tmpdatabuf]
        pp_ts = [ m[-1,-1] for m in tmpdatabuf]
        slopes = self.local_slope(pp_nsamp, pp_ts)
        self.fs = np.median( slopes ) * 1000.0# fs = nSamp/time
        print('Estimated pre-processed sample rate={}'.format(self.fs))

        # create the ring buffer, big enough to store the pre-processed data
        if self.data_ringbuffer:
            print("Warning: re-init data ring buffer")
        # TODO []: why does the datatype of the ring buffer matter so much? Is it because of uss?
        #  Answer[]: it's the time-stamps, float32 rounds time-stamps to 24bits
        self.data_ringbuffer = RingBuffer(maxsize=self.fs*self.datawindow_ms/1000, shape=tmpdatabuf[0].shape[1:], dtype=np.float32)

        # insert the warmup data into the ring buffer
        self.data_timestamp=None # reset last seen data
        nsamp=0
        # re-init the preprocessor for consistency with off-line
        if self.data_preprocessor:
            self.data_preprocessor.fit(np.array(databuf[0].samples)[0:1,:], fs=self.raw_fs)
        # use linear trend tracker to de-jitter the sample timestamps
        if self.sample2timestamp is None or isinstance(self.sample2timestamp,str):
            self.sample2timestamp = timestamp_interpolation(fs=self.fs,
                                                            sample2timestamp=self.sample2timestamp)
        for m in databuf:
            # apply the pre-processing again (this time with fs estimated)
            d = self.processDataPacket(m)
            self.data_ringbuffer.extend(d)
            nsamp = nsamp + d.shape[0]

        return (nsamp, nmsg)

    def local_slope(self,nsamp,data_ts):
        """compute a robust local slope estimate

        Args:
            nsamp ([type]): [description]
            data_ts ([type]): [description]

        Returns:
            [type]: [description]
        """        
        snsamp = np.cumsum(nsamp)
        step = max(1,len(snsamp)//10)
        local_slope = [ (snsamp[i+step] - snsamp[i]) / (data_ts[i+step]-data_ts[i]) for i in range(0,len(snsamp)-step,step//2) ]
        return local_slope

    def initStimulusRingBuffer(self):
        '''initialize the data ring buffer, by getting some seed messages and datapackets to get the data sizes etc.'''
        # TODO []: more efficient memory use, with different dtype for 'real' data and the time-stamps?
        self.stimulus_ringbuffer = RingBuffer(maxsize=self.fs*self.datawindow_ms/1000, shape=(257,), dtype=np.float32)

    def preprocess_message(self, m:UtopiaMessage):
        """[apply pre-processing to topia message before any more work]

        Args:
            m (UtopiaMessage): [description]

        Returns:
            [type]: [description]
        """        
        
        #  WARNING BODGE: fit time-stamp in 24bits for float32 ring buffer
        #  Note: this leads to wrap-arroung in (1<<24)/1000/3600 = 4.6 hours
        #        but that shouldn't matter.....
        m.timestamp = m.timestamp % (1<<24)
        return m
    
    def processDataPacket(self, m: DataPacket):
        """[pre-process a datapacket message ready to be inserted into the ringbuffer]

        Args:
            m (DataPacket): [description]

        Returns:
            [type]: [description]
        """        
        
        #print("DP: {}".format(m))
        # extract the raw data
        d = np.array(m.samples, dtype=np.float32) # process as singles
        # apply the pre-processor, if one was given

        if self.data_preprocessor:
            d_raw = d.copy()
            # warning-- with agressive downsample this may not produce any data!
            d = self.data_preprocessor.transform(d)

            # BODGE: running estimate of the electrode-quality, ONLY after initialization!
            if self.send_signalquality and self.data_ringbuffer is not None:
                self.update_and_send_ElectrodeQualities(d_raw, d, m.timestamp)

                #if self.VERBOSITY > 0 and self.data_ringbuffer is not None:
                #    self.plot_raw_preproc_data(d_raw,d,m.timestamp)

        if d.size > 0 :
            # If have data to add to the ring-buffer, guarding for time-stamp wrap-around
            # TODO [ ]: de-jitter and better timestamp interpolation
            # guard for wrap-around!
            if self.data_timestamp is not None and m.timestamp < self.data_timestamp:
                print("Warning: Time-stamp wrap-around detected!!")

            d = self.add_sample_timestamps(d,m.timestamp,self.fs)

        # update the last time-stamp tracking
        self.data_timestamp= m.timestamp
        return d

    def add_sample_timestamps(self,d:np.ndarray,timestamp:float,fs:float):
        """add per-sample timestamp information to the data matrix

        Args:
            d (np.ndarray): (t,d) the data matrix to attach time stamps to
            timestamp (float): the timestamp of the last sample of d
            fs (float): the nomional sample rate of d

        Returns:
            np.ndarray: (t,d+1) data matrix with attached time-stamp channel
        """
        if self.sample2timestamp is not None and not isinstance(self.sample2timestamp,str):
            sample_ts = self.sample2timestamp.transform(timestamp, len(d))
        else: # all the same ts
            sample_ts = np.ones((len(d),),dtype=int)*timestamp
        # combine data with timestamps, ensuring type is preserved
        d = np.append(np.array(d), sample_ts[:, np.newaxis], -1).astype(d.dtype)
        return d

    def plot_raw_preproc_data(self, d_raw, d_preproc, ts):
        """[debugging function to check the diff between the raw and pre-processed data]

        Args:
            d_raw ([type]): [description]
            d_preproc ([type]): [description]
            ts ([type]): [description]
        """        
        
        if not hasattr(self,'rawringbuffer'):
            self.preprocringbuffer=RingBuffer(maxsize=self.fs*3,shape=(d_preproc.shape[-1]+1,))
            self.rawringbuffer=RingBuffer(maxsize=self.raw_fs*3,shape=(d_raw.shape[-1]+1,))
        d_preproc = self.add_sample_timestamps(d_preproc,ts,self.fs)
        self.preprocringbuffer.extend(d_preproc)
        d_raw = self.add_sample_timestamps(d_raw,ts,self.raw_fs)
        self.rawringbuffer.extend(d_raw)
        if self.last_sigquality_ts is None or ts > self.last_sigquality_ts + self.send_sigquality_interval:
            import matplotlib.pyplot as plt
            plt.figure(10);plt.clf();
            idx = np.flatnonzero(self.rawringbuffer[:,-1])[0]
            plt.subplot(211); plt.cla(); plt.plot(self.rawringbuffer[idx:,-1],self.rawringbuffer[idx:,:-1])
            idx = np.flatnonzero(self.preprocringbuffer[:,-1])[0]
            plt.subplot(212); plt.cla(); plt.plot(self.preprocringbuffer[idx:,-1],self.preprocringbuffer[idx:,:-1])
            plt.show(block=False)


    def processStimulusEvent(self, m: StimulusEvent):
        """[pre-process a StimulusEvent message ready to be inserted into the stimulus ringbuffer]

        Args:
            m (StimulusEvent): [description]

        Returns:
            [type]: [description]
        """        
        
        # get the vector to hold the stimulus info
        d = np.zeros((257,),dtype=np.float32)

        if self.stimulus_ringbuffer is not None and self.stimulus_timestamp is not None:
            # hold value of used objIDs from previous time stamp
            d[:] = self.stimulus_ringbuffer[-1,:]

        # insert the  updated state
        d[m.objIDs] = m.objState
        d[-1] = m.timestamp
        # apply the pre-processor, if one was given
        if self.stimulus_preprocessor:
            d = self.stimulus_preprocessor.transform(d)

        # update the last time-stamp tracking
        self.stimulus_timestamp= m.timestamp
        return d

    def update_and_send_ElectrodeQualities(self, d_raw: np.ndarray, d_preproc: np.ndarray, ts: int):
        """[compute running estimate of electrode qality and stream it]

        Args:
            d_raw (np.ndarray): [description]
            d_preproc (np.ndarray): [description]
            ts (int): [description]
        """        
        
        raw_power, preproc_power = self.update_electrode_powers(d_raw, d_preproc)

        # convert to average amplitude
        raw_amp = np.sqrt(np.maximum(float(1e-8),raw_power)) # guard negatives
        preproc_amp = np.sqrt(np.maximum(float(1e-8),preproc_power)) # guard negatives

        # noise2signal estimated as removed raw amplitude (assumed=noise) to preprocessed amplitude (assumed=signal)
        noise2sig = np.maximum(float(1e-6), np.abs(raw_amp - preproc_amp)) /  np.maximum(float(1e-8),preproc_amp)

        # hack - detect disconnected channels
        noise2sig[ raw_power < 1e-6 ] = 100

        # hack - detect filter artifacts = preproc power is too big..
        noise2sig[ preproc_amp > raw_amp*10 ] = 100

        # hack - cap to 100
        noise2sig = np.minimum(noise2sig,100)

        # rate limit sending of signal-quality messages
        if self.last_sigquality_ts is None or ts > self.last_sigquality_ts + self.send_sigquality_interval:
            print("SigQ:\nraw_power=({}/{})\npp_power=({}/{})\nnoise2sig={}".format(
                   raw_amp,d_raw.shape[0],
                   preproc_amp,d_preproc.shape[0],
                   noise2sig))
            # N.B. use *our* time-stamp for outgoing messages!
            self.sendMessage(SignalQuality(None, noise2sig))
            self.last_sigquality_ts = ts

            if self.VERBOSITY>2:
                # plot the sample time-stamp jitter...
                import matplotlib.pyplot as plt
                plt.figure(10)
                ts = self.data_ringbuffer[:,-1]
                idx = np.flatnonzero(ts)
                if len(idx)>0:
                    ts = ts[idx[0]:]
                    plt.subplot(211); plt.cla(); plt.plot(np.diff(ts)); plt.title('diff time-sample')
                    plt.subplot(212); plt.cla(); plt.plot((ts-ts[0])-np.arange(len(ts))*1000.0/self.fs); plt.title('regression against sample-number')
                    plt.show(block=False)

    def update_electrode_powers(self, d_raw: np.ndarray, d_preproc:np.ndarray):
        """track exp-weighted-moving average centered power for 2 input streams -- the raw and the preprocessed

        Args:
            d_raw (np.ndarray): [description]
            d_preproc (np.ndarray): [description]

        Returns:
            [type]: [description]
        """        
        
        if self.raw_power is None:
            mu_hl, pow_hl = self.noise2sig_halflife_ms
            self.raw_power = power_tracker(mu_hl, pow_hl, self.raw_fs)
            self.preproc_power = power_tracker(mu_hl, pow_hl, self.fs)
        self.raw_power.transform(d_raw)
        self.preproc_power.transform(d_preproc)
        return (self.raw_power.power(), self.preproc_power.power())


    def update(self, timeout_ms=None, mintime_ms=None):
        '''Update the tracking state w.r.t. the data source

        By adding data to the data_ringbuffer, stimulus info to the stimulus_ringbuffer, 
        and other messages to the messages ring buffer.

        Args
         timeout_ms : int
             max block waiting for messages before returning
         mintime_ms : int
             min time to accumulate messages before returning
        Returns
          newmsgs : [newMsgs :UtopiaMessage]
             list of the *new* utopia messages from the server
          nsamp: int
             number of new data samples in this call
             Note: use data_ringbuffer[-nsamp:,...] to get the new data
          nstimulus : int
             number of new stimulus events in this call
             Note: use stimulus_ringbuffer[-nstimulus:,...] to get the new data
        '''
        if timeout_ms is None:
            timeout_ms = self.timeout_ms
        if mintime_ms is None:
            mintime_ms = self.mintime_ms
        if not self.isConnected():
            self.connect()
        if not self.isConnected():
            return [],0,0

        t0 = self.getTimeStamp()
        nsamp = 0
        nmsg = 0
        nstimulus = 0

        if self.data_ringbuffer is None: # do special init stuff if not done
            nsamp, nmsg = self.initDataRingBuffer()
        if self.stimulus_ringbuffer is None: # do special init stuff if not done
            self.initStimulusRingBuffer()
        if self.last_log_ts is None:
            self.last_log_ts = self.getTimeStamp()
        if t0 is None:
            t0 = self.getTimeStamp()

        # record the list of new messages from this call
        newmsgs = self.newmsgs # start with any left-overs from old calls 
        self.newmsgs=[] # clear the  left-over messages stack
        
        ttg = timeout_ms - (self.getTimeStamp() - t0) # time-to-go in the update loop
        while ttg > 0:

            # rate limit
            if ttg >= mintime_ms:
                sleep(mintime_ms/1000.0)
                ttg = timeout_ms - (self.getTimeStamp() - t0) # udate time-to-go
                
            # get the new messages
            msgs = self.getNewMessages(ttg)

            # process the messages - basically to split datapackets from the rest
            print(".",end='')
            #print("{} in {}".format(len(msgs),self.getTimeStamp()-t0),end='',flush=True)
            for m in msgs:
                m = self.preprocess_message(m)
                
                print("{:c}".format(m.msgID), end='', flush=True)
                
                if m.msgID == DataPacket.msgID: # data-packets are special
                    d = self.processDataPacket(m) # (samp x ...)
                    self.data_ringbuffer.extend(d)
                    nsamp = nsamp + d.shape[0]
                    
                elif m.msgID == StimulusEvent.msgID: # as are stmiuluse events
                    d = self.processStimulusEvent(m) # (nY x ...)
                    self.stimulus_ringbuffer.append(d)
                    nstimulus = nstimulus + 1
                    
                else:
                    # NewTarget/Selection are also special in that they clear stimulus state...
                    if m.msgID == NewTarget.msgID or m.msgID == Selection.msgID :
                        # Make a dummy stim-event to reset all objIDs to off
                        d = self.processStimulusEvent(StimulusEvent(m.timestamp,
                                                                    np.arange(255,dtype=np.int32),
                                                                    np.zeros(255,dtype=np.int8)))
                        self.stimulus_ringbuffer.append(d)
                        self.stimulus_timestamp= m.timestamp
                    
                    if len(self.msg_ringbuffer)>0 and m.timestamp > self.msg_ringbuffer[0].timestamp + self.msgwindow_ms: # slide msg buffer
                        self.msg_ringbuffer.popleft()
                    self.msg_ringbuffer.append(m)
                    newmsgs.append(m)
                    nmsg = nmsg+1
                    self.msg_timestamp = m.timestamp
                
            # update time-to-go
            ttg = timeout_ms - (self.getTimeStamp() - t0)

        # new line
        if self.getTimeStamp() > self.last_log_ts + 2000:
            print("",flush=True)
            self.last_log_ts = self.getTimeStamp()
        
        # return new messages, and count new samples/stimulus 
        return (newmsgs, nsamp, nstimulus)



    def push_back_newmsgs(self,oldmsgs):
        """[put unprocessed messages back onto the newmessages queue]

        Args:
            oldmsgs ([type]): [description]
        """        
        
        # TODO []: ensure  this preserves message time-stamp order?
        self.newmsgs.extend(oldmsgs)




    def extract_data_segment(self, bgn_ts, end_ts=None):
        """extract a segment of data based on a start and end time-stamp

        Args:
            bgn_ts (float): segment start time-stamp
            end_ts (float, optional): segment end time-stamp. Defaults to None.

        Returns:
            (np.ndarray): the data between these time-stamps, or None if timestamps invalid
        """        
        return extract_ringbuffer_segment(self.data_ringbuffer,bgn_ts,end_ts)
    
    def extract_stimulus_segment(self, bgn_ts, end_ts=None):
        """extract a segment of the stimulus stream based on a start and end time-stamp

        Args:
            bgn_ts (float): segment start time-stamp
            end_ts (float, optional): segment end time-stamp. Defaults to None.

        Returns:
            (np.ndarray): the stimulus events between these time-stamps, or None if timestamps invalid
        """        
        return extract_ringbuffer_segment(self.stimulus_ringbuffer,bgn_ts,end_ts)
    
    def extract_msgs_segment(self, bgn_ts, end_ts=None):
        """[extract the messages between start/end time stamps]

        Args:
            bgn_ts ([type]): [description]
            end_ts ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        
        msgs = [] # store the trial stimEvents
        for m in reversed(self.msg_ringbuffer):
            if m.timestamp <= bgn_ts:
                # stop as soon as earlier than bgn_ts
                break
            if end_ts is None or m.timestamp < end_ts:
                msgs.append(m)
        # reverse back to input order
        msgs.reverse()
        return msgs

    def run(self, timeout_ms=30000):
        """[test run the interface forever, just getting and storing data]

        Args:
            timeout_ms (int, optional): [description]. Defaults to 30000.
        """        
        
        t0 = self.getTimeStamp()
        # test getting 5s data
        tstart = self.data_timestamp
        trlen_ms = 5000
        while self.getTimeStamp() < t0+timeout_ms:
            self.update()
            # test getting a data segment
            if tstart is None :
                tstart = self.data_timestamp
            if tstart and self.data_timestamp > tstart + trlen_ms:
                X = self.extract_data_segment(tstart, tstart+trlen_ms)
                print("Got data: {}->{}\n{}".format(tstart, tstart+trlen_ms, X[:, -1]))
                Y = self.extract_stimulus_segment(tstart, tstart+trlen_ms)
                print("Got stimulus: {}->{}\n{}".format(tstart, tstart+trlen_ms, Y[:, -1]))
                tstart = self.data_timestamp + 5000
            print('.', flush=True)


try:
    from sklearn.base import TransformerMixin
except:
    # fake the class if sklearn is not available, e.g. Android/iOS
    class TransformerMixin:
        def __init__():
            pass
        def fit(self,X):
            pass
        def transform(self,X):
            pass






#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
from mindaffectBCI.decoder.utils import sosfilt, butter_sosfilt, sosfilt_zi_warmup
class butterfilt_and_downsample(TransformerMixin):
    """Incremental streaming transformer to provide filtering and downsampling data transformations

    Args:
        TransformerMixin ([type]): sklearn compatible transformer
    """    
    def __init__(self, stopband=((0,5),(5,-1)), order:int=6, fs:float =250, fs_out:float =60, ftype='butter'):
        self.stopband = stopband
        self.fs = fs
        self.fs_out = fs_out if fs_out is not None and fs_out < fs else fs
        self.order = order
        self.axis = -2
        if not self.axis == -2:
            raise ValueError("axis != -2 is not yet supported!")
        self.nsamp = 0
        self.ftype = ftype

    def fit(self, X, fs:float =None, zi=None):
        """[summary]

        Args:
            X ([type]): [description]
            fs (float, optional): [description]. Defaults to None.
            zi ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        if fs is not None: # parameter overrides stored fs
            self.fs = fs

        # preprocess -> spectral filter
        if isinstance(self.stopband, str):
            import pickle
            import os
            # load coefficients from file -- when scipy isn't available
            if os.path.isfile(self.stopband):
                fn = self.stopband 
            else: # try relative to our py file
                fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),self.stopband)
            with open(fn,'rb') as f:
                self.sos_ = pickle.load(f)
                self.zi_ = pickle.load(f)
                f.close()
            # tweak the shape/scale of zi to the actual data shape
            self.zi_ = sosfilt_zi_warmup(self.zi_, X, self.axis)
            print("X={} zi={}".format(X.shape,self.zi_.shape))

        else:
            # estimate them from the given information
            X, self.sos_, self.zi_ = butter_sosfilt(X, self.stopband, self.fs, order=self.order, axis=self.axis, zi=zi, ftype=self.ftype)
            
        # preprocess -> downsample
        self.nsamp = 0
        self.resamprate_ = int(round(self.fs*2.0/self.fs_out))/2.0 if self.fs_out is not None else 1
        self.out_fs_  = self.fs/self.resamprate_
        print("resample: {}->{}hz rsrate={}".format(self.fs, self.out_fs_, self.resamprate_))

        return self

    def transform(self, X, Y=None):
        """[summary]

        Args:
            X ([type]): [description]
            Y ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """        
        # propogate the filter coefficients between calls
        if not hasattr(self,'sos_'):
            self.fit(X[0:1,:])

        if self.sos_ is not None:
            X, self.zi_ = sosfilt(self.sos_, X, axis=self.axis, zi=self.zi_)

        nsamp = self.nsamp
        self.nsamp = self.nsamp + X.shape[self.axis] # track *raw* sample counter

        # preprocess -> downsample @60hz
        if self.resamprate_ > 1:
            # number samples through this cycle due to remainder of last block
            resamp_start = nsamp%self.resamprate_
            # convert to number samples needed to complete this cycle
            # this is then the sample to take for the next cycle
            if resamp_start > 0:
                resamp_start = self.resamprate_ - resamp_start
            
            # allow non-integer resample rates
            idx =  np.arange(resamp_start,X.shape[self.axis],self.resamprate_,dtype=X.dtype)

            if self.resamprate_%1 > 0 and idx.size>0 : # non-integer re-sample, interpolate
                idx_l = np.floor(idx).astype(int) # sample above
                idx_u = np.ceil(idx).astype(int) # sample below
                # BODGE: guard for packet ending at sample boundary.
                idx_u[-1] = idx_u[-1] if idx_u[-1]<X.shape[self.axis] else X.shape[self.axis]-1
                w_u   = (idx - idx_l).astype(X.dtype) # linear weight of the upper sample
                X = X[...,idx_u,:] * w_u[:,np.newaxis] + X[...,idx_l,:] * (1-w_u[:,np.newaxis]) # linear interpolation
                if Y is not None:
                    Y = Y[...,idx_u,:] * w_u[:,np.newaxis] + Y[...,idx_l,:] * (1-w_u[:,np.newaxis])

            else:
                idx = idx.astype(int)
                X = X[..., idx, :] # decimate X (trl, samp, d)
                if Y is not None:
                    Y = Y[..., idx, :] # decimate Y (trl, samp, y)
        
        return X if Y is None else (X, Y)

    @staticmethod
    def testcase():
        ''' test the filt+downsample transformation filter by incremental calling '''
        #X=np.cumsum(np.random.randn(100,1),axis=0)
        X=np.sin(np.arange(100)[:,np.newaxis]*2*np.pi/30)
        xs = np.arange(X.shape[0])[:,np.newaxis]
        # high-pass and decimate
        bands = ((0,20,'bandpass'))
        fs = 200
        fs_out = 130
        fds = butterfilt_and_downsample(stopband=bands,fs=fs,fs_out=fs_out)

        
        print("single step")
        fds.fit(X[0:1,:])
        m0,xs0 = fds.transform(X,xs) # (samp,ny,ne)
        print("M0 -> {}".format(m0[:20]))

        step=6
        print("Step size = {}".format(step))
        fds.fit(X[0:1,:])
        m1=np.zeros(m0.shape,m0.dtype)
        xs1 = np.zeros(xs0.shape,xs0.dtype)
        t=0
        for i in range(0,len(X),step):
            idx=np.arange(i,min(i+step,len(X)))
            mm, idx1=fds.transform(X[idx,:],idx[:,np.newaxis])
            m1[t:t+mm.shape[0],:]=mm
            xs1[t:t+mm.shape[0]]=idx1
            t = t +mm.shape[0]
        print("M1 -> {}".format(m1[:20]))
        print("diff: {}".format(np.max(np.abs(m0-m1))))

        import matplotlib.pyplot as plt 
        plt.plot(xs,X,'*-',label='X')
        plt.plot(xs0,m0,'*-',label='{} {}->{}Hz single'.format(bands,fs,fs_out))
        plt.plot(xs1,m1,'*-',label='{} {}->{}Hz incremental'.format(bands,fs,fs_out))
        plt.legend()
        plt.show()





#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
from mindaffectBCI.decoder.stim2event import stim2event
class stim2eventfilt(TransformerMixin):
    ''' Incremental streaming transformer to transform a sequence of stimulus states to a brain event sequence
    
    For example by transforming a sequence of stimulus intensities, to rising and falling edge events.
    '''
    def __init__(self, evtlabs=None, histlen=20):
        self.evtlabs = evtlabs
        self.histlen = histlen
        self.prevX = None

    def fit(self, X):
        """[summary]

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return self

    def transform(self, X):
        """[transform Stimulus-encoded to brain-encoded]

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """        
        
        if X is None:
            return None
        
        # keep old fitler state for the later transformation call
        prevX = self.prevX

        # grab the new filter state (if wanted)
        if self.histlen>0:
            #print('prevX={}'.format(prevX))
            #print("X={}".format(X))
            if X.shape[0] >= self.histlen or prevX is None:
                self.prevX = X
            else:
                self.prevX = np.append(prevX, X, 0)
            # only keep the last bit -- copy in case gets changed in-place
            self.prevX = self.prevX[-self.histlen:,:].copy()
            #print('new_prevX={}'.format(self.prevX))

        # convert from stimulus coding to brain response coding, with old state
        X = stim2event(X, self.evtlabs, axis=-2, oM=prevX)
        return X

    def testcase():
        ''' test the stimulus transformation filter by incremental calling '''
        M=np.array([0,0,0,1,0,0,1,1,0,1])[:,np.newaxis] # samp,nY
        s2ef = stim2eventfilt(evtlabs=('re','fe'),histlen=3)

        print("single step")
        m0=s2ef.transform(M) # (samp,ny,ne)
        print("{} -> {}".format(M,m0))

        print("Step size = 1")
        m1=np.zeros(m0.shape,m0.dtype)
        for i in range(len(M)):
            idx=slice(i,i+1)
            mm=s2ef.transform(M[idx,:])
            m1[idx,...]=mm
            print("{} {} -> {}".format(i,M[idx,...],mm))

        print("Step size=4")
        m4=np.zeros(m0.shape,m0.dtype)
        for i in range(0,len(M),4):
            idx=slice(i,i+4)
            mm=s2ef.transform(M[idx,:])
            m4[idx,...]=mm
            print("{} {} -> {}".format(i,M[idx,...],mm))

        print("m0={}\nm1={}\n,m4={}\n".format(m0,m1,m4))
            




#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
class power_tracker(TransformerMixin):
    """Incremental streaming transformer from raw n-channel data, to exponientially smoothed channel powers

    Args:
        TransformerMixin ([type]): sklearn compatiable transformer
    """

    def __init__(self,halflife_mu_ms, halflife_power_ms, fs, car=True):
        # convert to per-sample decay factor
        self.alpha_mu = self.hl2alpha(fs * halflife_mu_ms / 1000.0 ) 
        self.alpha_power= self.hl2alpha(fs * halflife_power_ms / 1000.0 )
        self.car = car
        self.sX_N = None
        self.sX = None
        self.sXX_N = None
        self.sXX = None

    def hl2alpha(self,hl):
        """[summary]

        Args:
            hl ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return np.exp(np.log(.5)/hl)

    def fit(self,X):
        """[summary]

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.sX_N = X.shape[0]
        if self.car and X.shape[-1]>4:
            X = X.copy() - np.mean(X,-1,keepdims=True)
        self.sX = np.sum(X,axis=0)
        self.sXX_N = X.shape[0]
        self.sXX = np.sum((X-(self.sX/self.sX_N))**2,axis=0)
        return self.power()

    def transform(self, X: np.ndarray):
        """compute the exponientially weighted centered power of X

        Args:
            X (np.ndarray): samples x channels data

        Returns:
            np.ndarray: the updated per-channel power
        """        
        
        if self.sX is None or np.any(np.isnan(self.sX)) or self.sXX is None or np.any(np.isnan(self.sXX)): # not fitted yet!
            return self.fit(X)
        if self.car and X.shape[-1]>4:
            ch_power = self.power()
            # identify the active channels, i.e. are attached and have some signal
            act_ch = ch_power > np.max(ch_power)*1e-3
            X = X.copy() - np.mean(X[...,act_ch], -1, keepdims=True)
        # compute updated mean
        alpha_mu   = self.alpha_mu ** X.shape[0]
        self.sX_N  = self.sX_N*alpha_mu + X.shape[0]
        self.sX    = self.sX*alpha_mu + np.sum(X, axis=0)
        # center and compute updated power
        alpha_pow  = self.alpha_power ** X.shape[0]
        self.sXX_N = self.sXX_N*alpha_pow + X.shape[0]
        self.sXX   = self.sXX*alpha_pow + np.sum((X-(self.sX/self.sX_N))**2, axis=0)
        return self.power()
    
    def mean(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
        return self.sX / self.sX_N
    def power(self):
        """[summary]

        Returns:
            [type]: [description]
        """        
        return self.sXX / self.sXX_N
    
    @staticmethod
    def testcase():
        """[summary]
        """        
        import matplotlib.pyplot as plt
        X = np.random.randn(10000,2)
        X[:500,:] = 0 # start with 0 to induce invalid values
        X[5000:50050,:] = 0 # flatline later to induce invalid values
        #X = np.cumsum(X,axis=0)
        pt = power_tracker(100,100,100)
        print("All at once: power={}".format(pt.transform(X)))  # all at once
        pt = power_tracker(100,1000,1000)
        print("alpha_mu={} alpha_pow={}".format(pt.alpha_mu,pt.alpha_power) )
        step = 30
        idxs = list(range(step,X.shape[0],step))
        powers = np.zeros((len(idxs),X.shape[-1]))
        mus = np.zeros((len(idxs),X.shape[-1]))
        for i,j in enumerate(idxs):
            powers[i,:] = np.sqrt(pt.transform(X[j-step:j,:]))
            mus[i,:]=pt.mean()
        for d in range(X.shape[-1]):
            plt.subplot(X.shape[-1],1,d+1)
            plt.plot(X[:,d])
            plt.plot(idxs,mus[:,d])
            plt.plot(idxs,powers[:,d])
        plt.show(block=True)




#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
class timestamp_interpolation(TransformerMixin):
    """Incremental streaming tranformer to transform from per-packet time-stamps to per-sample timestamps 
    with time-stamp smoothing, de-jittering, and dropped-sample detection.
    """

    def __init__(self,fs=None,sample2timestamp=None, max_delta=200):
        """tranform from per-packet (i.e. multiple-samples) to per-sample timestamps

        Args:
            fs (float): default sample rate, used when no other timing info is available
            sample2timestamp (transformer, optional): class to de-jitter timestamps based on sample-count. Defaults to None.
        """        
        self.fs=fs
        a0 = 1000/self.fs if self.fs is not None else 1
         # BODGE: special cases for particular mapping functions so can include the prior slope
        if sample2timestamp=='lower_bound_tracker':
            self.sample2timestamp = lower_bound_tracker(a0=a0)
        elif sample2timestamp=='linear_trend_tracker':
            self.sample2timestamp = linear_trend_tracker(a0=a0)
        else:
            self.sample2timestamp = sample2timestamp
        self.max_delta = max_delta

    def fit(self,ts,nsamp=1):
        """[summary]

        Args:
            ts ([type]): [description]
            nsamp (int, optional): [description]. Defaults to 1.
        """        
        self.last_sample_timestamp_ = ts
        self.n_ = 0

    def transform(self,timestamp:float,nsamp:int=1):
        """add per-sample timestamp information to the data matrix

        Args:
            timestamp (float): the timestamp of the last sample of d
            nsamp(int): number of samples to interpolate

        Returns:
            np.ndarray: (nsamp) the interpolated time-stamps
        """
        if not hasattr(self,'last_sample_timestamp_'):
            self.fit(timestamp,nsamp)

        # update tracking number samples processed
        self.n_ = self.n_ + nsamp

        if self.last_sample_timestamp_ < timestamp or self.sample2timestamp is not None:
            # update the tracker for the sample-number to sample timestamp mapping
            if self.sample2timestamp is not None:
                #print("n={} ts={}".format(n,timestamp))
                newtimestamp = self.sample2timestamp.transform(self.n_, timestamp)
                #print("ts={} newts={} diff={}".format(timestamp,newtimestamp,timestamp-newtimestamp))
                # use the corrected de-jittered time-stamp -- if it's not tooo different
                if abs(timestamp-newtimestamp) < self.max_delta:
                    timestamp = int(newtimestamp)

            # simple linear interpolation for the sample time-stamps
            samples_ts = np.linspace(self.last_sample_timestamp_, timestamp, nsamp+1, endpoint=True, dtype=int)
            samples_ts = samples_ts[1:]
        else:
            if self.fs :
                # interpolate with the estimated sample rate                    
                samples_ts = np.arange(-nsamp+1,1,dtype=int)*(1000/self.fs) + timestamp
            else:
                # give all same timestamp
                samples_ts = np.ones(nsamp,dtype=int)*timestamp

        # update the tracking info
        self.last_sample_timestamp_ = timestamp

        return samples_ts

    def testcase(self, npkt=1000, fs=100):
        """[summary]

        Args:
            npkt (int, optional): [description]. Defaults to 1000.
            fs (int, optional): [description]. Defaults to 100.
        """        
        # generate random packet sizes
        nsamp = np.random.random_integers(0,10,size=(npkt,))
        # generate true sample timestamps
        ts_true = np.arange(np.sum(nsamp))*1000/fs
        # packet end indices
        idx = np.cumsum(nsamp)-1
        # packet end time-stamps
        pkt_ts = ts_true[idx]
        # add some time-stamp jitter, always positive..
        pkt_ts = pkt_ts + np.random.uniform(0,.5*1000/fs,size=pkt_ts.shape)
        # apply the time-stamp interplotation
        sts=[]
        tsfn = timestamp_interpolation(fs=fs,sample2timestamp = 'lower_bound_tracker')
        for i,(n,t) in enumerate(zip(nsamp,pkt_ts)):
            samp_ts = tsfn.transform(t,n)
            sts.extend(samp_ts)
        # plot the result.
        import matplotlib.pyplot as plt
        plt.plot(ts_true - sts)
        plt.show()


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
from mindaffectBCI.decoder.preprocess import temporally_decorrelate
class temporal_decorrelator(TransformerMixin):
    """Incremental streaming tranformer to decorrelate temporally channels in an input stream
    """

    def __init__(self, order=10, reg=1e-4, eta=1e-5, axis=-2):
        self.reg=reg
        self.eta=eta
        self.axis=axis

    def fit(self,X):
        """[summary]

        Args:
            X ([type]): [description]
        """        
        self.W_ = np.zeros((self.order,X.shape[-1]),dtype=X.dtype)
        self.W_[-1,:]=1
        _, self.W_ = self.transform(X[1:,:])

    def transform(self,X):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate
            nsamp(int): number of samples to interpolate

        Returns:
            np.ndarray: the decorrelated data
        """
        if not hasattr(self,'W_'):
            self.fit(X)

        X, self.W_ = temporally_decorrelate(X, W=self.W_, reg=self.reg, eta=self.eta, axis=self.axis)

        return X

    def testcase(self, dur=3, fs=100, blksize=10):
        """[summary]

        Args:
            dur (int, optional): [description]. Defaults to 3.
            fs (int, optional): [description]. Defaults to 100.
            blksize (int, optional): [description]. Defaults to 10.
        """        
        import numpy as np
        import matplotlib.pyplot as plt
        from mindaffectBCI.decoder.preprocess import plot_grand_average_spectrum
        fs=100
        X = np.random.standard_normal((2,fs*dur,2)) # flat spectrum
        #X = X + np.sin(np.arange(X.shape[-2])*2*np.pi/10)[:,np.newaxis]
        X = X[:,:-1,:]+X[:,1:,:] # weak low-pass

        #X = np.cumsum(X,-2) # 1/f spectrum
        print("X={}".format(X.shape))
        plt.figure(1)
        plot_grand_average_spectrum(X, fs)
        plt.suptitle('Raw')
        plt.show(block=False)

        tdc = temporal_decorrelator()
        wX = np.zeros(X.shape,X.dtype)
        for i in range(0,X.shape[-1],blksize):
            idx = range(i,i+blksize)
            wX[idx,:] = tdc.transform(X[idx,:])
        
        # compare raw vs summed filterbank
        plt.figure(2)
        plot_grand_average_spectrum(wX,fs)
        plt.suptitle('Decorrelated')
        plt.show()


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
from mindaffectBCI.decoder.preprocess import standardize_channel_power
class channel_power_standardizer(TransformerMixin):
    """Incremental streaming tranformer to channel power normalization in an input stream
    """

    def __init__(self, reg=1e-4, axis=-2):
        self.reg=reg
        self.axis=axis

    def fit(self,X):
        """[summary]

        Args:
            X ([type]): [description]
        """        
        self.sigma2_ = np.zeros((X.shape[-1],), dtype=X.dtype)
        self.sigma2_ = X[0,:]*X[0,:] # warmup with 1st sample power
        self.transform(X[1:,:])

    def transform(self,X):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate

        Returns:
            np.ndarray: the decorrelated data
        """
        if not hasattr(self,'sigma2_'):
            self.fit(X)

        X, self.W_ = standardize_channel_power(X, sigma2=self.sigma2_, reg=self.reg, axis=self.axis)

        return X

    def testcase(self, dur=3, fs=100, blksize=10):
        """[summary]

        Args:
            dur (int, optional): [description]. Defaults to 3.
            fs (int, optional): [description]. Defaults to 100.
            blksize (int, optional): [description]. Defaults to 10.
        """        
        import numpy as np
        import matplotlib.pyplot as plt
        from mindaffectBCI.decoder.preprocess import plot_grand_average_spectrum
        fs=100
        X = np.random.standard_normal((2,fs*dur,2)) # flat spectrum
        #X = X + np.sin(np.arange(X.shape[-2])*2*np.pi/10)[:,np.newaxis]
        X = X[:,:-1,:]+X[:,1:,:] # weak low-pass

            #X = np.cumsum(X,-2) # 1/f spectrum
        print("X={}".format(X.shape))
        plt.figure(1)
        plot_grand_average_spectrum(X, fs)
        plt.suptitle('Raw')
        plt.show(block=False)

        cps = channel_power_standardizer()
        wX = np.zeros(X.shape,X.dtype)
        for i in range(0,X.shape[-1],blksize):
            idx = range(i,i+blksize)
            wX[idx,:] = cps.transform(X[idx,:])
        
        # compare raw vs summed filterbank
        plt.figure(2)
        plot_grand_average_spectrum(wX,fs)
        plt.suptitle('Decorrelated')
        plt.show()


def testRaw():
    """[summary]
    """    
    # test with raw
    ui = UtopiaDataInterface()
    ui.connect()
    sigViewer(ui,30000) # 30s sigviewer

def testPP():
    """[summary]
    """    
    from sigViewer import sigViewer
    # test with a filter + downsampler
    ppfn= butterfilt_and_downsample(order=4, stopband=((0,1),(25,-1)), fs_out=100)
    #ppfn= butterfilt_and_downsample(order=4, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=80)     
    ui = UtopiaDataInterface(data_preprocessor=ppfn, stimulus_preprocessor=None)
    ui.connect()
    sigViewer(ui)

def testFileProxy(filename,fs_out=999):
    """[summary]

    Args:
        filename ([type]): [description]
        fs_out (int, optional): [description]. Defaults to 999.
    """    
    from mindaffectBCI.decoder.FileProxyHub import FileProxyHub
    U = FileProxyHub(filename)
    from sigViewer import sigViewer
    # test with a filter + downsampler
    #ppfn= butterfilt_and_downsample(order=4, stopband=((0,3),(25,-1)), fs_out=fs_out)
    ppfn= butterfilt_and_downsample(order=4, stopband=(1,15,'bandpass'), fs_out=fs_out)
    #ppfn = None
    ui = UtopiaDataInterface(data_preprocessor=ppfn, stimulus_preprocessor=None, mintime_ms=0, U=U)
    ui.connect()
    sigViewer(ui)

def testFileProxy2(filename):
    """[summary]

    Args:
        filename ([type]): [description]
    """    
    from mindaffectBCI.decoder.FileProxyHub import FileProxyHub
    U = FileProxyHub(filename)
    fs = 200
    fs_out = 200
    # test with a filter + downsampler
    ppfn= butterfilt_and_downsample(order=4, stopband=((45,65),(0,3),(25,-1)), fs=fs, fs_out=fs_out)
    ui = UtopiaDataInterface(data_preprocessor=ppfn, stimulus_preprocessor=None, mintime_ms=0, U=U, fs=fs)
    ui.connect()
    # run in bits..
    data=[]
    stim=[]
    emptycount = 0 
    while True:
        newmsg, nsamp, nstim = ui.update()
        if len(newmsg) == 0 and nsamp == 0 and nstim == 0: 
            emptycount = emptycount + 1
            if emptycount > 10:
                break
        else:
            emptycount=0
        if nsamp > 0:
            data.append(ui.data_ringbuffer[-nsamp:,:].copy())
        if nstim > 0:
            stim.append(ui.stimulus_ringbuffer[-nstim:,:].copy())
    # convert to single data block
    data = np.vstack(data)
    stim = np.vstack(stim)
    # dump as pickle
    import pickle
    if ppfn is None:
        pickle.dump(dict(data=data,stim=stim),open('raw_udi.pk','wb'))
    else:
        pickle.dump(dict(data=data,stim=stim),open('pp_udi.pk','wb'))

def testERP():
    """[summary]
    """    
    ui = UtopiaDataInterface()
    ui.connect()
    erpViewer(ui,evtlabs=None) # 30s sigviewer

def testElectrodeQualities(X,fs=200,pktsize=20):
    """[summary]

    Args:
        X ([type]): [description]
        fs (int, optional): [description]. Defaults to 200.
        pktsize (int, optional): [description]. Defaults to 20.

    Returns:
        [type]: [description]
    """    
    # recurse if more dims than we want...
    if X.ndim>2:
        sigq=[]
        for i in range(X.shape[0]):
            sigqi = testElectrodeQualities(X[i,...],fs,pktsize)
            sigq.append(sigqi)
        sigq=np.concatenate(sigq,0)
        return sigq
    
    ppfn= butterfilt_and_downsample(order=6, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=100)
    ppfn.fit(X[:10,:],fs=200)
    noise2sig = np.zeros((int(X.shape[0]/pktsize),X.shape[-1]),dtype=np.float32)
    for pkti in range(noise2sig.shape[0]):
        t = pkti*pktsize
        Xi = X[t:t+pktsize,:]
        Xip = ppfn.transform(Xi)
        raw_power, preproc_power = update_electrode_powers(Xi,Xip)
        noise2sig[pkti,:] = np.maximum(float(1e-6), (raw_power - preproc_power)) /  np.maximum(float(1e-8),preproc_power)
    return noise2sig

    
if __name__ == "__main__":
    power_tracker.testcase()
    quit()

    #timestamp_interpolation().testcase()
    #butterfilt_and_downsample.testcase()
    #testRaw()
    #testPP()
    #testERP()
    filename="~/Desktop/mark/mindaffectBCI_*.txt"
    testFileProxy(filename)
    #testFileProxy2(filename)
    # "C:\\Users\\Developer\\Downloads\\mark\\mindaffectBCI_brainflow_200911_1229_90cal.txt")
    #"..\..\Downloads\khash\mindaffectBCI_noisetag_bci_200907_1433.txt"
