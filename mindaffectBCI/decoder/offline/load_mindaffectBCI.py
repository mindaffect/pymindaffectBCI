import os
import glob
from mindaffectBCI.utopiaclient import DataHeader, NewTarget
import numpy as np
from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_data_messages
from mindaffectBCI.decoder.devent2stimsequence import devent2stimSequence, upsample_stimseq, devent2stimchannels, devent2markerchannels, stimchannels2markerchannels
from mindaffectBCI.decoder.utils import block_randomize, butter_sosfilt, window_axis, unwrap, askloadsavefile
from mindaffectBCI.decoder.UtopiaDataInterface import butterfilt_and_downsample
from mindaffectBCI.decoder.preprocess import ola_fftfilter

def load_mindaffectBCI(source, datadir:str=None, sessdir:str=None, load_from_cached:bool=True,
                        fs_out:float=10000, filterband=((45,65),(.5,45,'bandpass')), order:int=6, ftype:str='butter',  
                        iti_ms:float=1500, trlen_ms:float=None, offset_ms:float=(-2000,2000), subtriallen_ms:float=None,
                        min_trlen_ms : float = None, max_trlen_percentile:float = .9,
                        zero_before_stimevents:bool=False,  sample2timestamp='lower_bound_tracker',#'robust_timestamp_regression',
                        ch_names=None, verb:int=0, **kwargs):
    """Load and pre-process a mindaffectBCI offline save-file and return the EEG data, and stimulus information

    Args:
        source (str, stream): the source to load the data from, use '-' to load the most recent file from the logs directory.
        fs_out (float, optional): [description]. Defaults to 100.
        filterband (tuple, optional): Specification for a (cascade of) temporal (IIR) filters, in the format used by `mindaffectBCI.decoder.utils.butter_sosfilt`. Defaults to ((45,65),(5.5,25,'bandpass')).
        order (int, optional): the order of the temporal filter. Defaults to 6.
        ftype (str, optional): The type of filter design to use.  One of: 'butter', 'bessel'. Defaults to 'butter'.
        verb (int, optional): General verbosity/logging level. Defaults to 0.
        iti_ms (int, optional): Inter-trial interval. Used to detect trial-transitions when gap between stimuli is greater than this duration. Defaults to 1000.
        trlen_ms (float, optional): Trial duration in milli-seconds.  If None then this is deduced from the stimulus information. Defaults to None.
        min_trlen_ms (float, optional): Minimal acceptable Trial duration in milli-seconds.  If None then this is deduced from the stimulus information. Defaults to None.
        max_trlen_percentile (float, optional): limit the used trial-length to the length of the trial with at this percentile position of the trial length distribution.  Defaults to .9.
        offset_ms (tuple, (2,) optional): Offset in milliseconds from the trial start/end for the returned data such that X has range [tr_start+offset_ms[0] -> tr_end+offset_ms[0]]. Defaults to (-500,500).
        ch_names (tuple, optional): Names for the channels of the EEG data.

    Returns:
        X (np.ndarray (nTrl,nSamp,nCh)): the pre-processed per-trial EEG data
        Y (np.ndarray (nTrl,nSamp,nY)): the up-sampled stimulus information for each output
        coords (list-of-dicts (3,)): dictionary with meta-info for each dimension of X & Y.  As a minimum this contains
                          "name"- name of the dimension, "unit" - the unit of measurment, "coords" - the 'value' of each element along this dimension
                          N.B. coords[3] contains all the non-data or stimulus event messages in the source data file
    """
    # load the raw data as a continuous stream
    X, Me, stim_ts, objIDs, messages, header_events, nt_events, ch_names, source = \
        load_mindaffectBCI_raw(source,sample2timestamp,ch_names=ch_names,verb=verb,load_from_cached=load_from_cached,**kwargs)

    # strip the data time-stamp channel
    data_ts = X[...,-1].astype(np.float64) # (nsamp,)
    data_ts = unwrap(data_ts)
    X = X[...,:-1] # (nsamp,nch)
    
    # estimate the sample rate from the data -- robustly?
    idx = range(0,data_ts.shape[0],1000)
    samp2ms = np.median( np.diff(data_ts[idx])/1000.0 ) 
    fs = 1000.0 / samp2ms
    if not fs_out is None:
        fs_out = min(fs,fs_out)

    if verb > 0: print("X={} @{}Hz".format(X.shape,fs),flush=True)

    # pre-process: spectral filter + downsample
    # add anti-aliasing filter
    if fs_out is not None and fs_out < fs:
        if filterband is None:
            filterband = []
        elif not hasattr(filterband[0],'__iter__'):
            filterband = [filterband]
        else:
            filterband = list(filterband)
        filterband.append((fs_out*.45, -1))

    # incremental call in bits
    ppfn = butterfilt_and_downsample(filterband=filterband, order=order, fs=fs, fs_out=fs_out, ftype=ftype)
    #ppfn = None
    if ppfn is not None:
        if verb > 0:
            print("preFilter: {}th {} {}Hz & downsample {}->{}Hz".format(order,ftype,filterband,fs,fs_out))
        #ppfn.fit(X[0:1,:])
        # process in blocks to be like the on-line, use time-stamp as Y to get revised ts
        if False:
            idxs = np.arange(0,X.shape[0],6); idxs[-1]=X.shape[0]
            Xpp=[]
            tspp=[]
            for i in range(len(idxs)-1):
                idx = range(idxs[i],idxs[i+1])
                Xi, tsi = ppfn.modify(X[idx,:], data_ts[idx,np.newaxis])
                Xpp.append(Xi)
                tspp.append(tsi)
            X=np.concatenate(Xpp,axis=0)
            data_ts = np.concatenate(tspp,axis=0)
        else:
            X, data_ts = ppfn.modify(X, data_ts[:,np.newaxis])
        data_ts = data_ts[...,0] # map back to 1d
        fs  = ppfn.out_fs_ # update with actual re-sampled data rate
        #dur_s = (data_ts[-1] - data_ts[0])/1000.0
        #fs  = X.shape[0] / dur_s

    #import pickle
    #pickle.dump(dict(data=np.append(X,data_ts[:,np.newaxis],-1),stim=np.append(Me,stim_ts[:,np.newaxis],-1)),open('pp_lmbci.pk','wb'))

    if zero_before_stimevents:
        # insert an all-zero stimEvent in the *sample* before each current stim-event
        sampdur_ms = int(np.round(1000/fs))
        Me0 = Me
        stim_ts0 = stim_ts
        Me = np.zeros( (Me.shape[0]*2,Me.shape[1]), dtype=Me.dtype)
        stim_ts = np.zeros( (stim_ts.shape[0]*2,), dtype=stim_ts.dtype)
        for ei in range(stim_ts0.shape[0]):
            Me[ei*2, :]=0
            stim_ts[ei*2] = stim_ts0[ei]-sampdur_ms
            Me[ei*2+1, :] = Me0[ei, :]
            stim_ts[ei*2+1]=stim_ts0[ei]

    # up-sample to stim rate
    Y, stim_samp = upsample_stimseq(data_ts, Me, stim_ts, objIDs)
    Y_ts = np.zeros((Y.shape[0],),dtype=int)
    Y_ts[stim_samp[stim_samp>0]]=stim_ts[stim_samp>0]
    # remove un-used stim info
    stim_ts = stim_ts[stim_samp>0]
    stim_samp = stim_samp[stim_samp>0]
    if verb > 0: print("Y={} @{}Hz".format(Y.shape,fs),flush=True)

    # slice into trials
    if len(nt_events)>1: # use the new-target events to identify new trials
        trl_stim_idx = []
        for nt in nt_events:
            idx = np.flatnonzero(stim_ts > nt.timestamp)[0]
            trl_stim_idx.append(idx)
        # add end of stimulus indicator
        trl_stim_idx.append(len(stim_ts)-1)
        trl_stim_idx=np.array(trl_stim_idx)
    else: 
        # isi = interval *before* every stimulus --
        #  include data-start so includes 1st stimulus
        isi = np.diff(np.concatenate((data_ts[:1],stim_ts,data_ts[-2:]),axis=0))
        #print('isi={}'.format(isi))
        # get trial indices in stimulus messages as sufficiently large inter-stimulus gap
        # N.B. this is the index in stim_ts of the *start* of the new trial
        trl_stim_idx = isi > iti_ms if iti_ms is not None else isi > 15 * max(1,np.median(isi[isi>0]))
        trl_stim_idx = np.flatnonzero(trl_stim_idx)

    # get duration of stimulus in each trial, in milliseconds (rather than number of stimulus events)
    # last-stim in each trial minus first stim in each trial
    # N.B. breaks if trail has only a single stimulus!
    if len(trl_stim_idx)>1:
        if len(stim_ts) > len(trl_stim_idx)*3:
            trl_dur = stim_ts[trl_stim_idx[1:]-1] - stim_ts[trl_stim_idx[:-1]]
        else:
            trl_dur = stim_ts[trl_stim_idx[1:]] - stim_ts[trl_stim_idx[:-1]]
    elif len(stim_ts)>1:
        trl_dur = [stim_ts[-1]-stim_ts[0]] 
    else:
        trl_dur = [fs]
    if verb>1 : print('{} trl_dur (ms) : {}'.format(len(trl_dur),trl_dur))
    if verb>1 : print("{} trl_stim : {}".format(len(trl_stim_idx),[trl_stim_idx[1:]-trl_stim_idx[:-1]]))
    # estimate the best trial-length to use
    if min_trlen_ms is None and len(trl_dur)>0:
        min_trlen_ms = np.sort(trl_dur)[int(len(trl_dur)*max_trlen_percentile)]
    # strip any trial too much shorter than trlen_ms (50%)
    keep = np.flatnonzero(trl_dur>min_trlen_ms*.2)
    if verb>1 : print('Got {} trials, keeping {}'.format(len(trl_stim_idx)-1,len(keep)))
    # re-compute the trlen_ms for the good trials
    trl_stim_idx = trl_stim_idx[keep]
    trl_dur = trl_dur[keep]
    if trlen_ms is None:
        trlen_ms = np.sort(trl_dur)[int(len(trl_dur)*max_trlen_percentile)]

    # get the trial starts as indices & ms into the data array
    trl_samp_idx = stim_samp[trl_stim_idx]
    trl_ts       = stim_ts[trl_stim_idx]
    trl_data_ts  = data_ts[trl_samp_idx]
    if np.max(np.abs(trl_ts-trl_data_ts)) > 20: # sanity check the trial alignment info
        print("Warning: big different between sample and stimulus trial times...")

    if verb>1 : print('{} trl_dur (samp): {}'.format(len(trl_samp_idx),np.diff(trl_samp_idx)))
    if verb>1 : print('{} trl_dur (ms) : {}'.format(len(trl_ts),np.diff(trl_ts)))
    if verb>1 : print('trlen_ms : {}'.format(trlen_ms))

    # compute the trial start/end relative to the trial-start
    trlen_samp  = int(trlen_ms *  fs / 1000)
    offset_samp = [int(o*fs/1000) for o in offset_ms]
    bgnend_samp = (offset_samp[0], trlen_samp+offset_samp[1]) # start end slice window
    xlen_samp = bgnend_samp[1]-bgnend_samp[0]
    
    # extract the slices
    Xraw = X.copy()
    Yraw = Y.copy()
    X = np.zeros((len(trl_samp_idx), xlen_samp, Xraw.shape[-1]), dtype=Xraw.dtype) # (nTrl,nsamp,d)
    Y = np.zeros((len(trl_samp_idx), xlen_samp, Yraw.shape[-1]), dtype=Yraw.dtype)
    Xe_ts = np.zeros((len(trl_samp_idx), xlen_samp),dtype=int)
    Ye_ts = np.zeros((len(trl_samp_idx), xlen_samp),dtype=int)
    ep_idx = np.zeros((len(trl_samp_idx), xlen_samp),dtype=int)
    if verb>0 : 
        print("slicing {} trials =[{} - {}] samples @ {}Hz = [{} - {}] ms".format(len(trl_samp_idx),bgnend_samp[0], bgnend_samp[1],fs,bgnend_samp[0]*1000//fs, bgnend_samp[1]*1000//fs))
    for ti, si in enumerate(trl_samp_idx):
        bgn_idx = si+bgnend_samp[0]
        end_idx_x = min(Xraw.shape[0],si+bgnend_samp[1])
        idx = range(si+bgnend_samp[0],end_idx_x)
        nsamp = len(idx) #min(si+bgnend_samp[1],Xraw.shape[0])-(si+bgnend_samp[0])
        X[ti, :nsamp, :] = Xraw[idx, :]
        Xe_ts[ti,:nsamp] = data_ts[idx]

        # ignore stimuli after end of this trial
        end_idx_y = min(end_idx_x,trl_samp_idx[ti+1]) if ti+1 < len(trl_samp_idx) else end_idx_x
        idx = range(si+bgnend_samp[0],end_idx_y)
        nsamp = len(idx) #min(si+bgnend_samp[1],Xraw.shape[0])-(si+bgnend_samp[0])
        Y[ti, :nsamp, :] = Yraw[idx, :]
        Ye_ts[ti,:nsamp] = Y_ts[idx]
        ep_idx[ti,:nsamp] = list(idx)
        
    del Xraw, Yraw
    if verb > 0: print("X={}\nY={}".format(X.shape,Y.shape))

    nsubtrials = int(X.shape[1]*1000/fs/subtriallen_ms) if subtriallen_ms is not None else 1
    if nsubtrials > 1:
        winsz = int(X.shape[1]//nsubtrials)
        print('{} subtrials -> winsz={}'.format(nsubtrials,winsz))
        # slice into sub-trials
        X = window_axis(X,axis=1,winsz=winsz,step=winsz) # (trl,win,samp,d)
        Y = window_axis(Y,axis=1,winsz=winsz,step=winsz) # (trl,win,samp,nY)
        # concatenate windows into trial dim
        X = X.reshape((X.shape[0]*X.shape[1],)+X.shape[2:])
        Y = Y.reshape((Y.shape[0]*Y.shape[1],)+Y.shape[2:])
        if verb>1 : print("X={}".format(X.shape))
        if verb>1 : print("Y={}".format(Y.shape))

    # ensure the channelnames match the data
    if ch_names is not None:
        eeg_ch_names=ch_names[:X.shape[-1]]  # remove extra names
        ch_names = eeg_ch_names + [ str(i) for i in range(len(eeg_ch_names),X.shape[-1]) ] # add missing channel names

    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = dict(name='trial', coords=trl_ts, trl_idx=ep_idx, trl_ts=Xe_ts, Y_ts=Ye_ts)
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':np.arange(X.shape[1])/fs, \
                 'fs':fs}
    coords[2] = dict(name='channel', coords=ch_names, objIDs=objIDs)
    coords[0]['messages']=messages # save the list of un-processed messages for other users
    # return data + metadata
    return X, Y, coords


def load_mindaffectBCI_raw(source, sample2timestamp='lower_bound_tracker',
                        ch_names=None, load_from_cached:bool=True, verb:int=0, **kwargs):
    if source is None or source == '-':
        # default to last log file if not given
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
    elif source == 'askloadsavefile':
        source = askloadsavefile(initialdir=os.path.dirname(os.path.abspath(__file__)))

    # most recent file in directory
    files = glob.glob(os.path.expanduser(source))
    source = max(files, key=os.path.getctime)

    # check for pickle version if available
    loaded_from_cache=False
    source_pk = os.path.splitext(source)[0] + '.pk'
    if os.path.exists(source_pk) and load_from_cached:
        try:
            import pickle
            with open(source_pk,'rb') as f:
                d = pickle.load(f)
            # extract contents and put into local name-space variables
            X,Me,stim_ts,objIDs,messages,header_events,nt_events = \
                 d['X'],d['Me'],d['stim_ts'],d['objIDs'],d['messages'],d['header_events'],d['nt_events']
            loaded_from_cache=True
        except:
            loaded_from_cache=False
            print("Warning: error loading from cached version")        
    
    if not loaded_from_cache:
        if verb >= 0 and isinstance(source,str): print("Loading {}".format(source))
        # TODO [X]: convert to use the on-line time-stamp code
        X, messages = read_mindaffectBCI_data_messages(source, sample2timestamp=sample2timestamp, **kwargs)

        header_events = [m for m in messages if m.msgID==DataHeader.msgID]
        # get the new target events
        nt_events = [ m for m in messages if m.msgID==NewTarget.msgID ]
        # extract the stimulus sequence
        Me, stim_ts, objIDs, isstimevent = devent2stimSequence(messages)
        stim_ts = unwrap(stim_ts.astype(np.float64)) if len(stim_ts)>0 else stim_ts

        # remove parsed messages from the remaining messages info
        messages = [ m for m,isstim in zip(messages,isstimevent) if not isstim and not m.msgID==NewTarget.msgID and not m.msgID==DataHeader.msgID]

        # save a pickle version for later faster loading
        try:
            import pickle
            with open(source_pk,'wb') as f:
                pickle.dump(dict(X=X,Me=Me,stim_ts=stim_ts,objIDs=objIDs,messages=messages,header_events=header_events,nt_events=nt_events,source=source),f)
        except:
            print("Error saving pickled file.  Ignored.")

    hdr_ch_names=None
    if header_events and len(header_events[-1].labels)>1:
        hdr_ch_names = header_events[-1].labels

    # channel names file if given over-rides
    ch_file = os.path.join(os.path.dirname(source),'ch_names.txt')
    if os.path.exists(ch_file):
        with open(ch_file,'r') as f:
            hdr_ch_names = [l.strip() for l in f.readlines()]
        hdr_ch_names = [ c.strip()  for l in hdr_ch_names for c in l.replace('"','').replace("'",'').replace('[','').replace(']','').split(',')]

    if hdr_ch_names is not None:
        if ch_names is None: # use the hdr-names
            ch_names = hdr_ch_names
            #print("Ch: {}".format(ch_names))

        else: # merge the two sets of channel names
            # hdr_ch_names override if non-numeric
            arg_ch_names = ch_names
            ch_names = [c for c in hdr_ch_names]
            for ci in range(min(len(hdr_ch_names),len(arg_ch_names))):
                hnm = hdr_ch_names[ci]
                try:
                    int(hnm) # try to make an int
                    ch_names[ci]=arg_ch_names[ci]
                except ValueError:
                    pass
    return X, Me,stim_ts,objIDs,messages,header_events,nt_events,ch_names,source



def load_mindaffectBCI_raw_continuous(source, sample2timestamp='lower_bound_tracker',
                        ch_names=None, verb:int=0, **kwargs):

    X, stimSeq, stim_ts, objIDs, messages, header_events, nt_events, ch_names, source = \
        load_mindaffectBCI_raw(source,sample2timestamp,ch_names=ch_names,verb=verb,**kwargs)

    # strip the data time-stamp channel
    data_ts = X[...,-1].astype(np.float64) # (nsamp,)
    data_ts = unwrap(data_ts)
    X = X[...,:-1] # (nsamp,nch)
    
    # estimate the sample rate from the data -- robustly?
    idx = range(0,data_ts.shape[0],1000)
    samp2ms = np.median( np.diff(data_ts[idx])/1000.0 ) 
    fs = 1000.0 / samp2ms

    if verb >= 0: print("X={} @{}Hz".format(X.shape,fs),flush=True)

    # convert from stim-seq at stim=rate to stim-channel at data sample rate
    Y, stim_names = devent2stimchannels(data_ts, stimSeq=stimSeq, stimulus_ts=stim_ts, objIDs=objIDs, nt_events=nt_events)

    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = {'name':'time','unit':'ms', \
                 'coords':data_ts, \
                 'fs':fs}
    coords[1] = dict(name='channel', coords=ch_names)
    
    return X,Y,coords


def load_mindaffectBCI_raw_mne(source, sample2timestamp:str='lower_bound_tracker',
                        ch_names=None, verb:int=0, non_eeg_ch=['acc','eog','ref','status'], **kwargs):
    """load a mindaffectBCI and return in an MNE compatiable raw format with stim_channels with unique integer event IDs

    Args:
        source (str): the file to load the data from
        sample2timestamp (str, optional): function to use mapping from time-stamps to sample numbers. Defaults to 'lower_bound_tracker'.
        ch_names (list-of-str, optional): list of channel names. Defaults to None.
        verb (int, optional): verbosity level. Defaults to 0.

    Returns:
        raw (RawArray): (nsamp, (nch+nstim)) raw MNE array with continous data with EEG channels + stimulus channels
        marker_dict (dict): dict mapping marker UIDs to strings with objID + level info
    """
    X, stimSeq, stim_ts, objIDs, messages, header_events, nt_events, ch_names, source = \
        load_mindaffectBCI_raw(source,sample2timestamp,ch_names=ch_names,verb=verb,**kwargs)

    # strip the data time-stamp channel
    data_ts = X[...,-1].astype(np.float64) # (nsamp,)
    data_ts = unwrap(data_ts)
    X = X[...,:-1] # (nsamp,nch)
    
    # estimate the sample rate from the data -- robustly?
    idx = range(0,data_ts.shape[0],1000)
    samp2ms = np.median( np.diff(data_ts[idx])/1000.0 ) 
    fs = 1000.0 / samp2ms

    if ch_names is None: 
        ch_names = [] 

    if verb >= 0: print("X={} @{}Hz".format(X.shape,fs),flush=True)

    # convert to stimchannels, i.e. @sample_rate channel per-objID
    stim_channels, stim_labels = devent2stimchannels(data_ts, stimSeq=stimSeq, stimulus_ts=stim_ts, objIDs=objIDs, nt_events=nt_events)
    Y, marker_dict = stimchannels2markerchannels(stim_channels, stim_labels)
    marker_labels = stim_labels

    # combine data + stim info, channel_labels, and marker_dict
    data = np.concatenate((X,Y),axis=-1)
    eeg_ch_names=ch_names[:X.shape[-1]]  # remove extra names
    ch_names = eeg_ch_names + [ str(i) for i in range(len(eeg_ch_names),X.shape[-1]) ] # add missing channel names
    ch_names = ch_names + marker_labels # add marker channel names
    ch_types = ['eeg']*len(eeg_ch_names) + ['bio']*(X.shape[-1]-len(eeg_ch_names)) + ['stim']*Y.shape[-1]

    non_eeg = [i for i,c in enumerate(ch_names) if any(t in c.lower() for t in non_eeg_ch)]
    for i in  non_eeg:
        ch_types[i]='bio'

    print(ch_names)
    print(ch_types)

    import mne
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=fs) 
    info.set_montage('standard_1020', on_missing='ignore') #10-20 montage ignoring missing electrodes 

    raw = mne.io.RawArray(data.T,info)
    return raw, marker_dict 


def make_onset_offset_events(data,event_id,output='onset',sep='.'):
    """convert raw event stream to onset/offset events

    Args:
        events ([type]): [description]
        event_id ([type]): [description]
    """    
    import mne
    if output =='onset':
        events = mne.find_events(data,output='onset',verbose=False,consecutive=True)
        event_id = { k+sep+'onset':v for k,v in event_id.items() }

    elif output in 'offset':
        events = mne.find_events(data,output='offset',verbose=False,consecutive=True)
        event_id = { k+sep+"offset":v for k,v in event_id.items() }

    if output in ('both','onset_offset'):
        onset_events = mne.find_events(data,output='onset',verbose=False,consecutive=True)
        onset_event_id = { k+sep+'onset':v for k,v in event_id.items() }
        offset_events = mne.find_events(data,output='offset',verbose=False,consecutive=True)
        offset_event_id = { k+sep+"offset":v+1000 for k,v in event_id.items() }
        # merge into single event stream
        offset_events[:,1:] = offset_events[:,1:]+1000
        onoff_events = np.concatenate((onset_events,offset_events),axis=0)

        # sort the result to increasing time
        idx = np.argsort(onoff_events[:,0])
        events = onoff_events[idx,:]

        event_id = onset_event_id.copy()
        event_id.update(offset_event_id)

    return events, event_id


def make_coords(dim_names=('trial','time','channel'),dim_coords=(None,None,None), fs=None, ch_names=None):
    coords = []
    for i in range(dim_names):
        di = dict(name=dim_names[i], coords=dim_coords[i])
        if di['name'] == 'time' and fs is not None:
            di['fs']=fs
        if di['name'] == 'channel' and ch_names is not None:
            di['coords'] = ch_names
        coords.append(di)
    return coords

def testcase():
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
    from mindaffectBCI.decoder.analyse_datasets import plot_trial
    source = askloadsavefile(initialdir=os.path.dirname(os.path.abspath(__file__)))

    X, Y, coords = load_mindaffectBCI(source, fs_out=100, regress=True, zero_before_stimevents=True, min_trlen_ms=200, verb=2)
    plot_trial(X[0,...],Y[0,...])
    plt.show()


    raw, event_id = load_mindaffectBCI_raw_mne(source)
    print(event_id)
    X, Y, coords = load_mindaffectBCI_raw_continuous(source)
    times = coords[-2]['coords']
    fs = coords[-2]['fs']
    ch_names = coords[-1]['coords']

    plot_trial(X,Y,fs,ch_names,show=True)

    print("X({}){}".format([c['name'] for c in coords],X.shape))
    print("Y={}".format(Y.shape))
    print("fs={}".format(fs))

    # visualize the dataset
    from mindaffectBCI.decoder.analyse_datasets import debug_test_dataset
    debug_test_dataset(X[:,:,:], Y[:,:,:], coords,
                        preprocess_args=dict(badChannelThresh=None, badTrialThresh=3, filterband=None, whiten_spectrum=False, whiten=False),
                        tau_ms=450, evtlabs=('re','fe'), rank=1, model='cca')
    
    
if __name__=="__main__":
    testcase()
