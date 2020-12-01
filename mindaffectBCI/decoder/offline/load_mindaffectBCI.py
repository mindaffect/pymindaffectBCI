import os
import glob
import numpy as np
from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_data_messages
from mindaffectBCI.decoder.devent2stimsequence import devent2stimSequence, upsample_stimseq
from mindaffectBCI.decoder.utils import block_randomize, butter_sosfilt, upsample_codebook, lab2ind, window_axis, unwrap
from mindaffectBCI.decoder.UtopiaDataInterface import butterfilt_and_downsample

def load_mindaffectBCI(source, datadir:str=None, sessdir:str=None, fs_out:float=100, stopband=((45,65),(5.5,25,'bandpass')), order:int=6, ftype:str='butter', verb:int=0, iti_ms:float=1000, trlen_ms:float=None, offset_ms:float=(-500,500), ch_names=None, **kwargs):
    """Load and pre-process a mindaffectBCI offline save-file and return the EEG data, and stimulus information

    Args:
        source (str, stream): the source to load the data from, use '-' to load the most recent file from the logs directory.
        fs_out (float, optional): [description]. Defaults to 100.
        stopband (tuple, optional): Specification for a (cascade of) temporal (IIR) filters, in the format used by `mindaffectBCI.decoder.utils.butter_sosfilt`. Defaults to ((45,65),(5.5,25,'bandpass')).
        order (int, optional): the order of the temporal filter. Defaults to 6.
        ftype (str, optional): The type of filter design to use.  One of: 'butter', 'bessel'. Defaults to 'butter'.
        verb (int, optional): General verbosity/logging level. Defaults to 0.
        iti_ms (int, optional): Inter-trial interval. Used to detect trial-transitions when gap between stimuli is greater than this duration. Defaults to 1000.
        trlen_ms (float, optional): Trial duration in milli-seconds.  If None then this is deduced from the stimulus information. Defaults to None.
        offset_ms (tuple, (2,) optional): Offset in milliseconds from the trial start/end for the returned data such that X has range [tr_start+offset_ms[0] -> tr_end+offset_ms[0]]. Defaults to (-500,500).
        ch_names (tuple, optional): Names for the channels of the EEG data.

    Returns:
        X (np.ndarray (nTrl,nSamp,nCh)): the pre-processed per-trial EEG data
        Y (np.ndarray (nTrl,nSamp,nY)): the up-sampled stimulus information for each output
        coords (list-of-dicts (3,)): dictionary with meta-info for each dimension of X & Y.  As a minimum this contains
                          "name"- name of the dimension, "unit" - the unit of measurment, "coords" - the 'value' of each element along this dimension
    """    
    if source is None or source == '-':
        # default to last log file if not given
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
        source= max(files, key=os.path.getctime)

    if isinstance(source,str):
        if sessdir:
            source = os.path.join(sessdir, source)
        if datadir:
            source = os.path.join(datadir, source)
        files = glob.glob(os.path.expanduser(source))
        source = max(files, key=os.path.getctime)

    if verb >= 0 and isinstance(source,str): print("Loading {}".format(source))
    # TODO []: convert to use the on-line time-stamp code
    X, messages = read_mindaffectBCI_data_messages(source, **kwargs)

    # TODO[]: get the header if there is one?

    import pickle
    pickle.dump(dict(data=X),open('raw_lmbci.pk','wb'))

    # strip the data time-stamp channel
    data_ts = X[...,-1].astype(np.float64) # (nsamp,)
    data_ts = unwrap(data_ts)
    X = X[...,:-1] # (nsamp,nch)
    
    # estimate the sample rate from the data -- robustly?
    idx = range(0,data_ts.shape[0],1000)
    samp2ms = np.median( np.diff(data_ts[idx])/1000.0 ) 
    fs = 1000.0 / samp2ms
    ch_names = ch_names

    if verb >= 0: print("X={} @{}Hz".format(X.shape,fs),flush=True)

    # pre-process: spectral filter + downsample
    # incremental call in bits
    ppfn = butterfilt_and_downsample(stopband=stopband, order=order, fs=fs, fs_out=fs_out, ftype=ftype)
    #ppfn = None
    if ppfn is not None:
        if verb >= 0:
            print("preFilter: {}th {} {}Hz & downsample {}->{}Hz".format(order,ftype,stopband,fs,fs_out))
        #ppfn.fit(X[0:1,:])
        # process in blocks to be like the on-line, use time-stamp as Y to get revised ts
        if False:
            idxs = np.arange(0,X.shape[0],6); idxs[-1]=X.shape[0]
            Xpp=[]
            tspp=[]
            for i in range(len(idxs)-1):
                idx = range(idxs[i],idxs[i+1])
                Xi, tsi = ppfn.transform(X[idx,:], data_ts[idx,np.newaxis])
                Xpp.append(Xi)
                tspp.append(tsi)
            X=np.concatenate(Xpp,axis=0)
            data_ts = np.concatenate(tspp,axis=0)
        else:
            X, data_ts = ppfn.transform(X, data_ts[:,np.newaxis])
        data_ts = data_ts[...,0] # map back to 1d
        fs  = ppfn.out_fs_ # update with actual re-sampled data rate
        #dur_s = (data_ts[-1] - data_ts[0])/1000.0
        #fs  = X.shape[0] / dur_s


    # extract the stimulus sequence
    Me, stim_ts, objIDs, _ = devent2stimSequence(messages)
    stim_ts = unwrap(stim_ts.astype(np.float64))

    import pickle
    pickle.dump(dict(data=np.append(X,data_ts[:,np.newaxis],-1),stim=np.append(Me,stim_ts[:,np.newaxis],-1)),open('pp_lmbci.pk','wb'))

    # up-sample to stim rate
    Y, stim_samp = upsample_stimseq(data_ts, Me, stim_ts, objIDs)
    Y_ts = np.zeros((Y.shape[0],),dtype=int); 
    Y_ts[stim_samp]=stim_ts
    if verb >= 0: print("Y={} @{}Hz".format(Y.shape,fs),flush=True)

    # slice into trials
    # isi = interval *before* every stimulus --
    #  include data-start so includes 1st stimulus
    isi = np.diff(np.concatenate((data_ts[0:1],stim_ts,data_ts[-2:-1]),axis=0))
    #print('isi={}'.format(isi))
    # get trial indices in stimulus messages as sufficiently large inter-stimulus gap
    # N.B. this is the index in stim_ts of the *start* of the new trial
    trl_stim_idx = np.flatnonzero(isi > iti_ms)
    # get duration of stimulus in each trial, in milliseconds (rather than number of stimulus events)
    trl_dur = stim_ts[trl_stim_idx[1:]-1] - stim_ts[trl_stim_idx[:-1]]
    print('{} trl_dur (ms) : {}'.format(len(trl_dur),np.diff(trl_dur)))
    # estimate the best trial-length to use
    if trlen_ms is None:
        trlen_ms = np.percentile(trl_dur,90)
    # strip any trial too much shorter than trlen_ms (50%)
    keep = np.flatnonzero(trl_dur>trlen_ms*.2)
    print('Got {} trials, keeping {}'.format(len(trl_stim_idx)-1,len(keep)))
    # re-compute the trlen_ms for the good trials
    trl_stim_idx = trl_stim_idx[keep]

    # get the trial starts as indices & ms into the data array
    trl_samp_idx = stim_samp[trl_stim_idx]
    trl_ts       = stim_ts[trl_stim_idx]   

    print('{} trl_dur (samp): {}'.format(len(trl_samp_idx),np.diff(trl_samp_idx)))
    print('{} trl_dur (ms) : {}'.format(len(trl_ts),np.diff(trl_ts)))

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
    print("slicing {} trials =[{} - {}] samples @ {}Hz".format(len(trl_samp_idx),bgnend_samp[0], bgnend_samp[1],fs))
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

    import pickle
    pickle.dump(dict(X=X,Y=Y,X_ts=Xe_ts,Y_ts=Ye_ts,ep_idx=ep_idx),open('sliced_lmbci.pk','wb'))


    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = dict(name='trial', coords=trl_ts, trl_idx=ep_idx, trl_ts=Xe_ts, Y_ts=Ye_ts)
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':np.arange(X.shape[1])/fs, \
                 'fs':fs}
    coords[2] = dict(name='channel', coords=ch_names)
    # return data + metadata
    return (X, Y, coords)

def testcase():
    import sys

    if len(sys.argv)>1:
        sessfn = sys.argv[1]
    else:
        # default to last log file if not given
        import glob
        import os
        fileregexp = '../../../logs/mindaffectBCI*.txt'
        #fileregexp = '../../../../utopia/java/utopia2ft/UtopiaMessages*.log'
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),fileregexp)) # * means all if need specific format then *.csv
        sessfn = max(files, key=os.path.getctime)
 
    #sessfn = "C:\\Users\\Developer\\Downloads\\mark\\mindaffectBCI_brainflow_200911_1229_90cal.txt"
    from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
    print("Loading: {}".format(sessfn))
    X, Y, coords = load_mindaffectBCI(sessfn, fs_out=100, regress=False)
    times = coords[1]['coords']
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']

    print("X({}){}".format([c['name'] for c in coords],X.shape))
    print("Y={}".format(Y.shape))
    print("fs={}".format(fs))

    # visualize the dataset
    from mindaffectBCI.decoder.analyse_datasets import debug_test_dataset
    debug_test_dataset(X[:,:,:], Y[:,:,:], coords,
                        preprocess_args=dict(badChannelThresh=None, badTrialThresh=3, stopband=None, whiten_spectrum=False, whiten=False),
                        tau_ms=450, evtlabs=('re','fe'), rank=1, model='cca')
    
    
if __name__=="__main__":
    testcase()
