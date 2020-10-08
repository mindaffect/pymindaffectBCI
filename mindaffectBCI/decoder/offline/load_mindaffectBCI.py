import os
import numpy as np
from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_data_messages
from mindaffectBCI.decoder.devent2stimsequence import devent2stimSequence, upsample_stimseq
from mindaffectBCI.decoder.utils import block_randomize, butter_sosfilt, upsample_codebook, lab2ind, window_axis, unwrap
from mindaffectBCI.decoder.UtopiaDataInterface import butterfilt_and_downsample

def load_mindaffectBCI(datadir, sessdir=None, sessfn=None, fs_out=100, stopband=((45,65),(5.5,25,'bandpass')), order=6, ftype='butter', verb=0, iti_ms=1000, trlen_ms=None, offset_ms=(-500,500), **kwargs):
    
    # load the data file
    Xfn = datadir
    if sessdir:
        Xfn = os.path.join(Xfn, sessdir)
    if sessfn:
        Xfn = os.path.join(Xfn, sessfn)
    sessdir = os.path.dirname(Xfn)

    if verb >= 0: print("Loading {}".format(Xfn))
    # TODO []: convert to use the on-line time-stamp code
    X, messages = read_mindaffectBCI_data_messages(Xfn, **kwargs)

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
    ch_names = None

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
    if verb > 0: print("Y={} @{}Hz".format(Y.shape,fs),flush=True)

    # slice into trials
    # isi = interval *before* every stimulus --
    #  include data-start so includes 1st stimulus
    isi = np.diff(np.concatenate((data_ts[0:1],stim_ts,data_ts[-2:-1]),axis=0))
    #print('isi={}'.format(isi))
    # get trial indices in stimulus messages as sufficiently large inter-stimulus gap
    # N.B. this is the index in stim_ts of the *start* of the new trial
    trl_stim_idx = np.flatnonzero(isi > iti_ms)
    # get duration of stimulus in each trial
    trl_dur = stim_ts[trl_stim_idx[1:]-1] - stim_ts[trl_stim_idx[:-1]]
    print('trl_dur: {}'.format(trl_dur))
    # estimate the best trial-length to use
    if trlen_ms is None:
        trlen_ms = np.median(trl_dur)
    # strip any trial too much shorter than trlen_ms (50%)
    keep = np.flatnonzero(trl_dur>trlen_ms*.5)
    # re-compute the trlen_ms for the good trials
    trl_stim_idx = trl_stim_idx[keep]
    
    # compute the trial start/end relative to the trial-start
    trlen_samp  = int(trlen_ms *  fs / 1000)
    offset_samp = [int(o*fs/1000) for o in offset_ms]
    bgnend_samp = (offset_samp[0], trlen_samp+offset_samp[1]) # start end slice window
    xlen_samp = bgnend_samp[1]-bgnend_samp[0]
    
    # get the trial starts as indices & ms into the data array
    trl_samp_idx = stim_samp[trl_stim_idx]
    trl_ts       = stim_ts[trl_stim_idx]
    
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
        idx = range(si+bgnend_samp[0],min(Xraw.shape[0],si+bgnend_samp[1]))
        nsamp = len(idx) #min(si+bgnend_samp[1],Xraw.shape[0])-(si+bgnend_samp[0])
        X[ti, :nsamp, :] = Xraw[idx, :]
        Xe_ts[ti,:nsamp] = data_ts[idx]
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
