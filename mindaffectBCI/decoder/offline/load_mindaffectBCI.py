import os
import numpy as np
from .read_mindaffectBCI import read_mindaffectBCI_data_messages
from devent2stimsequence import devent2stimSequence, upsample_stimseq
from utils import block_randomize, butter_sosfilt, upsample_codebook, lab2ind, window_axis

def load_mindaffectBCI(datadir, sessdir=None, sessfn=None, ofs=60, stopband=((0,1),(25,-1)), order=6, verb=0, iti_ms=1000, trlen_ms=None, offset_ms=(-1000,1000)):
    
    # load the data file
    Xfn = datadir
    if sessdir:
        Xfn = os.path.join(Xfn, sessdir)
    if sessfn:
        Xfn = os.path.join(Xfn, sessfn)
    sessdir = os.path.dirname(Xfn)

    if verb > 1: print("Loading {}".format(Xfn))
    X, messages =read_mindaffectBCI_data_messages(Xfn)

    # strip the data time-stamp channel
    data_ts = X[...,-1] # (nsamp,)
    X = X[...,:-1] # (nsamp,nch)
    
    # estimate the sample rate from the data.
    dur_s = (data_ts[-1] - data_ts[0])/1000.0
    fs  = X.shape[0] / dur_s
    ch_names = None

    if verb > 0: print("X={} @{}Hz".format(X.shape,fs),flush=True)

    # extract the stimulus sequence
    Me, stim_ts, objIDs, _ = devent2stimSequence(messages)
        
    # up-sample to stim rate
    Y, stim_samp = upsample_stimseq(data_ts, Me, stim_ts, objIDs)
    if verb > 0: print("Y={} @{}Hz".format(Y.shape,fs),flush=True)
    
    # preprocess -> spectral filter, in continuous time!
    if stopband is not None:
        if verb > 0:
            print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X,stopband,fs,order=order)
    
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
    print("slicing {} trials =[{} - {}] samples @ {}Hz".format(len(trl_samp_idx),bgnend_samp[0], bgnend_samp[1],fs))
    for ti, si in enumerate(trl_samp_idx):
        idx = slice(si+bgnend_samp[0],si+bgnend_samp[1])
        X[ti, :, :] = Xraw[idx, :]
        Y[ti, :, :] = Yraw[idx, :]
    del Xraw, Yraw
    if verb > 0: print("X={}\nY={}".format(X.shape,Y.shape))

    # preprocess -> downsample
    #resamprate = int(round(fs/ofs))
    #if resamprate > 1:
    #    if verb > 0:
    #        print("resample by {}: {}->{}Hz".format(resamprate, fs, fs/resamprate))
    #    X = X[..., ::resamprate, :] # decimate X (trl, samp, d)
    #    Y = Y[..., ::resamprate, :] # decimate Y (OK as we latch Y)
    #    fs = fs/resamprate
    resamprate = round(2*fs/ofs)/2 # round to nearest .5
    if resamprate > 1:
        print("resample by {}: {}->{}Hz".format(resamprate, fs, fs/resamprate))
        # TODO []: use better re-sampler also in ONLINE
        idx = np.arange(0,X.shape[1],resamprate).astype(np.int)
        X = X[..., idx, :] # decimate X (trl, samp, d)
        Y = Y[..., idx, :] # decimate Y (OK as we latch Y)
        fs = fs/resamprate
    
    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = {'name':'trial','coords':trl_ts}
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':np.arange(X.shape[1])/fs, \
                 'fs':fs}
    coords[2] = {'name':'channel','coords':ch_names}
    # return data + metadata
    return (X, Y, coords)

def testcase():
    import sys

    sessfn = '../../resources/example_data/mindaffectBCI.txt'
 
    from load_mindaffectBCI import load_mindaffectBCI
    print("Loading: {}".format(sessfn))
    X, Y, coords = load_mindaffectBCI(sessfn, ofs=60)
    times = coords[1]['coords']
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']

    print("X({}){}".format([c['name'] for c in coords],X.shape))
    print("Y={}".format(Y.shape))
    print("fs={}".format(fs))

    # visualize the dataset
    from analyse_datasets import debug_test_dataset
    debug_test_dataset(X, Y, coords, tau_ms=100, evtlabs=('re','fe'), rank=1, model='cca')
    
    
if __name__=="__main__":
    testcase()
