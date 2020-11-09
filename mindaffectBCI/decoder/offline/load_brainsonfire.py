import os
import numpy as np
from .read_buffer_offline import read_buffer_offline_data, read_buffer_offline_events, read_buffer_offline_header
from mindaffectBCI.decoder.utils import block_randomize, butter_sosfilt, upsample_codebook, lab2ind, window_axis

trigger_event='stimulus.note.play' # the actual times the user hit the button

def load_brainsonfire(datadir, sessdir=None, sessfn=None, fs_out=60, stopband=((45,65),(0,1),(25,-1)), subtriallen=10, nvirt=20, chIdx=slice(64), verb=2):
    
    # load the data file
    Xfn = datadir
    if sessdir:
        Xfn = os.path.join(Xfn, sessdir)
    if sessfn:
        Xfn = os.path.join(Xfn, sessfn)
    sessdir = os.path.dirname(Xfn)

    if verb > 1: print("Loading header")
    hdr=read_buffer_offline_header(Xfn)
    if verb > 1: print("Loading data")
    X = read_buffer_offline_data(Xfn,hdr) # (nsamp,nch)
    if verb > 1: print("Loading events")
    evts=read_buffer_offline_events(Xfn)

    fs = hdr.fs
    ch_names = hdr.labels

    if chIdx is not None:
        X = X [...,chIdx]
        ch_names = ch_names[chIdx] if ch_names is not None else None

    # pre-resample to save memory
    rsrate = int(fs//120)
    if rsrate > 1:
        if verb > 0: print("Pre-re-sample by {}: {}->{}Hz".format(rsrate,fs,fs/rsrate))
        X = X [::rsrate,:]
        for e in evts:
            e.sample = e.sample/rsrate
        fs = fs/rsrate

    if verb > 0: print("X={} @{}Hz".format(X.shape,fs),flush=True)

    # extract the trigger info
    trigevts = [e for e in evts if e.type.lower() == trigger_event]
    trig_samp= np.array([e.sample for e in trigevts],dtype=int)
    trig_val = [e.value for e in trigevts]
    trig_ind, lab2class = lab2ind(trig_val) # convert to indicator (ntrig,ncls)
    # up-sample to stim rate
    Y = np.zeros((X.shape[0],trig_ind.shape[-1]),dtype=bool)
    Y[trig_samp,:] = trig_ind
    if verb > 0:
            print("Y={}".format(Y.shape))
    
    # BODGE: trim to useful data range
    if .1 < (trig_samp[0]-fs)/X.shape[0] or (trig_samp[-1]+fs)/X.shape[0] < .9:
        if verb>0 : print('Trimming range: {}-{}s'.format(trig_samp[0]/fs,trig_samp[-1]/fs))
        # limit to the useful data range
        rng = slice(int(trig_samp[0]-fs), int(trig_samp[-1]+fs))
        X = X[rng, :]
        Y = Y[rng, ...]
        if verb > 0: print("X={}".format(X.shape))
        if verb > 0: print("Y={}".format(Y.shape))

    # preprocess -> spectral filter, in continuous time!
    if stopband is not None:
        if verb > 0:
            print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X,stopband,fs)
    
    # preprocess -> downsample
    resamprate = int(fs/fs_out)
    if resamprate > 1:
        if verb > 0:
            print("resample by {}: {}->{}Hz".format(resamprate, fs, fs/resamprate))
        X = X[..., ::resamprate, :] # decimate X (trl, samp, d)
        # re-sample Y, being sure to keep any events in the re-sample window
        Y = window_axis(Y,winsz=resamprate,step=resamprate,axis=-2) # (trl, samp, win, e)
        Y = np.max(Y,axis=-2) # (trl,samp,e)  N.B. use max so don't loose single sample events
        fs = fs/resamprate

    # make virtual targets
    Y = Y[:,np.newaxis,:] # (nsamp,1,e)
    Y_virt = block_randomize(Y, nvirt, axis=-3) # (nsamp,nvirt,e)
    Y = np.concatenate((Y, Y_virt), axis=-2) # (nsamp,1+nvirt,e)
    if verb > 0: print("Y={}".format(Y.shape))

    # cut into sub-trials
    nsubtrials = X.shape[0]/fs/subtriallen
    if nsubtrials > 1:
        winsz = int(X.shape[0]//nsubtrials)
        if verb > 0: print('subtrial winsz={}'.format(winsz))
        # slice into sub-trials
        X = window_axis(X,axis=0,winsz=winsz,step=winsz) # (trl,win,d)
        Y = window_axis(Y,axis=0,winsz=winsz,step=winsz) # (trl,win,nY)
        if verb > 0: print("X={}".format(X.shape))
        if verb > 0: print("Y={}".format(Y.shape))
    
    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = {'name':'trial'}
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':np.arange(X.shape[1])/fs, \
                 'fs':fs}
    coords[2] = {'name':'channel','coords':ch_names}
    # return data + metadata
    return (X, Y, coords)

def testcase():
    import sys

    if os.path.isdir('D:\external_data'):
        datadir = 'D:/own_experiments/'
    else:
        datadir = '/home/jadref/data/bci/own_experiments'
    sessfn = os.path.join(datadir,'motor_imagery/brainsonfire/brains_on_fire_online/subject01/raw_buffer/0001')
    # command-line, for testing
    if len(sys.argv) > 1:
        sessfn = sys.argv[1]
 
    from load_brainsonfire import load_brainsonfire
    print("Loading: {}".format(sessfn))
    oX, oY, coords = load_brainsonfire(sessfn, fs_out=60)
    times = coords[1]['coords']
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']
    X=oX.copy()
    Y=oY.copy()

    print("X({}){}".format([c['name'] for c in coords],X.shape))
    print("Y={}".format(Y.shape))
    print("fs={}".format(fs))

    tau=fs*.3
    evtlabs = None
    times=np.arange(int(tau))/fs
    rank=10

    # visualize the dataset
    from stim2event import stim2event
    from updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics, idOutliers
    import matplotlib.pyplot as plt
    
    Cxx, Cxy, Cyy = updateSummaryStatistics(X, Y[...,0:1,:], tau=tau)

    plt.figure(1);
    print("summary stats")
    plot_summary_statistics(Cxx, Cxy, Cyy, evtlabs, times, ch_names)

    plt.figure(2);
    print("ERP")
    plot_erp(Cxy, ch_names=ch_names, evtlabs=evtlabs, times=times, plottype='plot', axis=-1)

    from model_fitting import MultiCCA
    from decodingCurveSupervised import decodingCurveSupervised
    cca = MultiCCA(tau=tau, evtlabs=evtlabs, rank=rank)
    scores = cca.cv_fit(X, Y)
    Fy = scores['estimator']
    print("Fy={}".format(Fy.shape))
    (_)=decodingCurveSupervised(Fy)

    # plot the solution
    from scoreStimulus import factored2full
    print("Plot Model")
    plt.figure(3)
    plot_erp(factored2full(cca.W_, cca.R_), ch_names=ch_names, evtlabs=evtlabs, times=times)
    #   plot Fy
    plt.figure(4)
    for ti in range(min(Fy.shape[0],25)):
        plt.subplot(5,5,ti+1)
        plt.imshow(np.cumsum(Fy[ti,:,:],axis=-2),aspect='auto')
    plt.show()
    
    
if __name__=="__main__":
    testcase()
