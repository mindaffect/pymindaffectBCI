import os
import numpy as np
from scipy.io import loadmat, whosmat
from mindaffectBCI.decoder.utils import block_randomize, butter_sosfilt, upsample_codebook, lab2ind, window_axis

marker2stim=dict(lh=(1,3),rh=(2,4))

def load_twofinger(datadir, sessdir=None, sessfn=None, fs_out=60, stopband=((45,65),(0,1),(25,-1)), subtriallen=10, nvirt=20, verb=0, ch_idx=slice(32)):
    
    # load the data file
    Xfn = datadir
    if sessdir:
        Xfn = os.path.join(Xfn, sessdir)
    if sessfn:
        Xfn = os.path.join(Xfn, sessfn)
    sessdir = os.path.dirname(Xfn)

    data = loadmat(Xfn)

    def squeeze(v):
        while v.size == 1 and v.ndim > 0:
            v = v[0]
        return v

    fs = 512
    ch_names = [c[0] for c in squeeze(data['chann']).ravel()]
    X = squeeze(data['X'])  # (ch,samp)
    X = np.moveaxis(X,(0,1),(1,0)) # (samp,ch)
    X = X.astype(np.float32)
    X = np.ascontiguousarray(X)
    if ch_idx is not None:
        X = X[:, ch_idx]
        ch_names = ch_names[ch_idx]
    if verb>0: print("X={}".format(X.shape),flush=True)
    
    lab = squeeze(data['Y']).astype(int).ravel() # (samp,)

    # preprocess -> spectral filter, in continuous time!
    if stopband is not None:
        if verb > 0:
            print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X,stopband,fs)
    
    # make the targets, for the events we care about
    Y, lab2class = lab2ind(lab,marker2stim.values()) # (nTrl, e) # feature dim per class
    if verb>0: print("Y={}".format(Y.shape))
        
    # preprocess -> downsample
    resamprate = int(fs/fs_out)
    if resamprate > 1:
        if verb > 0:
            print("resample: {}->{}hz rsrate={}".format(fs, fs/resamprate, resamprate))
        X = X[..., ::resamprate, :] # decimate X (trl, samp, d)
        # re-sample Y, being sure to keep any events in the re-sample window
        Y = window_axis(Y,winsz=resamprate,step=resamprate,axis=-2) # (trl, samp, win, e)
        Y = np.max(Y,axis=-2) # (trl,samp,e)  N.B. use max so don't loose single sample events
        fs = fs/resamprate
        if verb > 0:
            print("X={}".format(X.shape))
            print("Y={}".format(Y.shape))

    # make virtual targets
    Y = Y[:,np.newaxis,:] # (nsamp,1,e)
    Y_virt = block_randomize(Y, nvirt, axis=-3) # (nsamp,nvirt,e)
    Y = np.concatenate((Y, Y_virt), axis=-2) # (nsamp,1+nvirt,e)
    if verb>0: print("Y={}".format(Y.shape))

    # cut into sub-trials
    nsubtrials = X.shape[0]/fs/subtriallen
    if nsubtrials > 1:
        winsz = int(X.shape[0]//nsubtrials)
        if verb>0: print('subtrial winsz={}'.format(winsz))
        # slice into sub-trials
        X = window_axis(X,axis=0,winsz=winsz,step=winsz) # (trl,win,d)
        Y = window_axis(Y,axis=0,winsz=winsz,step=winsz) # (trl,win,nY)
        if verb>0: 
            print("X={}".format(X.shape))
            print("Y={}".format(Y.shape))
    
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
        sessfn = 'D:\\external_data\twente\twofinger\S00.mat'
    else:
        sessfn = '/home/jadref/data/bci/external_data/twente/twofinger/S00.mat'
    # command-line, for testing
    if len(sys.argv) > 1:
        sessfn = sys.argv[1]
 
    from load_twofinger import load_twofinger
    oX, oY, coords = load_twofinger(sessfn, fs_out=60, nsubtrials=40)
    times = coords[1]['coords']
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']
    X=oX.copy()
    Y=oY.copy()

    print("X({}){}".format([c['name'] for c in coords],X.shape))
    print("Y={}".format(Y.shape))
    print("fs={}".format(fs))

    tau=fs*.7
    evtlabs = None
    times=np.arange(int(tau))/fs
    rank=1

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
