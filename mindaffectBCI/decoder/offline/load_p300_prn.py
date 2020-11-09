import os
import numpy as np
from scipy.io import loadmat
from mindaffectBCI.decoder.utils import butter_sosfilt, window_axis, block_randomize

def load_p300_prn(datadir, sessdir=None, sessfn=None, fs_out=60, offset_ms=(-1000,1000), ifs=None, fr=None, stopband=((45,65), (1,25,'bandpass')), order=6, subtriallen=10, verb=0, nvirt=20, chidx=slice(64)):

    # load the data file
    Xfn = datadir
    if sessdir:
        Xfn = os.path.join(Xfn, sessdir)
    if sessfn:
        Xfn = os.path.join(Xfn, sessfn)
    sessdir = os.path.dirname(Xfn)
    try:
        data = loadmat(Xfn)
    except NotImplementedError:
        # TODO[] : make this work correctly -- HDF5 field access is different to loadmat's
        #import h5py
        #data = h5py.File(Xfn, 'r')
        pass

    X = data['X']
    X = X.astype("float32") # [ ch x samp x trl ]:float - raw eeg
    X = np.moveaxis(X, (0, 1, 2), (2, 1, 0)) # (nTrl, nSamp, d)
    X = np.ascontiguousarray(X) # ensure memory efficient layout

    ch_names = np.stack(data['di'][0]['vals'][0][0]).ravel()
    # Extract the sample rate.  Argh!, why so deeply nested?
    if ifs is None:
        fs = data['di'][1]['info'][0]['fs'][0,0][0,0]
    else:
        fs = ifs

    extrainfo = data['di'][2]['extra'][0]
    try:
        Ye = np.stack(extrainfo['flipgrid'][0], -1) # (nY,nEp,nTrl)
    except:
        Ye = None
    Ye0= np.stack(extrainfo['flash'][0], -1) # true-target  (1,nEp,nTrl)
    tgtLetter = extrainfo['target']  # target letter, not needed

    samptimes = data['di'][1]['vals'][0].ravel()   # (nSamp,)
    flashi_ms = np.stack(extrainfo['flashi_ms'][0], -1) #(1,nEp,nTrl)

    # convert flashi_ms to flashi_samp and upsampled Ye to sample rate
    Ye0= np.moveaxis(Ye0, (0, 1, 2), (2, 1, 0)) # (nTrl, nEp, 1)
    stimTimes_ms = np.moveaxis(flashi_ms, (0, 1, 2), (2, 1, 0)) # (nTrl, nEp, 1)
    if Ye is not None:
        Ye = np.moveaxis(Ye,  (0, 1, 2), (2, 1, 0)) # (nTrl, nEp, nY)
    else:
        # make a pseudo-set of alternative targets
        Ye = block_randomize(Ye0[...,np.newaxis],nvirt,-3) #(nTrl,nEp,nvirt,1)
        Ye = Ye[...,0] # (nTrl,nEp,nvirt)
        print("{} virt targets".format(Ye.shape[-1]))
    
    # upsample to sample rate
    stimTimes_samp = np.zeros(stimTimes_ms.shape, dtype=int) # index from trial start for each flash
    Y = np.zeros(X.shape[:-1]+(Ye.shape[-1]+Ye0.shape[-1],), dtype='float32') # (nTrl, nEP, nY+1)
    for ti in range(Y.shape[0]):
        lastflash = None
        flashi_trli = stimTimes_ms[ti, :, 0]
        for fi, flash_time_ms in enumerate(flashi_trli):
            # find nearest sample time
            si = np.argmin(np.abs(samptimes - flash_time_ms))
            stimTimes_samp[ti, fi, 0] = si
            
            if lastflash: # hold until new values
                Y[ti, lastflash+1:si, :] = Y[ti, lastflash, :]
                
            Y[ti, si, 0]  = Ye0[ti, fi, 0] # true info always 1st row
            Y[ti, si, 1:] = Ye[ti, fi, :]  # rest possiblities
            lastflash = si
    # for comparsion...
    #print("{}".format(np.array(np.mean(stimTimes_samp, axis=0),dtype=int).ravel()))
    
    # preprocess -> ch-seln
    if chidx is not None:
        X=X[...,chidx]
        ch_names = ch_names[chidx]

    # Trim to useful data range
    stimRng = ( np.min(stimTimes_samp[:,0,0]+offset_ms[0]*fs/1000),
                np.max(stimTimes_samp[:,-1,0]+offset_ms[1]*fs/1000) )
    print("stimRng={}".format(stimRng))
    if  0 < stimRng[0] or stimRng[1] < X.shape[-2]:
        if verb>-1 : print('Trimming range: {}-{}ms'.format(stimRng[0]/fs,stimRng[-1]/fs))
        # limit to the useful data range
        rng = slice(int(max(0,stimRng[0])), int(min(X.shape[-2],stimRng[1])))
        X = X[..., rng, :]
        Y = Y[..., rng, :]
        if verb > 0: print("X={}".format(X.shape))
        if verb > 0: print("Y={}".format(Y.shape))

    # preprocess -> spectral filter
    if stopband is not None:
        if verb > 0:
            print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X,stopband,fs,order=order)
    
    # preprocess -> downsample 
    resamprate = int(round(fs/fs_out))
    if resamprate > 1:
        if verb > 0:
            print("resample: {}->{}hz rsrate={}".format(fs, fs_out, resamprate))
        X = X[:, ::resamprate, :] # decimate X (trl, samp, d)
        Y = Y[:, ::resamprate, :] # decimate Y (trl, samp, y)
        fs = fs/resamprate

    nsubtrials = X.shape[1]/fs/subtriallen if subtriallen is not None else 0
    if nsubtrials > 1:
        winsz = int(X.shape[1]//nsubtrials)
        print('{} subtrials -> winsz={}'.format(nsubtrials,winsz))
        # slice into sub-trials
        X = window_axis(X,axis=1,winsz=winsz,step=winsz) # (trl,win,samp,d)
        Y = window_axis(Y,axis=1,winsz=winsz,step=winsz) # (trl,win,samp,nY)
        # concatenate windows into trial dim
        X = X.reshape((X.shape[0]*X.shape[1],)+X.shape[2:])
        Y = Y.reshape((Y.shape[0]*Y.shape[1],)+Y.shape[2:])
        if verb>0 : print("X={}".format(X.shape))

    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = {'name':'trial','coords':np.arange(X.shape[0])}
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':np.arange(X.shape[1]) * 1000/fs, \
                 'fs':fs}
    coords[2] = {'name':'channel','coords':ch_names}
    
    return (X, Y, coords)

def testcase():
    import sys
    from readCapInf import getPosInfo

    datadir = '/home/jadref/data/bci/own_experiments/visual/p300_prn_2'
    sessdir = ''
    sessfn = 'peter/20100714/jf_prep/peter_rc_5_flash.mat'
    sessfn = 'peter/20100714/jf_prep/peter_prn_5_flip.mat'

    #sessfn = 'alex/20100722/jf_prep/alex_prn_10_flip.mat'
    
    from offline.load_p300_prn import load_p300_prn
    X, Y, coords = load_p300_prn(datadir, sessdir, sessfn, fs_out=32, stopband=((0,1),(12,-1)), order=6); oX=X.copy(); oY=Y.copy();
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']

    # attach the electrode positions
    cnames, xy, xyz, iseeg =getPosInfo(ch_names)
    coords[2]['pos2d'] = xy
    
    # set the analysis paramters
    tau = int(.6*fs)
    evtlabs = ('re','anyre') # ('re', 'ntre') #('re', 'fe') # 
    times = [i/fs for i in range(tau)]
    rank = 10

    from analyse_datasets import debug_test_dataset
    debug_test_dataset(X, Y, coords, tau_ms=800, evtlabs=evtlabs, rank=rank, regy=.1, regx=.1)
    

    # try with cross-validation
    res = cca.cv_fit(X, Y)
    Fy=res['estimator']
    (_) = decodingCurveSupervised(Fy)
    
    print('FWD')
    fwd = FwdLinearRegression(tau=tau, evtlabs=evtlabs)
    fwd.fit(X, Y)
    print('fwd = {}'.format(fwd))
    Fy = fwd.predict(X, Y, dedup0=True)
    print("Fy={}".format(Fy.shape))
    (_) = decodingCurveSupervised(Fy)

    plot_erp(factored2full(fwd.W_, fwd.R_), evtlabs, times, ch_names)

    
    print('BWD')
    bwd = BwdLinearRegression(tau=tau, evtlabs=evtlabs)
    bwd.fit(X, Y)
    print('bwd = {}'.format(bwd))
    Fy = bwd.predict(X, Y, dedup0=True)
    print("Fy={}".format(Fy.shape))
    (_) = decodingCurveSupervised(Fy)


if __name__ == "__main__":
    testcase()
