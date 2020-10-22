import os
import numpy as np
from scipy.io import loadmat
from mindaffectBCI.decoder.utils import butter_sosfilt

def load_brainstream(datadir, sessdir=None, sessfn=None, fs_out=60, ifs=None, fr=None, stopband=((45,65),(5,25,'bandpass')), verb=0, ch_names=None):
    """Load and pre-process a brainstream offline save-file and return the EEG data, and stimulus information

    Args:
        datadir (str): root of the data directory tree
        sessdir (str, optional): sub-directory for the session to load. Defaults to None.
        sessfn (str, optional): filename for the session information. Defaults to None.
        fs_out (float, optional): [description]. Defaults to 100.
        ifs (float, optional): the input data sample rate.
        fr (float, optional): the input stimulus frame rate.
        stopband (tuple, optional): Specification for a (cascade of) temporal (IIR) filters, in the format used by `mindaffectBCI.decoder.utils.butter_sosfilt`. Defaults to ((45,65),(5.5,25,'bandpass')).
        ch_names (tuple, optional): Names for the channels of the EEG data.

    Returns:
        X (np.ndarray (nTrl,nSamp,nCh)): the pre-processed per-trial EEG data
        Y (np.ndarray (nTrl,nSamp,nY)): the up-sampled stimulus information for each output
        coords (list-of-dicts (3,)): dictionary with meta-info for each dimension of X & Y.  As a minimum this contains
                          "name"- name of the dimension, "unit" - the unit of measurment, "coords" - the 'value' of each element along this dimension
    """    

    # load the data file
    Xfn = datadir
    if sessdir:
        Xfn = os.path.join(Xfn, sessdir)
    if sessfn:
        Xfn = os.path.join(Xfn, sessfn)
    sessdir = os.path.dirname(Xfn)
    data = loadmat(Xfn)
    if 'v' in data.keys(): # silly mat struct stuff..
        data = data['v'] # [d x samp x trl ]
    #print("data.keys={}".format(data.keys()))

    if 'X' in data: # all in 1 file, raw plos_one format
        X  = data['X'] # (d x samp x trl)
        cb = data['codes'] if 'codes' in data else data['V'] # (samples x nY)
        lab = data['y'] # (trl)

    elif 'data' in data: # pre-converted data
        X    = data['data']     # [ d x samp_seq x nTrl ]
        cb   = data['codebooks'] # [ samp_code x nY ]
        lab  = data['labels']   # [ nTrl x 1] # index into codes array

    else: # extract info from other files...

        X = data
        trainmode = 'train' in fn
        # general config = codebook info
        cfgfn = os.path.join(datadir, sessdir, 'cfg.mat')
        try:
            cfg = loadmat(cfgfn)
            stim = cfg['stimulation']
            fr = stim['rate']*stim['uprate']
            if trainmode:
                subset = stim['trainsubset']
                layout = stim['trainlayout']
                cb = stim['U'] # [ samp_code x nY ]
            else:
                subset = stim['testsubset']
                layout = stim['testlayout']
                cb = stim['V'] # [ samp_code x nY ]
            cb = cb[:, trainlayout]
            cb = cb[:, trainlayout] # [ samp_code x nY ]
        except:
            raise ValueError('Couldnt load the configuration file!')

        # extract the header to get the sample rate
        hdrfn = os.path.join(datadir, sessdir, 'hdr.mat')
        try:
            hdr=loadmat(hdrfn)
            if 'v' in hdr.keys():
                hdr = hdr['v']
                ifs = hdr['Fs'][0]
        except:
            print('Warning: Couldnt load the header file');

        # load per-trial labels
        if trainmode:
            labfn = os.path.join(datadir, sessdir, 'train'+'labels.mat')
        else:
            labfn = os.path.join(datadir, sessdir, 'test'+'labels.mat')
        lab = loadmat(labfn)
        if 'v' in lab.keys(): lab = lab['v'] # [ trl ]

    # get the sample rate,  and downsample rate
    if ifs is None: # get the sample rate info
        if 'fs' in data:
            ifs = data['fs'][0][0]
        elif 'fSample' in data:
            ifs = data['fSample'][0][0]
        else:
            ifs = 180
    if fr is None:
        if 'fr' in data:
            fr = data['fr'][0][0]
        else:
            fr = 60
    fs = ifs
    if fs_out is None:
        fs_out = ifs
        
    X  = X.astype("float32") # [ ch x samp x trl ]:float - raw eeg
    X  = np.moveaxis(X, (0, 1, 2), (2, 1, 0)) # (nTrl, nSamp, d)
    X  = np.ascontiguousarray(X) # ensure memory efficient layout
    lab = lab.astype("uint8").ravel() # [ trl ]:int - index into codebook
    cb = cb.astype("bool") # [ samp x nY ]:bool -- codebook

    # convert lab+code into Y + samp-times,  where 1st row is always the true label
    cb = cb[np.mod(np.arange(0, X.shape[1]), cb.shape[0]), :] # loop cb up to X size [ samp x nY]
    Y  = np.zeros((len(lab), X.shape[1], cb.shape[1]+1), dtype=bool) # (nTr, nSamp, nY) [nY x samp x trl]
    for ti, l in enumerate(lab):
        Y[ti, :, 0]  = cb[:, l-1] # copy in true label
        Y[ti, :, 1:] = cb # copy in other outputs

    # preprocess -> spectral filter
    if stopband is not None or passband is not None:
        # BODGE: pre-center X
        X = X - X[...,0:1,:]
        if verb > 0:
            print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X, stopband, fs)
    
    # preprocess -> downsample
    resamprate = round(2*fs/fs_out)/2 # round to nearest .5
    if resamprate > 1:
        if 1 or verb > 0:
            print("resample by {}: {}->{}Hz".format(resamprate, fs, fs/resamprate))
        idx = np.arange(0,X.shape[1],resamprate).astype(np.int)
        X = X[:, idx, :] # decimate X (trl, samp, d)
        Y = Y[:, idx, :] # decimate Y (trl, samp, y)
        fs = fs/resamprate

    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = {'name':'trial'}
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':np.arange(X.shape[1]) *1000/fs, \
                 'fs':fs}
    coords[2] = {'name':'channel', 'coords':ch_names}
    
    return (X, Y, coords)

def testcase():
    import sys

    # plos_one
    #datadir = '/home/jadref/removable/SD Card/data/bci/own_experiments/noisetagging_v3'
    #sessdir = 's01'
    #sessfn = 'traindata.mat'

    # lowlands
    datadir = '/home/jadref/removable/SD Card/data/bci/own_experiments/lowlands'
    sessdir = ''
    sessfn = 'LL_ENG_39_20170820_tr_train_1.mat' # should perform 100%

    # command-line, for testing
    if len(sys.argv) > 1:
        datadir = sys.argv[1]
    if len(sys.argv) > 2:
        sessdir = sys.argv[2]
    if len(sys.argv) > 3:
        fn = sys.argv[3]

    from load_brainstream import load_brainstream
    X, Y, coords = load_brainstream(datadir, sessdir, sessfn, fs_out=60, stopband=((0,5.5),(24,-1)))
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']
    
    from model_fitting import MultiCCA
    from decodingCurveSupervised import decodingCurveSupervised
    cca = MultiCCA(tau=18)
    cca.fit(X, Y)
    print('cca = {}'.format(cca))
    Fy = cca.predict(X, Y, dedup0=True)
    print("Fy={}".format(Fy.shape))
    (_) = decodingCurveSupervised(Fy)

    # test w.r.t. matlab
    from scipy.io import loadmat
    from utils import window_axis
    import numpy as np
    Xe = window_axis(X, axis=-2, winsz=18)
    Ye = window_axis(Y, axis=-2, winsz=1)
    
    matdata=loadmat('/home/jadref/'+sessfn)
    Xm = np.moveaxis(matdata['X'], (0, 1, 2), (2, 1, 0)) # (t,s,d)
    Ym = np.moveaxis(matdata['Y'], (0, 1, 2), (2, 1, 0)) # (t,s,y)
    stem = np.moveaxis(matdata['stimTimes_samp'], (0, 1), (1, 0)) # (t,e)

    cca = MultiCCA(tau=18)
    cca.fit(Xm, Ym)
    cca.score(Xm, Ym)
    (res) = decodingCurveSupervised(cca.predict(Xm,Ym))
    
    print("X-Xm={}".format(np.max(np.abs(X-Xm).ravel())))
    print("Y-Ym={}".format(np.max(np.abs(Y[:, :, 0]-Ym[:, :, 0]).ravel())))

    Xem = np.moveaxis(matdata['Xe'], (0, 1, 2, 3), (3, 2,1,0))# (t,e,tau,d)
    Yem = np.moveaxis(matdata['Ye'], (0, 1, 2), (2, 1, 0)) # (t,e, y)

    # off by 1 in the sliced version, w.r.t. Matlab
    print("Xe-Xem={}".format(np.max(np.abs(Xe[:,1:Xem.shape[1]+1,:,:]-Xem).ravel())))
    
    print("Ye-Yem={}".format(np.max(np.abs(Ye[:,1:Yem.shape[1]+1, 0, 0]-Yem[:, :, 0]).ravel())))
    
    
    cca = MultiCCA(tau=Xem.shape[-2])
    cca.fit(Xem, Yem, stem)
    print('cca = {}'.format(cca))
    Fy = cca.predict(Xem, Yem)
    print("Fy={}".format(Fy.shape))
    (res)=decodingCurveSupervised(Fy);

    # run in stages
    from updateSummaryStatistics import updateSummaryStatistics
    from stim2event import stim2event
    Yeem=np.moveaxis(matdata['Yee'], (0, 1, 2, 3), (3, 2, 1, 0))
    Yeme=stim2event(Yem, ['re', 'fe'], -2) # py convert to brain events
    print("Yeem-Yeme={}".format(np.max(np.abs(Yeem-Yeme).ravel())))
    
    Cxx,Cxy,Cyy=updateSummaryStatistics(Xem, Yeem[:,:,0:1,:], stem)

    Cxxm=matdata['Cxx']
    Cxym=np.moveaxis(matdata['Cxy'],(0,1,2),(2,1,0))
    Cyym=np.moveaxis(matdata['Cyy'],(0,1,2,3),(3,2,1,0))

    print('Cxx-Cxxm={}'.format(np.max(np.abs(Cxx-Cxxm).ravel())))
    print('Cxy-Cxym={}'.format(np.max(np.abs(Cxy-Cxym).ravel())))
    print('Cyy-Cyym={}'.format(np.max(np.abs(Cyy-Cyym).ravel())))
    
    import matplotlib.pyplot as plt
    plt.clf();plt.plot(X[1,:,0]);plt.plot(Xm[0,:,0])
    plt.clf();plt.plot(Y_true.ravel());plt.plot(Ym_true.ravel());

if __name__ == "__main__":
    testcase()
