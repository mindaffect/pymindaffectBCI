import numpy as np
from scipy.io import loadmat
from mindaffectBCI.decoder.utils import butter_sosfilt, window_axis, block_randomize
from mindaffectBCI.decoder.multipleCCA import robust_whitener
from mindaffectBCI.decoder.updateSummaryStatistics import updateCxx
import matplotlib.pyplot as plt

def load_ninapro_db2(datadir, stopband=((0,15), (45,65), (95,125), (250,-1)), envelopeband=(10,-1), trlen_ms=None, fs_out=60, nvirt=20, rectify=True, whiten=True, log=True, plot=False, filterbank=None, zscore_y=True, verb=1):
    d = loadmat(datadir, variable_names=('emg', 'glove', 'stimulus'))
    X = d['emg'] # (nSamp,d)
    Y = d['glove'] # (nSamp,e)
    lab = d['stimulus'].ravel() # (nSamp,1) - use to slice out trials+labels
    fs = 2000
    if fs_out is  None:
        fs_out = fs

    # get trial start/end info
    trl_start = np.flatnonzero(np.diff(lab)>0)
    lab = lab[trl_start+1]
    print('trl_start={}'.format(trl_start))
    print('label={}'.format(lab))
    print("diff(trl_start)={}".format(np.diff(trl_start)))
    if trlen_ms is None:
        trlen_ms = np.max(np.diff(trl_start))*1000/fs
        print('trlen_ms={}'.format(trlen_ms))

    if not stopband is None:
        if verb > 0:
            print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X,stopband,fs)
        if plot:plt.figure(101);plt.plot(X);plt.title("hp+notch+lp")
        # preprocess -> spatial whiten
        # TODO[] : make this  fit->transform method

    if whiten:
        if verb > 0:print("spatial whitener")
        Cxx = updateCxx(None,X,None)
        W,_ = robust_whitener(Cxx)
        X = np.einsum("sd,dw->sw",X,W)
        if plot:plt.figure(102);plt.plot(X);plt.title("+whiten")            

    if not filterbank is None:
        if verb > 0:  print("Filterbank: {}".format(filterbank))
        # apply filter bank to frequency ranges into virtual channels
        Xs=[]
        # TODO: make a nicer shape, e.g. (tr,samp,band,ch)
        for bi,band in enumerate(filterbank):
            Xf, _, _ = butter_sosfilt(X,band,fs)
            Xs.append(Xf)
        # stack the bands as virtual channels
        X = np.concatenate(Xs,-1)
        
    X = np.abs(X) # rectify

    if log:
        if verb > 0: print("log amplitude")
        X = np.log(np.maximum(X,1e-6))
    if plot:plt.figure(103);plt.plot(X);plt.title("+abs")
    if envelopeband is not None:
        if verb>0: print("Envelop band={}".format(envelopeband))
        X, _, _ = butter_sosfilt(X,envelopeband,fs) # low-pass = envelope extraction
        if plot:plt.figure(104);plt.plot(X);plt.title("env")

    # preprocess -> downsample
    resamprate = int(fs/fs_out)
    if resamprate > 1:
        if verb > 0:
            print("resample: {}->{}hz rsrate={}".format(fs, fs/resamprate, resamprate))
        X = X[..., ::resamprate, :] # decimate X (trl, samp, d)
        Y = Y[..., ::resamprate, :] # decimate Y (trl, samp, e)
        trl_start = trl_start/resamprate
        fs = fs/resamprate

    # pre-process : z-trans Y
    if zscore_y:
        if verb > 0:print("Z-trans Y")
        mu = np.mean(Y, axis=-2, keepdims=True)
        std = np.std(Y, axis=-2, keepdims=True)
        std[std < 1e-6] = 1 # guard divide by 0
        Y = (Y - mu) / std

    # generate artificial other stimulus streams, for testing
    # TODO: randomize in better way
    Y = Y[:, np.newaxis, :] # (nSamp, nY, e)
    Y_test = block_randomize(Y, nvirt, axis=-3, block_size=Y.shape[0]//100//2)
    Y = np.concatenate((Y, Y_test), -2) # (nSamp, nY, e)
   
    # slice X,Y into trials
    oX = X # (nSamp,d)
    oY = Y # (nSamp,nY,e)
    trlen_samp = int(trlen_ms * fs / 1000)
    X = np.zeros((trl_start.size, trlen_samp, X.shape[-1]))
    Y = np.zeros((trl_start.size, trlen_samp) + Y.shape[-2:])
    print("Slicing {} trials of {}ms".format(len(trl_start),trlen_ms))
    for ti,tii in enumerate(trl_start):
        tii = int(tii)
        trl_len = min(oX.shape[0],tii+trlen_samp) - tii
        X[ti, :trl_len, ...] = oX[tii:tii+trl_len, ...]
        Y[ti, :trl_len, ...] = oY[tii:tii+trl_len, ...]

    # make meta-info
    coords = [None]*X.ndim
    coords[0] = {'name':'trial', 'coords':lab}
    coords[1] = {'name':'time', 'fs':fs, 'units':'ms', 'coords':np.arange(X.shape[1])*1000/fs}
    coords[2] = {'name':'channel', 'coords':None}

    return (X, Y, coords)

def testcase():
    try:
        from offline.load_ninapro_db2 import load_ninapro_db2
    except:
        pass
    
    datadir = '/home/jadref/data/bci/external_data/ninapro/DB2_s1/S1_E1_A1.mat'

    fs_out = 30
    X, Y, coords = load_ninapro_db2(datadir, fs_out=fs_out, envelopeband=((0,.01),(5,-1)), log=True)
    from analyse_datasets import debug_test_dataset
    debug_test_dataset(X,Y,coords, rank=5, evtlabs=None, tau_ms=120, offset_ms=0, model='cca')
    
if __name__=="__main__":
    testcase()
