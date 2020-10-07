from scipy.io import loadmat
from mindaffectBCI.decoder.utils import window_axis, block_randomize, butter_sosfilt
import numpy as np
def load_mTRF_audio(datadir, regressor='envelope', ntrl=15, stopband=((45,65),(0,.5),(15,-1)), fs_out=60, nvirt_out=30, verb=1):
    d = loadmat(datadir)
    X = d['EEG'] # (nSamp,d)
    Y = d[regressor] # (nSamp,e)
    Y = Y[:, np.newaxis, :] # (nSamp, nY, e)
    fs = d['Fs'][0][0]
    if fs_out is  None:
        fs_out = fs

    # preprocess -> spectral filter, in continuous time!
    if stopband is not None:
        if verb > 0:
            print("preFilter: {}Hz".format(stopband))
        X,_,_ = butter_sosfilt(X, stopband, fs, axis=-2)
    
    # generate artificial other stimulus streams, for testing
    Y_test = block_randomize(Y, nvirt_out, axis=-3, block_size=Y.shape[0]//ntrl//2)
    Y = np.concatenate((Y, Y_test), -2) # (nSamp, nY, e)
    
    # slice X,Y into 'trials'
    if  ntrl > 1:
        winsz = X.shape[0]//ntrl
        X = window_axis(X, axis=0, winsz=winsz, step=winsz) # (ntrl,nSamp,d)
        Y = window_axis(Y, axis=0, winsz=winsz, step=winsz) # (nTrl,nSamp,nY,e)
    else:
        X = [np.newaxis, ...]
        Y = [np.newaxis, ...]

    # preprocess -> downsample
    resamprate = int(fs/fs_out)
    if resamprate > 1:
        if verb > 0:
            print("resample: {}->{}hz rsrate={}".format(fs, fs/resamprate, resamprate))
        X = X[:, ::resamprate, :] # decimate X (trl, samp, d)
        Y = Y[:, ::resamprate, :] # decimate Y (trl, samp, y)
        fs = fs/resamprate
        
    # make meta-info
    coords = [None]*X.ndim
    coords[0] = {'name':'trial'}
    coords[1] = {'name':'time', 'fs':fs, 'units':'ms', 'coords':np.arange(X.shape[1])*1000/fs}
    coords[2] = {'name':'channel', 'coords':None}

    return (X, Y, coords)
