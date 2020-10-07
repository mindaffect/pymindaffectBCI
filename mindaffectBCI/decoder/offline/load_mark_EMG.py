import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from mindaffectBCI.decoder.utils import block_randomize, butter_sosfilt, upsample_codebook, lab2ind
from mindaffectBCI.decoder.multipleCCA import robust_whitener
from mindaffectBCI.decoder.updateSummaryStatistics import updateCxx

def load_mark_EMG(datadir, sessdir=None, sessfn=None, fs_out=60, stopband=((45,65),(0,10),(45,55),(95,105),(145,-1)), filterbank=None, verb=0, log=True, whiten=True, plot=False):

    fs=1000
    ch_names=None
    
    # load the data file
    Xfn = os.path.expanduser(datadir)
    if sessdir:
        Xfn = os.path.join(Xfn, sessdir)
    if sessfn:
        Xfn = os.path.join(Xfn, sessfn)
    sessdir = os.path.dirname(Xfn)

    print("Loading {}".format(Xfn))
    data = loadmat(Xfn)

    def squeeze(v):
        while v.size == 1 and v.ndim > 0:
            v = v[0]
        return v

    X = np.array([squeeze(d['buf']) for d in squeeze(data['data'])]) # ( nTrl, nch, nSamp)
    X = np.moveaxis(X,(0,1,2),(0,2,1)) # ( nTr, nSamp, nCh)
    X = np.ascontiguousarray(X) # ensure memory efficient
    lab = np.array([squeeze(e['value']) for e in data['devents']],dtype=int) # (nTrl,)

    import matplotlib.pyplot as plt
    if plot: plt.figure(100);plt.plot(X[0,:,:]);plt.title("raw")

    # preprocess -> spectral filter, in continuous time!
    if stopband is not None:
        if verb > 0:
            print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X,stopband,fs)
        if plot:plt.figure(101);plt.plot(X[0,:,:]);plt.title("hp+notch+lp")
        # preprocess -> spatial whiten
        # TODO[] : make this  fit->transform method
        if whiten:
            print("spatial whitener")
            Cxx = updateCxx(None,X,None)
            W,_ = robust_whitener(Cxx)
            X = np.einsum("tsd,dw->tsw",X,W)
            if plot:plt.figure(102);plt.plot(X[0,:,:]);plt.title("+whiten")
            
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
        print("log amplitude")
        X = np.log(np.maximum(X,1e-6))
    if plot:plt.figure(103);plt.plot(X[0,:,:]);plt.title("+abs")
    X, _, _ = butter_sosfilt(X,(40,-1),fs) # low-pass = envelope extraction
    if plot:plt.figure(104);plt.plot(X[0,:,:]);plt.title("env")
        
    # preprocess -> downsample @60hz
    resamprate=int(fs/fs_out)
    if resamprate > 1:
        if verb > 0:
            print("resample: {}->{}hz rsrate={}".format(fs,fs_out,resamprate))
        X = X[:, ::resamprate, :] # decimate X (trl, samp, d)
        fs = fs/resamprate

    # get Y
    Y_true, lab2class = lab2ind(lab) # (nTrl, e)
    Y_true = Y_true[:, np.newaxis, :] # ( nTrl,1,e)
    # TODO[] : exhaustive list of other targets...
    Yall = np.eye(Y_true.shape[-1],dtype=bool) # (nvirt,e)
    Yall = np.tile(Yall,(Y_true.shape[0],1,1)) # (nTrl,nvirt,e)
    Y = np.append(Y_true,Yall,axis=-2) # (nTrl,nvirt+1,e)
    # upsample to fs_out
    Y = np.tile(Y[:,np.newaxis,:,:],(1,X.shape[1],1,1)) #  (nTrl, nSamp, nY, e)
    Y = Y.astype(np.float32)
    
    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = {'name':'trial'}
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':np.arange(X.shape[1])*1000/fs, \
                 'fs':fs}
    coords[2] = {'name':'channel','coords':ch_names}
    # return data + metadata
    return (X, Y, coords)

def testcase():
    import sys

    sessfn = '~/data/bci/own_experiments/emg/facial/training_data_SV_6.mat'
    # command-line, for testing
    if len(sys.argv) > 1:
        sessfn = sys.argv[1]
 
    #from offline.load_mark_EMG import load_mark_EMG
    oX, oY, coords = load_mark_EMG(sessfn, fs_out=125, stopband=((0,10),(45,55),(95,105),(145,-1)), plot=False)
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']
    X=oX.copy()
    Y=oY.copy()

    print("X({}){}".format([c['name'] for c in coords],X.shape))
    print("Y={}".format(Y.shape))
    print("fs={}".format(fs))

    # try as simple linear classification problem
    from sklearn.linear_model import Ridge, LogisticRegression, LogisticRegressionCV
    from sklearn.svm import LinearSVR, LinearSVC
    # trial average
    Xc = np.sum(X,axis=-2)
    Yc = np.argmax(Y[:,0,0,:],axis=-1)
    clsfr = LogisticRegression(C=1e8) # turn off regularization
    clsfr.fit(Xc,Yc)
    Yest = clsfr.predict(Xc)
    acc = sum([i==j for i, j in zip(Yc,Yest)]) / len(Yc); print("Acc:{}".format(acc))
    W = clsfr.coef_
    b = clsfr.intercept_
    Fe = np.einsum("Td,ed->Te",Xc,W) + b
    Fy = Fe
    Yest = np.argmax(Fy,-1)
    acc = sum([i==j for i, j in zip(Yc,Yest)]) / len(Yc); print("Acc:{}".format(acc))

    # per-sample
    clsfr = LogisticRegression(C=1e8)
    Xc = X.reshape((-1,X.shape[-1]))
    Yc = np.argmax(Y[...,0,:],axis=-1).reshape((-1,))
    clsfr.fit(Xc,Yc)
    W = clsfr.coef_
    b = clsfr.intercept_
    Fe = np.einsum("Td,ed->Te",Xc,W) + b
    Fy = Fe
    Yest = np.argmax(Fy,-1)
    acc = sum([i==j for i, j in zip(Yc,Yest)]) / len(Yc); print('Acc: {}'.format(acc))
    
    
    # visualize the dataset
    from analyse_datasets import debug_test_dataset
    debug_test_dataset(X, Y, coords, tau_ms=20, evtlabs=None, rank=15, model='cca')
    

    
    
if __name__=="__main__":
    testcase()
