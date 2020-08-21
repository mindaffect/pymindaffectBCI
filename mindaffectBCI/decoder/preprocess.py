from utils import idOutliers, butter_sosfilt
from mindaffectBCI.decoder.updateSummaryStatistics import updateCxx
from mindaffectBCI.decoder.multipleCCA import robust_whitener
import numpy as np

def preprocess(X, Y, coords, whiten=False, whiten_spectrum=False, badChannelThresh=None, badTrialThresh=None, center=False, car=False, stopband=None):
    """apply simple pre-processing to an input dataset

    Args:
        X ([type]): the EEG data (tr,samp,d)
        Y ([type]): the stimulus (tr,samp,e)
        coords ([type]): [description]
        whiten (float, optional): if >0 then strength of the spatially regularized whitener. Defaults to False.
        whiten_spectrum (float, optional): if >0 then strength of the spectrally regularized whitener. Defaults to False.
        badChannelThresh ([type], optional): threshold in standard deviations for detection and removal of bad channels. Defaults to None.
        badTrialThresh ([type], optional): threshold in standard deviations for detection and removal of bad trials. Defaults to None.
        center (bool, optional): flag if we should temporally center the data. Defaults to False.
        car (bool, optional): flag if we should spatially common-average-reference the data. Defaults to False.

    Returns:
        X ([type]): the EEG data (tr,samp,d)
        Y ([type]): the stimulus (tr,samp,e)
        coords ([type]): meta-info for the data
    """    
    if center:
        X = X - np.mean(X, axis=-2, keepdims=True)

    if badChannelThresh is not None:
        X, Y, coords = rmBadChannels(X, Y, coords, badChannelThresh)

    if badTrialThresh is not None:
        X, Y, coords = rmBadTrial(X, Y, coords, badTrialThresh)

    if car:
        print('CAR')
        X = X - np.mean(X, axis=-1, keepdims=True)

    if whiten>0:
        print("whiten:{}".format(whiten))
        X, W = spatially_whiten(X,reg=whiten)

    if stopband is not None:
        X, _, _ = butter_sosfilt(X,stopband,fs=coords[-2]['fs'])

    if whiten_spectrum > 0:
        print("Spectral whiten:{}".format(whiten_spectrum))
        X, W = spectrally_whiten(X, axis=-2, reg=whiten_spectrum)

    return X, Y, coords


def rmBadChannels(X:np.ndarray, Y:np.ndarray, coords, thresh=3.5):
    """remove bad channels from the input dataset

    Args:
        X ([np.ndarray]): the eeg data as (trl,sample,channel)
        Y ([np.ndarray]): the associated stimulus info as (trl,sample,stim)
        coords ([type]): the meta-info about the channel
        thresh (float, optional): threshold in standard-deviations for removal. Defaults to 3.5.

    Returns:
        X (np.ndarray)
        Y (np.ndarray)
        coords
    """    
    isbad, pow = idOutliers(X, thresh=thresh, axis=(0,1))
    keep = isbad[0,0,...]==False
    X=X[...,keep]

    #print("power={}".format(pow))
    if 'coords' in coords[-1] and coords[-1]['coords'] is not None:
        rmd = coords[-1]['coords'][isbad[0,0,...]]
        print("Bad Channels Removed: {} = {}={}".format(np.sum(isbad),np.flatnonzero(isbad[0,0,...]),rmd))

        coords[-1]['coords']=coords[-1]['coords'][keep]
    else:
        print("Bad Channels Removed: {} = {}".format(np.sum(isbad),np.flatnonzero(isbad[0,0,...])))

    if 'pos2d' in coords[-1] and coords[-1]['pos2d'] is not None:  
        coords[-1]['pos2d'] = coords[-1]['pos2d'][keep]

    return X,Y,coords

def rmBadTrial(X, Y, coords, thresh=3.5, verb=1):
    """[summary]

    Args:
        X ([np.ndarray]): the eeg data as (trl,sample,channel)
        Y ([np.ndarray]): the associated stimulus info as (trl,sample,stim)
        coords ([type]): the meta-info about the channel
        thresh (float, optional): threshold in standard-deviations for removal. Defaults to 3.5.
    Returns:
        X (np.ndarray)
        Y (np.ndarray)
        coords
    """
    isbad,pow = idOutliers(X, thresh=thresh, axis=(1,2))
    X=X[isbad[...,0,0]==False,...]
    Y=Y[isbad[...,0,0]==False,...]

    if 'coords' in coords[0]:
        rmd = coords[0]['coords'][isbad[...,0,0]]
        print("BadTrials Removed: {} = {}".format(np.sum(isbad),rmd))

        coords[0]['coords']=coords[0]['coords'][isbad[...,0,0]==False]
    return X,Y,coords


def spatially_whiten(X:np.ndarray, *args, **kwargs):
    """spatially whiten the nd-array X

    Args:
        X (np.ndarray): the data to be whitened, with channels/space in the *last* axis

    Returns:
        X (np.ndarray): the whitened X
        W (np.ndarray): the whitening matrix used to whiten X
    """    
    Cxx = updateCxx(None,X,None)
    W,_ = robust_whitener(Cxx, *args, **kwargs)
    X = X @ W #np.einsum("...d,dw->...w",X,W)
    return (X,W)

def spectrally_whiten(X:np.ndarray, reg=.01, axis=-2):
    """spatially whiten the nd-array X

    Args:
        X (np.ndarray): the data to be whitened, with channels/space in the *last* axis

    Returns:
        X (np.ndarray): the whitened X
        W (np.ndarray): the whitening matrix used to whiten X
    """    
    from scipy.fft import fft, ifft

    # TODO[]: add hanning window to reduce spectral leakage

    Fx = fft(X,axis=axis)
    H=np.abs(Fx)
    if Fx.ndim+axis > 0:
        H = np.mean(H, axis=tuple(range(Fx.ndim+axis))) # grand average spectrum (freq,d)

    # compute *regularized* whitener (so don't amplify low power noise components)
    W = 1./(H+np.max(H)*reg)

    # apply the whitener
    Fx = Fx * W 

    # map back to time-domain
    X = np.real(ifft(Fx,axis=axis))
    return (X,W)

def plot_grand_average_spectrum(X, fs:float, axis:int=-2, ch_names=None):
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    freqs, FX = welch(X, axis=axis, fs=fs, nperseg=fs//2, return_onesided=True, detrend=False) # FX = (nFreq, nch)
    print('FX={}'.format(FX.shape))
    plt.figure(18);plt.clf()
    muFX = np.median(FX,axis=0,keepdims=True)
    ylim = (0,2*np.median(np.max(muFX,axis=-2),axis=-1))
    plot_erp(muFX, ch_names=ch_names, evtlabs=None, times=freqs, ylim=ylim)
    plt.suptitle("Grand average spectrum")

def extract_envelope(X,fs,
                     stopband=None,whiten=True,filterbank=None,log=True,env_stopband=(10,-1),
                     verb=False, plot=False):
    """extract the envelope from the input data

    Args:
        X ([type]): [description]
        fs ([type]): [description]
        stopband ([type], optional): pre-filter stop band. Defaults to None.
        whiten (bool, optional): flag if we spatially whiten before envelope extraction. Defaults to True.
        filterbank ([type], optional): set of filters to apply to extract the envelope for each filter output. Defaults to None.
        log (bool, optional): flag if we return raw power or log-power. Defaults to True.
        env_stopband (tuple, optional): post-filter on the extracted envelopes. Defaults to (10,-1).
        verb (bool, optional): verbosity level. Defaults to False.
        plot (bool, optional): flag if we plot the result of each preprocessing step. Defaults to False.

    Returns:
        X: the extracted envelopes
    """                     
    from multipleCCA import robust_whitener
    from updateSummaryStatistics import updateCxx
    from utils import butter_sosfilt
    
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(100);plt.clf();plt.plot(X[:int(fs*10),:].copy());plt.title("raw")

    if not stopband is None:
        if verb > 0:  print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X,stopband,fs)
        if plot:plt.figure(101);plt.clf();plt.plot(X[:int(fs*10),:].copy());plt.title("hp+notch+lp")
        # preprocess -> spatial whiten
        # TODO[] : make this  fit->transform method
    
    if whiten:
        if verb > 0:  print("spatial whitener")
        Cxx = updateCxx(None,X,None)
        W,_ = robust_whitener(Cxx)
        X = np.einsum("sd,dw->sw",X,W)
        if plot:plt.figure(102);plt.clf();plt.plot(X[:int(fs*10),:].copy());plt.title("+whiten")            

    if not filterbank is None:
        if verb > 0:  print("Filterbank: {}".format(filterbank))
        if plot:plt.figure(103);plt.clf();
        # apply filter bank to frequency ranges into virtual channels
        Xs=[]
        # TODO: make a nicer shape, e.g. (tr,samp,band,ch)
        # TODO[]: check doesn't modify in place
        for bi,band in enumerate(filterbank):
            Xf, _, _ = butter_sosfilt(X.copy(),band,fs)
            Xs.append(Xf)
            if plot:plt.subplot(len(filterbank),1,bi+1);plt.plot(Xf[:int(fs*10),:]);plt.title("+filterbank {}".format(band))
        # stack the bands as virtual channels
        X = np.concatenate(Xs,-1)
        
    X = np.abs(X) # rectify
    if plot:plt.figure(104);plt.plot(X[:int(fs*10),:]);plt.title("+abs")

    if log:
        if verb > 0: print("log amplitude")
        X = np.log(np.maximum(X,1e-6))
        if plot:plt.figure(105);plt.clf();plt.plot(X[:int(fs*10),:]);plt.title("+log")

    if env_stopband is not None:
        if verb>0: print("Envelop band={}".format(env_stopband))
        X, _, _ = butter_sosfilt(X,env_stopband,fs) # low-pass = envelope extraction
        if plot:plt.figure(104);plt.clf();plt.plot(X[:int(fs*10),:]);plt.title("+env")
    return X

