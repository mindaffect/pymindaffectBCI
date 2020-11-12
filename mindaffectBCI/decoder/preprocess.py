#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
# This file is part of pymindaffectBCI <https://github.com/mindaffect/pymindaffectBCI>.
#
# pymindaffectBCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pymindaffectBCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pymindaffectBCI.  If not, see <http://www.gnu.org/licenses/>

from mindaffectBCI.decoder.utils import idOutliers, butter_sosfilt
from mindaffectBCI.decoder.updateSummaryStatistics import updateCxx
from mindaffectBCI.decoder.multipleCCA import robust_whitener
import numpy as np

def preprocess(X, Y, coords, fs=None, whiten=False, whiten_spectrum=False, decorrelate=False, badChannelThresh=None, badTrialThresh=None, center=False, car=False, standardize=False, stopband=None, filterbank=None, nY=None, fir=None):
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
        reg = whiten if not isinstance(whiten,bool) else 0
        print("whiten:{}".format(reg))
        X, W = spatially_whiten(X,reg=reg)

    if stopband is not None and stopband is not False:
        X, _, _ = butter_sosfilt(X,stopband,fs=coords[-2]['fs'])

    if whiten_spectrum > 0:
        reg = whiten_spectrum if not isinstance(whiten_spectrum,bool) else .1
        print("Spectral whiten:{}".format(reg))
        X, W = spectrally_whiten(X, axis=-2, reg=reg)

    if decorrelate > 0:
        reg = decorrelate if not isinstance(decorrelate,bool) else .4
        print("Temporally decorrelate:{}".format(reg))
        X, W = temporally_decorrelate(X, axis=-2, reg=reg)

    if standardize > 0:
        reg = standardize if not isinstance(standardize,bool) else 1
        print("Standardize channel power:{}".format(reg))
        X, W = standardize_channel_power(X, axis=-2, reg=reg)

    if filterbank is not None and filterbank is not False:
        if fs is None and coords is not None: 
            fs = coords[-2]['fs']
        #X, _, _ = butter_filterbank(X,filterbank,fs=fs)
        X = fft_filterbank(X,filterbank,fs=fs)
        # make filterbank entries into virtual channels
        X = np.reshape(X, X.shape[:-2]+(-1,))
        # update meta-info
        if coords is not None and 'coords' in coords[-1] and coords[-1]['coords'] is not None:
            ch_names = coords[-1]['coords']
            ch_names = ["{}_{}".format(c,f) for f in filterbank for c in coords]

    if fir is not None:
        X = fir(X,**fir)
        # make taps into virtual channels
        X = np.reshape(X, X.shape[:-2]+(-1,))
        # update meta-info
        if coords is not None and 'coords' in coords[-1] and coords[-1]['coords'] is not None:
            ch_names = coords[-1]['coords']
            ch_names = ["{}_{}".format(c,f) for f in ntap for c in coords]

    if nY is not None:
        Y = Y[...,:nY+1,:]

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
    print("Ch-power={}".format(pow.ravel()))
    keep = isbad[0,0,...]==False
    X=X[...,keep]

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
    print("Trl-power={}".format(pow.ravel()))
    X=X[isbad[...,0,0]==False,...]
    Y=Y[isbad[...,0,0]==False,...]

    if 'coords' in coords[0] and np.sum(isbad) > 0:
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


def fir(X:np.ndarray, ntap=3, dilation=1):
    from mindaffectBCI.decoder.utils import window_axis
    X = window_axis(X, axis=-2, winsz=ntap*dilation)
    if dilation > 1:
        X = X[...,::dilation,:] # extract the dilated points    
    return X

def standardize_channel_power(X:np.ndarray, sigma2:np.ndarray=None, axis=-2, reg=1e-1, alpha=1e-3):
    """Adaptively standardize the channel powers

    Args:
        X (np.ndarray): The data to standardize
        sigma2 (np.ndarray, optional): previous channel powers estimates. Defaults to None.
        axis (int, optional): dimension of X which is time. Defaults to -2.
        reg ([type], optional): Regularisation strength for power estimation. Defaults to 1e-1.
        alpha ([type], optional): learning rate for power estimation. Defaults to 1e-3.

    Returns:
        sX: the standardized version of X
        sigma2 : the estimated channel power at the last sample of X
    """
    assert axis==-2, "Only currently implemeted for axis==-2"

    # 3d-X recurse over trials 
    if X.ndim == 3:
        for i in range(X.shape[0]):
            X[i,...], sigma2 = standardize_channel_power(X[i,...], sigma2=sigma2, reg=reg, alpha=alpha)
        return X, sigma2

    if sigma2 is None:
        sigma2 = np.zeros((X.shape[-1],), dtype=X.dtype)
        sigma2 = X[0,:]*X[0,:] # warmup with 1st sample power

    # 2-d X
    # return copy to don't change in-place!
    sX = np.zeros(X.shape,dtype=X.dtype)
    for t in range(X.shape[axis]):
        # TODO[] : robustify this, e.g. clipping/windsorizing
        sigma2 = sigma2 * (1-alpha) + X[t,:]*X[t,:]*alpha
        sigma2[sigma2==0] = 1
        # set to unit-power - but regularize to stop maginfication of low-power, i.e. noise, ch
        sX[t,:] = X[t,:] / np.sqrt((sigma2 + reg*np.median(sigma2))/2)

    return sX,sigma2


def temporally_decorrelate(X:np.ndarray, W:np.ndarray=50, reg=.5, eta=1e-7, axis=-2, verb=0):
    """temporally decorrelate each channel of X by fitting and subtracting an AR model

    Args:
        X (np.ndarray trl,samp,d): the data to be whitened, with channels/space in the *last* axis
        W ( tau,d): per channel AR coefficients
        reg (float): regularization strength for fitting the AR model. Defaults to 1e-2
        eta (float): learning rate for the SGD. Defaults to 1e-5

    Returns:
        X (np.ndarray): the whitened X
        W (np.ndarray (tau,d)): the AR model used to sample ahead predict X
    """    
    assert axis==-2, "Only currently implemeted for axis==-2"
    if W is None:  W=10
    if isinstance(W,int):
        # set initial filter and order
        W = np.zeros((W,X.shape[-1]))
        W[-1,:]=1

    if X.ndim > 2: # 3-d version, loop and recurse
        wX = np.zeros(X.shape,dtype=X.dtype)
        for i in range(X.shape[0]):
            # TODO[]: why does propogating the model between trials reduce the decorrelation effectivness?
            wX[i,...], W = temporally_decorrelate(X[i,...],W=W,reg=reg,eta=eta,axis=axis,verb=verb)
        return wX, W
    
    # 2-d X
    wX = np.zeros(X.shape,dtype=X.dtype)
    dH = np.ones(X.shape[-1],dtype=X.dtype)
    for t in range(X.shape[-2]):
        if t < W.shape[0]:
            wX[t,:] = X[t,:]

        else:
            Xt = X[t,:] # current input (d,)
            Xtau = X[t-W.shape[0]:t,:] # current prediction window (N,d)


            # compute the prediction error:
            Xt_est = np.sum(W*Xtau,-2) # (d,)
            err = Xt - Xt_est # (d,)

            if t<W.shape[0]+0 or verb>1:
                print('Xt={} Xt_est={} err={}'.format(Xt[0],Xt_est[0],err[0]))

            # remove the predictable part => decorrelate with the window
            wX[t,:] = Xt - reg * Xt_est # y=x - w'x_tau

            # smoothed diag hessian estimate
            dH = dH*(1-eta) + eta * Xt*Xt
        
            # update the linear prediction model - via. SGD
            W = W + eta * (err * Xtau / dH ) #- reg * W)  # w = w + eta x*x_tau

    return (wX,W)


def butter_filterbank(X:np.ndarray, filterbank, fs:float, axis=-2, order=4, ftype='butter', verb=1):
    if verb > 0:  print("Filterbank: {}".format(filterbank))
    if not axis == -2:
        raise ValueError("axis other than -2 not supported yet!")
    if sos is None:
        sos = [None]*len(filterbank)
    if zi is None:
        zi = [None]*len(filterbank)
    # apply filter bank to frequency ranges into virtual channels
    Xf=np.zeros(X.shape[:axis+1]+(len(filterbank),)+X.shape[axis+1:],dtype=X.dtype)
    for bi,stopband in enumerate(filterbank):
        if verb>1: print("{}) band={}\n".format(bi,stopband))
        if sos[bi] is None:
            Xf[...,bi,:], sos[bi], zi[bi] = butter_sosfilt(X.copy(),stopband=stopband,axis=axis,fs=fs,order=order,ftype=ftype)
        else:
            Xf[...,bi,:], sos[bi], zi[bi] = sosfilt(sos[bi],X.copy(),zi=zi[bi],axis=axis)

    # TODO[X]: make a nicer shape, e.g. (tr,samp,band,ch)
    #X = np.concatenate([X[...,np.newaxis,:] for X in Xs],-2)
    return Xf, sos, zi

def fft_filterbank(X:np.ndarray, filterbank, fs:float, axis=-2, verb=1):
    from scipy.signal import sosfilt
    if verb > 0:  print("Filterbank: {}".format(filterbank))

    # apply filter bank to frequency ranges into virtual channels
    Xf=np.zeros(X.shape[:axis+1]+(len(filterbank),)+X.shape[axis+1:],dtype=X.dtype)
    Fx = np.fft.fft(X,axis=axis)
    freqs = np.fft.fftfreq(X.shape[axis], d=1/fs)
    for bi,stopband in enumerate(filterbank):
        if verb>1: print("{}) band={}\n".format(bi,stopband))
        mask = np.logical_and(stopband[0] <= np.abs(freqs), np.abs(freqs) < stopband[1])
        Xf[...,bi,:] = np.fft.ifft(Fx*mask[:,np.newaxis], axis=axis).real
    return Xf

def plot_grand_average_spectrum(X, fs:float, axis:int=-2, ch_names=None, log=False):
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    freqs, FX = welch(X, axis=axis, fs=fs, nperseg=fs//2, return_onesided=True, detrend=False) # FX = (nFreq, nch)
    print('FX={}'.format(FX.shape))
    #plt.figure(18);plt.clf()
    muFX = np.median(FX,axis=0,keepdims=True)
    if log:
        muFX = 10*np.log10(muFX)
        unit='db (10*log10(uV^2))'
        ylim = (2*np.median(np.min(muFX,axis=tuple(range(muFX.ndim-1))),axis=-1),
                2*np.median(np.max(muFX,axis=tuple(range(muFX.ndim-1))),axis=-1))
    else:
        unit='uV^2'
        ylim = (0,2*np.median(np.max(muFX,axis=tuple(range(muFX.ndim-1))),axis=-1))

    plot_erp(muFX, ch_names=ch_names, evtlabs=None, times=freqs, ylim=ylim)       
    plt.suptitle("Grand average spectrum ({})".format(unit))

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
    from mindaffectBCI.decoder.multipleCCA import robust_whitener
    from mindaffectBCI.decoder.updateSummaryStatistics import updateCxx
    from mindaffectBCI.decoder.utils import butter_sosfilt
    
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
        X = filterbank(X,filterbank,fs)
        # make filterbank entries into virtual channels
        X = np.reshape(X,X.shape[:-2]+(prod(X.shape[-2:],)))
        
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


def testCase_spectralwhiten():
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    fs=100
    X = np.random.standard_normal((2,fs*3,2)) # flat spectrum
    X = X[:,:-1,:]+X[:,1:,:] # weak low-pass
    #X = np.cumsum(X,-2) # 1/f spectrum
    print("X={}".format(X.shape))
    plt.figure(1)
    plot_grand_average_spectrum(X, fs)
    plt.suptitle('Raw')
    plt.show(block=False)

    wX, _ = spectrally_whiten(X)
    
    # compare raw vs summed filterbank
    plt.figure(2)
    plot_grand_average_spectrum(wX,fs)
    plt.suptitle('Whitened')
    plt.show()


def testCase_temporallydecorrelate(X=None,fs=100):
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    fs=100
    if X is None:
        X = np.random.standard_normal((2,fs*3,2)) # flat spectrum
        #X = X + np.sin(np.arange(X.shape[-2])*2*np.pi/10)[:,np.newaxis]
        X = X[:,:-1,:]+X[:,1:,:] # weak low-pass

        #X = np.cumsum(X,-2) # 1/f spectrum
    print("X={}".format(X.shape))
    plt.figure(1)
    plot_grand_average_spectrum(X, fs)
    plt.suptitle('Raw')
    plt.show(block=False)

    wX, _ = temporally_decorrelate(X)
    
    # compare raw vs summed filterbank
    plt.figure(2)
    plot_grand_average_spectrum(wX,fs)
    plt.suptitle('Decorrelated')
    plt.show()


def testCase_filterbank():
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    fs=100
    X = np.random.standard_normal((2,fs*3,2)) # flat spectrum
    X = X[:,:-1,:]+X[:,1:,:] # weak low-pass
    #X = np.cumsum(X,-2) # 1/f spectrum
    print("X={}".format(X.shape))
    #plt.figure()
    #plot_grand_average_spectrum(X, fs)
    #plt.show()


    bands = ((1,10,'bandpass'),(10,20,'bandpass'),(20,40,'bandpass'))
    #Xf,Yf,coordsf = preprocess(X, None, None, filterbank=bands, fs=100)
    #Xf, _, _ = butter_filterbank(X,bands,fs,order=3,ftype='butter') # tr,samp,band,ch
    Xf = fft_filterbank(X,bands,fs=100)
    print("Xf={}".format(Xf.shape))
    # bands -> virtual channels
    # make filterbank entries into virtual channels
    plt.figure()
    Xf_s = np.sum(Xf,-2,keepdims=False)
    plot_grand_average_spectrum(np.concatenate((X[:,np.newaxis,...],Xf_s[:,np.newaxis,...],np.moveaxis(Xf,(0,1,2,3),(0,2,1,3))),1), fs)
    plt.legend(('X','Xf_s','X_bands'))
    plt.show()
    
    # compare raw vs summed filterbank
    plt.figure()
    plot_erp(np.concatenate((X[0:1,...],Xf_s[0:1,...],np.moveaxis(Xf[0,...],(0,1,2),(1,0,2))),0),
             evtlabs=['X','Xf_s']+['Xf_{}'.format(b) for b in bands])
    plt.suptitle('X, Xf_s')
    plt.show()

def test_fir():
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    fs=100
    X = np.random.standard_normal((2,fs*3,2)) # flat spectrum
    X = X[:,:-1,:]+X[:,1:,:] # weak low-pass
    #X = np.cumsum(X,-2) # 1/f spectrum
    print("X={}".format(X.shape))
    #plt.figure()
    #plot_grand_average_spectrum(X, fs)
    #plt.show()


    bands = ((1,10,'bandpass'),(10,20,'bandpass'),(20,40,'bandpass'))
    Xf,Yf,coordsf = preprocess(X, None, None, fir=dict(ntap=3,dilation=3), fs=100)
    #Xf, _, _ = butter_filterbank(X,bands,fs,order=3,ftype='butter') # tr,samp,band,ch
    print("Xf={}".format(Xf.shape))
    # bands -> virtual channels
    # make filterbank entries into virtual channels
    plt.figure()
    Xf_s = np.sum(Xf,-2,keepdims=False)
    plot_grand_average_spectrum(np.concatenate((X[:,np.newaxis,...],Xf_s[:,np.newaxis,...],np.moveaxis(Xf,(0,1,2,3),(0,2,1,3))),1), fs)
    plt.legend(('X','Xf_s','X_bands'))
    plt.show()

if __name__=="__main__":

    savefile = '~/Desktop/mark/mindaffectBCI*.txt'

    import glob
    import os
    files = glob.glob(os.path.expanduser(savefile)); 
    #os.path.join(os.path.dirname(os.path.abspath(__file__)),fileregexp)) # * means all if need specific format then *.csv
    savefile = max(files, key=os.path.getctime)

    # load
    from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
    X, Y, coords = load_mindaffectBCI(savefile, stopband=((45,65),(5.5,25,'bandpass')), order=6, ftype='butter', fs_out=100)
    # output is: X=eeg, Y=stimulus, coords=meta-info about dimensions of X and Y
    print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,coords[1]['fs']))
    print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:1]]+['output'],Y.shape))


    testCase_temporallydecorrelate(X)
    #testCase_spectralwhiten()
