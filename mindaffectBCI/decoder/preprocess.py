#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jadref@gmail.com>
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

from os import fsdecode
from mindaffectBCI.decoder.utils import idOutliers, butter_sosfilt, block_permute, InfoArray, pool_axis
from mindaffectBCI.decoder.updateSummaryStatistics import updateCxx, plot_trial
from mindaffectBCI.decoder.multipleCCA import robust_whitener
import copy
import numpy as np

try:
    from sklearn.base import TransformerMixin
except:
    # fake the class if sklearn is not available, e.g. Android/iOS
    class TransformerMixin:
        def __init__():
            pass
        def fit(self,X):
            pass
        def transform(self,X):
            pass

def preprocess(X_TSd, Y_TSy, coords, pipeline:list=None, **kwargs):
    if pipeline is None:
        return preprocess_old(X_TSd, Y_TSy, coords, **kwargs)
    else:
        return preprocess_pipeline(X_TSd, Y_TSy, coords, pipeline=pipeline, **kwargs)


def preprocess_pipeline(X_TSd, Y_TSy, coords, pipeline:list=None, fs=None, ch_names=None, test_idx=None):
    """wrapper function to make a preprocess pipeline, apply it and return the preprocessed data

    Args:
        X_TSd ([type]): EEG data
        Y_TSy ([type]): stimulus info
        coords ([type]): meta info
        pipeline (list, optional): pipeline to apply as list of 2-uples with (name:str, args:dict) see `preprocess_transforms.py` for examples/stage names. Defaults to None.
        fs ([type], optional): sample rate of X/Y. Overrides coords. Defaults to None.
        ch_names ([type], optional): channel names for X.  Overrides coords. Defaults to None.
        test_idx ([type], optional): test trails -- if given then only fit parameters on non-test idx. Defaults to None.

    Returns:
        tuple: (X_TSd, Y_TSy, coords) the preprocessed data/targets
    """
    from mindaffectBCI.decoder.preprocess_transforms import make_preprocess_pipeline

    if fs is None and coords is not None: 
        fs = coords[-2]['fs']
    if ch_names is None and coords is not None: 
        ch_names = coords[-1]['coords']

    X_TSd = InfoArray(X_TSd, info=dict(fs=fs,ch_names=ch_names,coords=coords))
    Y_TSy = InfoArray(Y_TSy, info=dict(fs=fs))

    ppp = make_preprocess_pipeline(pipeline)
    print(ppp)

    if test_idx is not None:
        trn_idx = np.ones(X_TSd.shape[0],bool)
        trn_idx[test_idx]=False
        ppp.fit(X_TSd[trn_idx,...], Y_TSy[trn_idx,...])
        X_TSd, Y_TSy = ppp.modify(X_TSd.copy(), Y_TSy.copy())
    else:
        X_TSd, Y_TSy = ppp.fit_modify(X_TSd.copy(), Y_TSy.copy())
    
    if coords is not None and hasattr(X_TSd,'info'):
        coords[1]['fs'] = X_TSd.info['fs']
        coords[-1]['coords'] = X_TSd.info['ch_names']

    return X_TSd, Y_TSy, coords



def preprocess_old(X_TSd, Y_TSy, coords, fs=None, test_idx=None, *args):
    """apply simple pre-processing to an input dataset

    Args:
        X ([type]): the EEG data (tr,samp,d)
        Y_TSy ([type]): the stimulus (tr,samp,e)
        coords ([type]): [description]
        args (list): list of 2-uples pre-processing functions to apply, in the format (pipelinestagename:str, arg:dict)

        filterband (list): list of bandpass filters to apply to the data.  Defaults to None.
        prewhiten (float, optional): spatially whiten before band-pass if >0 then strength of the spatially regularized whitener. Defaults to False.
        whiten (float, optional): if >0 then strength of the spatially regularized whitener. Defaults to False.
        whiten_spectrum (float, optional): if >0 then strength of the spectrally regularized whitener. Defaults to False.
        badChannelThresh ([type], optional): threshold in standard deviations for detection and removal of bad channels. Defaults to None.
        badTrialThresh ([type], optional): threshold in standard deviations for detection and removal of bad trials. Defaults to None.
        center (bool, optional): flag if we should temporally center the data. Defaults to False.
        car (bool, optional): flag if we should spatially common-average-reference the data. Defaults to False.

    Returns:
        X ([type]): the EEG data (tr,samp,d)
        Y_TSy ([type]): the stimulus (tr,samp,e)
        coords ([type]): meta-info for the data
    """
    if fs is None and coords is not None: 
        fs = coords[-2]['fs']

    if args is None:
        return X_TSd, Y_TSy, coords

    for si, stage in enumerate(args):
        if isinstance(stage,str):
            stagename=stage
            args=dict()
        else:
            stagename, args = stage
        stagename = stagename.lower()
        print("{}) {} X.shape={}".format(si,stagename,X_TSd.shape))

        if stagename == 'center':
            print('center')
            X_TSd = X_TSd - np.mean(X_TSd, axis=-2, keepdims=True)

        elif stagename == 'rmBadChannels'.lower():
            print('rmbadch')
            X_TSd, Y_TSy, coords = rmBadChannels(X_TSd, Y_TSy, coords, **args)

        elif stagename == 'rmBadTrial'.lower():
            print('rmbadtr')
            X_TSd, Y_TSy, coords = rmBadTrial(X_TSd, Y_TSy, coords, **args)

        elif stagename == 'car':
            print('CAR')
            X_TSd = X_TSd - np.mean(X_TSd, axis=-1, keepdims=True)

        elif stagename == 'spatially_whiten':
            print("spatially_whiten: {}".format(args))
            X_TSd, W = spatially_whiten(X_TSd, test_idx=test_idx, **args)

        elif stagename == 'butter_sosfilt':
            print('filter: {} @ {}'.format(args,fs))
            X_TSd, _, _ = butter_sosfilt(X_TSd,fs=fs, **args)

        elif stagename == 'fftfilter':
            print('fftfilter: {} @ {}'.format(args,fs))
            X_TSd = fftfilter(X_TSd,fs=fs, **args)

        elif stagename == 'ola_fftfilter':
            print('ola_fftfilter: {} @ {}'.format(args,fs))
            X_TSd = ola_fftfilter(X_TSd,fs=fs, **args)

        elif stagename in ('fftfilterbank', 'ola_fftfilterbank'):
            print('filterbank: {}'.format(args))
            args = copy.deepcopy(args)
            filterbank = args.pop('filterbank',None) # grab the key argument
            whiten_band = args.pop('whiten_band',False)
            if stagename.startswith('ola'):
                X_TSd = ola_fftfilterbank(X_TSd, filterbank=filterbank, fs=fs, **args) # X_TSfd
            else:
                X_TSd = fftfilterbank(X_TSd, filterbank=filterbank, fs=fs, **args) # X_TSfd

            # BODGE []: apply per-band whitener
            if whiten_band:
                print("whiten_band: {}".format(whiten_band))
                for fi in range(X_TSd.shape[-2]):
                    print('{}'.format(fi),end='')
                    X_TSd[...,fi,:], _ = spatially_whiten(X_TSd[...,fi,:], whiten=whiten_band, symetric=True, test_idx=test_idx)
                print()

            # make filterbank entries into virtual channels
            X_TSd = np.reshape(X_TSd, X_TSd.shape[:-2]+(-1,))
            # update meta-info
            if coords is not None and 'coords' in coords[-1] and coords[-1]['coords'] is not None:
                ch_names = coords[-1]['coords']
                ch_names = ["{}_{}".format(c,f) for f in filterbank for c in ch_names]
                coords[-1]['coords']=ch_names

        elif stagename == 'resample':
            print("resample: {}".format(args))
            X_TSd, Y_TSy, coords = resample(X_TSd, Y_TSy, coords, **args)

        elif stagename == 'block_cov_mx':
            print('block_cov_mx :  {} @{}hz'.format(args,fs))
            X_TSd, Y_TSy, coords = block_cov(X_TSd, Y_TSy, coords, fs=fs, **args)

            # make cov matrix entries into virtual channels
            X_TSd = np.reshape(X_TSd, X_TSd.shape[:-2]+(-1,))
            # update meta-info
            if coords is not None and 'coords' in coords[-1] and coords[-1]['coords'] is not None:
                ch_names = coords[-1]['coords']
                ch_names = ["{}_{}".format(c,f) for f in ch_names for c in ch_names]
                coords[-1]['coords']=ch_names

        elif stagename in ('adaptive_whiten','adaptive_spatially_whiten'):
            print("adaptive_whiten:{}".format(args))
            X_TSd, W = adaptive_spatially_whiten(X_TSd,test_idx=test_idx, **args)

        elif stagename == 'whiten_spectrum':
            print("Spectral whiten:{}".format(args))
            X_TSd, W = spectrally_whiten(X_TSd, axis=-2, test_idx=test_idx, **args)

        elif stagename == 'decorrelate':
            print("Temporally decorrelate:{}".format(args))
            X_TSd, W = temporally_decorrelate(X_TSd, axis=-2, **args)

        elif stagename == 'standardize':
            print("Standardize channel power:{}".format(args))
            X_TSd, W = standardize_channel_power(X_TSd, axis=-2, **args)

        elif stagename == 'log':
            print('log')
            X_TSd = np.log(np.maximum(X_TSd,1e-3,dtype=X_TSd.dtype),dtype=X_TSd.dtype)

        elif stagename == 'fir':
            X_TSd = temporal_embedding(X_TSd,**args)
            # make taps into virtual channels
            ntaps = X_TSd.shape[-2]
            X_TSd = np.reshape(X_TSd, X_TSd.shape[:-2]+(-1,))
            # update meta-info
            if coords is not None and 'coords' in coords[-1] and coords[-1]['coords'] is not None:
                ch_names = coords[-1]['coords']
                ch_names = ["{}_{}".format(c,f) for f in ntaps for c in ch_names]
                coords[-1]['coords']=ch_names


        elif stagename == 'centery':
            print('centerY')
            Y_TSy = centerY(Y_TSy,axis=-1)

        elif stagename == 'ny':
            Y_TSy = Y_TSy[...,:args['nY']+1]

        elif stagename == 'add_virtual_outputs':
            print("add virtual outputs: {}".format(args))
            X_TSd, Y_TSy, coords = add_virtual_outputs(X_TSd, Y_TSy, coords, **args)
        
        else:
            raise ValueError("Unrecognised pp name {}".format(stagename))

    return X_TSd, Y_TSy, coords


def resample(X:np.ndarray, Y:np.ndarray, coords, axis=-2, resamprate:int=None, fs:float=None, fs_out:float=None, nsamp:int=None):
    assert axis==-2
    if resamprate is None:
        fs = coords[axis]['fs'] if fs is None else fs
        resamprate = int(round(fs*2.0/fs_out))/2.0
    # number samples through this cycle due to remainder of last block
    resamp_start = nsamp%resamprate if nsamp else 0
    # convert to number samples needed to complete this cycle
    # this is then the sample to take for the next cycle
    if resamp_start > 0:
        resamp_start = resamprate - resamp_start
    
    # allow non-integer resample rates
    idx =  np.arange(resamp_start,X.shape[axis],resamprate,dtype=X.dtype)

    if resamprate%1 > 0 and idx.size>0 : # non-integer re-sample, interpolate
        idx_l = np.floor(idx).astype(int) # sample above
        idx_u = np.ceil(idx).astype(int) # sample below
        # BODGE: guard for packet ending at sample boundary.
        idx_u[-1] = idx_u[-1] if idx_u[-1]<X.shape[axis] else X.shape[axis]-1
        w_u   = (idx - idx_l).astype(X.dtype) # linear weight of the upper sample
        X = X[...,idx_u,:] * w_u[:,np.newaxis] + X[...,idx_l,:] * (1-w_u[:,np.newaxis]) # linear interpolation
        if Y is not None:
            Y = Y[...,idx_u,:] * w_u[:,np.newaxis] + Y[...,idx_l,:] * (1-w_u[:,np.newaxis])

    else:
        idx = idx.astype(int)
        X = X[..., idx, :] # decimate X (trl, samp, d)
        if Y is not None:
            Y = Y[..., idx, :] # decimate Y (trl, samp, y)
    
    # update coords
    if coords and 'fs' in coords[axis]:
        coords[axis]['fs'] = coords[axis]['fs'] / resamprate

    return X, Y, coords


def rmBadChannels(X_TSd:np.ndarray, Y_TSy:np.ndarray, coords, thresh=3.5):
    """remove bad channels from the input dataset

    Args:
        X_TSd ([np.ndarray]): the eeg data as (trl,sample,channel)
        Y_TSy ([np.ndarray]): the associated stimulus info as (trl,sample,stim)
        coords ([type]): the meta-info about the channel
        thresh (float, optional): threshold in standard-deviations for removal. Defaults to 3.5.

    Returns:
        X_TSd (np.ndarray)
        Y_TSy (np.ndarray)
        coords
    """    
    isbad, pow = idOutliers(X_TSd, thresh=thresh, axis=(0,1))
    print("Ch-power={}".format(pow.ravel()))
    keep = isbad[0,0,...]==False
    X_TSd=X_TSd[...,keep]

    if 'coords' in coords[2] and coords[2]['coords'] is not None:
        rmd = [ c for i,c in enumerate(coords[2]['coords']) if keep[i] ] 
        print("Bad Channels Removed: {} = {}={}".format(np.sum(isbad),np.flatnonzero(keep==True),rmd))

        coords[2]['coords']=[ c for i,c in enumerate(coords[2]['coords']) if keep[i] ]
    else:
        print("Bad Channels Removed: {} = {}".format(np.sum(isbad),np.flatnonzero(isbad[0,0,...])))

    if 'pos2d' in coords[2] and coords[2]['pos2d'] is not None:  
        coords[2]['pos2d'] = [ p for i,p in enumerate(coords[2]['pos2d']) if keep[i] ]

    return X_TSd,Y_TSy,coords

def rmBadTrial(X_TSd, Y_TSy, coords, thresh=3.5, verb=1):
    """[summary]

    Args:
        X_TSd ([np.ndarray]): the eeg data as (trl,sample,channel)
        Y_TSy ([np.ndarray]): the associated stimulus info as (trl,sample,stim)
        coords ([type]): the meta-info about the channel
        thresh (float, optional): threshold in standard-deviations for removal. Defaults to 3.5.
    Returns:
        X_TSd (np.ndarray)
        Y_TSy (np.ndarray)
        coords
    """
    isbad,pow = idOutliers(X_TSd, thresh=thresh, axis=(1,2))
    print("Trl-power={}".format(pow.ravel()))
    X_TSd=X_TSd[isbad[...,0,0]==False,...]
    Y_TSy=Y_TSy[isbad[...,0,0]==False,...]

    if 'coords' in coords[0] and np.sum(isbad) > 0:
        rmd = coords[0]['coords'][isbad[...,0,0]]
        print("BadTrials Removed: {} = {}".format(np.sum(isbad),rmd))

        coords[0]['coords']=coords[0]['coords'][isbad[...,0,0]==False]
    return X_TSd,Y_TSy,coords

def add_virtual_outputs(X_TSd:np.ndarray, Y_TSy:np.ndarray, coords:dict, nvirt_out:int=0):
    # make virtual targets
    # generate virtual outputs for testing -- not from the 'true' target though
    virt_Y = block_permute(Y_TSy, nvirt_out, axis=-1, perm_axis=-2) 
    print("Added {} virtual outputs".format(virt_Y.shape[-1]))
    Y_TSy = np.concatenate((Y_TSy, virt_Y), axis=-1)
    return X_TSd, Y_TSy, coords


def spatially_whiten(X_TSd:np.ndarray, whiten=True, symetric:bool=False, test_idx=None, **kwargs):
    """spatially whiten the nd-array X_TSd

    Args:
        X_TSd (np.ndarray): the data to be whitened, with channels/space in the *last* axis

    Returns:
        X_TSd (np.ndarray): the whitened X_TSd
        W (np.ndarray): the whitening matrix used to whiten X_TSd
    """
    rcond=1e-6
    if whiten == True:
        reg=0
        symetric=True
    elif whiten>0:
        reg = 1-whiten
        symetric=True
    elif whiten<0: # output rank
        reg =0
        rcond=whiten

    if test_idx is None:
        Xtrn = X_TSd
    else:
        trn_idx = np.ones(X_TSd.shape[0],bool)
        trn_idx[test_idx]=False
        Xtrn = X_TSd[trn_idx,...]        
    Cxx = updateCxx(None,Xtrn,None)
    W,_ = robust_whitener(Cxx, symetric=symetric, reg=reg, rcond=rcond, **kwargs)
    X_TSd = X_TSd @ W
    return (X_TSd,W)


def est_matrix_variate_gaussian(X_TSd:np.ndarray,axis=(-2,-1),rcond=1e-12):
    cov = [ np.eye(X_TSd.shape[i],X_TSd.shape[i]) for i in axis]
    icov = [ np.eye(X_TSd.shape[i],X_TSd.shape[i]) for i in axis]

    # build the index expressions
    sumidx = [ chr(ord('a')+i) for i in range(X_TSd.ndim)] # unique char idx 
    lsub = sumidx.copy();    rsub = sumidx.copy() 
    lsub[axis[0]]='u';       rsub[axis[0]]='v'
    lsub[axis[1]]='w';       rsub[axis[1]]='x'
    idxstr = ["{},{},{}".format("".join(lsub),'uv',"".join(rsub)),
              "{},{},{}".format("".join(lsub),'wx',"".join(rsub))]
    print('idxstrs={}'.format(idxstr))
    for iter in range(40):
        # axis[0]
        icov[0] = np.linalg.pinv(cov[0],rcond=rcond,hermitian=True)
        cov[1] = np.einsum(idxstr[0], X_TSd, icov[0], X_TSd)/X_TSd.shape[axis[0]]
        # axis[1]
        icov[1] = np.linalg.pinv(cov[1],rcond=rcond,hermitian=True)
        cov[0] = np.einsum(idxstr[1], X_TSd, icov[1], X_TSd)/X_TSd.shape[axis[1]]
        # balance norm over components
        norm = [ np.mean(np.diag(c)) for c in cov ]
        nf = np.sqrt(np.prod(norm))
        for i in range(len(cov)):
            cov[i] = cov[i] * nf / norm[i]
        # TODO[]: convergence testing   
    return cov

def centerY(Y_TSy,axis=-1):
    Y_TSy = Y_TSy - np.mean(Y_TSy,axis=axis,keepdims=True)
    return Y_TSy

def matrix_var_gaussian_test():
    nL=5; nR=9
    L = np.random.standard_normal((1000,nL))@np.random.standard_normal((nL,nL))
    R = np.random.standard_normal((1000,nR))@np.random.standard_normal((nR,nR))
    X = np.einsum("il,ir->ilr",L,R)

    covfull = X.reshape((X.shape[0],-1)).T@X.reshape((X.shape[0],-1))
    print("Full\n{}".format(covfull))
    print('{}'.format(np.diag(covfull)))
    plt.imshow(covfull)

    cov = est_matrix_variate_gaussian(X)
    for i,c in enumerate(cov):
        print("{})\n {}".format(i,c))

    decomp = np.kron(cov[0],cov[1])
    print('Decomp\n{}'.format(decomp))
    print("{}".format(np.diag(decomp)))
    plt.imshow(decomp);plt.colorbar()

    (covfull-decomp)/covfull
    plt.imshow(covfull-decomp);plt.colorbar()

def adaptive_spatially_whiten(X_TSd:np.ndarray, test_idx=None, wght=.1, symetric=True, **kwargs):
    """spatially whiten the nd-array X_TSd

    Args:
        X_TSd (np.ndarray): the data to be whitened, with channels/space in the *last* axis

    Returns:
        X_TSd (np.ndarray): the whitened X_TSd
        W (np.ndarray): the whitening matrix used to whiten X_TSd
    """
    N = 0
    Cxx = 0
    for ti in range(X_TSd.shape[0]):
        Cxxi = updateCxx(None,X_TSd[ti,...])
        Cxx = Cxx*wght + Cxxi 
        N   = N*wght + 1
        W,_ = robust_whitener(Cxx/N, symetric=symetric, **kwargs)
        X_TSd[ti,...] = X_TSd[ti,...] @ W
    return (X_TSd,W)


def spectrally_whiten(X_TSd:np.ndarray, reg=.01, axis=-2, test_idx=None):
    """spatially whiten the nd-array X_TSd

    Args:
        X_TSd (np.ndarray): the data to be whitened, with channels/space in the *last* axis

    Returns:
        X_TSd (np.ndarray): the whitened X_TSd
        W (np.ndarray): the whitening matrix used to whiten X_TSd
    """    
    from scipy.fft import fft, ifft

    # TODO[]: add hanning window to reduce spectral leakage

    Fx = fft(X_TSd,axis=axis)

    # limit data we fit the model on
    if test_idx is None:
        Fxtrn = Fx
    else:
        trn_idx = np.ones(X_TSd.shape[0],bool)
        trn_idx[test_idx]=False
        Fxtrn = Fx[trn_idx,...]        

    H=np.abs(Fxtrn)
    if Fx.ndim+axis > 0:
        H = np.mean(H, axis=tuple(range(Fx.ndim+axis))) # grand average spectrum (freq,d)

    # compute *regularized* whitener (so don't amplify low power noise components)
    W = 1./(H+np.max(H)*reg)

    # apply the whitener
    Fx = Fx * W 

    # map back to time-domain
    X_TSd = np.real(ifft(Fx,axis=axis))
    return (X_TSd,W)


def temporal_embedding(X_TSd:np.ndarray, ntap=3, dilation=1):
    from mindaffectBCI.decoder.utils import window_axis
    X_TSd = window_axis(X_TSd, axis=-2, winsz=ntap*dilation)
    if dilation > 1:
        X_TSd = X_TSd[...,::dilation,:] # extract the dilated points    
    return X_TSd

def standardize_channel_power(X_TSd:np.ndarray, sigma2:np.ndarray=None, axis=-2, reg=1e-1, alpha=1e-3):
    """Adaptively standardize the channel powers

    Args:
        X_TSd (np.ndarray): The data to standardize
        sigma2 (np.ndarray, optional): previous channel powers estimates. Defaults to None.
        axis (int, optional): dimension of X_TSd which is time. Defaults to -2.
        reg ([type], optional): Regularisation strength for power estimation. Defaults to 1e-1.
        alpha ([type], optional): learning rate for power estimation. Defaults to 1e-3.

    Returns:
        sX: the standardized version of X_TSd
        sigma2 : the estimated channel power at the last sample of X
    """
    assert axis==-2, "Only currently implemeted for axis==-2"

    # ensure 3-d input
    X_TSd = X_TSd.reshape( (-1,)*(3-X_TSd.ndim) + X_TSd.shape )

    if sigma2 is None:
        sigma2 = np.zeros((X_TSd.shape[-1],), dtype=X_TSd.dtype)
        sigma2 = X_TSd[0,0,:]*X_TSd[0,0,:] # warmup with 1st sample power

    # 2-d X_TSd
    # return copy to don't change in-place!
    sX = np.zeros(X_TSd.shape,dtype=X_TSd.dtype)
    for i in range(X_TSd.shape[0]):
        for t in range(X_TSd.shape[axis]):
            # TODO[] : robustify this, e.g. clipping/windsorizing
            sigma2 = sigma2 * (1-alpha) + X_TSd[i,t,:]*X_TSd[i,t,:]*alpha
            sigma2[sigma2==0] = 1
            # set to unit-power - but regularize to stop maginfication of low-power, i.e. noise, ch
            sX[i,t,:] = X_TSd[i,t,:] / np.sqrt((sigma2 + reg*np.median(sigma2))/2)

    return sX,sigma2


def temporally_decorrelate(X_TSd:np.ndarray, W:np.ndarray=50, reg=.5, eta=1e-7, axis=-2, verb=0):
    """temporally decorrelate each channel of X_TSd by fitting and subtracting an AR model

    Args:
        X_TSd (np.ndarray trl,samp,d): the data to be whitened, with channels/space in the *last* axis
        W ( tau,d): per channel AR coefficients
        reg (float): regularization strength for fitting the AR model. Defaults to 1e-2
        eta (float): learning rate for the SGD. Defaults to 1e-5

    Returns:
        X_TSd (np.ndarray): the whitened X_TSd
        W (np.ndarray (tau,d)): the AR model used to sample ahead predict X_TSd
    """    
    assert axis==-2, "Only currently implemeted for axis==-2"
    if W is None:  W=10
    if isinstance(W,int):
        # set initial filter and order
        W = np.zeros((W,X_TSd.shape[-1]))
        W[-1,:]=1

    if X_TSd.ndim > 2: # 3-d version, loop and recurse
        wX = np.zeros(X_TSd.shape,dtype=X_TSd.dtype)
        for i in range(X_TSd.shape[0]):
            # TODO[]: why does propogating the model between trials reduce the decorrelation effectivness?
            wX[i,...], W = temporally_decorrelate(X_TSd[i,...],W=W,reg=reg,eta=eta,axis=axis,verb=verb)
        return wX, W
    
    # 2-d X
    wX = np.zeros(X_TSd.shape,dtype=X_TSd.dtype)
    dH = np.ones(X_TSd.shape[-1],dtype=X_TSd.dtype)
    for t in range(X_TSd.shape[-2]):
        if t < W.shape[0]:
            wX[t,:] = X_TSd[t,:]

        else:
            Xt = X_TSd[t,:] # current input (d,)
            Xtau = X_TSd[t-W.shape[0]:t,:] # current prediction window (N,d)


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


def butter_filterbank(X_TSd:np.ndarray, filterbank, fs:float, axis=-2, order=4, ftype='butter', verb=1):
    # TODO[]: lag the banks to phase align for re-construction!
    if verb > 0:  print("Filterbank: {}".format(filterbank))
    if not axis == -2:
        raise ValueError("axis other than -2 not supported yet!")
    if sos is None:
        sos = [None]*len(filterbank)
    if zi is None:
        zi = [None]*len(filterbank)
    # apply filter bank to frequency ranges into virtual channels
    Xf=np.zeros(X_TSd.shape[:axis+1]+(len(filterbank),)+X_TSd.shape[axis+1:],dtype=X_TSd.dtype)
    for bi,filterband in enumerate(filterbank):
        if verb>1: print("{}) band={}\n".format(bi,filterband))
        if sos[bi] is None:
            Xf[...,bi,:], sos[bi], zi[bi] = butter_sosfilt(X_TSd.copy(),filterband=filterband,axis=axis,fs=fs,order=order,ftype=ftype)
        else:
            Xf[...,bi,:], sos[bi], zi[bi] = sosfilt(sos[bi],X_TSd.copy(),zi=zi[bi],axis=axis)

    # TODO[X]: make a nicer shape, e.g. (tr,samp,band,ch)
    #X = np.concatenate([X[...,np.newaxis,:] for X in Xs],-2)
    return Xf, sos, zi


def fir_filterbank(X_TSd:np.ndarray, filterbank, fs:float, axis=-2, order=4, ftype='butter', verb=1):
    if verb > 0:  print("Filterbank: {}".format(filterbank))
    if not axis == -2:
        raise ValueError("axis other than -2 not supported yet!")
    if sos is None:
        sos = [None]*len(filterbank)
    if zi is None:
        zi = [None]*len(filterbank)
    # apply filter bank to frequency ranges into virtual channels
    Xf=np.zeros(X_TSd.shape[:axis+1]+(len(filterbank),)+X_TSd.shape[axis+1:],dtype=X_TSd.dtype)
    for bi,filterband in enumerate(filterbank):
        if verb>1: print("{}) band={}\n".format(bi,filterband))
        if sos[bi] is None:
            Xf[...,bi,:], sos[bi], zi[bi] = butter_sosfilt(X.copy(),filterband=filterband,axis=axis,fs=fs,order=order,ftype=ftype)
        else:
            Xf[...,bi,:], sos[bi], zi[bi] = sosfilt(sos[bi],X.copy(),zi=zi[bi],axis=axis)

    # TODO[X]: make a nicer shape, e.g. (tr,samp,band,ch)
    #X = np.concatenate([X[...,np.newaxis,:] for X in Xs],-2)
    return Xf, sos, zi


def fftfilter_masks(fftlen:int,bands:list,fs:float,freq_sigma:float=1,dtype=np.float):
    """[summary]

    Args:
        fftlen (int): [description]
        bands ([type]): list of pass bands, specified as either (low_cut,high_cut, type) or (low_cut,low_pass,high_pass,high_cut, type)
        fs (float): sampling rate of the data
        freq_sigma (float, optional): side-band size if not explicitly given. Defaults to 1.
        dtype ([type], optional): type of the generated mask. Defaults to np.float.

    Returns:
        (masks, bandtypes, freqs): masks (#masks,fftlen) mask for each band specficiation,  bandtypes (fftlen,) type info for each mask
    """    
    if not hasattr(bands[0],'__iter__') or isinstance(bands[-1],str): bands=(bands,) # ensure is list of lists
    # compute the band masks
    freqs = np.fft.fftfreq(fftlen, d=1/fs)
    mask = np.zeros((len(bands),len(freqs)),dtype=dtype)
    bandtype = ['bandpass' for _ in range(len(bands))]
    for bi,band in enumerate(bands):
        if len(band)==fftlen: # allow to directly use a frequency weighting function
            mask[bi]=band
            continue

        # extract the type info
        if isinstance(band[-1],str):
            bandtype[bi] = band[-1]
            band = band[:-1]
        else:
            bandtype[bi] = 'bandpass'

        # normalize specfication to: low_cut, low_pass, high_pass, high_cut
        if len(band)==2: 
            if freq_sigma is not None: # low_pass-sigma, low_pass, high_pass, high_pass+sigma
                band = (band[0]-freq_sigma, band[0], band[1], band[1]+freq_sigma)
            else: 
                band = (band[0], band[0], band[1], band[1])
        elif len(band)==3: # low_cut, low_pass, low_pass, high_cut
            band = (band[0], band[1], band[1], band[2])
        band = [b if b>=0 else fs/2-b for b in band] # allow neg freq spec

        mask[bi,:] = np.logical_and(band[1] <= np.abs(freqs), np.abs(freqs) <= band[2])
        if band[0]<band[1]:
            # hanning sholders band in freq domain to reduce ringing artifacts
            idx = np.logical_and(band[0] <= np.abs(freqs), np.abs(freqs)< band[1])
            mask[bi,idx] = (np.abs(freqs[idx])-band[0])/(band[1]-band[0]) 
            #mask[bi,idx] = .5*(1 - np.cos(2*np.pi*mask[bi,idx]/4)) #np.exp(-.5* (np.abs(freqs[idx])-mu)**2/sigma2) 

        if band[2]<band[3]:
            idx = np.logical_and(band[2] < np.abs(freqs), np.abs(freqs)<= band[3])
            mask[bi,idx] = (band[3]-np.abs(freqs[idx]))/(band[3]-band[2]) #np.exp(-.5* (np.abs(freqs[idx])-mu)**2/sigma2) 
            #mask[bi,idx] = .5*(1 - np.cos(2*np.pi*mask[bi,idx]/4)) #np.exp(-.5* (np.abs(freqs[idx])-mu)**2/sigma2) 

        if bandtype[bi] == 'hilbert': # positive frequencies only
            mask[bi,freqs<=0] = 0
        elif bandtype[bi] == 'bandstop': # invert the mask
            mask[bi,:] = 1-mask[bi,:]
    return mask, bandtype, freqs

def get_window_step(blksz:int,window:str=None,overlap:float=.5,dtype=np.float32):
    """make a temporal window and compute step size for Constant Overlap Add property

    Args:
        blksz (int): [description]
        window (str, optional): type of window to make, one-of: rectangle, triangle, bartlet, hanning (raised cosine), hamming. Defaults to None.
             window functions from: https://www.dsprelated.com/freebooks/sasp/Overlap_Add_Decomposition.html
        overlap (float, optional): step size as overlap between windows. Defaults to .5.
        dtype ([type], optional): dtype of the produced window. Defaults to np.float32.

    Returns:
        tuple(np.ndarray,int): (blksz,) the computed window, step-size
    """    
    blksz = int(blksz)
    # TODO[]: BODGE ensure is even so equal sized shifts work!
    #blksz = blksz + blksz%2 
    step = int(blksz*(1-overlap))
    if window in ('tophat','rectangle') or window is None or window==1:
        window = np.ones((blksz,),dtype=dtype) # bartlet window = triangle
    elif window in ('bartlet','bartlett','triangle') or window is None:
        window = 1 - np.abs(np.linspace(-1,1,blksz,dtype=dtype)) # bartlet window = triangle
    elif window == 'hanning':
        window = .5*(1 - np.cos(2*np.pi*np.arange(1,blksz,dtype=dtype)/(blksz+1)))
    elif window == 'hamming' or window is None:
        if blksz%2==0:
            window = .54 - .46*np.cos(2*np.pi*np.arange(0,blksz+1,dtype=dtype)/(blksz))
            window = window[:-1] # remove last entry
            step = int(blksz/2)
        else:
            window = .54 - .46*np.cos(2*np.pi*np.arange(0,blksz,dtype=dtype)/(blksz-1))
            # symetric first/final entries
            window[0] = window[0]/2
            window[-1] = window[-1]/2
            step = int((blksz-1)/2)
    return window, step

def test_get_widow_step(window=None,step=None,blksz=51):
    import matplotlib.pyplot as plt
    if window is None or isinstance(window,str):
        window, step = get_window_step(blksz, window, .5)
    xx = np.zeros((300,))
    for i in range(0,xx.size,step):
        blksz = min(window.size, xx.size-i)
        xx[i:i+blksz] = xx[i:i+blksz] + window[:blksz]
    plt.plot(window,label='window')
    plt.plot(xx,label='summed')
    plt.show()


def fftfilter(X_TSd:np.ndarray, filterband, fs:float=100, axis=-2, freq_sigma=1, center:bool=True, verb=1):
    """run X through a filterbank computed using the fft transform

    Args:
        X_TSd (np.ndarray): input data with time in axis
        filterband ([type]): filter band specified as (low-cut,high-cut,type), or (low-cut,low-pass,high-pass,high-cut,type), where type is one-of: 'bandpass','bandstop','hilbert'
        fs (float): sample rate of X_TSd, for conversion to Hz. Defaults to 100
        axis (int, optional): 'time' axis of X_TSd. Defaults to -2.
        verb (int, optional): verbosity level. Defaults to 1.

    Returns:
        X_TSfd: band pass filtered blocks of X_TSd
    """    
    if verb > 0:  print("fftfilter: {}".format(filterband))
    assert axis==-2 or axis == X_TSd.ndim-2
    # ensure 3-d input
    if X_TSd.ndim<3:
        X_TSd = X_TSd.reshape( (-1,)*(3-X_TSd.ndim) + X_TSd.shape )

    masks_bf, bandtype, freqs = fftfilter_masks(X_TSd.shape[axis],filterband,fs,freq_sigma,dtype=X_TSd.dtype)
    # actual mask we use is *and* over the individual masks (so same logic as for sosfilter)
    mask_f = np.prod(masks_bf,0)
    X_TSfd = inner_fftfilterbank(X_TSd, mask_f[np.newaxis,:], bandtype[0], axis=axis, center=center)
    X_TSd = X_TSfd.squeeze(2)
    return X_TSd

def fftfilterbank(X_TSd:np.ndarray, filterbank, fs:float=100, axis=1, freq_sigma=1, center:bool=True, window=None, verb=1):
    """run X through a filterbank computed using the fft transform

    Args:
        X_TSd (np.ndarray): input data with time in axis
        filterbank ([type]): list of filterbank bands, in Hz specified as (low-cut,high-cut,type), or (low-cut,low-pass,high-pass,high-cut,type), where type is one-of: 'bandpass','bandstop','hilbert'
        fs (float): sample rate of X_TSd, for conversion to Hz. Defaults to 100
        axis (int, optional): 'time' axis of X_TSd. Defaults to -2.
        verb (int, optional): verbosity level. Defaults to 1.

    Returns:
        X_TSfd: band pass filtered blocks of X_TSd
    """    
    if verb > 0:  print("Filterbank: {}".format(filterbank))
    assert axis==1

    # ensure 3-d input
    if X_TSd.ndim<3:
        X_TSd = X_TSd.reshape( (-1,)*(3-X_TSd.ndim) + X_TSd.shape )

    # compute the band masks
    mask_bf, feature_b, freqs = fftfilter_masks(X_TSd.shape[axis],filterbank,fs,freq_sigma,dtype=X_TSd.dtype)

    X_TSfd = inner_fftfilterbank(X_TSd,mask_bf=mask_bf,feature_b=feature_b,axis=axis,window=window,center=center)
    return X_TSfd

def inner_fftfilterbank(X_TSd:np.ndarray, mask_bf, feature_b, axis=1, window=None, center:bool=True, verb=1):
    axis = axis if axis>0 else X_TSd.ndim+axis
    #import matplotlib.pyplot as plt; plt.plot(mask.T);plt.plot(np.sum(mask,0),'k-');plt.show(block=True)
    mask_bf = np.reshape(mask_bf, mask_bf.shape+(1,)*(X_TSd.ndim-axis-1))
    if window is not None:
        window = np.reshape(window, window.shape+(1,)*(X_TSd.ndim-axis-1))
        window = np.sqrt(window)

    #print("X.shape={}  mask.shape={}".format(X_TSd.shape, mask_bf.shape))
    if center:
        X_TSd = X_TSd - np.mean(X_TSd,axis=axis,keepdims=True)

    # apply filter bank to frequency ranges into virtual channels
    X_TSfd=np.zeros(X_TSd.shape[:axis+1]+(mask_bf.shape[0],)+X_TSd.shape[axis+1:],dtype=X_TSd.dtype)
    for ti in range(X_TSd.shape[0]):
        X = X_TSd[ti:ti+1,...]
        if window is not None:
            X = X * window
        Fx = np.fft.fft(X,axis=axis)
        for bi in range(mask_bf.shape[0]):
            iFxbi = np.fft.ifft(Fx*mask_bf[bi,...], axis=axis)
            if window is not None:
                iFxbi = iFxbi * window
            if feature_b[bi].lower() in ('abs','hilbert'):
                iFxbi = np.abs(iFxbi)
            else:
                iFxbi = np.real(iFxbi)
            # make index expr to insert in right place
            bidx = [slice(None)]*X_TSfd.ndim; bidx[0]=ti; bidx[axis+1]=bi
            X_TSfd[tuple(bidx)] = iFxbi
    return X_TSfd

def ola_fftfilter(X_TSd:np.ndarray, filterband, fs:float=100, axis=1,  blksz=None, window=None, fftlen:int=None, freq_sigma:float=1, overlap:float=.5, center:bool=True, verb=1):
    axis = axis if axis>0 else X_TSd.ndim+axis
    assert axis==1
    # ensure 3-d input
    X_TSd = X_TSd.reshape( (-1,)*(3-X_TSd.ndim) + X_TSd.shape )

    blksz = int(blksz) if blksz is not None else int(fs//2)
    if fftlen is None:  fftlen = int(blksz*2)
    # pre-compute the mask
    filter_bf, bandtype, freqs = fftfilter_masks(fftlen,filterband,fs,freq_sigma,dtype=X_TSd.dtype)
    # actual mask we use is *and* over the individual masks (so same logic as for sosfilter)
    filter_f = np.prod(filter_bf,0)
    # get the window and step
    window, step = get_window_step(blksz, window, overlap)
    X_TSfd = inner_ola_fftfilterbank(X_TSd, filters_bf=filter_f, feature_b=bandtype, window_t=window, step=step, axis=axis, center=center)
    X_TSd = X_TSfd.squeeze(2)
    return X_TSd


def ola_fftfilterbank(X_TSd:np.ndarray, filterbank, fs:float=100, axis=1,  blksz=None, window=None, fftlen:int=None, freq_sigma:float=1, overlap:float=.5, center:bool=True, verb=1):
    """overlap-add FFT filterbank / filter, suitable for on-line application

    Args:
        X_TSd (np.ndarray): [description]
        filterbank ([type]): list of filterbank bands, in Hz specified as (low-cut,high-cut,type), or (low-cut,low-pass,high-pass,high-cut,type), where type is one-of: 'bandpass','bandstop','hilbert'
        fs (float): sample rate of X_TSd, for conversion to Hz. Defaults to 100.
        axis (int, optional): 'time' axis of X_TSd. Defaults to -2.
        winsz ([type], optional): block size for over-lap-add blocks. Defaults to None.
        fftlen (int, optional): size of the inner FFT window -> used to increase spectral resolution for small window sizes.  Defaults to None=winsz
        window ([type], optional): temporal window to apply. Bartlett (triangle) used if not given. Defaults to None.
        overlap (float, optional): fractional overlap for the blocks, such that overlap-summed window has equal strength. Defaults to .5

    Returns:
        X_TSfd: band pass filtered blocks of X_TSd
    """    
    if verb > 0:  print("Filterbank: {}".format(filterbank))
    assert axis==1

    # ensure 3-d input
    if X_TSd.ndim<3:
        X_TSd = X_TSd.reshape( (-1,)*(3-X_TSd.ndim) + X_TSd.shape )

    # setup the window
    if blksz is None:
        blksz = fs // 2
    blksz = int(blksz)
    if fftlen is None:
        fftlen = int(blksz)
    fftlen=int(fftlen)
    window, step = get_window_step(blksz, window, overlap)

    mask_bf, bandtype, freqs = fftfilter_masks(fftlen, filterbank, freq_sigma=freq_sigma, fs=fs, dtype=X_TSd.dtype)
    X_Sfd = inner_ola_fftfilterbank(X_TSd, mask_bf, bandtype, window, step, axis=axis, center=center)
    return X_Sfd


def inner_ola_fftfilterbank(X_TSd,filters_bf, feature_b, window_t, step, axis=1, center:bool=True, prefix_band_dim:bool=False):
    """(multi-band) spectral filter, implemented via. overlap-and-add fft transforms

    Args:
        X_TSd (np.ndarray): data with dimesion for FFT 
        filters_bf (np.ndarray): set of spectral filters to apply over f-frequency bins
        feature_b (str, optional): the type of feature of the filtered data for each band. one-of: 'real', 'abs', None.  Use real for normal filter, 'abs' in combination with a 'hilbert' filter (postive freq only) for band-power extraction, or None for complex envelope. Defaults to 'real'.
        window_t (np.ndarray): window pre-applied to X in blocks in time-domain
        step (int): step-size for overlapping windows.  N.B. it's important that the shifted summed windows have the Constant-OverLap-Add (COLA) property
        axis (int,tuple): axis of X which represents 'time'
        prefix_band_dim (bool): flag if we pre or post-fix the band dimension

    Returns:
        X_TSbd: X after application of the (multi-band) filters, with different filters in the additional axis+1 dimension, b.
    """    
    axis = axis if axis>0 else X_TSd.ndim+axis
    #assert axis==1  # not implemented for general axis specification (yet)

    # ensure 3-d input
    if X_TSd.ndim<3:
        X_TSd = X_TSd.reshape( (-1,)*(3-X_TSd.ndim) + X_TSd.shape )

    if filters_bf.ndim==1 : filters_bf=filters_bf[np.newaxis,:]
    if isinstance(feature_b,str):
        feature_b = [feature_b]*filters_bf.shape[0]

    # shift window to the left to align with axis
    # N.B. we use a weighted overlap-add strategy to reduced block boundary effects
    #      hence we pre and post weight with the sqrt of the window
    window_t = np.sqrt(window_t)
    window_t = np.reshape(window_t.flat, (-1,)+(1,)*(X_TSd.ndim-axis-1))
    filters_bf = np.reshape(filters_bf, filters_bf.shape+(1,)*(X_TSd.ndim-axis-1))

    #test_get_widow_step(window_t.ravel(),step)

    # apply in overlap-add format
    if prefix_band_dim:
        shape = X_TSd.shape[:axis]+(filters_bf.shape[0],)+X_TSd.shape[axis:]
    else:
        shape = X_TSd.shape[:axis+1]+(filters_bf.shape[0],)+X_TSd.shape[axis+1:]
    X_TSfd=np.zeros(shape,dtype=X_TSd.dtype)
    for ti in range(X_TSd.shape[0]):
        #acc[...]=0
        Xti = X_TSd[ti:ti+1,...]
        if center:
            Xti = Xti - np.mean(Xti, axis, keepdims=True)
        idx = [slice(None)]*X_TSd.ndim
        for i in range(0,X_TSd.shape[axis],step): # overlap segments
            # get this segment
            blksamp = min(window_t.size, X_TSd.shape[axis]-i)
            blkslice = slice(i,i+blksamp)
            idx = tuple(slice(None) if d!=axis else blkslice for d in range(X_TSd.ndim))

            # extract, apply window and FFT
            Xi = Xti[idx]
            # apply (truncated) window (pre-window)
            Xi = Xi * window_t[:Xi.shape[axis],...]
            # compute block fft
            Fxi = np.fft.fft(Xi, n=filters_bf.shape[1], axis=axis) # zero-padded FFT to blks

            # per band, apply frequency filter and inverse fft and sum into output
            
            for bi in range(filters_bf.shape[0]):
                # apply spectral filter, truncate to right length, map back to time-domain
                iFxi = np.fft.ifft(Fxi*filters_bf[bi,:,...], axis=axis)
                iFxi = iFxi[idx[:axis]+(slice(blksamp),)+idx[axis+1:]]
                # apply (truncated) window (post-window)
                iFxi = iFxi * window_t[:iFxi.shape[axis],...]
                #iFxi = window_t[:iFxi.shape[axis],...]
                if feature_b[bi].lower() in ('abs','hilbert'):
                    iFxi = np.abs(iFxi)
                else:
                    iFxi = np.real(iFxi)
                if prefix_band_dim:
                    bidx = [slice(None)]*X_TSfd.ndim; bidx[0]=ti; bidx[axis]=bi; bidx[axis+1]=blkslice
                else:
                    bidx = [slice(None)]*X_TSfd.ndim; bidx[0]=ti; bidx[axis]=blkslice; bidx[axis+1]=bi
                X_TSfd[tuple(bidx)] = X_TSfd[tuple(bidx)] + iFxi
        #         # N.B. sum the complex features
        #         outsamp = min(iFxi.shape[axis], acc.shape[0]-i)
        #         idx_out = slice(i,i+outsamp)
        #         acc[idx_out,bi,...] = acc[idx_out,bi,...] + iFxi[:,:outsamp,...]
        # # extract the wanted feature of the filtered data
        # for bi,feat in enumerate(feature):
        #     if feat.lower() in ('abs','hilbert'):
        #         X_TSfd[ti,:,bi,...] = np.abs(acc[:,bi,...])
        #     else:
        #         X_TSfd[ti,:,bi,...] = np.real(acc[:,bi,...])

    return X_TSfd


def ola_welch(X_TSd, axis, window_t, step, feature='real', center:bool=True):
    """(multi-band) spectral filter, implemented via. overlap-and-add fft transforms

    Args:
        X_TSd (np.ndarray): data with dimesion for FFT 
        axis (int,tuple): axis of X which represents 'time'
        window_t (np.ndarray): window pre-applied to X in blocks in time-domain
        step (int): step-size for overlapping windows.  N.B. it's important that the shifted summed windows have the Constant-OverLap-Add (COLA) property
        feature (str, optional): the type of feature of the filtered data to return. one-of: 'real', 'abs', None.  Use real for normal filter, 'abs' in combination with a 'hilbert' filter (postive freq only) for band-power extraction, or None for complex envelope. Defaults to 'real'.

    Returns:
        X_TBfd: X after application of the (multi-band) filters, with different filters in the additional axis+1 dimension, b.
    """
    axis = axis if axis>0 else X_TSd.ndim+axis
    assert axis==X_TSd.ndim-2  # not implemented for general axis specification (yet)
    # ensure 3-d input
    X_TSd = X.reshape( (-1,)*(3-X_TSd.ndim) + X_TSd.shape )

    # shift window to the left to align with axis
    window_t = np.reshape(window_t.flat,(-1,)+(1,)*(X_TSd.ndim-axis-1))

    # apply in overlap-add format
    start_idx = list(range(0,X_TSd.shape[axis],step)) # window start locations
    X_TBfd=np.zeros(X_TSd.shape[:axis]+(len(start_idx),window_t.size//2)+X_TSd.shape[axis+1:],dtype=X_TSd.dtype)
    for bi,i in enumerate(start_idx): # overlap segments
        # get this segment
        blksamp = window_t.size if i+window_t.size < X_TSd.shape[axis] else X_TSd.shape[axis]-i
        idx = slice(i,i+blksamp)

        # extract, apply window and FFT
        Xi = X_TSd[...,idx,:]
        if center:
            Xi = Xi - np.mean(Xi, axis, keepdims=True)
        # apply (truncated) window
        Xi = Xi * window_t[:Xi.shape[axis],...]
        # compute block fft
        Fxi = np.fft.fft(Xi, axis=axis) # zero-padded FFT to blks

        # per band, apply frequency filter and inverse fft and sum into output 
        if feature=='complex':
            pass
        elif feature=='real':
            Fxi = Fxi.real
        elif feature=='imag':
            Fxi = Fxi.imag
        else:
            Fxi = np.abs(Fxi)
        X_TBfd[...,bi,:,:] = Fxi
    return X_TBfd



def plot_grand_average_spectrum(X_TSd, fs:float, axis:int=-2, ch_names=None, log=False, show:bool=None):
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    # ensure 3-d input
    X_TSd = X_TSd.reshape( (-1,)*(3-X_TSd.ndim) + X_TSd.shape )

    freqs, FX = welch(X_TSd, axis=axis, fs=fs, nperseg=fs//2, return_onesided=True, detrend=False) # FX = (nFreq, nch)
    #print('FX={}'.format(FX.shape))
    #plt.figure(18);plt.clf()
    muFX = np.median(FX,axis=0,keepdims=True) if FX.ndim>2 else FX
    if log:
        muFX = 10*np.log10(muFX)
        unit='db (10*log10(uV^2))'
        ylim = (2*np.median(np.min(muFX,axis=tuple(range(muFX.ndim-1))),axis=-1),
                2*np.median(np.max(muFX,axis=tuple(range(muFX.ndim-1))),axis=-1))
    else:
        unit='uV^2'
        ylim = (0,2*np.median(np.max(muFX,axis=tuple(range(muFX.ndim-1))),axis=-1))

    plot_erp(muFX, ch_names=ch_names, evtlabs=None, times=freqs, ylim=ylim, show=show)       
    plt.suptitle("Grand average spectrum ({})".format(unit))
    if show is not None: plt.show(block=show)


def extract_envelope(X,fs,
                     filterband=None,whiten=True,filterbank=None,log=True,env_filterband=(10,-1),
                     verb=False, plot=False):
    """extract the envelope from the input data

    Args:
        X ([type]): [description]
        fs ([type]): [description]
        filterband ([type], optional): pre-filter stop band. Defaults to None.
        whiten (bool, optional): flag if we spatially whiten before envelope extraction. Defaults to True.
        filterbank ([type], optional): set of filters to apply to extract the envelope for each filter output. Defaults to None.
        log (bool, optional): flag if we return raw power or log-power. Defaults to True.
        env_filterband (tuple, optional): post-filter on the extracted envelopes. Defaults to (10,-1).
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

    if not filterband is None:
        if verb > 0:  print("preFilter: {}Hz".format(filterband))
        X, _, _ = butter_sosfilt(X,filterband,fs)
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
        X = ola_fftfilterbank(X,filterbank,fs)
        # make filterbank entries into virtual channels
        X = np.reshape(X,X.shape[:-2]+(np.prod(X.shape[-2:],)))
        
    X = np.abs(X) # rectify
    if plot:plt.figure(104);plt.plot(X[:int(fs*10),:]);plt.title("+abs")

    if log:
        if verb > 0: print("log amplitude")
        X = np.log(np.maximum(X,1e-6))
        if plot:plt.figure(105);plt.clf();plt.plot(X[:int(fs*10),:]);plt.title("+log")

    if env_filterband is not None:
        if verb>0: print("Envelop band={}".format(env_filterband))
        X, _, _ = butter_sosfilt(X,env_filterband,fs) # low-pass = envelope extraction
        if plot:plt.figure(104);plt.clf();plt.plot(X[:int(fs*10),:]);plt.title("+env")
    return X


def block_cov(X_TSd, Y_TSy, coords, axis=-2, blksz=None, blksz_ms=500, fs=100, resample_y='max', verb=1):
    """overlap-add FFT filterbank / filter, suitable for on-line application

    Args:
        X_TSd (np.ndarray): [description]
        Y_TSy (np.ndarray)
        fs (float): sample rate of X_TSd, for conversion to Hz. Defaults to 100.
        axis (int, optional): 'time' axis of X_TSd. Defaults to -2.
        winsz ([type], optional): block size for over-lap-add blocks. Defaults to None.
        winsz_ms (float): block size in milliseconds
        window ([type], optional): temporal window to apply. Bartlett (triangle) used if not given. Defaults to None.
        overlap (float, optional): fractional overlap for the blocks, such that overlap-summed window has equal strength. Defaults to .5

    Returns:
        (X_TSdd, Y_TSy, coords): band pass filtered blocks of X_TSd
    """
    if verb > 0:  print("block_cov")
    assert axis==-2

    # setup the window
    if blksz is None:
        blksz = blksz_ms /1000 * fs if blksz_ms is not None else fs / 2
    blksz=int(blksz)
    print('blksz={}'.format(blksz))

    window, step = get_window_step(blksz, window='bartlet', overlap=.5)
    # get to right shape
    window = window[:,np.newaxis]
    
    # apply in overlap-add format
    blkidxs = np.arange(0,X_TSd.shape[axis],step)
    Y_Tby =np.zeros(Y_TSy.shape[:axis]+(len(blkidxs),)+Y_TSy.shape[-1:]) if Y_TSy is not None else None# downsampled Y
    X_Tbdd=np.zeros(X_TSd.shape[:axis]+(len(blkidxs),X_TSd.shape[-1],X_TSd.shape[-1]),dtype=X_TSd.dtype) # block cov downsampled X
    for bi,i in enumerate(blkidxs): # overlap segments
        # get this segment
        blksamp = blksz if i+blksz < X_TSd.shape[axis] else X_TSd.shape[axis]-i
        idx = slice(i,i+blksamp)

        # extract, apply window and cov
        if blksamp==len(window):
            Xi = X_TSd[...,idx,:] * window
        else: # not a full window of data available -- reverse pad?
            Xi = X_TSd[...,idx,:] * window[:blksamp,:]
        XXi = np.einsum("TSd,TSe->Ted",Xi,Xi)
        X_Tbdd[...,bi,:,:] = X_Tbdd[...,bi,:,:] + XXi

        # downsample Y
        if Y_TSy is not None:
            Y_Tby[...,bi,:] = pool_axis(Y_TSy[...,idx,:],resample_y,axis)

    # udpate meta-info
    if coords is not None:
        # downsample
        coords[axis]['coords'] = coords[axis]['coords'][blkidxs]
        if 'fs' in coords[axis]:
            coords[axis]['fs'] = fs / step
            print('outfs={}'.format(coords[axis]['fs']))
        # new dim
        coords = coords[:X_TSd.ndim] + [coords[X_TSd.ndim-1]] + coords[X_TSd.ndim:]

    return X_Tbdd, Y_Tby, coords

def testCase_pipeline(X=None, Y=None, fs=100):
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    np.random.seed(1)
    if X is None:
        X = np.random.standard_normal((2,fs*3,2)) # flat spectrum
        X[...,0] = X[...,0] + X[...,1] # induce some cross correlation
        X = X[:,:-1,:]+X[:,1:,:] # weak low-pass

        # target 
        Y = np.random.standard_normal(X.shape[:-1]+(1,))>1
        X[...,0:1] = X[...,0:1] + Y*3  # signal injector

    print("X={}".format(X.shape))

    plt.figure()
    plot_trial(X,Y,fs)

    pipeline=['SpatialWhitener',
#                ('butter_sosfilt',dict(filterband=(3,35,'bandpass'))),
                ('FFTfilter',dict(filterbank=(1,10,'bandpass'))),
                "Log",
                'SpatialWhitener',
                ["ButterFilterAndResampler", {"filterband":[20,-1], "fs_out":50 }]
    ]
    X,Y,coords = preprocess(X,Y,None,fs=fs,pipeline=pipeline)
    print("X={}".format(X.shape))
    print("Y={}".format(Y.shape))

    plt.figure()
    plot_trial(X,Y,fs)
    plt.suptitle(pipeline)
    plt.show()


    pipeline=('spatially_whiten',
                ('butter_sosfilt',dict(filterband=(3,35,'bandpass'))),
                'spatially_whiten',
                ('fftfilter',dict(filterband=(0,100,'bandpass'))),
                ('resample',dict(fs=fs,fs_out=10)),
             )
    X,Y,coords = preprocess_old(X,Y,None,fs=fs,pipeline=pipeline)
    print("X={}".format(X.shape))
    print("Y={}".format(Y.shape))

    plt.figure()
    plot_trial(X,Y,fs)
    plt.suptitle(pipeline)
    plt.show()






def testCase_spectralwhiten(X=None, Y=None, fs=100):
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    if X is None:
        np.random.seed(1)
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


def testCase_block_cov(X=None, Y=None, fs=100):
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    if X is None:
        np.random.seed(1)
        X = np.random.standard_normal((2,fs*3,2)) # flat spectrum
        X[...,0] = X[...,0] + X[...,1] # induce some cross correlation
        X = X[:,:-1,:]+X[:,1:,:] # weak low-pass

        # target 
        Y = np.random.standard_normal(X.shape[:-1]+(1,))>3
        X[...,0:1] = X[...,0:1] + Y*3  # signal injector

    plot_trial(X,Y,fs)

    XX, YY, _ = block_cov(X,Y,None,blksz=fs)
    
    # compare raw vs summed filterbank
    plt.figure(1)
    fig, ax = plt.subplots(nrows=XX.shape[0],ncols=XX.shape[1])
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            ax[i][j].imshow(XX[i,j,:,:],aspect='auto')
    plt.suptitle('Whitened')

    plot_trial(XX.reshape(XX.shape[:-2]+(-1,)),YY)

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


def testCase_fftfilter(X_TSd=None, Y_TSy=None, fs=100, bands=((0,10,'bandpass'))):
    if bands is None:
        bands=(1,10, 'bandpass') # simple HP
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    if X_TSd is None:
        np.random.seed(0)
        X_TSd = np.random.standard_normal((2,fs*6,2)) # flat spectrum
        X_TSd = X_TSd[:,:-1,:]+X_TSd[:,1:,:] # weak low-pass
        X_TSd = X_TSd + 1e3 # DC shift
    #X_TSd = np.cumsum(X_TSd,-2) # 1/f spectrum
    print("X={}".format(X_TSd.shape))
    #plt.figure()
    #plot_grand_average_spectrum(X_TSd, fs)
    #plt.show()

    oX_TSd = X_TSd.copy()

    X_TSd0 = oX_TSd

    X_TSd0, _, _ = butter_sosfilt(oX_TSd.copy(),bands,fs=fs)

    X_TSd  = ola_fftfilter(oX_TSd.copy(),bands,fs=fs,center=True)

    plt.subplot(311);plt.plot(oX_TSd[0,:,:]); plt.title('raw')
    plt.subplot(323); plt.plot(X_TSd0[0,:,:]); plt.title('butter_sosfilt')
    plt.subplot(324); plt.plot(X_TSd[0,:,:]); plt.title('fft_ola_filt')
    plt.subplot(325); plt.plot(oX_TSd[0,:,:]-X_TSd0[0,:,:]);  plt.title('raw - butter')
    plt.subplot(326); plt.plot(oX_TSd[0,:,:]-X_TSd[0,:,:]);  plt.title('raw - fft_ola')

    plt.show()


def testCase_filterbank(X_TSd=None, Y_TSy=None, fs=100):
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    if X_TSd is None:
        np.random.seed(0)
        X_TSd = np.random.standard_normal((2,fs*3,2)) # flat spectrum
        X_TSd = X_TSd[:,:-1,:]+X_TSd[:,1:,:] # weak low-pass

        X_TSd = X_TSd + 1e3
        #X_TSd = np.cumsum(X_TSd,-2) # 1/f spectrum
        print("X={}".format(X_TSd.shape))
        #plt.figure()
        #plot_grand_average_spectrum(X_TSd, fs)
        #plt.show()


    X_TSd, _, _ = butter_sosfilt(X_TSd,(0,6),fs=fs)

    bands = ((3,6,12,15,'bandpass'),(12,15,20,25,'bandpass'),(20,25,40,45,'bandpass'),(40,45,-1,-1,'bandpass'))
    bands = ((40,45,125,125,'hilbert'))
    #bands = np.linspace(0,50,5)

    masks, types, freqs = fftfilter_masks(100,bands,100)
    plt.figure()
    plt.plot(freqs, masks.T)
    plt.plot(freqs, np.sum(masks,0),'k')
    plt.title('frequency domain filters')
    plt.show(block=False)


    #X_TSd[0,:,0]=1 # use fixed input to check window overlap

    Xff  = fftfilterbank(X_TSd,bands,fs=fs)

    Xf = ola_fftfilterbank(X_TSd,bands,fs=fs)#,blksz=fs//3,fftlen=fs//3)
    #print("Xf={}".format(Xf.shape))

    plt.figure()
    plt.subplot(311);plt.plot(Xff[0,:,:,0]); plt.title('fft_filterbank')
    plt.subplot(312); plt.plot(Xf[0,:,:,0]); plt.title('fft_ola_filterbank')
    plt.subplot(313); plt.plot(Xff[0,:,:,0]-Xf[0,:,:,0]); plt.title('fft-fft_ola')
    plt.show(block=False)

    #Xf,Yf,coordsf = preprocess(X, None, None, filterbank=bands, fs=100)
    #Xf, _, _ = butter_filterbank(X,bands,fs,order=3,ftype='butter') # tr,samp,band,ch
    #Xf = fft_filterbank(X,bands,fs=100)
    print("Xf={}".format(Xf.shape))
    # bands -> virtual channels
    # make filterbank entries into virtual channels
    plt.figure()
    Xf_s = np.sum(Xf,-2,keepdims=False)
    plot_grand_average_spectrum(np.concatenate((X_TSd[:,np.newaxis,...],Xf_s[:,np.newaxis,...],np.moveaxis(Xf,(0,1,2,3),(0,2,1,3))),1), fs)
    plt.legend(('X','Xf_s','X_bands'))
    plt.show()
    
    # compare raw vs summed filterbank
    plt.figure()
    plot_erp(np.concatenate((X_TSd[0:1,...],Xf_s[0:1,...],np.moveaxis(Xf[0,...],(0,1,2),(1,0,2))),0),
             evtlabs=['X','Xf_s']+['Xf_{}'.format(b) for b in bands])
    plt.suptitle('fft_ola: X, Xf_s')
    plt.show()

    # test hilbert transform
    bands = ((6,12,'bandpass'),(6,12,'hilbert'),(12,20,'bandpass'),(12,20,'hilbert'),(20,40,'bandpass'),(20,40,'hilbert'),(40,-1,'bandpass'))
    Xf = ola_fftfilterbank(X_TSd[:,:,0:1],bands,fs=fs)
    #Xf = ola_fftfilterbank(X_TSd[:,:,0:1],bands,fs=fs,blksz=fs//3,fftlen=2*fs//3)
    #print("Xf={}".format(Xf.shape))

    plt.figure()
    plot_trial(Xf.reshape(Xf.shape[:-2]+(-1,)),None,fs,outputs=bands)
    plt.suptitle('fft_ola \n {}'.format(bands))
    plt.show()


def test_fir(X_TSd=None, Y_TSy=None, fs=100):
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    if X_TSd is None:
        X = np.random.standard_normal((2,fs*3,2)) # flat spectrum
        X = X[:,:-1,:]+X[:,1:,:] # weak low-pass
        X = X + 1e3
        #X = np.cumsum(X,-2) # 1/f spectrum
    print("X={}".format(X.shape))
    #plt.figure()
    #plot_grand_average_spectrum(X, fs)
    #plt.show()


    bands = ((1,10,'bandpass'),(10,20,'bandpass'),(20,40,'bandpass'))
    ###Xf,Yf,coordsf = preprocess(X, None, None, fir=dict(ntap=3,dilation=1), fs=100)
    print("Xf={}".format(Xf.shape))
    # bands -> virtual channels
    # make filterbank entries into virtual channels
    plt.figure()
    plot_grand_average_spectrum(np.concatenate((X[:,np.newaxis,...],Xf[:,np.newaxis,...],np.moveaxis(Xf,(0,1,2,3),(0,2,1,3))),1), fs)
    plt.legend(('X','Xf_s','X_bands'))
    plt.show()

if __name__=="__main__":

    testCase_pipeline()

    # test_get_widow_step('hamming',blksz=50)
    # test_get_widow_step('hamming',blksz=51)

    X, Y, fs = (None, None, 100)
    if True:
        # load
        from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
        X, Y, coords = load_mindaffectBCI("askloadsavefile", filterband=((45,55),(95,105)), order=6, ftype='butter', fs_out=250)
        # output is: X=eeg, Y=stimulus, coords=meta-info about dimensions of X and Y
        fs = coords[1]['fs']
        print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,fs))
        print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:1]]+['output'],Y.shape))

    #testCase_pipeline()

    testCase_fftfilter(X,Y,fs)

    testCase_filterbank(X,Y,fs)

    testCase_block_cov(X,Y,fs)

    testCase_temporallydecorrelate(X)
    #testCase_spectralwhiten()
