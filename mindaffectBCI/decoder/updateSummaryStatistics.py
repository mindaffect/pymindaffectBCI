# Copyright (c) 2019 MindAffect B.V. 
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

import numpy as np
from mindaffectBCI.decoder.utils import window_axis, idOutliers, zero_outliers
#@function
def updateSummaryStatistics(X, Y, stimTimes=None, 
                            Cxx=None, Cxy=None, Cyy=None, 
                            badEpThresh=4, badWinThresh=3,  halflife_samp=1, cxxp=True, cyyp=True, tau=None,
                            offset=0, center=True, unitnorm=True, zeropadded:bool=True, perY=True):
    '''
    Compute updated summary statistics (Cxx_dd, Cxy_yetd, Cyy_yetet) for new data in X with event-info Y

    Args:
      X_TSd (nTrl, nEp, tau, d): raw response for the current stimulus event
               d=#electrodes tau=#response-samples  nEpoch=#stimulus events to process
         OR
          (nTrl, nSamp, d)
      Y_TSye = (nTrl, nSamp, nY, nE): Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      stimTimes_samp = (nTrl, nEp) sample times for start each epoch.  Used to detect
               overlapping responses
      tau = int - the length of the impulse response in samples (X.shape[1] or Cxy.shape[1] if given)
      offset:int = relative shift of Y w.r.t. X
      Cxx_dd (d,d) : current data covariance
      Cxy_yetd (nY, nE, tau, d) : current per output ERPs
      Cyy_yetet (nY, nE, tau, nE, tau) : current response covariance for each output
      badEpThresh (float): threshold for removing bad-data before fitting the summary statistics
      center (bool): do we center the X data before computing the summary statistics? (True)
      halflife_samp (float): forgetting factor for the updates
               Note: alpha = exp(log(.5)./(half-life)), half-life = log(.5)/log(alpha)   
    Returns:
      Cxx_dd (d,d) : updated data covariance
      Cxy_yetd (nY, nE, tau, d) : updated per output ERPs
      Cyy_yetet (nY, nE, tau, nE, tau): updated response covariance for each output

    Examples:
      # Supervised CCA
      Y = (nEpoch/nSamp, nY, nE) [nE x nY x nEpoch] indicator for each event-type and output of it's type in each epoch
      X = (nEpoch/nSamp, tau, d) [d x tau x nEpoch] sliced pre-processed raw data into per-stimulus event responses
      stimTimes = [nEpoch] sample numbers of the stimulus events
                # OR None if X, Y are samples
      [Cxx, Cxy, Cyy] = updateSummaryStatistics(X, Y, 3*np.arange(X.shape[0]));
      [J, w, r, a, I]=multipleCCA(Cxx, Cxy, Cyy)
    '''
    if X is None:
        return Cxx, Cxy, Cyy
    if tau is None:
        if Cxy is not None:
            tau = Cxy.shape[-2]
        elif X.ndim > 3: # assume X is already sliced so we can infer the tau..
            print("Warning: guessing tau from shape of X")
            tau = X.shape[-2]
        else:
            raise ValueError("tau not set and Cxy is None!")

    # add missing trial dim to X if not there
    if X.ndim==2: # put trial dim back in
        Y = Y.reshape( (1,)*(3-X.ndim) + Y.shape) 
        X = X.reshape( (1,)*(3-X.ndim) + X.shape)

    tau = int(tau) # ensure tau is integer
    if tau > X.shape[1]:
        print("Warning!!!!! tau is bigger than number samples in X... truncated")
        tau = X.shape[1]
    wght = 1

    X, Y = zero_outliers(X, Y, badEpThresh, badWinThresh=badWinThresh, winsz=tau, verb=0)
    
    # update the cross covariance XY
    Cxy = updateCxy(Cxy, X, Y, stimTimes, tau, wght, offset=offset, center=center, unitnorm=unitnorm)
    # Update Cxx
    if cxxp:
        if cxxp is True:
            Cxx = updateCxx(Cxx, X, stimTimes, tau, wght, offset=offset, center=center, unitnorm=unitnorm)
        elif cxxp == 'noise': # estimate using only the non-stimulus data
            Cxx = updateCxx_n(Cxx, X, Y, stimTimes, tau, wght, offset=offset, center=center, unitnorm=unitnorm)
        elif cxxp == 'signal': # estimate using only the non-stimulus data
            Cxx = updateCxx_s(Cxx, X, Y, stimTimes, tau, wght, offset=offset, center=center, unitnorm=unitnorm)

    # ensure Cyy has the right size if not entry-per-model
    if (cyyp):
        # TODO [] : support overlapping info between update calls
        Cyy = updateCyy(Cyy, Y, stimTimes, tau, wght, offset=offset, unitnorm=unitnorm, perY=perY, zeropadded=zeropadded)
        
    return Cxx, Cxy, Cyy


def match_labels(Y_TSye, label_matcher=None):
    match_ax = (-1,-2) if Y_TSye.ndim==4 else -1 # axis to match over
    if label_matcher is None or label_matcher == 'non_zero':
        sig_TS = np.any(Y_TSye!=0,axis=match_ax) # samples at which stime-event happens
    elif callable(label_matcher):
        sig_TS = label_matcher(Y_TSye,axis=match_ax)
    elif hasattr(label_matcher,'__iter__'): # list matching values
        sig_TS = False
        for v in label_matcher:
            sig_TS = np.logical_or(sig_TS, np.any(Y_TSye==v,axis=match_ax))
    return sig_TS


def get_signal_indicator(Y_TSye, tau:int=10, offset:int=0, label_matcher=None):
    """identify the samples in the dataset which should contain a stimulus response

    Args:
        Y_TSye ([type]): [description]
    Returns:
        sig_TS : bool indicator of which samples contain signal
    """
    assert offset==0, "Non-zero offset not supported yet"

    # get time points which contain a signal (of the interesting type)
    sig_TS = match_labels(Y_TSye, label_matcher)

    # Any sample up-to tau samples after stimulus event is also assumed to contain signal
    sig_TS = np.concatenate((np.zeros((sig_TS.shape[0],tau-1),dtype=sig_TS.dtype),sig_TS),axis=-1) # pad so any stim tau-before TS is true
    sig_TSt = window_axis(sig_TS, winsz=tau, axis=-1, step=1)
    sig_TS = np.any(sig_TSt,axis=-1) # any stim in tau after this time point
    return sig_TS

def updateCxx_n(Cxx, X_TSd, Y_TSye=None, stimTimes=None, tau:int=None, wght:float=1, offset:int=0, center:bool=False, unitnorm:bool=True):
    '''
    Args:
        Cxx_dd (ndarray (d,d)): current data covariance
        X_TSd (nTrl, nSamp, d): raw response at sample rate
        Y_TSye = (nTrl, nSamp, nY, nE): Indicator for which events occured for which outputs
                nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
        stimTimes_samp (ndarray (nTrl, nEp)): sample times for start each epoch.  Used to detect
        wght (float): weight to accumulate this Cxx with the previous data, s.t. Cxx = Cxx_old*wght + Cxx_new.  Defaults to 1.
        center (bool): flag if center the data before computing covariance? Defaults to True.
    Returns:
        Cxx_dd (ndarray (d,d)): current noise data covariance
    '''
    # use Y non-zero to estimate the data which does not contain stimulus responses
    sig_TS = get_signal_indicator(Y_TSye,tau,offset)
    # only use the *non*-signal data to compute the spatial whitener
    noise_TS = np.logical_not(sig_TS)
    X = X_TSd[noise_TS,...] if np.sum(noise_TS) > noise_TS.size * .05 else X_TSd
    Cxx = updateCxx(Cxx,X,stimTimes,tau,wght,offset,center,unitnorm)
    return Cxx

def updateCxx_s(Cxx, X_TSd, Y_TSye=None, stimTimes=None, tau:int=None, wght:float=1, offset:int=0, center:bool=False, unitnorm:bool=True):
    '''
    Args:
        Cxx_dd (ndarray (d,d)): current data covariance
        X_TSd (nTrl, nSamp, d): raw response at sample rate
        Y_TSye = (nTrl, nSamp, nY, nE): Indicator for which events occured for which outputs
                nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
        stimTimes_samp (ndarray (nTrl, nEp)): sample times for start each epoch.  Used to detect
        wght (float): weight to accumulate this Cxx with the previous data, s.t. Cxx = Cxx_old*wght + Cxx_new.  Defaults to 1.
        center (bool): flag if center the data before computing covariance? Defaults to True.
    Returns:
        Cxx_dd (ndarray (d,d)): current noise data covariance
    '''
    # use Y non-zero to estimate the data which does not contain stimulus responses
    sig_TS = get_signal_indicator(Y_TSye,tau,offset)
    # only use the *non*-signal data to compute the spatial whitener
    X = X_TSd[sig_TS,...] if np.sum(sig_TS) > sig_TS.size * .1 else X_TSd
    Cxx = updateCxx(Cxx,X,stimTimes,tau,wght,offset,center,unitnorm)
    return Cxx


#@function
def updateCxx(Cxx, X, stimTimes=None, tau:int=None, wght:float=1, offset:int=0, center:bool=False, unitnorm:bool=True):
    '''
    Args:
        Cxx_dd (ndarray (d,d)): current data covariance
        X_TSd (nTrl, nSamp, d): raw response at sample rate
        Y_TSye = (nTrl, nSamp, nY, nE): Indicator for which events occured for which outputs
                nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
        stimTimes_samp (ndarray (nTrl, nEp)): sample times for start each epoch.  Used to detect
        wght (float): weight to accumulate this Cxx with the previous data, s.t. Cxx = Cxx_old*wght + Cxx_new.  Defaults to 1.
        center (bool): flag if center the data before computing covariance? Defaults to True.
    Returns:
        Cxx_dd (ndarray (d,d)): current data covariance
    '''
    # ensure 3d
    X = X.reshape( (1,)*(3-X.ndim) + X.shape)

    # use X as is, computing purely spatial cov
    # N.B. reshape is expensive with window_axis
    XX = X.reshape((-1,X.shape[-1])).T @ X.reshape((-1,X.shape[-1]))

    if center:
        sX = np.sum(X.reshape((-1,X.shape[-1])),axis=0) # feature mean (d,)
        XX = XX - np.einsum("i,j->ij",sX,sX) *X.shape[-1]/X.size
        
    if unitnorm:
        N = X.shape[0]*X.shape[1]
        XX = XX / N

    Cxx = wght*Cxx + XX if Cxx is not None else XX
    return Cxx

#@function
def updateCxy(Cxy, X, Y, stimTimes=None, tau=None, wght=1, offset=0, center=False, verb=0, unitnorm=True):
    '''
    Args:
        X_TSd (nTrl, nSamp, d): raw response at sample rate
        Y_TSye (nTrl, nSamp, nY, nE): event indicator at sample rate
        stimTimes_samp = (nTrl, nEp) sample times for start each epoch.  Used to detect
                overlapping responses
        Cxy_yetd (nY, nE, tau, d): current per output ERPs

    Returns:
        Cxy_yetd (nY, nE, tau, d): current per output ERPs
    '''
    if tau is None: # estimate the tau
        tau = Cxy.shape[-2]
    if X.ndim==2: # put trial dim back in
        Y = Y.reshape( (1,)*(3-X.ndim) + Y.shape)
        X = X.reshape( (1,)*(3-X.ndim) + X.shape)
    if Y.ndim == 3: # add an output dim
        Y = Y.reshape(Y.shape+(1,)*(4-Y.ndim)) 
    if X.ndim >3 : # support multiple feature dims
        X = X.reshape(X.shape[:2]+(-1,))
    if tau > X.shape[1]:
        tau = X.shape[1]
    #if offset<-tau+1:
    #    raise NotImplementedError("Cant offset backwards by more than window size")
    #if offset>0:
    #    raise NotImplementedError("Cant offset by positive amounts")
    if verb > 1: print("tau={}".format(tau))
    if stimTimes is None:
        # X, Y are at sample rate, slice X every sample
        #Xe = window_axis(X, winsz=tau, axis=-2) # (nTrl, nSamp, tau, d)
        inner_len = max(1,X.shape[1]-tau+1)
        # shrink Y w.r.t. the window and shift to align with the offset
        if offset ==0 : 
            Ye = Y[:,:inner_len,...]
        elif offset <0:# shift Y forwards
            Ye = Y[:, -offset:inner_len-offset, ...] # shift fowards and shrink
            if offset < -tau: # zero pad to size
                pad = np.zeros(Y.shape[:1]+(inner_len-Ye.shape[1],)+Y.shape[2:], dtype=Y.dtype)
                Ye = np.append(Ye,pad,1)        
        elif offset>0: # shift and pad
            Ye = Y[:, :inner_len-offset, ...] # shrink
            pad = np.zeros(Y.shape[:1]+(inner_len-Ye.shape[1],)+Y.shape[2:], dtype=Y.dtype)
            Ye = np.append(pad,Ye,1) # pad to shift forwards

    else:
        #Xe = X
        inner_len = max(1,X.shape[1]-tau+1)
        Ye = Y[:,:inner_len,...]

    if verb > 1: 
        print("X={}\nYe={}".format(X.shape, Ye.shape))
        print(np.max(Ye))

    # LOOPY version as einsum doens't manage the memory well...
    XY = np.zeros( (Ye.shape[-2],Ye.shape[-1],tau,X.shape[-1]), dtype=X.dtype)
    for taui in range(tau):
        # extract shifted X
        if (tau-taui-1) > 0:
            Xet = X[:,taui:-(tau-taui-1),...] 
        else:
            Xet = X[:,taui:,:,...] 
        Yet = Y[:,:Xet.shape[1],...]
        #print("{}/{} X={}  Xet={}".format(taui,tau,X.shape,Xet.shape))
        XY[:,:,taui,:] = np.einsum("TSye, TSd->yed", Yet, Xet, casting='unsafe', dtype=X.dtype)


    #XY = np.einsum("TSye, TStd->yetd", Ye.reshape((-1,)+Ye.shape[-2:]), Xe.reshape((-1,)+Xe.shape[-2:]))

    if center:
        if verb > 1: print("center")
        # N.B. (X-mu*1.T)@Y.T = X@Y.T - (mu*1.T)@Y.T = X@Y.T - mu*1.T@Y.T = X@Y.T - mu*\sum(Y)  
        muX = np.mean(X.reshape((-1,X.shape[-1])),axis=0) # mean X (d,)
        muY = np.sum(Ye,(0,1)) # summed Y (nY, nE)
        muXY= np.einsum("ye,d->yed",muY,muX) #(nY,nE,d)
        XY = XY - muXY[...,np.newaxis,:] #(nY,nE,tau,d)

    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        XY = XY / (Y.shape[0]*Y.shape[1]) # / nTrl*nEp
    
    Cxy = wght*Cxy + XY if Cxy is not None else XY
    return Cxy

# @function
def updateCyy(Cyy, Y, stimTime=None, tau=None, wght=1, offset:float=0, zeropadded=True, unitnorm=True, perY=True):
    '''
    Compute the Cyy tensors given new data
    Args:
      Cyy_yetet : old Cyy info
      Y_TSye (nTrl, nSamp, nY, nE): the new stim info to be added
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      tau (int) : number of samples in the stimulus response
      zeropadded (bool) : flag variable length Y are padded with 0s
      wght (float) : weighting for the new vs. old data
      unitnorm(bool) : flag if we normalize the Cyy with number epochs
    Returns:
      Cyy_yetet (nY, tau, nE, tau, nE):
    '''
    if perY:
        MM = compCyy_diag_perY(Y,tau,unitnorm=unitnorm) # (tau, nY, nE, nE)
    else:
        MM = compCyy_diag(Y,tau,unitnorm=unitnorm) # (tau,nY,nE,nY,nE)
    MM = Cyy_diag2full(MM) # (nY,nE,tau,nE,tau)
    Cyy = wght*Cyy + MM if Cyy is not None else MM
    return Cyy

def Cyy_diag2full(Cyy_tyee):
    """convert diag compressed Cyy to full version

    Args:
        Cyy ([type]): (tau,nY,nE,nY,nE) or (tau,nY,nE,nE)

    Returns:
        Cyy_yetyet (nY,nE,tau,nE,tau): block up-scaled Cyy
    """    
    # BODGE: tau-diag Cyy entries to the 'correct' shape
    # (tau,nY,nE,nE) -> (nY,nE,tau,nE,tau)
    Cyy4_tyee = Cyy_tyee if Cyy_tyee.ndim>3 else Cyy_tyee[:, np.newaxis, ...] # (tau,nE,nE) -> (tau,1,nE,nE)
    Cyy_yetet = np.zeros((Cyy4_tyee.shape[-3],Cyy4_tyee.shape[-2],Cyy4_tyee.shape[-4],Cyy4_tyee.shape[-2],Cyy4_tyee.shape[-4]),dtype=Cyy_tyee.dtype) # (nY,nE,tau,nE,tau)
    # fill in the block diagonal entries
    for i in range(Cyy4_tyee.shape[-4]):
        Cyy_yetet[...,:,i,:,i] = Cyy4_tyee[0,:,:,:]
        for j in range(i+1,Cyy4_tyee.shape[-4]):
            Cyy_yetet[...,:,i,:,j] = Cyy4_tyee[j-i,:,:,:]
            Cyy_yetet[...,:,j,:,i] = Cyy4_tyee[j-i,:,:,:].swapaxes(-2,-1) # transpose the event types
    if Cyy_tyee.ndim==3: # ( 1,nE,tau,nE,tau) -> (nE,tau,nE,tau)
        Cyy_yetet = Cyy_yetet[0,...]
    return Cyy_yetet

def compCxx_diag(X_TSd, tau:float, offset:float=0, unitnorm:bool=True, center:bool=False):
    '''
    Compute the main tau diagonal entries of a Cyy tensor
    Args:
      X_TSd (nTrl, nEp/nSamp, d): the new stim info to be added
               Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      tau (int): number of time-shifts to compute over
      offset (int): offset in for tau=0 (Note: ignored here)
      unitnorm(bool) : flag if we normalize the Cyy with number epochs. Defaults to True.
    Returns:
      Cxx_tdd -- (tau, d, d)
    '''
    if X_TSd.ndim == 2:  # ensure is 3-d
        X_TSd = X_TSd[np.newaxis, ...] 

    if False:
        # temporal embedding
        X_TStd = window_axis(X_TSd, winsz=tau, axis=-2)
        # shrink X w.r.t. the window and shift to align with the offset
        #X_TSd = X_TSd[:, tau-1:, ...] # shift fowards and shrink
        X_TSd = X_TSd[:, :X_TStd.shape[-3], ...] # shift fowards and shrink
        # compute the cross-covariance
        Cxx_tdd = np.einsum("TStd, TSe->tde", X_TStd, X_TSd, dtype=X_TSd.dtype)

    else: # loopy computation
        Cxx_tdd = np.zeros((tau,X_TSd.shape[-1],X_TSd.shape[-1]),dtype=X_TSd.dtype) # tau,y,e,y,e
        Cxx_tdd[0,...] = np.einsum('TSd,TSe->de',X_TSd,X_TSd) # special case as python makes :end+1 hard ...
        for t in range(1,tau): # manually slide over Y -- as einsum doesn't manage the memory well
            Cxx_tdd[t,...] = np.einsum('TSd,TSe->de',X_TSd[:,t:,...],X_TSd[:,:-t,...])

    if center:
        N = X_TSd.shape[0]*X_TSd.shape[1]
        sX = np.sum(X_TSd.reshape((N,X.shape[-1])),axis=0)
        muXX = np.einsum("i,j->ij",sX,sX) / N
        Cxx_tdd = Cxx_tdd - muXX

    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        Cxx_tdd = Cxx_tdd / (X_TSd.shape[0]*X_TSd.shape[1]) # / nTrl*nEp
    return Cxx_tdd

def compCxx_full(X_TSd, tau:float, offset=0, unitnorm:bool=False, center:bool=False):
    '''
    Compute the main tau diagonal entries of a Cyy tensor
    Args:
      X_TSd (nTrl, nSamp, d): the input d dimensional data
      tau (int): number of time-shifts to compute over
      offset (int): offset in for tau=0 (Note: ignored here)
      unitnorm(bool) : flag if we normalize the Cyy with number epochs. Defaults to True.
    Returns:
      Cxx_fdfd -- (tau, d, tau d) the full cross auto-covariance
    '''
    assert unitnorm==False
    assert center==False
    assert offset==0
    # full manual computation
    #X_TStd = window_axis(X_TSd, winsz=tau, axis=-2) 
    #Cxx_fdfd = np.einsum('TSfd,TSge->fdge',X_TStd,X_TStd)

    # loopy computation for debugging
    Cxx_fdfd = np.zeros( (tau,X_TSd.shape[-1],tau,X_TSd.shape[-1]), dtype=X_TSd.dtype)
    # N.B. tau is **always** the amount of backward shifting of X -> amount truncated from the front, i.e. -tau->0 = 0->tau
    for t1 in range(tau): # shift 1st part
        for t2 in range(tau): # shift 2nd part
            # N.B. this is very confusing.  But:
            #   t1,t2 are *backwards* shifts in X1,X2 resp
            #   shifting X2 backwards by tau means matching: X1_t with X2_(t-tau)
            #   do this by aligning: X1_tau with X2_(tau-tau)=X2_0
            #      which can be achieved by truncating X1 -> X1[tau:] 
            #   Similarly for X2.
            #   This leads to the non-intutive result that we 
            #      *shift* X1 with X2's tau  and vice.versa!
            if t1>t2:  # t1 is bigger, truncate 2nd to match lengths
                Cxx_fdfd[t1,:,t2,:] = np.einsum('TSd,TSe->de',X_TSd[:,t2:-(t1-t2),:], X_TSd[:,t1:,:])
            elif t2==t1:
                Cxx_fdfd[t1,:,t2,:] = np.einsum('TSd,TSe->de',X_TSd[:,t2:,:],         X_TSd[:,t1:,:])
            elif t2>t1: # t2 is bigger, truncate 1st to match lenghts
                Cxx_fdfd[t1,:,t2,:] = np.einsum('TSd,TSe->de',X_TSd[:,t2:,:],         X_TSd[:,t1:-(t2-t1),:])

    return Cxx_fdfd

def Cxx_diag2full(Cxx_tdd):
    """ convert compressed cross-auto-cov representation to a full one

    Args:
        Cxx_tdd ([type]): [description]

    Returns:
        Cxx_tdtd: the fully completed version
    """    
    MM = np.zeros((Cxx_tdd.shape[0],Cxx_tdd.shape[1],Cxx_tdd.shape[0],Cxx_tdd.shape[2]), dtype=Cxx_tdd.dtype)
    for t1 in range(Cxx_tdd.shape[0]):
        for t2 in range(Cxx_tdd.shape[0]):
            dt = t2 - t1
            if dt>=0:
                MM[t1,:,t2,:] = Cxx_tdd[abs(dt),:,:]
            else:
                MM[t1,:,t2,:] = Cxx_tdd[abs(dt),:,:].T
    return MM

def test_compCxx_diag():
    from mindaffectBCI.decoder.utils import testSignal
    import matplotlib.pyplot as plt

    irf=(.5,0,0,0,0,0,0,0,0,1)
    offset=0; # X->lag-by-10
    X,Y,st,A,B = testSignal(nTrl=1,nSamp=500,d=5,nE=1,nY=1,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)
    print("A={}\nB={}".format(A, B))
    print("X{}={}".format(X.shape, X[:30, np.argmax(np.abs(A))]))
    print("Y{}={}".format(Y.shape, Y[0, :30, 0]))

    tau = 6

    Cxx_tdtd0 = compCxx_full(X,tau,unitnorm=False)
    #print(f"Cxx_tdtd0={Cxx_tdtd0}")
    Cxx_tdd0 = Cxx_tdtd0[:,:,0,:]

    print("Cxx_tdd0={}".format(Cxx_tdd0.shape))

    # diag computation
    Cxx_tdd = compCxx_diag(X,tau,unitnorm=False)
    print("Cxx_tdd={}".format(Cxx_tdd.shape))

    print('delta Cxx={}'.format(np.max(np.abs(Cxx_tdd-Cxx_tdd0))))

    plt.figure()
    plt.subplot(311)
    plt.imshow(Cxx_tdd0.reshape((Cxx_tdd0.shape[0],-1)))
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(Cxx_tdd.reshape((Cxx_tdd.shape[0],-1)))
    plt.colorbar()
    plt.subplot(313)
    plt.imshow((Cxx_tdd-Cxx_tdd0).reshape((Cxx_tdd.shape[0],-1)))
    plt.colorbar()
    plt.show()

    Cxx_tdtd = Cxx_diag2full(Cxx_tdd)
    print("N.B. can be approx because of the diagnonal copy in the up-sample")
    print('delta Cxx_full={}'.format(np.max(np.abs(Cxx_tdtd-Cxx_tdtd0))))
    #print(f"Cxx_tdtd0={Cxx_tdtd0}")
    #print(f"Cxx_tdtd={Cxx_tdtd}")

    plt.figure()
    plt.subplot(311)
    plt.imshow(Cxx_tdtd0.reshape((Cxx_tdtd0.shape[0]*Cxx_tdtd0.shape[1],-1)))
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(Cxx_tdtd.reshape((Cxx_tdtd.shape[0]*Cxx_tdtd.shape[1],-1)))
    plt.colorbar()
    plt.subplot(313)
    plt.imshow((Cxx_tdtd-Cxx_tdtd0).reshape((Cxx_tdtd.shape[0]*Cxx_tdtd.shape[1],-1)))
    plt.colorbar()
    plt.show()


def compCyx_diag(X_TSd, Y_TSye, tau=None, offset=0, center=False, verb=0, unitnorm=True):
    '''
    Args:
    X_TSd (nTrl, nSamp, d):  raw response for the current stimulus event
            d=#electrodes
    Y_TSye (nTrl, nSamp, nY, nE): Indicator for which events occured for which outputs
      OR Y_TSy : per-output sequence
            nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
    Returns:
        Cyx_tyed (nY, nE, tau, d): cross covariance at different offsets
        taus : relative offsets (t_y - t_x)
    '''
    if X_TSd.ndim == 2: 
        X_TSd = X_TSd[np.newaxis, ...] 
        Y_TSye = Y_TSye[np.newaxis, ...]
    if Y_TSye.ndim == 3: # add feature dim if needed
        Y_TSye = Y_TSye[...,np.newaxis] 
    if verb > 1: print("tau={}".format(tau))
    if not hasattr(tau,'__iter__'): tau=(0,tau)
    if offset is None : offset = 0
    if not hasattr(offset,'__iter__'): offset=(0,offset)

    # TODO[]: optimized version for self-computation
    if X_TSd is Y_TSye : # same input
        pass
        #Cyx_tyefd = comp_Cxxdiag(X_TSd, tau=tau, offset=offset, unitnorm=unitnorm, center=center)

    # work out a single shift-length for shifting X
    taus = list(range(-(offset[0] + tau[0]-1 - offset[1]), (offset[1] + tau[1]-1 - offset[0])+1))

    Cyx_tyed = np.zeros((len(taus),)+Y_TSye.shape[2:]+X_TSd.shape[2:],dtype=X_TSd.dtype)
    for i,t in enumerate(taus):
        if t < 0 : # shift Y backwards -> X preceeds Y
            Cyx_tyed[i,...] = np.einsum('TS...,TSd->...d',Y_TSye[:,-t:,:,:],X_TSd[:,:t,:])

        elif t == 0 : # aligned
            Cyx_tyed[i,...] = np.einsum('TS...,TSd->...d',Y_TSye,X_TSd)

        elif t > 0: # shift X backwards -> Y preceeds X
            Cyx_tyed[i,...] = np.einsum('TS...,TSd->...d',Y_TSye[:,:-t,:,:],X_TSd[:,t:,:])

    if center:
        if verb > 1: print("center")
        # N.B. (X-mu*1.T)@Y.T = X@Y.T - (mu*1.T)@Y.T = X@Y.T - mu*1.T@Y.T = X@Y.T - mu*\sum(Y)  
        muX = np.mean(X_TSd,axis=(0,1)) 
        muY = np.sum(Y_TSye,(0,1))
        muXY_yed= np.einsum("ye,d->yed",muY,muX) 
        Cyx_tyed = Cyx_tyed - muXY_yed[np.newaxis,...] #(nY,nE,tau,d)

    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        Cyx_tyed = Cyx_tyed / (Y_TSye.shape[0]*Y_TSye.shape[1]) # / nTrl*nEp
    
    return Cyx_tyed, taus


def compCyx_full(X_TSd, Y_TSye, tau=None, offset=0, center=False, verb=0, unitnorm=True):
    '''
    Args:
    X_TSd (nTrl, nSamp, d):  raw response for the current stimulus event
            d=#electrodes
    Y_TSye (nTrl, nSamp, nY, nE): Indicator for which events occured for which outputs
            nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
    Returns:
        Cyx_tyetd (nY, nE, tau, d): cross covariance at different offsets
    '''
    if X_TSd.ndim == 2: 
        X_TSd = X_TSd[np.newaxis, ...] 
    if Y_TSye.ndim == 3:
        Y_TSye = Y_TSye[np.newaxis, ...] 
    if verb > 1: print("tau={}".format(tau))
    if not hasattr(tau,'__iter__'): tau=(0,tau)
    if offset is None : offset = 0
    if not hasattr(offset,'__iter__'): offset=(0,offset)

    assert unitnorm==False
    assert center==False
    assert offset==(0,0)

    # loopy computation
    Cyx_tyefd = np.zeros((tau[1],Y_TSye.shape[-2],Y_TSye.shape[-1],tau[0],X_TSd.shape[-1]), dtype=X_TSd.dtype)
    for f in range(tau[0]): # backwards shift X
        for t in range(tau[1]): # backwards shift y
            # N.B. this is very confusing.  But:
            #   f,t are *backwards* shifts in X,Y resp
            #   shifting Y backwards by tau means matching: X_t with Y_(t-tau)
            #   do this by aligning: X_tau with Y_(tau-tau)=Y_0
            #      which can be achieved by truncating X -> X[tau:] 
            #   Similarly for Y.
            #   This leads to the non-intutive result that we 
            #      *shift* X with Y's tau  and vice.versa!
            if t > f: # Y shift is bigger, truncate X to match lengths
                Cyx_tyefd[t,:,:,f,:] = np.einsum("TSye,TSd->yed",Y_TSye[:,f:-(t-f),:,:], X_TSd[:,t:,:])
            elif t == f:
                Cyx_tyefd[t,:,:,f,:] = np.einsum("TSye,TSd->yed",Y_TSye[:,f:,:,:],  X_TSd[:,t:,:])
            elif f > t: # X shift is bigger, truncate Y to match lengths
                Cyx_tyefd[t,:,:,f,:] = np.einsum("TSye,TSd->yed",Y_TSye[:,f:,:,:], X_TSd[:,t:-(f-t),:])

    # X_TSfd = window_axis(X_TSd, winsz=tau[0], axis=-2) 
    # Y_TStye = window_axis(Y_TSye, winsz=tau[1], axis=-3)
    # if tau[0] < tau[1]: # truncate X
    #    Cyx_tyefd = np.einsum('TStye,TSfd->tyefd',Y_TStye,X_TSfd[:,tau[1]-tau[0]:,...])
    # else: # truncate Y
    #    Cyx_tyefd = np.einsum('TStye,TSfd->tyefd',Y_TStye[:,tau[0]-tau[1]:,...],X_TSfd)
 
    return Cyx_tyefd

def Cyx_diag2full(Cyx_tyed,tau,offset=None):
    """ convert compressed cross-auto-cov representation to full one

    Args:
        Cyx_tyed ([type]): [description]
        tau ([type]): [description]
        offset ([type], optional): [description]. Defaults to None.

    Returns:
        Cyx_tyefd: t x f inflated representation
    """    
    if not hasattr(tau,'__iter__'): tau=(0,tau)
    assert offset is None or offset == 0 

    i0 = tau[0]-1 # index of t_y - t_x == 0 in dtau list

    Cyx_tyefd = np.zeros((tau[1],Cyx_tyed.shape[1],Cyx_tyed.shape[2],tau[0],Cyx_tyed.shape[3]),dtype=Cyx_tyed.dtype)
    for f in range(tau[0]):
        for t in range(tau[1]):
            dt = t - f 
            i = i0 + dt
            Cyx_tyefd[t,:,:,f,:] = Cyx_tyed[i,...]

    return Cyx_tyefd

def test_compCyx_diag():
    from mindaffectBCI.decoder.utils import testSignal
    import matplotlib.pyplot as plt

    irf=(.5,0,0,0,0,0,0,0,0,1)
    offset=0; # X->lag-by-10
    X,Y,st,A,B = testSignal(nTrl=1,nSamp=500,d=1,nE=1,nY=1,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)
    print("A={}\nB={}".format(A, B))
    print("X{}".format(X.shape))
    print("Y{}".format(Y.shape))

    tau = (5,15)

    # full manual computation
    Cyx_tyefd0 = compCyx_full(X,Y,tau,center=False,unitnorm=False)

    print("Cyx_tyefd0={} = {}".format(Cyx_tyefd0.shape,np.squeeze(Cyx_tyefd0)))
    #print(f"Cyx_tyetd0={Cyx_tyetd0}")
    Cyx_x_tyed =  np.moveaxis(Cyx_tyefd0[0,:,:,:0:-1,:],-2,0) # WARNING: time-reversed!
    Cyx_y_tyed =  Cyx_tyefd0[:,:,:,0,:]   
    Cyx_tyed0 = np.concatenate((Cyx_x_tyed,Cyx_y_tyed),0)

    print("Cyx_tyed0={}".format(Cyx_tyed0.shape))

    # diag computation
    Cyx_tyed, taus = compCyx_diag(X,Y,tau,unitnorm=False)
    print("taus={}".format(taus))
    print("Cyx_tyed={}".format(Cyx_tyed.shape))
    #print(f"Cyx_tyed={Cyx_tyed}")

    print('delta Cxy={}'.format(np.max(np.abs(Cyx_tyed-Cyx_tyed0))))

    plt.figure()
    plt.subplot(311)
    plt.imshow(Cyx_tyed0.reshape((Cyx_tyed0.shape[0],-1)))
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(Cyx_tyed.reshape((Cyx_tyed.shape[0],-1)))
    plt.colorbar()
    plt.subplot(313)
    plt.imshow((Cyx_tyed-Cyx_tyed0).reshape((Cyx_tyed.shape[0],-1)))
    plt.colorbar()
    plt.show()

    Cyx_tyefd = Cyx_diag2full(Cyx_tyed,tau)
    print("Cyx_tyefd = {}  {}".format(Cyx_tyefd.shape,np.squeeze(Cyx_tyefd)))
    print("N.B. can be approx because of the diagnonal copy in the up-sample")
    print('delta Cxy_full={}'.format(np.max(np.abs(Cyx_tyefd-Cyx_tyefd0))))
    #print(f"Cyx_tyetd0={Cyx_tyetd0}")
    #print(f"Cyx_tyetd={Cyx_tyetd}")
    plt.figure()
    plt.subplot(311)
    plt.imshow(Cyx_tyefd0[:,:,:,:,:].reshape((-1,Cyx_tyefd0.shape[-1]*Cyx_tyefd0.shape[-2])))
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(Cyx_tyefd.reshape((-1,Cyx_tyefd.shape[-1]*Cyx_tyefd.shape[-2])))
    plt.colorbar()
    plt.subplot(313)
    plt.imshow((Cyx_tyefd-Cyx_tyefd0[:,:,:,:,:]).reshape((-1,Cyx_tyefd.shape[-1]*Cyx_tyefd.shape[-2])))
    plt.colorbar()
    plt.suptitle('Cyx_tyefd')
    plt.show()

def compCyy_diag_perY(Y, tau:float, unitnorm:bool=True, perY:bool=True):
    '''
    Compute the main tau diagonal entries of a Cyy tensor for each output independently
    Args:
      Y_TSye (nTrl, nSamp, nY, nE): the new stim info to be added
               Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      tau (int): number of samples in the stimulus response
      unitnorm(bool) : flag if we normalize the Cyy with number epochs. Defaults to True.
    Returns:
      Cyy_tyee (tau, nY, nE, nE):
    '''
    if Y.ndim == 3:  # ensure is 4-d
        Y = Y[np.newaxis, :, :, :] # (nTrl, nEp/nSamp, nY, nE)

    if not np.issubdtype(Y.dtype, np.floating): # all at once
        Y = Y.astype(np.float32)

    Cyy_tyee = np.zeros((tau,Y.shape[-2],Y.shape[-1],Y.shape[-1]),dtype=Y.dtype) # tau,y,e,e
    Cyy_tyee[0,...] = np.einsum('TSye,TSyf->yef',Y,Y) # special case as python makes :end+1 hard ...
    for t in range(1,tau): # manually slide over Y -- as einsum doesn't manage the memory well
        Cyy_tyee[t,...] = np.einsum('TSye,TSyf->yef',Y[:,t:,:,:],Y[:,:-t,:,:])

    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        Cyy_tyee = Cyy_tyee / (Y.shape[0]*Y.shape[1]) # / nTrl*nEp
    return Cyy_tyee

def compCyy_diag(Y, tau:float, unitnorm:bool=True):
    '''
    Compute the main tau diagonal entries of a Cyy tensor
    Args:
      Y (nTrl, nSamp, nY, nE): the new stim info to be added
               Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      tau (int): number of samples in the stimulus response
      unitnorm(bool) : flag if we normalize the Cyy with number epochs. Defaults to True.
    Returns:
      Cyy_tyeye (tau,nY,nE,nY,nE):
    '''
    if Y.ndim == 3:  # ensure is 4-d
        Y = Y[np.newaxis, :, :, :] # (nTrl, nEp/nSamp, nY, nE)

    if not np.issubdtype(Y.dtype, np.floating): # all at once
        Y = Y.astype(np.float32)

    Cyy_tyeye = np.zeros((tau,Y.shape[-2],Y.shape[-1],Y.shape[-2],Y.shape[-1]),dtype=Y.dtype) # tau,y,e,y,e
    Cyy_tyeye[0,...] = np.einsum('TSye,TSzf->yezf',Y,Y) # special case as python makes :end+1 hard ...
    for t in range(1,tau): # manually slide over Y -- as einsum doesn't manage the memory well
        Cyy_tyeye[t,...] = np.einsum('TSye,TSzf->yezf',Y[:,t:,:,:],Y[:,:-t,:,:])

    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        Cyy_tyeye = Cyy_tyeye / (Y.shape[0]*Y.shape[1]) # / nTrl*nEp
    return Cyy_tyeye


def Cyy_tyeye_diag2full(Cyy_tyeye):
    """[summary]

    Args:
        Cyy_tyeye ([type]): the input compressed version of Cyy

    Returns:
        Cyy_yetyet: the expanded version of Cyy
    """    
    t=Cyy_tyeye.shape[0]
    y=Cyy_tyeye.shape[1]
    e=Cyy_tyeye.shape[2]
    
    Cyy_yetyet = np.zeros((y,e,t,y,e,t),dtype=Cyy_tyeye.dtype)
    # fill in the block diagonal entries
    for i in range(t):
        Cyy_yetyet[:,:,i,:,:,i] = Cyy_tyeye[0,:,:,:,:]
        for j in range(i+1,t):
            Cyy_yetyet[:,:,i,:,:,j] = Cyy_tyeye[j-i,:,:,:,:]
            # lower diag, transpose the event types
            Cyy_yetyet[:,:,j,:,:,i] = Cyy_tyeye[j-i,:,:,:,:].swapaxes(-3,-1).swapaxes(-4,-2)
    return Cyy_yetyet

def Cyy_yetet_diag2full(Cyy_yetet):
    """[summary]

    Args:
        Cyy_yetet ([type]): the input compressed version of Cyy

    Returns:
        Cyy_yetyet: the expanded version of Cyy
    """    
    assert Cyy_yetet.shape[0]==1
    return Cyy_yetet[:,:,:,np.newaxis,:,:]


def compCyy_full(Y_TSye, tau:int, offset:int=0, unitnorm:bool=True):
    '''
    Compute the full YY cross auto-covariance
    
    Args:
      Y_TSye (nTrl, nSamp, nY, nE): the new stim info to be added
               Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      tau (int): number of samples in the stimulus response
      offset (int): offset from time-zero for the cross computation (Note: ignored here)
      unitnorm(bool) : flag if we normalize the Cyy with number epochs. Defaults to True.
    Returns:
      Cyy_yetyet -- (nY, nE, tau, nY, nE, tau)
    '''

    if Y_TSye.ndim == 3:  # ensure is 4-d
        Y_TSye = Y_TSye[np.newaxis, :, :, :] # (nTrl, nEp/nSamp, nY, nE)

    if not np.issubdtype(Y_TSye.dtype, np.floating): # all at once
        Y_TSye = Y_TSye.astype(np.float32)

    # loopy computation, for debugging
    Cyy_yetyet = np.zeros((Y_TSye.shape[-2],Y_TSye.shape[-1],tau,Y_TSye.shape[-2],Y_TSye.shape[-1],tau),dtype=Y_TSye.dtype) # tau,y,e,y,e
    for t1 in range(tau): # manually slide over Y -- as einsum doesn't manage the memory well
        for t2 in range(tau):
            # N.B. this is very confusing.  But:
            #   t1,t2 are *backwards* shifts in X1,X2 resp
            #   shifting X2 backwards by tau means matching: X1_t with X2_(t-tau)
            #   do this by aligning: X1_tau with X2_(tau-tau)=X2_0
            #      which can be achieved by truncating X1 -> X1[tau:] 
            #   Similarly for X2.
            #   This leads to the non-intutive result that we 
            #      *shift* X1 with X2's tau  and vice.versa!           
            if t1>t2: # t1 shift is bigger, truncate 2 to match lengths
                Cyy_yetyet[:,:,t1,:,:,t2] = np.einsum('TSye,TSzf->yezf',Y_TSye[:,t2:-(t1-t2),:,:],Y_TSye[:,t1:,:,:])
            elif t1==t2:
                Cyy_yetyet[:,:,t1,:,:,t2] = np.einsum('TSye,TSzf->yezf',Y_TSye[:,t2:,:,:],Y_TSye[:,t1:,:,:])
            elif t2>t1:
                Cyy_yetyet[:,:,t1,:,:,t2] = np.einsum('TSye,TSzf->yezf',Y_TSye[:,t2:,:,:],Y_TSye[:,t1:-(t2-t1),:,:])

    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        Cyy_yetyet = Cyy_yetyet / (Y.shape[0]*Y.shape[1]) # / nTrl*nEp
    return Cyy_yetyet

def autocov(X, tau):
    '''
    Compute the cross-auto-correlation of X, i.e. spatial-temporal covariance
    Inputs:
      X -- (nTrl, nSamp, d) the data to have autocov computed
               nSamp = temporal dim, d = spatial dim
      tau  -- number of samples in the autocov
    Outputs:
      Ctdtd -- (tau, d, tau, d)
    '''
    if X.ndim == 2: # ensure 3-d
        X = X[np.newaxis, :, :]
    if X.ndim > 4: # ensure 3-d
        X = X.reshape((np.prod(X.shape[:-2]),)+X.shape[-2:])

    if np.issubdtype(X.dtype, np.float): # all at once
        Xs = window_axis(X, winsz=tau, axis=-2) # window of length tau (nTrl, nSamp-tau, tau, d)
        Ctdtd=np.einsum("Tstd, Tsue -> tdue", Xs, Xs, optimize='optimal') # compute cross-covariance (tau, d, tau, d)

    else: # trial at a time + convert to float
        Ctdtd = np.zeros((tau, X.shape[-1], tau, X.shape[-1]))
        # different slice time for every trial, up-sample per-trial
        for ti in range(X.shape[0]):  # loop over trials            
            Xi = X[ti, :, :,]
            if not np.issubdtype(X.dtype, np.float):
                Xi = np.array(Xi, np.float)
            Xi = window_axis(Xi, winsz=tau, axis=-2)
            Ctdtd = Ctdtd + np.einsum("Etd, Eue->tdue", Xi, Xi)
            
    return Ctdtd

def cov(X):
    '''
    Compute the spatial covariance
    Input:
     X = (...,d) data, with features in the last dimension
    Output:
     Cxx = (d,d)
    '''
    # make X 2d
    X2d = np.reshape(X, (np.prod(X.shape[:-1]), X.shape[-1]))
    Cxx  = np.einsum("Td, Te->de", X2d, X2d)
    return Cxx

def crossautocov(X, Y, tau, offset=0, per_trial:bool=False, verb:int=0):
    '''
    Compute the cross-auto-correlation between 2 datasets, X,Y, i.e. the double spatial-temporal covariance
    Inputs:
      X -- (nTrl, nSamp, dx) the data to have autocov computed
               nSamp = temporal dim, dx = spatial dim x
      Y -- (nTrl, nSamp, dy) the data to have autocov computed
               nSamp = temporal dim, dy = spatial dim y
      tau  -- :int number of samples in the cov
             OR
              [int,int] time lags for x and y
    Outputs:
      Ctdtd -- (taux, dx, tauy, dy)
    '''
    if X.ndim == 2: # ensure 3-d
        X = X[np.newaxis, :, :]
    if X.ndim > 4: # ensure 3-d
        X = X.reshape((np.prod(X.shape[:-2]),)+X.shape[-2:])
    if Y.ndim == 2: # ensure 3-d
        Y = Y[np.newaxis, :, :]
    if Y.ndim > 4: # ensure 3-d
        Y = Y.reshape((np.prod(Y.shape[:-2]),)+Y.shape[-2:])
    if not hasattr(tau, '__iter__'): # scalar tau, same for x and y
        tau = (tau, tau)
    if offset!=0:
        raise NotImplementedError("Offsets not supported yet...")
        
    #TODO[]: optimize for memory usage?
    # TODO[]: tau at a time version!
    if False and X.shape[0] < 100 and np.issubdtype(X.dtype, np.float) and np.issubdtype(Y.dtype, np.float): # all at once
        Xs = window_axis(X, winsz=tau[0], axis=-2) # window of length tau (nTrl, nSamp-tau, tau, d)
        Ys = window_axis(Y, winsz=tau[1], axis=-2) # window of length tau (nTrl, nSamp-tau, tau, d)
        
        # reshape back to same shape if diff window sizes
        if tau[0] < tau[1]:
            Xs = Xs[..., :Ys.shape[-3], :, :]
        else:
            Ys = Ys[..., -offset:Xs.shape[-3]-offset, :, :]

        # compute cross-covariance (tau, d, tau, d)
        Ctdtd = np.einsum("Tstd, Tsue -> tdue", Xs, Ys, optimize='optimal')
        
    elif False: # trial at a time + convert to float
        Ctdtd = np.zeros((tau[0], X.shape[-1], tau[1], Y.shape[-1]),dtype=X.dtype)
        # different slice time for every trial, up-sample per-trial
        for ti in range(X.shape[0]):  # loop over trials
            
            Xi = X[ti, :, :]
            if not np.issubdtype(X.dtype, np.float32) and not np.issubdtype(X.dtype,np.float64):
                Xi = np.array(Xi, np.float)
            Xi = window_axis(Xi, winsz=tau[0], axis=-2)
            
            Yi = Y[ti, :, :]
            if not np.issubdtype(Y.dtype, np.float32) and not np.issubdtype(Y.dtype,np.float64):
                Yi = np.array(Yi, np.float)
            Yi = window_axis(Yi, winsz=tau[1], axis=-2)
            
            # reshape back to same shape if diff window-sizes
            if tau[0] < tau[1]:
                Xi = Xi[..., :Yi.shape[-3], :, :]
            else:
                Yi = Yi[..., offset:Xi.shape[-3]+offset, :, :]

            # compute cross-covariance (tau, d, tau, d)
            Ctdtd = Ctdtd + np.einsum("Etd, Eue -> tdue", Xi, Yi)
        
    else:
        tau_xs = range(tau[0])
        tau_ys = range(tau[1])
        # if not hasattr(tau[0],'__iter__'):
        #     tau_xs = list(range(tau[0])) if tau[0]>=0 else list(range(0,tau[0],-1))
        # if not hasattr(tau[1],'__iter__'):
        #     tau_ys = list(range(tau[1])) if tau[1]>=0 else list(range(0,tau[0],-1))

        Ctdtd = np.zeros((len(tau_xs),X.shape[-1],len(tau_ys),Y.shape[-1]),dtype=X.dtype)
        # different slice time for every trial, up-sample per-trial'
        if verb>0: print("{}Tr:".format(X.shape[0]),end='')
        for ti in range(X.shape[0]):  # loop over trials
            Xi = X[ti,...]
            Yi = Y[ti,...]
            Ctdtdi = np.zeros(Ctdtd.shape,dtype=Ctdtd.dtype)
            if verb>0: print("{} ".format(ti),end='')
            for tau_x in tau_xs: # manually slide over Y -- as einsum doesn't manage the memory well
                for tau_y in tau_ys:
                    # N.B. this is very confusing.  But:
                    #   t1,t2 are *backwards* shifts in X1,X2 resp
                    #   shifting X2 backwards by tau means matching: X1_t with X2_(t-tau)
                    #   do this by aligning: X1_tau with X2_(tau-tau)=X2_0
                    #      which can be achieved by truncating X1 -> X1[tau:] 
                    #   Similarly for X2.
                    #   This leads to the non-intutive result that we 
                    #      *shift* X1 with X2's tau  and vice.versa!           
                    if tau_x>tau_y: # t1 shift is bigger, truncate 2 to match lengths
                        Ctdtdi[tau_x,:,tau_y,:] = Xi[tau_y:-(tau_x-tau_y),:].T @ Yi[tau_x:,:]
                    elif tau_x==tau_y:
                        Ctdtdi[tau_x,:,tau_y,:] = Xi[tau_y:,:].T @ Yi[tau_x:,:]
                    elif tau_x<tau_y:
                        Ctdtdi[tau_x,:,tau_y,:] = Xi[tau_y:,:].T @ Yi[tau_x:-(tau_y-tau_x),:]
            # accumulate
            if per_trial: # store
                Ctdtd[ti,...] = Ctdtdi
            else: # accumulate
                Ctdtd = Ctdtd + Ctdtdi
        if verb>0: print()
    return Ctdtd

def plot_summary_statistics(Cxx_dd, Cyx_yetd, Cyy_yetet, 
                            evtlabs=None, outputs=None, times=None, ch_names=None, fs=None, offset:int=0, label:str=None):
    """Visualize the summary statistics (Cxx_dd, Cyx_yetd, Cyy) of a dataset

    It is assumed the data has 'd' channels, with 'nE' different types of
    trigger event, and a response length of 'tau' for each trigger.

    Args:
        Cxx_dd (d,d): spatial covariance
        Cxy_yetd (nY, nE, tau, d): per-output event related potentials (ERPs)
        Cyy_yetet (nY, nE, tau, nE, tau): updated response covariance for each output
        evtlabs ([type], optional): the labels for the event types. Defaults to None.
        times ([type], optional): values for the time-points along tau. Defaults to None.
        ch_names ([type], optional): textual names for the channels. Defaults to None.
        fs ([type], optional): sampling rate for the data along tau (used to make times if not given). Defaults to None.
    """    
    import matplotlib.pyplot as plt
    if times is None:
        times = np.arange(Cyx_yetd.shape[-2]) + offset
        if fs is not None:
            times = times / fs
    if ch_names is None:
        ch_names = ["{}".format(i) for i in range(Cyx_yetd.shape[-1])]
    if evtlabs is None:
        evtlabs = ["{}".format(i) for i in range(Cyx_yetd.shape[-3])]
    if outputs is None:
        outputs = ["{}".format(i) for i in range(Cyx_yetd.shape[0])]

    plt.clf()
    # Cxx_dd
    plt.subplot(311)
    plt.imshow(Cxx_dd, origin='lower', extent=[0, Cxx_dd.shape[0], 0, Cxx_dd.shape[1]])
    plt.colorbar()
    # TODO []: use the ch_names to add lables to the  axes
    plt.title('Cxx_dd')

    # Cxy
    nout = Cyx_yetd.shape[-4] if Cyx_yetd.ndim>3 else 1
    nevt = Cyx_yetd.shape[-3]
    if Cyx_yetd.ndim > 3:
        if Cyx_yetd.shape[0] > 1:
            print("Warning: Y's merged into event types")
        Cyx_yetd = Cyx_yetd.reshape((-1,Cyx_yetd.shape[-2],Cyx_yetd.shape[-1]))
    clim = [np.min(Cyx_yetd.flat), np.max(Cyx_yetd.flat)]
    # TODO[]: make into 2d nout x nevt]
    for ei in range(nevt*nout):
        if ei==0:
            ax = plt.subplot(3,nevt*nout,nevt*nout+ei+1) # N.B. subplot indexs from 1!
            plt.xlabel('time (s)')
            plt.ylabel('space')
        else: # no axis on other sub-plots
            plt.subplot(3, nevt*nout, nevt*nout+ei+1, sharey=ax, sharex=ax)
            plt.tick_params(labelbottom=False, labelleft=False)
        # TODO []: use the ch_names, times to add lables to the  axes
        plt.imshow(Cyx_yetd[ei, :, :].T, aspect='auto', origin='lower', extent=(times[0], times[-1], 0, Cyx_yetd.shape[-1]))
        plt.clim(clim)
        try:
            if nevt>1 and nout>1:
                title = '{}:{}'.format(outputs[ei//nevt],evtlabs[ei%nevt])
            elif nevt==1:
                title = '{}'.format(outputs[ei] if outputs else None)
            elif nout==1:
                title = '{}'.format(evtlabs[ei] if evtlabs else None)
            else:
                title=[]
            plt.title(title)
        except:
            pass
    # only last one has colorbar
    plt.colorbar()

    # Cyy_yetet
    if Cyy_yetet.ndim > 4 and Cyy_yetet.shape[0] > 1:
            print("Warning: Y's merged into event types")
    Cyy2d = np.reshape(Cyy_yetet, (-1, Cyy_yetet.shape[-2]*Cyy_yetet.shape[-1]))
    plt.subplot(313)
    plt.imshow(Cyy2d, origin='lower', aspect='auto', extent=[0, Cyy2d.shape[0], 0, Cyy2d.shape[1]])
    plt.colorbar()
    plt.title('Cyy')
    if label:
        plt.suptitle(label)

def plotCxy(Cyx_yetd,evtlabs=None,fs=None):
    import matplotlib.pyplot as plt
    times = np.arange(Cyx_yetd.shape[-2])
    if fs is not None: times = times/fs
    ncols = Cyx_yetd.shape[1] if Cyx_yetd.shape[1]<10 else Cyx_yetd.shape[1]//10
    nrows = Cyx_yetd.shape[1]//ncols + (1 if Cyx_yetd.shape[1]%ncols>0 else 0)
    clim = (np.min(Cyx_yetd), np.max(Cyx_yetd))
    fit,ax = plt.subplots(nrows=nrows,ncols=ncols,sharey='row',sharex='col')
    for ei in range(Cyx_yetd.shape[1]):
        plt.sca(ax.flat[ei])
        plt.imshow(Cyx_yetd[0,ei,:,:].T,aspect='auto',extent=[times[0],times[-1],0,Cyx_yetd.shape[-1]])
        plt.clim(clim)
        if evtlabs is not None: plt.title(evtlabs[ei])

def plot_trial(X_TSd,Y_TSy,fs:float=None,
                ch_names=None,evtlabs=None,outputs=None,times=None, 
                ylabel:str='ch + output', suptitle:str=None, show:bool=None):
    """visualize a single trial with data and stimulus sequences

    Args:
        X_TSd ([type]): raw data sequence
        Y_TSy ([type]): raw stimulus sequence
        fs (float, optional): sample rate of data and stimulus. Defaults to None.
        ch_names (list-of-str, optional): channel names for X. Defaults to None.
    """    
    import matplotlib.pyplot as plt
    # if X_TSd.shape[1]==1:
    #     print("Warning: data with no samples?  colapsed")
    #     X_TSd=X_TSd[:,0,...]
    #     Y_TSy=Y_TSy[:,0,...]
    if not X_TSd is None:
        if X_TSd.ndim<3:
            X_TSd = X_TSd.reshape( (1,)*(3-X_TSd.ndim) + X_TSd.shape )

    if not Y_TSy is None and Y_TSy.ndim==2 : Y_TSy=Y_TSy[np.newaxis,...]
    ntrl = X_TSd.shape[0] if X_TSd is not None else Y_TSy.shape[0]
    nsamp = X_TSd.shape[1] if X_TSd is not None else Y_TSy.shape[1]

    times = np.arange(nsamp)/fs if fs is not None else np.arange(nsamp)
    if ch_names is None and X_TSd is not None: ch_names = np.arange(X_TSd.shape[2],dtype=int)
    if outputs is None and not Y_TSy is None:  outputs  = np.arange(Y_TSy.shape[2])
    if evtlabs is None and not Y_TSy is None and Y_TSy.ndim>=4: evtlabs  = np.arange(np.prod(Y_TSy.shape[3:]))
    # strip unused outputs to simplify plot
    if not Y_TSy is None :
        if Y_TSy.ndim < 4:
            Y_TSy=Y_TSy[...,np.any(Y_TSy,axis=tuple(range(Y_TSy.ndim-1)))]
        else:
            Y_TSy=Y_TSy[...,np.any(Y_TSy,axis=tuple(range(Y_TSy.ndim-2))+(-1,)),:]

    #print(Y_TSy.shape)
    linespace = 2 #np.mean(np.abs(X_TSd[X_TSd!=0]))*2
    for i in range(ntrl):
        if i==0:
            ax1 = plt.subplot(ntrl,1,i+1)
        else:
            plt.subplot(ntrl,1,i+1,sharex=ax1,sharey=ax1)

        if X_TSd is not None:
            xscale = np.percentile(np.abs(X_TSd),70)
            #print(xscale)
            trlen = np.flatnonzero(np.any(X_TSd[i,...],-1))[-1]
            for c in range(X_TSd.shape[-1]):
                tmp = X_TSd[i,...,c]
                tmp = tmp[...,:trlen].ravel()
                tmp = (tmp - np.mean(tmp)) / xscale / 2
                lab = ch_names[c] if c < len(ch_names) else c
                if tmp.size < times.size:
                    plt.plot(times[:trlen],tmp+c*linespace,label='X {}'.format(lab))
                else:
                    plt.plot(tmp+c*linespace,label='X {}'.format(lab))

        if Y_TSy is not None:
            nd = X_TSd.shape[-1] if X_TSd is not None else 0
            if Y_TSy.ndim==3:
                trlen = Y_TSy.shape[1]
                for y in range(Y_TSy.shape[2]):
                    tmp = Y_TSy[i,...,y].ravel() / np.max(Y_TSy)
                    if tmp.size == times.size:
                        plt.plot(times[:trlen],tmp+y+nd*linespace+2,'.-',label='Y {}'.format(y))
                    else:
                        plt.plot(tmp+y+nd*linespace+2,'.-',label='Y {}'.format(y))

            elif Y_TSy.ndim>=4:
                if Y_TSy.ndim>4:
                    Y_TSy = Y_TSy.reshape(Y_TSy.shape[:3]+(-1,))
                trlen = np.flatnonzero(np.any(Y_TSy[i,...],-1))[-1]
                for y in range(Y_TSy.shape[2]):
                    out = outputs[y] if outputs is not None and y < len(outputs) else y
                    for e in range(Y_TSy.shape[-1]):
                        evt = evtlabs[e] if evtlabs is not None and e < len(evtlabs) else e
                        tmp = Y_TSy[i,...,y,e] / np.max(Y_TSy)
                        plt.plot(times[:trlen],tmp.ravel()+y+nd*linespace+2+e+y*Y_TSy.shape[2],'.-',label='Y {}:{}'.format(out,evt))

        plt.title('Trl {}'.format(i))
        plt.grid(True)
        plt.xlabel('time (s)' if fs is not None else 'time (samp)')
        plt.ylabel(ylabel)
    plt.legend()
    if suptitle:
        plt.suptitle(suptitle)
    if show is not None: plt.show(block=show)


def plot_erp(erp_yetd, evtlabs=None, outputs=None, times=None, fs:float=None, ch_names=None, 
             axis:int=-1, plottype='plot', offset:int=0, ylim=None, suptitle:str=None, show:bool=None):
    '''
    Make a multi-plot of the event ERPs (as stored in erp)
    erp = (nY, nE, tau, d) current per output ERPs
    '''

    # ensure 4-d
    if erp_yetd.ndim<4:
        erp_yetd=erp_yetd.reshape( (1,)*(4-erp_yetd.ndim)+erp_yetd.shape)

    nevt = erp_yetd.shape[1] if erp_yetd.ndim>2 else 1
    nout = erp_yetd.shape[0] if erp_yetd.ndim>3 else 1

    if outputs is None:
        outputs = ["{}".format(i) for i in range(nout)]
    if evtlabs is None:
        evtlabs = ["{}".format(i) for i in range(nevt)]
    if times is None:
        times = list(range(offset,erp_yetd.shape[2]+offset))
        if fs is not None:
            times = [ t/fs for t in times ]

    if ch_names is None:
        ch_names = ["{}".format(i) for i in range(erp_yetd.shape[-1])]

    if erp_yetd.shape[0]>1 :
        print("Multiple Y's merged!")
    erp_etd = erp_yetd.reshape((-1,)+erp_yetd.shape[2:])
    # update evtlabs with the outputs info
    if nevt>1 and nout>1:
        evtlabs = [ "{}:{}".format(o,e) for o in outputs for e in evtlabs]
    elif nevt==1 and nout>1:
        evtlabs = outputs

    # BODGE: for no-time dim and multiple feature dims...
    if erp_etd.shape[1]==1:
        print("Warning: no time values? compressing to show features")
        erp_etd=erp_etd[:,0,...]
        if erp_etd.ndim<3: 
            erp_etd=erp_etd[...,np.newaxis] # add in ch-dim
            times = ch_names
            ch_names = '?'
        else:
            times = list(range(erp_etd.shape[1]))

    icoords = evtlabs
    jcoords = times
    kcoords = ch_names 
    if axis<0:
        axis = erp_etd.ndim+axis
    clim = [np.nanmedian(np.nanmin(erp_etd,axis=(1,2))), np.nanmedian(np.nanmax(erp_etd,axis=(1,2)))]
    if clim[0]==clim[1]: clim=[-1,1]
    if any(np.isnan(clim)) or any(np.isinf(clim)): clim=None
    import matplotlib.pyplot as plt

    # print("erp_etd={}".format(erp_etd.shape))
    # print(" {}, {}, {} ".format(icoords, jcoords, kcoords))
    
    # plot per channel
    ncols = int(np.ceil(np.sqrt(erp_etd.shape[axis])))
    nrows = int(np.ceil(erp_etd.shape[axis]/ncols))
    # make the bottom left axis to share its limits..
    axploti = ncols*(nrows-1)
    ax = plt.subplot(nrows,ncols,axploti+1)
    #fig, plts = plt.subplots(nrows, ncols, sharex='all', sharey='all', squeeze=False)
    for ci in range(erp_etd.shape[axis]):
        # make the axis
        if ci==axploti: # common axis plot
            pl = ax
        else: # normal plot
            pl = plt.subplot(nrows,ncols,ci+1,sharex=ax, sharey=ax) # share limits
            plt.tick_params(labelbottom=False,labelleft=False) # no labels

        # get the slice for the data to plot,  and it's coords
        if axis == 0:
            A = erp_etd[ci, :, :] # (time,ch)
            pltcoords = icoords
            xcoords = jcoords
            ycoords = kcoords
        elif axis == 1:
            A = erp_etd[:, ci, :] # (evtcoords,ch)
            xcoords = icoords
            pltcoords = jcoords
            ycoords = kcoords
        elif axis == 2:
            A = erp_etd[:, :, ci] # (evtlabs,time)
            xcoords = icoords
            ycoords = jcoords
            pltcoords = kcoords

        # make the plot of the desired type
        if plottype == 'plot':
            for li in range(A.shape[0]):
                pl.plot(ycoords, A[li, :], '.-', label=xcoords[li] if li<len(xcoords) else None, markersize=1, linewidth=1)
            if clim: 
                plt.ylim(clim)
            pl.grid(True)
        elif plottype == 'plott':
            for li in range(A.shape[1]):
                pl.plot(xcoords, A[:, li], '.-', label=ycoords[li] if li<len(ycoords) else None, markersize=1, linewidth=1)
            if clim: 
                plt.ylim(clim)
            pl.grid(True)
        elif plottype == 'imshow':
            # TODO[] : add the labels to the rows
            img=pl.imshow(A, aspect='auto')#,extent=[times[0],0,times[-1],erp_yetd.shape[-3]])
            pl.set_xticks(np.arange(len(ycoords)));pl.set_xticklabels(ycoords)
            pl.set_yticks(np.arange(len(xcoords)));pl.set_yticklabels(xcoords)
            if clim:
                img.set_clim(clim)
        pl.title.set_text("{}".format(pltcoords[ci] if ci<len(pltcoords) else ci))
    # legend only in the last plot
    pl.legend()
    if suptitle:
        plt.suptitle(suptitle)
    if show is not None:
        plt.show(block=show)

def get_ch_pos(ch_names):
    print("trying to get pos from cap file!")
    from mindaffectBCI.decoder.readCapInf import getPosInfo
    cnames, xy, xyz, iseeg =getPosInfo(ch_names)
    return xy, iseeg


def topoplot(A,ch_names=None, ch_pos=None, ax=None, levels=None, cRng=None, channel_labels:bool=True, cmap:str='bwr', colorbar:bool=True):
    import matplotlib.pyplot as plt
    if ch_pos is None and not ch_names is None:
        try:
            ch_pos, iseeg = get_ch_pos(ch_names)
            if sum(iseeg)>len(iseeg)*.7:
                print("Warning: couldnt get position for all channels.  Plotting subset.")
                A=A[...,iseeg] # guard against in-place changes
                ch_pos = ch_pos[iseeg,:]
                ch_names = [c for c,e in zip(ch_names,iseeg) if e]
        except:
            pass

    if cRng is None: cRng= np.max(np.abs(A.reshape((-1))))
    if not hasattr(cRng,'__iter__'): cRng = (-cRng,cRng)
    
    if ax is None: ax=plt.gca()
    if not ch_pos is None: # make as topoplot
        if levels is None: levels = np.linspace(cRng[0],cRng[-1],20)
        tt=ax.tricontourf(ch_pos[:,0],ch_pos[:,1],A[:ch_pos.shape[0]],levels=levels,cmap=cmap)
        # BODGE: deal with co-linear inputs by replacing each channel with a triangle of points
        #interp_pos = np.concatenate((ch_pos+ np.array((0,.1)), ch_pos + np.array((-.1,-.05)), ch_pos + np.array((+.1,-.05))),0)
        #tt=pA.tricontourf(interp_pos[:,0],interp_pos[:,1],np.tile(A[:ch_pos.shape[0]]*sgn,3),levels=levels,cmap='Spectral')
        if channel_labels:
            for i,n in enumerate(ch_names):
                #pA.plot(ch_pos[i,0],ch_pos[i,1],'.',markersize=5) # marker
                ax.text(ch_pos[i,0],ch_pos[i,1],n,ha='center',va='center') # label
        ax.set_aspect(aspect='equal')
        ax.set_frame_on(False) # no frame
        ax.tick_params(labelbottom=False,labelleft=False,which='both',bottom=False,left=False) # no labels, ticks
        if colorbar:
            plt.colorbar(tt)
    else:
        ax.plot(ch_names,A,'.-')
        ax.set_ylim(cRng)
        ax.grid(True)

def topoplots(A,ch_names=None, ch_pos=None, axs=None, nrows=None, ncols=None, levels=None, cRng=None, titles:list=None, channel_labels:bool=False, cmap:str='bwr',colorbar:bool=False):
    import matplotlib.pyplot as plt
    if ch_pos is None and not ch_names is None:
        ch_pos, iseeg = get_ch_pos(ch_names)
        if not np.all(iseeg) and sum(iseeg)>len(iseeg)*.7:
            print("Warning: couldnt get position for all channels.  Plotting subset.")
            A=A[...,iseeg] # guard against in-place changes
            ch_pos = ch_pos[iseeg,:]
            ch_names = [c for c,e in zip(ch_names,iseeg) if e]

    if cRng is None: cRng= np.max(np.abs(A))
    A_kd = A.reshape((-1,A.shape[-1]))

    # get the axes
    if axs is None:
        if nrows is None and ncols is None: 
            nrows = int(np.ceil(np.sqrt(A_kd.shape[0])))
        if nrows is None:
            nrows = int(np.ceil(A_kd.shape[0]/ncols))
        if ncols is None:
            ncols = int(np.ceil(A_kd.shape[0]/nrows))
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True)

    # do the plots
    for i, (A_d, ax) in enumerate(zip(A_kd,axs.ravel())):
        #ax.set_()
        topoplot(A_d, ch_names, ch_pos,ax=ax, cRng=cRng, channel_labels=channel_labels,colorbar=colorbar,cmap=cmap)
        if titles is not None:
            ax.title(titles[i])

    # turn the frames off from an unused plots
    for ax in axs.ravel()[A_kd.shape[0]:]:
        ax.set_frame_on(False)
        ax.tick_params(labelbottom=False,labelleft=False,which='both',bottom=False,left=False)

    # TODO[]: big colorbar at the side of the full set of plots
    return axs


def plot_spatial_components(A,ch_names=None,ncols=2,nrows=None,ch_pos=None,channel_labels=True,colorbar=True,normalize:bool=False):
    import matplotlib.pyplot as plt
    A=A.copy()
    if A.ndim > 2:
        if A.shape[0]>1 :
            print("Warning: multiple filters ploted at once")
        A = A.reshape((-1,A.shape[-1]))
    if A.ndim < 2:
        A = A[np.newaxis, :]

    if nrows==None: nrows=A.shape[0]

    if ch_pos is None and ch_names is not None:
        if not len(ch_names) == A.shape[-1]:
            print("Warning: channel names don't match dimension size!")
            if len(ch_names)>A.shape[-1]*.75:
                ch_names=ch_names[:A.shape[-1]]
            else:
                ch_names=None
    if not ch_names is None:
        # try to load position info from capfile
        try: 
            print("trying to get pos from cap file!")
            from mindaffectBCI.decoder.readCapInf import getPosInfo
            cnames, xy, xyz, iseeg =getPosInfo(ch_names)
            if all(iseeg):
                ch_pos = xy
            elif sum(iseeg)>len(iseeg)*.7:
                print("Warning: couldnt get position for all channels.  Plotting subset.")
                A=A[...,iseeg] # guard against in-place changes
                ch_pos = xy[iseeg,:]
                ch_names = [c for c,e in zip(ch_names,iseeg) if e]
        except:
            pass
    if ch_names is None:
        ch_names = np.arange(A.shape[-1])


    cRng= np.max(np.abs(A.reshape((-1))))
    levels = np.linspace(-cRng,cRng,20)

    # plot per component
    # start at the bottom to share the axis
    for ci in range(A.shape[0]):
        # make the axis
        subploti=(nrows-1-ci)*ncols
        if ci==0: # common axis plot
            axA = plt.subplot(nrows, ncols, subploti+1) # share limits
            axA.set_xlabel("Space")
            pA = axA
            channel_label = channel_labels
        else: # normal plot
            pA = plt.subplot(nrows, ncols, subploti+1, sharex=axA, sharey=axA) 
            plt.tick_params(labelbottom=False,labelleft=False) # no labels
            channel_label = False

        # make the spatial plot
        sign = 1 #np.sign(R[ci,...].flat[np.argmax(np.aabs(R[ci,...]))]) # normalize directions
        try:
            topoplot(A[ci,:]*sign,ch_names=ch_names,ch_pos=ch_pos,ax=pA,levels=levels,colorbar=colorbar, channel_labels=channel_label)
        except:
            pA.plot(ch_names,A[ci,:]*sign,'.-')
            pA.set_ylim((-cRng,cRng))
            pA.grid(True)

        #pA.title.set_text("Spatial {} #{}".format(spatial_filter_type,ci))

def plot_temporal_components(R,fs=1,ncols=2,nrows=None,normalize=False):
    import matplotlib.pyplot as plt
    R=R.copy()
    if R.ndim > 3:
        if R.shape[0]==1: R = R[0, ...] # remove uncessary dim
    if R.ndim < 3:
        R = R[np.newaxis, :]
    if nrows is None: nrows=R_keyt.shape[0]

    if times is None:
        times = np.arange(R.shape[-1])
        if offset is not None:
            times = times + offset
        if fs is not None:
            times = times / fs
        if offset_ms is not None:
            times = times + offset_ms/1000

    if evtlabs is None:
        evtlabs = np.arange(R.shape[-2])

    rRng =  np.max(np.abs(R.reshape((-1))))

    for ci in range(R.shape[0]):
        # make the axis
        subploti=(nrows-1-ci)*ncols
        if ci==0: # common axis plot
            axR = plt.subplot(nrows, ncols, subploti+2) # share limits
            axR.set_xlabel("time (s)")
            axR.grid(True)
            pR = axR
        else: # normal plot
            pR = plt.subplot(nrows, ncols, subploti+2, sharex=axR, sharey=axR) 
            pR.grid(True)
            pR.tick_params(axis='both',which='both',bottom=False, left=False, labelbottom=False,labelleft=False) # no labels

        sign = 1 #np.sign(R[ci,...].flat[np.argmax(np.abs(R[ci,...]))]) # normalize directions
        # make the temporal plot, with labels, N.B. use loop so can set each lines label

        for e in range(R.shape[-2]):
            Re = R[ci,e,:]*sign
            if norm_temporal: Re = Re / np.sqrt(np.sum(Re*Re))
            pR.plot(times,Re,'.-',label=evtlabs[e] if e<len(evtlabs) else e)
        pR.set_ylim((-rRng,rRng))
        # legend in the temporal plot
        axR.legend()

def plot_per_output_temporal_components(R_kyet,fs=1,ncols=2,nrows=None,normalize=False,evtlabs=None,outputs=None):
    import matplotlib.pyplot as plt
    R_kyet=R_kyet.copy()
    if R_kyet.ndim > 4:
        if R_kyet.shape[0]==1: R_kyet = R_kyet[0, ...] # remove uncessary dim
    if R_kyet.ndim < 4:
        R_kyet = R_kyet[np.newaxis, :]
    if nrows is None: nrows=R_kyet.shape[0]

    yscale = np.max(np.abs(R_kyet))
    xscale = 1
    times  = np.linspace(0,1,R_kyet.shape[-1])
    for ci in range(R_kyet.shape[0]):
        # make the axis
        subploti=(nrows-1-ci)*ncols
        if ci==0: # common axis plot
            axR = plt.subplot(nrows, ncols, subploti+2) # share limits
            axR.set_xlabel("Event")
            axR.set_ylabel('Output')
            axR.grid(True)
            pR = axR
            if outputs is not None:
                axR.set_yticks(np.arange(R_kyet.shape[1]))
                axR.set_yticklabels(outputs)
            if evtlabs is not None:
                axR.set_xticks(np.arange(R_kyet.shape[2]))
                axR.set_xticklabels(evtlabs)
        else: # normal plot
            pR = plt.subplot(nrows, ncols, subploti+2, sharex=axR, sharey=axR) 
            pR.grid(True)
            pR.tick_params(axis='both',which='both',bottom=False, left=False, labelbottom=False,labelleft=False) # no labels

        sign = 1 #np.sign(R[ci,...].flat[np.argmax(np.abs(R[ci,...]))]) # normalize directions
        # make the temporal plot, with labels, N.B. use loop so can set each lines label

        for yi in range(R_kyet.shape[1]):
            for ei in range(R_kyet.shape[2]):
                plt.plot(times+xscale*ei,R_kyet[ci,yi,ei,:]*.7/yscale+yi)



def plot_factoredmodel(A, R, S=None, 
                        evtlabs=None, times=None, ch_names=None, ch_pos=None, fs=None, 
                        offset_ms=None, offset=None, 
                        spatial_filter_type="Filter", temporal_filter_type="Filter", 
                        norm_temporal:bool=False, norm_spatial:bool=False, suptitle=None, ncol=2, 
                        channel_labels:bool=True, colorbar:bool=True, show:bool=None):
    '''
    Make a multi-plot of a factored model
    A_kd = k components and d sensors
    R_ket = k components, e events, tau samples response-duration
    '''
    import matplotlib.pyplot as plt

    print("A={} R={}".format(A.shape if A is not None else None, R.shape if R is not None else None ))
    ncols = ncol #int(np.ceil(np.sqrt(A.shape[0])))
    nrows = A.shape[0] if A is not None else R.shape[0] #int(np.ceil(A.shape[0]/ncols))

    # plot the spatial components
    if A is not None:
        plot_spatial_components(A,ch_names,ncols,channel_labels=channel_labels,colorbar=colorbar, normalize=norm_temporal)

    # plot temmporal components
    if R is not None:
        plot_temporal_components(R,fs,ncols, normalize=norm_temporal)
        #pR.title.set_text("Temporal {} #{}".format(temporal_filter_type,ci))

    if suptitle:
        plt.suptitle(suptitle)
    if show is not None:
        plt.show(block=show)



def plot_factoredmodel(A, R, S=None, 
                        evtlabs=None, times=None, ch_names=None, ch_pos=None, fs=None, 
                        offset_ms=None, offset=None, 
                        spatial_filter_type="Filter", temporal_filter_type="Filter", 
                        norm_temporal:bool=False, norm_spatial:bool=False, suptitle=None, ncol=2, 
                        channel_labels:bool=True, colorbar:bool=False, show:bool=None):
    '''
    Make a multi-plot of a factored model
    A_kd = k components and d sensors
    R_ket = k components, e events, tau samples response-duration
    '''
    import matplotlib.pyplot as plt

    print("A={} R={}".format(A.shape if A is not None else None, R.shape if R is not None else None ))
    ncols = ncol #int(np.ceil(np.sqrt(A.shape[0])))
    nrows = A.shape[0] if A is not None else R.shape[0] #int(np.ceil(A.shape[0]/ncols))

    if A is not None:
        A=A.copy()
        if A.ndim > 2:
            if A.shape[0]>1 :
                print("Warning: multiple filters ploted at once")
            A = A.reshape((-1,A.shape[-1]))
        if A.ndim < 2:
            A = A[np.newaxis, :]

        if ch_names is not None:
            if not len(ch_names) == A.shape[-1]:
                print("Warning: channel names don't match dimension size!")
                if len(ch_names)>A.shape[-1]*.75:
                    ch_names=ch_names[:A.shape[-1]]
                else:
                    ch_names=None
        if ch_pos is None and not ch_names is None:
            # try to load position info from capfile
            try: 
                print("trying to get pos from cap file!")
                from mindaffectBCI.decoder.readCapInf import getPosInfo
                cnames, xy, xyz, iseeg =getPosInfo(ch_names)
                if all(iseeg):
                    ch_pos = xy
                elif sum(iseeg)>len(iseeg)*.7:
                    print("Warning: couldnt get position for all channels.  Plotting subset.")
                    A=A[...,iseeg] # guard against in-place changes
                    ch_pos = xy[iseeg,:]
                    ch_names = [c for c,e in zip(ch_names,iseeg) if e]
            except:
                pass
        if ch_names is None:
            ch_names = np.arange(A.shape[-1])


        cRng= np.max(np.abs(A.reshape((-1))))
        levels = np.linspace(-cRng,cRng,20)

        # plot per component
        # start at the bottom to share the axis
        for ci in range(A.shape[0]):
            # make the axis
            subploti=(nrows-1-ci)*ncols
            if ci==0: # common axis plot
                axA = plt.subplot(nrows, ncols, subploti+1) # share limits
                axA.set_xlabel("Space")
                pA = axA
                channel_label = channel_labels
            else: # normal plot
                pA = plt.subplot(nrows, ncols, subploti+1, sharex=axA, sharey=axA) 
                plt.tick_params(labelbottom=False,labelleft=False) # no labels
                channel_label = False

            # make the spatial plot
            sign = 1 #np.sign(R[ci,...].flat[np.argmax(np.aabs(R[ci,...]))]) # normalize directions
            try:
                topoplot(A[ci,:]*sign,ch_names=ch_names,ch_pos=ch_pos,ax=pA,levels=levels,colorbar=colorbar, channel_labels=channel_label)
            except:
                pA.plot(ch_names,A[ci,:]*sign,'.-')
                pA.set_ylim((-cRng,cRng))
                pA.grid(True)

            #pA.title.set_text("Spatial {} #{}".format(spatial_filter_type,ci))

    # plot temmporal components
    if R is not None:
        R=R.copy()
        if R.ndim > 3:
            if R.shape[0]>1 :
                print("Warning: only the 1st set ERPs is plotted")
            R = R[0, ...]
        if R.ndim < 3:
            R = R[np.newaxis, :]

        if times is None:
            times = np.arange(R.shape[-1])
            if offset is not None:
                times = times + offset
            if fs is not None:
                times = times / fs
            if offset_ms is not None:
                times = times + offset_ms/1000

        if evtlabs is None:
            evtlabs = np.arange(R.shape[-2])

        rRng =  np.max(np.abs(R.reshape((-1))))

        for ci in range(R.shape[0]):
            # make the axis
            subploti=(nrows-1-ci)*ncols
            if ci==0: # common axis plot
                axR = plt.subplot(nrows, ncols, subploti+2) # share limits
                axR.set_xlabel("time (s)")
                axR.grid(True)
                pR = axR
            else: # normal plot
                pR = plt.subplot(nrows, ncols, subploti+2, sharex=axR, sharey=axR) 
                pR.grid(True)
                pR.tick_params(axis='both',which='both',bottom=False, left=False, labelbottom=False,labelleft=False) # no labels

            sign = 1 #np.sign(R[ci,...].flat[np.argmax(np.abs(R[ci,...]))]) # normalize directions
            # make the temporal plot, with labels, N.B. use loop so can set each lines label
            for e in range(R.shape[-2]):
                Re = R[ci,e,:]*sign
                if norm_temporal: Re = Re / np.sqrt(np.sum(Re*Re))
                pR.plot(times,Re,'.-',label=evtlabs[e] if e<len(evtlabs) else e)
            pR.set_ylim((-rRng,rRng))


            #pR.title.set_text("Temporal {} #{}".format(temporal_filter_type,ci))

        # legend in the temporal plot
        axR.legend()
    # rotate the ticks to be readable?
    if ch_pos is None:
        # TODO[]: get and only set for the tick locations!
        #axA.set_xticklabels(ch_names,rotation=65)
        pass
    if suptitle:
        plt.suptitle(suptitle)
    if show is not None:
        plt.show(block=show)





def plot_subspace(X_TSfd, Y_TSye, W_kd, R_ket, S_y, f_f, offset:int=0, fs:float=100, show:bool=False, label:str=None):
    """ plot the subspace source activity of a fwd-bwd model

    Args:
        X_TSfd ([type]): [description]
        Y_TSye ([type]): [description]
        W_kd ([type]): [description]
        R_ket ([type]): [description]
        S_y ([type]): [description]
        f_f ([type]): [description]
        offset (int, optional): [description]. Defaults to 0.
        fs (float, optional): [description]. Defaults to 100.
        show (bool, optional): [description]. Defaults to False.
        label (str, optional): [description]. Defaults to None.
    """
    if W_kd.ndim==3:
        W_kd=W_kd[0,...]
    if R_ket.ndim==4:
        R_ket=R_ket[0,...]

    if f_f is not None:
        wXf_TSk = np.einsum("kd,TSfd,f->TSk",W_kd,X_TSfd,f_f)
    else:
        wXf_TSk = np.einsum("kd,TSd->TSk",W_kd,X_TSfd)

    if S_y is not None:
        Y_TSye = np.einsum("TSye,yz->TSze", Y_TSye, S_y[:,np.newaxis])
    Y_TStye = window_axis(Y_TSye,axis=-3,winsz=R_ket.shape[-1])
    tmp = np.einsum("TStye,ket->TSyk",Y_TStye,R_ket[:,:,::-1]) # N.B. time-reverse R when applied to Y
    Yr_TSyk = np.zeros(Y_TSye.shape[:2]+tmp.shape[-2:],dtype=tmp.dtype)
    if offset==0:
        Yr_TSyk[:,R_ket.shape[-1]-1:,:]=tmp # pad and shift forward
    elif offset<0:
        Yr_TSyk[:,R_ket.shape[-1]-1+offset:offset,:]=tmp # pad and shift forward
    elif offset>0:
        Yr_TSyk[:,R_ket.shape[-1]-1+offset:,:] = tmp[:,:-offset,:]

    plot_trial(wXf_TSk, Yr_TSyk, fs=fs, show=show, ylabel="g(X) g(Y)", suptitle="sub_space {}".format(label))


def testSlicedvsContinuous():
    import numpy as np
    from utils import testSignal, sliceData, sliceY
    irf=(0,0,0,0,0,1,0,0,0,0)
    offset=0; # o->lag-by-5, -5=>no-lag, -9->lead-by-4
    X,Y,st,A,B = testSignal(nTrl=1,nSamp=500,d=1,nE=1,nY=1,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)
    #plt.subplot(311);plt.plot(X[0,:,0],label='X');plt.plot(Y[0,:,0,0],label='Y');plt.title("offset={}, irf={}".format(offset,irf));plt.legend()
    print("A={}\nB={}".format(A, B))
    print("X{}={}".format(X.shape, X[:30, np.argmax(np.abs(A))]))
    print("Y{}={}".format(Y.shape, Y[0, :30, 0]))
    Y_true = Y[..., 0:1, :]

    # slice then compute
    Xe=sliceData(X, stimTimes, tau=tau)
    Ye=sliceY(Y, stimTimes)
    Ye_true = Ye[..., 0:1, :]
    print('Xe{}={}\nYe{}={}\nst{}={}'.format(Xe.shape, Xe[:5, :, 0], Ye.shape, Ye[:5, 0, 0], stimTimes.shape, stimTimes[:5]))
    Cxx, Cxy, Cyy = updateSummaryStatistics(Xe, Ye_true, stimTimes, tau=Xe.shape[-2])
    print("Cxx_dd={} Cxy={} Cyy={}".format(Cxx_dd.shape, Cxy.shape, Cyy.shape))

    plot_summary_statistics(Cxx_dd, Cxy, Cyy)

    plotCxy(Cxy)
    
    dCxx = np.max((np.abs(Cxx/np.trace(Cxx)-Cxx1/np.trace(Cxx1))/(np.abs(Cxx/np.trace(Cxx))+1e-5)).ravel())
    dCxy = np.max((np.abs(Cxy-Cxy1)/(np.abs(Cxy)+1e-5)).ravel())
    dCyy = np.max((np.abs(Cyy-Cyy1)/(np.abs(Cyy)+1e-5)).ravel())
    print("dCxx= {} dCxy={} dCyy={}".format(dCxx, dCxy, dCyy))

    # timing tests
    from timeit import timeit
    dur = timeit(lambda: updateSummaryStatistics(Xe, Ye_true, stimTimes, tau=10), number=1000, globals=globals())
    print("Sliced={}s".format(dur/1000))

    def slicenuss(X, Y, stimTimes, tau):
        Xe = sliceData(X, stimTimes, tau)
        Ye = sliceY(Y, stimTimes)
        Cxx, Cxy, Cyy = updateSummaryStatistics(Xe, Ye, stimTimes, tau=tau)
    dur = timeit(lambda: slicenuss(X, Y_true, stimTimes, tau=10), number=1000, globals=globals())
    print("Slice+USS={}s".format(dur/1000))

    dur = timeit(lambda: updateSummaryStatistics(X, Y_true, None, tau=10), number=1000, globals=globals())
    print("Raw={}s".format(dur/1000))

Y=None
X=None
def testCyy2():
    updateCyy(None,Y,None,tau=18)

def testComputationMethods():
    global X, Y
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.utils import testSignal, sliceData, sliceY
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, autocov, cov, crossautocov, updateCyy, updateCxx, updateCxy
    irf=(1,0,0,0,0,0,0,0,0,0)
    offset=0; # o->lag-by-5, -5=>no-lag, -9->lead-by-4
    X,Y,st,A,B = testSignal(nTrl=30,nSamp=1000,d=1,nE=2,nY=1,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)
    #plt.subplot(311);plt.plot(X[0,:,0],label='X');plt.plot(Y[0,:,0,0],label='Y');plt.title("offset={}, irf={}".format(offset,irf));plt.legend()
    print("A={}\nB={}".format(A, B))
    print("X{}={}".format(X.shape, X[:30, np.argmax(np.abs(A))]))
    print("Y{}={}".format(Y.shape, Y[0, :30, 0]))
    Y_true = Y[..., 0:1, :]

    tau=30

    Cyy=updateCyy(None,Y,None,tau=tau) #(nY,yd,tau,yd,tau)
    Cyy2=updateCyy_old(None,Y,None,tau=tau) #(nY,yd,tau,yd,tau)
    print('Cyy-Cyy2={}'.format(np.max(np.abs(Cyy.ravel()-Cyy2.ravel()))))
    fig,axs=plt.subplots(nrows=3,ncols=1,sharex='all',sharey='all')
    im=axs[0].imshow(Cyy[0,...].reshape((Cyy.shape[-1]*Cyy.shape[-2],Cyy.shape[-1]*Cyy.shape[-2]))); plt.colorbar(im,ax=axs[0]); axs[0].set_title("Cyy")
    im=axs[1].imshow(Cyy2[0,...].reshape((Cyy.shape[-1]*Cyy.shape[-2],Cyy.shape[-1]*Cyy.shape[-2]))); plt.colorbar(im,ax=axs[1]);axs[1].set_title("Cyy2")
    im=axs[2].imshow((Cyy-Cyy2)[0,...].reshape((Cyy.shape[-1]*Cyy.shape[-2],Cyy.shape[-1]*Cyy.shape[-2]))); plt.colorbar(im,ax=axs[2]);axs[2].set_title("Cyy2-Cyy")
    plt.show()

    import cProfile, pstats
    cProfile.run("testCyy2()", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats("time").print_stats(10)

    import timeit
    t = timeit.timeit(lambda: updateCyy_old(None,Y,None,tau=tau),number=10)
    print("updateCyy_old: {}".format(t))
    t = timeit.timeit(lambda: updateCyy(None,Y,None,tau=tau),number=10)
    print("updateCyy: {}".format(t))

    quit()


    # other measures....
    Cxdtxdt = autocov(X, tau=tau) # (tau[0],dx,tau[1],dx)
    Cxdtydt = crossautocov(X, Y[...,0, :], tau=tau) # (tau[0],dx,tau[1],dy)
    Cydxdt  = crossautocov(Y[..., 0, :], X, tau=[1,tau]) # (tauy,dy,taux,dx)
    Cydtydt = np.stack([crossautocov(Y[:, :, yi:yi+1], Y[:, :, yi:yi+1], tau=tau) for yi in range(Y.shape[-1])], 0) # (nY,tau,d,tau,d)
    Cxxdd   = cov(X)

    # compare
    Cxx=updateCxx(None,X,None)
    Cxy=updateCxy(None,X,Y[:,:,0:1,np.newaxis],None,tau=tau)#(1,dy,tau,dx)
    Cyy=updateCyy(None,Y[:,:,:,np.newaxis],None,tau=tau) #(nY,yd,tau,yd,tau)

    Cyy2=updateCyy2(None,Y[:,:,:,np.newaxis],None,tau=tau) #(nY,yd,tau,yd,tau)
    print('Cyy-Cyy2={}'.format(np.max(np.abs(Cyy.ravel()-Cyy2.ravel()))))

    print('Cxx-Cxxdd={}'.format(np.max(np.abs(Cxxdd-Cxx).ravel())))
    print('Cxy-Cydxdt={}'.format(np.max(np.abs(Cxy-Cydxdt).ravel())))
    print('Cyy-Cydtydt={}'.format(np.max(np.abs(Cyy.ravel()-Cydtydt.ravel()))))

    # test centering
    cX = X - np.mean(X,(0,1))
    Cxx = updateCxx(None,X,None,center=True)
    cCxx = updateCxx(None,cX,None,center=False)
    print("Cxx(center) - cCxx={}".format(np.max(np.abs(Cxx-cCxx).ravel())))

    Cxy = updateCxy(None,X,Y,None,tau=3,center=True)
    cCxy= updateCxy(None,cX,Y,None,tau=3,center=False)
    print("Cxy(center) - cCxy={}".format(np.max(np.abs(Cxy-cCxy).ravel())))



    
def testCases():
    import numpy as np
    from utils import testSignal, sliceData, sliceY
    from updateSummaryStatistics import updateSummaryStatistics, plot_summary_statistics
    import matplotlib.pyplot as plt

    irf=(.5,0,0,0,0,0,0,0,0,1)
    offset=0; # X->lag-by-10
    X,Y,st,A,B = testSignal(nTrl=1,nSamp=500,d=1,nE=1,nY=1,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)
    print("A={}\nB={}".format(A, B))
    print("X{}={}".format(X.shape, X[:30, np.argmax(np.abs(A))]))
    print("Y{}={}".format(Y.shape, Y[0, :30, 0]))
    
    tau=10; uss_offset=0
    Y_true = Y[...,0:1,:]
    Cxx, Cxy, Cyy = updateSummaryStatistics(X, Y_true, None, tau=tau, offset=uss_offset, center=False, unitnorm=True)
    print("Cxx={} Cxy={} Cyy={}".format(Cxx.shape, Cxy.shape, Cyy.shape))
    plt.figure(1);plot_summary_statistics(Cxx,Cxy,Cyy); plt.show()

    N=Y_true.shape[0]*Y_true.shape[1]
    # compare with direct computation.
    Cxx0 = np.einsum('tsd,tse->de',X,X)/N
    print('Cxx={}'.format(np.max(np.abs(Cxx-Cxx0))))

    Ytau = window_axis(Y_true, winsz=tau, axis=-3)    # window
    Ytau = pad_dim(Ytau,axis=-4,pad=tau-1,prepend=True) # shift
    Ytau = Ytau[:,:,::-1,...] # time-reverse
    Cxy0 = np.einsum('TSd,TStye->yetd',X[...,:Ytau.shape[1],:],Ytau)/N # yetd
    #Cxy0 = Cxy0[...,::-1,:] # time-reverse to match updateCxy
    print('Cxy={}'.format(np.max(np.abs(Cxy-Cxy0))))
    
    Cyy0 = np.einsum('TStye,TSuyf->yetfu',Ytau,Ytau)/N
    print('Cyy={}'.format(np.max(np.abs(Cyy-Cyy0))))
    

    
    Cxys=[]
    offsets = list(range(-15,15))
    for i,offset in enumerate(offsets):
        Cxx, Cxy, Cyy = updateSummaryStatistics(X, Y, None, tau=20, offset=offset)
        Cxys.append(Cxy)
    Cxys = np.concatenate(Cxys,1)
    plotCxy(Cxys,offsets)
    plt.show()


    # leading X
    irf=(1,0,0,0,0,0,0,0,0,0)
    offset=-9; # X leads by 9
    X,Y,st,A,B = testSignal(nTrl=1,nSamp=5000,d=1,nE=1,nY=1,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)
    plt.figure(0);plt.clf();plt.plot(X[0,:,0],label='X');plt.plot(Y[0,:,0,0],label='Y');plt.title("offset={}, irf={}".format(offset,irf));plt.legend()

    # no-shift in analysis window
    tau=10; uss_offset=0
    Cxx, Cxy, Cyy = updateSummaryStatistics(X, Y, None, tau=tau, offset=uss_offset)
    print("Cxx={} Cxy={} Cyy={}".format(Cxx.shape, Cxy.shape, Cyy.shape))
    plt.figure(1);plt.clf();plot_summary_statistics(Cxx,Cxy,Cyy);plt.show()

    # shifted analysis window
    tau=10; uss_offset=-9
    Cxx, Cxy, Cyy = updateSummaryStatistics(X, Y, None, tau=tau, offset=uss_offset)
    print("Cxx={} Cxy={} Cyy={}".format(Cxx.shape, Cxy.shape, Cyy.shape))
    plt.figure(2);plt.clf();plot_summary_statistics(Cxx,Cxy,Cyy);plt.show()

    plt.figure(3);plt.clf();plot_erp(Cxy);plt.show()

    
if __name__=="__main__":

    topoplots(np.random.standard_normal((5,8)),ch_names=['FPz','C3','Cz','C4','CP3','CPz','CP4','Pz'])

    test_compCyx_diag()
    testComputationMethods()
    testCases()
