# Copyright (c) 2019 MindAffect B.V. 
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

import numpy as np
from mindaffectBCI.decoder.utils import window_axis, idOutliers, zero_outliers
#@function
def updateSummaryStatistics(X, Y, stimTimes=None, 
                            Cxx=None, Cxy=None, Cyy=None, 
                            badEpThresh=4, halflife_samp=1, cxxp=True, cyyp=True, tau=None,
                            offset=0, center=True, unitnorm=True, perY=True):
    '''
    Compute updated summary statistics (Cxx, Cxy, Cyy) for new data in X with event-info Y
      [Cxx, Cxy, Cyy, state]=updateSummaryStatistics(X, Y, stimTime_samp, Cxx, Cxy, Cyy, state, halflife_samp, badEpThresh)
    Inputs:
      X = (nTrl, nEp, tau, d) [d x tau x nEpoch x nTrl] raw response for the current stimulus event
               d=#electrodes tau=#response-samples  nEpoch=#stimulus events to process
         OR
          (nTrl, nSamp, d) [d x nSamp x nTrl ]
      Y = (nTrl, nEp, nY, nE) [nE x nY x nEpoch x nTrl] Indicator for which events occured for which outputs
         OR
           (nTrl, nSamp, nY, nE) [nE x nY x nSamp x nTrl]
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      stimTimes_samp = (nTrl, nEp) [nEpoch x nTrl] sample times for start each epoch.  Used to detect
               overlapping responses
      tau = int - the length of the impulse response in samples (X.shape[1] or Cxy.shape[1] if given)
      offset:int = relative shift of Y w.r.t. X
      Cxx = [d x d] current data covariance
      Cxy = (nY, nE, tau, d) [d x tau x nE x nY ] current per output ERPs
      Cyy = (nY, nE, tau, nE, tau) [tau x nE x tau x nE x nY ] current response covariance for each output
      badEpThresh = float, threshold for removing bad-data before fitting the summary statistics
      center:bool - do we center the X data before computing the summary statistics? (True)
      halflife_samp = float, forgetting factor for the updates
               Note: alpha = exp(log(.5)./(half-life)), half-life = log(.5)/log(alpha)

    
    Outputs:
      Cxx = [d x d] updated data covariance
      Cxy = (nY, nE, tau, d) [d x tau x nE x nY] updated per output ERPs
      Cyy = (nY, nE, tau, nE, tau) [tau x nE x tau x nE x nY] updated response covariance for each output
    Examples:
      # Supervised CCA
      Y = (nEpoch/nSamp, nY, nE) [nE x nY x nEpoch] indicator for each event-type and output of it's type in each epoch
      X = (nEpoch/nSamp, tau, d) [d x tau x nEpoch] sliced pre-processed raw data into per-stimulus event responses
      stimTimes = [nEpoch] sample numbers of the stimulus events
                # OR None if X, Y are samples
      [Cxx, Cxy, Cyy] = updateSummaryStatistics(X, Y, 3*np.arange(X.shape[0]));
      [J, w, r]=multipleCCA(Cxx, Cxy, Cyy)
    '''
    if X is None:
        return Cxx, Cxy, Cyy
    if tau is None:
        if Cxy is not None:
            tau = Cxy.shape[-2]
        elif X.shape[-2] < 100: # assume X is already sliced so we can infer the tau..
            print("Warning: guessing tau from shape of X")
            tau = X.shape[-2]
        else:
            raise ValueError("tau not set and Cxy is None!")
    tau = int(tau) # ensure tau is integer
    wght = 1

    X, Y = zero_outliers(X, Y, badEpThresh)
    
    # update the cross covariance XY
    Cxy = updateCxy(Cxy, X, Y, stimTimes, tau, wght, offset, center, unitnorm=unitnorm)
    # Update Cxx
    if (cxxp):
        Cxx = updateCxx(Cxx, X, stimTimes, None, wght, center, unitnorm=unitnorm)

    # ensure Cyy has the right size if not entry-per-model
    if (cyyp):
        # TODO [] : support overlapping info between update calls
        Cyy = updateCyy(Cyy, Y, stimTimes, tau, wght, unitnorm=unitnorm, perY=perY)
        
    return Cxx, Cxy, Cyy

#@function
def updateCxx(Cxx, X, stimTimes=None, tau=None, wght=1, center=False, unitnorm=True):
    '''
    Cxx (ndarray (d,d)): current data covariance
    X (ndarray):  (nTrl, nEp, tau, d) raw response for the current stimulus event
            d=#electrodes tau=#response-samples  nEpoch=#stimulus events to process
        OR
       (nTrl, nSamp, d) 
    stimTimes_samp (ndarray (nTrl, nEp)): sample times for start each epoch.  Used to detect
    wght (float): weight to accumulate this Cxx with the previous data, s.t. Cxx = Cxx_old*wght + Cxx_new.  Defaults to 1.
    center (bool): flag if center the data before computing covariance? Defaults to True.
    '''
    # ensure 3d
    if X.ndim == 2:
        X = X[np.newaxis, ...]

    # use X as is, computing purely spatial cov
    # N.B. reshape is expensive with window_axis
    if X.ndim == 3:
        XX = np.einsum("TEd,TEe->de", X, X)
    elif X.ndim == 4:
        XX = np.einsum("TEtd,TEte->de", X, X)

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
    X = (nTrl, nEp, tau, d) raw response for the current stimulus event
            d=#electrodes tau=#response-samples  nEpoch=#stimulus events to process
        OR
        (nTrl, nSamp, d) raw response at sample rate
    Y = (nTrl, nEp, nY, nE) Indicator for which events occured for which outputs
            nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
        OR 
        (nTrl, nSamp, nY, nE) event indicator at sample rate
    stimTimes_samp = (nTrl, nEp) sample times for start each epoch.  Used to detect
            overlapping responses
    Cxy = (nY, nE, tau, d) current per output ERPs
    '''
    if tau is None: # estimate the tau
        tau = Cxy.shape[-2]
    if Y.ndim == 3:
        Y = Y[np.newaxis, :, :, :] # add missing trial dim
    #if offset<-tau+1:
    #    raise NotImplementedError("Cant offset backwards by more than window size")
    #if offset>0:
    #    raise NotImplementedError("Cant offset by positive amounts")
    if verb > 1: print("tau={}".format(tau))
    if stimTimes is None:
        # X, Y are at sample rate, slice X every sample
        Xe = window_axis(X, winsz=tau, axis=-2) # (nTrl, nSamp, tau, d)
        # shrink Y w.r.t. the window and shift to align with the offset
        if offset <=0 : # shift Y forwards
            Ye = Y[:, -offset:Xe.shape[-3]-offset, :, :] # shift fowards and shrink
            if offset < -tau: # zero pad to size
                pad = np.zeros(Y.shape[:1]+(Xe.shape[-3]-Ye.shape[1],)+Y.shape[2:],dtype=Y.dtype)
                Ye = np.append(Ye,pad,1)        
        elif offset>0: # shift and pad
            Ye = Y[:,:Xe.shape[-3]-offset,:,:] # shrink
            pad = np.zeros(Y.shape[:1]+(Xe.shape[-3]-Ye.shape[1],)+Y.shape[2:],dtype=Y.dtype)
            Ye = np.append(pad,Ye,1) # pad to shift forwards

    else:
        Xe = X
        Ye = Y
    if Xe.ndim == 3:
        Xe = Xe[np.newaxis, :, :, :]

    if verb > 1: print("Xe={}\nYe={}".format(Xe.shape, Ye.shape))
    XY = np.einsum("TEye, TEtd->yetd", Ye, Xe)

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
def updateCyy(Cyy, Y, stimTime=None, tau=None, wght=1, zeropadded=True, unitnorm=True, perY=perY):
    '''
    Compute the Cyy tensors given new data
    Inputs:
      Cyy -- (nY, nE, tau, nE, tau) old Cyy info
      Y -- (nTrl, nEp/nSamp, nY, nE) the new stim info to be added
               Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      tau  -- number of samples in the stimulus response
      stimTime -- [] the times (in samples) of the epochs in Y
                  None : if Y is already at sample rate
      zeropadded - bool - flag variable length Y are padded with 0s
      wght -- weighting for the new vs. old data
      unitnorm(bool) : flag if we normalize the Cyy with number epochs
    Outputs:
      Cyy -- (nY, tau, nE, nE)
    '''
    if stimTime is None:
        MM = compCyy_diag(Y,tau,wght,zeropadded,unitnorm,perY=perY)
        MM = Cyy_diag2full(MM)
        Cyy = wght*Cyy + MM if Cyy is not None else MM
    else:
        Cyy = updateCyy_old(Cyy,Y,stimTime,tau,wght,zeropadded,unitnorm)
            # accumulate into the running total
    return Cyy

def Cyy_diag2full(MM):
    # BODGE: tau-diag Cyy entries to the 'correct' shape
    # (tau,nY,nE,nE) -> (nY,nE,tau,nE,tau)
    if MM.ndim == 3: # (tau,nE,nE) -> (tau,1,nE,nE)
        MM = np.reshape(MM,(MM.shape[0],1,MM.shape[1],MM.shape[2]))
    MM2 = np.zeros((MM.shape[-3],MM.shape[-2],MM.shape[-4],MM.shape[-2],MM.shape[-4]),dtype=MM.dtype) # (nY,nE,tau,nE,tau)
    # fill in the block diagonal entries
    for i in range(MM.shape[-4]):
        MM2[...,:,i,:,i] = MM[0,:,:,:]
        for j in range(i+1,MM.shape[-4]):
            MM2[...,:,i,:,j] = MM[j-i,:,:,:]
            MM2[...,:,j,:,i] = MM[j-i,:,:,:].swapaxes(-2,-1) # transpose the event types
    if MM.ndim==3: # ( 1,nE,tau,nE,tau) -> (nE,tau,nE,tau)
        MM2 = MM2[0,...]
    MM = MM2
    return MM2


def compCyy_diag(Y, tau:float, wght:float=1, zeropadded:bool=True, unitnorm:bool=True, perY:bool=True):
    '''
    Compute the main tau diagonal entries of a Cyy tensor
    Args:
      Y (nTrl, nEp/nSamp, nY, nE): the new stim info to be added
               Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      tau (int): number of samples in the stimulus response
      stimTime (list): the times (in samples) of the epochs in Y.  Default to None.
                  None : if Y is already at sample rate
      zeropadded (bool): flag variable length Y are padded with 0s
      wght (float): weighting for the new vs. old data. Defaults to 1.
      unitnorm(bool) : flag if we normalize the Cyy with number epochs. Defaults to True.
    Returns:
      Cyy -- (tau, nY, nE, nE)
    '''
    if Y.ndim == 3:  # ensure is 4-d
        Y = Y[np.newaxis, :, :, :] # (nTrl, nEp/nSamp, nY, nE)

    if not np.issubdtype(Y.dtype, np.floating): # all at once
        Y = Y.astype(np.float32)
    #print("Y={}".format(Y.shape))
    Ys = window_axis(Y, winsz=tau, axis=-3) # window of length tau (nTrl, nSamp, tau, nY, nE)

    # shrink Y w.r.t. the window and shift to align with the offset
    Ye = Y[:, :Ys.shape[-4], :, :] # shift fowards and shrink

    #print("Ys={}".format(Ys.shape))
    if perY:
        MM = np.einsum("TStye, TSyf->tyef", Ys, Ye) # compute cross-covariance (tau, nY, nE, nE)
    else: # cross output covariance
        MM = np.einsum("TStye, TSzf->tyezf", Ys, Ye) # compute cross-covariance (tau, nY, nE, nE)

    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        MM = MM / (Y.shape[0]*Y.shape[1]) # / nTrl*nEp
    return MM

def updateCyy_old(Cyy, Y, stimTime=None, tau=None, wght=1, zeropadded=True, unitnorm=True):
    '''
    Compute the Cyy tensors given new data
    Inputs:
      Cyy -- (nY, nE, tau, nE, tau) old Cyy info
      Y -- (nTrl, nEp/nSamp, nY, nE) the new stim info to be added
               Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      tau  -- number of samples in the stimulus response
      stimTime -- [] the times (in samples) of the epochs in Y
                  None : if Y is already at sample rate
      zeropadded - bool - flag variable length Y are padded with 0s
      wght -- weighting for the new vs. old data
      unitnorm(bool) : flag if we normalize the Cyy with number epochs
    Outputs:
      Cyy -- (nY, nE, tau, nE, tau)
    '''
    if tau is None: # estimate the tau
        tau=Cyy.shape[-1]
    if Cyy is None:
        Cyy = np.zeros((Y.shape[-2], Y.shape[-1], tau, Y.shape[-1], tau),dtype=Y.dtype)
    if Y.ndim == 3:  # ensure is 4-d
        Y = Y[np.newaxis, :, :, :] # (nTrl, nEp/nSamp, nY, nE) [nE x nY x nEpoch/nSamp x nTrl]

    if stimTime is None: # fast-path, already at sample rate
        if np.issubdtype(Y.dtype, np.floating): # all at once
            #print("Y={}".format(Y.shape))
            Ys = window_axis(Y, winsz=tau, axis=-3) # window of length tau (nTrl, nSamp, tau, nY, nE)
            #print("Ys={}".format(Ys.shape))
            MM = np.einsum("TStye, TSuyf->yetfu", Ys, Ys) # compute cross-covariance (nY, nE, tau, nE, tau)
        else: # trial at a time + convert to float
            MM = np.zeros(Cyy.shape)
            # different slice time for every trial, up-sample per-trial
            for trli in range(Y.shape[0]):  # loop over trials
                Yi = Y[trli, :, :, :]
                if not np.issubdtype(Y.dtype, np.floating):
                    Yi = np.array(Yi, np.float)
                Yi = window_axis(Yi, winsz=tau, axis=-3)
                MM = MM+np.einsum("Etye, Euyf->yetfu", Yi, Yi)

    else: # upsample and accumulate
        if stimTime.ndim == 1:  # ensure is 2d
            stimTime = stimTime[np.newaxis, :]

        MM = np.zeros(Cyy.shape)
        # different slice time for every trial, up-sample per-trial
        for trli in range(Y.shape[0]):  # loop over trials
            # up-sample to sample rate
            if trli < stimTime.shape[0]:
                nEp    = np.flatnonzero(stimTime[trli, :] != 0)[-1]
                sampIdx = np.array(stimTime[trli, :nEp] - stimTime[trli, 0], dtype=int)
            Yi = np.zeros((sampIdx[-1]+1+tau, )+Y.shape[-2:])
            Yi[sampIdx, :, :] = Y[trli, :nEp, :, :] # (nSamp, nY, nE)
            # compute the outer product for the covariance
            Yi = window_axis(Yi, winsz=tau, axis=-3) # (nEp/nSamp, tau, nY, nE)
            #print("Yi={}".format(Yi.shape))
            MM = MM + np.einsum("Etye, Euyf->yetfu", Yi, Yi) # (nY, nE, tau, nE, tau)
    
    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        MM = MM / (Y.shape[0]*Y.shape[1]) # / nTrl*nEp

    # accumulate into the running total
    Cyy = wght*Cyy + MM if Cyy is not None else MM
    return Cyy


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

def crossautocov(X, Y, tau, offset=0):
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
    if X.shape[0] < 100 and np.issubdtype(X.dtype, np.float) and np.issubdtype(Y.dtype, np.float): # all at once
        Xs = window_axis(X, winsz=tau[0], axis=-2) # window of length tau (nTrl, nSamp-tau, tau, d)
        Ys = window_axis(Y, winsz=tau[1], axis=-2) # window of length tau (nTrl, nSamp-tau, tau, d)
        
        # reshape back to same shape if diff window sizes
        if tau[0] < tau[1]:
            Xs = Xs[..., :Ys.shape[-3], :, :]
        else:
            Ys = Ys[..., -offset:Xs.shape[-3]-offset, :, :]

        # compute cross-covariance (tau, d, tau, d)
        Ctdtd = np.einsum("Tstd, Tsue -> tdue", Xs, Ys, optimize='optimal')
        
    else: # trial at a time + convert to float
        Ctdtd = np.zeros((tau[0], X.shape[-1], tau[1], Y.shape[-1]))
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
        
    return Ctdtd

def plot_summary_statistics(Cxx, Cxy, Cyy, evtlabs=None, times=None, ch_names=None, fs=None):
    """Visualize the summary statistics (Cxx, Cxy, Cyy) of a dataset

    It is assumed the data has 'd' channels, with 'nE' different types of
    trigger event, and a response length of 'tau' for each trigger.

    Args:
        Cxx (d,d): spatial covariance
        Cxy (nY, nE, tau, d): per-output event related potentials (ERPs)
        Cyy (nY, nE, tau, nE, tau): updated response covariance for each output
        evtlabs ([type], optional): the labels for the event types. Defaults to None.
        times ([type], optional): values for the time-points along tau. Defaults to None.
        ch_names ([type], optional): textual names for the channels. Defaults to None.
        fs ([type], optional): sampling rate for the data along tau (used to make times if not given). Defaults to None.
    """    
    import matplotlib.pyplot as plt
    if times is None:
        times = np.arange(Cxy.shape[-2])
        if fs is not None:
            times = times / fs
    if ch_names is None:
        ch_names = ["{}".format(i) for i in range(Cxy.shape[-1])]
    if evtlabs is None:
        evtlabs = ["{}".format(i) for i in range(Cxy.shape[-3])]

    plt.clf()
    # Cxx
    plt.subplot(311)
    plt.imshow(Cxx, origin='lower', extent=[0, Cxx.shape[0], 0, Cxx.shape[1]])
    plt.colorbar()
    # TODO []: use the ch_names to add lables to the  axes
    plt.title('Cxx')

    # Cxy
    if Cxy.ndim > 3:
        if Cxy.shape[0] > 1:
            print("Warning: only the 1st set ERPs is plotted")
        Cxy = Cxy[0, ...]
    nevt = Cxy.shape[-3]
    for ei in range(nevt):
        if ei==0:
            ax = plt.subplot(3,nevt,nevt+ei+1) # N.B. subplot indexs from 1!
            plt.xlabel('time (s)')
            plt.ylabel('space')
        else: # no axis on other sub-plots
            plt.subplot(3, nevt, nevt+ei+1, sharey=ax, sharex=ax)
            plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(Cxy[ei, :, :].T, aspect='auto', origin='lower', extent=(times[0], times[-1], 0, Cxy.shape[-1]))
        # TODO []: use the ch_names to add lables to the  axes
        plt.title('{}'.format(evtlabs[min(len(evtlabs),ei)]))
    # only last one has colorbar
    plt.colorbar()

    # Cyy
    if Cyy.ndim > 4:
        if Cyy.shape[0] > 1:
            print("Warning: only the 1st set ERPs is plotted")
        Cyy = Cyy[0, ...]
    Cyy2d = np.reshape(Cyy, (Cyy.shape[0]*Cyy.shape[1], Cyy.shape[2]*Cyy.shape[3]))
    plt.subplot(313)
    plt.imshow(Cyy2d, origin='lower', extent=[0, Cyy2d.shape[0], 0, Cyy2d.shape[1]])
    plt.colorbar()
    plt.title('Cyy')

def plotCxy(Cxy,evtlabs=None,fs=None):
    import matplotlib.pyplot as plt
    times = np.arange(Cxy.shape[-2])
    if fs is not None: times = times/fs
    ncols = Cxy.shape[1] if Cxy.shape[1]<10 else Cxy.shape[1]//10
    nrows = Cxy.shape[1]//ncols + (1 if Cxy.shape[1]%ncols>0 else 0)
    clim = (np.min(Cxy), np.max(Cxy))
    fit,ax = plt.subplots(nrows=nrows,ncols=ncols,sharey='row',sharex='col')
    for ei in range(Cxy.shape[1]):
        plt.sca(ax.flat[ei])
        plt.imshow(Cxy[0,ei,:,:].T,aspect='auto',extent=[times[0],times[-1],0,Cxy.shape[-1]])
        plt.clim(clim)
        if evtlabs is not None: plt.title(evtlabs[ei])

def plot_erp(erp, evtlabs=None, times=None, fs=None, ch_names=None, axis=-1, plottype='plot', offset=0, ylim=None):
    '''
    Make a multi-plot of the event ERPs (as stored in erp)
    erp = (nE, tau, d) current per output ERPs
    '''
    if erp.ndim > 3:
        if erp.shape[0]>1 :
            print("Warning: only the 1st set ERPs is plotted")
        erp = erp[0, ...]
    elif erp.ndim == 2:
        erp = erp[np.newaxis,...]
    icoords = evtlabs if not evtlabs is None else list(range(erp.shape[-3]))
    if times is None:
        times = list(range(offset,erp.shape[-2]+offset))
        if fs is not None:
            times = [ t/fs for t in times ]
    jcoords = times if not times is None else list(range(offset,erp.shape[-2]+offset))
    kcoords = ch_names if not ch_names is None else list(range(erp.shape[-1]))
    if axis<0:
        axis = erp.ndim+axis
    import matplotlib.pyplot as plt

    print("erp={}".format(erp.shape))
    
    # plot per channel
    ncols = int(np.ceil(np.sqrt(erp.shape[axis])))
    nrows = int(np.ceil(erp.shape[axis]/ncols))
    # make the bottom left axis to share its limits..
    axploti = ncols*(nrows-1)
    ax = plt.subplot(nrows,ncols,axploti+1)
    #fig, plts = plt.subplots(nrows, ncols, sharex='all', sharey='all', squeeze=False)
    for ci in range(erp.shape[axis]):
        # make the axis
        if ci==axploti: # common axis plot
            pl = ax
        else: # normal plot
            pl = plt.subplot(nrows,ncols,ci+1,sharex=ax, sharey=ax) # share limits
            plt.tick_params(labelbottom=False,labelleft=False) # no labels

        # get the slice for the data to plot,  and it's coords
        if axis == 0:
            A = erp[ci, :, :] # (time,ch)
            pltcoords = icoords
            xcoords = jcoords
            ycoords = kcoords
        elif axis == 1:
            A = erp[:, ci, :] # (evtcoords,ch)
            xcoords = icoords
            pltcoords = jcoords
            ycoords = kcoords            
        elif axis == 2:
            A = erp[:, :, ci] # (evtlabs,time)
            xcoords = icoords
            ycoords = jcoords
            pltcoords = kcoords

        # make the plot of the desired type
        if plottype == 'plot':
            for li in range(A.shape[0]):
                pl.plot(ycoords, A[li, :], '.-', label=xcoords[li], markersize=1, linewidth=1)
                if ylim: plt.ylim(ylim)
        elif plottype == 'plott':
            for li in range(A.shape[1]):
                pl.plot(xcoords, A[:, li], '.-', label=ycoords[li], markersize=1, linewidth=1)
                if ylim: plt.ylim(ylim)
        elif plottype == 'imshow':
            # TODO[] : add the labels to the rows
            pl.imshow(A, aspect='auto')#,extent=[times[0],0,times[-1],erp.shape[-3]])
            
        pl.title.set_text("{}".format(pltcoords[ci]))
    # legend only in the last plot
    pl.legend()


def plot_factoredmodel(A, R, evtlabs=None, times=None, ch_names=None, ch_pos=None, fs=None, offset_ms=None, offset=None, spatial_filter_type="Filter", ncol=2):
    '''
    Make a multi-plot of a factored model
    A = (k,d)
    R = (k,e,tau)
    '''
    A=A.copy()
    R=R.copy()
    if A.ndim > 2:
        if A.shape[0]>1 :
            print("Warning: only the 1st set ERPs is plotted")
        A = A[0, ...]
    if A.ndim < 2:
        A = A[np.newaxis, :]
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
    if ch_names is None:
        ch_names = np.arange(A.shape[-1])
    elif ch_pos is None and len(ch_names) > 0:
        # try to load position info from capfile
        try: 
            print("trying to get pos from cap file!")
            from mindaffectBCI.decoder.readCapInf import getPosInfo
            cnames, xy, xyz, iseeg =getPosInfo(ch_names)
            if all(iseeg):
                ch_pos = xy
        except:
            pass
    if evtlabs is None:
        evtlabs = np.arange(R.shape[-2])
        
    import matplotlib.pyplot as plt

    print("A={} R={}".format(A.shape, R.shape))
    
    # plot per component
    ncols = ncol #int(np.ceil(np.sqrt(A.shape[0])))
    nrows = A.shape[0] #int(np.ceil(A.shape[0]/ncols))
    # start at the bottom to share the axis
    for ci in range(A.shape[0]):
        # make the axis
        subploti=(nrows-1-ci)*ncols
        if ci==0: # common axis plot
            axR = plt.subplot(nrows, ncols, subploti+2) # share limits
            axA = plt.subplot(nrows, ncols, subploti+1) # share limits
            axA.set_xlabel("Space")
            axR.set_xlabel("time (s)")
            axR.grid(True)
            pA = axA
            pR = axR
        else: # normal plot
            pR = plt.subplot(nrows, ncols, subploti+2, sharex=axR, sharey=axR) 
            pA = plt.subplot(nrows, ncols, subploti+1, sharex=axA, sharey=axA) 
            plt.tick_params(labelbottom=False,labelleft=False) # no labels
            pR.grid(True)

        # make the spatial plot
        sign = np.sign(A[ci,np.argmax(np.abs(A[ci,:]))]) # normalize directions
        if not ch_pos is None: # make as topoplot
            cRng= np.max(np.abs(A.reshape((-1))))
            levels = np.linspace(-cRng,cRng,20)
            tt=pA.tricontourf(ch_pos[:,0],ch_pos[:,1],A[ci,:]*sign,levels=levels,cmap='Spectral')
            # BODGE: deal with co-linear inputs by replacing each channel with a triangle of points
            #interp_pos = np.concatenate((ch_pos+ np.array((0,.1)), ch_pos + np.array((-.1,-.05)), ch_pos + np.array((+.1,-.05))),0)
            #tt=pA.tricontourf(interp_pos[:,0],interp_pos[:,1],np.tile(W[ri,:]*sgn,3),levels=levels,cmap='Spectral')
            for i,n in enumerate(ch_names):
                #pA.plot(ch_pos[i,0],ch_pos[i,1],'.',markersize=5) # marker
                pA.text(ch_pos[i,0],ch_pos[i,1],n,ha='center',va='center') # label
            pA.set_aspect(aspect='equal')
            pA.set_frame_on(False) # no frame
            plt.tick_params(labelbottom=False,labelleft=False,which='both',bottom=False,left=False) # no labels, ticks
            plt.colorbar(tt)
        else:
            pA.plot(ch_names,A[ci,:]*sign,'.-')
            pA.grid(True)

        pA.title.set_text("Spatial {} #{}".format(spatial_filter_type,ci))
        # make the temporal plot, with labels, N.B. use loop so can set each lines label
        for e in range(R.shape[-2]):
            pR.plot(times,R[ci,e,:]*sign,'.-',label=evtlabs[e])
        pR.title.set_text("Impulse Response #{}".format(ci))

    # legend in the temporal plot
    axR.legend()
    # rotate the ticks to be readable?
    if ch_pos is None:
        # TODO[]: get and only set for the tick locations!
        #axA.set_xticklabels(ch_names,rotation=65)
        pass


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
    print("Cxx={} Cxy={} Cyy={}".format(Cxx.shape, Cxy.shape, Cyy.shape))

    plot_summary_statistics(Cxx, Cxy, Cyy)

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
    testComputationMethods()
    testCases()
