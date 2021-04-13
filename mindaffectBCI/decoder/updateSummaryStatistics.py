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
                            offset=0, center=True, unitnorm=True, zeropadded:bool=True, perY=True):
    '''
    Compute updated summary statistics (Cxx_dd, Cxy_yetd, Cyy_yetet) for new data in X with event-info Y

    Args:
      X_TSd (nTrl, nEp, tau, d): raw response for the current stimulus event
               d=#electrodes tau=#response-samples  nEpoch=#stimulus events to process
         OR
          (nTrl, nSamp, d)
      Y_TSye = (nTrl, nEp, nY, nE): Indicator for which events occured for which outputs
         OR
           (nTrl, nSamp, nY, nE)
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
    Cxy = updateCxy(Cxy, X, Y, stimTimes, tau, wght, offset=offset, center=center, unitnorm=unitnorm)
    # Update Cxx
    if (cxxp):
        Cxx = updateCxx(Cxx, X, stimTimes, None, wght, offset=offset, center=center, unitnorm=unitnorm)

    # ensure Cyy has the right size if not entry-per-model
    if (cyyp):
        # TODO [] : support overlapping info between update calls
        Cyy = updateCyy(Cyy, Y, stimTimes, tau, wght, offset=offset, unitnorm=unitnorm, perY=perY, zeropadded=zeropadded)
        
    return Cxx, Cxy, Cyy

#@function
def updateCxx(Cxx, X, stimTimes=None, tau:int=None, wght:float=1, offset:int=0, center:bool=False, unitnorm:bool=True):
    '''
    Args:
        Cxx_dd (ndarray (d,d)): current data covariance
        X_TSd (nTrl, nSamp, d): raw response at sample rate
        stimTimes_samp (ndarray (nTrl, nEp)): sample times for start each epoch.  Used to detect
        wght (float): weight to accumulate this Cxx with the previous data, s.t. Cxx = Cxx_old*wght + Cxx_new.  Defaults to 1.
        center (bool): flag if center the data before computing covariance? Defaults to True.
    Returns:
        Cxx_dd (ndarray (d,d)): current data covariance
    '''
    # ensure 3d
    if X.ndim == 2:
        X = X[np.newaxis, ...]

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
                pad = np.zeros(Y.shape[:1]+(Xe.shape[-3]-Ye.shape[1],)+Y.shape[2:], dtype=Y.dtype)
                Ye = np.append(Ye,pad,1)        
        elif offset>0: # shift and pad
            Ye = Y[:, :Xe.shape[-3]-offset, :, :] # shrink
            pad = np.zeros(Y.shape[:1]+(Xe.shape[-3]-Ye.shape[1],)+Y.shape[2:], dtype=Y.dtype)
            Ye = np.append(pad,Ye,1) # pad to shift forwards

    else:
        Xe = X
        Ye = Y
    if Xe.ndim == 3:
        Xe = Xe[np.newaxis, :, :, :]

    if verb > 1: print("Xe={}\nYe={}".format(Xe.shape, Ye.shape))

    # LOOPY version as einsum doens't manage the memory well...
    XY = np.zeros( (Ye.shape[-2],Ye.shape[-1],Xe.shape[-2],Xe.shape[-1]), dtype=Xe.dtype)
    for tau in range(Xe.shape[-2]):
        XY[:,:,tau,:] = np.einsum("TSye, TSd->yed", Ye, Xe[:,:,tau,:], casting='unsafe', dtype=Xe.dtype)

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
      Cyy_yetet (nY, tau, nE, nE):
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
    if Cyy_tyee.ndim == 3: # (tau,nE,nE) -> (tau,1,nE,nE)
        Cyy_tyee = np.reshape(Cyy_tyee,(Cyy_tyee.shape[0],1,Cyy_tyee.shape[1],Cyy_tyee.shape[2]))
    Cyy_yetet = np.zeros((Cyy_tyee.shape[-3],Cyy_tyee.shape[-2],Cyy_tyee.shape[-4],Cyy_tyee.shape[-2],Cyy_tyee.shape[-4]),dtype=Cyy_tyee.dtype) # (nY,nE,tau,nE,tau)
    # fill in the block diagonal entries
    for i in range(Cyy_tyee.shape[-4]):
        Cyy_yetet[...,:,i,:,i] = Cyy_tyee[0,:,:,:]
        for j in range(i+1,Cyy_tyee.shape[-4]):
            Cyy_yetet[...,:,i,:,j] = Cyy_tyee[j-i,:,:,:]
            Cyy_yetet[...,:,j,:,i] = Cyy_tyee[j-i,:,:,:].swapaxes(-2,-1) # transpose the event types
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

    # temporal embedding
    X_TStd = window_axis(X_TSd, winsz=tau, axis=-2)
    # shrink X w.r.t. the window and shift to align with the offset
    #X_TSd = X_TSd[:, tau-1:, ...] # shift fowards and shrink
    X_TSd = X_TSd[:, :X_TStd.shape[-3], ...] # shift fowards and shrink
    # compute the cross-covariance
    MM = np.einsum("TStd, TSe->tde", X_TStd, X_TSd, dtype=X_TSd.dtype)

    if center:
        N = X_TSd.shape[0]*X_TSd.shape[1]
        sX = np.sum(X_TSd.reshape((N,X.shape[-1])),axis=0)
        muXX = np.einsum("i,j->ij",sX,sX) / N
        MM = MM - muXX

    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        MM = MM / (X_TSd.shape[0]*X_TSd.shape[1]) # / nTrl*nEp
    return MM

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
            nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
    Returns:
        Cyx_tyed (nY, nE, tau, d): cross covariance at different offsets
        taus : relative offsets (t_y - t_x)
    '''
    if X_TSd.ndim == 2: 
        X_TSd = X_TSd[np.newaxis, ...] 
    if Y_TSye.ndim == 3:
        Y_TSye = Y_TSye[np.newaxis, ...] 
    if verb > 1: print("tau={}".format(tau))
    if not hasattr(tau,'__iter__'): tau=(0,tau)
    if offset is None : offset = 0
    if not hasattr(offset,'__iter__'): offset=(0,offset)

    # work out a single shift-length for shifting X
    taus = list(range(-(offset[0] + tau[0]-1 - offset[1]), (offset[1] + tau[1]-1 - offset[0])+1))

    MM = np.zeros((len(taus),Y_TSye.shape[-2],Y_TSye.shape[-1],X_TSd.shape[-1]),dtype=X_TSd.dtype)
    for i,t in enumerate(taus):
        if t < 0 : # shift Y backwards -> X preceeds Y
            MM[i,...] = np.einsum('TSye,TSd->yed',Y_TSye[:,-t:,:,:],X_TSd[:,:t,:])

        elif t == 0 : # aligned
            MM[i,...] = np.einsum('TSye,TSd->yed',Y_TSye,X_TSd)

        elif t > 0: # shift X backwards -> Y preceeds X
            MM[i,...] = np.einsum('TSye,TSd->yed',Y_TSye[:,:-t,:,:],X_TSd[:,t:,:])

    if center:
        if verb > 1: print("center")
        # N.B. (X-mu*1.T)@Y.T = X@Y.T - (mu*1.T)@Y.T = X@Y.T - mu*1.T@Y.T = X@Y.T - mu*\sum(Y)  
        muX = np.mean(X_TSd,axis=(0,1)) 
        muY = np.sum(Y_TSye,(0,1))
        muXY_yed= np.einsum("ye,d->yed",muY,muX) 
        MM = MM - muXY_yed[np.newaxis,...] #(nY,nE,tau,d)

    if unitnorm:
        # normalize so the resulting constraint on the estimated signal is that it have
        # average unit norm
        MM = MM / (Y_TSye.shape[0]*Y_TSye.shape[1]) # / nTrl*nEp
    
    return MM, taus

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

def plot_summary_statistics(Cxx_dd, Cyx_yetd, Cyy_yetet, 
                            evtlabs=None, outputs=None, times=None, ch_names=None, fs=None, label:str=None):
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
        times = np.arange(Cyx_yetd.shape[-2])
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
        if nevt>1 and nout>1:
            title = '{}:{}'.format(outputs[ei//nevt],evtlabs[ei%nevt])
        elif nevt==1:
            title = '{}'.format(outputs[ei] if outputs else None)
        elif nout==1:
            title = '{}'.format(evtlabs[ei] if evtlabs else None)
        else:
            title=[]
        plt.title(title)
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
                ylabel:str='ch + output', suptitle:str=None, block:bool=False):
    """visualize a single trial with data and stimulus sequences

    Args:
        X_TSd ([type]): raw data sequence
        Y_TSy ([type]): raw stimulus sequence
        fs (float, optional): sample rate of data and stimulus. Defaults to None.
        ch_names (list-of-str, optional): channel names for X. Defaults to None.
    """    
    import matplotlib.pyplot as plt
    if X_TSd.ndim==2 : X_TSd=X_TSd[np.newaxis,...]
    if not Y_TSy is None and Y_TSy.ndim==2 : Y_TSy=Y_TSy[np.newaxis,...]
    times = np.arange(X_TSd.shape[1])/fs if fs is not None else np.arange(X_TSd.shape[1])
    if ch_names is None: ch_names = np.arange(X_TSd.shape[-1],dtype=int)
    if outputs is None and not Y_TSy is None:  outputs  = np.arange(Y_TSy.shape[-1 if Y_TSy.ndim==3 else -2])
    if evtlabs is None and not Y_TSy is None and Y_TSy.ndim==4: evtlabs  = np.arange(Y_TSy.shape[-1])
    # strip unused outputs to simplify plot
    if not Y_TSy is None :
        if Y_TSy.ndim < 4:
            Y_TSy=Y_TSy[...,np.any(Y_TSy,axis=tuple(range(Y_TSy.ndim-1)))]
        else:
            Y_TSy=Y_TSy[...,np.any(Y_TSy,axis=tuple(range(Y_TSy.ndim-2))+(-1,)),:]

    print(Y_TSy.shape)
    linespace = 2 #np.mean(np.abs(X_TSd[X_TSd!=0]))*2
    for i in range(X_TSd.shape[0]):
        plt.subplot(X_TSd.shape[0],1,i+1)
        trlen = np.flatnonzero(np.any(X_TSd[i,...],-1))[-1]
        for c in range(X_TSd.shape[-1]):
            tmp = X_TSd[i,...,c]
            tmp = tmp[...,:trlen]
            tmp = (tmp - np.mean(tmp.ravel())) / max(1,np.std(tmp[tmp!=0].ravel())) / 2
            lab = ch_names[c] if c < len(ch_names) else c
            plt.plot(times[:trlen],tmp+c*linespace,label='X {}'.format(lab))

        if Y_TSy is not None:
            if Y_TSy.ndim==3:
                for y in range(Y_TSy.shape[-1]):
                    tmp = Y_TSy[i,...,y] / np.max(Y_TSy)
                    plt.plot(times,tmp+y+X_TSd.shape[-1]*linespace+2,'.-',label='Y {}'.format(y))
            elif Y_TSy.ndim==4:
                for y in range(Y_TSy.shape[-2]):
                    out = outputs[y] if y < len(outputs) else y
                    for e in range(Y_TSy.shape[-1]):
                        evt = evtlabs[e] if e < len(evtlabs) else e
                        tmp = Y_TSy[i,...,y,e] / np.max(Y_TSy)
                        plt.plot(times,tmp+y+X_TSd.shape[-1]*linespace+2+e,'.-',label='Y {}:{}'.format(out,evt))

        plt.title('Trl {}'.format(i))
        plt.grid(True)
        plt.xlabel('time (s)' if fs is not None else 'time (samp)')
        plt.ylabel(ylabel)
    plt.legend()
    if suptitle:
        plt.suptitle(suptitle)
    plt.show(block=block)


def plot_erp(erp, evtlabs=None, outputs=None, times=None, fs=None, ch_names=None, 
             axis=-1, plottype='plot', offset=0, ylim=None, suptitle:str=None, block:bool=False):
    '''
    Make a multi-plot of the event ERPs (as stored in erp)
    erp = (nY, nE, tau, d) current per output ERPs
    '''
    nevt = erp.shape[-3]
    nout = erp.shape[-4] if erp.ndim>3 else 1
    if outputs is None:
        outputs = ["{}".format(i) for i in range(nout)]
    if evtlabs is None:
        evtlabs = ["{}".format(i) for i in range(nevt)]
    if times is None:
        times = list(range(offset,erp.shape[-2]+offset))
        if fs is not None:
            times = [ t/fs for t in times ]
    if ch_names is None:
        ch_names = ["{}".format(i) for i in range(erp.shape[-1])]

    if erp.ndim > 3:
        if erp.shape[0]>1 :
            print("Multiple Y's merged!")
        erp = erp.reshape((-1,erp.shape[-2],erp.shape[-1]))
        # update evtlabs with the outputs info
        if nevt>1 and nout>1:
            evtlabs = [ "{}:{}".format(o,e) for o in outputs for e in evtlabs]
        elif nevt==1 and nout>1:
            evtlabs = outputs
    elif erp.ndim == 2:
        erp = erp[np.newaxis,...]

    icoords = evtlabs
    jcoords = times
    kcoords = ch_names 
    if axis<0:
        axis = erp.ndim+axis
    clim = [np.min(erp.flat), np.max(erp.flat)]
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
            pl.grid(True)
        elif plottype == 'plott':
            for li in range(A.shape[1]):
                pl.plot(xcoords, A[:, li], '.-', label=ycoords[li], markersize=1, linewidth=1)
            if ylim: plt.ylim(ylim)
            pl.grid(True)
        elif plottype == 'imshow':
            # TODO[] : add the labels to the rows
            img=pl.imshow(A, aspect='auto')#,extent=[times[0],0,times[-1],erp.shape[-3]])
            pl.set_xticks(np.arange(len(ycoords)));pl.set_xticklabels(ycoords)
            pl.set_yticks(np.arange(len(xcoords)));pl.set_yticklabels(xcoords)
            img.set_clim(clim)
        pl.title.set_text("{}".format(pltcoords[ci] if ci<len(pltcoords) else ci))
    # legend only in the last plot
    pl.legend()
    if suptitle:
        plt.suptitle(suptitle)
    if block is not None:
        plt.show(block=block)


def plot_factoredmodel(A, R, S=None, 
                        evtlabs=None, times=None, ch_names=None, ch_pos=None, fs=None, 
                        offset_ms=None, offset=None, spatial_filter_type="Filter", label=None, ncol=2):
    '''
    Make a multi-plot of a factored model
    A_kd = k components and d sensors
    R_ket = k components, e events, tau samples response-duration
    '''
    A=A.copy()
    R=R.copy()
    if A.ndim > 2:
        if A.shape[0]>1 :
            print("Warning: multiple filters ploted at once")
        A = A.reshape((-1,A.shape[-1]))
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
    if ch_names is not None:
        if not len(ch_names) == A.shape[-1]:
            print("Warning: channel names don't match dimension size!")
            if len(ch_names)>A.shape[-1]:
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
        except:
            pass
    if ch_names is None:
        ch_names = np.arange(A.shape[-1])
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
            axA = plt.subplot(nrows, ncols, subploti+1) # share limits
            axA.set_xlabel("Space")
            pA = axA
        else: # normal plot
            pA = plt.subplot(nrows, ncols, subploti+1, sharex=axA, sharey=axA) 
            plt.tick_params(labelbottom=False,labelleft=False) # no labels

        # make the spatial plot
        sign = 1 #np.sign(R[ci,...].flat[np.argmax(np.abs(R[ci,...]))]) # normalize directions
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

    # plot temmporal components
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

        sign = 1 #np.sign(R[ci,...].flat[np.argmax(np.abs(R[ci,...]))]) # normalize directions
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
    if label:
        plt.suptitle(label)


def plot_subspace(X_TSfd, Y_TSye, W_kd, R_ket, S_y, f_f, offset:int=0, fs:float=100, block:bool=False, label:str=None):
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
        block (bool, optional): [description]. Defaults to False.
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

    plot_trial(wXf_TSk, Yr_TSyk, fs=fs, block=block, ylabel="g(X) g(Y)", suptitle="sub_space {}".format(label))


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
    test_compCyx_diag()
    testComputationMethods()
    testCases()
