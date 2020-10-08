import numpy as np
from mindaffectBCI.decoder.utils import window_axis, idOutliers, zero_outliers
#@function
def updateSummaryStatistics(X, Y, stimTimes=None, 
                            Cxx=None, Cxy=None, Cyy=None, 
                            badEpThresh=4, halflife_samp=1, cxxp=True, cyyp=True, tau=None, offset=0, center=True):
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
    Cxy = updateCxy(Cxy, X, Y, stimTimes, tau, wght, offset, center)
    # Update Cxx
    if (cxxp):
        Cxx = updateCxx(Cxx, X, stimTimes, None, wght, center)

    # ensure Cyy has the right size if not entry-per-model
    if (cyyp):
        # TODO [] : support overlapping info between update calls
        Cyy = updateCyy(Cyy, Y, stimTimes, tau, wght)
        
    return Cxx, Cxy, Cyy

#@function
def updateCxx(Cxx, X, stimTimes=None, tau=None, wght=1, center=False, unitnorm=True):
    '''
    Cxx = [d x d] current data covariance
    X = (nTrl, nEp, tau, d) raw response for the current stimulus event
            d=#electrodes tau=#response-samples  nEpoch=#stimulus events to process
        OR
        (nTrl, nSamp, d) 
    stimTimes_samp = (nTrl, nEp) sample times for start each epoch.  Used to detect
    wght=[1x1]
    center:bool - compute centered covariance? (True)
    '''
    # ensure 3d
    if X.ndim == 2:
        X = X[np.newaxis, ...]

    # use X as is, computing purely spatial cov
    if tau is None:
        # N.B. reshape is expensive with window_axis
        if X.ndim == 3:
            XX = np.einsum("TEd,TEe->de", X, X)
            N = X.shape[0]
        elif X.ndim == 4:
            XX = np.einsum("TEtd,TEte->de", X, X)
            N = X.shape[0]*X.shape[1]

        if center:
            sX = np.sum(X.reshape((-1,X.shape[-1])),axis=0) # feature mean (d,)
            XX = XX - np.einsum("i,j->ij",sX,sX) *X.shape[-1]/X.size
        

    else: # compute time-shifted covariance
        Xe = window_axis(X, winsz=tau, axis=-2) # (nTrl, nSamp, tau, d)
        # N.B. reshape is expensive if use window axis
        XX = np.einsum("TEtd,TEue->tdue",Xe, Xe)

        if center:
            raise NotImplementedError("Centering with shifts!")

    if unitnorm:
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
    if offset<-tau+1:
        raise NotImplementedError("Cant offset backwards by more than window size")
    if offset>0:
        raise NotImplementedError("Cant offset by positive amounts")
    if verb > 1: print("tau={}".format(tau))
    if stimTimes is None:
        # X, Y are at sample rate, slice X every sample
        Xe = window_axis(X, winsz=tau, axis=-2) # (nTrl, nSamp, tau, d)
        Ye = Y[:, -offset:Xe.shape[-3]-offset, :, :] # shrink Y to fit w.r.t. window
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
        #print('Cxy - unitnorm!')
        #if verb>0 : 
        #print("tau={}".format(tau))
        #XY = XY / np.sqrt(tau)
        pass
    
    Cxy = wght*Cxy + XY if Cxy is not None else XY
    return Cxy

# @function
def updateCyy(Cyy, Y, stimTime=None, tau=None, wght=1, zeropadded=True, unitnorm=True):
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
        Cyy = np.zeros((Y.shape[-2], Y.shape[-1], tau, Y.shape[-1], tau))
    if Y.ndim == 3:  # ensure is 4-d
        Y = Y[np.newaxis, :, :, :] # (nTrl, nEp/nSamp, nY, nE) [nE x nY x nEpoch/nSamp x nTrl]

    if stimTime is None: # fast-path, already at sample rate
        if np.issubdtype(Y.dtype, np.floating): # all at once
            #print("Y={}".format(Y.shape))
            Ys = window_axis(Y, winsz=tau, axis=-3) # window of length tau (nTrl, nSamp, tau, nY, nE) [ nE x nY x nSamp x tau x nTrl ]
            #print("Ys={}".format(Ys.shape))
            MM = np.einsum("TStye, TSuyf->yetfu", Ys, Ys) # compute cross-covariance (nY, nE, tau, nE, tau) [ nE x tau x nE x tau x nY ]
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
    Cyy = wght*Cyy + MM if Cyy is not None else Cyy
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
    '''
    Cxx =(d, d) updated data covariance
    Cxy = (nY, nE, tau, d)  updated per output ERPs
    Cyy = (nY, nE, tau, nE, tau) updated response covariance for each output
    '''
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
    plt.subplot(311);
    plt.imshow(Cxx, origin='lower', extent=[0, Cxx.shape[0], 0, Cxx.shape[1]])
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
        plt.title('{}'.format(evtlabs[ei]))
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
    plt.title('Cyy')

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


def plot_factoredmodel(A, R, evtlabs=None, times=None, ch_names=None, ch_pos=None, fs=None, spatial_filter_type="Filter"):
    '''
    Make a multi-plot of a factored model
    A = (k,d)
    R = (k,e,tau)
    '''
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
        if fs is not None:
            times = times / fs
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
    ncols = 2 #int(np.ceil(np.sqrt(A.shape[0])))
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
            axA.grid(True)
            axR.grid(True)
            pA = axA
            pR = axR
        else: # normal plot
            pR = plt.subplot(nrows, ncols, subploti+2, sharex=axR, sharey=axR) 
            pA = plt.subplot(nrows, ncols, subploti+1, sharex=axA, sharey=axA) 
            plt.tick_params(labelbottom=False,labelleft=False) # no labels
            pA.grid(True)
            pR.grid(True)

        # make the spatial plot
        sign = np.sign(A[ci,np.argmax(np.abs(A[ci,:]))]) # normalize directions
        if not ch_pos is None:
            cRng= np.max(np.abs(A.reshape((-1))))
            levels = np.linspace(-cRng,cRng,20)
            tt=pA.tricontourf(ch_pos[:,0],ch_pos[:,1],A[ci,:]*sign,levels=levels,cmap='Spectral')
            pA.plot(ch_pos[:,0],ch_pos[:,1],'ko',markersize=5)
            plt.colorbar(tt)
        else:
            pA.plot(ch_names,A[ci,:]*sign,'.-')
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

    plot_Cxy(Cxy)
    
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


def testComputationMethods():
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

    # other measures....
    from updateSummaryStatistics import updateSummaryStatistics, autocov, cov, crossautocov, updateCyy, updateCxx, updateCxy
    Cxdtxdt = autocov(X, tau=18) # (tau[0],dx,tau[1],dx)
    Cxdtydt = crossautocov(X, Y[:, :, 0:1], tau=18) # (tau[0],dx,tau[1],dy)
    Cydxdt  = crossautocov(Y[:, :, 0:1], X, tau=[1,18]) # (tauy,dy,taux,dx)
    Cydtydt = np.stack([crossautocov(Y[:, :, yi:yi+1], Y[:, :, yi:yi+1], tau=18) for yi in range(Y.shape[-1])], 0) # (nY,tau,d,tau,d)
    Cxxdd   = cov(X)

    # compare
    Cxx=updateCxx(None,X,None)
    Cxy=updateCxy(None,X,Y[:,:,0:1,np.newaxis],None,tau=18)#(1,dy,tau,dx)
    Cyy=updateCyy(None,Y[:,:,:,np.newaxis],None,tau=18) #(nY,yd,tau,yd,tau)

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

    irf=(0,0,0,0,0,0,0,0,0,1)
    offset=0; # X->lag-by-10
    X,Y,st,A,B = testSignal(nTrl=1,nSamp=500,d=1,nE=1,nY=1,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)
    print("A={}\nB={}".format(A, B))
    print("X{}={}".format(X.shape, X[:30, np.argmax(np.abs(A))]))
    print("Y{}={}".format(Y.shape, Y[0, :30, 0]))
    
    tau=10; uss_offset=0
    Cxx, Cxy, Cyy = updateSummaryStatistics(X, Y, None, tau=tau, offset=uss_offset)
    print("Cxx={} Cxy={} Cyy={}".format(Cxx.shape, Cxy.shape, Cyy.shape))
    plt.figure(1);plot_summary_statistics(Cxx,Cxy,Cyy); plt.show()
    
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
    testCases()
