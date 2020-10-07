import numpy as np
import warnings

def multipleCCA(Cxx=None, Cxy=None, Cyy=None,
                reg=0, rank=1, CCA=True, rcond=1e-4, symetric=False):
    '''
    Compute multiple CCA decompositions using the given summary statistics
      [J,W,R]=multiCCA(Cxx,Cxy,Cyy,regx,regy,rank,CCA)
    Inputs:
      Cxx  = (d,d) current data covariance
      Cxy  = (nM,nE,tau,d) current per output ERPs
      Cyy  = (nM,nE,tau,nE,tau) current response covariance for each output
      reg  = (1,) :float or (2,):float linear weighting reg strength or separate values for Cxx and Cyy 
            OR (d,) regularisation ridge coefficients for Cxx  (0)
      rank= [1x1] number of top cca components to return (1)
      CCA = [bool] or (2,):bool  flag if we normalize the spatial dimension. (true)
      rcond  = [float] tolerance for singular eigenvalues.        (1e-4)
               or [2x1] separate rcond for Cxx (rcond(1)) and Cyy (rcond(2))
               or [2x1] (-1<-0) negative values = keep this fraction of eigen-values
               or (2,1) <-1 keep this many eigenvalues
      symetric = [bool] us symetric whitener?
    Outputs:
      J     = (nM,) optimisation objective scores
      W     = (nM,rank,d) spatial filters for each output
      R     = (nM,rank,nE,tau) responses for each stimulus event for each output
    Examples:
      # Supervised CCA
      Y = (nEp/nSamp,nY,nE) indicator for each event-type and output of it's type in each epoch
      X = (nEp/nSamp,tau,d) sliced pre-processed raw data into per-stimulus event responses
      stimTimes = (nEp) # sample number for each epoch or None if sample rate
      Cxx,Cxy,Cyy=updateSummaryStatistics(X, Y, stimTimes)
      J,w, r=multiCCA(Cxx, Cxy, Cyy, reg) # w=(nM,d)[dx1],r=(nM,nE,tau)[tau x nE] are the found spatial,spectral filters
      # incremental zeroTrain
      Cxx=[];Cxy=[];Cyy=[]; prediction=[];
      while isCalibration:
         # Slicing
         newX,newE,stimTimes= preprocessnSliceTrial(); #function which get 1 trials data+stim
         # Model Fitting
         Cxx, Cxy, Cyy= updateSummaryStatistics(newX, newE, stimTimes, Cxx, Cxy, Cyy);
         J, w, r      = multipleCCA(Cxx, Cxy, Cyy, regx, regy) # w=[dx1],r=[tau x nE] are the found spatial,spectral filters
      end
      # re-compute with current state, can try different regulisors, e.g. for adaptive bad-ch
      regx = updateRegFromBadCh();
      J, w, r=multipleCCA(Cxx, Cxy, Cyy, reg)
    Copyright (c) MindAffect B.V. 2018
    '''
    rank = int(max(1, rank))
    if not hasattr(reg, '__iter__'):
        reg = (reg, reg)  # ensure 2 element list
    if not hasattr(CCA, '__iter__'):
        CCA = (CCA, CCA)  # ensure 2 element list
    if not hasattr(rcond, '__iter__'):
        rcond = (rcond, rcond)  # ensure 2 element list
        
    # 3d Cxy, Cyy for linear alg stuff, (nM,feat,feat) - N.B. add model dim if needed
    nM = Cxy.shape[0] if Cxy.ndim > 3 else 1
    if Cxx.ndim > 2:
        Cxx2d = np.reshape(Cxx, (nM, Cxx.shape[-4]*Cxx.shape[-3], Cxx.shape[-2]*Cxx.shape[-1]))
    else:
        Cxx2d = np.reshape(Cxx, (nM, Cxx.shape[-2], Cxx.shape[-1]))

    if Cxx.ndim < 4 and Cyy.ndim >= 4:
        Cxy2d = np.reshape(Cxy, (nM, Cxy.shape[-3]*Cxy.shape[-2], Cxy.shape[-1]))  # (nM,(nE*tau),d)
    elif Cxx.ndim >= 4 and Cyy.ndim < 4:
        Cxy2d = np.reshape(Cxy, (nM, Cxy.shape[-3], Cxy.shape[-2]*Cxy.shape[-1]))  # (nM,nE,(tau*d))
    elif Cxx.ndim >= 4 and Cyy.ndim >= 4:
        raise NotImplementedError("Not immplemented yet for double temporally embedded inputs")
        
    if Cyy.ndim > 2:
        Cyy2d = np.reshape(Cyy, (nM, Cyy.shape[-4]*Cyy.shape[-3], Cyy.shape[-2] * Cyy.shape[-1]))  # (nM,(nE*tau),(nE*tau))
    else:
        Cyy2d = np.reshape(Cyy, (nM, Cyy.shape[-2], Cyy.shape[-1]))  # (nM,e,e)    
    rank = min(min(rank, Cxy2d.shape[-1]), Cyy2d.shape[-1])

    
    # convert to double precision if needed
    if np.issubdtype(Cxx2d.dtype, np.float32) or np.issubdtype(Cxx2d.dtype, np.signedinteger):
        Cxx2d = np.array(Cxx2d, dtype=np.float64)
    if np.issubdtype(Cxy2d.dtype, np.float32) or np.issubdtype(Cxy2d.dtype, np.signedinteger):
        Cxy2d = np.array(Cxy2d, dtype=np.float64)
    if np.issubdtype(Cyy2d.dtype, np.float32) or np.issubdtype(Cyy2d.dtype, np.signedinteger):
        Cyy2d = np.array(Cyy2d, dtype=np.float64)
    
    # do CCA/PLS for each output in turn
    J = np.zeros((Cxy2d.shape[0]))  # [ nM ]
    W = np.zeros((Cxy2d.shape[0], rank, Cxy2d.shape[2]))  # (nM,rank,d)
    R = np.zeros((Cxy2d.shape[0], rank, Cxy2d.shape[1]))  # (nM,rank,(nE*tau)) 
    for mi in range(Cxy2d.shape[0]):  # loop over posible models
        # get output specific ERPs
        Cxym = Cxy2d[mi, :, :]  # ((tau*nE),d)

        # Whitener for X
        if CCA[0]:
            # compute model specific whitener, or re-use the  last one
            if mi <= Cxx2d.shape[0]:
                isqrtCxx, _ = robust_whitener(Cxx2d[mi, :, :], reg[0], rcond[0], symetric)
            # compute whitened Cxy
            isqrtCxxCxym = np.dot(Cxym, isqrtCxx) # (nM,(nE*tau),d)
        else:
            isqrtCxxCxym = Cxym

        # Whitener for Y
        if CCA[1]:
            # compute model-specific whitener, or re-use the last one
            if mi <= Cyy2d.shape[0]:
                isqrtCyy, _ = robust_whitener(Cyy2d[mi, :, :], reg[1], rcond[1], symetric)
            isqrtCxxCxymisqrtCyy = np.dot(isqrtCyy.T, isqrtCxxCxym)
        else:
            isqrtCxxCxymisqrtCyy = isqrtCxxCxym

        # SVD for the double  whitened cross covariance
        #N.B. Rm=((nE*tau),rank),lm=(rank),Wm=(rank,d)
        Rm, lm, Wm = np.linalg.svd(isqrtCxxCxymisqrtCyy, full_matrices=False)  
        Wm = Wm.T  # (d,rank)

        # include relative component weighting directly in the  Left/Right singular values
        nlm = lm / np.max(lm)  # normalize so predictions have unit average norm
        Wm = Wm #* np.sqrt(nlm[np.newaxis, :])
        Rm = Rm * nlm[np.newaxis, :] #* np.sqrt(nlm[np.newaxis, :])

        # pre-apply the pre-whitener so can apply the result directly on input data
        if CCA[0]:
            Wm = np.dot(isqrtCxx, Wm)
        if CCA[1]:
            Rm = np.dot(isqrtCyy, Rm)

        # extract the desired parts of the solution information, largest singular values first
        slmidx = np.argsort(lm)  # N.B. ASCENDING order
        slmidx = slmidx[::-1]  # N.B. DESCENDING order
        r = min(len(lm), rank)  # guard rank>effective dim
        W[mi, :r, :] = Wm[:, slmidx[:r]].T  #(nM,rank,d)
        J[mi] = lm[slmidx[-1]]  # N.B. this is *wrong* for rank>1
        R[mi, :r, :] = Rm[:, slmidx[:r]].T  #(nM,rank,(nE*tau))
        
    # Map back to input shape
    if Cyy.ndim > 2:
        R = np.reshape(R, (R.shape[0], R.shape[1], Cyy.shape[-2], Cyy.shape[-1])) # (nM,rank,nE,tau)
    if Cxx.ndim > 2:
        W = np.reshape(W, (W.shape[0], W.shape[1], Cxx.shape[-2], Cxx.shape[-1]))

    # strip model dim if not needed
    if Cxy.ndim == 3:
        R = R[0, ...]
        W = W[0, ...]
        J = J[0, ...]
        
    return J, W, R


def robust_whitener(C:np.ndarray, reg:float=0, rcond:float=1e-6, symetric:bool=True, verb:int=0):
    """compute a robust whitener for the input covariance matrix C, s.t. isqrtC*C*isqrtC.T = I
    Args:
        C ((d,d) np.ndarray): Sample covariance matrix of the data
        reg (float, optional): regularization strength when computing the whitener. Defaults to 0.
        rcond (float, optional): reverse-condition number for truncating degenerate eigen-values when computing the inverse. Defaults to 1e-6.
        symetric (bool, optional): flag to produce a symetric-whitener (which preserves location) or not. Defaults to True.
        verb (int, optional): verbosity level. Defaults to 0.

    Returns:
        W (np.ndarray): The whitening matrix
        iW (np.ndarray): The inverse whitening matrix
    """    

    # ensure symetric
    C = (C + C.T) / 2

    # include the regularisor if needed
    # TODO[]: include the ridge later, i.e. after the eigendecomp?
    if not reg is None and not reg == 0:
        # TODO[]: Optimal shrinkage?
        if np.ndim(reg) == 0 or len(reg) == 1:  # weight w.r.t. same amplitude identity
            C = (1-reg)*C + reg*np.eye(C.shape[0], dtype=C.dtype)*np.median(C.diagonal())
        elif len(reg) == C.shape[0]:  # diag entries
            np.fill_diagonal(C, C.diagonal()+reg)
        else:  # full reg matrix
            C = C + reg

    # eigen decomp
    sigma, U = np.linalg.eig(C)  # sigma=(r,) U=(d,k)
    U = U.real
    
    # identify bad/degenerate eigen-values, complex, negative, inf, nan or too small
    bad = np.any(np.vstack((np.abs(sigma.imag > np.finfo(sigma.dtype).eps),  # complex
                            np.isinf(sigma),  # inf
                            np.isnan(sigma),  # nan
                            np.abs(sigma)<np.finfo(sigma.dtype).eps)), 0)  # too-small
    if verb:
        print("{} bad eig".format(sum(bad)))
    # zero these bad ones
    sigma[bad] = 0

    # additional badness conditions based on eigen-spectrum
    if not rcond is None:
        #print('rcond={}\nsigma\{}'.format(rcond,sigma))
        # identify eigen-values we want to remove due to rcond
        if 0 <= rcond:  # value threshold
            #print('bad={}'.format(bad))
            bad = np.logical_or(bad, sigma.real < rcond*np.median(np.abs(sigma)))
            #print('bad={}'.format(bad))
        elif -1 < rcond and rcond < 0:  # fraction
            si = np.argsort(sigma)  # N.B. Ascending
            bad[si[:int(len(si)*abs(rcond))]] = True  # up to this fraction are bad
        elif rcond < -1:  # number to discard
            si = np.argsort(sigma)
            bad[si[:abs(rcond)]] = True  # this many are bad

    if verb:
        print("{} rcond+bad eig".format(sum(bad)))

    # compute the whitener (and it's inverse)
    if not all(bad):
        keep = np.logical_not(bad)
        Ukeep = U[:, keep]  # (d,r)
        sqrtsigmakeep = np.sqrt(np.abs(sigma[keep]))  #(r,)
        # non-symetric (and rank  reducing) version
        sqrtC = Ukeep * sqrtsigmakeep[np.newaxis, :]
        isqrtC= Ukeep * (1.0/sqrtsigmakeep[np.newaxis, :])
        
        # post apply U to get symetric version if wanted
        if symetric:
            sqrtC = np.dot(sqrtC, Ukeep.T)
            isqrtC= np.dot(isqrtC, Ukeep.T)
    else:
        warnings.warn('Degenerate C matrices input!')
        sqrtC = 1
        isqrtC = 1
    return (isqrtC, isqrtC)


def plot_multicca_solution(w, r):
    # plot the solutions, spatial-filter / temporal-impulse-response
    import matplotlib.pyplot as plt
    plt.clf()
    plt.subplot(211); plt.plot(np.squeeze(w)); plt.title("Spatial")
    plt.subplot(212); plt.plot(np.squeeze(r).T); plt.title("Temporal")
    plt.show()


def cvSupervised(Xe, Me, stimTimes, evtlabs=('re', 'fe'), n_splits=10, rank=1):
    ''' do a cross-validated training of a multicca model using sklearn'''
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, crossautocov, updateCyy, updateCxy, autocov, updateCxx
    from mindaffectBCI.decoder.multipleCCA import multipleCCA
    from mindaffectBCI.decoder.scoreOutput import scoreOutput, dedupY0
    from mindaffectBCI.decoder.scoreStimulus import scoreStimulus, scoreStimulusCont
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
    from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
    from mindaffectBCI.decoder.stim2event import stim2event
    from sklearn.model_selection import StratifiedKFold
    print("Xe = {}\nMe = {}".format(Xe.shape, Me.shape))
    # convert from stimulus coding to brain response coding
    Ye = stim2event(Me, evtlabs, axis=1) # (nTrl, nEp, nY, nE) [ nE x y x ep x trl ]

    # cross validate in folds over trials
    Ye_true = Ye[:, :, 0:1, :] # N.B. horrible slicing trick to keep the dim
    print("Ye = {}\nYe_true = {}".format(Ye.shape, Ye_true.shape))
    Fy = np.zeros((1, Ye.shape[0], Ye.shape[1], Ye.shape[2])) # (nModel, nTrl, nEp, nY) [ nY x nEp x nTrl x nModel ]
    print("Fy = {}".format(Fy.shape))

    kf = StratifiedKFold(n_splits=n_splits)
    train_idx = np.arange(30)
    valid_idx = np.arange(30, Ye.shape[0])
    print("CV:", end='')
    for i, (train_idx, valid_idx) in enumerate(kf.split(np.zeros(Xe.shape[0]), np.zeros(Ye_true.shape[0]))):
        print(".", end='', flush=True)
        Cxx, Cxy, Cyy = updateSummaryStatistics(Xe[train_idx, :, :, :], Ye_true[train_idx, :, :, :], stimTimes)
        J, W, R = multipleCCA(Cxx, Cxy, Cyy, rank=rank)
        # valid performance
        Fei = scoreStimulus(Xe[valid_idx, :, :, :], W, R)
        #print("Fei = {}".format(Fei.shape))
        Fyi = scoreOutput(Fei, Ye[valid_idx, :, :, :], dedup0=True)
        #print("Fyi = {}".format(Fyi.shape))
        Fy[:, valid_idx, :, :] = Fyi
    decodingCurveSupervised(Fy)

    # retrain with all the data
    Cxx, Cxy, Cyy = updateSummaryStatistics(Xe, Ye_true, stimTimes)
    J, W, R = multipleCCA(Cxx, Cxy, Cyy)
    return (W, R, Fy)
    

def testcase():
    from mindaffectBCI.decoder.utils import testNoSignal, testSignal, sliceData, sliceY
    #from multipleCCA import *
    if False:
        X, Y, st = testNoSignal()
    else:
        X, Y, st, A, B = testSignal(tau=10, noise2signal=10)
    # TODO[]: summary stats directly without first slicing
    Y_true = Y[:, :, 0:1, :] if Y.ndim > 3 else Y[:, 0:1, :] # N.B. hack with [0] to stop removal of dim...
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_summary_statistics
    Cxx, Cxy, Cyy = updateSummaryStatistics(X, Y_true, tau=10)

    #plot_summary_statistics(Cxx,Cxy,Cyy)

    # single supervised training
    from mindaffectBCI.decoder.multipleCCA import multipleCCA, plot_multicca_solution
    J, w, r = multipleCCA(Cxx, Cxy, Cyy, rcond=-.3, symetric=True) # 50% rank reduction
    
    plot_multicca_solution(w, r)

    # apply to the data
    from scoreStimulus import scoreStimulus
    Fe = scoreStimulus(X, w, r)
    import matplotlib.pyplot as plt;
    plt.clf();
    plt.plot(np.einsum("Tsd,d->s",X,w.ravel()),'b',label="Xw");
    plt.plot(Y_true.ravel(),'g',label="Y");
    plt.plot(Fe.ravel()/np.max(Fe.ravel())/2,'r',label="Fe");
    plt.legend()
    from scoreOutput import scoreOutput
    print("Fe={}".format(Fe.shape))
    print("Y={}".format(Y.shape))
    Fy = scoreOutput(Fe, Y, r) # (nM,nTrl,nEp,nY)
    print("Fy={}".format(Fy.shape))
    import matplotlib.pyplot as plt
    sFy=np.cumsum(Fy, -2)
    plt.clf();plt.plot(sFy[0, 0, :, :]);plt.xlabel('epoch');plt.ylabel('output');plt.show()

    from decodingSupervised import decodingSupervised
    decodingSupervised(Fy)
    from decodingCurveSupervised import decodingCurveSupervised, plot_decoding_curve
    dc=decodingCurveSupervised(Fy)
    plot_decoding_curve(*dc)

def testcase_matlab_summarystatistics():
    from scipy.io import loadmat
    ss = loadmat('C:/Users/Developer/Desktop/utopia/matlab/buffer/SummaryStatistics.mat')
    print(ss)
    Cxx=ss['Cxx']
    Cxy=np.moveaxis(ss['Cxy'],(0,1,2),(2,1,0)) # (d,tau,e)->(e,tau,d)
    Cyy=np.moveaxis(ss['Cyy'],(0,1,2,3),(3,2,1,0)) # (e,tau,e,tau) -> (tau,e,tau,e)
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_summary_statistics, plot_erp, plot_factoredmodel
    plot_summary_statistics(Cxx,Cxy,Cyy)
    plt.show()
    from mindaffectBCI.decoder.multipleCCA import multipleCCA
    J,W,R = multipleCCA(Cxx,Cxy,Cyy)
    plot_factoredmodel(W,R)
    plot.show()

if __name__=="__main__":
    testcase()
