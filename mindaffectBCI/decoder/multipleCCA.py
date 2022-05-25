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

import numpy as np
import warnings

def multipleCCA(Cxx_dd:np.ndarray=None, Cyx_metd:np.ndarray=None, Cyy_metet:np.ndarray=None,
                reg:float=1e-9, rank:int=1, CCA:bool=True, rcond:float=(1e-8,1e-8), 
                symetric:bool=False, whiten_alg:str="eigh", normalize_sign:bool=True):
    '''
    Compute multiple CCA decompositions using the given summary statistics
      [J,W,R]=multiCCA(Cxx_dd,Cyx_metd,Cyy_metet,regx,regy,rank,CCA)
    Args:
      Cxx_dd (d,d): current data covariance
      Cyx_metd (nM,nE,tau,d): current per output ERPs
      Cyy_metet (nM,nE,tau,nE,tau): current response covariance for each output
           OR
             (nM,tau,nE,nE) compressed cov for each output at different time-lags
      reg  = (1,) :float or (2,):float linear weighting reg strength or separate values for Cxx_dd and Cyy_metet 
            OR (d,) regularisation ridge coefficients for Cxx_dd.  Penalizes small eigenvalues.  (0)
      rank (float): number of top cca components to return (1)
      CCA (bool): [bool] or (2,):bool  flag if we normalize the spatial dimension. (true)
      rcond (float): tolerance for singular eigenvalues.        (1e-4)
               or [2x1] separate rcond for Cxx_dd (rcond(1)) and Cyy_metet (rcond(2))
               or [2x1] (-1<-0) negative values = keep this fraction of eigen-values
               or (2,1) <-1 keep this many eigenvalues
      symetric (bool): use symetric whitener?
    Returns:
      J (nM,): optimisation objective scores
      W (nM,rank,d): spatial filters for each output
      R (nM,rank,nE,tau): responses for each stimulus event for each output
      A (nM,rank,d): spatial patterns for each output
      I (nM,rank,nE,tau): Impulse responses for each stimulus event for each output


    Examples:
      # Supervised CCA
      Y = (nEp/nSamp,nY,nE) indicator for each event-type and output of it's type in each epoch
      X = (nEp/nSamp,tau,d) sliced pre-processed raw data into per-stimulus event responses
      stimTimes = (nEp) # sample number for each epoch or None if sample rate
      Cxx_dd,Cyx_metd,Cyy_metet=updateSummaryStatistics(X, Y, stimTimes)
      J,w, r=multiCCA(Cxx_dd, Cyx_metd, Cyy_metet, reg) # w=(nM,d)[dx1],r=(nM,nE,tau)[tau x nE] are the found spatial,spectral filters
      # incremental zeroTrain
      Cxx_dd=[];Cyx_metd=[];Cyy_metet=[]; prediction=[];
      while isCalibration:
         # Slicing
         newX,newE,stimTimes= preprocessnSliceTrial(); #function which get 1 trials data+stim
         # Model Fitting
         Cxx_dd, Cyx_metd, Cyy_metet= updateSummaryStatistics(newX, newE, stimTimes, Cxx_dd, Cyx_metd, Cyy_metet);
         J, w, r, _,_      = multipleCCA(Cxx_dd, Cyx_metd, Cyy_metet, regx, regy) # w=[dx1],r=[tau x nE] are the found spatial,spectral filters
      end
      # re-compute with current state, can try different regulisors, e.g. for adaptive bad-ch
      regx = updateRegFromBadCh();
      J, w, r, _, _=multipleCCA(Cxx_dd, Cyx_metd, Cyy_metet, reg)
    Copyright (c) MindAffect B.V. 2018
    '''
    rank = int(max(1, rank))
    if not hasattr(reg, '__iter__'):
        reg = (reg, reg)  # ensure 2 element list
    if not hasattr(CCA, '__iter__'):
        CCA = (CCA, CCA)  # ensure 2 element list
    if not hasattr(rcond, '__iter__'):
        rcond = (rcond, rcond)  # ensure 2 element list
        
    # 3d Cyx_metd, Cyy_metet for linear alg stuff, (nM,feat,feat) - N.B. add model dim if needed
    nM = Cyx_metd.shape[0] if Cyx_metd.ndim > 3 else 1
    if Cxx_dd.ndim > 2:
        Cxx2d = np.reshape(Cxx_dd, (nM, Cxx_dd.shape[-4]*Cxx_dd.shape[-3], Cxx_dd.shape[-2]*Cxx_dd.shape[-1]))
    else:
        Cxx2d = np.reshape(Cxx_dd, (nM, Cxx_dd.shape[-2], Cxx_dd.shape[-1]))

    if Cxx_dd.ndim < 4 and Cyy_metet.ndim >= 4:
        Cyx2d = np.reshape(Cyx_metd, (nM, Cyx_metd.shape[-3]*Cyx_metd.shape[-2], Cyx_metd.shape[-1]))  # (nM,(nE*tau),d)
    elif Cxx_dd.ndim >= 4 and Cyy_metet.ndim < 4:
        Cyx2d = np.reshape(Cyx_metd, (nM, Cyx_metd.shape[-3], Cyx_metd.shape[-2]*Cyx_metd.shape[-1]))  # (nM,nE,(tau*d))
    elif Cxx_dd.ndim >= 4 and Cyy_metet.ndim >= 4:
        raise NotImplementedError("Not immplemented yet for double temporally embedded inputs")
        
    if Cyy_metet.ndim > 2:
        Cyy2d = np.reshape(Cyy_metet, (nM, Cyy_metet.shape[-4]*Cyy_metet.shape[-3], Cyy_metet.shape[-2] * Cyy_metet.shape[-1]))  # (nM,(nE*tau),(nE*tau))
    else:
        Cyy2d = np.reshape(Cyy_metet, (nM, Cyy_metet.shape[-2], Cyy_metet.shape[-1]))  # (nM,e,e)    
    rank = min(rank, min(Cyx2d.shape[1:]))

    
    # convert to double precision if needed
    if np.issubdtype(Cxx2d.dtype, np.float32) or np.issubdtype(Cxx2d.dtype, np.signedinteger):
        Cxx2d = np.array(Cxx2d, dtype=np.float64)
    if np.issubdtype(Cyx2d.dtype, np.float32) or np.issubdtype(Cyx2d.dtype, np.signedinteger):
        Cyx2d = np.array(Cyx2d, dtype=np.float64)
    if np.issubdtype(Cyy2d.dtype, np.float32) or np.issubdtype(Cyy2d.dtype, np.signedinteger):
        Cyy2d = np.array(Cyy2d, dtype=np.float64)
    
    # do CCA/PLS for each output in turn
    J_M = np.zeros((Cyx2d.shape[0]))  # [ nM ]
    W_Mrd = np.zeros((Cyx2d.shape[0], rank, Cyx2d.shape[2]),dtype=Cxx_dd.dtype)  # (nM,rank,d)
    A_Mrd = np.zeros(W_Mrd.shape,dtype=W_Mrd.dtype) # spatial patterns
    R_M_r_te = np.zeros((Cyx2d.shape[0], rank, Cyx2d.shape[1]),dtype=Cxx_dd.dtype)  # (nM,rank,(nE*tau)) 
    I_M_r_te = np.zeros(R_M_r_te.shape,dtype=R_M_r_te.dtype) # temporal patterns
    for mi in range(Cyx2d.shape[0]):  # loop over posible models
        # get output specific ERPs
        Cyxm_te_d = Cyx2d[mi, :, :]  # ((tau*nE),d)

        # Whitener for X
        if CCA[0]:
            # compute model specific whitener, or re-use the  last one
            if mi <= Cxx2d.shape[0]:
                isqrtCxx_dm, sqrtCxx_dm = robust_whitener(Cxx2d[mi, :, :], reg[0], rcond[0], symetric, alg=whiten_alg)
            # compute whitened Cyx
            CyxmisqrtCxx_te_m = Cyxm_te_d @ isqrtCxx_dm # (nM,(nE*tau),d)
        else:
            CyxmisqrtCxx_te_m = Cyxm_te_d

        # Whitener for Y
        if CCA[1]:
            # compute model-specific whitener, or re-use the last one
            if mi <= Cyy2d.shape[0]:
                isqrtCyy_te_o, sqrtCyy_te_o = robust_whitener(Cyy2d[mi, :, :], reg[1], rcond[1], symetric, alg=whiten_alg)
            isqrtCyyCyxmisqrtCxx_om = isqrtCyy_te_o.T @ CyxmisqrtCxx_te_m
        else:
            isqrtCyyCyxmisqrtCxx_om = CyxmisqrtCxx_te_m

        # SVD for the double  whitened cross covariance
        #N.B. Rm=((nE*tau),rank),lm=(rank),Wm=(rank,d)
        Rm_or, lm_r, Wm_rm = np.linalg.svd(isqrtCyyCyxmisqrtCxx_om, full_matrices=False)  
        Wm_mr = Wm_rm.T  # (d,rank) 

        # include relative component weighting directly in the  Left/Right singular values
        nlm = (lm_r / np.max(lm_r)) if np.max(lm_r)>0 else np.ones(lm_r.shape,dtype=lm_r.dtype)  # normalize so predictions have unit average norm
        Wm_mr = Wm_mr * np.sqrt(nlm[np.newaxis, :])
        Rm_or = Rm_or * np.sqrt(nlm[np.newaxis, :]) #* np.sqrt(nlm[np.newaxis, :])

        # pre-apply the pre-whitener so can apply the result directly on input data
        Am_dr = Wm_mr
        if CCA[0]:
            #Am_dr = sqrtCxx_dm @ Wm_mr # pattern - what detect in raw
            Wm_dr = isqrtCxx_dm @ Wm_mr # filter - how detect from raw
        else:
            Am_dr = Wm_mr
            Wm_dr = Wm_mr

        Im_te_r = Rm_or
        if CCA[1]:
            #Im_te_r = sqrtCyy_te_o @ Rm_or # pattern - what detect in raw
            Rm_te_r = isqrtCyy_te_o @ Rm_or # filter - how detect from raw
        else:
            Im_te_r = Rm_or # pattern - what detect in raw
            Rm_te_r = Rm_or # filter - how detect from raw

        # extract the desired parts of the solution information, largest singular values first
        slmidx = np.argsort(lm_r)  # N.B. ASCENDING order
        slmidx = slmidx[::-1]  # N.B. DESCENDING order
        r = min(len(lm_r), rank)  # guard rank>effective dim
        W_Mrd[mi, :r, :] = Wm_dr[:, slmidx[:r]].T  #(nM,rank,d)
        A_Mrd[mi, :r, :Am_dr.shape[0]] = Am_dr[:, slmidx[:r]].T
        J_M[mi] = lm_r[slmidx[-1]]  # N.B. this is *wrong* for rank>1
        R_M_r_te[mi, :r, :] = Rm_te_r[:, slmidx[:r]].T  #(nM,rank,(nE*tau))
        I_M_r_te[mi, :r, :] = Im_te_r[:, slmidx[:r]].T
    # Map back to input shape
    if Cyy_metet.ndim > 2:
        R_Mrte = np.reshape(R_M_r_te, (R_M_r_te.shape[0], R_M_r_te.shape[1], Cyy_metet.shape[-2], Cyy_metet.shape[-1])) # (nM,rank,nE,tau)
        I_Mrte = np.reshape(I_M_r_te, (I_M_r_te.shape[0], I_M_r_te.shape[1], Cyy_metet.shape[-2], Cyy_metet.shape[-1])) # (nM,rank,nE,tau)
    if Cxx_dd.ndim > 2:
        W_Mrd = np.reshape(W_Mrd, (W_Mrd.shape[0], W_Mrd.shape[1], Cxx_dd.shape[-2], Cxx_dd.shape[-1]))
        A_Mrd = np.reshape(A_Mrd, (A_Mrd.shape[0], A_Mrd.shape[1], Cxx_dd.shape[-2], Cxx_dd.shape[-1]))

    if normalize_sign:
        sgn_k = (np.median(A_Mrd,axis=(-3,-1),keepdims=False) >= 0)*2 - 1.0
        A_Mrd = A_Mrd * sgn_k[:,np.newaxis]
        W_Mrd = W_Mrd * sgn_k[:,np.newaxis]
        R_Mrte = R_Mrte * sgn_k[:,np.newaxis,np.newaxis]
        I_Mrte = I_Mrte * sgn_k[:, np.newaxis, np.newaxis]

    # strip model dim if not needed
    if Cyx_metd.ndim == 3:
        R_Mrte = R_Mrte[0, ...]
        W_Mrd = W_Mrd[0, ...]
        J_M = J_M[0, ...]
        
    return J_M, W_Mrd, R_Mrte, A_Mrd, I_Mrte


def robust_whitener(C:np.ndarray, reg:float=0, rcond:float=1e-6, symetric:bool=True, verb:int=0, pca:bool=False, alg='eigh'):
    """compute a robust whitener for the input covariance matrix C, s.t. isqrtC*C*isqrtC.T = I
    Args:
        C_dd ((d,d) np.ndarray): Sample covariance matrix of the data
        reg (float, optional): regularization strength when computing the whitener, 0=no-reg, 1=no-whiten. Defaults to 0.
        rcond (float, optional): reverse-condition number for truncating degenerate eigen-values when computing the inverse. Defaults to 1e-6.
        symetric (bool, optional): flag to produce a symetric-whitener (which preserves location) or not. Defaults to True.
        verb (int, optional): verbosity level. Defaults to 0.

    Returns:
        W_dk (np.ndarray): The whitening matrix
        iW_dk (np.ndarray): The inverse whitening matrix
    """
    assert not np.any(np.isnan(C.ravel())) and not np.any(np.isinf(C.ravel())), "NaN or Inf in inputs!"

    # ensure symetric
    C = (C + C.T) / 2

    # include the regularisor if needed
    # TODO[]: include the ridge later, i.e. after the eigendecomp?
    if reg==1:
        sqrtC = np.eye(C.shape[0], dtype=C.dtype)
        isqrtC = sqrtC
        return isqrtC, sqrtC
    elif not reg is None and not reg == 0:
        # TODO[]: Optimal shrinkage?
        if callable(reg):
            C = reg(C)
        elif np.ndim(reg) == 0 or len(reg) == 1:  # weight w.r.t. same amplitude identity
            C = (1-reg)*C + reg*np.eye(C.shape[0], dtype=C.dtype)*np.mean(C.diagonal())
        elif len(reg) == C.shape[0]:  # diag entries
            np.fill_diagonal(C, C.diagonal()+reg)
        else:  # full reg matrix
            C = C + reg

    if alg=='chol':
        sqrtC = np.linalg.cholesky(C)
        isqrtC = np.linalg.pinv(sqrtC)
        return isqrtC, sqrtC

    # eigen decomp
    # N.B. use eigh as eig is numerically crap!
    sigma_k, U_dk = np.linalg.eigh(C)  # sigma=(r,) U=(d,k)
    U_dk = U_dk.real
    
    # identify bad/degenerate eigen-values, complex, negative, inf, nan or too small
    bad = np.any(np.vstack((np.abs(sigma_k.imag > np.finfo(sigma_k.dtype).eps),  # complex
                            np.isinf(sigma_k),  # inf
                            np.isnan(sigma_k),  # nan
                            np.abs(sigma_k)<np.finfo(sigma_k.dtype).eps)), 0)  # too-small
    if verb>1:
        print("{}/{} bad eig".format(sum(bad),len(bad)))
    # zero these bad ones
    #sigma_k[bad] = 0

    # additional badness conditions based on eigen-spectrum
    if not rcond is None:
        #print('rcond={}\nsigma\{}'.format(rcond,sigma))
        # identify eigen-values we want to remove due to rcond
        if 0 <= rcond:  # value threshold
            #print('bad={}'.format(bad))
            #bad = np.logical_or(bad, sigma_k.real < rcond*np.median(np.abs(sigma_k)))
            bad = np.logical_or(bad, sigma_k.real*(1-rcond) < rcond*np.mean(np.abs(sigma_k)))
            #print('bad={}'.format(bad))
        elif -1 < rcond and rcond < 0:  # fraction
            si = np.argsort(sigma_k)  # N.B. Ascending
            bad[si[:int(len(si)*abs(rcond))]] = True  # up to this fraction are bad
        elif rcond < -1:  # number to keep
            si = np.argsort(sigma_k) # ascending order
            bad[si[:-abs(rcond)]] = True  # up-to rcond smallest are marked as bad

    if verb>1:
        print("{}/{} rcond+bad eig".format(sum(bad),len(bad)))

    # compute the whitener (and it's inverse)
    if not all(bad):
        keep = np.logical_not(bad)
        Ukeep_dk = U_dk[:, keep]  # (d,r)
        sqrtsigmakeep = np.sqrt(np.abs(sigma_k[keep])) if not pca else np.ones(sum(keep)) #(r,)
        
        # post apply U to get symetric version if wanted
        if symetric:
            # non-symetric (and rank  reducing) version
            sqrtC_dk = Ukeep_dk * np.sqrt(sqrtsigmakeep)[np.newaxis, :]
            isqrtC_dk= Ukeep_dk * np.sqrt(1.0/sqrtsigmakeep)[np.newaxis, :]
            sqrtC_dk = sqrtC_dk @ sqrtC_dk.T
            isqrtC_dk= isqrtC_dk @ isqrtC_dk.T
        else:
            # non-symetric (and rank  reducing) version
            sqrtC_dk = Ukeep_dk * sqrtsigmakeep[np.newaxis, :]
            isqrtC_dk= Ukeep_dk * (1.0/sqrtsigmakeep[np.newaxis, :])
            
    else:
        warnings.warn('Degenerate C matrices input!')
        sqrtC_dk = np.eye(C.shape[0], dtype=C.dtype)
        isqrtC_dk = sqrtC_dk
    return isqrtC_dk, sqrtC_dk


def edge_reg(N:int,alpha:float=10,C:float=1, symetric:bool=False):
    """generate an edge penalizing regulazor

    Args:
        N (int): length of regularizor to make
        symetric (bool): symetric (both-edges) or just the right edge.  Defaults to False.
        alpha (int, optional): slope of the reg, higher is stronger at edge. Defaults to 10.

    Returns:
        ndarray (N,): the reg-strength array 
    """    
    if symetric:
        reg = C*np.exp(alpha*np.abs(np.arange(N)-N/2)/(N/2-1)) / np.exp(alpha)
    else:
        reg = C*np.exp(alpha*np.arange(N)/(N-1)) / np.exp(alpha)
    return reg

def cvSupervised(Xe, Me, stimTimes, evtlabs=('re', 'fe'), n_splits=10, rank=1):
    ''' do a cross-validated training of a multicca model using sklearn'''
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, crossautocov, updateCyy, updateCyx, autocov, updateCxx
    from mindaffectBCI.decoder.multipleCCA import multipleCCA
    from mindaffectBCI.decoder.scoreOutput import scoreOutput, dedupY0
    from mindaffectBCI.decoder.scoreStimulus import scoreStimulus, scoreStimulusCont
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
    from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
    from mindaffectBCI.decoder.stim2event import stim2event
    from sklearn.model_selection import StratifiedKFold
    print("Xe = {}\nMe = {}".format(Xe.shape, Me.shape))
    # convert from stimulus coding to brain response coding
    Ye, _, _ = stim2event(Me, evtlabs, axis=1) # (nTrl, nEp, nY, nE) [ nE x y x ep x trl ]

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
        Cxx, Cyx, Cyy = updateSummaryStatistics(Xe[train_idx, :, :, :], Ye_true[train_idx, :, :, :], stimTimes)
        J, W, R, _, _ = multipleCCA(Cxx, Cyx, Cyy, rank=rank)
        # valid performance
        Fei = scoreStimulus(Xe[valid_idx, :, :, :], W, R)
        #print("Fei = {}".format(Fei.shape))
        Fyi = scoreOutput(Fei, Ye[valid_idx, :, :, :], dedup0=True)
        #print("Fyi = {}".format(Fyi.shape))
        Fy[:, valid_idx, :, :] = Fyi
    decodingCurveSupervised(Fy)

    # retrain with all the data
    Cxx, Cyx, Cyy = updateSummaryStatistics(Xe, Ye_true, stimTimes)
    J, W, R,_, _ = multipleCCA(Cxx, Cyx, Cyy)
    return (W, R, Fy)
    

def fit_predict(X_TSd, Y_TSye, tau:int=10, offset:int=0, fs:float=100, 
                evtlabs:list=None, outputs:list=None, ch_names:list=None, label:str=None,
                center:bool=True, unitnorm:bool=True, centerY:bool=False,
                test_idx=slice(10,None), Ytrn:int=0, **kwargs):
    from mindaffectBCI.decoder.scoreStimulus import scoreStimulus
    from mindaffectBCI.decoder.scoreOutput import scoreOutput
    from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised, plot_decoding_curve
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_summary_statistics, plot_factoredmodel, plot_trial, plot_subspace, plot_erp
    from mindaffectBCI.decoder.multipleCCA import multipleCCA
    from mindaffectBCI.decoder.analyse_datasets import plot_trial_summary
    import matplotlib.pyplot as plt

    if centerY:
        Y_TSye = Y_TSye - np.mean(Y_TSye,axis=-2,keepdims=True)

    Xtrn_TSd = X_TSd
    Ytrn_TSye = Y_TSye
    if not test_idx is None:
        test_ind = np.zeros((X_TSd.shape[0],),dtype=bool)
        test_ind[test_idx] = True
        train_ind = np.logical_not(test_ind)
        print("Training Idx: {}\nTesting Idx :{}\n".format(np.flatnonzero(train_ind),np.flatnonzero(test_ind)))
        Xtrn_TSd = X_TSd[train_ind,...]
        Ytrn_TSye = Y_TSye[train_ind,...]


    # set to the outputs used during training
    if Ytrn is not None:
        Ytrn_TSye = Ytrn_TSye[:, :, (Ytrn,), :] if Ytrn_TSye.ndim > 3 else Ytrn_TSye[np.newaxis, :, (Ytrn,), :]

    Cxx, Cyx, Cyy = updateSummaryStatistics(Xtrn_TSd, Ytrn_TSye, 
                        tau=tau, offset=offset, center=center, unitnorm=unitnorm)

    # single supervised training
    J, w, r, _, _ = multipleCCA(Cxx, Cyx, Cyy, **kwargs) 
    Fe = scoreStimulus(X_TSd, w, r, offset=offset)
    Fy = scoreOutput(Fe, Y_TSye, R_mket=r, dedup0=True) # (nM,nTrl,nEp,nY)

    dc=decodingCurveSupervised(Fy[0,...])

    return dc, w, r, Cxx, Cyx, Cyy, Fe, Fy


def fit_predict_plot(X_TSd, Y_TSye, tau:int=10, offset:int=0, fs:float=100, 
                evtlabs:list=None, outputs:list=None, ch_names:list=None, label:str=None,
                center:bool=True, unitnorm:bool=True, centerY:bool=False,
                test_idx=slice(10,None), Ytrn:int=0, block=False, **kwargs):
    from mindaffectBCI.decoder.decodingCurveSupervised import plot_decoding_curve
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_summary_statistics, plot_factoredmodel, plot_trial, plot_subspace, plot_erp
    from mindaffectBCI.decoder.analyse_datasets import plot_trial_summary
    import matplotlib.pyplot as plt

    times = np.arange(tau)
    if offset is not None: 
        times = times + offset
    if fs is not None:
        times = times/fs

    dc, w, r, Cxx, Cyx, Cyy, Fe, Fy = fit_predict(X_TSd, Y_TSye, tau=tau, offset=offset, fs=fs, 
                center=center, unitnorm=unitnorm, centerY=centerY,
                test_idx=test_idx, Ytrn=Ytrn, **kwargs)

    plt.figure()
    plot_summary_statistics(Cxx,Cyx,Cyy,times=times,outputs=outputs,ch_names=ch_names,evtlabs=evtlabs,label=label)

    plt.figure()
    plot_erp(Cyx,times=times,outputs=outputs,ch_names=ch_names,evtlabs=evtlabs,suptitle=label)

    plt.figure()
    plot_factoredmodel(w, r, ncol=3, suptitle=label, times=times, ch_names=ch_names, evtlabs=evtlabs)
    plt.subplot(1,3,3)
    plot_decoding_curve(*dc)

    plt.figure()
    plot_subspace(X_TSd[:5,...],Y_TSye[:5,:,0:1,:],w,r,None,None,offset=offset,fs=fs,block=False,label="{} raw".format(label))

    plt.figure() 
    plot_trial_summary(X_TSd[0:1,...], Y_TSye[0:1,:,0,:], Fy[0,...], fs=fs, Yerr=None, Py=None, Fe=Fe[0,0:1,...], label=label, block=False)

    plt.show(block=block)
    return w, r, dc, Cxx, Cyx, Cyy, Fe, Fy

def filterbank_testcase():
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.utils import testNoSignal, testSignal
    from mindaffectBCI.decoder.preprocess import fft_filterbank
    from mindaffectBCI.decoder.scoreStimulus import scoreStimulus
    from mindaffectBCI.decoder.scoreOutput import scoreOutput
    from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised, plot_decoding_curve
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_summary_statistics, plot_factoredmodel
    from mindaffectBCI.decoder.multipleCCA import multipleCCA

    # get a dataset
    if False:
        X, Y, st = testNoSignal()
        fs = 100
    else:
        X, Y, st, A, B = testSignal(tau=10, noise2signal=1, nTrl=100, nSamp=300)
        fs = 100

    # 1) Test without filter bank
    fit_predict_plot(X, Y)
    plt.show(block=False)

    # 2) Test with filterbank, 5-bands of 4hz
    plt.figure(2)
    Xf, _, _ = fft_filterbank(X, ((0,10,'bandpass'),(8,14,'bandpass'),(12,18,'bandpass'),(16,-1,'bandpass')), fs=fs) #(tr,samp,band,ch)
    fit_predict( Xf.reshape(Xf.shape[:-2]+(-1,)), Y)
    plt.show(block=True)




def loaddata(savefile=None, filterband=((45,65),(3,12,'bandpass')),preprocess_args:dict=None,
             centerY:bool=False, evtlabs=None,fs_out:float=100,target_output:int=None):
    from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
    from mindaffectBCI.decoder.stim2event import stim2event
    from mindaffectBCI.decoder.analyse_datasets import debug_test_dataset, plot_stimseq, plot_stim_encoding
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp, plot_trial
    from mindaffectBCI.decoder.preprocess import preprocess

    if savefile is None or savefile=='askloadsavefile':
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        import os
        root = Tk()
        root.withdraw()
        savefile = askopenfilename(initialdir=os.path.dirname(os.path.abspath(__file__)),
                                    title='Chose mindaffectBCI save File',
                                    filetypes=(('mindaffectBCI','mindaffectBCI*.txt'),('All','*.*')))
        root.destroy()

    # load
    X_TSd, Y_TSy, coords = load_mindaffectBCI(savefile, filterband=filterband, order=6, ftype='butter', fs_out=fs_out)

    if preprocess_args is not None:
        X_TSd, Y_TSy, coords = preprocess(X_TSd, Y_TSy, coords, **preprocess_args)

    fs = coords[1]['fs'] if coords is not None else 100
    plot_trial(X_TSd[0:1,...],Y_TSy[0:1,...],fs=fs)

    #score, dc, Fy, clsfr, rawFy = debug_test_dataset(X_TSd, Y_TSy, coords,tau_ms=450, evtlabs=evtlabs, model='cca')
    Y_TSye = None
    outputs = None
    label=None

    label = savefile[-30:]
    if evtlabs is None:
        if 'rc'in savefile:
            evtlabs=('re','ntre')
        else:
            evtlabs=('re','fe')

    if evtlabs is not None and Y_TSye is None:
        Y_TSye, labs, _ = stim2event(Y_TSy,evtlabs)
    coords.append(dict(name='output', unit=None, coords=labs))

    # set the ch_names dependng on the file type...
    ch_names = ('P7','PO7','PO8','P8','Oz','Iz','POz','Cz')
    coords[2]['coords']=ch_names

    plot_stim_encoding(Y_TSy[0:1,...], Y_TSye[0:1,...], evtlabs=evtlabs, fs=fs, outputs=outputs, suptitle=label)

    return X_TSd, Y_TSye, coords, outputs, labs, label


def testcase(savefile=None, tau_ms:float=350, offset_ms:float=100, 
            filterband=((45,65),(4,25,'bandpass')), fs_out:float=100, evtlabs=('re','fe'),
            preprocess_args:dict=dict(badChannelThresh=3), 
            target_output:int=None, rank:int=1, nTrn:int=10, Ytrn:int=0, **kwargs):
    if savefile == 'simdata':
        X_TSd, Y_TSye, coords, outputs, evtlabs, label = simdata(nTrl=100, nSamp=500, nY=10, tau=10, noise2signal=3, irf=None, seed=1)
    else:
        X_TSd, Y_TSye, coords, outputs, evtlabs, label = loaddata(savefile, evtlabs=evtlabs, filterband=filterband,fs_out=fs_out, target_output=target_output, preprocess_args=preprocess_args)

    fs = coords[1]['fs'] if coords is not None else 100
    ch_names = coords[2]['coords'] if coords is not None else None
    tau = int(tau_ms*fs/1000)
    offset = int(offset_ms*fs/1000)

    fit_predict_plot(X_TSd,Y_TSye,
                fs=fs,evtlabs=evtlabs,ch_names=ch_names,outputs=outputs,label=label,
                tau=tau,offset=offset, rank=rank, Ytrn=Ytrn, test_idx=slice(nTrn,None), block=True, **kwargs)


def edge_regularisor(XX,C=.8,nE=2,symetric=False):
    if C==0:
        return XX
    if nE is None: nE=1
    tau = XX.shape[0]//nE
    r = edge_reg(tau,C=1,alpha=5,symetric=symetric)
    r = r / np.mean(r) / nE
    rs=[]
    for ei in range(nE):
        ri = r * np.mean(np.diag(XX)[ei*tau:(ei+1)*tau])
        rs.append(ri)
    r = np.concatenate(rs,0)
    return (1-C)*XX + C*r

if __name__=="__main__":
    regx = 0
    testcase(savefile='simdata',offset_ms=0,nTrn=30, rank=2, centerY=False, reg=(regx,edge_regularisor))
    #testcase(evtlabs='hoton_fe')