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

def normalizeOutputScores(Fy, validTgt=None, badFyThresh=4,
                          normSum=True, centFy=True, detrendFy=False, 
                          nEpochCorrection=0,
                          minDecisLen=0, maxDecisLen=0,
                          bwdAccumulate=False,
                          priorsigma=None, marginalizemodels=True):
    '''
    normalize the raw output scores to feed into the Perr computation

    Args:
      Fy (nM,nTr,nEp,nY): float
      validTgt (nM,nTr,nY): bool indication of which outputs are used in each trial
      badFyThresh: threshold for detction of bad Fy entry, in std-dev
      centFy (bool): do we center Fy before computing  *important*  (true)
      nEpochCorrection (int): number of epochs to use a base for correction of number epochs in the sum.
               basically, nEpoch < nEpochCorrection have highly increased Perr, so unlikely to be selected
      minDecisLen (int): int, number of epochs to use as base for distribution of time-based decision points.
                         i.e. decisions at, [1,2,4,...2^n]*exptDistDecis
                    OR: minDecisLen<0 => decision point every abs(minDeicsLen) epochs
      maxDecisLen(int): maximum number of epochs for a decision
      bwdAccumulate (bool): accumulate data backwards from last epoch gathered
      prior_sigma (float,float): prior estimate of the variance and it's equivalent samples weight (sigma,N)

    Returns:
      ssFy (nM,nTr,nDecis,nY): scaled summed scores
      sFy_scale (nM,nTr,nDecis,1): the normalization scaling factor
      Nep (nDecis,): number of epochs included in each score
      nEp,nY (nTrl): the detected number epoch/output for each trial

    Copyright (c) MindAffect B.V. 2018    
    '''
    if Fy is None or Fy.size == 0:
        ssFy = np.zeros(Fy.shape[:-2]+(1,Fy.shape[-1]),dtype=Fy.dtype)
        return ssFy, None, 0, None, None

    # print('args={}'.format(dict(marginalizemodels=marginalizemodels,normSum=normSum,
    #                         detrendFy=detrendFy,centFy=centFy,nEpochCorrection=nEpochCorrection,
    #                         priorsigma=priorsigma)))

    # compress out the model dimension
    # Fyshape = Fy.shape # (nM,nTrl,nEp,nY)
    # if Fy.ndim > 3:
    #     Fy = np.reshape(Fy, (np.prod(Fy.shape[:-2]), )+Fy.shape[-2:]) # ((nM*nTrl), nEp, nY) [ nY x nEp x (nTrl*nM*..)]
    #     if not validTgt is None : # (nM, nTr, nY)
    #         validTgt = np.reshape(validTgt, (np.prod(validTgt.shape[:-1]), ) + validTgt.shape[-1:]) # ((nM*nTrl), nY) [ nY x (nTrl*nM)]
    if Fy.ndim < 3: # ensure has trial dim
        Fy = Fy[np.newaxis, :, :]

    # get info on the number of valid Epochs or Outputs in each trial
    nY, nEp, lastEp, validTgt, validEp, validFy = get_valid_epochs_outputs(Fy)

    if np.max(nEp.ravel()) < 1: # guard no data to analyse
        ssFy = np.zeros(Fy.shape[:-2]+(1,Fy.shape[-1]),dtype=Fy.dtype)
        return ssFy, None, 0, None, None

    # estimate the points at which we may make a decision
    maxnEp = np.max(lastEp.ravel())
    if maxDecisLen > 0:     # Limit the max window size to test
        maxnEp = np.min(maxnEp,maxDecisLen)

    if minDecisLen > 0 and minDecisLen < maxnEp:
        # exp-distributed decision points
        nstep = int(np.ceil(np.log((maxnEp-1) / minDecisLen) / np.log(2)))
        decisIdx = [(2**i)*minDecisLen for i in range(nstep+1)]
        decisIdx = np.minimum(np.maximum(decisIdx, minDecisLen), maxnEp)
    elif minDecisLen < 0:
        # linearly distributed decision points
        decisIdx = np.arange(-minDecisLen-1, maxnEp, -minDecisLen)
        decisIdx = np.append(decisIdx, maxnEp)
    else :
        # default to single decision at max-length
        decisIdx = np.array([maxnEp], dtype=int) 
    #print("decisIdx={}".format(decisIdx))

    # compute the summed scores
    if min(decisIdx) >= Fy.shape[-2]:
        decisIdx = np.array([Fy.shape[-2]-1])
        sFy = np.sum(Fy, -2, keepdims=True)
    else:            
        if bwdAccumulate: # (nM, nTrl, nEp, nY)
            #print("Fy={} lastEp={}".format(Fy.shape,lastEp)) 
            oFy=Fy.copy()
            Fy=Fy.copy()
            for ti in range(Fy.shape[-3]): # time-reverse in the valid data range
                if lastEp[ti]>0:
                    Fy[..., ti, :lastEp[ti]+1, :] = Fy[..., ti, lastEp[ti]::-1, :]
                    validEp[..., :lastEp[ti]+1 ] = validEp[...,lastEp[ti]::-1]

        # For computational efficiency remove data outside the max decision points
        if False and np.max(decisIdx[-1]) < min(Fy.shape[-2]*0.7, Fy.shape[-2]-100): # (nM, nTrl, nEp, nY)
            maxLen = int(np.max(decisIdx[-1]))
            Fy = Fy[..., :maxLen+1, :]

        sFy = np.cumsum(Fy, -2)
        sFy = sFy[..., decisIdx, :]

    # (nTrl, nDecis) (average) number points in each sum at each decision point
    N   = np.mean(np.cumsum(validFy,-2,dtype=np.float32),-1 if validFy.ndim<4 else (-4,-1)) # (nTrl,nEp)
    N   = N[..., decisIdx]

    #N = np.tile(decisIdx[np.newaxis, :], (Fy.shape[-3], 1)) # (nTrl, nDecis) number points in each sum [ nDecis x nTrl ]
    #for ti, nti in enumerate(nEp): # and limit to trial length
    #    N[ti, :] = np.minimum(nti, N[ti, :])

    # compute the noise scale to transform to standardized units
    sigma2, Nsigma = estimate_Fy_noise_variance(Fy, decisIdx=decisIdx, centFy=centFy, detrendFy=detrendFy, priorsigma=priorsigma)

    # same variance for all models
    if marginalizemodels and sigma2.ndim>2:
        sigma2 = np.mean(sigma2,-3)

    # scale = std-deviation over outputs of smoothed summed score at each decison point
    if normSum is not None and normSum > 0:
        # scale up to summed variance
        # \sum_i N(0, sigma)) ~ N(0, sqrt(i)*sigma) 
        sigma2 = sigma2 * np.maximum(N,1).astype(sigma2.dtype) #  (nM,nTrl,nDecis)

    # get the std-dev
    sigma = np.sqrt(sigma2) 

    # correction factor for sampling bias in the std
    # estimation of of the correction factors to
    # from: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    # E[sigma]=c4(n)*sigma -> sigma = E[sigma]/c4(n)
    # where cf is the correction for the sampling bias in the estimator
    cf = 1/c4(np.maximum(N,1)) 
    if nEpochCorrection is not None and nEpochCorrection :
        
        if nEpochCorrection > 0 :
            cf = (1 + np.maximum(nEpochCorrection,1)/np.maximum(N,1))
        else:    
            cf = 1/c4(np.maximum(N,1)/np.maximum(nEpochCorrection, 1)) 

    sigma = sigma * cf.astype(sigma.dtype)
    
    # get the score scaling = std-def = sigma = sqrt(sigma^2)
    sFy_scale = sigma

    # apply the normalization to convert to z-score (i.e. unit-noise)
    sFy_scale[sFy_scale == 0] = 1
    ssFy = sFy/sFy_scale[..., np.newaxis].astype(sFy.dtype)

    # TODO[] : convert back to forward order
    if bwdAccumulate: # (nM, nTrl, nEp, nY)
        #print("Fy={} lastEp={}".format(Fy.shape,lastEp)) 
        pass

    return ssFy, sFy_scale, decisIdx, nEp, nY

def get_valid_epochs_outputs(Fy,validTgt=None):
    """get the number valid epoch and outputs from a zero-padded Fy set

    Args:
        Fy ([np.ndarray]): (nTrl,nEp,nY) the output scores
        validTgt ([np.ndarray]): (nTrl,nY) flag if this output is used in this trial 

    Returns:
        nY ([np.ndarray]): (nTrl,) number valid outputs per trial
        nEp ([np.ndarray]): (nTrl,) number valid epochs per trial
        lastEp (nTrl,): index of the last stimulus in each trial
        validTgt ([np.ndarray]): (nTrl,nY) flag if this output is used in this trial
        validEp (nTrl,nEp): flag if this epoch is a stimulus epoch in this trial
        validFy (np.ndarray): indicator which Fy have valid predictions 
    """
    validFy = Fy != 0    
    validTgt =  np.any(validFy, axis=(-2 if Fy.ndim<4 else (0,-2)) ) # (nTrl,nY)
    nY  = np.sum(validTgt, -1) # number active outputs in this trial (nTrl)

    validEp  =  np.any(validFy, axis=(-1 if Fy.ndim<4 else (0,-1)) ) # (nTrl,nEp)
    nStim = np.sum(validEp, -1) # numer active time-points in this trial
    lastEp = np.zeros((validEp.shape[0],),dtype=int)
    for ti in range(validEp.shape[0]):
         tmp = np.flatnonzero(validEp[ti,:])
         lastEp[ti] = tmp[-1] if tmp.size>0 else 0 # num active epochs/samples
    return nY, nStim, lastEp, validTgt, validEp, validFy

def filter_Fy(Fy, filtLen=None, B=None):
    # apply the temporal smoothing filter to the raw scores
    # i.e. low-pass,  to remove short-term temporal 
    # (anti)-correlations which add a lot of noise to the scale
    # estimation # (nM, nTrl, nEp, nY) # [nY, nEp, nTrl, nM]
    if B is None and (filtLen is None or filtLen == 1 or filtLen == 0):
        return Fy
    
    if B is None:
        filtLen = max(1, min(filtLen, Fy.shape[-2]))
        B = np.ones((filtLen))  # simple moving average filter
    else:
        filtLen = len(B)

    Fyshape = Fy.shape
    if len(Fyshape)>3 :
        Fy = Fy.reshape((-1,)+Fy.shape[-2:])

    #B = np.array((.5,-1.5,2,-1.5,.5))/2 # simple high-pass filter
    # TODO []: how to norm B to achive properties we want?
    B = B /  np.sum(np.abs(B))
    # TODO: good multi-dimensional filter... & check correctness of the convolve
    fFy=np.zeros(Fy.shape) # (nTrl, nEp, nY) [ nY x nEp x nTrl ]
    for ti in range(Fy.shape[-3]): # trials loop
        for yi in range(Fy.shape[-1]): # outputs loop
            fFy[ti, :, yi] = np.convolve(Fy[ti, :, yi], B)[:Fy.shape[-2]] # do the smoothing
    # compute the point weighting
    wB  = np.convolve(np.ones(Fy.shape[-2]), np.abs(B))[:Fy.shape[-2]] # point weighting for each element of fFy (nEp) [ nEp,  ]
    # normalize away the point weighting,  and startup effects
    fFy = fFy/wB[np.newaxis, :, np.newaxis]
    #w=np.ones(Fy.shape[1])
    # compare raw and filtered sum
    #sfFy  = np.cumsum(fFy, -2)[:, decisIdx, :] # (nTrl, nDecis, nY) [ nY x nDecis x nTrl] sfFy(y, T) = \sum_t=0^T fFy(y, t)

    if len(Fyshape)>3 :
        fFy=fFy.reshape(Fyshape)

    return fFy

def estimate_Fy_noise_variance(Fy, decisIdx=None, centFy=True, detrendFy=False, priorsigma=None,  verb=0):
    """Estimate the noise variance for Fy

    Args:
        Fy ([np.ndarray]): (nM,nTr,nEp,nY) the output scores
        decisIdx ([type], optional): [description]. Defaults to None.
        centFy (bool, optional): flag if we should center over outputs before computing variance. Defaults to True.
        detrendFy (bool, optional): flag if we should detrend Fy before computing it's variance. Defaults to True.
        priorsigma ([type], optional): Prior estimate for the variance. Defaults to None.
    returns:
        sigma2 ([np.ndarray]): (nTr, nDecis) estimated variance per sample at each decision point
    """
    validFy = Fy != 0
    nY  = np.sum(np.any(validFy, -2 if Fy.ndim<4 else (0,-2)), -1,dtype=Fy.dtype) # number active outputs in this trial (nM*nTrl) 
    # (nTrl, nEp) (average) number points in each sum at each decision point
    N   = np.mean(np.cumsum(validFy,-2,dtype=Fy.dtype),-1 if validFy.ndim<4 else (-4,-1)) # (nTrl,nEp)

    if decisIdx is None:
        decisIdx = np.array([Fy.shape[-2]-1],dtype=int)

    # TODO [] : generalize by all possible accumul lengths at ass poss start points
    # TODO [] : make more computationally efficient? e.g. no explicit centering
    cFy = Fy.copy()

    if detrendFy:
        # center over time also (to remove any signal)
        muFy_t = np.sum(cFy,-2,keepdims=True)/np.maximum(1,N[:,-1:,np.newaxis],dtype=cFy.dtype)
        cFy = cFy - muFy_t

    # compute the cumulative sum
    scFy = np.cumsum(cFy, -2, dtype=Fy.dtype) # cum-sum from start # (nTr, nEp, nY)

    # remove per-sample offset (if enough outputs to do reliably)
    if centFy and np.any(nY>3):
        # center at each time point, with guard for no active outputs (when mu should == 0)
        muFy_y = np.sum(scFy, -1, keepdims=True) / np.maximum(1,nY[:, np.newaxis, np.newaxis],dtype=scFy.dtype) # mean at each time-point
        scFy[...,nY>3,:,:] = scFy[...,nY>3,:,:] - muFy_y[...,nY>3,:,:]

    # variance of the summed scores over outputs for each time point
    # if independent then this should be a constant slope increase over time.
    var2csFy = np.sum(scFy**2, -1) / np.maximum(1, nY[:,np.newaxis]-1, dtype=scFy.dtype) # (nTr,nEp) var over outputs for each cumsum

    # normalize out the constant slope over time, to get the equivalent slope if 
    # we had N-indenpendent samples  
    nvar2csFy = var2csFy / np.maximum(1, N, dtype=scFy.dtype) #np.arange(1,var2csFy.shape[-1]+1) # ave var per-time-step

    # compute the average of the estimated slopes for each integeration length
    muvar2csFy = np.cumsum(nvar2csFy, -1, dtype=Fy.dtype) / np.maximum(1, np.cumsum(N>0,-1),dtype=scFy.dtype)  # ave per-stime-stamp vars before each time-point
    
    # return the ave-cumsum-slope-of-variance at each decision length
    sigma2 = muvar2csFy[...,decisIdx] # (nTr,nDecis)
    N = N[...,decisIdx]

    # include the prior if needed
    if priorsigma is not None and priorsigma[0] is not None and priorsigma[1]>0 :
        # include the effect of the prior, sigma is weighted combo pior and data
        #  sigma'^2 = 1 / ( N_0/sigma_0^2 + N / sigma^2) 
        #           = sigma_0^2 * sigma^2 / ( N_0 sigma^2 + N * sigma_0 )
        osigma2= sigma2
        sigma2 = ((sigma2*N).astype(sigma2.dtype) + np.array(priorsigma[0]*priorsigma[1],dtype=sigma2.dtype)) / ( N + priorsigma[1] ).astype(sigma2.dtype)
        sigma2[osigma2==0] = 0
        sigma2 = np.maximum(osigma2,sigma2) # sigma2.astype(Fy.dtype)
        #print('sigma2 = {}  prior={} -> {}'.format(np.mean(osigma2),priorsigma,np.mean(sigma2)))
    
    return sigma2, N


def estimate_Fy_noise_variance_old(Fy, decisIdx=None, centFy=True, detrendFy=False, priorsigma=None,  verb=0):
    """Estimate the noise variance for Fy

    Args:
        Fy ([np.ndarray]): (nTr,nEp,nY) the output scores
        decisIdx ([type], optional): [description]. Defaults to None.
        centFy (bool, optional): flag if we should center over outputs before computing variance. Defaults to True.
        detrendFy (bool, optional): flag if we should detrend Fy before computing it's variance. Defaults to True.
        priorsigma ([type], optional): Prior estimate for the variance. Defaults to None.
        
    Returns:
        sigma2 ([np.ndarray]): (nTr, nDecis) estimated variance per sample at each decision point
    """
    if decisIdx is None:
        decisIdx = np.array([Fy.shape[-2]-1],dtype=int)

    sigma2 = np.zeros((Fy.shape[0],decisIdx.size),dtype=Fy.dtype)
    N      = np.zeros((Fy.shape[0],decisIdx.size),dtype=int)
    for ti in range(Fy.shape[0]):
        Fyti = Fy[ti,...].copy()
        # subset to valid entries, samples/outputs
        vi = np.any(Fyti!=0,-1) # active epochs/samples
        Fyti = Fyti[vi,:] # remove the inactive 
        vY = np.any(Fyti!=0,-2) # active outputs
        Fyti = Fyti[:,vY]

        # get the decision points in the compressed representation
        Nti = np.cumsum(vi,-1) # #active epoch at each index == index-1
        decisIdxti = Nti[decisIdx]-1 # index of the decision points in compressed Fy
        N[ti, :] = Nti[decisIdx]

        if Fyti.size == 0:
            sigma2[ti,:]=np.nan # undefined if no active points!
            continue

        if detrendFy and Fyti.shape[-2]>10:
            # center over time-points
            muFy_t = np.mean(Fyti,-2, keepdims=True)
            Fyti = Fyti - muFy_t

        # compute the cumulative sum
        sFy = np.cumsum(Fyti, -2) # cum-sum from start # (nTr, nEp, nY)

        # remove per-example offset (if enough outputs to do reliably)
        if centFy and Fyti.shape[-1]>3:
            # center at each time point, with guard for no active outputs (when mu should == 0)
            muFy_y = np.mean(sFy, -1, keepdims=True) # mean at each time-point
            sFy = sFy - muFy_y

        # variance of the summed scores over outputs for each time point
        # if independent then this should be a constant slope increase over time.
        var2sFy = np.sum(sFy**2, -1) / np.maximum(.1,sFy.shape[-1]-1) # (nTr,nEp) var over outputs for each cumsum

        # normalize out the constant slope over time, to get the equivalent slope if 
        # we had N-indenpendent samples  
        nvar2sFy = var2sFy / np.arange(1,var2sFy.shape[-1]+1) # ave var per-time-step

        # compute the average of the estimated slopes for each integeration length
        muvar2sFy = np.cumsum(nvar2sFy, -1) / np.arange(1,var2sFy.shape[-1]+1) # np.arange(1,nvar2csFy.shape[-1]+1) # ave per-stime-stamp vars before each time-point
    
        # return the ave-cumsum-var-slope at each decision length
        sigma2[ti,:] = muvar2sFy[decisIdxti] # (nTr,nDecis)
        sigma2[ti,decisIdxti<0]=np.nan # undefined if no active points!
        

    # include the prior if needed
    if priorsigma is not None and priorsigma[0]>0 :
        # include the effect of the prior, sigma is weighted combo pior and data
        #  sigma'^2 = 1 / ( N_0/sigma_0^2 + N / sigma^2) 
        #           = sigma_0^2 * sigma^2 / ( N_0 sigma^2 + N * sigma_0 )
        osigma2= np.mean(sigma2)
        sigma2 = (sigma2*N + priorsigma[0]*priorsigma[1]) / ( N + priorsigma[1] )
        #print('sigma2 = {}  prior={} -> {}'.format(osigma2,priorsigma,np.mean(sigma2)))
    
    return sigma2, N


#@function
def c4(n):
    # correction factor for bias in the standard deviation  
    # from: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    n = np.maximum(1.0, n)
    cf = 1 - 1 / 4*(n ** - 1) - 7 / 32*(n ** - 2) - 19 / 128*(n ** - 3)
    return cf

def plot_normalizedScores(Fy, ssFy, scale_sFy, decisIdx):
    '''plot the normalized score and raw summed score for comparsion '''
    sFy = np.cumsum(Fy, -2) # (nEp, nY)
    stdsFy = np.std(sFy,-1,keepdims=True)
    #stdsFy = np.sum((sFy-np.mean(sFy,-1,keepdims=True))**2, -1) /sFy.shape[-1] / np.arange(1,sFy.shape[-2]+1)
    #stdsFy = np.cumsum(stdsFy) / np.arange(1,stdsFy.shape[-1]+1)
    print('sFy={}'.format(sFy.shape))
    import matplotlib.pyplot as plt
    plt.clf()
    plt.subplot(211)
    plt.cla()
    plt.plot(decisIdx, sFy[decisIdx, :]) 
    plt.plot(decisIdx, scale_sFy, 'k', label='scale_sFy')
    plt.plot(stdsFy, 'k.', label='std cumsum(Fy)')
    plt.legend()
    plt.title('Non-normalized sum - scale')
    plt.grid('on')
    plt.subplot(212)
    plt.cla()
    plt.plot(decisIdx, ssFy)
    stdssFy = np.std(ssFy,-1,keepdims=True)
    plt.plot(decisIdx, stdssFy[decisIdx,:],'k.',label='std ssFy')
    plt.title("scaled sum")
    plt.ylabel('ssFy'); plt.xlabel('epoch')
    plt.grid('on')
    plt.show()


#@function
def mktestFy(nY=10, nM=1, nEp=360, nTrl=100, sigstr=.5, startupNoisefrac=.25, offsetstr=0, trlenfrac=.25):
    import numpy as np
    noise = np.random.standard_normal((nM, nTrl, nEp, nY))
    noise = noise - np.mean(noise,axis=(-1,-2),keepdims=True)
    noise = noise / np.std(noise,axis=(-1,-2),keepdims=True)
    # add an offset to the scores...
    offset = offsetstr*np.random.standard_normal((noise.shape[-2])) # (nEp)
    noise = noise+offset[:, np.newaxis] # (nM, nTrl, nEp, nY) [ nYxnEpxnTrlxnM ]

    sigamp = sigstr*np.ones((noise.shape[-2])) # (nEp)

    # make variable length trials,  and per-trial offset
    nEp = np.zeros((noise.shape[-3],), dtype=int)
    for ti in range(noise.shape[-3]):
        # trial offset
        mu = np.random.standard_normal()
        noise[:,  ti, :, :] = noise[:, ti, :, :] + mu
        # trial end time
        nEp[ti] = noise.shape[-2]*(np.random.uniform()*trlenfrac + (1-trlenfrac))
        noise[:, ti, nEp[ti]:, :] = 0

    # no signal at the start of the trial
    startupNoise_samp = int(noise.shape[1]*startupNoisefrac) if startupNoisefrac<1 else startupNoisefrac
    sigamp[:startupNoise_samp] = 0
    noise[:,:,:startupNoise_samp,:]=0

    # measure is sig + noise
    Fy = noise
    Fy[0, :, :, 0] = Fy[0, :, :, 0] + sigamp[np.newaxis, :] # (nM, nTrl, nEp, nY) [nEp x nTrl]

    return Fy, nEp

def testcase():
    detrendFy=False
    centFy=True
    normSum=1
    nEpochCorrection=50

    from normalizeOutputScores import mktestFy,  normalizeOutputScores
    import matplotlib.pyplot as plt
    Fy, nEp = mktestFy(sigstr=1,nM=2) #(nM, nTrl, nEp, nY)
    #Fy = Fy[0,...] # (nTrl,nEp,nY)
    # Introduce temporal and spatial sparsity like real data
    Fy = Fy * (np.random.standard_normal((1,Fy.shape[-2],Fy.shape[-1]))>0).astype(np.float)
    Fy[...,:20,:] = 0 # block zero at start

    # different 50% active for different outputs
    for i in range(Fy.shape[-1]):
        Fy[..., (i%3)::3, i] = 0 # only 50% active
        Fy[..., (i+1%3)::3, i] = 0 # only 50% active
    oFy=Fy.copy()

    print("Fy={}".format(Fy.shape))

    # test all at once, vs. incremental computation.
    step=50
    ssFy_all, scale_sFy_all, decisIdx, nEp, nY = normalizeOutputScores(Fy, minDecisLen=step, normSum=normSum, nEpochCorrection=nEpochCorrection, centFy=centFy, detrendFy=detrendFy, priorsigma=(0,0))
    for i,len in enumerate(decisIdx):
        ssFyL, scale_sFyl, _, _, _ = normalizeOutputScores(Fy[:,:,:len,...], minDecisLen=9999, normSum=normSum, nEpochCorrection=nEpochCorrection, centFy=centFy, detrendFy=detrendFy, bwdAccumulate=False, priorsigma=(0,0))
        print("{}) all={} once={}".format(len,scale_sFy_all[0,i],scale_sFyl[0,0]))


    priorsigma=(1,0)
    ssFy, scale_sFy, decisIdx, nEp, nY = normalizeOutputScores(Fy, minDecisLen=-1, normSum=normSum, nEpochCorrection=nEpochCorrection, centFy=centFy, detrendFy=detrendFy, priorsigma=priorsigma)
    print('ssFy={} scale_sFy={}'.format(ssFy.shape,scale_sFy.shape))
    #%matplotlib
    plt.figure(1)
    plot_normalizedScores(oFy[0,0,...],ssFy[0,0,:,:],scale_sFy[0,:],decisIdx)


    # visualize all trials true-target normalized scores
    priorsigma=(1,1e6)
    ssFy, scale_sFy, decisIdx, nEp, nY = normalizeOutputScores(Fy, minDecisLen=-1, normSum=normSum, nEpochCorrection=nEpochCorrection, centFy=centFy, detrendFy=detrendFy)
    print('ssFy={} scale_sFy={}'.format(ssFy.shape,scale_sFy.shape))
    #%matplotlib
    plt.figure(1)
    plot_normalizedScores(oFy[0,0,...],ssFy[0,0,:,:],scale_sFy[0,:],decisIdx)

    # introduce temporal correlations and visualize
    Fy = filter_Fy(Fy, filtLen=10)
    ssFy, scale_sFy, decisIdx, nEp, nY = normalizeOutputScores(Fy, minDecisLen=-1, normSum=normSum, nEpochCorrection=nEpochCorrection, centFy=centFy, detrendFy=detrendFy)
    print('pre-filtered ssFy={}'.format(ssFy.shape))
    #%matplotlib
    plt.figure(1)
    plt.suptitle("filtered {}".format(10))
    plot_normalizedScores(Fy[0,0,:,:],ssFy[0,0,:,:],scale_sFy[0,:],decisIdx)

    # introduce temporal correlations and visualize
    Fy = filter_Fy(Fy, B=np.array([1,0,0,-1]))
    ssFy, scale_sFy, decisIdx, nEp, nY = normalizeOutputScores(Fy, minDecisLen=-1, nEpochCorrection=nEpochCorrection, centFy=centFy, detrendFy=detrendFy)
    print('pre-filtered ssFy={}'.format(ssFy.shape))
    #%matplotlib
    plt.figure(1)
    plt.suptitle('anti-correlated')
    plot_normalizedScores(Fy[0,0,:,:],ssFy[0,0,:,:],scale_sFy[0,:],decisIdx)

if __name__=="__main__":
    testcase()
