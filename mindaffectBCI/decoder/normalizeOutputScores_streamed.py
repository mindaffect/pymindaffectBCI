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

import numpy as np
def normalizeOutputScores_streamed(Fy, validTgt=None, badFyThresh=6,
                                   centFy=True, blockSize=30, filtLen=5):
    '''
    normalize the raw output scores to feed into the Perr computation
    Inputs:
      Fy   - (nM,nTr,nEp,nY):float
      validTgt - (nM,nTr,nY):bool indication of which outputs are used in each trial
          badFyThresh - threshold for detction of bad Fy entry, in std-dev
          centFy   - bool, do we center Fy before computing  *important*  (true)
          blockSize  - maximum number of epochs single normalized sum
    Outputs:
      ssFy - (nM,nTr,nDecis,nY):float  scaled summed scores
      scale_sFy - (nM,nTr,nDecis):float scaling factor for each decision point
      N - (nDecis):int the number of elements summed in each block
    Copyright (c) MindAffect B.V. 2018
    
    '''
    normSum = True
    if Fy is None:
        return None, None, None, None, None

    # compress out the model dimension
    Fyshape = Fy.shape # (nM,nTrl,nEp,nY)
    if Fy.ndim > 3:
        Fy = np.reshape(Fy, (np.prod(Fy.shape[:-2]), )+Fy.shape[-2:]) # ((nM*nTrl), nEp, nY)
        if not validTgt is None : # (nM, nTr, nY)
            validTgt = np.reshape(validTgt, (np.prod(validTgt.shape[:-1]), ) + validTgt.shape[-1:]) # ((nM*nTrl), nY)
    if Fy.ndim < 3: # ensure has trial dim
        Fy = Fy[np.newaxis, :, :]

    # get info on the number of valid Epochs or Outputs in each trial
    if validTgt is None : # (nM, nTr, nEp, nY)
        validTgt = np.any(Fy != 0, -2) # (nM, nTr, nY) [ nY x nTrl ]

    # loop over trials
    nBlk = Fy.shape[1]//blockSize
    ssFy = np.zeros((Fy.shape[0], nBlk, Fy.shape[2]), dtype='float32')
    scale_sFy = np.zeros((Fy.shape[0], nBlk), dtype='float32')
    N = np.minimum(np.arange(1, nBlk+1)*blockSize, Fy.shape[1])
    for ti in range(Fy.shape[-3]):
        # N.B. sub-slice in steps as mixing integer with logical idexing is ackward
        Fyti = Fy[ti,...]
        # get num active epochs
        tmp = np.flatnonzero(np.any(Fy[ti, :, :] != 0, axis=-1))
        nEp = tmp[-1] if tmp.size > 0 else 0 # num active epochs/samples
        # limit to valid epochs
        Fyti = Fyti[:nEp, :]
        # limit to valid outputs
        validY = validTgt[ti, :]
        Fyti = Fyti[..., validY]
        
        # get the  start/end of each block
        decisIdx = np.arange(0, nEp-(blockSize//2), blockSize)
        decisIdx = np.append(decisIdx, nEp)

        # compute the noise-scale *incrementally* for each block
        sigma2 = None
        for bi in range(len(decisIdx)-1):
            Fytibi = Fyti[decisIdx[bi]:decisIdx[bi+1], ...]
            sigma2 = incremental_estimate_noise_variance(Fytibi, sigma2, filtLen)
            
            # apply the normalization and compute the summed score
            scale_sFy[ti,bi] = np.sqrt(Fytibi.shape[0]*sigma2)
            ssFy[ti,bi,validY] = np.sum(Fytibi,0) / scale_sFy[ti,bi]            

    if len(Fyshape)>3 : # convert back to have model dimension
        ssFy      = np.reshape(ssFy, (Fyshape[:-2]+ssFy.shape[-2:]))

    return (ssFy, scale_sFy, N)


def incremental_estimate_noise_variance(Fy, old_sigma2, filtLen):
    # TODO []: pre-filter Fy?
    cFy = Fy - np.mean(Fy, -1, keepdims=True) # per-output offset centered
    cFy = cFy - np.mean(cFy, 0, keepdims=True) # per-time point offset centered
    sigma2 = np.sum(cFy**2)/cFy.size # average residual
    # accumulate over calls
    if old_sigma2:
        sigma2 = old_sigma2*.5 + .5*sigma2
    return sigma2


def testcase(Fy=None):
    from normalizeOutputScores import mktestFy,  normalizeOutputScores, plot_normalizedScores
    from normalizeOutputScores_streamed import normalizeOutputScores_streamed, incremental_estimate_noise_variance
    from zscore2Ptgt_softmax import softmax
    import matplotlib.pyplot as plt
    if Fy is None:
        Fy, _ = mktestFy() #(nM, nTrl, nEp, nY)
        Fy = Fy[0, ...] # strip model dim
    print("Fy={}".format(Fy.shape))
    # visualize all trials true-target normalized scores
    blk_len = 15
    zFy, scale_sFy, N = normalizeOutputScores_streamed(Fy, blockSize=blk_len)
    print('zFy={}'.format(zFy.shape))
    sumcorr = np.sqrt(np.arange(zFy.shape[-2])+1) # correction for block sums
    plt.figure(2);plt.clf()
    plot_normalizedScores(Fy[0,:,:],np.cumsum(zFy[0,:,:],-2)/sumcorr[:,np.newaxis],np.cumsum(scale_sFy[0,:],-1)/sumcorr,N)

    # compare: sum->pVal with pval->combine
    # sum -> pval
    ssFy = np.cumsum(zFy,-2) /sumcorr[:,np.newaxis] # sum
    Ptgt_ssFy = softmax(ssFy)

    # pval -> combine
    # N.B. not numerically robust
    Ptgt_zFy = softmax(zFy)
    sPtgt_zFy = np.cumprod(Ptgt_zFy, -2) # combine
    sPtgt_zFy0 = sPtgt_zFy / np.sum(sPtgt_zFy, -1, keepdims=True) # normalize

    # alt computation,  softmax sum log(P)
    sPtgt_zFy =softmax(np.cumsum(np.log(Ptgt_zFy),-2))

    # incremental computation
    sPtgt_zFy = Ptgt_zFy.copy()
    for di in range(1,Ptgt_zFy.shape[-2]):
        #       prior            * likeihood
        postlikelihood = sPtgt_zFy[..., di-1, :] * sPtgt_zFy[..., di,:]
        #          post-likelihood / Z
        sPtgt_zFy[..., di,:] = postlikelihood / np.sum(postlikelihood,-1,keepdims=True)


    # N.B. correct way to do this is to integerate over the unit variance noise in the softmax? right?
    # Noise corrected computation, taking into account the error in Fy, which  scales as sqrt(N)
    # N.B. this is equivalent to having the scale in the softmax go as sqrt(N)
    if True:
        sPtgt_zFy =softmax(np.cumsum(np.log(Ptgt_zFy),-2)) # no summed error compenstation
    else:
        sPtgt_zFy =softmax(np.cumsum(np.log(Ptgt_zFy),-2)/sumcorr[:,np.newaxis]) # sum compensation

    plt.figure(3)
    plt.clf()
    p1=plt.subplot(311)
    plt.cla()
    plt.plot(ssFy[0,:,:])
    plt.grid(True)
    plt.title("summed scores")
    plt.subplot(312,sharex=p1)
    plt.cla()
    plt.plot(Ptgt_ssFy[0,:,:])
    plt.grid(True)
    plt.title("integerate->pVal")
    plt.subplot(313,sharex=p1)
    plt.cla()
    plt.title("pVal->integerate")
    plt.plot(sPtgt_zFy[0,:,:])
    plt.grid(True)
    plt.show()


def compute_pval_curve(X,pval):
    thresh=np.zeros((X.shape[1]))
    for t in range(X.shape[1]):
        Xt=X[:,t,:]
        for tst in np.linspace(np.min(Xt.ravel()),np.max(Xt.ravel()),40):
            if np.sum(Xt.ravel()>tst)/Xt.size < pval:
                thresh[t]=tst
                break
    return thresh

def compute_softmax_curve(X,pval,scale=3):
    thresh=np.ones((X.shape[1]))*np.nan
    for t in range(X.shape[1]):
        Xt=X[:,t,:]
        Z = np.sum(np.exp(scale*Xt),-1)
        for tst in np.linspace(np.min(Xt.ravel()),np.max(Xt.ravel())*2,40):
            softmax = np.exp(scale*tst) / (np.exp(scale*tst)+Z) # p-(tst==max)
            if np.mean(softmax.ravel()) > 1-pval:
                thresh[t]=tst
                break
    return thresh

def softmax_nout_corr(n):
    ''' approximate correction factor for probabilities out of soft-max to correct for number of outputs'''
    return min(2.45,1.25+np.log2(n)/5.5)

def softmax_vs_nout(nruns,nout,sscale):
    F = np.random.randn(nruns,np.max(nout)) # noise
    F0= np.random.randn(nruns,1) # signal noise
    Ftrue = np.linspace(0,5,50) # signal
    Ptrue = np.zeros((Ftrue.size,len(nout)))
    Pest  = np.zeros((Ftrue.size,len(nout)))
    for ni,nnoise in enumerate(nout):
        scale = sscale*softmax_nout_corr(nnoise)
        sexpF = np.sum(np.exp(scale*F[:,:nnoise]),-1) # sum exp-noise scores
        for ampi,sigamp in enumerate(Ftrue):
            # empherical estimate signal is biggest
            Ptrue[ampi,ni] = np.mean(np.all(sigamp+F0 > F[:,:nnoise],-1),0)
            # softmax estimate signal is biggest
            expF0 = np.exp( scale*(sigamp+F0) )
            softmax = expF0  /  (expF0 + sexpF[:,np.newaxis])
            # N.B. median so test the most-likely p-val est is similar to the true..
            Pest[ampi,ni] = np.median( softmax )
    # visualize the result....
    import matplotlib.pyplot as plt
    plt.clf();plt.subplot(221);
    plt.plot(Ptrue);plt.grid(True);plt.ylim([.8,1]);plt.xlabel('sig-amp');plt.subplot(222);
    plt.plot(Pest);plt.grid(True);plt.ylim([.8,1]);plt.xlabel('sig-amp'); plt.subplot(212);
    plt.plot(Ptrue-Pest);plt.legend(nout);plt.grid(True)
    return (Ptrue,Pest)

if __name__=='__main__':
    Fy=None
    # load plos-one dataset
    if True:
        from datasets import get_dataset
        from analyse_datasets import analyse_dataset
        l,f,_ = get_dataset('plos_one')
        X,Y,coords=l(f[0])
        _,_,Fy,_=analyse_dataset(X,Y,coords,evtlabs=('re','fe'),model='cca',rank=1)

    testcase(Fy)
