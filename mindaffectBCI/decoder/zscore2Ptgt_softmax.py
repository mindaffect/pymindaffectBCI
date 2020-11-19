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

from mindaffectBCI.decoder.normalizeOutputScores import mktestFy
import numpy as np

def zscore2Ptgt_softmax(f, softmaxscale:float=2, prior:np.ndarray=None, validTgt=None, marginalizemodels:bool=True, marginalizedecis:bool=False, peroutputmodel:bool=True):
    '''
    convert normalized output scores into target probabilities

    Args:
     f (nM,nTrl,nDecis,nY):  normalized accumulated scores
     softmaxscale (float, optional): slope to scale from scores to probabilities.  Defaults to 2.
     validtgtTrl (bool nM,nTrl,nY): which targets are valid in which trials 
     peroutputmode (bool, optional): assume if nM==nY that we have 1 model per output.  Defaults to True.
     prior ([nM,nTrl,nDecis,nY]): prior probabilities over the different dimesions of f. e.g. if want a prior over Models should be shape (nM,1,1,1), for a prior over outputs (nY,). Defaults to None. 

    Returns:
     Ptgt (nTrl,nY): target probability for each trial (if marginalizemodels and marginalizedecis is True)
    '''

    # fix the nuisance parameters to get per-output per trial score
    # pick the model

    # make 4d->3d for simplicity
    if f.ndim > 3:
        origf = f.copy()
        if f.shape[0] == 1:
            f = f[0,...]
        else:
            if f.shape[0]==f.shape[-1] and peroutputmodel:
                # WARNING: assume that have unique model for each output..
                # WARNING: f must have same mean and scale for this to be valid!!
                f=np.zeros(f.shape[1:])
                for mi in range(origf.shape[0]):
                    f[..., mi]=origf[mi, :, :, mi]

    elif f.ndim == 2:
        f = f[np.newaxis, :]

    elif f.ndim == 1:
        f = f[np.newaxis, np.newaxis, :]
    # Now : f= ((nM,)nTrl,nDecis,nY)

    # identify the valid targets for each trial
    if validTgt is None: # get which outputs are used in which trials..
        validTgt = np.any(f != 0, axis=(-4,-2) if f.ndim>3 else -2) # (nTrl,nY)
    else:
        # ensure valid target info is in the shape we expect
        if validTgt.ndim == 3: #(nM,nTrl,nY)
            validTgt = np.any(validTgt,0)
        elif validTgt.ndim == 1: # (nY,)
            validTgt = validTgt[np.newaxis, :] 

    noutcorr = softmax_nout_corr(np.sum(validTgt,-1)) # (nTrl,)
    softmaxscale = softmaxscale * noutcorr #(nTrl,)

    # include the scaling
    # N.B. horrible np-hack to make softmaxscale the right size, by adding 1 dims to end
    f = f * softmaxscale.reshape((-1,1,1)).astype(f.dtype)

    # inlude the prior
    if prior is not None:
        logprior = np.log(np.maximum(prior,1e-8))
        f = f + logprior.astype(f.dtype)
    
    # marginalization over multiple models
    # get the axis to marginalize over
    marginalize_axis=[]
    if marginalizemodels and f.ndim>3 and f.shape[0] > 1: # marginalize the models axis
        marginalize_axis.append(-4)
    if marginalizedecis and f.shape[-2] > 1: # marginalize the decision points axis
        marginalize_axis.append(-2)
    marginalize_axis=tuple(marginalize_axis)

    if marginalize_axis:
        if False: # marginalize then softmax
            f = marginalize_scores(f, marginalize_axis, keepdims=False)

            # get the prob each output conditioned on the model
            Ptgt = softmax(f,axis=-1,validTgt=validTgt) # ((nM,)nTrl,nDecis,nY)
        else: # softmax then sum

            Ptgt = softmax(f,axis=(-1,)+marginalize_axis, validTgt=validTgt)
            Ptgt = np.sum(Ptgt, axis=marginalize_axis, keepdims=True)
    else:
        Ptgt = softmax(f,axis=(-1,)+marginalize_axis, validTgt=validTgt)


    if any(np.isnan(Ptgt.ravel())):
        if not all(np.isnan(Ptgt.ravel())):
            print('Error NaNs in target probabilities')
        Ptgt[:] = 0
    return Ptgt


def entropy(p,axis=-1):
    ent = np.sum(p*np.log(np.maximum(p, 1e-08)), axis) # / -np.log(Ptgtepmdl.shape[-1]) # (nDecis)
    return ent


def softmax(f, axis=-1, validTgt=None):
    ''' simple softmax over final dim of input array, with compensation for missing inputs with validTgt mask. '''
    p = np.exp( f - np.max(f, axis, keepdims=True) ) # (nTrl,nDecis,nY) [ nY x nDecis x nTrl ]
    # cancel out the missing outputs
    if validTgt is not None and not all(validTgt.ravel()):
        p = p * validTgt[..., np.newaxis, :].astype(f.dtype)
    # convert to softmax, with guard for div by zero
    p = p / np.maximum(np.sum(p, axis, keepdims=True), 1e-6, dtype=f.dtype)
    return p

def softmax_nout_corr(n):
    ''' approximate correction factor for probabilities out of soft-max to correct for number of outputs'''
    #return np.minimum(2.45,1.25+np.log2(np.maximum(1,n))/5.5)/2.45
    return np.ones(n.shape) #np.minimum(2.45,1.25+np.log2(np.maximum(1,n))/5.5)/2.45

def marginalize_scores(f, axis, prior=None, keepdims=False):
    """marginalize the output scores to remove nuisance parameters, e.g. decis-pts, models.

    Args:
        f (np.ndarray (nModel,nTrial,nDecisPts,nOutput)): the scores
        axis ([listInt]): the axis of f to marginalize over
        prior ([type], optional): prior over the dimesions of f. Defaults to None.
        keepdims (bool, optional): flag if we keep or compress the dims of f.  Defaults to False.

    Returns:
        f: (np.ndarray): the marginalized f scores
    """
    if f.size == 0:
        return f

    if prior is not None:
        logprior = np.log(np.maximum(prior,1e-8))
        f = f + logprior.astype(f.dtype)

    maxf = np.max(f, axis=axis, keepdims=True) # remove for numerical robustness
    z = np.exp( f - maxf ).astype(f.dtype) # non-normalized Ptgt
    p = z / np.sum(z, axis, keepdims=True) # normalized Ptgt
    #p = softmax(f,axis=axis)
    f = np.sum(f * p, axis, keepdims=keepdims) # marginalized score

    return f

def calibrate_softmaxscale(f, validTgt=None, scales=(.01,.02,.05,.1,.2,.3,.4,.5,1,1.5,2,2.5,3,3.5,4,5,7,10,15,20,30), MINP=.01, marginalizemodels=True, marginalizedecis=False, eta=.05, error_weight=2):
    '''
    attempt to calibrate the scale for a softmax decoder to return calibrated probabilities

    Args:
     f ((nM,)nTrl,nDecis,nY): normalized accumulated scores]
     validTgt(bool (nM,)nTrl,nY): which targets are valid in which trials
     scales (list:int): set of possible soft-max scales to try
     MINP (float): minimium P-value.  We clip the true-target p-val to this level as a way
            of forcing the fit to concentrate on getting the p-val right when high, rather than
            over penalizing when it's wrong

    Returns:
     softmaxscale (float): slope for softmax to return calibrated probabilities
    '''
    if validTgt is None: # get which outputs are used in which trials..
        validTgt = np.any(f != 0, axis=(-4,-2) if f.ndim>3 else -2) # (nTrl,nY)
    elif validTgt.ndim == 1:
        validTgt = validTgt[np.newaxis, :]

    # remove trials with no-true-label info
    axis = (-3,-1) if f.ndim>3 else (-1)
    keep = np.any(f[..., 0], axis) # [ nTrl ]
    if not np.all(keep):
        f = f[..., keep, :, :]
        if validTgt.shape[0] > 1 :
            validTgt = validTgt[..., keep, :]
 
    # include the nout correction on a per-trial basis
    noutcorr = softmax_nout_corr(np.sum(validTgt,1)) # (nTrl,)

    # simply try a load of scales - as input are normalized shouldn't need more than this
    Ed = np.zeros(len(scales),)
    for i,s in enumerate(scales):
        # apply the soft-max with this scaling
        Ptgt = zscore2Ptgt_softmax(f,softmaxscale=s,validTgt=validTgt,marginalizemodels=marginalizemodels,marginalizedecis=marginalizedecis)
        # Compute the loss = cross-entropy.  
        # As Y==0 is *always* the true-class, this becomes simply sum log this class 
        # Compute the loss = cross-entropy, with soft targets
        # identify trials where prediction is incorrect
        Yerr = Ptgt[...,0:1] < np.max(Ptgt[...,1:],-1,keepdims=True)
        wght = np.ones(Yerr.shape,dtype=Ptgt.dtype)
        wght[Yerr] = error_weight # relative weighting for these errors
        if False:
            # As Y==0 is *always* the true-class, this becomes simply sum log this class 
            Ed[i] = (1-eta)*np.sum(-wght*np.log(np.maximum(Ptgt[...,0:1],MINP))) + (eta/Ptgt.shape[-1])*np.sum(-wght*np.log(np.maximum(Ptgt[...,1:],MINP)))
        else:
            Ed[i] = -np.sum(np.log(np.maximum(Ptgt[...,0:1],1e-5)))

        #print("{}) scale={} Ed={}".format(i,s,Ed[i]))
    # use the max-entropy scale
    mini = np.argmin(Ed)
    softmaxscale = scales[mini]
    print("softmaxscale={}".format(softmaxscale))
    return softmaxscale

def mkTestFy(nY,nM,nEp,nTrl,sigstr,startup_lag):
    np.random.seed(0)
    noise = np.random.standard_normal((nM,nTrl,nEp,nY))
    noise = noise - np.mean(noise.ravel())
    noise = noise / np.std(noise.ravel())

    sigamp=sigstr*np.ones(noise.shape[-2]) # [ nEp ]
    # no signal ast the start of the trial
    startuplag_samp=int(nEp*startup_lag)
    sigamp[:startuplag_samp]=0
    Fy = np.copy(noise)
    # add the signal
    Fy[0, :, :, 0] = Fy[0, :, :, 0] + sigamp
    #print("Fy={}".format(Fy))
    return Fy, noise, sigamp


def visPtgt(Fy, normSum, centFy, detrendFy, bwdAccumulate,
            marginalizemodels, marginalizedecis, 
            nEpochCorrection, priorweight, minDecisLen=-1):
    import numpy as np
    #print("{}".format(locals()))
    
    from mindaffectBCI.decoder.normalizeOutputScores import normalizeOutputScores, estimate_Fy_noise_variance
    sigma0, _ = estimate_Fy_noise_variance(Fy, priorsigma=None)
    #print('Sigma0{} = {}'.format(self.sigma0_.shape,self.sigma0_))
    sigma0 = np.nanmedian(sigma0)  # ave
    print('Sigma0 = {}'.format(sigma0))

    sFy=np.cumsum(Fy,-2)
    ssFy,scale_sFy,N,_,_=normalizeOutputScores(Fy.copy(),minDecisLen=-1, nEpochCorrection=nEpochCorrection, 
                                normSum=normSum, detrendFy=detrendFy, centFy=centFy, bwdAccumulate=False,
                                marginalizemodels=marginalizemodels, priorsigma=(sigma0,priorweight))
    softmaxscale = calibrate_softmaxscale(ssFy,marginalizemodels=marginalizemodels)

    ssFy,scale_sFy,N,_,_=normalizeOutputScores(Fy.copy(),minDecisLen=-1, nEpochCorrection=nEpochCorrection, 
                                normSum=normSum, detrendFy=detrendFy, centFy=centFy, bwdAccumulate=False,
                                marginalizemodels=marginalizemodels, priorsigma=(sigma0,priorweight))
    #print('ssFy={}'.format(ssFy.shape))
    from mindaffectBCI.decoder.zscore2Ptgt_softmax import zscore2Ptgt_softmax, softmax
    smax = softmax(ssFy*softmaxscale,axis=((-4,-1) if ssFy.ndim>3 else -1))
    #print("{}".format(smax.shape))
    Ptgt=zscore2Ptgt_softmax(ssFy, marginalizemodels=marginalizemodels, marginalizedecis=False, softmaxscale=softmaxscale) # (nTrl,nEp,nY)
    if Ptgt.ndim>3 and Ptgt.shape[0]==1: # strip singlenton model dim
        Ptgt=Ptgt[0,...]
    #print("Ptgt={}".format(Ptgt.shape))
    import matplotlib.pyplot as plt
    plt.clf()
    tri=min(1,Fy.shape[-3]-1)
    nM=Fy.shape[-4] if Fy.ndim>3 else 1
    for mi in range(nM):
        sFyi  = sFy[mi,tri,:,:] if sFy.ndim>3 else sFy[tri,:,:]
        ssFyi = ssFy[mi,tri,:,:] if ssFy.ndim>3 else ssFy[tri,:,:]
        smaxi = smax[mi,tri,:,:] if smax.ndim>3 else smax[tri,:,:]

        if mi==0 :
            a = plt.subplot(4,nM,0*nM+mi+1)
            xax=a
            yax0=a
        else:
            a = plt.subplot(4,nM,0*nM+mi+1,sharex=xax, sharey=yax0)
        plt.cla()
        plt.plot(sFyi)
        plt.plot(scale_sFy[tri,:],'k')
        if mi==0 : plt.ylabel('sFy')
        plt.ylim((np.min(sFy.ravel()),np.max(sFy.ravel())))
        plt.grid()
    
        if mi==0 :
            a=plt.subplot(4,nM,1*nM+mi+1, sharex=xax)
            yax1=a
        else:
            a=plt.subplot(4,nM,1*nM+mi+1, sharex=xax, sharey=yax1)
        plt.cla()
        plt.plot(ssFyi)
        plt.ylim((np.min(ssFy.ravel()),np.max(ssFy.ravel())))
        if mi==0 : plt.ylabel('ssFy')
        plt.grid()
        
        if mi==0:
            a=plt.subplot(4,nM,2*nM+mi+1, sharex=xax)
            yax2=a
        else:
            a=plt.subplot(4,nM,2*nM+mi+1, sharex=xax, sharey=yax2)
        plt.cla()
        plt.plot(smaxi)
        plt.ylim((np.min(smax.ravel()),np.max(smax.ravel())))
        if mi==0: plt.ylabel('softmax')
        plt.grid()
        if Ptgt.ndim>3 :
            if mi==0:
                a=plt.subplot(4,nM,3*nM+mi+1, sharex=xax)
                yax2=a
            else:
                a=plt.subplot(4,nM,3*nM+mi+1, sharex=xax, sharey=yax2)
            plt.cla()
            plt.plot(Ptgt[mi,tri,:,:])
            plt.ylim((0,1))
            if mi==0: plt.ylabel('Ptgt')
            plt.grid()

    if Ptgt.ndim<4 or Ptgt.shape[0]==1:
        plt.subplot(414);plt.cla()    
        plt.plot(Ptgt[tri,...])
        plt.ylabel('Ptgt')
        plt.grid()

    plt.show(block=False)
    plt.tight_layout()

    # make the decoding curve
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised, plot_decoding_curve
    #ssFy,scale_sFy,N,_,_=normalizeOutputScores(Fy,minDecisLen=-1, nEpochCorrection=nEpochCorrection, normSum=normSum, marginalizemodels=marginalizemodels)
    #softmaxscale = calibrate_softmaxscale(ssFy,marginalizemodels=marginalizemodels)
    (dc) = decodingCurveSupervised(Fy,nInt=(100,50),
                    marginalizemodels=marginalizemodels, marginalizedecis=marginalizedecis, minDecisLen=minDecisLen,
                    normSum=normSum, detrendFy=detrendFy, centFy=centFy, bwdAccumulate=bwdAccumulate, 
                    nEpochCorrection=nEpochCorrection,
                    softmaxscale=softmaxscale,priorsigma=(sigma0,priorweight))
    plt.figure()
    plot_decoding_curve(*dc)
    plt.show()

def testcase(nY=10, nM=4, nEp=340, nTrl=500, sigstr=.4, normSum=True, centFy=True, detrendFy=True, marginalizemodels=True, marginalizedecis=False, bwdAccumulate=False, nEpochCorrection=20, priorweight=1e2, startup_lag=.1):

    # mk the test dataset
    Fy, noise, sigamp = mkTestFy(nY,nM,nEp,nTrl,sigstr,startup_lag)

    # visualize it
    visPtgt(Fy,normSum,centFy,detrendFy,bwdAccumulate,marginalizemodels,marginalizedecis,nEpochCorrection,priorweight)



if __name__=="__main__":
    if True:
        Fy, noise, sigamp = mkTestFy(nY=10,nM=1,nEp=640,nTrl=200,sigstr=.15,startup_lag=.3)
    else:
        import pickle
        stopping=pickle.load(open('stopping.pk','rb'))
        Fy = stopping['Fy']
        Y=stopping['Y']
        keep = np.any(Fy>0,axis=(-2,-1) if Fy.ndim<4 else (-4,-2,-1))
        Fy=Fy[...,keep,:,:]
        Y=Y[...,keep,:,:]

    visPtgt(Fy.copy(),normSum=False, centFy=True, detrendFy=False, bwdAccumulate=False, minDecisLen=100,
            marginalizemodels=True,marginalizedecis=True,nEpochCorrection=50,priorweight=200)
