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
    f = f * softmaxscale.reshape((-1,1,1))

    # inlude the prior
    if prior is not None:
        logprior = np.log(np.maximum(prior,1e-8))
        f = f + logprior
    
    # multiple models
    if ((marginalizemodels and f.ndim > 3 and f.shape[0] > 1) or 
        (marginalizedecis and f.shape[-2] > 1)): # mulitple decis points 

        # marginalize for the P values.
        # get the axis to marginalize over
        axis=[]
        if f.ndim>3 and marginalizemodels: # marginalize the models axis
            axis.append(-4)
        if marginalizedecis: # marginalize the decision points axis
            axis.append(-2)
        axis=tuple(axis)

        f = marginalize_scores(f, axis, keepdims=False)

    # get the prob each output conditioned on the model
    Ptgt = softmax(f,axis=-1,validTgt=validTgt) # ((nM,)nTrl,nDecis,nY)

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
        p = p * validTgt[..., np.newaxis, :]
    # convert to softmax, with guard for div by zero
    p = p / np.maximum(np.sum(p, axis, keepdims=True),1e-6)
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
        f = f + logprior

    maxf = np.max(f, axis=axis, keepdims=True) # remove for numerical robustness
    z = np.exp( f - maxf ) # non-normalized Ptgt
    f = np.log(np.sum(z, axis=axis, keepdims=keepdims)) + maxf

    return f

    maxf = np.max(f, axis=axis, keepdims=True) # remove for numerical robustness
    z = np.exp( f - maxf ) # non-normalized Ptgt
    p = z / np.sum(z, axis, keepdims=True) # normalized Ptgt
    #p = softmax(f,axis=axis)
    f = np.sum(f * p, axis, keepdims=keepdims) # marginalized score

    return f

def calibrate_softmaxscale(f, validTgt=None, scales=(.05,.1,.2,.5,1,1.5,2,2.5,3,3.5,4,5,7,10,15,20,30), MINP=.01, marginalizemodels=True, marginalizedecis=False):
    '''
    attempt to calibrate the scale for a softmax decoder to return calibrated probabilities

    Args:
     f ((nM,)nTrl,nDecis,nY): normalized accumulated scores
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
        Ptgt = zscore2Ptgt_softmax(f,s,validTgt=validTgt,marginalizemodels=marginalizemodels,marginalizedecis=marginalizedecis)
        # Compute the loss = cross-entropy.  
        # As Y==0 is *always* the true-class, this becomes simply sum log this class 
        Ed[i] = np.sum(-np.log(np.maximum(Ptgt[...,0],MINP)))
        #print("{}) scale={} Ed={}".format(i,s,Ed[i]))
    # use the max-entropy scale
    mini = np.argmin(Ed)
    softmaxscale = scales[mini]
    print("softmaxscale={}".format(softmaxscale))
    return softmaxscale

#@function
def testcase(nY=10, nM=4, nEp=340, nTrl=100, sigstr=.5, marginalizemodels=True, marginalizedecis=False):
    import numpy as np
    np.random.seed(0)
    noise = np.random.standard_normal((nM,nTrl,nEp,nY))
    noise = noise - np.mean(noise.ravel())
    noise = noise / np.std(noise.ravel())

    sigamp=sigstr*np.ones(noise.shape[-2]) # [ nEp ]
    # no signal ast the start of the trial
  #startupNoise_samp=nEp*.5;
  #sigamp((1:size(sigamp,2))<startupNoise_samp)=0;
    Fy = np.copy(noise)
    # add the signal
    Fy[0, :, :, 0] = Fy[0, :, :, 0] + sigamp
    #print("Fy={}".format(Fy))
    
    sFy=np.cumsum(Fy,-2)
    from mindaffectBCI.decoder.normalizeOutputScores import normalizeOutputScores
    ssFy,scale_sFy,N,_,_=normalizeOutputScores(Fy,minDecisLen=-1)
    #print('ssFy={}'.format(ssFy.shape))
    from mindaffectBCI.decoder.zscore2Ptgt_softmax import zscore2Ptgt_softmax, softmax
    smax = softmax(ssFy)
    #print("{}".format(smax.shape))
    Ptgt=zscore2Ptgt_softmax(ssFy,marginalizemodels=marginalizemodels, marginalizedecis=marginalizedecis) # (nTrl,nEp,nY)
    #print("Ptgt={}".format(Ptgt.shape))
    import matplotlib.pyplot as plt
    plt.clf()
    tri=1
    for mi in range(nM):
        plt.subplot(4,nM,0*nM+mi+1);plt.cla();
        plt.plot(sFy[mi,tri,:,:])
        plt.plot(scale_sFy[mi,tri,:],'k')
        plt.title('sFy')
        plt.grid()
    
        plt.subplot(4,nM,1*nM+mi+1);plt.cla();
        plt.plot(ssFy[mi,tri,:,:])
        plt.title('ssFy')
        plt.grid()
        
        plt.subplot(4,nM,2*nM+mi+1);plt.cla()
        plt.plot(smax[mi,tri,:,:])
        plt.title('softmax')
        plt.grid()
        
    plt.subplot(414);plt.cla()    
    plt.plot(Ptgt[tri,...])
    plt.title('Ptgt')
    plt.grid()
    plt.show()
    
    maxP=np.max(Ptgt,-1) # (nTrl,nEp) [ nEp x nTrl ]
    estopi=[ np.flatnonzero(maxP[tri,:]>.9)[-1] for tri in range(Ptgt.shape[0])]

if __name__=="__main__":
    testcase()
