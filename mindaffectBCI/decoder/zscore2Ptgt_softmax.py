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

def zscore2Ptgt_softmax(f, softmaxscale:float=2, prior:np.ndarray=None, validTgt=None, marginalizemodels:bool=True, marginalizedecis:bool=False, peroutputmodel:bool=False):
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

def calibrate_softmaxscale(f, validTgt=None, 
                           scales=(.01,.02,.05,.1,.2,.3,.4,.5,1,1.5,2,2.5,3,3.5,4,5,7,10,15,20,30,50,100), 
                           MINP=1e-5, marginalizemodels=True, marginalizedecis=False, 
                           nocontrol_condn=.1, verb=0):
    '''
    attempt to calibrate the scale for a softmax decoder to return calibrated probabilities, optionally simulate a no-control condition for controlling the false-positive detections

    Args:
     f ((nM,)nTrl,nDecis,nY): normalized accumulated scores
     validTgt(bool (nM,)nTrl,nY): which targets are valid in which trials.
     scales (list:int): set of possible soft-max scales to try.
     marginalizemodels (bool): marginalize over models when computing the output pvals? Defaults to True.
     marginalizedecis (bool): marginalize over decision points when computing the output pvals? Defaults to False.
     nocontrol_condn (float): if >0, then simulate a set of no-control condition trials by only including
            the non-target outputs, and include the loss on this dataset weighted by `nocontrol_condn` in the loss
     MINP (float): minimium P-value.  We clip the true-target p-val to this level as a way
            of forcing the fit to concentrate on getting the p-val right when high, rather than
            over penalizing when it's wrong

    Returns:
     softmaxscale (float): slope for softmax to return calibrated probabilities
    '''
    # remove trials with no-true-label info
    axis = (-3,-1) if f.ndim>3 else (-1)
    keep = np.any(f[..., 0], axis) # [ nTrl ]
    if not np.all(keep):
        f = f[..., keep, :, :]
    # strip empty outputs (over all trials)
    axis = (-4,-3,-2) if f.ndim>3 else (-3,-2)
    keep = np.any(f!=0,axis=axis) # nY
    if not np.all(keep):
        f = f[..., keep]
    if f.size==0: # guard against no data to calibrate on
        return None

    validTgt = np.any(f != 0, axis=(-4,-2) if f.ndim>3 else -2) # (nTrl,nY)

    # compute label confidence weighting, by estimating the performance at each time-point
    ycorr = np.argmax(f,axis=-1)[...,np.newaxis]==0
    axis = (-4,-3) if f.ndim>3 else (-3,)
    pcorr = np.sum(ycorr,axis=axis,keepdims=True)/np.prod([ycorr.shape[i] for i in axis])

    f_nc = None
    if nocontrol_condn and f.shape[-1]>5:
        f_nc = f[..., 1:]
        vtgt_nc = np.any(f_nc != 0, axis=(-4,-2) if f.ndim>3 else -2) # (nTrl,nY)
        if verb>0: print('No-control amplitude {} on {} nc-outputs'.format(nocontrol_condn,f_nc.shape[-1]))

    # include the nout correction on a per-trial basis
    noutcorr = softmax_nout_corr(np.sum(validTgt,1)) # (nTrl,)

    # simply try a load of scales - as input are normalized shouldn't need more than this
    Ed = np.zeros(len(scales),)
    for i,s in enumerate(scales):
        # apply the soft-max with this scaling
        Ptgt = zscore2Ptgt_softmax(f,softmaxscale=s,validTgt=validTgt,marginalizemodels=marginalizemodels,marginalizedecis=marginalizedecis)

        # Compute the loss = cross-entropy.  
        # As Y==0 is *always* the true-class, this becomes simply sum log this class
        Ptrue = Ptgt[...,0:1]
        Edi = -np.sum( pcorr*np.log(np.maximum(Ptrue,MINP)) + (1-pcorr)*np.log(np.maximum(1-np.maximum(Ptrue,.8),MINP)) )

        if nocontrol_condn and not f_nc is None:
            # inlude a non-control class loss
            Ptgt_nc = zscore2Ptgt_softmax(f_nc,softmaxscale=s,validTgt=vtgt_nc,marginalizemodels=marginalizemodels,marginalizedecis=marginalizedecis)
            Edi_nc = np.sum( -np.log(np.maximum(1-np.maximum(np.max(Ptgt_nc,axis=-1),.9),MINP)) ) #/ (f.shape[-1]-1)
            Edii = Edi + Edi_nc * nocontrol_condn
            if verb > 0: print("{:3d}) scale={:5.1f} Ed={:5.1f} = {:5.1f} + {:5.1f}".format(i,s,Edii, Edi, Edi_nc * nocontrol_condn))
            Edi = Edii
        else:
            if verb > 0: print("{:3d}) scale={:5.1f} Ed={:5.1f}".format(i,s,Edi))

        Ed[i] = Edi
    # use the max-entropy scale
    mini = np.argmin(Ed)
    softmaxscale = scales[mini]
    if verb>0: print("softmaxscale={}".format(softmaxscale))
    return softmaxscale

def visPtgt(Fy, normSum, centFy, detrendFy, bwdAccumulate,
            marginalizemodels, marginalizedecis, 
            nEpochCorrection, priorweight, minDecisLen=-1, nocontrol_condn=True, n_virt_outputs=None):
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
    softmaxscale = calibrate_softmaxscale(ssFy,marginalizemodels=marginalizemodels, nocontrol_condn=nocontrol_condn)

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

if __name__=="__main__":
    if True:
        from mindaffectBCI.decoder.normalizeOutputScores import mktestFy
        Fy, noise, sigamp = mkTestFy(nY=10, nM=1, nEp=800, nTrl=100, sigstr=.25, startup_lag=.20)
    else:
        import pickle
        stopping=pickle.load(open('stopping.pk','rb'))
        Fy = stopping['Fy']
        Y=stopping['Y']
        keep = np.any(Fy>0,axis=(-2,-1) if Fy.ndim<4 else (-4,-2,-1))
        Fy=Fy[...,keep,:,:]
        Y=Y[...,keep,:,:]

    visPtgt(Fy.copy(),normSum=True, centFy=True, detrendFy=False, bwdAccumulate=False, minDecisLen=100,
            marginalizemodels=True,marginalizedecis=True,nEpochCorrection=30,priorweight=0,
            nocontrol_condn=.1, n_virt_outputs=0)

    visPtgt(Fy.copy(),normSum=True, centFy=True, detrendFy=False, bwdAccumulate=False, minDecisLen=100,
            marginalizemodels=True,marginalizedecis=True,nEpochCorrection=30,priorweight=0,
            nocontrol_condn=False, n_virt_outputs=0)
