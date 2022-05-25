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
from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
from mindaffectBCI.decoder.scoreOutput import dedupY0
from mindaffectBCI.decoder.utils import block_permute

def decodingCurveSupervised(Fy,objIDs=None,nInt=(30,25),dedup0:bool=True,nvirt_out:int=-20, verb:int=0, **kwargs):
    '''
    Compute a decoding curve, i.e. mistake-probability over time for probability based stopping from the per-epoch output scores
    
    Args:
        Fy (nModel,nTrl,nEp,nY) : similarity score for each input epoch for each output
                N.B. Supervised decoder only has 1 model!!!
        objIDs (nY,) : mapping from rows of Fy to output object IDs.  N.B. assumed objID==0 is true target
                N.B. if objIDs > size(Fy,2), then additional virtual outputs are added
        nInt (2,) : the number of integeration lengths to use, numThresholds. Defaults to ([30,25])
        nvirt_out (int): number of virtual outputs to add to the true outputs
    Returns:
        integerationLengths (int (nInt,)) : the actual integeration lengths in samples
        ProbErr (float (nInt,)) : empherical error probablility at this integeration length
        ProbErrEst (float (nInt,)) : decoder estimate of the error rate for each integeration length
        StopPerr (nInt,) : error rate at this average trial length when using ProbErrEst-thresholding based stopping
        StopThresh (nInt,) : ProbErrEst threshold used to get this average trial length.
        Yerr (bool (nTrl,nInt)) : flag if prediction was *incorrect* at this integeration length for this trial 
        Perr (float (nTrl,nInt)) : compute probability of incorrect prediction for this integeration length and trial
    '''
    if objIDs is None:
        objIDs = np.arange(Fy.shape[-1])   
    if nInt is None:
        nInt = [min(Fy.shape[-2], 30), 25]
    if not hasattr(nInt,'__iter__'):
        nInt = (nInt,nInt)
    if Fy is None:
        return 1, 1, None, None, None, None, -1, 1

    #print("kwargs={}".format(kwargs))

    # remove trials with no-true-label info
    keep = np.any(Fy[..., objIDs == 0], (-2, -1) if Fy.ndim<=3 else (0,-2,-1)) # [ nTrl ]
    if not np.all(keep):
        if not any(keep):
            print('No trials with true label info!')
        else:
            Fy = Fy[..., keep, :, :]
            if verb>0 : print('Discarded %d trials without true-label info'%(sum(np.logical_not(keep))))
    
    if nvirt_out is not None and not nvirt_out==0:
        # generate virtual outputs for testing -- not from the 'true' target though
        virt_Fy = block_permute(Fy[...,1:], nvirt_out, axis=-1, perm_axis=-2) 
        Fy = np.append(Fy,virt_Fy,axis=-1)
        if verb>0 : print("Added {} virtual outputs".format(virt_Fy.shape[-1]))

    if dedup0 is not None and dedup0 is not False: # remove duplicate copies output=0
        Fy = dedupY0(Fy, zerodup=dedup0>0, yfeatdim=False)

    # get the points at which we compute performances
    if len(nInt) < 3:
        if nInt[0] > 0 and nInt[0] < Fy.shape[-2]: #  number steps
            integerationLengths = np.linspace(0, Fy.shape[-2], min(Fy.shape[-2],nInt[0] + 1), dtype=int, endpoint=True)
        elif nInt[0] < 0: # step sized
            integerationLengths = np.arange(0, Fy.shape[-2]+1, min(Fy.shape[-2],-nInt[0]), dtype=int)
        else: 
            integerationLengths = np.arange(0, Fy.shape[-2]+1, dtype=int)
        integerationLengths = integerationLengths[1:]
    else:
        integerationLengths = nInt
    #print("intlen={}".format(integerationLengths))
        
    Yerr, Perr, aveProbErr, aveProbErrEst = compute_decoding_curve(Fy, objIDs, integerationLengths, verb=verb, **kwargs)

    # up-size Yerr, Perr to match input number of trials
    if not np.all(keep) and np.any(keep):
        tmp=Yerr; Yerr=np.ones((len(keep),)+Yerr.shape[1:],dtype=Yerr.dtype); Yerr[keep,...]=tmp
        tmp=Perr; Perr=np.ones((len(keep),)+Perr.shape[1:],dtype=Perr.dtype); Perr[keep,...]=tmp

    stopPerrThresh,stopYerr = compute_stopping_curve(nInt,integerationLengths,Perr,Yerr)
    
    if verb>=0:
        print(print_decoding_curve(integerationLengths,aveProbErr,aveProbErrEst,stopYerr,stopPerrThresh))

    return integerationLengths, aveProbErr, aveProbErrEst, stopYerr, stopPerrThresh, Yerr, Perr


def compute_decoding_curve(Fy:np.ndarray, objIDs, integerationLengths, verb:int=0, **kwargs):
    """compute the decoding curves from the given epoch+output scores in Fy

    Args:
        Fy (float: ((nM,)nTrl,nEp,nY)) : per-epoch output scores
        objIDs (float: (nY,)) : the objectIDs for the outputs in Fy
        integerationLengths (float (nInt,)) : a list of integeration lengths to compute peformance at

    Returns:
        Yerr (nTrl,nInt : bool) : flag if prediction was *incorrect* at this integeration length for this trial 
        Perr (nTrl,nInt : float) : compute probability of error for this integeration length and trial
        aveProbErr (nInt: float) : average error probablility at this integeration length
        aveProbErrEst (nInt: float):  average estimated error probability for this integeration length
    """    
    Yidx=-np.ones((Fy.shape[-3], len(integerationLengths)),dtype=int) # (nTrl,nInt)
    Yest=-np.ones((Fy.shape[-3], len(integerationLengths)),dtype=int) # (nTrl,nInt)
    Perr= np.ones((Fy.shape[-3], len(integerationLengths)),dtype=np.float32) # (nTrl,nInt)

    if verb>=0: print("Int Lens:", end='')
    for li,nep in enumerate(integerationLengths):
        Yidxli,Perrli,_,_,_=decodingSupervised(Fy[..., :nep, :], **kwargs)
        # BODGE: only use result from first-model & last decision point!!!!
        if Yidxli.ndim>1:
            if  Yidxli.shape[-1]>1 or (Yidxli.ndim>2 and Yidxli.shape[0]>1):
                print("Warning: multiple decision points or models, taking the last one!")
            Yidxli=Yidxli[:,-1] if Yidxli.ndim==2 else Yidxli[-1,:,-1]
            Perrli=Perrli[:,-1] if Perrli.ndim==2 else Perrli[-1,:,-1]
        Yidx[:,li]=Yidxli
        Perr[:,li]=Perrli
        # convert from Yidx to Yest, note may be invalid = -1
        Yest[:,li]=[ objIDs[yi] if yi in objIDs else -1 for yi in Yidxli ]
        if verb>=0: print('.',end='',flush=True)
    if verb>=0: print("\n")

    Yerr = Yest!=0 # (nTrl, nEp)
    aveProbErr   =np.mean(Yerr,0) #(nEp)
    aveProbErrEst=np.mean(Perr,0) #(nEp)
    return (Yerr,Perr,aveProbErr,aveProbErrEst)


def compute_stopping_curve(nInt,integerationLengths,Perr,Yerr):
    """compute the stopping curve -- which is the performance at times when stopping threshold (Perr) is passed

    Args:
        nInt (int): number of time points to compute the stopping curve at
        integerationLengths (list int): the set of integeration lengths at which stopping curve is computed
        Perr ( nTrl,nInt): Probability of error at each time point
        Yerr ( nTrl,nInt): For each time point if the 'best' prediction is correct or not

    Returns:
        [type]: [description]
    """    
    
    nthresh=nInt[1] if len(nInt)>1 else nInt[0]
    if nthresh < 0:
        nthresh=20
    #thresholds =linspace(min(Perr(:)),max(Perr(:)),nthresh)'; #set thresho
    thresholds=1 - np.exp(np.linspace(-0.25, -5, nthresh))
    thresholds=np.append(np.linspace(thresholds[0]*.2,thresholds[0]*.8,3),thresholds,0) # [ nThresh ]
    perrstopiYerr=np.zeros((Perr.shape[0],len(thresholds),4)) # (nTrl,nThresh,4) [ 4 x nThresh x nTrl ]
    for trli in range(Perr.shape[0]):
        tmp=Perr[trli,:]
        for ti,thresh in enumerate(thresholds):
            stopi=np.argmax(tmp < thresh) # returns 1st time test is true
            if not tmp[stopi]<thresh: 
                stopi = Perr.shape[1]-1 # set to end trial if didn't pass
            perrstopiYerr[trli,ti,:]=[thresh,Perr[trli,stopi],integerationLengths[stopi],Yerr[trli,stopi]]
    aveThreshPerrIntYerr=np.mean(perrstopiYerr,0) # (nThresh,4) [4 x nThresh] average stopping time for each threshold
    # BODGE: map into integeration lengths to allow use the same ploting routines

    # threshold with closest average stopping time
    mi=np.argmin(np.abs(integerationLengths[:,np.newaxis]-aveThreshPerrIntYerr[:,2:3].T),1) # [ nInt ] 

    stopThresh=aveThreshPerrIntYerr[mi,0]
    stopPerrThresh=aveThreshPerrIntYerr[mi,1]
    stopYerr=aveThreshPerrIntYerr[mi,3]
    return stopPerrThresh,stopYerr

def comp_audc(aveProbErr): 
    """ area under decoding curve """
    return 100.0*np.mean(aveProbErr)# ** 0.6)
def comp_psae(aveProbErrEst, aveProbErr, MINSCALEPERR=.1):  
    """ sum squared perr estimate error -- weighted by estimated value, so small estvalues cost more """
    return 100.0*np.mean(np.abs(aveProbErrEst - aveProbErr) / np.maximum(aveProbErrEst,MINSCALEPERR))
def comp_ausc(stopYerr):
    """Area under stopping curve"""
    return 100.0*np.mean(stopYerr) 
def comp_ssae(stopYerr, stopPerrThresh, MINSCALEPERR=.1):
    """ sum squared stopping threshold estimate error -- weighted by estimated value, so small estvalues cost more """
    return 100.0*np.mean(np.abs(stopYerr - stopPerrThresh) / np.maximum(stopPerrThresh,MINSCALEPERR))


def score_decoding_curve(integerationLengths,aveProbErr,aveProbErrEst=None,stopYerr=None,stopPerrThresh=None, Yerr=None, Perr=None, MINSCALEPERR=.1):
    """compute score metrics from a decoding curve

    Args:
        integerationLengths ([type]): [description]
        aveProbErr ([type]): [description]
        aveProbErrEst ([type], optional): [description]. Defaults to None.
        stopYerr ([type], optional): [description]. Defaults to None.
        stopPerrThresh ([type], optional): [description]. Defaults to None.
        Yerr ([type], optional): [description]. Defaults to None.
        Perr ([type], optional): [description]. Defaults to None.
        MINSCALEPERR (float, optional): [description]. Defaults to .1.

    Returns:
        dict: the 4 computed scores: 
            AUDC = Area Under Decoding Curve
            PSAE = Perr Sum Absolute Errors (measure of accuracy of Perr estimation)
            AUSC = Area Under Stopping Curve
            SSAE = Stopping Sum Absolute Error (measure of accuracy of the stopping criteria) 
    """    
    audc=comp_audc(aveProbErr)
    psae=comp_psae(aveProbErrEst,aveProbErr,MINSCALEPERR)
    ausc=comp_ausc(stopYerr)
    ssae=comp_ssae(stopYerr, stopPerrThresh, MINSCALEPERR)
    score = (100-audc)/100
    return {'score':score, 'audc':audc, "psae":psae, "ausc":ausc, "ssae":ssae}

def print_decoding_curve(integerationLengths,aveProbErr,aveProbErrEst=None,stopYerr=None,stopPerrThresh=None, Yerr=None, Perr=None, MINSCALEPERR=.1):
    """[summary]

    Args:
        integerationLengths ([type]): [description]
        aveProbErr ([type]): [description]
        aveProbErrEst ([type], optional): [description]. Defaults to None.
        stopYerr ([type], optional): [description]. Defaults to None.
        stopPerrThresh ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """    
    s=''    
    # get set lengths to print performance for
    logIdx=np.linspace(0,len(integerationLengths)-1,min(len(integerationLengths),9),dtype=int,endpoint=True)
    if  len(logIdx)>1 :
        logIdx=logIdx[1:]

    #print("logidx={}".format(logIdx))
    # make a formated summary string
    s+='%18s  '%('IntLen') + " ".join(['%4d '%(i) for i in integerationLengths[logIdx]]) + "\n"
    s+='%18s  '%('Perr')   + " ".join(['%4.2f '%(i) for i in aveProbErr[logIdx]])
    s+='  AUDC %4.1f'%(comp_audc(aveProbErr)) + "\n"

    # PERREST
    if not aveProbErrEst is None:
        s+='%18s  '%('Perr(est)')+ " ".join(['%4.2f '%(i) for i in aveProbErrEst[logIdx]])    
        s+='  PSAE %4.1f'%(comp_psae(aveProbErrEst,aveProbErr,MINSCALEPERR)) + '\n'    

    # STOPPING CURVE
    if not stopYerr is None:
        s+="%18s  "%("StopErr") + " ".join(['%4.2f '%(i) for i in stopYerr[logIdx]])
        s+="  AUSC %4.1f"%(comp_ausc(stopYerr)) + "\n"
        s+='%18s  '%('StopThresh(P)') + " ".join(['%4.2f '%(i) for i in stopPerrThresh[logIdx]])
        s+="  SSAE %4.1f"%(comp_ssae(stopYerr,stopPerrThresh,MINSCALEPERR)) + "\n"
        
    return s



def plot_decoding_curve(integerationLengths, aveProbErr, aveProbErrEst=None,stopYerr=None,stopPerrThresh=None, Yerr=None, Perr=None, labels=None, fs:float=None, xunit:str='samples', title:str=''):
    """
    plot the decoding curve

    Args:
        integerationLengths ([type]): [description]
        aveProbErr ([type]): [description]
    """    

    import matplotlib.pyplot as plt
    if fs is not None:
        xunit='s'
    else:
        fs = 1

    if aveProbErr.ndim > 1:
        # multiple datasets
        for i in range(integerationLengths.shape[0]):
            if labels is not None:
                plt.plot(integerationLengths[i,:]/fs,aveProbErr[i,:],label=labels[i])
            else:
                plt.plot(integerationLengths[i,:]/fs,aveProbErr[i,:],color=(.8,.8,.8)) # grey lines
        plt.plot(np.nanmean(integerationLengths,0)/fs, np.nanmean(aveProbErr,0), 'k', linewidth=5, label="mean")
        plt.title('Decoding Curve\n(nDatasets={}) {}'.format(aveProbErr.shape[0],title))
    
    else:
        # single dataset
        if not Yerr is None and not Perr is None:
            # plot the trialwise estimates, when is single subject
            #Yerr = args[5-2] #(nTrl,nInt), flag if was right or not
            oPerr = Perr.copy() #args[6-2] #(nTrl,nInt)
            keep = np.any(oPerr<1,axis=-1) #(nTrl)
            if np.any(keep.ravel()):
                Yerr=Yerr[keep,:]
                oPerr=oPerr[keep,:]

                Perr=oPerr.copy()
                plt.plot(integerationLengths.T/fs,Perr.T,color='.95') # line per trial
                Perr[Yerr<0]=np.NaN
                Perr[Yerr==True]=np.NaN # disable points where it was in error
                # est when was correct
                plt.plot(integerationLengths.T/fs,Perr[0,:].T,'.',markerfacecolor=(0,1,0,.2),markeredgecolor=(0,1,0,.2),label='Perr(correct)')
                plt.plot(integerationLengths.T/fs,Perr.T,'.',markerfacecolor=(0,1,0,.2),markeredgecolor=(0,1,0,.2))
                # est when incorrect..
                Perr = oPerr.copy() #(nTrl,nInt)
                Perr[Yerr<0]=np.NaN
                Perr[Yerr==False]=np.NaN # disable points where it was in error, or not available
                plt.plot(integerationLengths.T/fs,Perr[0,:].T,'.', markerfacecolor=(1,.0,.0,.2), markeredgecolor=(1,.0,.0,.2),label='Perr(incorrect)')
                plt.plot(integerationLengths.T/fs,Perr.T,'.', markerfacecolor=(1,.0,.0,.2), markeredgecolor=(1,.0,.0,.2))
            plt.title('Decoding Curve\n(nTrl={}) {}'.format(Yerr.shape[0],title))

        plt.plot(integerationLengths.T/fs,aveProbErr.T,'.-',label='avePerr')

    plt.ylim((0,1))
    plt.xlabel('Integeration Length\n({})'.format(xunit))
    plt.ylabel('Perr')
    plt.legend()
    plt.grid(True)


def flatten_decoding_curves(decoding_curves):
    ''' take list of (potentially variable length) decoding curves and make into a single array '''
    il = np.zeros((len(decoding_curves),decoding_curves[0][0].size))
    pe = np.zeros(il.shape)
    pee = np.zeros(il.shape)
    se = np.zeros(il.shape)
    st = np.zeros(il.shape)
    # TODO [] : insert according to the actual int-len
    for di,dc in enumerate(decoding_curves):
        il_di = dc[0]
        ll = min(il.shape[1],il_di.size)
        il[di,:ll] = dc[0][:ll]
        pe[di,:ll] = dc[1][:ll]
        pee[di,:ll] = dc[2][:ll]
        se[di,:ll] = dc[3][:ll]
        st[di,:ll] = dc[4][:ll] 
    return il,pe,pee,se,st


def testcase():
    """[summary]
    """    
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.normalizeOutputScores import mktestFy,  normalizeOutputScores
    Fy, nEp = mktestFy(sigstr=.5,nM=1,nY=10,nTrl=10,startupNoisefrac=25) #(nM, nTrl, nEp, nY)
    #Fy = Fy[0,...] # (nTrl,nEp,nY)
    # Introduce temporal and spatial sparsity like real data
    Fy = Fy * (np.random.standard_normal((1,Fy.shape[-2],Fy.shape[-1]))>0).astype(np.float)
    Fy[...,:50,:] = 0 # block zero at start

    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
    (dc)=decodingCurveSupervised(Fy,nInt=(25,25),nvirt_out=-12,softmaxscale=2)
    plot_decoding_curve(*dc)
    plt.show(block=True)

    quit()

    sFy = np.cumsum(Fy,-2)
    Yi = np.argmax(sFy,-1)
    audc=np.sum((Yi==0).ravel())/Yi.size
    print("1-Audc_score={}".format(1-audc))
    
    # test with multiple lines plotting
    il=np.tile(dc[0][np.newaxis,:],(4,1))
    pe=np.tile(dc[1][np.newaxis,:],(4,1))
    pe= pe+np.random.standard_normal(pe.shape)*.1 # add some noise
    plt.figure()
    plot_decoding_curve(il,pe)
    plt.show()
    

if __name__=="__main__":
    testcase()
