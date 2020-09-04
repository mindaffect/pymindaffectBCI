import numpy as np
from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
def decodingCurveSupervised(Fy,objIDs=None,nInt=(30,25),**kwargs):
    '''
    Compute a decoding curve, i.e. mistake-probability over time for probability based stopping from the per-epoch output scores
    
    Args:
        Fy  = (nModel,nTrl,nEp,nY) [nY x nEpoch x nTrl x nModel ] similarity score for each input epoch for each output
                N.B. Supervised decoder only has 1 model!!!
        objIDs = (nY,) mapping from rows of Fy to output object IDs.  N.B. assumed objID==0 is true target
                N.B. if objIDs > size(Fy,2), then additional virtual outputs are added
        nInt = (2,) the number of integeration lengths to use, numThresholds. Defaults to ([30,25])
    Returns:
        integerationLengths - [nInt] the actual integeration lengths in samples
        ProbErr: (nInt,)float empherical error probablility at this integeration length
        ProbErrEst(np.ndarray): (nInt: float) decoder estimate of the error rate for each integeration length
        StopPerr - (nInt,) error rate at this average trial length when using ProbErrEst-thresholding based stopping
        StopThresh - (nInt,) ProbErrEst threshold used to get this average trial length.
        Yerr(np.ndarray): (nTrl,nInt : bool) flag if estimate was correct at this integeration length for this trial 
        Perr(np.ndarray): (nTrl,nInt : float) compute probability of error for this integeration length and trial
    '''
    if objIDs is None:
        objIDs = np.arange(Fy.shape[-1])   
    if nInt is None:
        nInt = [min(Fy.shape[-2], 30), 25]
    if not hasattr(nInt,'__iter__'):
        Int = (nInt,nInt)
    if Fy is None:
        return 1, 1, None, None, None, None, -1, 1
    if Fy.ndim > 3:
        Fyshape = Fy.shape
        Fy = np.reshape(Fy, (np.prod(Fy.shape[:-2]),)+Fy.shape[-2:]) # ((nM*nTrl),nEp,nY)

    # remove trials with no-true-label info
    keep = np.any(Fy[:, :, objIDs == 0], (-2, -1)) # [ nTrl ]
    if not np.all(keep):
        Fy = Fy[keep, :, :]
        print('Discarded %d trials without true-label info'%(sum(np.logical_not(keep))))
        if not any(keep):
            print('No trials with true label info!')
    
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
        
    Yerr, Perr, aveProbErr, aveProbErrEst = compute_decoding_curve(Fy,objIDs,integerationLengths, **kwargs)

    stopPerrThresh,stopYerr = compute_stopping_curve(nInt,integerationLengths,Perr,Yerr)
    
    print(print_decoding_curve(integerationLengths,aveProbErr,aveProbErrEst,stopYerr,stopPerrThresh))
    
    return integerationLengths,aveProbErr,aveProbErrEst,stopYerr,stopPerrThresh, Yerr, Perr


def compute_decoding_curve(Fy:np.ndarray, objIDs, integerationLengths, **kwargs):
    """compute the decoding curves from the given epoch+output scores in Fy

    Args:
        Fy ([np.ndarray]): (nTrl,nEp,nY) per-epoch output scores
        objIDs ([float]): (nY,) the objectIDs for the outputs in Fy
        integerationLengths ([type]): (nInt,) a list of integeration lengths to compute peformance at

    Returns:
        Yerr(np.ndarray): (nTrl,nInt : bool) flag if estimate was correct at this integeration length for this trial 
        Perr(np.ndarray): (nTrl,nInt : float) compute probability of error for this integeration length and trial
        aveProbErr: (nInt: float) average error probablility at this integeration length
        aveProbErrEst(np.ndarray): (nInt: float) averaget estimated error probability for this integeration length
    """    
    Yidx=np.zeros((Fy.shape[0], len(integerationLengths))) # (nTrl,nInt)
    Yest=np.zeros((Fy.shape[0], len(integerationLengths))) # (nTrl,nInt)
    Perr=np.zeros((Fy.shape[0], len(integerationLengths))) # (nTrl,nInt)

    print("Int Lens:", end='')
    for li,nep in enumerate(integerationLengths):
        Yidxli,Perrli,_,_,_=decodingSupervised(Fy[:, :nep, :], **kwargs)
        # BODGE: only use result from last decision point
        Yidx[:,li]=Yidxli[:,-1]
        Perr[:,li]=Perrli[:,-1]
        # convert from Yidx to Yest, note may be invalid = -1
        Yest[:,li]=[ objIDs[yi] if yi in objIDs else -1 for yi in Yidxli[:,-1] ]
        print('.',end='',flush=True)
    print("\n")

    Yerr = Yest!=0 # (nTrl, nEp)
    aveProbErr   =np.mean(Yerr,0) #(nEp)
    aveProbErrEst=np.mean(Perr,0) #(nEp)
    return (Yerr,Perr,aveProbErr,aveProbErrEst)


def compute_stopping_curve(nInt,integerationLengths,Perr,Yerr):
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

    # threshold with closes average stopping time
    mi=np.argmin(np.abs(integerationLengths[:,np.newaxis]-aveThreshPerrIntYerr[:,2:3].T),1) # [ nInt ] 

    stopThresh=aveThreshPerrIntYerr[mi,0]
    stopPerrThresh=aveThreshPerrIntYerr[mi,1]
    stopYerr=aveThreshPerrIntYerr[mi,3]
    return stopPerrThresh,stopYerr

    
def print_decoding_curve(integerationLengths,aveProbErr,aveProbErrEst=None,stopYerr=None,stopPerrThresh=None):
    MINSCALEPERR=0.1
    
    s=''    
    # get set lengths to print performance for
    logIdx=np.linspace(0,len(integerationLengths)-1,min(len(integerationLengths),9),dtype=int,endpoint=True)
    if  len(logIdx)>1 :
        logIdx=logIdx[1:]

    #print("logidx={}".format(logIdx))
    # make a formated summary string
    # area under decoding curve (weighted)
    audc=100.0*np.mean(aveProbErr)# ** 0.6)
    s+='%18s  '%('IntLen') + " ".join(['%4d '%(i) for i in integerationLengths[logIdx]]) + "\n"
    s+='%18s  '%('Perr')   + " ".join(['%4.2f '%(i) for i in aveProbErr[logIdx]])
    s+='  AUDC %4.1f'%(audc) + "\n"

    # PERREST
    if not aveProbErrEst is None:
        # sum squared perr estimate error
        psae=100.0*np.mean(np.abs(aveProbErrEst - aveProbErr) / np.maximum(aveProbErr,MINSCALEPERR))
        
        s+='%18s  '%('Perr(est)')+ " ".join(['%4.2f '%(i) for i in aveProbErrEst[logIdx]])    
        s+='  PSAE %4.1f'%(psae) + '\n'    

    # STOPPING CURVE
    if not stopYerr is None:
        # area under stopping curve
        ausc=100.0*np.mean(stopYerr) 
        # sum-squared error in stopping estimate
        ssae=100.0*np.mean(np.abs(stopYerr - stopPerrThresh) / np.maximum(stopYerr,MINSCALEPERR))
        s+="%18s  "%("StopErr") + " ".join(['%4.2f '%(i) for i in stopYerr[logIdx]])
        s+="  AUSC %4.1f"%(ausc) + "\n"
        s+='%18s  '%('StopThresh(P)') + " ".join(['%4.2f '%(i) for i in stopPerrThresh[logIdx]])
        s+="  SSAE %4.1f"%(ssae) + "\n"
        
    return s


def plot_decoding_curve(integerationLengths, aveProbErr, *args):
    ''' plot the decoding curve '''
    import matplotlib.pyplot as plt

    if aveProbErr.ndim > 1:
        # multiple datasets
        plt.plot(integerationLengths.T,aveProbErr.T)
        plt.plot(np.mean(integerationLengths,0), np.mean(aveProbErr,0), 'k', linewidth=5, label="mean")
    
    else:
        # single dataset
        plt.plot(integerationLengths.T,aveProbErr.T,label='avePerr')
        if len(args)>=7-2:
            # plot the trialwise estimates, when is single subject
            Yerr = args[5-2] #(nTrl,nInt), flag if was right or not
            Perr = args[6-2].copy() #(nTrl,nInt)
            Perr[Yerr==True]=np.NaN # disable points where it was in error
            # est when was correct
            plt.plot(integerationLengths.T,Perr[0,:].T,'.',markerfacecolor=(0,1,0,.2),markeredgecolor=(0,1,0,.2),label='Perr(objID == 0)')
            plt.plot(integerationLengths.T,Perr.T,'.',markerfacecolor=(0,1,0,.2),markeredgecolor=(0,1,0,.2))
            # est when incorrect..
            Perr = args[6-2].copy() #(nTrl,nInt)
            Perr[Yerr==False]=np.NaN # disable points where it was in error
            plt.plot(integerationLengths.T,Perr[0,:].T,'.', markerfacecolor=(1,.0,.0,.2), markeredgecolor=(1,.0,.0,.2),label='Perr(objID ~= 0)')
            plt.plot(integerationLengths.T,Perr.T,'.', markerfacecolor=(1,.0,.0,.2), markeredgecolor=(1,.0,.0,.2))

    plt.ylim((0,1))
    plt.xlabel('Integeration Length (samples)')
    plt.ylabel('Perr')
    plt.legend()
    plt.grid(True)

def testcase():
    import numpy as np
    Fy=np.random.standard_normal((10,100,50))
    Fy[:,:,0]=Fy[:,:,0] + 0.3
    from decodingCurveSupervised import decodingCurveSupervised
    (dc)=decodingCurveSupervised(Fy)
    plot_decoding_curve(*dc)

    sFy = np.cumsum(Fy,-2)
    Yi = np.argmax(sFy,-1)
    audc=np.sum((Yi==0).ravel())/Yi.size
    print("1-Audc_score={}".format(1-audc))
    
    # test with multiple lines plotting
    il=np.tile(dc[0][np.newaxis,:],(4,1))
    pe=np.tile(dc[1][np.newaxis,:],(4,1))
    pe= pe+np.random.standard_normal(pe.shape)*.1 # add some noise
    plot_decoding_curve(il,pe)
    

if __name__=="__main__":
    testcase()
