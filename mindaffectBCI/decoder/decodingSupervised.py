import numpy as np
from mindaffectBCI.decoder.normalizeOutputScores import normalizeOutputScores
from mindaffectBCI.decoder.zscore2Ptgt_softmax import zscore2Ptgt_softmax
#@function
def decodingSupervised(Fy, softmaxscale=3.5, badFyThresh=4,
                       centFy=True, detrendFy=True, 
                       nEpochCorrection=30,
                       minDecisLen=0, maxDecisLen=0,
                       bwdAccumulate=False,
                       marginalizemodels=True, 
                       nocontrolamplitude=None,
                       priorsigma=(-1,120),
                       tiebreaking_noise=1e-8):
    '''
    true-target estimator and error-probility estimator for each trial
    Inputs:
      Fy   - (nModel,nTrl,nEp,nY) [#Y x #Epoch x #Trials x nModels]
      softmaxscale - the scale length to pass to zscore2Ptgt_softmax.py
      badFyThresh - threshold for detction of bad Fy entry, in std-dev
      centFy   - bool, do we center Fy before computing  *important*  (true)
      nEpochCorrection - int, number of epochs to use a base for correction of number epochs in the sum.
               such that: zscore = Fy / ( sigma * ( 1+1/(nEpoch/nEpochErrorCorrection)) ),
               basically, nEpoch < nEpochCorrection have highly increased Perr, so unlikely to be selected
      minDecisLen - int, number of epochs to use as base for distribution of time-based decision points.
                         i.e. decisions at, [1,2,4,...2^n]*exptDistDecis
                    OR: minDecisLen<0 => decision point every abs(minDeicsLen) epochs
      bwdAccumulate - [bool], accumulate data backwards from last epoch gathered
      maxDecisLen   - maximum number of epochs for a decision
      priorsigma = (sigma,N) prior estimate of sigma2 and number pseudo-points (10,30)
    Outputs:
      Yest - (nTrl,nDecis) [ nDecis x nTrl ] the most likely / minimum error output for each decision point
      Perr - (nTrl,nDecis) [ nDecix x nTrl ] the probability that this selection is an ERROR for each decision point
      Ptgt - (nTrl,nDecis,nY) [ nY x nDecis x nTrl ] the probability each target is the true target for each decision point

    Copyright (c) MindAffect B.V. 2018
    '''
    if Fy is None:
        return -1, 1, None, None, None
    
    # get the info on which outputs are zero in each trial
    validTgt = np.any(Fy != 0, axis=-2) # valid if non-zero for any epoch..# (nModel,nTrl,nY)  
    # normalize the raw scores for each model to have nice distribution
    ssFy,varsFy,decisIdx,nEp,nY=normalizeOutputScores(Fy, validTgt=validTgt,
                                               badFyThresh=badFyThresh, 
                                               centFy=centFy, detrendFy=detrendFy,
                                               nEpochCorrection=nEpochCorrection,
                                               minDecisLen=minDecisLen, maxDecisLen=maxDecisLen,
                                               bwdAccumulate=bwdAccumulate,
                                               priorsigma=priorsigma)

    if nocontrolamplitude is not None:
      raise NotImplementedError('no-control signal not yet implemented correctly')
      # # add a no-control pseudo-output to simulate not looking
      # #Median rather than mean?
      # mussFy=np.sum(ssFy,0)/np.maximum(2,np.sum(validTgt,0,keepdims=True)) # mean score for each trial/decisPt [ 1 x nDecis x nTrl x nMdl ]
      # # add to the scores & validTgt info
      # ssFy=np.concatenate((ssFy,nocontrolamplitude+mussFy),0)
      # validtgtTrl=np.concatenate((validtgtTrl,np.ones([1,size(validtgtTrl,2),size(validtgtTrl,3),size(validtgtTrl,4)])))

    # compute the target probabilities over output for each model+trial
    # use the softmax approach to get Ptgt for all outputs
    Ptgt = zscore2Ptgt_softmax(ssFy,
                               softmaxscale,
                               validTgt=validTgt,
                               marginalizemodels=marginalizemodels) # (nM,nTrl,nDecis,nY) [ nY x nDecis x nTrl]
    # extract the predicted output and it's probability of being the target
    Ptgt2d = Ptgt.reshape((np.prod(Ptgt.shape[:-1]), Ptgt.shape[-1])) # make 2d-copy
    # add tie-breaking noise
    if tiebreaking_noise > 0:
        Ptgt2d = Ptgt2d + np.random.standard_normal(Ptgt2d.shape)*tiebreaking_noise

    Yestidx = np.argmax(Ptgt2d, -1) # max over outputs
    Ptgt_max = Ptgt2d[np.arange(Ptgt2d.shape[0]), Yestidx] # value at max, indexing trick to find..
    Yestidx = Yestidx.reshape(Ptgt.shape[:-1]) # -> (nM,nTrl,nY)
    Ptgt_max = Ptgt_max.reshape(Ptgt.shape[:-1])# -> (nM,nTrl,nY)

    # pick a model for this maxY
    decisMdl = Yestidx if ssFy.ndim>3 else 1
    decisEp = 1   
    Yest = Yestidx
    # p(maxz != tgt ) = 1 - p(maxz==tgt)
    Perr = 1 - Ptgt_max
    return Yest, Perr, Ptgt, decisMdl, decisEp


def decodingSupervised_streamed(Fy,softmaxscale=3,nocontrolamplitude=None,badFyThresh=6,
                                centFy=True,nEpochCorrection=15,
                                minDecisLen=0,maxDecisLen=0,
                                bwdAccumulate=False, filtLen=15,
                                marginalizemodels=False):
    '''
    true-target estimator and error-probility estimator for each trial
    Inputs:
      Fy   - (nModel,nTrl,nEp,nY) [#Y x #Epoch x #Trials x nModels]
      softmaxscale - the scale length to pass to zscore2Ptgt_softmax.py
      badFyThresh - threshold for detction of bad Fy entry, in std-dev
      centFy   - bool, do we center Fy before computing  *important*  (true)
      nEpochCorrection - int, number of epochs to use a base for correction of number epochs in the sum.
               such that: zscore = Fy / ( sigma * ( 1+1/(nEpoch/nEpochErrorCorrection)) ),
               basically, nEpoch < nEpochCorrection have highly increased Perr, so unlikely to be selected
      minDecisLen - int, number of epochs to use as base for distribution of time-based decision points.
                         i.e. decisions at, [1,2,4,...2^n]*exptDistDecis
                    OR: minDecisLen<0 => decision point every abs(minDeicsLen) epochs
      bwdAccumulate - [bool], accumulate data backwards from last epoch gathered
      maxDecisLen   - maximum number of epochs for a decision
    Outputs:
      Yest - (nTrl,nDecis) [ nDecis x nTrl ] the most likely / minimum error output for each decision point
      Perr - (nTrl,nDecis) [ nDecix x nTrl ] the probability that this selection is an ERROR for each decision point
      Ptgt - (nTrl,nDecis,nY) [ nY x nDecis x nTrl ] the probability each target is the true target for each decision point

    Copyright (c) MindAffect B.V. 2018
    '''
    if Fy is None:
        return -1, 1, None, None, None
    
    # get the info on which outputs are zero in each trial
    validTgt = np.any(Fy != 0, axis=-2) # valid if non-zero for any epoch..# (nModel,nTrl,nY) [ nY x nTrl x nModel ]   
    # normalize the raw scores for each model to have nice distribution
    stdFy,varsFy,N,nEp,nY=normalizeOutputScores_streamed(Fy, validTgt=validTgt,
                                                         badFyThresh=badFyThresh, centFy=centFy,
                                                         minDecisLen=minDecisLen, filtLen=filtLen)



    # compute the target probabilities over output for each block+model+trial
    # use the softmax approach to get Ptgt for all outputs
    Ptgt = zscore2Ptgt_softmax(ssFy,
                               softmaxscale,
                               validTgt=validTgt,
                               marginalizemodels=False) # (nM,nTrl,nDecis,nY) [ nY x nDecis x nTrl]
    # extract the predicted output and it's probability of being the target
    Ptgt2d = Ptgt.reshape((np.prod(Ptgt.shape[:-1]), Ptgt.shape[-1])) # make 2d-copy
    Yestidx = np.argmax(Ptgt2d, -1) # max over outputs
    Ptgt_max = Ptgt2d[np.arange(Ptgt2d.shape[0]), Yestidx] # value at max, indexing trick to find..
    Yestidx = Yestidx.reshape(Ptgt.shape[:-1]) # -> (nM,nTrl,nY)
    Ptgt_max = Ptgt_max.reshape(Ptgt.shape[:-1])# -> (nM,nTrl,nY)

    # pick a model for this maxY
    decisMdl = Yestidx if ssFy.ndim>3 else 1
    decisEp = 1   
    Yest = Yestidx
    # p(maxz != tgt ) = 1 - p(maxz==tgt)
    Perr = 1 - Ptgt_max
    return Yest, Perr, Ptgt, decisMdl, decisEp



#@function
def testcase():
    from normalizeOutputScores import mktestFy
    Fy,nEp=mktestFy(startupNoisefrac=0,trlenfrac=0,sigstr=.4)
    from decodingSupervised import decodingSupervised
    print("Fy={}".format(Fy.shape))
    Yest,Perr,Ptgt,decismdl,decisEp=decodingSupervised(Fy)
    print("Fy={}".format(Fy[:,[0],:,:].shape))
    Yest,maxp,ssFy,decismdl,decisEp=decodingSupervised(Fy[:,[0],:,:])
    np.mean(Yest == 0,1) 
    from decodingCurveSupervised import decodingCurveSupervised
    decodingCurveSupervised(Fy);

if __name__=="__main__":
    testcase()

