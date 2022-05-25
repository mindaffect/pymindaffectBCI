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
from mindaffectBCI.decoder.utils import window_axis
#@function
def scoreOutput(Fe_mTSe, Y_TSye, dedup0=None, R_mket=None, offset=None, outputscore='ip'):
    '''
    score each output given information on which stim-sequences corrospend to which inputs

    Args

      Fe_mTSe (nM,nTrl,nSamp,nE): similarity score for each event type for each stimulus
      Y_TSye (nTrl,nSamp,nY,nE): Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      R_mket (nM,nfilt,nE,tau): FWD-model (impulse response) for each of the event types, used to correct the 
            scores for correlated responses.
      offset (int): A (set of) offsets to try when decoding.  Defaults to None.
      dedup0 (int): remove duplicate copies of output O, >0 remove the copy, <0 remove objID==0 (used when cross validating calibration data)
      outputscore (str): type of score to compute. one-of: 'ip', 'sse'.  Defaults to 'ip' 

    Returns
      Fy_mTSy  (nM,nTrl,nSamp,nY): similarity score for each input epoch for each output
    
    Copyright (c) MindAffect B.V. 2018
    '''
    if Fe_mTSe.size == 0:
        Fy_mTSy = np.zeros(Fe_mTSe.shape[:-1] + (Y_TSye.shape[-2],),dtype=np.float32)
        return Fy_mTSy

    # ensure correct input number dims
    Y_TSye = Y_TSye.reshape((1,)*(4-Y_TSye.ndim)+Y_TSye.shape)    
    Fe_mTSe = Fe_mTSe.reshape((1,)*(4-Fe_mTSe.ndim)+Fe_mTSe.shape)    

    if dedup0 is not None and dedup0 is not False: # remove duplicate copies output=0
        Y_TSye = dedupY0(Y_TSye, zerodup=dedup0>0)
    # ensure Y_TSye has same type of Fe_mTSe
    Y_TSye = Y_TSye.astype(Fe_mTSe.dtype)

    # inner-product score
    if offset is None:
        Fy_mTSy = np.einsum("mTEe,TEYe->mTEY", Fe_mTSe, Y_TSye, dtype=Fe_mTSe.dtype)
        Fy_mTSy = Fy_mTSy.astype(Fe_mTSe.dtype)

    else:
        assert Fe_mTSe.shape[0]==1, "Offsets only for single models!"
        if not hasattr(offset,'__iter__'): 
            offset=[offset]
        # list of possible offsets to try
        Fy_mTSy= np.zeros((len(offset), Fe_mTSe.shape[1], Fe_mTSe.shape[2], Y_TSye.shape[-2]), dtype=Fe_mTSe.dtype)
        for i,o in enumerate(offset):
            # +offset -> Y is later than it 'should' be
            if o == 0:
                Fyi = np.einsum("mTEe,TEYe->mTEY", Fe_mTSe, Y_TSye, dtype=Fe_mTSe.dtype)
                Fy_mTSy[i,...] = Fyi.astype(Fe_mTSe.dtype)

            elif o > 0: # offset>0 -> shift Y FORWARDS in time, t_x=offset -> t_y=0 => t_x=0 -> t_y=-offset
                Fyi = np.einsum("mTEe,TEYe->mTEY", Fe_mTSe[..., o: ,: ], Y_TSye[..., :-o , :, :], dtype=Fe_mTSe.dtype)
                Fy_mTSy[i, ..., o:, :] = Fyi.astype(Fe_mTSe.dtype)

            else: # offset<0 -> shift Y BACKWARDS in time, t_x=0 -> t_y=offset => t_x=-offset -> t_y=0
                Fyi = np.einsum("mTEe,TEYe->mTEY", Fe_mTSe[..., :o ,:], Y_TSye[..., -o: , :, :], dtype=Fe_mTSe.dtype)
                Fy_mTSy[i, ..., :o, :] = Fyi.astype(Fe_mTSe.dtype)


    # TODO[]: validate these score functions are acrrect.
    # add correction for other measures
    if outputscore == 'sse' or outputscore == 'msse':
        YR = convYR(Y_TSye,R_mket,offset) # (nM,nTrl,nSamp,nY,nFilt)
        # Apply the correction:
        #  SSE = (wX-Yr).^2
        #      = wX**2 - 2 wXYr + Yr**2
        #      = wX**2 - 2 Fe*Y + Yr**2
        #  -sse/2= Fy - Yr**2/2 - wX**2/2
        #  take negative, so big (i.e. 0) is good, and drop constant over Y and divide by 2 ->
        #  -SSE/2 = Fy - Yr**2/2 - wX**2/2 &= Fe*Y - Yr**2/2
        rYYr = np.sum(YR*YR,-1)
        if outputscore == 'msse': 
            # TODO[]: validate this scaling is actually optimal, i.e. minimal SSE
            # use optimal per-output scaling which minimizes the SSE (at the end)
            alpha = .5* np.sum(Fy_mTSy,-2,keepdims=True) / np.sum(rYYr,-2,keepdims=True)
        else:
            alpha = 1
        Fy_mTSy = Fy_mTSy - rYYr * alpha / 2

    elif outputscore == 'corr': # correlation based scoring...  hard to do in a summable way...
        YR = convYR(Y_TSye,R_mket,offset) # (nM,nTrl,nSamp,nY,nFilt)
        # Apply the correction:
        #  corr = (wXYr / sqrt(wXX**2) / sqrt(Yr**2)
        rYYr = np.sum(YR*YR,-1)
        # use scaling such that the **final** score is the actual correlation
        nrm = np.sqrt(np.sum(rYYr,-2,keepdims=True)) # sqrt(rYYr)
        Fy_mTSy = Fy_mTSy / nrm
    
    elif outputscore == 'ip':
        pass
    
    else:
        raise NotImplementedError("output scoring with {} isn't supported".format(outputscore))
    
    return Fy_mTSy

def dedupY0(Y, zerodup=True, yfeatdim=True, verb=0):
    ''' remove outputs which are duplicates of the first (objID==0) output
    Inputs:
      Y=(tr,ep,Y,e)
      zerodup : bool
         if True, then if mi is the duplicate, then zero the  duplicate, i.e. of Y[...,mi,:]=0
         else, we zero out ojID==0, i.e. Y[...,0,:]=0
    Outputs:
      Y=(tr,ep,Y,e) version of Y with duplicates of 1st row of Y set to 0'''

    # record input shape so can get it back later
    Yshape = Y.shape
    #print("Y={}".format(Yshape))
    
    Y = np.copy(Y) # copy so can modify, w/o killing orginal
    # make the shape we want
    if not yfeatdim: # hack in feature dim
        Y = Y[..., np.newaxis]
    if Y.shape[-2] == 1 or np.all(Y[...,0,:]==0):
        return Y.reshape(Yshape)
    if Y.ndim == 3: # add trial dim if not there
        Y = Y[np.newaxis, :, :, :]
    if Y.ndim > 4:
        # remove the model dimension!
        print("Warning: multiple models not really supported!")
        Y = Y[0,...] if Y.shape[0]==1 else np.sum(Y,0)
        
    for ti in range(Y.shape[0]):
        # Note: horrible numpy hacks to make work & only for idx==0
        sim = np.sum(np.equal(Y[ti, :, 0:1, :], Y[ti, :, 1:, :]), axis=(0, 2))/(Y.shape[1]*Y.shape[3])
        #print("sim={}".format(sim))
        mi = np.argmax(sim)
        if sim[mi] > .95:
            if verb>0 : print("{}) dup {}={} ".format(ti,0,mi+1),end='')
            if zerodup: # zero out the duplicate of objId=0
                Y[ti, :, mi+1, :] = 0
                if verb>0 : print(" {} removed".format(mi+1))
            else: # zero out the objId==0 line
                Y[ti, :, 0, :] = 0
                if verb>0: print(" {} removed".format(0))
                
    # reshape back to input shape
    Y = np.reshape(Y, Yshape)
    return Y

def convWX(X,W):
    ''' apply spatial filter W  to X '''
    W=W.reshape((1,)*(3-W.ndim)+W.shape)
    X=X.reshape((1,)*(3-X.ndim)+X.shape)

    WX_TSk = np.einsum("TSd,mkd->mTSk",X, W, dtype=W.dtype)
    return WX_TSk #(nM,nTrl,nSamp,nfilt)

def convYR(Y_TSye, R_Mket, offset=None):
    ''' compute the convolution of Y with R '''
    if R_Mket is None:
        return Y_TSye
    R_Mket = R_Mket.reshape((1,)*(4-R_Mket.ndim)+R_Mket.shape) # (nM,nfilt,nE,tau)
    Y_TSye = Y_TSye.reshape((1,)*(4-Y_TSye.ndim)+Y_TSye.shape) # (nTr,nSamp,nY,nE)

    if offset is None:
        offset=0
    #print("R={}".format(R.shape))
    #print("Y={}".format(Y.shape))
    
    # Compute the 'template' response
    Y_TStye = window_axis(Y_TSye, winsz=R_Mket.shape[-1], axis=-3) # (nTr,nSamp,tau,nY,nE)
    #print("Yt={} (TStYe)".format(Yt.shape))
    #print("R={} (mfet)".format(R[...,::-1].shape))
    # TODO []: check treating filters correctly
    # TODO []: check if need to time-reverse IRF - A: YES!
    YR_MTSyk = np.einsum("TStYe,mket->mTSYk", Y_TStye, R_Mket[...,::-1], casting='unsafe', dtype=R_Mket.dtype) # (nM,nTr,nSamp,nY,nE)
    #print("YtR={} (mTSYf)".format(YtR.shape))
    # TODO []: correct edge effect correction, rather than zero-padding....
    # zero pad to keep the output size
    # TODO[]: pad with the *actual* values...
    tmp = np.zeros(YR_MTSyk.shape[:-3]+(Y_TSye.shape[-3],)+YR_MTSyk.shape[-2:], dtype=R_Mket.dtype)
    #print("tmp={}".format(tmp.shape))
    tmp[..., R_Mket.shape[-1]-1-offset:YR_MTSyk.shape[-3]+R_Mket.shape[-1]-1-offset, :, :] = YR_MTSyk
    YR_MTSyk = tmp
    #print("YtR={}".format(YtR.shape))
    return YR_MTSyk #(nM,nTrl,nSamp,nY,nfilt)

def convWXYR(X,Y,W,R,offset=0):
    WX=convWX(X,W) # (nTr,nSamp,nfilt)
    YR=convYR(Y,R,offset) # (nTr,nSamp,nY,nfilt)
    # Sum out  filt dimesion
    WXYR = np.sum(WX[...,np.newaxis,:]*YR,-1) #(nTr,nSamp,nY)
    return WXYR,WX,YR

def sse(X,Y,W,R,offset=0):
    WX = convWX(X,W)
    YR = convYR(Y,R,offset)
    sse = np.sum((WX[...,np.newaxis,:]-YR)**2)
    return sse

def decomp_sse(X,Y,W,R,offset=0):
    WX_TSk=convWX(X,W) # (nTr,nSamp,nfilt)
    YR_TSyk=convYR(Y,R,offset) # (nTr,nSamp,nY,nfilt)
    sse = np.sum((WX_TSk[...,np.newaxis,:]-YR_TSyk)**2)
    WXXW = np.sum(WX**2)
    # Sum out  filt dimesion
    WXYR = np.sum(WX_TSk[...,np.newaxis,:]*YR_TSyk,-1) #(nTr,nSamp,nY)

def corr(X,Y,W,R,offset=0):
    WX_TSk = convWX(X,W)
    nWX_TSk = WX_TSk / np.sqrt(np.sum(WX_TSk**2, axis=(-2,-1), keepdims=True))
    YR_TSek = convYR(Y,R,offset)
    corr = np.sum((nWX_TSk[...,np.newaxis,:]-nYR_TSek)**2) / np.sum(WX_TSk**2)
    return corr


def corr_cov(Cxx_dd, Cyx_yetd, Cyy_yetet, W_kd, R_ket, offset=0):
    """compute correlation in latent space from pre-compute covariance matrices

    Args:
        Cxx_dd ([type]): [description]
        Cyx_yetd ([type]): [description]
        Cyy_yetet ([type]): [description]
        W_kd ([type]): [description]
        R_ket ([type]): [description]
        offset (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """    
    assert offset==0, "None-zero offset not supported yet"
    WXXW_k  = np.einsum("kd,de,ke->k", W_kd,Cxx_dd,W_kd)
    RYYR_yk = np.einsum("ket,yetfu,kfu->yk",R_ket,Cyy_yetet,R_ket)
    RYXW_yk = np.einsum("ket,yetd,kd->yk",R_ket,Cyx_yetd,W_kd)
    corr_yk = RYXW_yk / np.sqrt(RYYR_yk) / np.sqrt(WXXW_k)
    corr_y  = np.sum(RYXW_yk,-1) / np.sqrt(np.sum(RYYR_yk,-1)) / np.sqrt(np.sum(WXXW_k))
    return corr_y, corr_yk


def plot_Fy(Fy,cumsum=True, label=None, legend=False, maxplots=25):
    import matplotlib.pyplot as plt
    import numpy as np
    '''plot the output score function'''
    if cumsum:
        Fy = np.cumsum(Fy.copy(),-2)
    if label is None:
        label = ''
    plt.clf()
    nPlts=min(maxplots,Fy.shape[0])
    if Fy.shape[0]/2 > nPlts:
        tis = np.linspace(0,Fy.shape[0]/2-1,nPlts,dtype=int)
    else:
        tis = np.arange(0,nPlts,dtype=int)
    ncols = int(np.ceil(np.sqrt(nPlts)))
    nrows = int(np.ceil(nPlts/ncols))
    #fig, plts = plt.subplots(nrows, ncols, sharex='all', sharey='all', squeeze=False)
    axploti = ncols*(nrows-1)
    ax = plt.subplot(nrows,ncols,axploti+1)
    Yerr = np.any(Fy[...,-1,1:] > Fy[...,-1,:1],-1)
    for ci,ti in enumerate(tis):
        # make the axis
        if ci==axploti: # common axis plot
            pl = ax
        else: # normal plot
            pl = plt.subplot(nrows,ncols,ci+1,sharex=ax, sharey=ax) # share limits
            plt.tick_params(labelbottom=False,labelleft=False) # no labels        
        #pl = plts[ci//ncols, ci%ncols]
        pl.plot(Fy[ti,:,1:],color='.5')
        pl.plot(Fy[ti,:,0:1],'k')
        
        pl.set_title("{}{}".format(ti," " if Yerr[ti] else "*"))
        pl.grid(True)
    if legend:
        pl.legend(range(Fy.shape[-1]-1))
    plt.suptitle('{}\n {} Fy {}/{} correct'.format(label,"cumsum" if cumsum else "", sum(np.logical_not(Yerr)),len(Yerr)))
    
def plot_Fycomparsion(Fy,Fys,ti=0):    
    import matplotlib.pyplot as plt
    import numpy as np
    from normalizeOutputScores import normalizeOutputScores
    from decodingSupervised import decodingSupervised
    plt.subplot(231); plt.cla(); plt.plot(Fy[ti,:,:],label='Fy');  plt.title('Inner-product'); 

    # Ptgt every sample, accum from trial start
    ssFy,varsFy,N,nEp,nY=normalizeOutputScores(Fy, minDecisLen=-1, filtLen=5)
    plt.subplot(232); plt.cla(); plt.plot(np.cumsum(Fy[ti,:,:],-2),label='sFy');plt.plot(varsFy[ti,1:],'k-',linewidth=5,label='scale'); plt.title('cumsum(Inner-product)');  plt.title('cumsum(Inner-product)');
    Yest,Perr,Ptgt,decismdl,decisEp=decodingSupervised(Fy, minDecisLen=-1, marginalizemodels=False, filtLen=5)
    plt.subplot(233); plt.cla(); plt.plot(Ptgt[ti,:,:],label='Ptgt');  plt.title('Ptgt');

    #X2 = np.sum(X**2,axis=-1,keepdims=True); Fys=Fys-X2/2     #  include norm of X to be sure.
    plt.subplot(234); plt.cla(); plt.plot(Fys[ti,:,:],label='sFy');  plt.title('-SSE'); 

    # Ptgt every sample, accum from trial start
    ssFy,varsFy,N,nEp,nY=normalizeOutputScores(Fys, minDecisLen=-1, filtLen=5)
    Yest,Perr,Ptgt,decismdl,decisEp=decodingSupervised(Fys, minDecisLen=-1, marginalizemodels=False, filtLen=5)
    plt.subplot(235); plt.cla(); plt.plot(np.cumsum(Fys[ti,:,:],-2),label='sFy'); plt.plot(varsFy[ti,1:],'k-',linewidth=5,label='scale'); plt.title('cumsum(-SSE)');
    plt.subplot(236); plt.cla(); plt.plot(Ptgt[ti,:,:],label='Ptgt');  plt.title('Ptgt(-SSE)'); 


def plot_outputscore(X,Y,W=None,R=None,offset=0):
    import matplotlib.pyplot as plt
    import numpy as np
    from model_fitting import MultiCCA
    from scoreStimulus import scoreStimulus
    from scoreOutput import scoreOutput
    from decodingCurveSupervised import decodingCurveSupervised
    ''' plot the factored model to visualize the fit and the scoring functions'''
    if W is None:
        cca = MultiCCA(tau=R,evtlabs=None)
        if X.ndim < 3: # BODGE:
            cca.fit(X[np.newaxis,...], Y[np.newaxis,...])
        else:
            cca.fit(X, Y)
        W=cca.W_
        R=cca.R_

        Fy=cca.predict(X,Y)
        (_) = decodingCurveSupervised(Fy)

        
    WXYR,WX,YR=convWXYR(X,Y,W,R,offset)
    
    plt.clf();
    plt.subplot(511);plt.plot(np.squeeze(WX));plt.grid();plt.title("WX");
    plt.subplot(512);plt.plot(Y[...,0],label='Y');plt.plot(np.squeeze(R).T,'k',label='R',linewidth=5);plt.grid();plt.title("Y");
    plt.subplot(513);plt.plot(np.squeeze(YR[...,0]));plt.grid();plt.title("YR");
    
    plt.subplot(5,3,10);plt.plot(np.squeeze(WXYR));plt.grid();plt.title("WXYR")
    plt.subplot(5,3,11);plt.plot(np.squeeze(np.cumsum(WXYR,-2)));plt.grid();plt.title("cumsum(WXYR)")

    err = WX[...,np.newaxis,:]-YR
    sse = np.sum(err**2,-1)

    plt.subplot(5,3,12);plt.plot(np.squeeze(np.cumsum(-sse,-2)));plt.grid();plt.title("cumsum(sse)")

    #cor = np.cumsum(WXYR,-2)/np.sqrt(np.cumsum(np.sum(YR**2,-1),-2))
    #plt.subplot(5,3,12);plt.cla();plt.plot(np.squeeze(np.cumsum(-sse,-2)));plt.grid();plt.title("corr")

    Fe = scoreStimulus(X,W,R)
    Fy = scoreOutput(Fe,Y,R=R,outputscore='ip')
    Fys = scoreOutput(Fe,Y,R=R,outputscore='sse')

    plt.subplot(5,3,13);plt.plot(np.squeeze(Fy));plt.grid();plt.title("Fy")
    plt.subplot(5,3,14);plt.plot(np.squeeze(np.cumsum(Fy,-2)));plt.grid();plt.title("cumsum(Fy)")
    plt.subplot(5,3,15);plt.plot(np.squeeze(np.cumsum(2*Fys,-2)));plt.grid();plt.title("cumsum(Fy(sse))")


#@function
def testcases():
    from mindaffectBCI.decoder.utils import testSignal
    from mindaffectBCI.decoder.scoreOutput import scoreOutput, plot_outputscore, convWX, convYR, convWXYR
    from mindaffectBCI.decoder.scoreStimulus import scoreStimulus
    from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
    from mindaffectBCI.decoder.normalizeOutputScores import normalizeOutputScores
    import numpy as np
    import matplotlib.pyplot as plt

    # Fe  = (nM,nTrl,nSamp,nE) similarity score for each event type for each stimulus
    # Ye  = (nTrl,nSamp,nY,nE) Indicator for which events occured for which outputs
    nE=2
    nSamp=100
    nTrl=30
    nY=20
    nM=1
    sigstr = 1e-2
    N=np.random.standard_normal((nM,nTrl,nSamp,nE))
    Ye=np.random.standard_normal((nTrl,nSamp,nY,nE))
    # make Ye[0]=Fe
    Fe = N + Ye[...,0,:] * sigstr
    print("Fe={}".format(Fe.shape))
    print("Ye={}".format(Ye.shape))
    Fy = scoreOutput(Fe,Ye) # (nM,nTrl,nSamp,nY)
    print("Fy={}".format(Fy.shape))
    import matplotlib.pyplot as plt
    sFy=np.cumsum(Fy,axis=-2)
    plt.clf();plt.plot(sFy[0,0,:,:]);plt.xlabel('epoch');plt.ylabel('output');plt.show()

    # try with range offsets between Fe, Ye
    offset=2
    offsets=np.arange(-6,6)
    Fe = N 
    Fe[..., offset: , :] = Fe[..., offset:, :] + Ye[..., :-offset, 0, :]
    Fy = scoreOutput(Fe,Ye, offset=offsets) # (nM,nTrl,nSamp,nY), nM=num-offset
    print("Fy={}".format(Fy.shape))
    import matplotlib.pyplot as plt
    sFy=np.cumsum(Fy,axis=-2)
    plt.clf();
    for i,o in enumerate(offsets):
        plt.subplot(len(offsets),1,i+1)
        plt.plot(sFy[i,0,:,:]);plt.xlabel('epoch');plt.ylabel('output');
        plt.title("offset={}".format(o))
    plt.show()


    # more complex example with actual signal/noise
    irf=(1,1,-1,-1,0,0,0,0,0,0)
    X,Y,st,A,R = testSignal(nTrl=1,nSamp=1000,d=1,nE=1,nY=10,isi=2,irf=irf,noise2signal=0)
    Y=Y[...,0] # (nTrl,nSamp,nY)
    W=np.linalg.pinv(A)
    
    from mindaffectBCI.decoder.model_fitting import MultiCCA
    cca = MultiCCA(tau=len(irf), offset=0, rank=1, evtlabs=1)
    cca.fit(X,Y[...,0:1])
    Fy = cca.predict(X, Y, dedup0=True)
    (_) = decodingCurveSupervised(Fy)
    W=cca.W_
    R=cca.R_
    b=cca.b_

    WX=convWX(X,W) # (nTr,nSamp,nfilt)
    YR=convYR(Y[...,0:1,np.newaxis],R,0) # (nTr,nSamp,nY,nfilt)
    
    plot_outputscore(X[0,...],Y[0,:,:,np.newaxis],W,R)
    plt.show()
    
    # add a correlated output
    Y[:,:,1]=Y[:,:,0]*.5
    plot_outputscore(X[0,...],Y[0,:,:,np.newaxis],W,R)
    plt.show()
    

def datasettest():
    # N.B. imports in function to avoid import loop..
    from mindaffectBCI.decoder.offline.datasets import get_dataset
    from mindaffectBCI.decoder.model_fitting import MultiCCA, BwdLinearRegression, FwdLinearRegression
    from mindaffectBCI.decoder.analyse_datasets import debug_test_dataset
    from mindaffectBCI.decoder.scoreOutput import plot_outputscore
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised

    tau_ms=20
    offset_ms=0
    rank=8
    evtlabs=None
    l,f,_=get_dataset('mark_EMG')
    X,Y,coords=l(f[1],whiten=True,filterband=((0,10),(45,55),(200,-1)),filterbank=((10,20),(20,45),(55,95),(105,200)))
    oX=X.copy(); oY=Y.copy()

    X=oX.copy()
    Y=oY.copy()

    # test with reduced number  classes?
    nY=8
    Y=Y[:,:,:nY+1,:nY]

    plt.close('all')
    debug_test_dataset(X, Y, coords, tau_ms=tau_ms, offset_ms=offset_ms, evtlabs=evtlabs, rank=8, outputscore='ip', model='cca')
        
    fs = coords[1]['fs']
    tau = min(X.shape[-2],int(tau_ms*fs/1000))
    offset=int(offset_ms*fs/1000)
    cca = MultiCCA(tau=tau, offset=offset, rank=rank, evtlabs=evtlabs)
    cca.fit(X,Y)
    Fy = cca.predict(X, Y, dedup0=True)
    (_) = decodingCurveSupervised(Fy)
    W=cca.W_
    R=cca.R_
    b=cca.b_

    plot_outputscore(X[0,...],Y[0,...],W,R)

if __name__=="__main__":
    testcases()
