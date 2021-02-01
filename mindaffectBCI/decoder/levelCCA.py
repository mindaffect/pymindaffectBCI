from mindaffectBCI.decoder.multipleCCA import robust_whitener
from mindaffectBCI.decoder.updateSummaryStatistics import updateCxx, updateCxy, compCyy_diag
from mindaffectBCI.decoder.utils import zero_outliers
import numpy as np


def levelsCCA(X_TSd,Y_TSye,tau,offset=0,reg=1e-9,rank=1,CCA=True,rcond=(1e-8,1e-8), symetric=False, center=True, unitnorm=True, zeropadded=True):
    Cxx_dd, Cyx_yetd, Cyy_tyeye = levelsSummaryStatistics(X_TSd, Y_TSye, tau, offset, center, unitnorm, zeropadded)
    
    return levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, reg=reg, rank=rank, rcond=rcond, symetric=symetric)


def levelsSummaryStatistics(X_TSd,Y_TSye,tau, offset:int=0, center:bool=True, unitnorm:bool=True, zeropadded:bool=True, badEpThresh:float=4):
    X_TSd, Y_TSye = zero_outliers(X_TSd, Y_TSye, badEpThresh)
    # get the summary statistics
    Cxx_dd = updateCxx(None,X_TSd,None,tau=tau, offset=offset, center=center, unitnorm=unitnorm)
    Cyx_yetd = updateCxy(None, X_TSd, Y_TSye, None, tau=tau, offset=offset, center=center, unitnorm=unitnorm)
    Cyy_tyeye = compCyy_diag(Y_TSye, tau, zeropadded=zeropadded, unitnorm=unitnorm, perY=False)
    return Cxx_dd, Cyx_yetd, Cyy_tyeye


def levelsCCA_cov(Cxx_dd=None, Cyx_yetd=None, Cyy_tyeye=None, S_y=None,
                  reg=1e-9, rank=1, CCA=True, rcond=(1e-8,1e-8), symetric=False, tol=1e-3, 
                  max_iter:int=100, eta:float=.5, syopt=None, showplots=False):
    """
    Compute multiple CCA decompositions using the given summary statistics
      [J,W,R]=multiCCA(Cxx,Cyx,Cyy,regx,regy,rank,CCA)
    Inputs:
      Cxx  (d,d): current data covariance
      Cyx  (nY,nE,tau,d): current per output ERPs
      Cyy  (tau,nY,nE,nY,nE): compressed cov for each output at different time-lags
      reg  = (1,) :float or (2,):float linear weighting reg strength or separate values for Cxx and Cyy 
            OR (d,) regularisation ridge coefficients for Cxx  (0)
      rank (float): number of top cca components to return (1)
      CCA (bool): [bool] or (2,):bool  flag if we normalize the spatial dimension. (true)
      rcond (float): tolerance for singular eigenvalues.        (1e-4)
               or [2x1] separate rcond for Cxx (rcond(1)) and Cyy (rcond(2))
               or [2x1] (-1<-0) negative values = keep this fraction of eigen-values
               or (2,1) <-1 keep this many eigenvalues
      symetric (bool): use symetric whitener?

    Returns:
      J (float): optimisation objective scores
      W_dk (rank,d): spatial filters for each output
      R_etk (rank,nE,tau): responses for each stimulus event for each output
      S_y (nY): weighting over levels
    """
    import matplotlib.pyplot as plt
    rank = int(max(1, rank))
    if not hasattr(reg, '__iter__'):
        reg = (reg, reg)  # ensure 2 element list
    if not hasattr(CCA, '__iter__'):
        CCA = (CCA, CCA)  # ensure 2 element list
    if not hasattr(rcond, '__iter__'):
        rcond = (rcond, rcond)  # ensure 2 element list    

    # extract dimension sizes
    tau = Cyy_tyeye.shape[0]
    nE = Cyy_tyeye.shape[-1]
    nY = Cyy_tyeye.shape[-2]
    nD = Cxx_dd.shape[0]

    # pre-expand Cyy
    # TODO[]: remove the need to actually do this -- as it's a ram/compute hog
    Cyy_yetyet = Mtyeye2Myetyet(Cyy_tyeye)
    
    # pre-compute the spatial whitener part.
    isqrtCxx_dd, _ = robust_whitener(Cxx_dd,reg[0],rcond[0],symetric=True)
    #2.c) Whiten Cxy
    CyxisqrtCxx_yetd = np.einsum('yetd,df->yetf',Cyx_yetd,isqrtCxx_dd)
    isqrtCxxCxxisqrtCxx_dd = np.einsum('df,de,eg->fg',isqrtCxx_dd,Cxx_dd,isqrtCxx_dd)

    J = 1e9
    Js = np.zeros((max_iter,2))
    Js[:]=np.nan
    if showplots:
        plt.figure()
        plt.pause(.1)

    if S_y is None:
        S_y = np.ones((nY),dtype=Cyx_yetd.dtype)/nY  # seed weighting over y
    #S_y[0]=1; S_y[1:]=0;
    for iter in range(max_iter):
        oJ = J
        oS_y = S_y

        # 1) Apply the output weighting and make 2d Cyy
        sCyys_etet = np.einsum("y,yetzfu,z->etfu",S_y,Cyy_yetyet,S_y) 
        sCyxisqrtCxx_etd = np.einsum('yetd,y->etd',CyxisqrtCxx_yetd,S_y)
            
        #2) CCA solution
        #2.1) Make 2d
        sCyys_et_et = np.reshape(sCyys_etet,(nE*tau,nE*tau)) # (nE,tau,nE,tau) -> ((nE*tau),(nE*tau))
        #2.2) Compute updated right-whitener
        isqrtCyy_et_et, _ = robust_whitener(sCyys_et_et,reg[1],rcond[1],symetric=True)
        isqrtCyy_etet = isqrtCyy_et_et.reshape((nE,tau,nE,tau))
        #2.3) right-whiten Cyx
        isqrtCxxsCyxisqrtCyy_etd = np.einsum('etfu,etd->fud',isqrtCyy_etet,sCyxisqrtCxx_etd)
        # 2.3) SVD
        isqrtCxxsCyxisqrtCyy_et_d = np.reshape(isqrtCxxsCyxisqrtCyy_etd,(nE*tau, nD)) # (nE,tau,d) -> ((nE*tau),d)
        R_et_k, l_k, W_kd = np.linalg.svd(isqrtCxxsCyxisqrtCyy_et_d, full_matrices=False)  
        R_ket = np.moveaxis(R_et_k.reshape((nE, tau, R_et_k.shape[-1])),-1,0) #((nE*tau),k) -> (nE,tau,k) -> (k,nE,tau)
        # 2.4) get the largest components
        l_idx = np.argsort(l_k)[::-1]  # N.B. DECENDING order
        r = min(len(l_idx), rank)  # guard rank>effective dim
        # 2.5) extract the desired rank sub-space solutions
        R_ket = R_ket[l_idx[:r],:,:] 
        l_k = l_k[l_idx[:r]]  # l_k = coor for this component
        W_kd = W_kd[l_idx[:r],:]

        # 2.6) pre-apply the whitener + scaling so can apply directly to input
        R_ket = np.einsum('ket,etfu->kfu',R_ket,isqrtCyy_etet) #* np.sqrt(l_k[:,np.newaxis, np.newaxis])
        W_kd = np.einsum('kd,df->kf',W_kd,isqrtCxx_dd) #* np.sqrt(l_k[:,np.newaxis])
        
        #3) Update componenst for the output weighting
        wCxxw  = np.einsum('kd,de,ke', W_kd, Cxx_dd, W_kd) # N.B. should be I_k
        rCyyr_yy = np.einsum("ket,yetzfu,kfu->yz",R_ket,Cyy_yetyet,R_ket) # N.B. should be I_k
        rCyxw_y = np.einsum('ket,yetd,kd->y',R_ket,Cyx_yetd,W_kd) 

        # 4) Solve for optimal S, under the positivity constraint using
        # multi-updates
        J = wCxxw - 2 * rCyxw_y @ S_y + S_y.T @ rCyyr_yy @ S_y # should be 1 - 2* ip + 1
        J_wr = J
        # TODO [X] : non-negative least squares
        # TODO [] : norm-constrained optimization
        S_y = nonneglsopt(wCxxw, rCyxw_y, rCyyr_yy, S_y, syopt)
        S_y = S_y / np.sum(S_y) # maintain the norm
        #print("{:3d}) S_y = {}".format(iter,np.array_str(S_y,precision=3)))

        # 5) Goodness of fit tracking
        J = wCxxw - 2 * rCyxw_y @ S_y + S_y.T @ rCyyr_yy @ S_y
        J_s = J
        Js[iter,:]=[J_wr,J_s]
        if showplots and iter%100 == 0:
            plt.cla()
            plt.plot(Js[:iter,:])
            plt.legend(('pre','post'))
            plt.pause(.001)
        deltaS_y = np.sum(np.abs(S_y - oS_y))
        deltaJ = np.abs(J-oJ)
        if iter<10 or iter%100==0:
            print("{:3d}) |S_y|={:4.3f} dS_y={:4.3f}  J={:4.3f} dJ={:4.3f}".format(iter,np.sum(S_y),deltaS_y,J_s,deltaJ))
        #print("{:3d}) dS_y = {}".format(iter,dS_y,np.array_str(S_y,precision=3)))

        # convergence testing
        if deltaS_y < tol or deltaJ < tol:
            print("{:3d}) |S_y|={:4.3f} dS_y={:4.3f}  J={:4.3f} dJ={:4.3f}".format(iter,np.sum(S_y),deltaS_y,J_s,deltaJ))
            break
            
    if showplots:
        print("{:3d}) |S_y|={:4.3f} dS_y={:5.4f}  J={:4.3f} dJ={:5.4f}".format(iter,np.sum(S_y),deltaS_y,J_s,deltaJ))
        plt.cla()
        plt.plot(Js[:iter,:])
        plt.legend(('pre','post'))
        plt.pause(.001)

    # include relative component weighting directly in the  Left/Right singular values
    nl_k = (l_k / np.max(l_k)) if np.max(l_k)>0 else np.ones(l_k.shape,dtype=l_k.dtype)  
    W_kd = W_kd * np.sqrt(nl_k[:,np.newaxis])
    R_ket = R_ket * np.sqrt(nl_k[:,np.newaxis, np.newaxis]) 

    return J, W_kd, R_ket, S_y


def Mtyeye2Myetyet(M_tyeye):
    t=M_tyeye.shape[0]
    y=M_tyeye.shape[1]
    e=M_tyeye.shape[2]
    
    M_yetyet = np.zeros((y,e,t,y,e,t),dtype=M_tyeye.dtype)
    # fill in the block diagonal entries
    for i in range(t):
        M_yetyet[:,:,i,:,:,i] = M_tyeye[0,:,:,:,:]
        for j in range(i+1,t):
            M_yetyet[:,:,i,:,:,j] = M_tyeye[j-i,:,:,:,:]
            # lower diag, transpose the event types
            M_yetyet[:,:,j,:,:,i] = M_tyeye[j-i,:,:,:,:].swapaxes(-3,-1).swapaxes(-4,-2)
    return M_yetyet


def nonneglsopt(C_xx,C_yx,C_yy,s_y,syopt='negridge',max_iter=30,tol=1e-4,verb:int=0):
    if syopt is None:
        syopt = 'negridge'

    J = C_xx - 2*C_yx@s_y + s_y.T@C_yy@s_y
    for iter in range(max_iter):
        oJ=J
        if syopt=='ls': 
            # simple least squares
            s_y = np.linalg.pinv(C_yy, hermitian=True) @ C_yx
        elif syopt=='lspos': 
            s_y = np.linalg.pinv(C_yy, hermitian=True) @ C_yx
            s_y = np.maximum(s_y,0)
        elif syopt=='expgrad': # exponiated gradient - maintain non-negative
            ds_y = C_yy @ s_y - C_yx
            eds_y = np.exp(-eta*ds_y)
            s_y = s_y * eds_y
        elif syopt=='gd': # simple gradient descent
            ds_y = rC_yy @ s_y - C_yx
            s_y = s_y + eta * 1e-2 * ds_y
        elif syopt == 'mu': # mulplicative update
            s_y = s_y * np.abs(C_yx) / (C_yy@s_y + 1e-6)
        elif syopt == 'irwls' or syopt == 'negridge':
            # iterative re-weighted least squares to enforce the non-negavitivity constraint
            C = C_yy + np.diag(s_y<0).astype(s_y.dtype)*np.mean(np.diag(C_yy))*10
            s_y = np.linalg.pinv(C, hermitian=True) @ C_yx
        else:
            raise ValueError('unknown optimization type: {}'.format(syopt))

        # project onto the sum constraint
        s_y = s_y / np.sum(s_y)

        # updated objective
        J = C_xx - 2*C_yx@s_y + s_y.T@C_yy@s_y
        if verb>0 : print("{:2d} J={}  dJ={}".format(iter,J,oJ-J))
        if np.abs(oJ-J) < tol:
            break

    s_y = np.maximum(1e-6,s_y) # move away from 0 to prevent degeneracies 
    # project onto the sum constraint
    s_y = s_y / np.sum(s_y)
            
    return s_y



def testcase_levelsCCA_vs_multiCCA(X_TSd,Y_TSye,tau=15,offset=0,center=True,unitnorm=True,zeropadded=True,badEpThresh=4):
    from mindaffectBCI.decoder.utils import testNoSignal, testSignal, sliceData, sliceY
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_summary_statistics, plot_factoredmodel
    from mindaffectBCI.decoder.multipleCCA import multipleCCA
    from mindaffectBCI.decoder.scoreOutput import scoreOutput
    from mindaffectBCI.decoder.scoreStimulus import scoreStimulus
    import matplotlib.pyplot as plt

    Cxx_dd0, Cyx_yetd0, Cyy_yetet0 = updateSummaryStatistics(X_TSd,Y_TSye[...,0:1,:],tau=tau,offset=offset,center=center,unitnorm=unitnorm,zeropadded=zeropadded,badEpThresh=badEpThresh)

    print(f"Cxx_dd0 {Cxx_dd0.shape}")
    print(f"Cyz_yetd0 {Cyx_yetd0.shape}")
    print(f"Cyy_yetet0 {Cyy_yetet0.shape}")

    # plot SS
    plt.figure()
    plot_summary_statistics(Cxx_dd0,Cyx_yetd0,Cyy_yetet0)
    plt.suptitle(' uss ' )
    plt.show(block=False)

    Cxx_dd, Cyx_yetd, Cyy_tyeye = levelsSummaryStatistics(X_TSd,Y_TSye[...,0:1,:],tau=tau,offset=offset,center=center,unitnorm=unitnorm,zeropadded=zeropadded,badEpThresh=badEpThresh)
    
    Cyy_yetet = Mtyeye2Myetyet(Cyy_tyeye).squeeze(-3)
    
    print(f"Cxx_dd {Cxx_dd.shape}")
    print(f"Cyx_yetd {Cyx_yetd.shape}")
    print(f"Cyy_tyeye {Cyy_tyeye.shape}")
    print(f"Cyy_yetet {Cyy_yetet.shape}")
    
    # plot SS
    plt.figure()
    plot_summary_statistics(Cxx_dd,Cyx_yetd,Cyy_yetet)
    plt.suptitle(' levels ' )
    plt.show(block=False)

    print("dCxx_dd {}".format(np.max(np.abs(Cxx_dd0 - Cxx_dd))))
    print("dCyx_yetd {}".format(np.max(np.abs(Cyx_yetd0 - Cyx_yetd))))
    print("dCyy_yetet {}".format(np.max(np.abs(Cyy_yetet0 - Cyy_yetet))))


    J, W_kd0, R_ket0    = multipleCCA  (Cxx_dd0, Cyx_yetd0, Cyy_yetet0, rcond=1e-6, reg=1e-4, symetric=True)
    plt.figure()
    plot_factoredmodel(W_kd0,R_ket0)
    plt.suptitle(' mCCA soln') 
    
    J, W_kd, R_ket, S_y = levelsCCA_cov(Cxx_dd,  Cyx_yetd,  Cyy_tyeye,  rcond=1e-6, reg=1e-4, symetric=True, max_iter=1)
    plt.figure()
    plot_factoredmodel(W_kd,R_ket)
    plt.suptitle(' levelsCCA soln')

    print("d W_kd {}".format(np.max(np.abs(W_kd0 - W_kd))))
    print("d R_ket {}".format(np.max(np.abs(R_ket0 - R_ket))))


def debug_levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, syopt=None, label=None, **kwargs):
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_summary_statistics, plot_factoredmodel
    import matplotlib.pyplot as plt
    if syopt is None: syopt='negridge'

    J, W_kd, R_ket, S_y = levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, syopt=syopt, **kwargs)
    print("{}) J={}".format('opt',J))

    print(f"W_kd {W_kd.shape}")
    print(f"R_ket {R_ket.shape}")
    print(f"S_y {S_y.shape} = {S_y}")

    # plot soln
    plt.figure()
    plot_factoredmodel(W_kd,R_ket)
    plt.suptitle('{} {} soln'.format(label,syopt)) 

    # levels strength in separate plot
    plt.figure()
    plt.plot(S_y)
    plt.suptitle('{} {} output weight'.format(label,syopt)) 

    # plot SS
    # pre-expand Cyy
    Cyy_yetyet = Mtyeye2Myetyet(Cyy_tyeye)
    sCyys_etet = np.einsum("y,yetzfu,z->etfu",S_y,Cyy_yetyet,S_y) 
    sCyx_etd = np.einsum('yetd,y->etd',Cyx_yetd,S_y)
    plt.figure()
    plot_summary_statistics(Cxx_dd,sCyx_etd,sCyys_etet[np.newaxis,...])
    plt.suptitle(' {} {}'.format(label,syopt) )
    plt.show(block=False)


def testcase_levelsCCA(X_TSd,Y_TSye,tau=15,offset=0,center=True,unitnorm=True,zeropadded=True,badEpThresh=4):
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_summary_statistics, plot_factoredmodel
    import matplotlib.pyplot as plt

    Cxx_dd0, Cyx_yetd0, Cyy_yetet0 = updateSummaryStatistics(X_TSd,Y_TSye[...,0:1,:],tau=tau,offset=offset,center=center,unitnorm=unitnorm,zeropadded=zeropadded,badEpThresh=badEpThresh)

    # plot SS
    plt.figure()
    plot_summary_statistics(Cxx_dd0,Cyx_yetd0,Cyy_yetet0)
    plt.suptitle(' uss Y_true' )
    plt.show(block=False)

    print(" With all Y")
    
    Cxx_dd, Cyx_yetd, Cyy_tyeye = levelsSummaryStatistics(X_TSd,Y_TSye,tau=tau,offset=offset,center=center,unitnorm=unitnorm,zeropadded=zeropadded)

    print(f"Cxx_dd {Cxx_dd.shape}")
    print(f"Cyz_yetd {Cyx_yetd.shape}")
    print(f"Cyy_tyeye {Cyy_tyeye.shape}")

    # plot SS
    plt.figure()
    plot_summary_statistics(Cxx_dd,Cyx_yetd,Cyy_tyeye)
    plt.suptitle(' levels allY' )
    plt.show(block=False)
    
    # test with single output scores:
    nY = Cyx_yetd.shape[0]
    for yi in range(nY):
        S_y = np.zeros((nY))
        S_y[yi]=1
        J, W_kd, R_ket, S_y = levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, S_y=S_y, rcond=1e-6, reg=1e-4, symetric=True, max_iter=1)
        print("{}) J={}".format(yi,J))


    debug_levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, syopt='negridge')

    #debug_levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, syopt='lspos')

    debug_levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, syopt='mu')

    plt.show()    


def simdata(nTrl=10, nSamp=1000, nY:int=30, tau:int=10, offset:int=0, noise2signal=5):
    from mindaffectBCI.decoder.utils import testNoSignal, testSignal
    import matplotlib.pyplot as plt

    #from multipleCCA import *
    if False:
        X_TSd, Y_TSye, st = testNoSignal()
    elif True:
        X_TSd, Y_TSye, st, A, B = testSignal(nTrl=nTrl, nSamp=nSamp, nY=nY, tau=tau, noise2signal=noise2signal)

    return X_TSd, Y_TSye, None


def loaddata(stopband=((45,65),(5,25,'bandpass')),evtlabs=('re','fe')):
    from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
    from mindaffectBCI.decoder.stim2event import stim2event
    from mindaffectBCI.decoder.analyse_datasets import debug_test_dataset

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
    X_TSd, Y_TSy, coords = load_mindaffectBCI(savefile, stopband=stopband, order=6, ftype='butter', fs_out=100)

    #score, dc, Fy, clsfr, rawFy = debug_test_dataset(X_TSd, Y_TSy, coords,tau_ms=450, evtlabs=evtlabs, model='cca')

    if 'visual_acuity' in savefile:
        X_TSd = X_TSd[10:,...] 
        Y_TSy = Y_TSy[10:,...,1:] # strip true output info
    else:
        # limit to 1st 20 cal trials
        X_TSd = X_TSd[:10,...]
        Y_TSy = Y_TSy[:10,...]

    Y_TSye, labs = stim2event(Y_TSy,evtlabs)

    return X_TSd, Y_TSye, coords[1]['fs']



def testcase(tau_ms=650, offset_ms=0):
    if False:
        X_TSd, Y_TSye, fs = simdata()

    else:
        X_TSd, Y_TSye, fs = loaddata()

    if fs is None: fs=100
    tau = int(tau_ms*fs/1000)
    offset = int(offset_ms*fs/1000)

    #testcase_levelsCCA_vs_multiCCA(X_TSd,Y_TSye,tau=15,offset=0)
        
    testcase_levelsCCA(X_TSd,Y_TSye,tau=tau,offset=offset)


if __name__=="__main__":
    testcase()
