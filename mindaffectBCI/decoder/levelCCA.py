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


def levelsCCA_cov(Cxx_dd=None, Cyx_yetd=None, Cyy_tyeye=None,
                  reg=1e-9, rank=1, CCA=True, rcond=(1e-8,1e-8), symetric=False, tol=1e-6, max_iter:int=10):
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
    Cyy_yetyet = Mtyeye2Myetyet(Cyy_tyeye)
    
    # pre-compute the spatial whitener part.
    isqrtCxx_dd, _ = robust_whitener(Cxx_dd,reg[0],rcond[0],symetric=True)
    #2.c) Whiten Cxy
    CyxisqrtCxx_yetd = np.einsum('yetd,df->yetf',Cyx_yetd,isqrtCxx_dd)


    J = 1e9
    S_y = np.ones((nY),dtype=Cyx_yetd.dtype)/nY # seed weighting over y
    #S_y[0]=1; S_y[1:]=0;
    for iter in range(max_iter):

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
        l_k = l_k[l_idx[:r]]
        W_kd = W_kd[l_idx[:r],:]

        # 2.6) pre-apply the whitener so can apply directly to input
        R_ket = np.einsum('ket,etfu->kfu',R_ket,isqrtCyy_etet)
        W_kd = np.einsum('kd,df->kf',W_kd,isqrtCxx_dd)
        
        #3) Update componenst for the output weighting
        rCyyr_yy = np.einsum("ket,yetzfu,kfu->yz",R_ket,Cyy_yetyet,R_ket) 
        rCyxw_y = np.einsum('kd,yetd,ket->y',W_kd,Cyx_yetd,R_ket)

        # 4) Solve for optimal S, under the positivity constraint using
        # multi-updates
        oS_y = S_y
        if False:
            S_y = np.linalg.pinv(rCyyr_yy, hermitian=True) @ rCyxw_y
            #S_y = np.maximum(S_y,0)
            S_y = S_y / np.sqrt(S_y@S_y.T)
        else:
            S_y = 
        #print("{:3d}) S_y = {}".format(iter,np.array_str(S_y,precision=3)))

        # 5) Goodness of fit tracking
        oJ = J
        # TODO[]: object value computation
        J = 0
        dS_y = np.sum(np.abs(S_y - oS_y))
        print("{:3d}) dS_y = {}".format(iter,dS_y,np.array_str(S_y,precision=3)))
        if dS_y < tol:
            break

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


def testcase_levelsCCA(X_TSd,Y_TSye,tau=15,offset=0,center=True,unitnorm=True,zeropadded=True,badEpThresh=4):
    from mindaffectBCI.decoder.utils import testNoSignal, testSignal, sliceData, sliceY
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_summary_statistics, plot_factoredmodel
    from mindaffectBCI.decoder.multipleCCA import multipleCCA
    from mindaffectBCI.decoder.scoreOutput import scoreOutput
    from mindaffectBCI.decoder.scoreStimulus import scoreStimulus
    import matplotlib.pyplot as plt


    Cxx_dd0, Cyx_yetd0, Cyy_yetet0 = updateSummaryStatistics(X_TSd,Y_TSye[...,0:1,:],tau=tau,offset=offset,center=center,unitnorm=unitnorm,zeropadded=zeropadded,badEpThresh=badEpThresh)

    # plot SS
    plt.figure()
    plot_summary_statistics(Cxx_dd0,Cyx_yetd0,Cyy_yetet0)
    plt.suptitle(' uss Y_true' )
    plt.show(block=False)

    print(" With all Y")
    
    Cxx_dd, Cyx_yetd, Cyy_tyeye = levelsSummaryStatistics(X_TSd,Y_TSye,tau=15,offset=0,center=True,unitnorm=True,zeropadded=True)

    print(f"Cxx_dd {Cxx_dd.shape}")
    print(f"Cyz_yetd {Cyx_yetd.shape}")
    print(f"Cyy_tyeye {Cyy_tyeye.shape}")

    # plot SS
    plt.figure()
    plot_summary_statistics(Cxx_dd,Cyx_yetd,Cyy_tyeye)
    plt.suptitle(' levels allY' )
    plt.show(block=False)
    
    J, W_kd, R_ket, S_y = levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, rcond=1e-6, reg=1e-4, symetric=True, max_iter=50)

    print(f"W_kd {W_kd.shape}")
    print(f"R_ket {R_ket.shape}")
    print(f"S_y {S_y.shape} = {S_y}")

    # plot soln
    plt.figure()
    plot_factoredmodel(W_kd,R_ket)
    plt.suptitle(' levels allY soln') 
    
    # levels strength in separate plot
    plt.figure()
    plt.plot(S_y)
    plt.suptitle(' levels output weight') 

    plt.show()    

def testcase():
    from mindaffectBCI.decoder.utils import testNoSignal, testSignal, sliceData, sliceY
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_summary_statistics, plot_factoredmodel
    from mindaffectBCI.decoder.multipleCCA import multipleCCA
    from mindaffectBCI.decoder.scoreOutput import scoreOutput
    from mindaffectBCI.decoder.scoreStimulus import scoreStimulus
    import matplotlib.pyplot as plt

    #from multipleCCA import *
    if False:
        X_TSd, Y_TSye, st = testNoSignal()
    else:
        X_TSd, Y_TSye, st, A, B = testSignal(nY=40, tau=10, noise2signal=7)

    #testcase_levelsCCA_vs_multiCCA(X_TSd,Y_TSye,tau=15,offset=0)
        
    testcase_levelsCCA(X_TSd,Y_TSye,tau=15,offset=0)


if __name__=="__main__":
    testcase()
