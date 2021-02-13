from mindaffectBCI.decoder.multipleCCA import robust_whitener
from mindaffectBCI.decoder.levelCCA import plot_3factoredmodel, loaddata
from mindaffectBCI.decoder.updateSummaryStatistics import compCyy_diag, compCxx_diag, compCyx_diag, \
                Cxx_diag2full, Cyy_tyeye_diag2full, Cyx_diag2full, \
                plot_factoredmodel, plot_erp, plot_summary_statistics
from mindaffectBCI.decoder.utils import zero_outliers
import numpy as np

# TODO[] : work out the 'optimal' order of indices in the various arrays


def firCCA(X_TSd,Y_TSye,tau,offset=0,reg=1e-9,rank=1,rcond=(1e-8,1e-8), symetric=False, center=True, unitnorm=True, zeropadded=True):
    Cxx_tdd, Cyx_tyed, Cyy_tyeye = firSummaryStatistics(X_TSd, Y_TSye, tau, offset, center, unitnorm, zeropadded)
    return firCCA_cov(Cxx_tdd, Cyx_tyed, Cyy_tyeye, reg=reg, rank=rank, rcond=rcond, symetric=symetric)


def firSummaryStatistics(X_TSd,Y_TSye, tau, Cxx_tdd=None, Cyx_tyed=None, Cyy_tyeye=None, 
                         offset:int=0, center:bool=True, unitnorm:bool=False, zeropadded:bool=True, badEpThresh:float=4):
    X_TSd, Y_TSye = zero_outliers(X_TSd, Y_TSye, badEpThresh)
    # get the summary statistics
    nCxx_tdd = compCxx_diag(X_TSd, tau[0], offset=offset, center=center, unitnorm=unitnorm)
    nCyx_tyed, taus = compCyx_diag(X_TSd, Y_TSye, tau, offset=offset, center=center, unitnorm=unitnorm)
    nCyy_tyeye = compCyy_diag(Y_TSye, tau[1], zeropadded=zeropadded, unitnorm=unitnorm, perY=False)
    # accmulate if wanted
    Cxx_tdd = nCxx_tdd if Cxx_tdd is None else nCxx_tdd + Cxx_tdd
    Cyx_tyed = nCyx_tyed if Cyx_tyed is None else nCyx_tyed + Cyx_tyed
    Cyy_tyeye = nCyy_tyeye if Cyy_tyeye is None else nCyy_tyeye + Cyy_tyeye
    # return the updated info
    return Cxx_tdd, Cyx_tyed, Cyy_tyeye

def sse(Cxx_fdfd, Cyx_tyefd, Cyy_yetyet, W_kd, R_kte, f_f, S_y):
    if f_f is None: f_f = np.ones((1),dtype=Cxx_fdfd.dtype)
    if S_y is None: S_y = np.ones((1),dtype=Cyy_yetyet.dtype)
    fwCxxwf = np.einsum('f,kd,fdhg,kg,h',f_f,W_kd,Cxx_fdfd,W_kd,f_f, optimize=True)
    srCyxwf = np.einsum('y,kte,tyefd,f,kd',S_y,R_kte,Cyx_tyefd,f_f,W_kd, optimize=True)
    srCyyrs = np.einsum('y,kte,yetzfu,kuf,z',S_y,R_kte,Cyy_yetyet,R_kte,S_y, optimize=True)
    return fwCxxwf - 2 * srCyxwf + srCyyrs

def lsopt(Cxx, Cyx, Cyy, w, ftopt):
    w = Cyx @ np.linalg.pinv(Cxx,hermitian=True)
    return w

def firCCA_cov(Cxx_fdd=None, Cyx_tyed=None, Cyy_tyeye=None, S_y=None, f_f=None,
               reg:float=1e-9, rank:int=1, rcond:float=(1e-8,1e-8), 
               symetric:bool=False, tol:float=1e-3, 
               max_iter:int=100, eta:float=.5, syopt:str=None, ftopt:str=None, showplots:bool=False):
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
    if not hasattr(rcond, '__iter__'):
        rcond = (rcond, rcond)  # ensure 2 element list    

    # extract dimension sizes
    tau_x = Cxx_fdd.shape[0]
    tau_y = Cyy_tyeye.shape[0]
    nE = Cyy_tyeye.shape[-1]
    nY = Cyy_tyeye.shape[-2]
    assert nY==1
    nD = Cxx_fdd.shape[-1]

    # pre-expand Cyy
    # TODO[]: remove the need to actually do this -- as it's a ram/compute hog
    Cyy_yetyet = Cyy_tyeye_diag2full(Cyy_tyeye)
    Cxx_fdfd = Cxx_diag2full(Cxx_fdd)
    Cyx_tyefd = Cyx_diag2full(Cyx_tyed,(tau_x,tau_y))
    
    J = 1e9
    Js = np.zeros((max_iter,2))
    Js[:]=np.nan
    if showplots:
        plt.figure()
        plt.pause(.1)

    if f_f is None:
        f_f = np.zeros((tau_x),dtype=Cyx_tyed.dtype)
        f_f[0]=1  # seed weighting over y
        f_f = f_f / np.sqrt(f_f.T@f_f)

    assert S_y is None

    for iter in range(max_iter):
        oJ = J
        of_f = f_f

        # 1) Apply the fir to X
        fCxxf_dd = np.einsum("f,fdue,u->de",f_f,Cxx_fdfd,f_f) 
        Cyxf_tyed = np.einsum("tyefd,f->tyed",Cyx_tyefd,f_f) 

        # 2) Apply the outputs weighting to Y
        sCyys_etet = Cyy_yetyet[0,:,:,0,:,:]
        sCyxf_ted = Cyxf_tyed[:,0,:,:]
            
        #2) CCA solution
        #2.1) whiten X
        isqrtCxx_dd, _ = robust_whitener(fCxxf_dd,reg[0],rcond[0],symetric=True)
        CyxisqrtCxx_ted = np.einsum('ted,df->tef',sCyxf_ted,isqrtCxx_dd)

        #2.2) whiten Y
        #2.2.1) make Cyy 2d
        if S_y is not None or iter==0:
            Cyy_et_et = np.reshape(sCyys_etet,(nE*tau_y,nE*tau_y)) # (nE,tau,nE,tau) -> ((nE*tau),(nE*tau))
            #2.2.2) Compute updated right-whitener
            isqrtCyy_et_et, _ = robust_whitener(Cyy_et_et,reg[1],rcond[1],symetric=True)
            isqrtCyy_etet = isqrtCyy_et_et.reshape((nE,tau_y,nE,tau_y))

        #2.2.3) right-whiten Cyx
        isqrtCyyCyxisqrtCxx_ted = np.einsum('etfu,ted->ufd',isqrtCyy_etet,CyxisqrtCxx_ted)

        # 2.3) SVD
        isqrtCxxCyxisqrtCyy_te_d = np.reshape(isqrtCyyCyxisqrtCxx_ted,(tau_y*nE, nD)) # (nE,tau,d) -> ((nE*tau),d)
        R_te_k, l_k, W_kd = np.linalg.svd(isqrtCxxCyxisqrtCyy_te_d, full_matrices=False)  
        R_kte = R_te_k.T.reshape((R_te_k.shape[-1],tau_y,nE)) #((nE*tau),k) -> (k,(tau,nE)) -> (k,tau,nE)
        # 2.4) get the largest components
        l_idx = np.argsort(l_k)[::-1]  # N.B. DECENDING order
        r = min(len(l_idx), rank)  # guard rank>effective dim
        # 2.5) extract the desired rank sub-space solutions
        R_kte = R_kte[l_idx[:r],:,:] 
        l_k = l_k[l_idx[:r]]  # l_k = coor for this component
        W_kd = W_kd[l_idx[:r],:]

        # 2.6) pre-apply the whitener + scaling so can apply directly to input
        R_kte = np.einsum('kte,etfu->kuf', R_kte, isqrtCyy_etet) #* np.sqrt(l_k[:,np.newaxis, np.newaxis])
        W_kd = np.einsum('kd,df->kf', W_kd, isqrtCxx_dd) #* np.sqrt(l_k[:,np.newaxis])
        

        J_wr = sse(Cxx_fdfd, Cyx_tyefd, Cyy_yetyet, W_kd, R_kte, f_f, S_y)
        print("{:3d}) J_wr = {}".format(iter,J_wr))

        # 4) Solve for optimal S, under the positivity constraint using
        if not S_y is None:
            #fwCxxwf  = np.einsum('kd,de,ke', W_kd, Cxx_dd, W_kd) # N.B. should be I_k
            rCyyr_yy = np.einsum("kte,yetzfu,kuf->yz",R_kte,Cyy_yetyet,R_kte) # N.B. should be I_k
            rCyxwf_y = np.einsum('kte,tyefd,kd,f->y',R_kte,Cyx_tyefd,W_kd,f_f) 
            # multi-updates
            # TODO [X] : non-negative least squares
            # TODO [] : norm-constrained optimization
            S_y = nonneglsopt(None, rCyxwf_y, rCyyr_yy, S_y, syopt)
            S_y = S_y / np.sum(S_y) # maintain the norm
            if np.any(np.isnan(S_y)) or np.any(np.isinf(S_y)):
                print("Warning: nan or inf!")

        # 5) Solve for optimal f_f, under unit norm constraint
        if not f_f is None:
            wCxxw_ff  = np.einsum('kd,fdge,ke->fg', W_kd, Cxx_fdfd, W_kd)
            sCyx_tefd = Cyx_tyefd[:,0,...]
            srCyxw_f = np.einsum('kte,tefd,kd->f', R_kte, sCyx_tefd, W_kd) 
            f_f = lsopt(wCxxw_ff,srCyxw_f,None, f_f, ftopt)

            # updated loss
            J_f = sse(Cxx_fdfd, Cyx_tyefd, Cyy_yetyet, W_kd, R_kte, f_f, S_y)
            print("{:3d}) J_f = {}   df={}".format(iter, J_f, np.sum(np.abs(of_f-f_f))))

            f_f = f_f / np.sqrt(f_f@f_f)


        J = sse(Cxx_fdfd, Cyx_tyefd, Cyy_yetyet, W_kd, R_kte, f_f, S_y)
        #print("{:3d}) S_y = {}".format(iter,np.array_str(S_y,precision=3)))

        # 5) Goodness of fit tracking
        Js[iter,:]=[J_wr,J]
        if showplots and iter%100 == 0:
            plt.cla()
            plt.plot(Js[:iter,:])
            plt.legend(('pre','post'))
            plt.pause(.001)
        deltaJ = np.abs(J-oJ)
        if iter<10 or iter%100==0:
            print("{:3d}) J={:4.3f} dJ={:4.3f}".format(iter,J,deltaJ))

        # convergence testing
        if deltaJ < tol:
            print("{:3d}) J={:4.3f} dJ={:4.3f}".format(iter,J,deltaJ))
            break
            
    if showplots:
        print("{:3d}) J={:4.3f} dJ={:5.4f}".format(iter,J,deltaJ))
        plt.cla()
        plt.plot(Js[:iter,:])
        plt.legend(('pre','post'))
        plt.pause(.001)

    # include relative component weighting directly in the  Left/Right singular values
    nl_k = (l_k / np.max(l_k)) if np.max(l_k)>0 else np.ones(l_k.shape,dtype=l_k.dtype)  
    W_kd = W_kd * np.sqrt(nl_k[:,np.newaxis])
    R_kte = R_kte * np.sqrt(nl_k[:,np.newaxis, np.newaxis]) 

    return J, W_kd, R_kte, f_f, S_y

def plot_fir_summary_statistics(Cxx_fdd, Cyx_tyed, Cyy_tyeye, f_f, S_y=None):
    tau = (Cxx_fdd.shape[0], Cyy_tyeye.shape[0])
    Cyy_yetyet = Cyy_tyeye_diag2full(Cyy_tyeye)
    Cxx_fdfd = Cxx_diag2full(Cxx_fdd)
    Cyx_tyefd = Cyx_diag2full(Cyx_tyed,tau)
    # fir    
    fCxxf_dd = np.einsum("f,fdue,u->de",f_f,Cxx_fdfd,f_f) 
    Cyxf_tyed = np.einsum("tyefd,f->tyed",Cyx_tyefd,f_f) 
    Cyxf_yetd = np.moveaxis(Cyxf_tyed,0,-2)
    # output-weight
    if S_y is None:
        sCyys_etet = Cyy_yetyet[0,:,:,0,:,:]
        sCyxf_ted = Cyxf_yetd[0,:,:,:]
    else:
        raise NotImplementedError
    # plot
    plot_summary_statistics(fCxxf_dd, sCyxf_ted, sCyys_etet)


def test_firCCA(X_TSd,Y_TSye,tau=(2,15),offset=0,rank=2,center=True,unitnorm=True,zeropadded=True,badEpThresh=4):
    import matplotlib.pyplot as plt

    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_factoredmodel
    from mindaffectBCI.decoder.multipleCCA import multipleCCA

    Cxx_dd0, Cyx_yetd0, Cyy_yetet0 = updateSummaryStatistics(X_TSd,Y_TSye[...,0:1,:],tau=tau[1],
                    offset=offset,center=center,unitnorm=unitnorm,zeropadded=zeropadded,badEpThresh=badEpThresh)

    plt.figure()
    plot_erp(Cyx_yetd0)
    plt.suptitle('USS')

    plt.figure()
    plot_summary_statistics(Cxx_dd0, Cyx_yetd0, Cyy_yetet0)
    plt.suptitle('USS')

    J, W_kd0, R_ket0    = multipleCCA  (Cxx_dd0, Cyx_yetd0, Cyy_yetet0, rank=rank, rcond=1e-6, reg=1e-4, symetric=True)

    plt.figure()
    plot_factoredmodel(W_kd0,R_ket0)
    plt.suptitle(' mCCA soln') 

    Cxx_fdd, Cyx_tyed, Cyy_tyeye = firSummaryStatistics(X_TSd, Y_TSye, tau, center=False, unitnorm=False)

    plt.figure()
    Cyx_tyefd = Cyx_diag2full(Cyx_tyed,tau)
    Cyx_yetfd = np.moveaxis(Cyx_tyefd,(0,1,2),(2,0,1))
    Cyy_yetyet = Cyy_tyeye_diag2full(Cyy_tyeye)
    plot_summary_statistics(Cxx_fdd[0,...], Cyx_yetfd[...,0,:], Cyy_yetyet[...,0,:,:])
    plt.suptitle(' firUSS tau_x=0')

    plt.figure()
    f_f = np.zeros(Cxx_fdd.shape[0]); f_f[0]=1; S_y=None
    plot_fir_summary_statistics(Cxx_fdd, Cyx_tyed, Cyy_tyeye, f_f, S_y)
    plt.suptitle(' firCCA: f_f = delta_0')

    plt.figure()
    Cyx_yetd = np.moveaxis(Cyx_tyed,(0,1,2),(2,0,1))
    plot_erp(Cyx_yetd)
    plt.suptitle('fir SS')

    J, W_kd, R_kte, f_f, S_y = firCCA_cov(Cxx_fdd,  Cyx_tyed,  Cyy_tyeye,  rank=rank, rcond=1e-6, reg=1e-4, symetric=True)
    R_ket = R_kte.swapaxes(-2,-1)
    plt.figure()
    plot_3factoredmodel(W_kd,R_ket,f_f)
    plt.suptitle(' firCCA soln')

    # apply the FIR and plot the summary statistics
    plt.figure()
    plot_fir_summary_statistics(Cxx_fdd, Cyx_tyed, Cyy_tyeye, f_f, S_y)
    plt.suptitle(' firCCA USS')


def debug_levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, rank:int=1, syopt=None, label=None, outputs=None, fs:float=None, ch_names:list=None, **kwargs):
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_summary_statistics, plot_factoredmodel
    import matplotlib.pyplot as plt
    if syopt is None: syopt='negridge'
    if outputs is None: outputs = [ i for i in range(Cyy_tyeye.shape[1])]

    J, W_kd, R_ket, S_y = levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, syopt=syopt, rank=rank, **kwargs)
    print("{}) J={}".format('opt',J))

    print(f"W_kd {W_kd.shape}")
    print(f"R_ket {R_ket.shape}")
    print(f"S_y {S_y.shape} = {S_y}")

    # plot soln
    plot_3factoredmodel(W_kd,R_ket,S_y,fs=fs,ch_names=ch_names,outputs=outputs)
    plt.suptitle('{} {} soln'.format(label,syopt)) 

    # plot SS
    # pre-expand Cyy
    Cyy_yetyet = Mtyeye2Myetyet(Cyy_tyeye)
    sCyys_etet = np.einsum("y,yetzfu,z->etfu",S_y,Cyy_yetyet,S_y) 
    sCyx_etd = np.einsum('yetd,y->etd',Cyx_yetd,S_y)
    plt.figure()
    plot_summary_statistics(Cxx_dd,sCyx_etd,sCyys_etet[np.newaxis,...],fs=fs,ch_names=ch_names)
    plt.suptitle(' {} {}'.format(label,syopt) )
    plt.show(block=False)

    return W_kd, R_ket, S_y


def testcase_levelsCCA(X_TSd,Y_TSye,tau=15,offset=0,rank:int=1,
                       center=True,unitnorm=True,zeropadded=True,badEpThresh=4,
                       single_output_test:bool=False, 
                       fs:float=None, ch_names:list=None, outputs:list=None, label:str=None):
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
    if single_output_test:
        nY = Cyx_yetd.shape[0]
        for yi in range(nY):
            S_y = np.zeros((nY))
            S_y[yi]=1
            J, W_kd, R_ket, S_y = levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, S_y=S_y, rank=rank, rcond=1e-6, reg=1e-4, symetric=True, max_iter=1)
            print("{}) J={}".format(yi,J))


    W_kd, R_ket, S_y = debug_levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, rank=rank, syopt='negridge', fs=fs, ch_names=ch_names, outputs=outputs, label=label)

    if label=='visual_acuity':
        # 2d S_y
        w = int(np.sqrt(S_y.shape[-1])) 
        h = (S_y.shape[-1]) // w
        plt.figure()
        plt.imshow(S_y.reshape((w,h)),aspect='auto')
        plt.colorbar()
        plt.suptitle("S_y")


    #debug_levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, rank=rank, syopt='lspos')

    #debug_levelsCCA_cov(Cxx_dd, Cyx_yetd, Cyy_tyeye, rank=rank, syopt='mu')

    plt.show()
    return W_kd, R_ket, S_y

def simdata(nTrl=10, nSamp=500, nY=10, tau=10, noise2signal=2, irf=None):
    from mindaffectBCI.decoder.utils import testSignal
    if irf=='sin':
        irf = np.sin(np.linspace(0,2*np.pi*3,tau))
    X_TSd, Y_TSye, st, A, B = testSignal(nTrl=nTrl, nSamp=nSamp, nY=nY, tau=tau, irf=irf, noise2signal=noise2signal)
    coords = None
    outputs=None
    label = None
    return X_TSd, Y_TSye, coords, outputs, label


def testcase(tau_ms:float=650, offset_ms:float=0, rank:int=2):
    if False:
        X_TSd, Y_TSye, coords, outputs, label = simdata()

    else:
        X_TSd, Y_TSye, coords, outputs, label = loaddata()

    fs = coords[1]['fs'] if coords is not None else 100
    ch_names = coords[2]['coords'] if coords is not None else None
    tau = int(tau_ms*fs/1000)
    offset = int(offset_ms*fs/1000)

    #testcase_levelsCCA_vs_multiCCA(X_TSd,Y_TSye,tau=15,offset=0)
        
    testcase_levelsCCA(X_TSd,Y_TSye,tau=tau,offset=offset,rank=rank,fs=fs,ch_names=ch_names,outputs=outputs,label=label)


if __name__=="__main__":
    testcase()
