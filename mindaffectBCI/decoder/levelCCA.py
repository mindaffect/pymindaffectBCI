from mindaffectBCI.decoder.multiCCA import robust_whitener
import numpy as np

def levelsCCA(X,Y,tau,offset=0,reg=1e-9,rank=1,CCA=True,rcond=(1e-8,1e-8), symetric=False, center=True, unitnorm=True):
    # get the summary statistics
    Cxx = updateCxx(None,X,None,tau=tau, offset=offset, center=center, unitnorm=unitnorm)
    Cyx = updateCxy(None, X, Y, None, tau=tau, offset=offset, center=center, unitnorm=unitnorm)
    Cyy_diag = compCyy_diag(Y, tau, zeropadded=zeropadded, unitnorm=unitnorm, perY=perY)

    return levelsCCA_cov(Cxx, Cyx, Cyy_diag, reg=reg, rank=rank, rcond=rcond, symetric=symetric)
    

def levelsCCA_cov(Cxx_dd=None, Cyx_detd=None, Cyy_tyeye=None,
                reg=1e-9, rank=1, CCA=True, rcond=(1e-8,1e-8), symetric=False):
    """
    Compute multiple CCA decompositions using the given summary statistics
      [J,W,R]=multiCCA(Cxx,Cyx,Cyy,regx,regy,rank,CCA)
    Inputs:
      Cxx  = (d,d) current data covariance
      Cyx  = (nY,nE,tau,d) current per output ERPs
      Cyy  = (tau,nY,nE,nY,nE) compressed cov for each output at different time-lags
      reg  = (1,) :float or (2,):float linear weighting reg strength or separate values for Cxx and Cyy 
            OR (d,) regularisation ridge coefficients for Cxx  (0)
      rank= [1x1] number of top cca components to return (1)
      CCA = [bool] or (2,):bool  flag if we normalize the spatial dimension. (true)
      rcond  = [float] tolerance for singular eigenvalues.        (1e-4)
               or [2x1] separate rcond for Cxx (rcond(1)) and Cyy (rcond(2))
               or [2x1] (-1<-0) negative values = keep this fraction of eigen-values
               or (2,1) <-1 keep this many eigenvalues
      symetric = [bool] us symetric whitener?
    Outputs:
      J     = (nM,) optimisation objective scores
      W     = (nM,rank,d) spatial filters for each output
      R     = (nM,rank,nE,tau) responses for each stimulus event for each output
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
    isqrtCxx_dd, _ = robust_whitener(Cxx_dd,reg[0],rcond[0],symetric)
    #2.c) Whiten Cxy
    CyxisqrtCxx_yetd = np.einsum('yetd,df->yetf',isqrtCxx_dd,Cyx_yetd)


    S_y = np.ones((y),dtype=Cxy.dtype) # seed weighting over y
    for iter in range(30):

        # 1) Apply the output weighting and make 2d Cyy
        sCyys_etet = np.einsum("y,yetzfu,z->etfu",S_y,Cyy_yetyet,S_y) 
        sCyxisqrtCxx_etd = np.einsum('yetd,y->etd',CyxisqrtCxx_yetd,S_d)
            
        #2) CCA solution
        #2.1) Make 2d
        sCyys_et_et = np.reshape(sCyys_etet,(nE*tau,nE*tau)) # (nE,tau,nE,tau) -> ((nE*tau),(nE*tau))
        #2.2) Compute whiteners
        isqrtCyy_et_et, _ = robust_whitener(sCyys_et_et,reg[0],rcond[0],symetric)
        isqrtCyy_etet = isqrtCyy_et_et.reshape((e,t,e,t))
        #2.3) right-whiten Cxy
        CxyisqrtCxx_et_d = np.reshape(sCxy_etd,(nE*tau,d)) # (nE,tau,d) -> ((nE*tau),d)
        isqrtCxxsCyxisqrtCyy_et_d = np.einsum('ad,ab->bd',CxyisqrtCxx_et_d,isqrtCyy_et_et)
        # 2.3) SVD
        R_et_k, l_k, W_kd = np.linalg.svd(isqrtCxxsCyxisqrtCyy_et_d, full_matrices=False)  
        W_dk = W_kd.T  # (d,rank)
        R_etk = R.reshape((e,t,R_et_k.shape[-1]))
        # 2.4) get the largest components
        l_idx = np.argsort(l_k)[::-1]  # N.B. DECENDING order
        r = min(len(l_idx), rank)  # guard rank>effective dim
        # 2.5) extract the desired rank sub-space solutions
        R_etk = R_etk[:,l_idx[:r]] 
        l_k = l_k[l_idx[:r]]
        W_dk = W_dk[:,l_idx[:r]]

        # 2.6) pre-apply the whitener so can apply directly to input
        R_etk = np.einsum('etk,etfu->fuk',R_etk,isqrtCyy_etet)
        W_dk = np.einsum('dk,df->fk',W_dk,isqrtCxx_dd)
        
        #3) Update componenst for the output weighting
        rCyyr_yy = np.einsum("etk,yetzfu,fuk->yz",R_etk,Cyy_yetyet,R_etk) 
        rCyxw_y = np.einsum('dk,yetd,etk->y',W_dk,Cyx_yetd,R_etk)

        # 4) Solve for optimal S
        S_y = np.pinv(rCyyr_yy) @ rCyxw_y


    # include relative component weighting directly in the  Left/Right singular values
    nl_k = (l_k / np.max(l_k)) if np.max(l_k)>0 else np.ones(l_k.shape,dtype=l_k.dtype)  # normalize so predictions have unit average norm
    W_d_k = W_d_k * np.sqrt(nl_k[np.newaxis, :])
    R_et_k = R_et_k * np.sqrt(nl_k[np.newaxis, :]) #* np.sqrt(nlm[np.newaxis, :])



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
