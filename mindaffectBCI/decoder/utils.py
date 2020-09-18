import numpy as np

# time-series tests
def window_axis(a, winsz, axis=0, step=1, prependwindowdim=False):
    ''' efficient view-based slicing of equal-sized equally-spaced windows along a selected axis of a numpy nd-array '''
    if axis < 0: # no negative axis indices
        axis = len(a.shape)+axis

    # compute the shape/strides for the  windowed view of a
    if prependwindowdim: # window dim before axis
        shape  = a.shape[:axis] + (winsz, int((a.shape[axis]-winsz)/step)+1) +  a.shape[(axis+1):]
        strides =  a.strides[:axis] + (a.strides[axis], a.strides[axis]*step) + a.strides[(axis+1):]
    else: # window dim after axis
        shape  = a.shape[:axis] + (int((a.shape[axis]-winsz)/step)+1, winsz) +  a.shape[(axis+1):]
        strides =  a.strides[:axis] + (a.strides[axis]*step, a.strides[axis]) + a.strides[(axis+1):]
    #print("a={}".format(a.shape))
    #print("shape={} stride={}".format(shape,strides))
        
    # return the computed view
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def equals_subarray(a, pat, axis=-1, match=-1):
    ''' efficiently find matches of a 1-d sub-array along axis within an nd-array ''' 
    if axis < 0: # no negative dims
        axis = a.ndim+axis
    # reshape to match dims of a
    if not isinstance(pat, np.ndarray): pat = np.array(pat) # ensure is numpy
    pshape = np.ones(a.ndim+1, dtype=int); pshape[axis+1] = pat.size
    pat =  np.array(pat.ravel()).reshape(pshape) # [ ... x l x...]
    # window a into pat-len pieces
    aw = window_axis(a, pat.size, axis=axis) # [ ... x t-l x l x ...]
    # do the match
    F  = np.all(np.equal(aw, pat), axis=axis+1) # [... x t-l x ...]
    # pad to make the same shape as input
    padshape = list(a.shape); padshape[axis] = a.shape[axis]-F.shape[axis]
    if match == -1: # match at end of pattern -> pad before
        F  = np.append(np.zeros(padshape), F, axis)
    else: # match at start of pattern -> pad after
        F  = np.append(F, np.zeros(padshape), axis)
    return F


class RingBuffer:
    ''' time efficient linear ring-buffer for storing packed data, e.g. continguous np-arrays '''
    def __init__(self, maxsize, shape, dtype=np.float32):
        self.elementshape = shape
        self.bufshape = (int(maxsize), )+shape
        self.buffer = np.zeros((2*int(maxsize), np.prod(shape)), dtype=dtype) # store as 2d
        # position for the -1 element. N.B. start maxsize so pos-maxsize is always valid
        self.pos = int(maxsize)
        self.n = 0 # count of total number elements added to the buffer
        self.copypos = 0 # position of the last element copied to the 1st half
        self.copysize = 0 # number entries to copy as a block

    def append(self, x):
        '''add single element to the ring buffer'''
        return self.extend(x[np.newaxis, ...])
    
    def extend(self, x):
        '''add a group of elements to the ring buffer'''
        # TODO[] : incremental copy to the 1st half, to spread the copy cost?
        nx = x.shape[0]
        if self.pos+nx >= self.buffer.shape[0]:
            flippos = self.buffer.shape[0]//2
            # flippos-nx to 1st half
            self.buffer[:(flippos-nx), :] = self.buffer[(self.pos-(flippos-nx)):self.pos, :]
            # move cursor to end 1st half
            self.pos = flippos-nx

        # insert in the buffer
        self.buffer[self.pos:self.pos+nx, :] =  x.reshape((nx, self.buffer.shape[1]))
        # move the cursor
        self.pos = self.pos+nx
        # update the count
        self.n = self.n + nx
        return self
    
    @property
    def shape(self):
        return self.bufshape
    
    def unwrap(self):
        '''get a view on the valid portion of the ring buffer'''
        return self.buffer[self.pos-self.bufshape[0]:self.pos, :].reshape(self.shape)

    def __getitem__(self, item):
        return self.unwrap()[item]

    def __iter__(self):
        return iter(self.unwrap())

def extract_ringbuffer_segment(rb, bgn_ts, end_ts=None):
    ''' extract the data between start/end time stamps'''
    # get the data / msgs from the ringbuffers
    X = rb.unwrap() # (nsamp,nch+1)
    X_ts = X[:, -1] # last channel is timestamps
    # TODO: binary-search to make these searches more efficient!
    # search backwards for trial-start time-stamp
    # TODO[] : use a bracketing test.. (better with wrap-arround)
    bgn_samp = np.flatnonzero(np.logical_and(bgn_ts <= X_ts, X_ts != 0))
    # get the index of this timestamp, guarding for after last sample
    bgn_samp = bgn_samp[0] if len(bgn_samp) > 0 else len(X_ts)+1
    # and just to be sure the trial-end timestamp
    if  end_ts is not None:
        end_samp = np.flatnonzero(np.logical_and(X_ts < end_ts, X_ts != 0))
        # get index of this timestamp, guarding for after last data sample
        end_samp = end_samp[-1] if len(end_samp) > 0 else len(X_ts)
    else: # until now
        end_samp = len(X_ts)
    # extract the trial data, and make copy (just to be sure)
    X = X[bgn_samp:end_samp+1, :].copy()
    return X

# toy data generation

#@function
def randomSummaryStats(d=10, nE=2, tau=10, nY=1):
    import numpy as np
    # pure random test-case
    Cxx = np.random.standard_normal((d, d))
    Cxy = np.random.standard_normal((nY, nE, tau, d))
    Cyy = np.random.standard_normal((nY, nE, tau, nE, tau))
    return (Cxx, Cxy, Cyy)

def testNoSignal(d=10, nE=2, nY=1, isi=5, tau=None, nSamp=10000, nTrl=1):
    # Simple test-problem -- no real signal
    if tau is None:
        tau = 10*isi
    X = np.random.standard_normal((nTrl, nSamp, d))
    stimTimes_samp = np.arange(0, X.shape[-2] - tau, isi)
    Me = np.random.standard_normal((nTrl, len(stimTimes_samp), nY, nE))>1
    Y = np.zeros((nTrl, X.shape[-2], nY, nE))
    Y[:, stimTimes_samp, :, :] = Me
    return (X, Y, stimTimes_samp)

def testSignal(nTrl=1, d=5, nE=2, nY=30, isi=5, tau=None, offset=0, nSamp=10000, stimthresh=.6, noise2signal=1, irf=None):
    #simple test problem, with overlapping response
    import numpy as np
    if tau is None:
        tau = 10 if irf is None else len(irf)
    nEp = int((nSamp-tau)/isi)
    cb = np.random.standard_normal((nEp, nY, nE)) > stimthresh # codebook = per-epoch stimulus activity
    E  = cb # (nEp, nY, nE) # per-epoch stimulus activity
    # up-sample to sample rate
    stimTimes_samp = np.arange(0, nSamp-tau, isi) # (nEp)
    Y = np.zeros((nSamp, nY, E.shape[-1]))
    Y[stimTimes_samp, :, :] = E[:len(stimTimes_samp), :, :] #per-sample stimulus activity (nSamp, nY, nE) [nE x nY x nSamp]
    Y = np.tile(Y,(nTrl,1,1,1)) # replicate for the trials
    # generate the brain source
    A  = np.random.standard_normal((nE, d)) # spatial-pattern for the source signal
    if irf is None:
        B  = np.zeros((tau),dtype=np.float32)
        B[-3] = 1;         # true response filter (shift by 10 samples)
    else:
        B = np.array(irf,dtype=np.float32)
    Ytrue = Y[..., 0, :] # (nTrl, nSamp, nE)

    if True:
        # convolve with the impulse response - manually using window_axis
        # zero pad before for the sliding window
        Ys = np.zeros(Ytrue.shape[:-2]+(Ytrue.shape[-2]+tau-1,)+Ytrue.shape[-1:])
        Ys[..., tau-1+offset:Ytrue.shape[-2]+tau-1+offset, :] = Ytrue # zero-pad at front + include the offset.
        Yse = window_axis(Ys, winsz=len(B), axis=-2) # (nTr,nSamp,tau,nE)
        YtruecB = np.einsum("Tste,t->Tse", Yse, B[::-1]) # N.B. time-reverse irf (nTr,nSamp,nE)

    else:
        # use the np convolve function, N.B. implicitly time reverses B (like we want)
        YtruecB = np.array([np.convolve(Ytrue[:, ei], B, 'full') for ei in range(Ytrue.shape[-1])]).T #(nSamp+pad, nE) [nE x nSamp]
        YtruecB = YtruecB[:Ytrue.shape[0], :] # trim the padding

    #import matplotlib.pyplot as plt; plt.clf(); plt.plot(Ytrue[:100,0],'b*',label='Y'); plt.plot(YtruecB[:100,0],'g*',label='Y*B'); plt.plot(B,'k',label='B'); plt.legend()

    #print("Ytrue={}".format(Ytrue.shape))
    #print("YtruecB={}".format(YtruecB.shape))
    S  = YtruecB # (nTr, nSamp, nE) true response, i.e. filtered Y 
    N  = np.random.standard_normal(S.shape[:-1]+(d,)) # EEG noise (nTr, nSamp, d)
    X  = np.einsum("tse,ed->tsd", S, A) + noise2signal*N       # simulated data.. true source mapped through spatial pattern (nSamp, d) #[d x nSamp]
    return (X, Y, stimTimes_samp, A, B)

def testtestSignal():
    import matplotlib.pyplot as plt
    plt.clf()
    # shift by 5
    offset=0; irf=(0,0,0,0,0,1,0,0,0,0)
    X,Y,st,W,R = testSignal(nTrl=1,nSamp=500,d=1,nE=1,nY=1,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)
    plt.subplot(311);plt.plot(X[0,:,0],label='X');plt.plot(Y[0,:,0,0],label='Y');plt.title("offset={}, irf={}".format(offset,irf));plt.legend()
    # back-shift-by-5 -> 0 shift
    offset=-5
    X,Y,st,W,R = testSignal(nTrl=1,nSamp=500,d=1,nE=1,nY=1,isi=10,tau=10,offset=offset,irf=(0,0,0,0,0,1,0,0,0,0),noise2signal=0)
    plt.subplot(312);plt.plot(X[0,:,0],label='X');plt.plot(Y[0,:,0,0],label='Y');plt.title("offset={}, irf={}".format(offset,irf));plt.legend()
    # back-shift-by-10 -> -5 shift
    offset=-9
    X,Y,st,W,R = testSignal(nTrl=1,nSamp=500,d=1,nE=1,nY=1,isi=10,tau=10,offset=offset,irf=(0,0,0,0,0,1,0,0,0,0),noise2signal=0)
    plt.subplot(313);plt.plot(X[0,:,0],label='X');plt.plot(Y[0,:,0,0],label='Y');plt.title("offset={}, irf={}".format(offset,irf));plt.legend()
    

def sliceData(X, stimTimes_samp, tau=10):
    # make a sliced version
    dst = np.diff(stimTimes_samp)
    if np.all(dst == dst[0]) and stimTimes_samp[0] == 0: # fast path equaly spaced stimTimes
        Xe = window_axis(X, winsz=tau, axis=-2, step=int(dst[0]), prependwindowdim=False) # (nTrl, nEp, tau, d) #d x tau x ep x trl
    else: 
        Xe = np.zeros(X.shape[:-2] + (len(stimTimes_samp), tau, X.shape[-1])) # (nTrl, nEp, tau, d) [ d x tau x nEp x nTrl ]
        for ei, si in enumerate(stimTimes_samp):
            idx = range(si, si+tau)
            Xe[:, ei, :, :] = X[:, idx, :] if X.ndim > 2 else X[idx, :]
    return Xe

def sliceY(Y, stimTimes_samp, featdim=True):
    '''
    Y = (nTrl, nSamp, nY, nE) if featdim=True
    OR
    Y=(nTrl, nSamp, nY) if featdim=False #(nE x nY x nSamp x nTrl)
    '''
    # make a sliced version
    si = np.array(stimTimes_samp, dtype=int)
    if featdim:
        return Y[:, si, :, :] if Y.ndim > 3 else Y[si, :, :]
    else:
        return Y[:, si, :] if Y.ndim > 2 else Y[si, :]

    
def block_randomize(true_target, npermute, axis=-3, block_size=None):
    ''' make a block random permutaton of the input array
    Inputs:
       npermute: int - number permutations to make
       true_target: (..., nEp, nY, e): true target value for nTrl trials of length nEp flashes
       axis : int the axis along which to permute true_target'''
    if true_target.ndim < 3:
        raise ValueError("true target info must be at least 3d")
    if not (axis == -3 or axis == true_target.ndim-2):
        raise NotImplementedError("Only implementated for axis=-2 currently")
    
    # estimate the number of blocks to use
    if block_size is None:
        block_size = max(1, true_target.shape[axis]/2/npermute)

    nblk = int(np.ceil(true_target.shape[axis]/block_size))
    blk_lims = np.linspace(0, true_target.shape[axis], nblk, dtype=int)
    # convert to start/end index for each block
    blk_lims = [(blk_lims[i], blk_lims[i+1]) for i in range(len(blk_lims)-1)]
    cb = np.zeros(true_target.shape[:axis+1] + (npermute, true_target.shape[-1]))
    for ti in range(cb.shape[axis+1]):
        for di, dest_blk in enumerate(blk_lims):
            
            yi = np.random.randint(true_target.shape[axis+1])
            si = np.random.randint(len(blk_lims))
            # ensure can't be the same block
            if si == di:
                si = si+1 if si < len(blk_lims)-1 else si-1
            src_blk = blk_lims[si]
            # guard for different lengths for source/dest blocks
            dest_len = dest_blk[1] - dest_blk[0]
            if dest_len > src_blk[1]-src_blk[0]:
                if src_blk[0]+dest_len < true_target.shape[axis]:
                    # enlarge the src
                    src_blk = (src_blk[0], src_blk[0]+dest_len)
                elif src_blk[1]-dest_len > 0:
                    src_blk = (src_blk[1]-dest_len, src_blk[1])
                else:
                    raise ValueError("can't fit source and dest")
            elif dest_len < src_blk[1]-src_blk[0]:
                src_blk = (src_blk[0], src_blk[0]+dest_len)
            
            cb[..., dest_blk[0]:dest_blk[1], ti, :] = true_target[..., src_blk[0]:src_blk[1], yi, :]

    return cb


def upsample_codebook(trlen, cb, ep_idx, stim_dur_samp, offset_samp=(0, 0)):
    ''' upsample a codebook definition to sample rate
    Inputs:
       trlen : (int) length after up-sampling
       cb : (nTr, nEp, ...) the codebook
       ep_idx : (nTr, nEp) the indices of the codebook  entries
       stim_dur_samp: (int) the amount of time the cb entry is held for
       offset_samp : (2,):int the offset for the stimulus in the upsampled trlen data
    Outputs:
       Y : ( nTrl, trlen, ...) the up-sampled codebook '''

    if ep_idx is not None:
        if not np.all(cb.shape[:ep_idx.ndim] == ep_idx.shape):
            raise ValueError("codebook and epoch indices must has same shape")
        trl_idx = ep_idx[:, 0] # start each trial
    else: # make dummy ep_idx with 0 for every trial!
        ep_idx = np.zeros((cb.shape[0],1),dtype=int)
        trl_idx = ep_idx
        
    Y  = np.zeros((cb.shape[0], trlen)+ cb.shape[2:], dtype='float32') # (nTr, nSamp, ...)
    for ti, trl_start_idx in enumerate(trl_idx):
        for ei, epidx in enumerate(ep_idx[ti, :]):
            if ei > 0 and epidx == 0: # zero indicates end of variable length trials
                break
            # start index for this epoch in this *trial*, including the 0-offset 
            ep_start_idx = -int(offset_samp[0])+int(epidx-trl_start_idx)
            Y[ti, ep_start_idx:(ep_start_idx+int(stim_dur_samp)), ...]  = cb[ti, ei, ...]
    return Y


def lab2ind(lab,lab2class=None):
    ''' convert a list of labels (as integers) to a class indicator matrix'''
    if lab2class is None:
        lab2class = [ (l,) for l in set(lab) ] # N.B. list of lists
    if not isinstance(lab,np.ndarray):
        lab=np.array(lab)
    Y = np.zeros(lab.shape+(len(lab2class),),dtype=bool)
    for li,ls in enumerate(lab2class):
        for l in ls:
            Y[lab == l, li]=True
    return (Y,lab2class)



def zero_outliers(X, Y, badEpThresh=4, badEpChThresh=None, verbosity=0):
    '''identify and zero-out bad/outlying data

    Inputs:
      X = (nTrl, nSamp, d)
      Y = (nTrl, nSamp, nY, nE) OR (nTrl, nSamp, nE)
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
    '''
    # remove whole bad epochs first
    if badEpThresh > 0:
        bad_ep, _ = idOutliers(X, badEpThresh, axis=(-2, -1)) # ave over time,ch
        if np.any(bad_ep):
            if verbosity > 0:
                print("{} badEp".format(np.sum(bad_ep.ravel())))
            # copy X,Y so don't modify in place!
            X = X.copy()
            Y = Y.copy()
            X[bad_ep[..., 0, 0], ...] = 0
            #print("Y={}, Ybad={}".format(Y.shape, Y[bad_ep[..., 0, 0], ...].shape))
            # zero out Y also, so don't try to 'fit' the bad zeroed data
            Y[bad_ep[..., 0, 0], ...] = 0
            
            
    # Remove bad individual channels next
    if badEpChThresh is None: badEpChThresh = badEpThresh*2 
    if badEpChThresh > 0:
        bad_epch, _ = idOutliers(X, badEpChThresh, axis=-2) # ave over time
        if np.any(bad_epch):
            if verbosity > 0:
                print("{} badEpCh".format(np.sum(bad_epch.ravel())))
            # make index expression to zero out the bad entries
            badidx = list(np.nonzero(bad_epch)) # convert to linear indices
            badidx[-2] = slice(X.shape[-2]) # broadcast over the accumulated dimensions
            if not np.any(bad_ep): # copy so don't update in place
                X = X.copy()
            X[tuple(badidx)] = 0

    return (X, Y)


def idOutliers(X, thresh=4, axis=-2, verbosity=0):
    ''' identify outliers with excessively high power in the input data
    Inputs:
      X:float the data to identify outliers in
      axis:int (-2)  axis of X to sum to get power
      thresh(float): threshold standard deviation for outlier detection
      verbosity(int): verbosity level
    Returns:
      badEp:bool (X.shape axis==1) indicator for outlying elements
      epPower:float (X.shape axis==1) power used to identify bad
    '''
    #print("X={} ax={}".format(X.shape,axis))
    power = np.sqrt(np.sum(X**2, axis=axis, keepdims=True))  
    #print("power={}".format(power.shape))
    good = np.ones(power.shape, dtype=bool)
    for _ in range(4):
        mu = np.mean(power[good])
        sigma = np.sqrt(np.mean((power[good] - mu) ** 2))
        badThresh = mu + thresh*sigma
        good[power > badThresh] = False
    good = good.reshape(power.shape) # (nTrl, nEp)
    #print("good={}".format(good.shape))
    bad = ~good
    if verbosity > 1:
        print("%d bad" % (np.sum(bad.ravel())))
    return (bad, power)

class linear_trend_tracker():
    """ linear trend tracker with adaptive forgetting factor
    """   

    def __init__(self,halflife=1000,int_err_halflife=50, K_int_err=1):
        self.alpha=np.exp(np.log(.5)/halflife) if halflife else .99
        self.warmup_weight= (1-self.alpha**20)/(1-self.alpha); # >20 points for warmup
        self.N=0
        if int_err_halflife is None and halflife:
            int_err_halflife = max(halflife/100,1) 
        self.int_err_alpha = np.exp(np.log(.5)/int_err_halflife) if int_err_halflife else .999
        self.K_int_err = K_int_err

    def reset(self, keep_err=False):
        self.N = 0
        self.X0 = None
        self.Y0 = None
        self.sX = 0
        self.sY = 0
        self.sXX = 0
        self.sYX = 0
        self.sYY = 0
        self.a=1
        self.b=0
        if not keep_err:
            self.int_err = 0
            self.abs_err=0
            self.int_err_N = 0

    def fit(self,X,Y,keep_err=False):
        self.reset(keep_err)
        if hasattr(X,'__iter__'):
            self.N = len(X)
            self.X0=X[0,...] 
            self.Y0=Y[0,...] 
            if len(X)>1 :
                self.transform(X[1:,...],Y[1:,...])
        else:
            self.N = 1
            self.X0 = X
            self.Y0 = Y
        self.a = 1
        self.b = self.Y0 - self.a * self.X0

    def transform(self,X,Y):
        if self.N==0:
            self.fit(X,Y)
            return Y

        #if np.all(X==self.Xlast) or np.all(Y==self.Ylast):
        #    return self.getY(X)
        # get our prediction for this point and use to track our prediction error
        Yest = self.getY(X)
        err  = np.mean(np.abs(Y-Yest))
  
        ## center x/y
        # N.B. be sure to do all the analysis in floating point to avoid overflow
        cX  = np.array(X - self.X0, dtype=float)
        cY  = np.array(Y - self.Y0, dtype=float)
        N = len(X) if hasattr(X,'__iter__') else 1
        # update the 1st and 2nd order summary statistics 
        wght    = self.alpha**N
        # adaptive learning rate as function of integerated error
        if self.N > self.warmup_weight:
            ptwght = max(1, abs(self.int_err/self.int_err_N)*self.K_int_err) #1/max(1,err) if self.N > self.warmup_weight else 1
        else:
            ptwght = 1
        self.N  = wght*self.N   + ptwght*N
        self.sY = wght*self.sY  + ptwght*np.sum(cY,axis=0)
        self.sX = wght*self.sX  + ptwght*np.sum(cX,axis=0)
        self.sYY= wght*self.sYY + ptwght*np.sum(cY*cY,axis=0)
        self.sYX= wght*self.sYX + ptwght*np.sum(cY*cX,axis=0)
        self.sXX= wght*self.sXX + ptwght*np.sum(cX*cX,axis=0)

        # update the fit parameters
        if self.N > 1:
            Yvar  = self.sYY - self.sY * self.sY / self.N
            Xvar  = self.sXX - self.sX * self.sX / self.N
            YXvar = self.sYX - self.sY * self.sX / self.N
            self.a = (YXvar / Xvar + Yvar/YXvar )/2        
        # update the bias given the estimated slope b = mu_y - a * mu_x
        # being sure to include the shift to the origin!
        self.b = (self.Y0 + self.sY / self.N) - self.a * (self.X0 + self.sX / self.N)
        #self.b = (self.sY / self.N) - self.a * (self.sX / self.N)

        # check for steps in the inputs
        # get our prediction for this point and use to track our prediction error
        Yest = self.getY(X)
        err  = np.mean(Y-Yest)

        # track the prediction error, with long and short half-life
        self.abs_err = self.abs_err*wght + abs(err)
        # track with step window
        int_err_wght = self.int_err_alpha**N
        self.int_err_N = self.int_err_N * int_err_wght + N
        self.int_err = self.int_err*int_err_wght + err

        # only return the estimate if we've warmed up the tracker
        if self.N < self.warmup_weight:
            return Y

        # detect change in statistics by significant difference in error statistics
        # between long and short halflife
        #if (self.step_err / self.step_N) > (self.err / self.N) * self.step_threshold:
        #    print("step-detected")
        #    #self.fit(X,Y,keep_err=True)
        #    #Yest = Y

        return Yest

    def getX(self,y):
        return ( y  - self.b ) / self.a 

    def getY(self,x):
        return self.a * x + self.b

    @staticmethod
    def testcase():
        X = np.arange(1000) + np.random.randn(1)*1e6
        a = 1000/50 
        b = 1000*np.random.randn(1)
        Ytrue= a*X+b
        Y    = Ytrue+ np.random.standard_normal(Ytrue.shape)*10

        import glob
        import os
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
        savefile = max(files, key=os.path.getctime)
        #savefile = "C:\\Users\\Developer\\Downloads\\mark\\mindaffectBCI_brainflow_200911_1339.txt" 
        #savefile = "C:/Users/Developer/Downloads/khash/mindaffectBCI_brainflow_ipad_200908_1938.txt"
        from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_messages
        from mindaffectBCI.utopiaclient import DataPacket
        print("Loading: {}".format(savefile))
        msgs = read_mindaffectBCI_messages(savefile,regress=None) # load without time-stamp fixing.
        dp = [ m for m in msgs if isinstance(m,DataPacket)]
        nsc = np.array([ (m.samples.shape[0],m.sts,m.timestamp) for m in dp])
        X = np.cumsum(nsc[:,0])
        Y = nsc[:,1]
        Ytrue = nsc[:,2]

        ltt = linear_trend_tracker(1000)
        ltt.fit(X[0],Y[0]) # check scalar inputs
        step = 1
        idxs = list(range(1,X.shape[0],step))
        ab = np.zeros((len(idxs),2))
        print("{}) a={} b={}".format('true',a,b))
        dts = np.zeros((Y.shape[0],))
        dts[0] = ltt.getY(X[0])
        for i,j in enumerate(idxs):
            dts[j:j+step] = ltt.transform(X[j:j+step],Y[j:j+step])
            ab[i,:] = (ltt.a,ltt.b)
            yest = ltt.getY(X[j])
            err =  yest - Y[j]
            if abs(err)> 1000:
                print("{}) argh! yest={} ytrue={} err={}".format(i,yest,Ytrue[j],err))
            if i < 100:
                print("{:4d}) a={:5f} b={:5f}\ty_est-y={:2.5f}".format(j,ab[i,0],ab[i,1],
                        Y[j]-yest))

        import matplotlib.pyplot as plt
        ab,res,_,_ = np.linalg.lstsq(np.append(X[:,np.newaxis],np.ones((X.shape[0],1)),1),Y,rcond=-1)
        ots = X*ab[0]+ab[1]        
        idx=range(X.shape[0])
        plt.plot(X[idx],Y[idx]- X[idx]*ab[0]-Y[0],label='server ts')
        plt.plot(X[idx],dts[idx] - X[idx]*ab[0]-Y[0],label='regressed ts (samp vs server)')
        plt.plot(X[idx],ots[idx] - X[idx]*ab[0]-Y[0],label='regressed ts (samp vs server) offline')

        err = Y - X*ab[0] - Y[0]
        cent = np.median(err); scale=np.median(np.abs(err-cent))
        plt.ylim((cent-scale*5,cent+scale*5))

        plt.legend()
        plt.show()
        

try:
    from scipy.signal import butter, bessel, sosfilt, sosfilt_zi
except:
#if True:
    # use the pure-python fallbacks
    def sosfilt(sos,X,axis,zi):
        return sosfilt_2d_py(sos,X,axis=axis,zi=zi)
    def sosfilt_zi(sos):
        return sosfilt_zi_py(sos)
    def butter(order,freq,btype,output):
        return butter_py(order,freq,btype,output)

def sosfilt_zi_warmup(zi, X, axis=-1, sos=None):
    if axis < 0: # no neg axis
        axis = X.ndim+axis
    # zi => (order,...,2,...)
    zi = np.reshape(zi, (zi.shape[0],) + (1,)*(axis) + (zi.shape[1],) + (1,)*(X.ndim-axis-1))

    # make a programattic index expression to support arbitary axis
    idx = [slice(None)]*X.ndim
    # get the index to start the warmup
    warmupidx = 0 if sos is None else min(sos.size*3,X.shape[axis]-1)

    # center on 1st warmup value
    idx[axis] = slice(warmupidx,warmupidx+1)
    zi = zi * X[tuple(idx)] 

    # run the filter on the rest of the warmup values
    if not sos is None and warmupidx>3: 
        idx[axis] = slice(warmupidx,1,-1)
        _, zi  = sosfilt(sos, X[tuple(idx)], axis=axis, zi=zi)
        
    return zi

def iir_sosfilt_sos(stopband, fs, order=4, ftype='butter', passband=None, verb=0):
    ''' given a set of filter cutoffs return butterworth sos coefficients '''
    sos=[]

    # convert to normalized frequency, Note: not to close to 0/1
    if stopband is not None:
        if not hasattr(stopband[0],'__iter__'):
            stopband=(stopband,)

        for sb in stopband:
            btype = None
            if type(sb[-1]) is str:
                btype = sb[-1]
                sb = sb[:-1]

            # convert to normalize frequency
            sb = np.array(sb,dtype=np.float32)
            sb[sb<0] = (fs/2)+sb[sb<0]+1 # neg freq count back from nyquist
            Wn  = sb/(fs/2)

            if  Wn[1] < .0001 or .9999 < Wn[0]: # no filter
                continue

            # identify type from frequencies used, cliping if end of frequency range
            if Wn[0] < .0001:
                Wn = Wn[1]
                btype = 'highpass' if btype is None or btype == 'bandstop' else 'lowpass'
            elif .9999 < Wn[1]:
                Wn = Wn[0]
                btype = 'lowpass' if btype is None or btype == 'bandstop' else 'highpass'

            elif btype is None: # .001 < Wn[0] and Wn[1] < .999:
                btype = 'bandstop'

            if verb>0: print("{}={}={}".format(btype,sb,Wn))

            if ftype == 'butter':
                sosi = butter(order, Wn, btype=btype, output='sos')
            elif ftype == 'bessel':
                sosi = bessel(order, Wn, btype=btype, output='sos', norm='phase')
            else:
                raise ValueError("Unrecognised filter type")

            sos.append(sosi)

    # single big filter cascade
    sos = np.concatenate(sos,axis=0)
    #print("sos={}".format(sos.shape))
    return sos

def butter_sosfilt(X, stopband, fs, order=6, axis=-2, zi=None, passband=None, verb=True, ftype='butter'):
    ''' use a (cascade of) butterworth SOS filter(s) to band-pass and (cascade of) band stop X along axis '''
    if axis < 0: # no neg axis
        axis = X.ndim+axis
    # TODO []: auto-order determination?
    sos = iir_sosfilt_sos(stopband, fs, order, passband=passband, ftype=ftype)
    sos = sos.astype(X.dtype) # keep as single precision

    if axis == X.ndim-2 and zi is None:
        zi = sosfilt_zi(sos) # (order,2)
        zi.astype(X.dtype)
        zi = sosfilt_zi_warmup(zi, X, axis, sos)

    else:
        zi = None
        print("Warning: not warming up...")

    # Apply the warmed up filter to the input data
    #print("zi={}".format(zi.shape))
    if not zi is None:
        #print("filt:zi X{} axis={}".format(X.shape,axis))
        X, zi  = sosfilt(sos, X, axis=axis, zi=zi)
    else:
        print("filt:no-zi")
        X  = sosfilt(sos, X, axis=axis) # zi=zi)

    # return filtered data, filter-coefficients, filter-state
    return (X, sos, zi)

def save_butter_sosfilt_coeff(filename=None, stopband=((0,5),(25,-1)), fs=200, order=6, ftype='butter'):
    ''' design a butterworth sos filter cascade and save the coefficients '''
    import pickle
    sos = iir_sosfilt_sos(stopband, fs, order, passband=None, fytpe=ftype)
    zi = sosfilt_zi(sos)
    if filename is None:
        # auto-generate descriptive filename
        filename = "{}_stopband{}_fs{}.pk".format(btype,stopband,fs)
    with open(filename,'wb') as f:
        pickle.dump(sos,f)
        pickle.dump(zi,f)
        f.close()

def test_butter_sosfilt():
    fs= 100
    X = np.random.randn(fs*10,2)
    X = np.cumsum(X,0)
    X = X + np.random.randn(1,X.shape[1])*100 # include start shift
    import matplotlib.pyplot as plt

    plt.clf();plt.subplot(511);plt.plot(X);

    pbs=(((0,1),(40,-1)),(10,-1),((5,10),(15,20),(45,50)))
    for i,pb in enumerate(pbs):
        Xf,_,_ = butter_sosfilt(X,pb,fs)
        plt.subplot(5,1,i+2);plt.plot(Xf);plt.title("{}".format(pb))

    # test incremental application
    pb=pbs[0]
    sos=None
    zi =None
    Xf=[]
    for i in range(0,X.shape[0],fs):
        if sos is None:
            # init filter and do 1st block
            Xfi,sos,zi = butter_sosfilt(X[i:i+fs,:],pb,fs,axis=-2)
        else: # incremenally apply
            Xfi,zi = sosfilt(sos,X[i:i+fs,:],axis=-2,zi=zi)
        Xf.append(Xfi)
    Xf = np.concatenate(Xf,axis=0)
    plt.subplot(5,1,5);plt.plot(Xf);plt.title("{} - incremental".format(pb))
    plt.show()

    # test diff specifications
    pb = ((0,1),(40,-1)) # pair stops
    Xf0,_,_ = butter_sosfilt(X,pb,fs,axis=-2)
    plt.subplot(3,1,1);plt.plot(Xf0);plt.title("{}".format(pb))
 
    pb = (1,40,'bandpass') # single pass
    Xfi,_,_ = butter_sosfilt(X,pb,fs,axis=-2)
    plt.subplot(3,1,2);plt.plot(Xfi);plt.title("{}".format(pb))

    pb = (1,40,'bandpass') # single pass
    Xfi,_,_ = butter_sosfilt(X,pb,fs,axis=-2,ftype='bessel')
    plt.subplot(3,1,3);plt.plot(Xfi);plt.title("{} - bessel".format(pb))
    plt.show()


# TODO[] : cythonize?
# TODO[X] : vectorize over d? ---- NO. 2.5x *slower*
def sosfilt_2d_py(sos,X,axis=-2,zi=None):
    ''' pure python fallback for second-order-sections filter in case scipy isn't available '''
    X = np.asarray(X)
    sos = np.asarray(sos)

    if zi is None:
        returnzi = False
        zi = np.zeros((sos.shape[0],2,X.shape[-1]),dtype=X.dtype)
    else:
        returnzi = True
        zi = np.asarray(zi)

    Xshape = X.shape
    if not X.ndim == 2:
        print("Warning: X>2d.... treating as 2d...")
        X = X.reshape((-1,Xshape[-1]))

    if axis < 0:
        axis = X.ndim + axis

    if not axis == X.ndim-2:
        raise ValueError("Only for time in dim 0/-2")

    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError('sos must be shape (n_sections, 6)')

    if zi.ndim != 3 or zi.shape[1] != 2 or zi.shape[2] != X.shape[1]:
        raise ValueError('zi must be shape (n_sections, 2, dim)')

    # pre-normalize sos if needed
    for j in range(sos.shape[0]):
        if sos[j,3] != 1.0:
            sos[j,:] = sos[j,:]/sos[j,3]
    
    n_signals = X.shape[1]
    n_samples = X.shape[0]
    n_sections = sos.shape[0]

    # extract the a/b
    b = sos[:,:3]
    a = sos[:,4:]

    # loop over outputs
    x_n = 0
    for i in range(n_signals):
        for n in range(n_samples):
            for s in range(n_sections):
                x_n = X[n, i]
                # use direct II transposed structure
                X[n, i] = b[s, 0] * x_n + zi[s, 0, i]
                zi[s, 0, i] = b[s, 1] * x_n - a[s, 0] * X[n, i] + zi[s, 1, i]
                zi[s, 1, i] = b[s, 2] * x_n - a[s, 1] * X[n, i]

    # back to input shape
    if not len(Xshape) == 2:
        X = X.reshape(Xshape)

    # match sosfilt, only return zi if given zi
    if returnzi :
        return X, zi
    else:
        return X

def sosfilt_zi_py(sos):
    ''' compute an initial state for a second-order section filter '''
    sos = np.asarray(sos)
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError('sos must be shape (n_sections, 6)')

    n_sections = sos.shape[0]
    zi = np.empty((n_sections, 2))
    scale = 1.0
    for section in range(n_sections):
        b = sos[section, :3]
        a = sos[section, 3:]
        
        if a[0] != 1.0:
            # Normalize the coefficients so a[0] == 1.
            b = b / a[0]
            a = a / a[0]
        IminusA = np.eye(n - 1) - linalg.companion(a).T
        B = b[1:] - a[1:] * b[0]
        # Solve zi = A*zi + B
        lfitler_zi = np.linalg.solve(IminusA, B)
        zi[section] = scale * lfilter_zi
        scale *= b.sum() / a.sum()
        
    return zi

def test_sosfilt_py():
    import pickle
    with open('butter_stopband((0, 5), (25, -1))_fs200.pk','rb') as f:
        sos = pickle.load(f)
        zi = pickle.load(f)

    X = np.random.randn(10000,3)
    print("X={} sos={}".format(X.shape,sos.shape))

    Xsci = sosfilt(sos,X.copy(),-2)
    Xpy = sosfilt_2d_py(sos,X.copy(),-2)

    import matplotlib.pyplot as plt
    plt.clf();
    plt.subplot(411);plt.plot(X[:500,:]);plt.title('X')
    plt.subplot(412);plt.plot(Xsci[:500,:]);plt.title('Xscipy')
    plt.subplot(413);plt.plot(Xpy[:500,:]);plt.title('Xpy')
    plt.subplot(414);plt.plot(Xsci-Xpy);plt.title('Xsci - Xpy')

# def butter_py(order,fc,fs,btype,output):
#     ''' pure python butterworth filter synthesis '''
#     if fc>=fs/2:
#         error('fc must be less than fs/2')

#     # I.  Find poles of analog filter
#     k= np.arange(order)
#     theta= (2*k -1)*np.pi/(2*order);
#     pa= -sin(theta) + j*cos(theta);     # poles of filter with cutoff = 1 rad/s
#     #
#     # II.  scale poles in frequency
#     Fc= fs/np.pi * tan(np.pi*fc/fs);          # continuous pre-warped frequency
#     pa= pa*2*np.pi*Fc;                     # scale poles by 2*pi*Fc
#     #
#     # III.  Find coeffs of digital filter
#     # poles and zeros in the z plane
#     p= (1 + pa/(2*fs))/(1 - pa/(2*fs))      # poles by bilinear transform
#     q= -np.ones((1,N));                   # zeros
#     #
#     # convert poles and zeros to polynomial coeffs
#     a= poly(p);                   # convert poles to polynomial coeffs a
#     a= real(a);
#     b= poly(q);                   # convert zeros to polynomial coeffs b
#     K= sum(a)/sum(b);             # amplitude scale factor
#     b= K*b;    

if __name__=='__main__':
    test_butter_sosfilt()
    linear_trend_tracker.testcase()