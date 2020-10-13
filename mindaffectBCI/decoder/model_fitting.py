import numpy as np
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, zero_outliers, crossautocov, updateCyy, updateCxy, autocov, updateCxx, plot_erp, plot_summary_statistics, plot_factoredmodel
from mindaffectBCI.decoder.multipleCCA import multipleCCA
from mindaffectBCI.decoder.scoreOutput import scoreOutput, dedupY0
from mindaffectBCI.decoder.utils import window_axis
from mindaffectBCI.decoder.scoreStimulus import scoreStimulus, scoreStimulusCont, factored2full
from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
from mindaffectBCI.decoder.stim2event import stim2event
from mindaffectBCI.decoder.normalizeOutputScores import normalizeOutputScores, estimate_Fy_noise_variance
from mindaffectBCI.decoder.zscore2Ptgt_softmax import calibrate_softmaxscale

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.exceptions import NotFittedError
except:
#  if True:
    # placeholder classes for when sklearn isn't available
    class StratifiedKFold:
        def __init__(self, n_splits):
            self.n_splits = n_splits

        def split(self, X, Y):
            shape = X.shape
            self.splits = np.linspace(0, shape[0], self.n_splits+1, dtype=int, endpoint=True).tolist()
            self._index = 0
            return self

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self.splits)-1:
                train_idx = list(range(self.splits[self._index]))+list(range(self.splits[self._index+1], self.splits[-1]))
                val_idx = list(range(self.splits[self._index], self.splits[self._index+1]))
                self._index = self._index + 1
            else:
                raise StopIteration
            return (train_idx, val_idx)


    class BaseEstimator:
        pass


    class ClassifierMixin:
        pass

    class NotFittedError(Exception):
        pass
    
class BaseSequence2Sequence(BaseEstimator, ClassifierMixin):
    '''Base class for sequence-to-sequence learning.  Provides, prediction and scoring functions, but not the fitting method'''
    def __init__(self, evtlabs=('re','fe'), tau=18, offset=0, outputscore='ip'):
        self.evtlabs = evtlabs if evtlabs is not None else ('re','fe')
        self.tau = tau
        self.offset = offset
        self.outputscore = outputscore
        self.verb = 0
        self.priorweight = 120
        if self.offset>0 or self.offset<-tau:
            raise NotImplementedError("Offsets of more than a negative window are not supported yet!")
        
    def stim2event(self, Y, prevY=None):
        '''transform Stimulus-encoded to brain-encoded, if needed'''
        # convert from stimulus coding to brain response coding
        if self.evtlabs is not None:
            Y = stim2event(Y, self.evtlabs, axis=-2, oM=prevY) # (tr, samp, Y, e)
        else:
            if Y.ndim == 3:
                Y = Y[:,:,:,np.newaxis] # (tr,samp,Y,e)

        return Y

    def fit(self, X, Y):
        '''fit model mapping 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, e)'''
        raise NotImplemented

    def is_fitted(self):
        return hasattr(self,"W_")

    def predict(self, X, Y, dedup0=None, prevY=None):
        '''make predictions on multi-dim time series: X = (tr, samp, d), Y = (tr, samp, e)
        
        N.B. this implementation assumes linear coefficients in W_ (nM,nfilt,d) and R_ (nM,nfilt,nE,tau)'''
        if not self.is_fitted():
            # only if we've been fitted!
            raise NotFittedError
        # convert from stimulus coding to brain response coding
        Y = self.stim2event(Y, prevY)
        # valid performance
        Fe = self.transform(X) # (nM, nTrl, nSamp, nE)
        Fy = scoreOutput(Fe, Y, outputscore=self.outputscore, dedup0=dedup0, R=self.R_, offset=self.offset) #(nM, nTrl, nSamp, nY)
        # BODGE: strip un-needed model dimension
        if Fy.shape[0] == 1 and Fy.ndim > 3:
            Fy = Fy.reshape(Fy.shape[1:])
        return Fy

    def transform(self, X):
        ''' estimate the stimulus properties from the raw data '''
        return scoreStimulus(X, self.W_, self.R_, self.b_, offset=self.offset)

    def decode_proba(self, Fy, minDecisLen=0, marginalizemodels=True):
        # build optional arguments if set
        kwargs=dict()
        if hasattr(self,'sigma0_') and self.sigma0_ is not None:
            # include the score prior
            #print('priorsigma=({},{})'.format(self.sigma0_,self.priorweight))
            kwargs['priorsigma']=(self.sigma0_,self.priorweight)
        if hasattr(self,'softmaxscale_') and self.softmaxscale_ is not None:
            kwargs['softmaxscale']=self.softmaxscale_

        Yest, Perr, Ptgt, _, _ = decodingSupervised(Fy, minDecisLen=minDecisLen, marginalizemodels=marginalizemodels, **kwargs)
        return Ptgt #(nTrl, nEp, nY)

    
    def predict_proba(self, X, Y, marginalizemodels=False, minDecisLen=-1, dedup0=None, prevY=None):
        Fy = self.predict(X, Y, dedup0=dedup0, prevY=prevY)
        return self.decode_proba(Fy,marginalizemodels=marginalizemodels, minDecisLen=minDecisLen)

    def score(self, X, Y):
        '''score this model on this data, N.B. for model_selection higher is *better*'''
        # score, removing duplicates for true target (objId==0) so can compute score
        Fy = self.predict(X, Y, dedup0=True) # (nM, nTrl, nSamp, e)

        return BaseSequence2Sequence.audc_score(Fy)

    @staticmethod
    def audc_score(Fy):
        '''compute area under decoding curve score from Fy, *assuming* Fy[:,:,0] is the *true* classes score'''
        sFy = np.cumsum(Fy, axis=-2) # (nM, nTrl, nSamp, nY)
        Yi  = np.argmax(sFy, axis=-1) # output for every model*trial*sample
        score = np.sum((Yi == 0).ravel())/Yi.size # total amount time was right, higher=better
        return score

    def cv_fit(self, X, Y, cv=5, fit_params:dict=dict(), verbose:bool=0, return_estimator:bool=True, calibrate_softmax:bool=True, dedup0:bool=True):
        ''' cross validated fit to the data.  N.B. write our own as sklearn doesn't work for getting the estimator values for structured output.'''
        # TODO [] : make more computationally efficient by pre-computing the updateSummaryStatistics etc.
        # TODO [] : conform to sklearn cross_validate signature
        # TODO [] : move into a wrapper class
        if cv == True:  cv = 5
        if isinstance(cv, int):
            if X.shape[0] > 1:
                cv = StratifiedKFold(n_splits=min(cv, X.shape[0])).split(np.zeros(X.shape[0]), np.zeros(Y.shape[0]))
            else: # single trial, train/test on all...
                cv = [(slice(1), slice(1))] # N.B. use slice to preserve dims..

        if return_estimator:
            Fy = np.zeros(Y.shape) if Y.ndim<=3 else np.zeros(Y.shape[:-1]) # (nTrl, nEp, nY)
        else:
            Fy = None
        scores = []
        if verbose > 0:
            print("CV:", end='')
        for i, (train_idx, valid_idx) in enumerate(cv):
            if verbose > 0:
                print(".", end='', flush=True)
            #print("trn={} val={}".format(train_idx,valid_idx))
            
            self.fit(X[train_idx, ...], Y[train_idx, ...], **fit_params)

            # predict, forcing removal of copies of  tgt=0 so can score
            Fyi = self.predict(X[valid_idx, ...], Y[valid_idx, ...], dedup0=dedup0)
            if return_estimator:
                Fy[valid_idx, ...] = Fyi
            
            scores.append(self.audc_score(Fyi))

        # final retrain with all the data
        self.fit(X, Y)

        self.sigma0_ = None
        self.softmaxscale_ = None
        if return_estimator:
            # estimate prior for Fy using the cv'd Fy
            #self.sigma0_ = np.sum(Fy.ravel()**2) / Fy.size
            #print('Fy={}'.format(Fy.shape))
            # N.B. need to match the filter used in the decoder..
            self.sigma0_, _ = estimate_Fy_noise_variance(Fy, priorsigma=None)  # per-trial
            #print('Sigma0{} = {}'.format(self.sigma0_.shape,self.sigma0_))
            self.sigma0_ = np.median(self.sigma0_.ravel())  # ave
            print('Sigma0 = {}'.format(self.sigma0_))

            if calibrate_softmax:
                # calibrate the softmax scale to give more valid probabilities
                nFy,_,_,_,_ = normalizeOutputScores(Fy, minDecisLen=-10, priorsigma=(self.sigma0_,self.priorweight))
                self.softmaxscale_ = calibrate_softmaxscale(nFy)

        return {'estimator': Fy, 'test_score': scores}
    
    def plot_model(self, **kwargs):
        if not self.R_ is None:
            print("Plot Factored Model")
            if hasattr(self, 'A_'):
                plot_factoredmodel(self.A_, self.R_, evtlabs=self.evtlabs, spatial_filter_type='Pattern', **kwargs)
            else:
                plot_factoredmodel(self.W_, self.R_, evtlabs=self.evtlabs, **kwargs)
        else:
            plot_erp(self.W_, evtlabs=self.evtlabs, **kwargs)

    
class MultiCCA(BaseSequence2Sequence):
    ''' Sequence 2 Sequence learning using CCA as a bi-directional forward/backward learning method '''
    def __init__(self, evtlabs=('re','fe'), tau=18, offset=0, rank=1, reg=(1e-8,None), rcond=(1e-4,1e-8), badEpThresh=6, symetric=False, center=True, CCA=True, **kwargs):
        super().__init__(evtlabs=evtlabs, tau=tau,  offset=offset, **kwargs)
        self.rank = rank
        self.reg  = reg
        self.rcond= rcond
        self.badEpThresh = badEpThresh
        self.symetric = symetric
        self.center = center
        self.CCA = CCA

    def fit(self, X, Y, stimTimes=None):
        '''fit 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, Y)'''
        # map to event sequence
        Y = self.stim2event(Y) # (tr,samp,nY,e)
        # extract the true target to fit to, using horible slicing trick
        Y_true = Y[..., 0:1, :] #  (tr,samp,1,e)
        # get summary statistics
        #print("X={} |X|={}".format(X.shape,np.sum(X**2,axis=(0,1))))
        Cxx, Cxy, Cyy = updateSummaryStatistics(X, Y_true, stimTimes, tau=self.tau, offset=self.offset, badEpThresh=self.badEpThresh, center=self.center)
        #print("diag(Cxx)={}".format(np.diag(Cxx)))
        # do the CCA fit
        J, W, R = multipleCCA(Cxx, Cxy, Cyy, reg=self.reg, rank=self.rank, rcond=self.rcond, CCA=self.CCA, symetric=self.symetric)
        # maintain type compatiability
        W = W.astype(X.dtype)
        R = R.astype(X.dtype)
        self.W_ = W #(nM,rank,d)
        self.R_ = R #(nM,rank,e,tau)
        self.fit_b(X) #(nM,e)
        # store A_ for plotting later
        self.A_ = np.einsum("de,Mkd->Mke",Cxx,W)


        return self

    def fit_b(self,X):
        """fit the bias parameter given the other parameters and a dataset

        Args:
            X (np.ndarray): the target data
        """        
        if self.center: # use bias terms to correct for data centering
            muX = np.mean(X.reshape((-1,X.shape[-1])),0)
            self.b_ = -np.einsum("Mkd,d,Mket->Me",self.W_,muX,self.R_) # (nM,e)
            self.b_ = self.b_[0,:] # strip model dim..
        else:
            self.b_ = None
        return self


    def cv_fit(self, X, Y, cv=5, fit_params:dict=dict(), verbose:bool=0, 
               return_estimator:bool=True, calibrate_softmax:bool=True, dedup0:bool=True, ranks=None):
        ''' cross validated fit to the data.  N.B. write our own as sklearn doesn't work for getting the estimator values for structured output.'''
        if ranks is None or len(ranks)==1 :
            return super(MultiCCA,self).cv_fit(X,Y,cv,fit_params,verbose,return_estimator,calibrate_softmaxscale,dedup0)

        # fast path for cross validation over rank
        cv_in = cv.copy() if hasattr(cv,'copy') else cv # save copy of cv info for later
        if cv == True:  cv = 5
        if isinstance(cv, int):
            if X.shape[0] > 1:
                cv = StratifiedKFold(n_splits=min(cv, X.shape[0])).split(np.zeros(X.shape[0]), np.zeros(Y.shape[0]))
            else: # single trial, train/test on all...
                cv = [(slice(1), slice(1))] # N.B. use slice to preserve dims..

        if return_estimator:
            Fy = np.zeros(Y.shape) if Y.ndim<=3 else np.zeros(Y.shape[:-1]) # (nTrl, nEp, nY)
        else:
            Fy = None
        scores = []
        if verbose > 0:
            print("CV:", end='')

        maxrank = max(ranks)
        self.rank = maxrank
        scores = [[] for i in range(len(ranks))]
        for i, (train_idx, valid_idx) in enumerate(cv):
            if verbose > 0:
                print(".", end='', flush=True)
            #print("trn={} val={}".format(train_idx,valid_idx))

            # 1) fit with max-rank            
            self.fit(X[train_idx, ...], Y[train_idx, ...], **fit_params)

            # 2) Extract the desired rank-sub-models and predict with them
            W = self.W_ #(nM,rank,d)
            R = self.R_ #(nM,rank,e,tau)
            for i,r in enumerate(ranks):
                self.W_ = W[...,:r,:]
                self.R_ = R[...,:r,:,:]
                self.fit_b(X[train_idx,...])
                # predict, forcing removal of copies of  tgt=0 so can score
                Fyi = self.predict(X[valid_idx, ...], Y[valid_idx, ...], dedup0=dedup0)
                scores[i].append(self.audc_score(Fyi))
        
        #3) get the *best* rank
        scores= np.mean(np.array(scores),axis=-1) # (ranks,folds) -> ranks
        print("Rank score: " + ", ".join(["{}={:3.2f}".format(r,s) for (r,s) in zip(ranks,scores)]),end='')
        maxri = np.argmax(scores)
        self.rank = ranks[maxri]
        print(" -> best={}".format(self.rank))

        # final retrain with all the data
        # TODO[]: just use a normal fit, and get the Fy from the above CV loop 
        res = super(MultiCCA,self).cv_fit(X, Y, cv_in, fit_params, verbose, return_estimator, calibrate_softmaxscale, dedup0)
        return res

    
class FwdLinearRegression(BaseSequence2Sequence):
    ''' Sequence 2 Sequence learning using forward linear regression  X = A*Y '''
    def __init__(self, evtlabs=('re','fe'), tau=18, offset=0, reg=None, rcond=1e-6, badEpThresh=6, center=True, **kwargs):
        super().__init__(evtlabs=evtlabs, tau=tau, offset=offset, **kwargs)
        self.reg = reg
        self.rcond = rcond
        self.badEpThresh = badEpThresh
        self.center = center
        if offset != 0:
            print("WARNING: not tested with offset!! YMMV")

    def fit(self, X, Y):
        '''fit 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, e)'''
        # map to event sequence
        Y = self.stim2event(Y) # (tr,samp,nY,e)
        # extract the true target to fit to
        Y_true = Y[..., 0, :] # (tr,samp, e)
        # get summary statistics
        # first clean up the data..
        X, Y_true = zero_outliers(X, Y_true, self.badEpThresh)
        if self.center:
            # TODO[]: make work in-place w/o modify X
            X = X.copy()
            muX = np.mean(X.reshape((-1,X.shape[-1])),0)
            X = X - muX
        # TODO: should be negative tau to represent causality of Y->X, so earlier Y cause X
        Cyteyte = crossautocov(Y_true, Y_true, tau=self.tau) # (tau,e,tau,e) cross-cov shifted Y with itself
        # N.B. should be shifted Y with X, but we want Y *backwards* in time, thus
        #      as a *BODGE* we instead window X, as forward for X == backwards for Y
        #Cytexd  = crossautocov(Y_true, X, tau=[-self.tau, 1]) # (tau,e,1,d) shifted Y with X
        Cytexd  = crossautocov(Y_true, X, tau=[1, self.tau], offset=self.offset) # (1,e,tau,d) shifted X with Y
        Cytexd  = np.moveaxis(Cytexd, (0, 1, 2, 3), (2, 1, 0, 3)) # (tau,e,1,d)
        # fit:
        # Make 2d
        # (tau,e,tau,e) -> ((tau*e),(tau*e))
        Cyy2d = Cyteyte.reshape((Cyteyte.shape[0]*Cyteyte.shape[1],
                                 Cyteyte.shape[2]*Cyteyte.shape[3]))
        # (tau,e,1,d) -> ((tau*e),d)
        Cyx2d = Cytexd.reshape( (Cytexd.shape[0]*Cytexd.shape[1],
                                 Cytexd.shape[3]))
        # add regularization
        if self.reg:
            # **IN-PLACE modify Cyy2d..**
            diag=Cyy2d.diagonal()
            np.fill_diagonal(Cyy2d, diag+self.reg*np.max(diag))
        
        # fit by solving the matrix equation: Cyy*A=Cyx -> A=Cyy**-1Cyx
        # Q: solve or lstsq?
        #R = np.linalg.solve(Cyy2d, Cyx2d) # ((tau*e),d)
        R,_,_,_ = np.linalg.lstsq(Cyy2d, Cyx2d, rcond=self.rcond) # ((tau*e),d)
        # convert back to 2d (tau,e,d)
        R = R.reshape((Cyteyte.shape[0], Cyteyte.shape[1], R.shape[1]))
        # BODGE: now moveaxis to make consistent with the prediction functions
        self.W_ = np.moveaxis(R, (0, 1, 2), (1, 0, 2)) #(tau,e,d)->(e,tau,d)
        self.A_ = self.W_
        self.R_ = None
        if self.center:
            self.b_ = -np.einsum("etd,d->e", self.W_, muX) #(e,)
        else:
            self.b_ = None

        return self

class BwdLinearRegression(BaseSequence2Sequence):
    ''' Sequence 2 Sequence learning using backward linear regression  W*X = Y '''
    def __init__(self, evtlabs=('re','fe'), tau=18, offset=0, reg=None, rcond=1e-5, badEpThresh=6, center=True, **kwargs):
        super().__init__(evtlabs=evtlabs, tau=tau, offset=offset, **kwargs)
        self.reg = reg
        self.rcond = rcond
        self.badEpThresh = badEpThresh
        self.center = center
        if offset != 0:
            print("WARNING: not tested with offset!! YMMV")

    def fit(self, X, Y):
        '''fit 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, e)'''
        # map to event sequence
        Y = self.stim2event(Y) # (tr,samp,nY,e)
        # extract the true target to fit to
        Y = Y[..., 0, :] # (tr,samp, e)
        # first clean up the data..
        X, Y = zero_outliers(X, Y, self.badEpThresh)
        if self.center:
            # TODO[]: make work in-place w/o modify X
            X = X.copy()
            muX = np.mean(X.reshape((-1,X.shape[-1])),0)
            X = X - muX
        # get summary statistics
        Cxtdxtd = crossautocov(X, X, tau=self.tau) # (tau,d,tau,d) cross-cov shifted X with itself
        Cxtdye  = crossautocov(X, Y, tau=[self.tau, 1], offset=self.offset) # (tau,d,1,e) shifted X with Y
        # fit:
        # Make 2d
        # (tau,d,tau,d) -> ((tau*d),(tua*d))
        Cxx2d = Cxtdxtd.reshape((Cxtdxtd.shape[0]*Cxtdxtd.shape[1],
                                 Cxtdxtd.shape[2]*Cxtdxtd.shape[3]))
        # (tau,d,1,e) -> ((tau*d),e)
        Cxy2d = Cxtdye.reshape( (Cxtdye.shape[0]*Cxtdye.shape[1],
                                 Cxtdye.shape[3]))
        # add regularization
        if self.reg:
            # **IN-PLACE modify Cxx2d..**
            diag = Cxx2d.diagonal()
            np.fill_diagonal(Cxx2d, diag+self.reg*np.max(diag))
        
        # fit by solving the matrix equation: Cxx*W=Cxy -> W=Cxx**-1Cxy
        W,_,_,_ = np.linalg.lstsq(Cxx2d, Cxy2d, rcond=self.rcond) # ((tau*d*),e)
        #W = np.linalg.solve(Cxx2d, Cxy2d) # ((tau*d*),e) # can have problems with singular inputs!
        # convert back to 2d
        W = W.reshape((Cxtdxtd.shape[0], Cxtdxtd.shape[1], W.shape[1])) # (tau,d,e)
        # BODGE: moveaxis to make consistent for the prediction functions
        W = np.moveaxis(W, (0, 1, 2), (1, 2, 0)) #(tau,d,e)->(e,tau,d)
        self.W_ = W
        self.R_ = None
        if self.center:
            self.b_ = -np.einsum("etd,d->e",self.W_,muX) #(e,)
        else:
            self.b_ = None

        return self

#from sklearn.linear_model import Ridge
class LinearSklearn(BaseSequence2Sequence):
    ''' Wrap a normal sk-learn classifier for sequence to sequence learning '''
    def __init__(self, clsfr, labelizeY=False, ignore_unlabelled=True, badEpThresh=None, **kwargs):
        super().__init__(**kwargs)
        self.clsfr = clsfr
        self.labelizeY = labelizeY
        self.badEpThresh = badEpThresh
        self.ignore_unlabelled = ignore_unlabelled
        
    @staticmethod
    def sklearn_fit(X, Y, clsfr, tau, offset, labelizeY=False, ignore_unlabelled=True, badEpThresh=None, verb=0):
        '''fit 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, e) using a given sklearn linear_model method'''
        # slice X to get block of data per sample
        X = window_axis(X, winsz=tau, axis=1) # (tr,samp-tau,tau,d)
        # sub-set Y to  be the same size
        Y = Y[:, offset:X.shape[1]+offset, :] # (tr,samp-tau,e)

        # clean up the data
        if badEpThresh is not None:
            X, Y = zero_outliers(X, Y, badEpThresh)

        # convert to a 2d array: (tr*samp, feat) before giving to sklearn
        X2d = np.reshape(X, (np.prod(X.shape[:2]), np.prod(X.shape[2:]))) #((tr*samp-tau),(tau*d))
        Y2d = np.reshape(Y, (np.prod(Y.shape[:2]), np.prod(Y.shape[2:]))) #((tr*samp-tau),e)
        if labelizeY: # convert Y to a label list
            Ylab = np.argmax(Y2d, axis=-1) # ((tr*samp),1)
            unlabelled = np.all(Y2d == 0, axis=-1)
            if np.any(unlabelled):
                if ignore_unlabelled:
                    if verb > 0: print("Warning: removing unlabelled examples from training")
                    keep = ~unlabelled
                    Ylab= Ylab[keep]
                    Y2d = Y2d[keep, ...]
                    X2d = X2d[keep, ...]
                    if verb > 0: print("X={} Y={} Ylab={}".format(X2d.shape,Y2d.shape,np.unique(Ylab)))
                else:
                    if verb > 0: print("Warning: added an extra 'REST' class")
                    # add extra class for anything which isn't labelled already
                    Ylab[unlabelled] = Y2d.shape[-1]
                
            Y2d = Ylab
        # fit the classifier
        clsfr.fit(X2d, Y2d)

        # log the binary training set performance.
        if verb > 0: print("Training score = {}".format(clsfr.score(X2d,Y2d)))

        return clsfr

    def fit(self, X, Y):
        # map to event sequence
        Y = self.stim2event(Y)
        # extract the true target to fit to
        Y_true = Y[..., 0, :] #  (tr,samp,e)
        # fit the model
        self.clsfr = LinearSklearn.sklearn_fit(X, Y_true, self.clsfr, tau=self.tau, offset=self.offset, labelizeY=self.labelizeY, ignore_unlabelled=self.ignore_unlabelled, badEpThresh=self.badEpThresh)
        # extract the solution coefficient and store in W_
        W = self.clsfr.coef_ # ( n_class, n_feat ) = (e,(tau*d)))
        b = self.clsfr.intercept_ # (e,1)
        #print("W={}".format(W.shape))
        #print("b={}".format(b.shape))
        W = np.reshape(W, (W.shape[0] if W.ndim > 1 else 1, self.tau, X.shape[-1])) # (e,tau,d)
        #print("W={}".format(W.shape))

        # ensure the weights vector is the right size ...
        if self.labelizeY and not W.shape[0] == Y_true.shape[-1]:

            if W.shape[0] > Y.shape[-1]:
                if self.verb > 0: print("More weights than outputs! Y={} W={} removed".format(Y.shape[-1],W.shape[0]))
                W = W[:Y.shape[-1], ...]
                b = b[:Y.shape[-1]]
            elif W.shape[0] == 1 and Y.shape[-1] == 2:
                if self.verb > 0: print("Fewer weights than outputs! Y={} W={} binary".format(Y.shape[-1],W.shape[0]))
                # Y is binary, so e==0 -> neg-class,  e==1 -> pos-class
                # TODO[]: check?
                W = np.concatenate((-W, W), axis=0)
                b = np.concatenate((-b, b), axis=0)
            else:
                raise ValueError("Don't know what to do with W")

        self.W_ = W
        self.R_ = None
        self.b_ = b
        #print("b={}".format(b))

        return self
    
def visualize_Fy_Py(Fy, Py):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.ion()
    for trli in range(Fy.shape[0]):
        print('{})'.format(trli), flush=True)
        plt.clf()
        plt.subplot(211);plt.plot(np.cumsum(Fy[trli, :, :], -2))
        plt.subplot(212);plt.plot(Py[trli, :, :])
        plt.title("{})".format(trli))
        plt.draw()
        plt.pause(1)

def plot_model_weights(model:BaseSequence2Sequence, ch_names=None):
    from updateSummaryStatistics import plot_erp
    from scoreStimulus import factored2full
    plot_erp(factored2full(model.W_, model.R_), ch_names=ch_names, evtlabs=model.evtlabs, offset=model.offset)

def testLeadLag():
    import numpy as np
    from utils import testSignal
    from model_fitting import MultiCCA    
    from decodingCurveSupervised import decodingCurveSupervised
    import matplotlib.pyplot as plt

    irf=(0,0,0,0,0,0,0,0,0,1)
    offset=0; # X->lag-by-10
    X,Y,st,A,B = testSignal(nTrl=1,nSamp=500,d=10,nE=1,nY=30,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)

    # reference case lagged response test
    evtlabs=None; tau=10; cca_offset=0
    cca = MultiCCA(evtlabs=evtlabs, tau=tau, offset=cca_offset)
    scores=cca.cv_fit(X,Y)
    Fy = scores['estimator']
    (_) = decodingCurveSupervised(Fy)
        
    # leading X
    irf=(1,0,0,0,0,0,0,0,0,0)
    offset=-9; # X leads by 9
    X,Y,st,A,B = testSignal(nTrl=1,nSamp=5000,d=10,nE=1,nY=30,isi=10,tau=10,offset=offset,irf=irf,noise2signal=0)
    plt.figure(0);plt.clf();plt.plot(X[0,:,0],label='X');plt.plot(Y[0,:,0,0],label='Y');plt.title("offset={}, irf={}".format(offset,irf));plt.legend()

    # no-shift in analysis window
    evtlabs=None; tau=10; cca_offset=0
    cca = MultiCCA(evtlabs=evtlabs, tau=tau,offset=cca_offset)
    scores=cca.cv_fit(X,Y)
    Fy = scores['estimator']
    (_) = decodingCurveSupervised(Fy)

    # shifted analysis window
    evtlabs=None; tau=20; cca_offset=-9
    cca = MultiCCA(evtlabs=evtlabs, tau=tau,offset=cca_offset)
    scores=cca.cv_fit(X,Y)
    Fy = scores['estimator']
    (_) = decodingCurveSupervised(Fy)

    plt.figure(1);plt.clf();plot_model_weights(cca)
    
        
def testcase(dataset='toy',loader_args=dict()):
    from model_fitting import MultiCCA, FwdLinearRegression, BwdLinearRegression, LinearSklearn
    from decodingCurveSupervised import decodingCurveSupervised
    from datasets import get_dataset
    
    loadfn, filenames, dataroot = get_dataset(dataset)
    if dataset=='toy':
        loader_args=dict(tau=10,isi=5,noise2signal=3,nTrl=20,nSamp=50)
    X, Y, coords = loadfn(filenames[0], **loader_args)
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']

    Y = Y[..., 0] # (nsamp, nY)
        
    # raw
    tau = int(.3*fs)
    evtlabs = ('re', 'fe', 'rest')#  ('re', 'ntre')#  ('re', 'ntre', 'rest')#  ('1', '0')#
    cca = MultiCCA(tau=tau, evtlabs=evtlabs)
    print('cca = {}'.format(cca))
    cca.fit(X, Y)
    Fy = cca.predict(X, Y, dedup0=True)
    print("score={}".format(cca.score(X,Y)))
    (_) = decodingCurveSupervised(Fy)

    # cca - cv-fit
    print("CV fitted")
    cca = MultiCCA(tau=tau, rank=1, reg=None, evtlabs=evtlabs)
    cv_res = cca.cv_fit(X, Y, ranks=(1,2,3,5))
    Fy = cv_res['estimator']
    (_) = decodingCurveSupervised(Fy,priorsigma=(cca.sigma0_,cca.priorweight))
        
    from model_fitting import MultiCCA, FwdLinearRegression, BwdLinearRegression
    from updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics
    from scoreStimulus import factored2full
    import matplotlib.pyplot as  plt
    from decodingCurveSupervised import decodingCurveSupervised
    tau = int(.3*fs)
    rank = 1
    evtlabs = ('re', 'fe', 'rest') #('re', 'ntre') # ('re', 'fe')  # ('1', '0') #
    # cca
    cca = MultiCCA(tau=tau, rank=rank, evtlabs=evtlabs, reg=None)
    print("{}".format(cca))
    cca.fit(X, Y)
    Fy = cca.predict(X, Y, dedup0=True)
    (_) = decodingCurveSupervised(Fy)

    plot_erp(factored2full(cca.W_, cca.R_), ch_names=ch_names, evtlabs=evtlabs)
    plt.savefig('W_cca.png')
    
    # fwd-model
    print("Forward Model")
    fwd = FwdLinearRegression(tau=tau, evtlabs=evtlabs, reg=None, badEpThresh=4)
    print("{}".format(fwd))
    fwd.fit(X, Y)
    Fy = fwd.predict(X, Y, dedup0=True)
    (_) = decodingCurveSupervised(Fy)
    
    # bwd-model
    print("Backward Model")
    bwd = BwdLinearRegression(tau=tau, evtlabs=evtlabs, reg=None, badEpThresh=4)
    print("{}".format(bwd))
    bwd.fit(X, Y)
    Fy = bwd.predict(X, Y, dedup0=True)
    (_) = decodingCurveSupervised(Fy)

    Py = bwd.predict_proba(X, Y, dedup0=True)
    visualize_Fy_Py(Fy, Py)

    # sklearn wrapper
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.svm import LinearSVR, LinearSVC
    print("sklearn-ridge")
    ridge = LinearSklearn(tau=tau, evtlabs=evtlabs, clsfr=Ridge(alpha=0))
    print("{}".format(ridge))
    ridge.fit(X, Y)
    Fy = ridge.predict(X, Y, dedup0=True)
    (_) = decodingCurveSupervised(Fy)


    print("sklear-lr")
    lr = LinearSklearn(tau=tau, evtlabs=evtlabs, clsfr=LogisticRegression(C=1, multi_class='multinomial', solver='sag'), labelizeY=True)
    print("{}".format(lr))
    lr.fit(X, Y)
    Fy = lr.predict(X, Y, dedup0=True)
    (_) = decodingCurveSupervised(Fy)

    print("sklear-svc")
    svc = LinearSklearn(tau=tau, evtlabs=evtlabs, clsfr=LinearSVC(C=1, multi_class='ovr'), labelizeY=True)
    print("{}".format(svc))
    svc.fit(X, Y)
    Fy = svc.predict(X, Y, dedup0=True)
    (_) = decodingCurveSupervised(Fy)

    
    plot_erp(factored2full(svc.W_, svc.R_), ch_names=ch_names, evtlabs=evtlabs)
    plt.savefig('W_svc.png')
   
    # hyper-parameter optimization with cross-validation
    from sklearn.model_selection import GridSearchCV
    tuned_parameters={'rank':[1, 2, 3, 5], 'tau':[int(dur*fs) for dur in [.2, .3, .5, .7]], 'evtlabs':[['re', 'fe'], ['re', 'ntre'], ['0','1']]}
    cv_cca = GridSearchCV(MultiCCA(), tuned_parameters)
    cv_cca.fit(X, Y)
    print("CVOPT:\n\n{} = {}\n".format(cv_cca.best_estimator_, cv_cca.best_score_))
    means = cv_cca.cv_results_['mean_test_score']
    stds = cv_cca.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, cv_cca.cv_results_['params']):
        print("{:5.3f} (+/-{:5.3f}) for {}".format(mean, std * 2, params))
    print()

    # use the best setting to fit a normal model, and get it's cv estimated predictions
    cca = MultiCCA()
    cca.set_params(**cv_cca.best_params_) # Note: **dict -> k,v argument array
    cv_res = cca.cv_fit(X, Y)
    Fy = cv_res['estimator']
    (_) = decodingCurveSupervised(Fy)
    
    # Slice data
    from utils import sliceData, sliceY
    stimTimes = st[st < X.shape[1]-tau+1] # limit valid stimTimes
    Xe = sliceData(X, stimTimes, tau=tau) # d x tau x ep x trl
    Ye = sliceY(Y, stimTimes, featdim=False) # y x ep  x trl
    print('cca = {}'.format(cca))
    Fy = cca.predict(X, Y)
    print("Fy={}".format(Fy.shape))
    Py = cca.predict_proba(X, Y)
    print("Py={}".format(Py.shape))
    score = cca.score(X, Y)
    print("score={}".format(score))
    decodingCurveSupervised(Fy)
    
if __name__ ==  "__main__":
    testcase()
