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
import scipy as sp
import copy
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, zero_outliers, crossautocov, updateCyy, updateCxy, autocov, updateCxx, plot_erp, plot_summary_statistics, plot_factoredmodel, compCxx_diag, compCyx_diag, compCyy_diag, Cxx_diag2full, Cyx_diag2full, Cyy_diag2full
from mindaffectBCI.decoder.multipleCCA import multipleCCA
from mindaffectBCI.decoder.scoreOutput import scoreOutput, dedupY0, corr_cov
from mindaffectBCI.decoder.utils import block_permute, window_axis
from mindaffectBCI.decoder.scoreStimulus import scoreStimulus, scoreStimulusCont, factored2full
from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised, score_decoding_curve, plot_decoding_curve
from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
from mindaffectBCI.decoder.stim2event import stim2event
from mindaffectBCI.decoder.normalizeOutputScores import normalizeOutputScores, estimate_Fy_noise_variance
from mindaffectBCI.decoder.zscore2Ptgt_softmax import softmax, calibrate_softmaxscale, marginalize_scores
from mindaffectBCI.decoder.temporal_basis import get_temporal_basis, apply_temporal_basis, invert_temporal_basis_mapping, apply_temporal_basis_X_TStd, invert_temporal_basis_mapping_spatiotemporal, apply_temporal_basis_tdtd, apply_temporal_basis_tde

try:
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.svm import LinearSVR, LinearSVC
    from sklearn.model_selection import StratifiedKFold, ShuffleSplit
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.exceptions import NotFittedError
except:
    #  if True:
    # placeholder classes for when sklearn isn't available
    class StratifiedKFold:
        def __init__(self, n_splits):
            self.n_splits = n_splits

        def split(self, X, y=None):
            shape = X.shape
            self.splits = np.linspace(0, shape[0], self.n_splits+1, dtype=int, endpoint=True).tolist()
            self._index = 0
            return self

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self.splits)-1:
                train_idx = list(range(self.splits[self._index])
                                 )+list(range(self.splits[self._index+1], self.splits[-1]))
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


def init_clsfr(
        model: str = "cca", tau_ms: int = None, offset_ms: int = 0, fs: float = 100, evtlabs: list = None, rank: int = 1,
        center: bool = True, C: float = 1.0, max_iter: int = 500, **kwargs):
    """create an instance of a BaseSequence2Sequence classifier from a given model name and parameters

    Args:
        model (str): the type of classifier to create. One-of; cca,bwd,fwd,ridge,lr,svr,svc,linearsklearn
        tau (int, optional): the length in samples of the stimulus response. Defaults to None.
        offset (int, optional): offset from the event time for the response window. Defaults to None.
        evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
        tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
        offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
        fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
        rank (int, optional): The rank (number of components) of the model to fit. Defaults to 1.
        center (bool, optional): center the data before fitting the model. Defaults to True.
        C (float, optional): regularization strength for the model.  Note: adjusted w.r.t. the model so higher-value means more regularization. Defaults to 1.0.

    Raises:
        NotImplementedError: [description]

    Returns:
        BaseSequence2Sequence: the fitted sequence 2 sequence model
    """
    if isinstance(model, BaseSequence2Sequence):
        clsfr = model
        
    elif model.lower() == 'cca' or model is None:
        clsfr = MultiCCA(tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, rank=rank, evtlabs=evtlabs, center=center, **kwargs)

    elif model.lower() == 'bwd':
        clsfr = BwdLinearRegression(tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, evtlabs=evtlabs, center=center, **kwargs)

    elif model.lower() == 'fwd':
        clsfr = FwdLinearRegression(tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, evtlabs=evtlabs, center=center, **kwargs)

    elif model.lower() == 'ridge':  # should be equivalent to BwdLinearRegression
        clsfr = LinearSklearn(tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, evtlabs=evtlabs,
                              clsfr=Ridge(alpha=C, fit_intercept=True, max_iter=max_iter), **kwargs)

    elif model.lower() in ('lr', 'linearlogisticregression'):
        clsfr = LinearLogisticRegression(tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, evtlabs=evtlabs, clsfr=LogisticRegression(
            C=C, fit_intercept=True, max_iter=max_iter, solver='lbfgs'), labelizeY=True, **kwargs)

    elif model.lower() in ('lrcv', 'linearlogisticregressioncv'):
        clsfr = LinearLogisticRegressionCV(tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, evtlabs=evtlabs, clsfr=LogisticRegression(
            C=C, fit_intercept=True, max_iter=max_iter, solver='lbfgs'), labelizeY=True, **kwargs)

    elif model.lower() == 'svr':
        clsfr = LinearSklearn(tau_ms=tau_ms, offset=offset_ms, fs=fs, evtlabs=evtlabs,
                              clsfr=LinearSVR(C=C, max_iter=max_iter), **kwargs)

    elif model.lower() == 'svc':
        clsfr = LinearSklearn(tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, evtlabs=evtlabs,
                              clsfr=LinearSVC(C=C, max_iter=max_iter), labelizeY=True, **kwargs)

    elif isinstance(model, BaseEstimator):
        clsfr = LinearSklearn(tau_ms=tau_ms, offset_ms=offset_ms, fs=fs,
                              evtlabs=evtlabs, clsfr=model, labelizeY=True, **kwargs)
    elif model == 'linearsklearn':
        clsfr = LinearSklearn(tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, evtlabs=evtlabs, **kwargs)
    elif isinstance(model, str):
        # treat the string as the fully qualified class name and try to load it!
        from mindaffectBCI.decoder.utils import import_and_make_class
        clsfr = import_and_make_class(model, tau_ms=tau_ms, offset_ms=offset_ms, fs=fs,
                                      rank=rank, evtlabs=evtlabs, center=center, **kwargs)
    else:
        raise NotImplementedError("don't  know this model: {}".format(model))

    # if pipeline is not None:
    #     pipeline = init_pipeline(pipeline)
    #     clsfr = Pipeline( pipeline +  ((model, clsfr),))

    return clsfr


# ----------------------------------------------------------
#
#  BASESEQUENCE2SEQUENCE
#
# -----------------------------------------------------------
class BaseSequence2Sequence(BaseEstimator, ClassifierMixin):
    '''Base class for sequence-to-sequence learning.  Provides, prediction and scoring functions, but not the fitting method'''

    def __init__(self, evtlabs=None, tau_ms: int = None, tau: int = None, offset_ms: int = None, offset: int = None,
                 fs: float = None, outputscore: str = 'ip', priorweight: float = 0, startup_correction: int = 50,
                 prediction_offsets: list = None, centerY: bool = False, minDecisLen: int = 0, bwdAccumulate: bool = False,
                 nvirt_out: int = -20, nocontrol_condn: float = .5, verb: int = 0, score_type:str='audc'):
        """Base class for general sequence to sequence models and inference

            N.B. this implementation assumes linear coefficients in W_ (nM,nfilt,d) and R_ (nM,nfilt,nE,tau)

        Args:
          evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
          tau (int, optional): the length in samples of the stimulus response. Defaults to None.
          offset (int, optional): offset from the event time for the response window. Defaults to None.
          tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
          offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
          fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
          priorweight (float, Optional): the weighting in pseudo-samples for the prior estimate for the prediction noise variance.  Defaults to 120.
          startup_correction (int, Optional): length in samples of addition startup correction where the noise-variance is artificially increased due to insufficient data.  Defaults to 100.
          prediction_offsets (list-of-int, optional): set of offsets to test when performing prediction to check/adapt to stimulus to response latency changes.  Defaults to None.
          centerY (bool, optional): center the stimulus-sequence information to be zero-mean per trial before fitting the response model.  Defaults to False.
          nvirt_out (int, optional): number of virtual outputs to add when testing.  If <0 then this is a limit on the totoal nubmer of outputs if >0 then add this number additional virtual outputs.  Defaults to -20.
          nocontrol_condn (float, optional): relative importance of correct prediction of nocontrol when no stimulus is present.  Defaults to .5
          verb (int, optional): verbosity level, 0 is default, higher number means be more talky in debug messages.  Defaults to 0.
        """
        self.evtlabs = evtlabs
        self.fs, self.tau_ms, self.tau, self.offset_ms, self.offset, self.priorweight, self.startup_correction, self.prediction_offsets = (
            fs, tau_ms, tau, offset_ms, offset, priorweight, startup_correction, prediction_offsets)
        self.verb, self.minDecisLen, self.bwdAccumulate, self.nvirt_out, self.nocontrol_condn, self.centerY, self.score_type = (
            verb, minDecisLen, bwdAccumulate, nvirt_out, nocontrol_condn, centerY, score_type)
        self.outputscore = outputscore

    def stim2event(self, Y_TSy, prevY=None, fit=False):
        '''transform Stimulus-encoded to brain-encoded, if needed'''
        # convert from stimulus coding to brain response coding
        if self.evtlabs is not None:
            if fit or not hasattr(self, 's2estate_') or self.s2estate_ is None:  # fit the event mapping
                Y_TSye, self.s2estate_, self.evtlabs_ = stim2event(
                    Y_TSy, evtypes=self.evtlabs, axis=-2, oM=prevY)  # (tr, samp, Y, e)
            else:  # use fitted event mapping
                Y_TSye, _, _ = stim2event(Y_TSy, evtypes=self.s2estate_, axis=-2, oM=prevY)  # (tr, samp, Y, e)
        else:
            if hasattr(Y_TSy, 'info'):
                self.evtlabs_ = Y_TSy.info.get('evtlabs', None)
            else:
                self.evtlabs_ = None
            Y_TSye = Y_TSy[..., np.newaxis] if Y_TSy.ndim == 3 else Y_TSy  # (tr,samp,Y,e)

        if self.centerY and Y_TSye.shape[2] > 1:
            Y_TSye = Y_TSye - np.mean(Y_TSye, axis=2, keepdims=True)

        if Y_TSye.ndim > 4:  # collapse additional event type dims into a single big one
            Y_TSye = Y_TSye.reshape(Y_TSye.shape[:3]+(-1,))

        return Y_TSye

    def fit_tau_offset(self, X, y, fs=None):
        """fit the window size and offset to the input data sample rate

        Args:
            X ([type]): [description]
            y ([type]): [description]
            fs ([type], optional): fall-back sample rate if not found in X's metainfo. Defaults to None.
        """
        if fs is not None:
            self.fs_ = fs
        elif hasattr(X, 'info'):  # meta-data ndarray
            self.fs_ = X.info.get('fs',None)
            self.ch_names_ = X.info.get('ch_names',None)
        elif self.fs is not None:
            self.fs_ = self.fs
        elif not hasattr(self, 'fs_'):
            self.fs_ = None
        
        if not hasattr(self, 'ch_names_'):
            self.ch_names_ = None

        self.tau_ = self.tau
        if self.tau_ is None and self.tau_ms is not None:
            self.tau_ = max(1, int(self.tau_ms * self.fs_ / 1000))
        if self.tau_ is None:
            self.tau_ = X.shape[1]
        #print("tau_ms={} tau={}".format(self.tau_ms, self.tau))
        if self.offset is not None:
            self.offset_ = self.offset
        elif not hasattr(self, 'offset_'):
            self.offset_ = None
        if self.offset_ is None:
            if self.offset_ms is None or self.offset_ms == 0:
                self.offset_ = 0
            else:
                self.offset_ = int(self.offset_ms * self.fs_ / 1000)

    def fit(self, X, y):
        '''fit model mapping 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, e)'''
        raise NotImplementedError

    def is_fitted(self):
        """test if this model has been fitted to data

        Returns:
            _type_: _description_
        """
        return hasattr(self, "W_")

    def clear(self):
        """clear the fitted model
        """
        if hasattr(self, "W_"):
            delattr(self, 'W_')
        if hasattr(self, "R_"):
            delattr(self, 'R_')
        if hasattr(self, "A_"):
            delattr(self, 'A_')
        if hasattr(self, "b_"):
            delattr(self, 'b_')

    def predict(self, X, y, dedup0=False, prevY=None, offsets=None):
        """Generate predictions with the fitted model for the paired data + stimulus-sequences

            N.B. this implementation assumes linear coefficients in W_ (nM,nfilt,d) and R_ (nM,nfilt,nE,tau)

        Args:
            X (np.ndarray (tr,samp,d)): the multi-trial data
            Y (np.ndarray (tr,samp,nY)): the multi-trial stimulus sequences
            dedup0 ([type], optional): remove duplicates of the Yidx==0, i.e. 1st, assumed true, output of Y. >0 remove the copy (used when calibrating), <0 remove objID==0 (used when predicting) Defaults to False.
            prevY ([type], optional): previous stimulus sequence information. for partial incremental calls. Defaults to None.
            offsets ([ListInt], optional): list of offsets in Y to try when decoding, to override the class variable.  Defaults to None.

        Raises:
            NotFittedError: raised if try to predict without first fitting

        Returns:
            Fy (np.ndarray (mdl,tr,samp,nY): score for each output in each trial.  Higher score means more 'likely' to be the 'true' target
        """
        if not self.is_fitted():
            # only if we've been fitted!
            raise NotFittedError

        X_TSd = X
        Y_TSy = y
        if X_TSd.ndim < 3:  # ensure trial dim if not there
            X_TSd = X_TSd.reshape((1,)*(3-X.ndim) + X.shape)  # add trial dim
            Y_TSy = Y_TSy.reshape((1,)*(3-X.ndim) + y.shape)

        # convert from stimulus coding to brain response coding
        Y_TSye = self.stim2event(Y_TSy, prevY)

        # apply the model to transform from raw data to stimulus scores
        # N.B. offset is already applied here?
        Fe_MTSe = self.transform(X_TSd)  # (nM, nTrl, nSamp, nE)

        # combine the stimulus information and the stimulus scores to
        # get output scores.  Optionally, include time-shifts in output.
        if offsets is None and self.prediction_offsets is not None:
            offsets = self.prediction_offsets
        Fy_MTSy = scoreOutput(Fe_MTSe, Y_TSye, dedup0=dedup0, R_mket=self.R_, offset=offsets,
                              outputscore=self.outputscore)  # (nM, nTrl, nSamp, nY)

        # BODGE: strip un-needed model dimension
        if Fy_MTSy.ndim > 3 and Fy_MTSy.shape[0] == 1:
            Fy_MTSy = Fy_MTSy[0, ...]
        return Fy_MTSy

    def transform(self, X, y=None):
        """ transform raw data into raw stimulus scores by convolving X with the model, i.e. Fe = X (*) (W*R)

        Args:
            X (np.ndarray (nTrl,nSamp,d): The raw eeg data.

        Returns:
            F_TSe (np.ndarray (nTrl,nSamp,nE)): The raw stimulus scores.
        """
        X_TSd = X
        W_Mkd = self.W_
        if X.ndim > 3:
            X_TSd = X_TSd.reshape(X_TSd.shape[:2]+(-1,))  # squash extra feature dims
            if W_Mkd.shape[-1] < X_TSd.shape[-1]: 
                W_Mkd = W_Mkd.reshape(W_Mkd.shape[:2]+(-1,))
        if X_TSd.ndim < 3:
            X_TSd = X_TSd.reshape((1,)*(3-X_TSd.ndim) + X.shape)  # add trial dim

        F_TSe = scoreStimulus(X_TSd, W_Mkd, self.R_, self.b_, offset=self.offset_, isepoched=False)
        if F_TSe.ndim > 3:  # strip confusing model dimension
            F_TSe = F_TSe[0, ...]
        return F_TSe

    def modify(self, X, Y):
        """compute the forward and backward latent space model activity

        Args:
            X (_type_): X data, i.e. EEG
            Y (_type_): Y data, i.e. raw stimulus time-series

        Returns:
            _type_: wX = g_x, g_y = Yr, the latent space activations in the forward and backward directions 
        """
        X_TSd = X
        if X_TSd.ndim > 3:
            X_TSd = X_TSd.reshape(X_TSd.shape[:2]+(-1,))  # squash extra feature dims
        elif X_TSd.ndim < 3:
            X_TSd = X_TSd.reshape((1,)*(3-X_TSd.ndim) + X_TSd.shape)  # add trial dim
        wX = np.einsum("kd,TSd->TSk", self.W_[0, ...], X_TSd)

        # convert from stimulus coding to brain response coding
        Y_TSy = Y
        if Y_TSy.ndim > 4:
            Y_TSy = Y_TSy.reshape(Y_TSy.shape[:3]+(-1,))  # squash extra feature dims
        elif Y_TSy.ndim < 3:
            Y_TSy = Y_TSy.reshape((1,)*(3-Y_TSy.ndim) + Y_TSy.shape)  # add trial dim
        Y_TSye = self.stim2event(Y_TSy)

        # TODO[]: add the offset!
        # convolve
        Y_TStye = window_axis(Y_TSye, winsz=self.R_.shape[-1], axis=1)
        Yr = np.einsum("TStye,ket->TSyk", Y_TStye, self.R_[0, ...])

        # strip X to be same shape...
        wX = wX[:, :Yr.shape[1], ...]
        return wX, Yr

    def decode_proba(self, Fy, minDecisLen=0, bwdAccumulate=None, marginalizemodels=True, marginalizedecis=False, dedup0=None):
        """Convert stimulus scores to stimulus probabities of being the target

        Args:
            Fy (np.ndarray (tr,samp,nY)): the multi-trial stimulus sequence scores, from predict
            minDecisLen (int,optional): minimum number of samples on which to make a prediction. Defaults to 0.
            bwdAccumulate (bool, optional): accumulate information backwards in the trials.  Defaults to True.
            marginalizemodels (bool,optional): flag if we should marginalize over models when have multiple prediction models.  Defaults to True.
            marginalizedecis (bool, optional): flag if we should marginalize over decision points when have multiple. Defaults to False.

        Raises:
            NotFittedError: [description]

        Returns:
            Ptgt (np.ndarray (tr,nDecis,nY)): score for each decision point in each trial for each output.  Higher score means more 'likely' to be the 'true' target
        """
        # build optional arguments if set
        kwargs = dict()
        if hasattr(self, 'sigma0_') and self.sigma0_ is not None:
            # include the score prior
            # print('priorsigma=({},{})'.format(self.sigma0_,self.priorweight))
            kwargs['priorsigma'] = (self.sigma0_, self.priorweight)
        if hasattr(self, 'softmaxscale_') and self.softmaxscale_ is not None:
            kwargs['softmaxscale'] = self.softmaxscale_
        if bwdAccumulate is None and hasattr(self, 'bwdAccumulate'):
            bwdAccumulate = self.bwdAccumulate

        if dedup0 is not None and dedup0 is not False:  # remove duplicate copies output=0
            Fy = dedupY0(Fy, zerodup=dedup0 > 0, yfeatdim=False)

        Yest, Perr, Ptgt, _, _ = decodingSupervised(
            Fy, minDecisLen=minDecisLen, bwdAccumulate=bwdAccumulate, marginalizemodels=marginalizemodels,
            marginalizedecis=marginalizedecis, nEpochCorrection=self.startup_correction, nvirt_out=self.nvirt_out, **
            kwargs)
        if marginalizemodels and Ptgt.ndim > 3 and Ptgt.shape[-4] > 0:  # hide our internal model dimension?
            Yest = Yest[0, ...]
            Perr = Perr[0, ...]
            Ptgt = Ptgt[0, ...]
        return Ptgt  # (nM, nTrl, nEp, nY)

    def predict_proba(self, X, Y, marginalizemodels=True, marginalizedecis=True, startup_correction=100,
                      minDecisLen=None, bwdAccumulate=True, dedup0=True, prevY=None):
        """Predict the probability of each output for paired data/stimulus sequences

        Args:
            X (np.ndarray (tr,samp,d)): the multi-trial data
            Y (np.ndarray (tr,samp,nY)): the multi-trial stimulus sequences
            dedup0 (bool, optional): remove duplicates of the Yidx==0, i.e. 1st, assumed true, output of Y. Defaults to True.
            prevY (np.ndarray, optional): previous stimulus sequence information. for partial incremental calls. Defaults to None.
            minDecisLen (int,optional): minimum number of samples on which to make a prediction
            marginalizemodels (bool,optional): flag if we should marginalize over models when have multiple prediction models.  Defaults to True.
            marginalizedecis (bool, optional): flag if we should marginalize over decision points when have multiple. Defaults to False.

        Raises:
            NotFittedError: [description]

        Returns:
            Ptgt (np.ndarray (tr,nDecis,nY): Probability of each output being the target.  Higher score means more 'likely' to be the 'true' target
        """
        Fy = self.predict(X, Y, dedup0=dedup0, prevY=prevY)
        if minDecisLen is None:
            minDecisLen = self.minDecisLen
        if bwdAccumulate is None:
            bwdAccumulate = self.bwdAccumulate
        return self.decode_proba(
            Fy, marginalizemodels=marginalizemodels, marginalizedecis=marginalizedecis, minDecisLen=minDecisLen,
            bwdAccumulate=bwdAccumulate, dedup0=False)

    def add_virt_out(self, Y, nvirt_out):
        """add a virtual output to the input stimulus info

        Args:
            Y (_type_): _description_
            nvirt_out (_type_): the number of virtual outputs to add.  These are generated as random perumations of Y.

        Returns:
            _type_: _description_
        """
        Ytst = Y
        if nvirt_out is not None:
            # generate virtual outputs for testing -- so get valid results even if only 1 sequence
            if Y.ndim == 3:  # no feature dim
                virt_Y = block_permute(Y[..., 1:], nvirt_out, axis=-1, perm_axis=-2)
                Ytst = np.concatenate((Y, virt_Y), axis=-1)
            elif Y.ndim == 4:  # feature dim
                virt_Y = block_permute(Y[..., 1:, :], nvirt_out, axis=-2, perm_axis=-3)
                Ytst = np.concatenate((Y, virt_Y), axis=-2)
        return Ytst

    def score(self,X,Y,score_type:str=None, **kwargs):
        if score_type is None: score_type=self.score_type
        if score_type=='corr':
            score = self.score_corr(X,Y,**kwargs)
        elif score_type == 'classifier':
            score = self.score_classifier(X,Y,**kwargs)
        elif callable(score_type):
            score = self.score_type(self,X,Y,**kwargs)
        elif score_type == 'audc' or score_type is None:
            score = self.score_audc(X,Y,**kwargs)
        else:
            raise ValueError("Unrecog score type: {}".format(self.score_type))
        return score

    def score_audc(self, X, Y, Fy=None, nvirt_out=None):
        '''score this model on this data, N.B. for model_selection higher is *better*'''
        # score, removing duplicates for true target (objId==0) so can compute score
        if np.any(Y[:, :, 0, ...] != 0):  # we have true-target info to fit
            if Fy is None:  # compute the Fy from X,Y
                Ytst = self.add_virt_out(Y, nvirt_out if nvirt_out is None else self.nvirt_out)
                Fy = self.predict(X, Ytst)
            score = BaseSequence2Sequence.inner_score_audc(Fy)
        else:  # no true target info, so try to score w.r.t. the summed activity over all outputs
            score = self.score_corr(X, Y)
        return score

    def decoding_curve(self, X_TSd, Y_TSy, cv=None, **kwargs):
        """compute the decoding curve information for this X,Y apr

        Args:
            X_TSd ([type]): [description]
            Y_TSy ([type]): [description]

        Returns:
            tuple: the information about the decoding curve as returned from `decodingCurveSupervised`
        """
        if cv is not None:
            res = self.cv_fit(X_TSd, Y_TSy, cv=cv, retrain_on_all=False)
            # TODO[]: make cv_predict..
            Fy_TSy = res['rawestimator'] if 'rawestimator' in res else res['estimator']
        else:
            Fy_TSy = self.predict(X_TSd, Y_TSy, dedup0=False)
        return decodingCurveSupervised(
            Fy_TSy, marginalizedecis=True, minDecisLen=self.minDecisLen, bwdAccumulate=self.bwdAccumulate,
            priorsigma=(self.sigma0_, self.priorweight),
            softmaxscale=self.softmaxscale_, nEpochCorrection=self.startup_correction, **kwargs)

    def plot_decoding_curve(self, X_TSd, Y_TSy, cv=None, fs: float = None, **kwargs):
        (dc) = self.decoding_curve(X_TSd, Y_TSy, cv=cv, **kwargs)
        plot_decoding_curve(*dc, fs=fs)

    def transform_cv(self, X_TSd, Y_TSy, cv=None):
        """cross-validated transformation of X, i.e. computation of Fe_TSe which is the predicted stimulus sequence

        Args:
            X_TSd (_type_): _description_
            Y_TSy (_type_): _description_
            cv (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if cv is not None:
            # cv-fit and predict in each fold
            cv = self.get_folding(X_TSd, Y_TSy, cv)
            for fi, (train_idx, val_idx) in enumerate(cv):
                # fit the model
                self.fit(X_TSd[train_idx, ...], Y_TSy[train_idx, ...])
                # get predicted stimulus on validation set
                Fei = self.transform(X_TSd[val_idx, ...], Y_TSy[val_idx, ...])
                # strip extra model dim
                if Fei.ndim > 3:
                    if Fei.shape[0] > 1:
                        print("Warning: extra models ignored!")
                    Fei = Fei[0, ...]
                if fi == 0:
                    Fe_TSe = np.zeros((X_TSd.shape[0],)+Fei.shape[1:], dtype=Fei.dtype)
                Fe_TSe[val_idx, ...] = Fei

        else:
            # apply the model to transform from raw data to stimulus scores
            Fe_TSe = self.transform(X_TSd)  # (nM, nTrl, nSamp, nE)
            if Fe_TSe.ndim > 3:
                if Fe_TSe.shape[0] > 1:
                    print("Warning, extra models ignored!")
                Fe_TSe = Fe_TSe[0, ...]
        return Fe_TSe

    def sequence2classification(self, Fe_TSe, Y_TSy, prevY=None, score_noise: float = 1e-8):
        """convert from sequence problem (with stimulus parameters) to classification problem, with sample labels

        Args:
            Fe_TSe ([type]): [description]
            Y_TSy ([type]): [description]
            prevY ([type], optional): [description]. Defaults to None.
            score_noise (float, optional): [description]. Defaults to 1e-8.

        Returns:
            [type]: [description]
        """
        # convert from stimulus coding to brain response coding
        Y_TSye = self.stim2event(Y_TSy, prevY)
        Ytrue_TSe = Y_TSye[..., 0, :]

        # include score noise to force tie-breaking
        Fe_TSe = Fe_TSe + np.random.standard_normal(Fe_TSe.shape)*score_noise

        # convert indicator to label
        Ytrue_TS = np.argmax(Ytrue_TSe, axis=-1) + 1  # class 1,2,3...
        Ytrue_TS[np.all(Ytrue_TSe == 0, -1)] = 0  # unlabelled has value 0
        Fe_TS = np.argmax(Fe_TSe, axis=-1) + 1

        return Ytrue_TS, Fe_TS

    def classifier_score(self, Ytrue_TS, Fe_TS, ignore_unlabelled: bool = True):
        """compute a goodness of fit score as a classification problem, i.e. prediction of the unique stimulus event info.

        Args:
            Ytrue_TS (_type_): _description_
            Fe_TS (_type_): _description_
            ignore_unlabelled (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # limit to only samples with any activity
        if ignore_unlabelled:
            keep = Ytrue_TS != 0
            Fe_TS = Fe_TS[keep, ...]  # now: (N,e)
            Ytrue_TS = Ytrue_TS[keep, ...]  # now: (N,e)
        cr = np.sum(Fe_TS == Ytrue_TS) / Ytrue_TS.size
        return cr

    def score_classifier(self, X_TSd, Y_TSy, cv=None, ignore_unlabelled: bool = True):
        """apply the model to the data, convert to a unique-event code classificaiton problem and then compute a classification score.

        Args:
            X_TSd (_type_): _description_
            Y_TSy (_type_): _description_
            cv (_type_, optional): _description_. Defaults to None.
            ignore_unlabelled (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        Fe_TSe = self.transform_cv(X_TSd, Y_TSy, cv=cv)
        Ytrue_TS, Fe_TS = self.sequence2classification(Fe_TSe, Y_TSy)
        cr = self.classifier_score(Ytrue_TS, Fe_TS, ignore_unlabelled=ignore_unlabelled)
        return cr

    def confusion_matrix(
            self, X_TSd, Y_TSy, cv=None, prevY=None, ignore_unlabelled=True, normalize='true', score_noise=1e-5):
        """ generate a stimulus event type confusion matrix
        Args:
            X (np.ndarray (tr,samp,d)): the multi-trial data
            Y (np.ndarray (tr,samp,nY)): the multi-trial stimulus sequences, Y[:,:,0] assumed to be the 'true' label sequence
            prevY ([type], optional): previous stimulus sequence information. for partial incremental calls. Defaults to None.
        Raises:
            NotFittedError: raised if try to predict without first fitting
        Returns:
            confmx_ee: event class confusion matrix
        """
        if not self.is_fitted():
            # only if we've been fitted!
            raise NotFittedError

        Fe_TSe = self.transform_cv(X_TSd, Y_TSy, cv=cv)
        Ytrue_TS, Fe_TS = self.sequence2classification(Fe_TSe, Y_TSy, score_noise=score_noise)
        if ignore_unlabelled:
            keep = Ytrue_TS != 0
            Fe_TS = Fe_TS[keep, ...]  # now: (N,e)
            Ytrue_TS = Ytrue_TS[keep, ...]  # now: (N,e)

        # compute confusion matrix
        import sklearn.metrics
        cm = sklearn.metrics.confusion_matrix(Ytrue_TS.ravel(), Fe_TS.ravel(), normalize=normalize)
        return cm

    def plot_confusion_matrix(self, X, Y, **kwargs):
        cm = self.confusion_matrix(X, Y, **kwargs)
        import sklearn.metrics
        import matplotlib.pyplot as plt
        if self.evtlabs_ is not None and len(self.evtlabs_) > 1:
            disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.evtlabs_)
        else:
            disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(cm.shape[0]))
        disp.plot(ax=plt.gca())

    @staticmethod
    def inner_score_audc(Fy_TSy, score_noise=1e-5):
        '''compute area under decoding curve score from Fy, *assuming* Fy[:,:,0] is the *true* classes score'''
        if Fy_TSy.ndim > 3:
            raise ValueError("Only for single models!")
        Fy_TSy = Fy_TSy.reshape((1,)*(3-Fy_TSy.ndim) + Fy_TSy.shape)
        Fy_TSy = dedupY0(Fy_TSy, zerodup=True, yfeatdim=False)  # remove duplicates of 'true' info
        sFy_TSy = np.cumsum(Fy_TSy, axis=-2)  # (nM, nTrl, nSamp, nY)
        hasTrueTgt = np.any(sFy_TSy[:, :, 0] != 0, -1)  # has true-target info
        hasStim = np.any(sFy_TSy != 0, axis=(-2, -1))  # (nTrl) flag if this trial is valid, i.e. some non-zero
        validTrl = np.logical_and(hasTrueTgt, hasStim)
        # add some label noise....
        sFy_TSy = sFy_TSy + np.random.standard_normal(sFy_TSy.shape)*score_noise
        Yi = np.argmax(sFy_TSy[validTrl, :, :], axis=-1)  # output for every model*trial*sample
        score = np.sum(Yi == 0)/Yi.size  # total amount time was right, higher=better
        return score

    # TODO[]: correct and ensure the goodness of fit score is actually working
    def score_corr(self, X_TSd, Y_TSy, featdim=False):
        """compute a goodness-of-fit (gof) score for this fited model, as the correlation between the latent-space time-series.

        Args:
            X_TSd (_type_): _description_
            Y_TSy (_type_): _description_
            featdim (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """        ''' compute goodness of fit score for the current model for X,Y'''
        from mindaffectBCI.decoder.scoreOutput import convWX, convYR
        Y_TSye = self.stim2event(Y_TSy)  # (nTr,nSamp,1,nE)
        if np.any(Y_TSye[..., 0, :]):  # use true target info
            Y_TSye = Y_TSye[..., 0, :]
        else:  # use average target info
            Y_TSye = np.sum(Y_TSye, 2, keepdims=True)
        WX_TSk = convWX(X_TSd, self.W_)[0, ...]  # (nM,nTr,nSamp,nFilt)
        nWX_TSk = WX_TSk / np.sqrt(np.sum(WX_TSk**2, axis=(-2, -1), keepdims=True))
        YR_TS1k = convYR(Y_TSye, self.R_, offset=self.offset)[0, ...]  # (nM,nTr,nSamp,nY,nFilt)
        # BODGE: unit norm Yr over time/filters -- shouldn't be necessary!
        nYR_TS1k = YR_TS1k / np.sqrt(np.abs(np.sum(YR_TS1k**2, axis=(-3, -1), keepdims=True))+1e-3)
        corr = np.sum(nWX_TSk * nYR_TS1k[:, :, 0, :]) / X_TSd.shape[0] / X_TSd.shape[1]
        return corr

    def get_folding(self, X, Y, cv):
        """get the folding of the data given the cross-validation specification

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            cv (_type_): _description_

        Returns:
            _type_: _description_
        """
        if cv == True:
            cv = 5
        if cv is None or X.shape[0] == 1 or cv == 0 or cv == -1:  # use all the data
            cv = [(slice(None), slice(None))]
        elif isinstance(cv, int):  # max folds equal number trials
            cv = StratifiedKFold(n_splits=min(cv, X.shape[0]))
        if hasattr(cv, 'split'):
            if cv.get_n_splits() <= X.shape[0]:
                cv = cv.split(np.zeros(X.shape[0]), y=np.zeros(Y.shape[0]))
            else:  # fall back on 50/50 shuffle-split
                cv = ShuffleSplit(n_splits=X.shape[0], train_size=.5).split(np.zeros(X.shape[0]))
        return cv

    def cv_fit(
            self, X, y, cv: int = 5, fs=None, fit_params: dict = dict(),
            verbose: bool = 0, return_estimator: bool = True, calibrate_softmax: bool = True, retrain_on_all: bool = True,
            score_type=None):
        """Cross-validated fit the model for generalization performance estimation

        N.B. write our own as sklearn doesn't work for getting the estimator values for structured output.

        Args:
            X ([type]): [description]
            y ([type]): [description]
            cv (int, optional): the number of folds to use, or a fold generator. Defaults to 5.
            fit_params (dict, optional): additional parameters to pass to the self.fit(X,Y,...) function. Defaults to dict().
            verbose (bool, optional): flag for model fitting verbosity. Defaults to 0.
            return_estimator (bool, optional): should we return the cross-validated predictions. Defaults to True.
            calibrate_softmax (bool, optional): after fitting the model, should we use the cross-validated predictions to calibrate the probability estimates. Defaults to True.
            dedup0 (bool, optional): should we de-duplicate copies of the 'true-target'. Defaults to True.
            retrain_on_all (bool, optional): should we retrain the model on all the data after the cv folds. Defaults to True.

        Returns:
            results (dict): dictionary with the results
        """

        # TODO [] : conform to sklearn cross_validate signature
        # TODO [] : move into a wrapper class
        Y_TSy = y  # BODGE []: rename to conform to sklearn naming...
        X_TSd = X
        if score_type is None:
            score_type = self.score_type

        cv = self.get_folding(X_TSd, Y_TSy, cv)
        scores = []  # decoding scores
        Fy_TSy = None
        if verbose > 0:
            print("CV:", end='')
        for i, (train_idx, valid_idx) in enumerate(cv):
            if verbose > 0:
                print(".", end='', flush=True)
            #print("trn={} val={}".format(train_idx,valid_idx))

            self.fit(X_TSd[train_idx, ...], Y_TSy[train_idx, ...], fs=fs, **fit_params)

            # predict, forcing removal of copies of  tgt=0 so can score
            if X_TSd[valid_idx, ...].size == 0:
                print("Warning: no-validation trials!!! using all data!")
                valid_idx = slice(X_TSd.shape[0])

            Fyi_TSy = self.predict(X_TSd[valid_idx, ...], Y_TSy[valid_idx, ...], dedup0=False)

            if i == 0:  # reshape Fy to include the extra model dim
                Fy_TSy = np.zeros((Y_TSy.shape[0],)+Fyi_TSy.shape[1:], dtype=X.dtype)
            # BODGE: guard for more outputs in later blocks
            Fy_TSy[valid_idx, ..., :Fyi_TSy.shape[-1]] = Fyi_TSy[..., :Fy_TSy.shape[-1]]

            val = None
            #try:
            if 1:
                if score_type is None:  # just call our score function
                    val = self.score(X_TSd[valid_idx, ...], Y_TSy[valid_idx, ...], Fy=Fyi_TSy)
                elif score_type == 'audc':
                    val = self.inner_score_audc(Fyi_TSy)
                elif score_type == 'gof':  # goodness of fit
                    val = self.score_corr(X_TSd[valid_idx, ...], Y_TSy[valid_idx, ...])
                elif callable(score_type):  # given score function
                    val = score_type(X_TSd[valid_idx, ...], Y_TSy[valid_idx, ...])
                elif isinstance(score_type, 'str'):
                    if hasattr(self, score_type):
                        val = getattr(self, score_type)(self, X_TSd[valid_idx, ...], Y_TSy[valid_idx, ...])
                    else:
                        # try to get a function in our name-space with this name
                        score_type = locals().get(score_type) or globals().get(score_type) or score_type
                        val = score_type(X_TSd[valid_idx, ...], Y_TSy[valid_idx, ...], self.W_, self.R_, self.b_)
            #except:
            #    pass
            scores.append(val)
        # final retrain with all the data
        if retrain_on_all:
            self.fit(X_TSd, Y_TSy, **fit_params)

        # TODO[]: move the calibration outside the cv_fit function
        self.sigma0_ = self.estimate_prior_sigma(Fy_TSy)
        self.softmaxscale_ = self.calibrate_softmaxscale(
            Fy_TSy) if calibrate_softmax and np.any(Y_TSy[:, :, 0, ...]) else 3
        #print("Sigma0={}  softmaxscale={}".format(self.sigma0_, self.softmaxscale_))

        Fy_raw = Fy_TSy
        if return_estimator and Fy_TSy.ndim > 3:
            # make into a Fy matrix
            # N.B. DO NOT marginalize directly on per-sample scores -- as they are v.v.v. noisy!!
            sFy = np.sum(Fy_TSy, -2, keepdims=True)
            p = softmax(sFy*self.softmaxscale_, axis=0)  # weight for each model
            print("model wght= {}".format(np.mean(p.reshape((p.shape[0], -1)), axis=1)))
            Fy_TSy = np.sum(Fy_TSy * p, axis=0)

        return dict(estimator=Fy_TSy, rawestimator=Fy_raw, scores_cv=scores, test_score=scores)

    def estimate_prior_sigma(self, FY_TSy):
        # estimate prior for Fy using the cv'd Fy
        #self.sigma0_ = np.sum(Fy.ravel()**2) / Fy.size
        # print('Fy={}'.format(Fy.shape))
        # N.B. need to match the filter used in the decoder..
        sigma0, _ = estimate_Fy_noise_variance(FY_TSy, priorsigma=None)
        #print('Sigma0{} = {}'.format(self.sigma0_.shape,self.sigma0_))
        sigma0 = np.nanmedian(sigma0.ravel())  # ave
        if self.verb > 0:
            print('Sigma0 = {}'.format(sigma0))
        return sigma0

    def priorsigma(self):
        return (getattr(self, 'sigma0_', None), self.priorweight)

    def softmaxscale(self):
        return getattr(self, 'softmaxscale_', 3)

    def gettau(self):
        """get the response length in samples from the fitted model

        Returns:
            _type_: _description_
        """
        if hasattr(self, 'R_'):
            return self.R_.shape[-1]
        elif hasattr(self, 'tau_'):
            return self.tau_
        return None

    def getfs(self):
        """get the assumed sample rate of the data we are fitting

        Returns:
            _type_: _description_
        """
        if hasattr(self, 'fs_'):
            return self.fs_
        else:
            return self.fs

    def calibrate_softmaxscale(self, Fy_TSy):
        """post-process prediction scores to better calibrate the proba values

        TODO[]: move this calibration step to a post-process or classifier wrapper..

        Args:
            Fy_TSy (np.ndarray): (nTrials, nSamples, nY) the prediction scores 
        """
        # calibrate the softmax scale to give more valid probabilities
        # add virtual outputs if wanted -- so it's a better test
        if self.nvirt_out is not None:
            # generate virtual outputs for testing -- not from the 'true' target though
            # N.B. can't add virtual outputs *after* score normalization as the normalization
            #      process breaks the IID assumption underlying the permutation approach
            virt_Fy = block_permute(Fy_TSy[..., 1:], self.nvirt_out, axis=-1, perm_axis=-2)
            nvirt_out = virt_Fy.shape[-1]
            if self.verb > 0:
                print("Added {} virtual outputs for calibration".format(nvirt_out))
            Fy_TSy = np.append(Fy_TSy, virt_Fy, axis=-1)
        nFy, _, _, _, _ = normalizeOutputScores(
            Fy_TSy, minDecisLen=-10, nEpochCorrection=self.startup_correction,
            priorsigma=(self.sigma0_, self.priorweight))
        softmaxscale = calibrate_softmaxscale(nFy, nocontrol_condn=self.nocontrol_condn, verb=self.verb-1)
        return softmaxscale

    def calibrate_proba(self, X, Y):
        """given some test data, calibrate the parameters needed to compute valid target probabilities, used for example for early stopping based on system estimated confidence.

        Args:
            X (_type_): _description_
            Y (_type_): _description_

        Returns:
            _type_: _description_
        """
        Fy = self.predict(X, Y)
        self.sigma0_ = self.estimate_prior_sigma(Fy)
        self.softmaxscale_ = self.calibrate_softmaxscale(Fy)
        return self

    def plot_model(self, fs: float = None, ncol: int = 2, plot_pattern: bool = True, ch_names: list = None, offset:float=None, offset_ms:float=None, **kwargs):
        """plot the fitted model, either as a factored spatial+temporal model, or combined spatiotemporal model

        Args:
            fs (float, optional): sample rate of the data.  If None then use the sample rate given at construction/fitting time. Defaults to None.
            ncol (int, optional): number of colums of the figure to make when ploting the figure.  The sptial + temporal will be in columns 0,1.  For example use ncols=3 to leave a 3rd col for additional model info plotted outside this function. Defaults to 2.
            plot_pattern (bool, optional): plot a spatial pattern or a spatial filter. Defaults to True.
        """
        evtlabs = self.evtlabs_ if hasattr(self, 'evtlabs_') else self.evtlabs
        if fs is None:        fs = self.fs_
        if ch_names is None:  ch_names = self.ch_names_
        if offset is None and offset_ms is None:  offset = self.offset_
        if hasattr(self, 'R_') and not self.R_ is None:
            #print("Plot Factored Model")
            sp = (self.A_, 'spatial-pattern') if plot_pattern and hasattr(self,
                                                                          'A_') and self.A_ is not None else (self.W_, 'spatial-filter')
            tp = (self.R_, 'temporal-pattern')
            plot_factoredmodel(
                sp[0].reshape((-1, sp[0].shape[-1])),
                tp[0].reshape((-1,) + tp[0].shape[-2:]),
                evtlabs=evtlabs, 
                offset=offset, offset_ms = offset_ms, fs=fs,
                ch_names=ch_names,  
                spatial_filter_type=sp[1],
                temporal_filter_type=tp[1],
                ncol=ncol, **kwargs)
        else:
            plot_erp(self.W_[np.newaxis,...], # add back in a epoch-dim
                    evtlabs=evtlabs, 
                    fs=fs, offset=offset, offset_ms = offset_ms, 
                    ch_names=ch_names, 
                    **kwargs)

        #import matplotlib.pyplot as plt
        # plt.figure()
        #plot_factoredmodel(self.W_, self.R_, evtlabs=evtlabs, offset=self.offset, fs=fs, spatial_filter_type='filt', temporal_filter_type='filt', ncol=ncol, **kwargs)


# ----------------------------------------------------------
#
#  M U L T I C C A
#
# -----------------------------------------------------------


class MultiCCA(BaseSequence2Sequence):
    ''' Sequence 2 Sequence learning using CCA as a bi-directional forward/backward learning method '''

    def __init__(self, rank=1, reg=(1e-8, 1e-8), rcond=(1e-4, 1e-8), badEpThresh=6, badWinThresh=3, symetric=True,
                 center=True, CCA=True, outputscore: str = 'ip', whiten_alg: str = 'eigh', cxxp=True, temporal_basis=None,
                 # explicit args so sklearn can autoset and clone
                 evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, normalize_sign=True, score_type:str=None,  **kwargs):
        """Sequence 2 Sequence learning using CCA as a bi-directional forward/backward learning method

        Args:
            rank (int, optional): The rank (number of components) of the model to fit. Defaults to 1.
            reg (tuple, optional): Regularization strength for (spatial,temporal) components when performing the CCA fit. See `multiCCA.py::robust_whiten` for more information. Defaults to (1e-8, 1e-8).
            rcond (tuple, optional): Recipol-condition-number (spatial,temporal) components when performing the CCA fit. See `multiCCA.py::robust_whiten` for more information.. Defaults to (1e-4, 1e-8).
            symetric (bool, optional): Use a symetric whitener in the CCA fitting.  See `multiCCA.py` for more information. Defaults to True.
            center (bool, optional): center the data before fitting the model. Defaults to True.
            CCA (bool, optional): do we run CCA or PLS or a hybrid.  If True is CCA, if False is PLS.  If 2-tuple, then 1st entry means spatial whiten in the CCA and second entry means temporal whiten in the CCA. Defaults to True.
            badEpThresh (int, optional): Bad epoch threshold for computing a robustified covariance matrix for the CCA fit.  See `utils.py::zero_outliers` for information on how the threshold is used, and `updateSummaryStatistics.py::updateSummaryStatistics` for usage in computing the summary statistics needed to compute a CCA model. Defaults to 6.
            badWinThresh (int, optional): Bad temporal-window threshold for computing a robustified covariance matrix for the CCA fit.  N.B. by default a single window is 1/50th of a single trial. See `utils.py::zero_outliers` for information on how the threshold is used, and `updateSummaryStatistics.py::updateSummaryStatistics` for usage in computing the summary statistics needed to compute a CCA model. Defaults to 3.
            outputscore (str, optional): _description_. Defaults to 'ip'.
            whiten_alg (str, optional): The type of algorithm used for the whitening step of the CCA algorithm. One of 'eigh' or 'chol'.  See `multiCCA.py` for more information. Defaults to 'eigh'.
            cxxp (bool, optional): _description_. Defaults to True.
            temporal_basis (_type_, optional): As temporal basis is an array mapping from the raw sample temporal basis to a new parametric basis, such as the amplitude of a particular sinusoidal frequency. See `temporal_basis.py` for more information on the available types of basis. Defaults to None.
            normalize_sign (bool, optional): When making the plots/models attempt to normalize the sign so different plots look more similar. Defaults to True.
            score_type (str, optional): When computing the model-fit score by cross-validation, which type of score to compute.  One of: 'score_auc' for Area-under-curve classification score, or 'corr' for correlation based goodness of fit in the latent space fit quality. Defaults to None.
            evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
            tau (int, optional): the length in samples of the stimulus response. Defaults to None.
            offset (int, optional): offset from the event time for the response window. Defaults to None.
            tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
            offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
            fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
            priorweight (float, Optional): the weighting in pseudo-samples for the prior estimate for the prediction noise variance.  Defaults to 120.
            startup_correction (int, Optional): length in samples of addition startup correction where the noise-variance is artificially increased due to insufficient data.  Defaults to 100.
            prediction_offsets (list-of-int, optional): set of offsets to test when performing prediction to check/adapt to stimulus to response latency changes.  Defaults to None.
            centerY (bool, optional): center the stimulus-sequence information to be zero-mean per trial before fitting the response model.  Defaults to False.
            nvirt_out (int, optional): number of virtual outputs to add when testing.  If <0 then this is a limit on the totoal nubmer of outputs if >0 then add this number additional virtual outputs.  Defaults to -20.
            nocontrol_condn (float, optional): relative importance of correct prediction of nocontrol when no stimulus is present.  Defaults to .5
            verb (int, optional): verbosity level, 0 is default, higher number means be more talky in debug messages.  Defaults to 0.
        """
        super().__init__(evtlabs=evtlabs, tau=tau,  offset=offset, tau_ms=tau_ms,
                         offset_ms=offset_ms, fs=fs, outputscore=outputscore, score_type=score_type, **kwargs)
        self.rank, self.reg, self.rcond, self.badEpThresh, self.badWinThresh, self.symetric, self.center, self.CCA, self.whiten_alg, self.cxxp, self.normalize_sign, self.temporal_basis = (
            rank, reg, rcond, badEpThresh, badWinThresh, symetric, center, CCA, whiten_alg, cxxp, normalize_sign, temporal_basis)

    def fit_cca(
            self, Cxx_dd, Cyx_yetd, Cyy_yetet, rank=None, reg=None, rcond=None, CCA=None, symetric=None,
            whiten_alg=None, temporal_basis=None):
        """fit a bi-directional CCA model from the summary statistics (covariance matrices) of the data with given parameters
        The fitted model is:
               X_Td w_d = Y_Te * r_te
        where * is temporal convolution

        Args:
            Cxx_dd (_type_): Spatial covariance of the data with d-channels/electrodes.  This has shape (d,d) = (nElectrodes, nElectrodes)
            Cyx_yetd (_type_): Spatio-temporal cross-covariance between the data (X_Td) and the stimuli (Y_Tye).   This has shape (y,e,t,d) = (nOuputs, num-event-type, response-length tau, n-electrodes)
            Cyy_yetet (ndarray): Temporal auto-covariance of the stimulus.  This has shape (y,e,t,e,t) = (number outputs, number stimulus event types, response length tau, number stimulus event types, response length tau)
            rank (int, optional): Rank of the model to fit, i.e. the number of components, or the number of sources to fit. Defaults to None.
            reg (float, optional): Regularization strength to use. 0=no-reg, 1=fully regularized. Defaults to None.
            rcond (float, optional): recipiol condition number limit when fitting.  components smaller than this threshold are removed to make the system more solvable. (see definition in SVD for it's use). Defaults to None.
            CCA (bool, optional): flag if we do CCA or PLS or hybrid.  If true is full CCA, false is PLS. Defaults to None.
            symetric (bool, optional): flag if we require the robust-whitener used internally to use a symetric model. Defaults to None.
            whiten_alg (_type_, optional): the algorithm to use to find the robust whitener.  See: multiCCA.robust_whitener for more info. Defaults to None.
            temporal_basis (str|ndarray, optional): The temporal basis map used when fiting the `R_yet` parameters of the model.   This mapping can be used to reduce the number of parameters to fit in R_yet by sharing parameters between differnt time-points, outputs or stimulus-events.
                            See 'temporal_basis.get_temporal_basis` for info on the types of basis maps currently supported. Defaults to None.

        Returns:
            W_kd: the spatial filter mapping from sensors to estimated source activations.  This has shape (k,d) = (number sources, number sensors)
            R_ket: the temporal response function mapping from stimulus events to impuluse-response-functions.  This has shape (k,e,t) = (number sources, number stimulus event types, response-duration tau)
        """
        if rank is None:
            rank = self.rank
        if reg is None:
            reg = self.reg
        if rcond is None:
            rcond = self.rcond
        if CCA is None:
            CCA = self.CCA
        if symetric is None:
            symetric = self.symetric
        if whiten_alg is None:
            whiten_alg = self.whiten_alg
        if temporal_basis is None:
            temporal_basis = self.temporal_basis

        # transform the temporal basis used
        Cxx_dd, Cyx_yetd, Cyy_yetet, temporal_basis_bt = apply_temporal_basis(
            temporal_basis, Cxx_dd, Cyx_yetd, Cyy_yetet)

        #print(" rank = {} reg={}".format(rank,reg))
        #print(" Cxx={} Cyx={} Cyy={}".format(Cxx_dd.shape, Cyx_yetd.shape, Cyy_yetet.shape))
        J, W_Mkd, R_Mket, A_Mkd, I_Mket = multipleCCA(
            Cxx_dd, Cyx_yetd, Cyy_yetet, reg=reg, rank=rank, rcond=rcond, CCA=CCA, symetric=symetric,
            whiten_alg=whiten_alg)
        # strip the model dim
        W_kd, R_ket, A_kd, I_ket = W_Mkd[0], R_Mket[0], A_Mkd[0], I_Mket[0]

        # invert the temporal basis mapping so is in sample space
        R_ket = invert_temporal_basis_mapping(temporal_basis_bt, R_ket)
        I_ket = invert_temporal_basis_mapping(temporal_basis_bt, I_ket)

        # normalize the sign of the model for each component
        if self.normalize_sign:
            sgn_k = (np.median(A_kd, axis=-1, keepdims=False) >= 0) * 2 - 1
            W_kd = W_kd * sgn_k[:, np.newaxis]
            R_ket = R_ket * sgn_k[:, np.newaxis, np.newaxis]
            A_kd = A_kd * sgn_k[:, np.newaxis]
            I_ket = I_ket * sgn_k[:, np.newaxis, np.newaxis]

        #print(" W={}, R={}".format(W.shape,R.shape))
        return W_kd, R_ket, A_kd, I_ket

    def fit(self, X, y, stimTimes=None, fs=None, calibrate_softmaxscale: bool = False):
        '''fit 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, Y)'''

        # print("mcca")
        self.fit_tau_offset(X, y, fs=fs)
        X_TSd = X
        Y_TSy = y
        if X_TSd.ndim > 3:  # collapse additional feature dims into a single big one
            X_TSd = X_TSd.reshape(X_TSd.shape[:2]+(-1,))
        # map to event sequence
        Y_TSye = self.stim2event(Y_TSy, fit=True)  # (tr,samp,nY,e)

        # extract the true target to fit to, keeping the dim shape intack
        Y_true = Y_TSye[:, :, 0:1, ...] if np.any(Y_TSye[:, :, 0, :]) else np.sum(
            Y_TSye, 2, keepdims=True)  # (tr,samp,1,e)
        # get summary statistics
        #print("X={} |X|={}".format(X.shape,np.sum(X**2,axis=(0,1))))
        Cxx, Cyx, Cyy = updateSummaryStatistics(
            X_TSd, Y_true, stimTimes, tau=self.tau_, offset=self.offset_, badEpThresh=self.badEpThresh,
            badWinThresh=self.badWinThresh, center=self.center, cxxp=self.cxxp)

        # import matplotlib.pyplot as plt
        # from mindaffectBCI.decoder.updateSummaryStatistics import plot_summary_statistics
        # plt.figure()
        # plot_summary_statistics(Cxx, Cxy, Cyy)

        # print("diag(Cxx)={}".format(np.diag(Cxx)))
        # do the CCA fit
        W, R, A, I = self.fit_cca(Cxx, Cyx, Cyy, self.rank, self.reg, temporal_basis=self.temporal_basis)
        # maintain type compatiability
        W = W.astype(X_TSd.dtype)
        R = R.astype(X_TSd.dtype)
        self.W_ = W  # (nM,rank,d)
        self.R_ = R  # (nM,rank,e,tau)
        self.A_ = A.astype(X_TSd.dtype)  # np.einsum("de,Mkd->Mke",Cxx,W) #
        self.I_ = I.astype(X_TSd.dtype)  # np.einsum("Metut,Mket->Mkut",Cyy,R) #
        self.fit_b(X_TSd)  # (nM,e)

        # just in case
        if calibrate_softmaxscale:
            self.calibrate_softmaxscale(X_TSd, Y_TSye)
        else:
            self.sigma0_ = None
            self.softmaxscale_ = 3

        return self

    def fit_b(self, X_TSd):
        """fit the bias parameter given the other parameters and a dataset

        Args:
            X (np.ndarray): the target data
        """
        if self.center:  # use bias terms to correct for data centering
            if X_TSd.ndim > 3:  # collapse additional feature dims into a single big one
                X_TSd = X_TSd.reshape(X_TSd.shape[:2]+(-1,))
            muX = np.mean(X_TSd.reshape((-1, X_TSd.shape[-1])), 0)
            self.b_ = -np.einsum("kd,d,ket->e", self.W_, muX, self.R_)  # (nM,e)
        else:
            self.b_ = None
        return self

    def cv_fit(self, X, Y, cv=None, fs=None, fit_params: dict = dict(), verbose: bool = 0, score_type=None,
               return_estimator: bool = True, calibrate_softmax: bool = True, retrain_on_all: bool = True, ranks=None):
        ''' cross validated fit to the data.  optimized wrapper for optimization of the model rank.'''
        if cv is None:
            cv = self.inner_cv_params
        # fast path for cross validation over rank
        cv_in = cv.copy() if hasattr(cv, 'copy') else cv  # save copy of cv info for later
        if ranks is None:
            # call the base version
            return BaseSequence2Sequence.cv_fit(
                self, X, Y, cv=cv_in, fit_params=fit_params, verbose=verbose, return_estimator=return_estimator,
                calibrate_softmax=calibrate_softmax, retrain_on_all=retrain_on_all)

        if score_type is None:
            score_type = self.score_type

        # compute the summary statistics for each trial
        maxrank = max(ranks)
        self.rank = maxrank
        scores = [[] for i in ranks]

        cv = self.get_folding(X, Y, cv)
        if verbose > 0:
            print("CV:", end='')

        for i, (train_idx, valid_idx) in enumerate(cv):
            if verbose > 0:
                print(".", end='', flush=True)
            #print("trn={} val={}".format(train_idx,valid_idx))
            if X[valid_idx, ...].size == 0:
                print("Warning: no-validation trials!!! using all data!")
                valid_idx = slice(X.shape[0])

            # 1) fit with max-rank
            self.fit(X[train_idx, ...], Y[train_idx, ...], fs=fs, **fit_params)

            # 2) Extract the desired rank-sub-models and predict with them
            W = self.W_  # (nM,rank,d)
            R = self.R_  # (nM,rank,e,tau)
            for ri, r in enumerate(ranks):
                self.W_ = W[..., :r, :]
                self.R_ = R[..., :r, :, :]
                self.fit_b(X[train_idx, ...])
                # predict, forcing removal of copies of  tgt=0 so can score
                Fyi = self.predict(X[valid_idx, ...], Y[valid_idx, ...], dedup0=False)
                if callable(score_type):
                    score = score_type(self, X[valid_idx,...], Y[valid_idx,...])
                elif score_type == 'corr':
                    score = self.score_corr(X[valid_idx, ...], Y[valid_idx, ...])
                else:
                    score = self.inner_score_audc(Fyi)
                if i == 0 and ri == 0:  # reshape Fy to include the extra model dim
                    Fy = np.zeros((len(ranks), Y.shape[0])+Fyi.shape[1:], dtype=np.float32)
                Fy[ri, valid_idx, ..., :Fyi.shape[-1]] = Fyi[..., :Fy.shape[-1]]

                scores[ri].append(score)

        # 3) get the *best* rank
        scores = np.mean(np.array(scores), axis=-1)  # (ranks,folds) -> ranks
        print("Rank score: " + ", ".join(["{}={:4.3f}".format(r, s) for (r, s) in zip(ranks, scores)]), end='')
        maxri = np.argmax(scores)
        self.rank = ranks[maxri]
        print(" -> best={}".format(self.rank))

        # final retrain with all the data
        # TODO[]: just use a normal fit, and get the Fy from the above CV loop
        res = BaseSequence2Sequence.cv_fit(
            self, X, Y, cv=cv_in, fit_params=fit_params, verbose=verbose, return_estimator=return_estimator,
            calibrate_softmax=calibrate_softmax, retrain_on_all=retrain_on_all)
        res['Fy_rank'] = Fy  # store the pre-rank info
        return res



class MultiCCACV(MultiCCA):
    def __init__(self, inner_cv: int = 5, inner_cv_params: dict = None,
                 # other parms explicilty given so sklearn auto-set and clone works....
                 rank=1, reg=(1e-8, 1e-8), rcond=(1e-4, 1e-8), badEpThresh=6, symetric=True,
                 center=True, CCA=True, outputscore: str = 'ip', whiten_alg: str = 'eigh', cxxp=True, temporal_basis=None,
                 evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, normalize_sign=True, score_type:str='audc',
                 **kwargs):
        """wrapper which does an inner cv to fit the inner_cv_params of a MultiCCA model
        Args:
            inner_cv (int|CrossValidation, optional): the number of inner folds to use. Defaults to 5.
            inner_cv_params (dict, optional): dictionary of lists of parameters to test, using a similar format as that for `sklearn.model_selection.cross_validate`'s 'fit_params' option.  Defaults to None.
            rank (int, optional): The rank (number of components) of the model to fit. Defaults to 1.
            reg (tuple, optional): Regularization strength for (spatial,temporal) components when performing the CCA fit. See `multiCCA.py::robust_whiten` for more information. Defaults to (1e-8, 1e-8).
            rcond (tuple, optional): Recipol-condition-number (spatial,temporal) components when performing the CCA fit. See `multiCCA.py::robust_whiten` for more information.. Defaults to (1e-4, 1e-8).
            symetric (bool, optional): Use a symetric whitener in the CCA fitting.  See `multiCCA.py` for more information. Defaults to True.
            center (bool, optional): center the data before fitting the model. Defaults to True.
            CCA (bool, optional): do we run CCA or PLS or a hybrid.  If True is CCA, if False is PLS.  If 2-tuple, then 1st entry means spatial whiten in the CCA and second entry means temporal whiten in the CCA. Defaults to True.
            badEpThresh (int, optional): Bad epoch threshold for computing a robustified covariance matrix for the CCA fit.  See `utils.py::zero_outliers` for information on how the threshold is used, and `updateSummaryStatistics.py::updateSummaryStatistics` for usage in computing the summary statistics needed to compute a CCA model. Defaults to 6.
            badWinThresh (int, optional): Bad temporal-window threshold for computing a robustified covariance matrix for the CCA fit.  N.B. by default a single window is 1/50th of a single trial. See `utils.py::zero_outliers` for information on how the threshold is used, and `updateSummaryStatistics.py::updateSummaryStatistics` for usage in computing the summary statistics needed to compute a CCA model. Defaults to 3.
            outputscore (str, optional): _description_. Defaults to 'ip'.
            whiten_alg (str, optional): The type of algorithm used for the whitening step of the CCA algorithm. One of 'eigh' or 'chol'.  See `multiCCA.py` for more information. Defaults to 'eigh'.
            cxxp (bool, optional): _description_. Defaults to True.
            temporal_basis (_type_, optional): As temporal basis is an array mapping from the raw sample temporal basis to a new parametric basis, such as the amplitude of a particular sinusoidal frequency. See `temporal_basis.py` for more information on the available types of basis. Defaults to None.
            normalize_sign (bool, optional): When making the plots/models attempt to normalize the sign so different plots look more similar. Defaults to True.
            score_type (str, optional): When computing the model-fit score by cross-validation, which type of score to compute.  One of: 'score_auc' for Area-under-curve classification score, or 'corr' for correlation based goodness of fit in the latent space fit quality. Defaults to None.
            evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
            tau (int, optional): the length in samples of the stimulus response. Defaults to None.
            offset (int, optional): offset from the event time for the response window. Defaults to None.
            tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
            offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
            fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
            priorweight (float, Optional): the weighting in pseudo-samples for the prior estimate for the prediction noise variance.  Defaults to 120.
            startup_correction (int, Optional): length in samples of addition startup correction where the noise-variance is artificially increased due to insufficient data.  Defaults to 100.
            prediction_offsets (list-of-int, optional): set of offsets to test when performing prediction to check/adapt to stimulus to response latency changes.  Defaults to None.
            centerY (bool, optional): center the stimulus-sequence information to be zero-mean per trial before fitting the response model.  Defaults to False.
            nvirt_out (int, optional): number of virtual outputs to add when testing.  If <0 then this is a limit on the totoal nubmer of outputs if >0 then add this number additional virtual outputs.  Defaults to -20.
            nocontrol_condn (float, optional): relative importance of correct prediction of nocontrol when no stimulus is present.  Defaults to .5
            verb (int, optional): verbosity level, 0 is default, higher number means be more talky in debug messages.  Defaults to 0.
        """        
        super().__init__(rank=rank, reg=reg, rcond=rcond, badEpThresh=badEpThresh, symetric=symetric, center=center,
                         CCA=CCA, outputscore=outputscore, whiten_alg=whiten_alg, cxxp=cxxp, temporal_basis=temporal_basis,
                         evtlabs=evtlabs, tau=tau, offset=offset, tau_ms=tau_ms, offset_ms=offset_ms, fs=fs,
                         normalize_sign=normalize_sign, score_type=score_type, **kwargs)
        self.inner_cv_params, self.inner_cv = (inner_cv_params, inner_cv)

    def fit(self, X, Y, fs=None, inner_cv_params: dict = None, retrain_on_all: bool = True):
        """override fit, to do a cv_fit to optimize the hyper-params if inner_cv_params is given, otherwise a normal fit

        Args:
            X ([type]): [description]
            Y ([type]): [description]
            fs ([type], optional): [description]. Defaults to None.
            inner_cv_params (dict, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if inner_cv_params is None:
            inner_cv_params = self.inner_cv_params
        if inner_cv_params is not None or self.inner_cv is not None:  # inner CV for fit-params
            self.cv_fit(X, Y, fs=fs, cv=self.inner_cv, fit_params=inner_cv_params, retrain_on_all=retrain_on_all)
        else:
            super().fit(X, Y, fs=fs)
        return self

    def per_trial_summaryStatistics(self, X_TSd, Y_TSye, tau, offset, center, cxxp, badEpThresh, badWinThresh):
        """compute the summary statistics (covariance matrices) needed to fit the CCA model on a per-trial basis

          This is used to cache this computation when performing multiple cross-validated fits.

        Args:
            X_TSd (_type_): The EEG data, with shape (T,S,d) = (number trials, number-samples in a trial, number electrodes)
            Y_TSye (_type_): The brain-trigger stimulus sequence.  with shape (T,S,y,e) = (number trials, number samples in trial, number outputs, number stimulus event types per output)
            tau (int): Response length in samples.
            offset (int): Offset in samples for the start of the response.  (Not TESTED)
            center (bool): flag if we center (zero-mean) the data (X) before fitting.
            cxxp (bool): flag if we run a full CCA with whitening of X before fitting.
            badEpThresh (float): threshold for datection of bad epochs, i.e. whole trials, in numbers of standard deviations of excess power above that of the average trial.
            badWinThresh (_type_): threshold for detection of bad temporal-windows, i.e. 3s blocks of data, in numbers of standard deviations of excess power above an average temporal window.

        Returns:
            Cxxs, Cxys, Cyys: lists of summary statistics for data-spatial-covaraince (Cxxs), the data-stimulus cross covaraince (Cyxs), and the data auto-covariance (Cyys) 
        """
        X_TSd, Y_TSye = zero_outliers(X_TSd.copy(), Y_TSye.copy(), badEpThresh, badWinThresh, winsz=tau)
        Cxxs, Cyxs, Cyys = [], [], []
        for ti in range(X_TSd.shape[0]):
            Cxxi, Cyxi, Cyyi = updateSummaryStatistics(
                X_TSd[ti:ti+1, ...], Y_TSye[ti:ti+1, ...],
                tau=tau, offset=offset, center=center, cxxp=cxxp,
                badEpThresh=None, badWinThresh=None)
            Cxxs.append(Cxxi)
            Cyys.append(Cyyi)
            Cyxs.append(Cyxi)
        Cxxs, Cyxs, Cyys = np.stack(Cxxs, 0), np.stack(Cyxs, 0), np.stack(Cyys, 0)
        return Cxxs, Cyxs, Cyys

    def cv_fit(self, X, Y, cv=None, fs=None, verbose: bool = 0, score_type: str = None, return_estimator: bool = True,
               calibrate_softmax: bool = True, retrain_on_all: bool = True, ranks=None, fit_params: dict = None):
        ''' cross validated fit to the data.  optimized wrapper for optimization of the model rank.'''
        if cv is None:
            cv = self.inner_cv_params

        # fast path for cross validation over rank
        cv_in = cv.copy() if hasattr(cv, 'copy') else cv  # save copy of cv info for later
        if fit_params is None:
            fit_params = dict()
        if ranks is None:
            ranks = (self.rank,)

        # get the tau-offset info
        self.fit_tau_offset(X, Y, fs=fs)

        # setup Y
        X_TSd = X
        Y_TSy = Y
        if X_TSd.ndim > 3:  # collapse additional feature dims into a single big one
            X_TSd = X_TSd.reshape(X_TSd.shape[:2]+(-1,))
        # map to event sequence
        Y_TSye = self.stim2event(Y_TSy, fit=True)  # (tr,samp,nY,e)

        # extract the true target to fit to, using horible slicing trick
        Y_true = Y_TSye[:, :, 0:1, ...] if np.any(Y_TSye[:, :, 0, :]) else np.sum(
            Y_TSye, 2, keepdims=True)  # (tr,samp,1,e)

        # compute the summary statistics for each trial
        Cxxs, Cyxs, Cyys = self.per_trial_summaryStatistics(
            X_TSd, Y_true, self.tau_, self.offset_, center=self.center, cxxp=self.cxxp, badEpThresh=self.badEpThresh,
            badWinThresh=self.badWinThresh)

        # Rank is special hyper-param, so remove it from the fit_params
        if fit_params is not None:
            ranks = fit_params.pop('rank', None) or ranks
        if ranks is None:
            ranks = [self.rank]

        from sklearn.model_selection import ParameterGrid
        fit_configs = ParameterGrid(fit_params) if fit_params is not None else dict()

        if self.verb > 0:
            print("CV over {} ranks {} and\n {} configs: {}".format(
                len(ranks), ranks, len(fit_configs), [c for c in fit_configs]))

        cv = self.get_folding(X, Y, cv)
        if verbose > 0:
            print("CV:", end='')

        maxrank = max(ranks)
        self.rank = maxrank
        if score_type is None:  score_type = self.score_type
        scores_cv = [[[] for _ in ranks] for _ in fit_configs]  # double nested list of lists..
        for i, (train_idx, valid_idx) in enumerate(cv):
            if verbose > 0:
                print(".", end='', flush=True)
            #print("trn={} val={}".format(train_idx,valid_idx))
            if X[valid_idx, ...].size == 0:
                print("Warning: no-validation trials!!! using all data!")
                valid_idx = slice(X.shape[0])

            # get this folds summary statistics
            Cxx, Cyx, Cyy = [np.sum(C[train_idx], 0) for C in (Cxxs, Cyxs, Cyys)]

            for ci, fit_config in enumerate(fit_configs):
                # 1) fit with max-rank
                W, R, A, I = self.fit_cca(Cxx, Cyx, Cyy, rank=maxrank, **fit_config)
                #print(" W={} R={} ".format(W.shape,R.shape))

                # 2) Extract the desired rank-sub-models and predict with them
                for ri, r in enumerate(ranks):
                    self.W_ = W[..., :r, :].copy()
                    self.R_ = R[..., :r, :, :].copy()
                    self.fit_b(X_TSd[train_idx, ...])

                    Fyi = self.predict(X[valid_idx, ...], Y[valid_idx, ...], dedup0=False)
                    # save the prediction for later
                    if i == 0 and ri == 0 and ci == 0:  # reshape Fy to include the extra model dim
                        Fy_cv = np.zeros((len(fit_configs), len(ranks), Y.shape[0])+Fyi.shape[1:], dtype=np.float32)
                    Fy_cv[ci, ri, valid_idx, ..., :Fyi.shape[-1]] = Fyi[..., :Fy_cv.shape[-1]]

                    if callable(score_type):
                        score = score_type(self, X[valid_idx,...], Y[valid_idx,...])
                    elif score_type == 'corr':
                        Cxxval, Cyxval, Cyyval = [np.sum(C[valid_idx], 0) for C in (Cxxs, Cyxs, Cyys)]
                        score_y, _ = corr_cov(Cxxval, Cyxval, Cyyval, self.W_, self.R_)
                        score = score_y[0]  # strip silly y dim
                    else:
                        score = self.inner_score_audc(Fyi)
                    scores_cv[ci][ri].append(score)

        # 3) get the *best* rank
        scores_cv = np.array(scores_cv)
        avescore = np.mean(scores_cv, axis=-1)  # (regs,ranks,folds) -> ranks
        if self.verb > 0:
            for ci, c in enumerate(fit_configs):
                print("Cfg {}: Rank: ".format(
                    c) + ", ".join(["{}={:4.3f}".format(r, avescore[ci, ri]) for ri, r in enumerate(ranks)]))
        maxci, maxri = np.unravel_index(np.argmax(avescore, axis=None), avescore.shape)
        self.rank = ranks[maxri]
        for k, v in fit_configs[maxci].items():
            setattr(self, k, v)  # self.reg = regs[maxci]
        Fy = Fy_cv[maxci, maxri]
        scores = scores_cv[maxci][maxri]
        print("     -> best={},{} = {:4.3f}".format(fit_configs[maxci], self.rank, avescore[maxci, maxri]))

        self.score_cv_ = scores

        # setup the p-value calibration
        self.sigma0_ = self.estimate_prior_sigma(Fy)
        self.softmaxscale_ = self.calibrate_softmaxscale(Fy) if calibrate_softmax else 3
        if self.sigma0_ is None:
            self.sigma0_ = 1
        if self.softmaxscale_ is None:
            self.softmaxscale_ = 3
        #print("Sigma0={:4.3f}  softmaxscale={:4.3f}".format(self.sigma0_, self.softmaxscale_))

        # retrain on all the data
        if retrain_on_all:
            Cxx, Cyx, Cyy = [np.sum(C, 0) for C in (Cxxs, Cyxs, Cyys)]
            # 1) fit with max-rank
            print(" Retraining on all data with: rank={} reg={}".format(self.rank, self.reg))
            W, R, A, I = self.fit_cca(Cxx, Cyx, Cyy, rank=self.rank, reg=self.reg)
            #print(" W={} R={} ".format(W.shape,R.shape))
            self.W_ = W.astype(X.dtype)
            self.R_ = R.astype(X.dtype)
            self.A_ = A.astype(X.dtype)
            self.I_ = I.astype(X.dtype)

        return dict(
            estimator=Fy, estimator_cv=Fy_cv, test_score=scores,
            scores_cv=scores_cv, ranks=ranks, fit_configs=[cf for cf in fit_configs])


class MultiCCA2(MultiCCA):

    def fit_cca(self, Cxx_dd, Cyx_yetd, Cyy_yetet, rank=None, reg=None):
        """alternative mathmathical formulation of the CCA problem solving it directly as a generalized-eigenvalue problem.

        Note: this is computationally much more expensive than that used above, but may be numerically more stable in some cases.

        Args:
            Cxx_dd (_type_): Spatial covariance of the data with d-channels/electrodes.  This has shape (d,d) = (nElectrodes, nElectrodes)
            Cyx_yetd (_type_): Spatio-temporal cross-covariance between the data (X_Td) and the stimuli (Y_Tye).   This has shape (y,e,t,d) = (nOuputs, num-event-type, response-length tau, n-electrodes)
            Cyy_yetet (ndarray): Temporal auto-covariance of the stimulus.  This has shape (y,e,t,e,t) = (number outputs, number stimulus event types, response length tau, number stimulus event types, response length tau)
            rank (int, optional): Rank of the model to fit, i.e. the number of components, or the number of sources to fit. Defaults to None.
            reg (float, optional): Regularization strength to use. 0=no-reg, 1=fully regularized. Defaults to None.

        Returns:
            _type_: _description_
        """
        # do the CCA fit
        if rank is None:
            rank = self.rank
        if reg is None:
            reg = self.reg
        # setup the generalized eigen-value problem.
        Cxx_dd = Cxx_dd.reshape((Cxx_dd.shape[0], Cxx_dd.shape[1]))
        Cyy_et_et = Cyy_yetet.reshape((Cyy_yetet.shape[1]*Cyy_yetet.shape[2], Cyy_yetet.shape[3]*Cyy_yetet.shape[4]))
        Cyx_et_d = Cyx_yetd.reshape((Cyx_yetd.shape[1]*Cyx_yetd.shape[2], Cyx_yetd.shape[3]))

        # CCA =>  [ 0   Cyx ] [x]  = lambda [ Cxx  0   ]
        #         [ Cxy   0 ] [y]           [  0   Cyy ]
        LHS = np.zeros((Cyx_et_d.shape[1]+Cyx_et_d.shape[0], Cyx_et_d.shape[0]+Cyx_et_d.shape[1]))
        LHS[:Cyx_et_d.shape[1], Cyx_et_d.shape[1]:] = Cyx_et_d.T
        LHS[Cyx_et_d.shape[1]:, :Cyx_et_d.shape[1]] = Cyx_et_d

        # add a ridge (if wanted)
        if reg[0] > 0:
            Cxx_dd = Cxx_dd.copy()*(1-reg[0]) + reg[0]*np.mean(Cxx_dd.diagonal())*np.eye(Cxx_dd.shape[0])
        if reg[1] > 0:
            Cyy_et_et = Cyy_et_et.copy()*(1-reg[1]) + reg[1]*np.mean(Cyy_et_et.diagonal())*np.eye(Cyy_et_et.shape[0])

        RHS = np.zeros((Cxx_dd.shape[0]+Cyy_et_et.shape[0], Cxx_dd.shape[1]+Cyy_et_et.shape[1]))
        RHS[:Cxx_dd.shape[0], :Cxx_dd.shape[1]] = Cxx_dd
        RHS[Cxx_dd.shape[0]:, Cxx_dd.shape[1]:] = Cyy_et_et

        # solve the generalized eigen-value problem
        lm, V_det_k = sp.linalg.eigh(a=LHS, b=RHS, eigvals=(LHS.shape[0]-1-rank, LHS.shape[0]-1))
        slmidx = np.argsort(lm)  # N.B. ASCENDING order
        slmidx = slmidx[::-1]  # N.B. DESCENDING order
        V_det_k = V_det_k[:, slmidx[:rank]]

        # extract the solution components from Vs
        W_kd = V_det_k[:Cxx_dd.shape[0], :].T
        R_k_et = V_det_k[Cxx_dd.shape[0]:, :].T
        R_ket = R_k_et.reshape((R_k_et.shape[0], Cyx_yetd.shape[1], Cyx_yetd.shape[2]))

        # maintain type compatiability
        W = W_kd.astype(Cxx_dd.dtype)  # add model dim
        R = R_ket.astype(Cxx_dd.dtype)
        A = np.einsum("de,kd->ke", Cxx_dd, W)
        I = np.einsum("etut,ket->kut", Cyy_et_et.reshape(Cyy_yetet.shape[1:]), R)
        return W, R, A, I

    def fit(self, X, y, stimTimes=None, fs=None, calibrate_softmaxscale: bool = False):
        '''fit 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, Y)'''
        self.fit_tau_offset(X, y, fs=fs)
        X_TSd = X
        Y_TSy = y
        if X_TSd.ndim > 3:  # collapse additional feature dims into a single big one
            X_TSd = X_TSd.reshape(X_TSd.shape[:2]+(-1,))
        # map to event sequence
        Y_TSye = self.stim2event(Y_TSy, fit=True)  # (tr,samp,nY,e)

        # extract the true target to fit to, using horible slicing trick
        Y_true = Y_TSye[:, :, 0:1, ...]  # (tr,samp,1,e)
        # get summary statistics
        #print("X={} |X|={}".format(X.shape,np.sum(X**2,axis=(0,1))))
        Cxx_dd, Cyx_yetd, Cyy_yetet = updateSummaryStatistics(
            X_TSd, Y_true, stimTimes, tau=self.tau_, offset=self.offset_, badEpThresh=self.badEpThresh,
            center=self.center)

        # print("mcca2")
        W, R, A, I = self.fit_cca(Cxx_dd, Cyx_yetd, Cyy_yetet)
        # store A_ for plotting later
        self.W_ = W[np.newaxis, ...]  # (M,rank,d)
        self.R_ = R[np.newaxis, ...]  # (M,rank,e,tau)
        self.A_ = A[np.newaxis, ...]
        self.I_ = I[np.newaxis, ...]

        self.fit_b(X_TSd)  # (nM,e)

        # just in case
        if calibrate_softmaxscale:
            self.calibrate_softmaxscale(X_TSd, Y_TSye)
        else:
            self.sigma0_ = None
            self.softmaxscale_ = 3

        return self


# ----------------------------------------------------------
#
#  FWD
#
# -----------------------------------------------------------
class FwdLinearRegression(BaseSequence2Sequence):
    ''' Sequence 2 Sequence learning using forward linear regression  X = A*Y '''

    def __init__(
            self, evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, reg=None, rcond=1e-6,
            badEpThresh=6, center=True, **kwargs):
        """ Sequence 2 Sequence learning using forward linear regression equation  X = A*Y

        Args:
            evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
            tau (int, optional): the length in samples of the stimulus response. Defaults to None.
            offset (int, optional): offset from the event time for the response window. Defaults to None.
            tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
            offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
            fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
            verb (int, optional): verbosity level, 0 is default, higher number means be more talky in debug messages.  Defaults to 0.
        """        
        super().__init__(evtlabs=evtlabs, tau=tau, offset=offset, tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, **kwargs)
        self.reg, self.rcond, self.badEpThresh, self.center = reg, rcond, badEpThresh, center
        if offset and offset != 0:
            print("WARNING: not tested with offset!! YMMV")

    def fit_fwd(self, Cyy_tete, Cyx_texd, reg=1e-4, rcond=0):
        """fit a spatiotemporal forward model based on the data summary statistics, i.e. covariance matrices
          The fitted model is :
               X_Td =  Y_Te * W_ted
           where * is temporal convolution.

        Args:
            Cyy_tete (_type_): the stimulus cross-covariance.  This has shape (t,e,t,e) = ( response length tau, number stimulus event types, response length tau, number stimulus event types)
            Cyx_texd (_type_): the stimulus data cross-covariance.  This has shape (texd) = (response length tau, number of stimulus event types, number of data-features, number of data channels)
            reg (float, optional): regularization strength. Defaults to 1e-4.
            rcond (float, optional): condition number for matrix inverse. Defaults to 0.

        Returns:
            _type_: _description_
        """
        # Make 2d
        # (tau,e,tau,e) -> ((tau*e),(tau*e))
        Cyy_te_te = Cyy_tete.reshape((Cyy_tete.shape[0]*Cyy_tete.shape[1], Cyy_tete.shape[2]*Cyy_tete.shape[3]))
        # (tau,e,1,d) -> ((tau*e),d)
        Cyx_te_d = Cyx_texd.reshape((Cyx_texd.shape[0]*Cyx_texd.shape[1], Cyx_texd.shape[-1]))
        # add regularization
        if reg:
            # **IN-PLACE modify Cyy2d..**
            diag = Cyy_te_te.diagonal().copy()
            Cyy_te_te = Cyy_te_te*(1-reg)  # shrink
            np.fill_diagonal(Cyy_te_te, diag*(1-reg)+reg*np.mean(diag))  # ridge

        # fit by solving the matrix equation: Cyy*A=Cyx -> A=Cyy**-1Cyx
        # Q: solve or lstsq?
        # R = np.linalg.solve(Cyy2d, Cyx2d) # ((tau*e),d)
        W_te_d, _, _, _ = np.linalg.lstsq(Cyy_te_te, Cyx_te_d, rcond=rcond)  # ((tau*e),d)
        # convert back to 2d (tau,e,d)
        W_ted = W_te_d.reshape((Cyy_tete.shape[0], Cyy_tete.shape[1], Cyx_texd.shape[-1]))
        # BODGE: now moveaxis to make consistent with the prediction functions
        W_etd = np.moveaxis(W_ted, (0, 1, 2), (1, 0, 2))  # (tau,e,d)->(e,tau,d)
        return W_etd

    def fit(self, X, y, fs=None):
        '''fit 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, e)'''
        if X.ndim > 3:  # collapse additional feature dims into a single big one
            X = X.reshape(X.shape[:2]+(-1,))

        self.fit_tau_offset(X, y, fs=fs)
        if X.ndim > 3:  # collapse additional feature dims into a single big one
            X = X.reshape(X.shape[:2]+(-1,))
        # map to event sequence
        Y_TSye = self.stim2event(y, fit=True)  # (tr,samp,nY,e)
        # extract the true target to fit to
        Y_TSe = Y_TSye[..., 0, :]  # (tr,samp, e)
        # get summary statistics
        # first clean up the data..
        X, Y_TSe = zero_outliers(X, Y_TSe, self.badEpThresh)
        if self.center:
            # TODO[]: make work in-place w/o modify X
            X = X.copy()
            muX = np.mean(X.reshape((-1, X.shape[-1])), 0)
            X = X - muX

        # compute the summary statistics
        # TODO[]: make more efficient by computing the diag-tau version first then mapping it up
        if True:
            Cyy_yetet = updateCyy(None, Y_TSe[:, :, np.newaxis, :], None, tau=self.tau_, offset=self.offset_)
            Cyy_tete = np.moveaxis(Cyy_yetet[0, ...], (0, 1, 2, 3), (1, 0, 3, 2))  # yetet -> tete
            Cyx_yetd = updateCxy(None, X, Y_TSe[:, :, np.newaxis, :], None, tau=self.tau_, offset=self.offset_)
            Cyx_ted = np.moveaxis(Cyx_yetd[0, ...], (0, 1, 2), (1, 0, 2))  # yetd -> ted

        else:

            # TODO: should be negative tau to represent causality of Y->X, so earlier Y cause X
            Cyy_tete = crossautocov(Y_TSe, Y_TSe, tau=self.tau_)  # (tau,e,tau,e) cross-cov shifted Y with itself
            # N.B. should be shifted Y with X, but we want Y *backwards* in time, thus
            #      as a *BODGE* we instead window X, as forward for X == backwards for Y
            # Cytexd  = crossautocov(Y_true, X, tau=[-self.tau, 1]) # (tau,e,1,d) shifted Y with X
            Cyx_te1d = crossautocov(Y_TSe, X, tau=[self.tau_, 1], offset=self.offset_)  # (1,e,tau,d) shifted X with Y
            # strip the singlenton dim
            Cyx_ted = Cyx_te1d.reshape(Cyx_te1d.shape[:2]+Cyx_te1d[3:])
            # Cyx_te1d  = np.moveaxis(Cyx_1etd, (0, 1, 2, 3), (2, 1, 0, 3)) # (tau,e,1,d)

        # fit:
        W_etd = self.fit_fwd(Cyy_tete, Cyx_ted, self.reg, self.rcond)
        self.W_ = W_etd
        self.R_ = None
        self.I_ = None
        if self.center:
            self.b_ = -np.einsum("etd,d->e", self.W_, muX)  # (e,)
        else:
            self.b_ = None

        self.sigma0_, self.softmaxscale_ = (None, 3)

        return self


class FwdLinearRegressionCV(FwdLinearRegression):

    ''' Sequence 2 Sequence learning using forward linear regression  X = A*Y '''

    def __init__(self, inner_cv: int = 5, inner_cv_params: dict = None,
                 # explicit args so sklearn can autoset and clone
                 evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, reg=None, badEpThresh=6, center=True,
                 **kwargs):
        super().__init__(evtlabs=evtlabs, tau=tau, offset=offset, tau_ms=tau_ms,
                         offset_ms=offset_ms, fs=fs, reg=reg, badEpThresh=badEpThresh, center=center, **kwargs)
        self.inner_cv, self.inner_cv_params = (inner_cv, inner_cv_params)

    def fit(self, X, Y, fs=None, inner_cv_params: dict = None, retrain_on_all: bool = True):
        """override fit, to do a cv_fit to optimize the hyper-params if inner_cv_params is given, otherwise a normal fit

        Args:
            X_TSd (_type_): The EEG data, with shape (T,S,d) = (number trials, number-samples in a trial, number electrodes)
            Y_TSye (_type_): The brain-trigger stimulus sequence.  with shape (T,S,y,e) = (number trials, number samples in trial, number outputs, number stimulus event types per output)
            tau (int): Response length in samples.
            fs ([type], optional): sample rate of the data, for plotting. Defaults to None.
            inner_cv_params (dict, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if inner_cv_params is None:
            inner_cv_params = self.inner_cv_params
        if inner_cv_params is not None:  # inner CV for fit-params
            self.cv_fit(X, Y, fs=fs, cv=self.inner_cv, fit_params=inner_cv_params, retrain_on_all=retrain_on_all)
        else:
            super().fit(X, Y, fs=fs)
        return self

    def cv_fit(
            self, X, Y, cv=None, fs=None, verbose: bool = 0, return_estimator: bool = True, calibrate_softmax: bool = True,
            retrain_on_all: bool = True, fit_params: dict = None):
        ''' cross validated fit to the data.  optimized wrapper for optimization of the model rank.'''
        if cv is None:
            cv = self.inner_cv_params
        # fast path for cross validation over rank
        if fit_params is None or len(fit_params) == 0:
            # call the base version
            return super().cv_fit(self, X, Y, cv=cv, fs=fs, verbose=verbose, return_estimator=return_estimator,
                                  calibrate_softmax=calibrate_softmax, retrain_on_all=retrain_on_all)

        # first clean up the data..
        X, Y = zero_outliers(X, Y, self.badEpThresh)
        if self.center:
            # TODO[]: make work in-place w/o modify X
            X = X.copy()
            muX = np.mean(X.reshape((-1, X.shape[-1])), 0)
            X = X - muX

        # setup Y
        X_TSd = X
        Y_TSy = Y
        if X_TSd.ndim > 3:  # collapse additional feature dims into a single big one
            X_TSd = X_TSd.reshape(X_TSd.shape[:2]+(-1,))
        # map to event sequence
        Y_TSye = self.stim2event(Y_TSy, fit=True)  # (tr,samp,nY,e)

        # extract the true target to fit to, using horible slicing trick
        Y_TSe = Y_TSye[:, :, 0, ...]  # (tr,samp,e)

        # get the tau-offset info
        self.fit_tau_offset(X, Y, fs=fs)

        # compute summary statistics for each trial individually, to speed up the CV fits
        # compute the summary statistics for each trial
        Cyys_tete = []
        Cyxs_ted = []
        for ti in range(X.shape[0]):
            # get summary statistics
            Cyy_yetet = updateCyy(None, Y_TSe[ti:ti+1, :, np.newaxis, :], None, tau=self.tau_, offset=self.offset_)
            Cyy_tete = np.moveaxis(Cyy_yetet[0, ...], (0, 1, 2, 3), (1, 0, 3, 2))  # yetet -> tete
            Cyys_tete.append(Cyy_tete)

            Cyx_yetd = updateCxy(
                None, X[ti: ti + 1, ...],
                Y_TSe[ti: ti + 1, :, np.newaxis, :],
                None, tau=self.tau_, offset=self.offset_)
            Cyx_ted = np.moveaxis(Cyx_yetd[0, ...], (0, 1, 2), (1, 0, 2))  # yetd -> ted
            Cyxs_ted.append(Cyx_ted)

        from sklearn.model_selection import ParameterGrid
        fit_configs = ParameterGrid(fit_params) if fit_params is not None else dict()

        if self.verb > 0:
            print("CV over {} configs: {}".format(len(fit_configs), fit_configs))

        cv = self.get_folding(X, Y, cv)
        if verbose > 0:
            print("CV:", end='')

        scores_cv = [[] for _ in fit_configs]  # list of lists, cv[params]
        for i, (train_idx, valid_idx) in enumerate(cv):
            if verbose > 0:
                print(".", end='', flush=True)
            #print("trn={} val={}".format(train_idx,valid_idx))
            if X[valid_idx, ...].size == 0:
                print("Warning: no-validation trials!!! using all data!")
                valid_idx = slice(X.shape[0])

            for ci, fit_config in enumerate(fit_configs):

                Cyy_tete = sum([Cyys_tete[ti] for ti in train_idx])
                Cyx_ted = sum([Cyxs_ted[ti] for ti in train_idx])

                W_etd = self.fit_fwd(Cyy_tete, Cyx_ted, **fit_config)

                self.W_, self.R_ = (W_etd, None)
                self.b_ = np.einsum("etd,d->e", self.W_, -muX) if self.center else None

                # predict, forcing removal of copies of  tgt=0 so can score
                Fyi = self.predict(X[valid_idx, ...], Y[valid_idx, ...], dedup0=False)
                score = self.inner_score_audc(Fyi)

                # save the prediction for later
                if i == 0 and ci == 0:  # reshape Fy to include the extra model dim
                    Fy_cv = np.zeros((len(fit_configs), Y.shape[0])+Fyi.shape[1:], dtype=np.float32)
                Fy_cv[ci, valid_idx, ..., :Fyi.shape[-1]] = Fyi

                scores_cv[ci].append(score)

        # 3) get the *best* reg
        avescore = np.mean(np.array(scores_cv), axis=-1)  # (cfgs,folds) -> ranks
        if self.verb > 0:
            for ci, c in enumerate(fit_configs):
                print("Cfg {} = {:4.3f} ".format(c, avescore[ci]))
        maxci = np.argmax(avescore)
        opt_config = fit_configs[maxci]
        for k, v in opt_config.items():
            setattr(self, k, v)  # self.reg = regs[maxci]
        Fy = Fy_cv[maxci, ...]
        scores = scores_cv[maxci]
        print("     -> best={} = {:4.3f}".format(fit_configs[maxci], avescore[maxci]))

        # setup the p-value calibration
        self.sigma0_ = self.estimate_prior_sigma(Fy)
        self.softmaxscale_ = self.calibrate_softmaxscale(Fy) if calibrate_softmax else 3
        #print("Sigma0={:4.3f}  softmaxscale={:4.3f}".format(self.sigma0_, self.softmaxscale_))

        # retrain on all the data
        if retrain_on_all:
            Cyy_tete = sum(Cyys_tete)
            Cyx_ted = sum(Cyxs_ted)
            # 1) fit with max-rank
            W_etd = self.fit_fwd(Cyy_tete, Cyx_ted)

            self.W_, self.R_ = (W_etd, None)
            self.b_ = np.einsum("etd,d->e", self.W_, -muX) if self.center else None

        return dict(
            estimator=Fy, score=scores_cv[maxci],
            estimator_cv=Fy_cv, scores_cv=scores_cv, fit_configs=[cf for cf in fit_configs])


# ----------------------------------------------------------
#
#  BWD
#
# -----------------------------------------------------------

class BwdLinearRegression(BaseSequence2Sequence):
    ''' Sequence 2 Sequence learning using backward linear regression  W*X = Y '''

    def __init__(
            self, evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, reg=None, badEpThresh=6,
            center=True, compdiagcxx: bool = True, temporal_basis: str = None, **kwargs):
        """ Sequence 2 Sequence learning using backward linear regression equation  W*X = Y

        Args:
            evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
            tau (int, optional): the length in samples of the stimulus response. Defaults to None.
            offset (int, optional): offset from the event time for the response window. Defaults to None.
            tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
            offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
            badEpThresh (int, optional): Bad epoch threshold for computing a robustified covariance matrix for the CCA fit.  See `utils.py::zero_outliers` for information on how the threshold is used, and `updateSummaryStatistics.py::updateSummaryStatistics` for usage in computing the summary statistics needed to compute a CCA model. Defaults to 6.
            temporal_basis (_type_, optional): As temporal basis is an array mapping from the raw sample temporal basis to a new parametric basis, such as the amplitude of a particular sinusoidal frequency. See `temporal_basis.py` for more information on the available types of basis. Defaults to None.
            fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
            verb (int, optional): verbosity level, 0 is default, higher number means be more talky in debug messages.  Defaults to 0.
        """        
        super().__init__(evtlabs=evtlabs, tau=tau, offset=offset, tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, **kwargs)
        # TODO[]: fit the bias term while learning
        self.reg, self.badEpThresh, self.center, self.compdiagcxx, self.temporal_basis = \
            reg, badEpThresh, center, compdiagcxx, temporal_basis
        if offset and offset != 0:
            print("WARNING: not tested with offset!! YMMV")

    def fit_bwd(self, Cxx_tdtd, Cxy_tde, reg: float = None, rcond: float = None, verb: int = 0):
        """fit a spatiotemporal backward model based on the data summary statistics, i.e. covariance matrices.
          The fitted model is :
               Y_Te = X_Td * W_ted
           where * is temporal convolution.

        Args:
            Cxx_tdtd (_type_): the data cross-covariance.  This has shape (t,d,t,d) = ( response length tau, number of sensors, response length tau, number of sensors) 
            Cxy_tde (_type_): the stimulus data cross-covariance.  This has shape (tde) = (response length tau, number of data sensors, number of stimulus event types)
            reg (float, optional): regularization strength. Defaults to 1e-4.
            rcond (float, optional): condition number for matrix inverse. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if reg is None:
            reg = self.reg

        # fit:
        # Make 2d
        # (tau,d,tau,d) -> ((tau*d),(tua*d))
        Cxx_td_td = Cxx_tdtd.reshape((Cxx_tdtd.shape[0]*Cxx_tdtd.shape[1],
                                      Cxx_tdtd.shape[2]*Cxx_tdtd.shape[3]))
        # (tau,d,e) -> ((tau*d),e)
        Cxy_td_e = Cxy_tde.reshape((Cxy_tde.shape[0]*Cxy_tde.shape[1],
                                    Cxy_tde.shape[2]))
        # add regularization
        if reg:
            # **IN-PLACE modify Cxx2d..**
            diag = Cxx_td_td.diagonal().copy()
            Cxx_td_td = Cxx_td_td*(1-reg)  # shrink
            np.fill_diagonal(Cxx_td_td, diag*(1-reg)+reg*np.mean(diag))  # ridge

        if verb > 0:
            print("Cxx={}  Cxy={}".format(Cxx_td_td.shape, Cxy_td_e.shape))

        # fit by solving the matrix equation: Cxx*W=Cxy -> W=Cxx**-1Cxy
        try:
            W_td_e = np.linalg.solve(Cxx_td_td, Cxy_td_e)
        except:
            W_td_e, _, _, _ = np.linalg.lstsq(Cxx_td_td, Cxy_td_e, rcond=rcond)  # ((tau*d*),e)
        # # ((tau*d*),e) # can have problems with singular inputs!
        # convert back to 2d
        W_tde = W_td_e.reshape((Cxx_tdtd.shape[0], Cxx_tdtd.shape[1], Cxy_tde.shape[-1]))  # (tau,d,e)
        # BODGE: moveaxis to make consistent for the prediction functions
        W_edt = np.moveaxis(W_tde, (0, 1, 2), (1, 2, 0))  # (tau,d,e)->(e,tau,d)
        return W_edt

    def fit(self, X, y, fs=None):
        '''fit 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, y, e)'''
        if X.ndim > 3:  # collapse additional feature dims into a single big one
            X = X.reshape(X.shape[:2]+(-1,))

        self.fit_tau_offset(X, y, fs=fs)
        if X.ndim > 3:  # collapse additional feature dims into a single big one
            X = X.reshape(X.shape[:2]+(-1,))
        # map to event sequence
        Y_TSye = self.stim2event(y, fit=True)  # (tr,samp,nY,e)
        # extract the true target to fit to
        Y_TSe = Y_TSye[..., 0, :]  # (tr,samp, e)
        # first clean up the data..
        X, Y_TSe = zero_outliers(X, Y_TSe, self.badEpThresh)
        if self.center:
            # TODO[]: make work in-place w/o modify X
            X = X.copy()
            muX = np.mean(X.reshape((-1, X.shape[-1])), 0)
            X = X - muX
        # get summary statistics
        if self.compdiagcxx:
            # compute the compressed cross-auto-cov
            Cxx_tdd = compCxx_diag(X, tau=self.tau_)  # (tau,d,d)
            Cxx_tdtd = Cxx_diag2full(Cxx_tdd)  # expand to the full version
            Cxy_tyed, taus = compCyx_diag(
                X, Y_TSe[:, :, np.newaxis, :],
                tau=[1, self.tau_],
                offset=self.offset_)  # (tau,d,1,e) shifted X with Y
            # permute dim order and strip the singlenton y dimension
            Cxy_tdey = np.moveaxis(Cxy_tyed, (0, 1, 2, 3), (0, 3, 2, 1))
            Cxy_tde = Cxy_tdey.reshape(Cxy_tdey.shape[:3])
        else:
            # compute directly the full t,d,t,d cross-auto-cov
            Cxx_tdtd = crossautocov(X, X, tau=(self.tau_, self.tau_))  # (tau,d,tau,d) cross-cov shifted X with itself
            Cxy_1dte = crossautocov(X, Y, tau=[1, self.tau_], offset=self.offset_)  # (tau,d,1,e) shifted X with Y
            Cxy_tde1 = np.moveaxis(Cxy_1dte, (0, 1, 2, 3), (3, 1, 0, 2))
            Cxy_tde = Cxy_tde1.reshape(Cxy_tde1.shape[:3])

        temporal_basis_bt = get_temporal_basis(self.tau_, self.temporal_basis, X.dtype)
        if temporal_basis_bt is not None:
            Cxx_tdtd = apply_temporal_basis_tdtd(temporal_basis_bt, Cxx_tdtd)
            Cxy_tde = apply_temporal_basis_tde(temporal_basis_bt, Cxy_tde)

        W_etd = self.fit_bwd(Cxx_tdtd, Cxy_tde, self.reg)

        if temporal_basis_bt is not None:
            W_etd = invert_temporal_basis_mapping_spatiotemporal(temporal_basis_bt, W_etd)

        self.W_ = W_etd
        self.R_ = None
        if self.center:
            self.b_ = np.einsum("etd,d->e", self.W_, -muX)  # (e,)
        else:
            self.b_ = None

        # calibration parameters
        self.sigma0_, self.softmaxscale_ = (None, 3)

        return self


class BwdLinearRegressionCV(BwdLinearRegression):

    ''' Sequence 2 Sequence learning using backward linear regression  W*X = Y '''

    def __init__(self, inner_cv: int = 5, inner_cv_params: dict = None,
                 # explicit args so sklearn can autoset and clone
                 evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, reg=None, badEpThresh=6, center=True, compdiagcxx: bool = True,
                 **kwargs):
        super().__init__(evtlabs=evtlabs, tau=tau, offset=offset, tau_ms=tau_ms, offset_ms=offset_ms,
                         fs=fs, reg=reg, badEpThresh=badEpThresh, center=center, compdiagcxx=compdiagcxx, **kwargs)
        self.inner_cv, self.inner_cv_params = (inner_cv, inner_cv_params)

    def fit(self, X, Y, fs=None, inner_cv_params: dict = None, retrain_on_all: bool = True):
        """override fit, to do a cv_fit to optimize the hyper-params if inner_cv_params is given, otherwise a normal fit

        Args:
            X_TSd (_type_): The EEG data, with shape (T,S,d) = (number trials, number-samples in a trial, number electrodes)
            Y_TSye (_type_): The brain-trigger stimulus sequence.  with shape (T,S,y,e) = (number trials, number samples in trial, number outputs, number stimulus event types per output)
            fs ([type], optional): [description]. Defaults to None.
            inner_cv_params (dict, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if inner_cv_params is None:
            inner_cv_params = self.inner_cv_params
        if inner_cv_params is not None:  # inner CV for fit-params
            self.cv_fit(X, Y, fs=fs, cv=self.inner_cv, fit_params=inner_cv_params, retrain_on_all=retrain_on_all)
        else:
            super().fit(X, Y, fs=fs)
        return self

    def cv_fit(
            self, X, Y, cv=None, fs=None, verbose: bool = 0, return_estimator: bool = True, calibrate_softmax: bool = True,
            retrain_on_all: bool = True, fit_params: dict = None):
        ''' cross validated fit to the data.  optimized wrapper for optimization of the model rank.'''
        if cv is None:
            cv = self.inner_cv_params
        # fast path for cross validation over rank
        if fit_params is None or len(fit_params) == 0:
            # call the base version
            return super().cv_fit(self, X, Y, cv=cv, fs=fs, verbose=verbose, return_estimator=return_estimator,
                                  calibrate_softmax=calibrate_softmax, retrain_on_all=retrain_on_all)

        # first clean up the data..
        X, Y = zero_outliers(X, Y, self.badEpThresh)
        if self.center:
            # TODO[]: make work in-place w/o modify X
            X = X.copy()
            muX = np.mean(X.reshape((-1, X.shape[-1])), 0)
            X = X - muX

        # setup Y
        X_TSd = X
        Y_TSy = Y
        if X_TSd.ndim > 3:  # collapse additional feature dims into a single big one
            X_TSd = X_TSd.reshape(X_TSd.shape[:2]+(-1,))
        # map to event sequence
        Y_TSye = self.stim2event(Y_TSy, fit=True)  # (tr,samp,nY,e)

        # extract the true target to fit to, using horible slicing trick
        Y_true = Y_TSye[:, :, 0:1, ...]  # (tr,samp,e)

        # get the tau-offset info
        self.fit_tau_offset(X, Y, fs=fs)

        # compute summary statistics for each trial individually, to speed up the CV fits
        # compute the summary statistics for each trial
        Cxxs_tdtd = []
        Cxys_tde = []
        for ti in range(X.shape[0]):
            # get summary statistics
            if self.compdiagcxx:
                # get summary statistics
                # TODO[X]: store Cxx_tdtd in compact form as Cxx_tdd
                # (tau,d,tau,d) cross-cov shifted X with itself
                Cxx_tdd = compCxx_diag(X_TSd[ti:ti+1, ...], tau=self.tau_)
                Cxxs_tdtd.append(Cxx_tdd)

                Cxy_tyed, taus = compCyx_diag(
                    X_TSd[ti: ti + 1, ...],
                    Y_true[ti: ti + 1, ...],
                    tau=[1, self.tau_],
                    offset=self.offset_)  # (tau,d,1,e) shifted X with Y
                Cxy_tdey = np.moveaxis(Cxy_tyed, (0, 1, 2, 3), (0, 3, 2, 1))
                Cxy_tde = Cxy_tdey.reshape(Cxy_tdey.shape[:3])
                Cxys_tde.append(Cxy_tde)
            else:
                # (tau,d,tau,d) cross-cov shifted X with itself
                Cxx_tdtd = crossautocov(X_TSd[ti:ti+1, ...], X_TSd[ti:ti+1, ...], tau=(self.tau_, self.tau_))
                Cxxs_tdtd.append(Cxx_tdtd)

                Cxy_1dte = crossautocov(
                    X_TSd[ti: ti + 1, ...],
                    Y_true[ti: ti + 1, :, 0, :],
                    tau=[1, self.tau_],
                    offset=self.offset_)  # (tau,d,1,e) shifted X with Y
                Cxy_tde1 = np.moveaxis(Cxy_1dte, (0, 1, 2, 3), (3, 1, 0, 2))
                Cxy_tde = Cxy_tde1.reshape(Cxy_tde1.shape[:3])
                Cxys_tde.append(Cxy_tde)

        from sklearn.model_selection import ParameterGrid
        fit_configs = ParameterGrid(fit_params) if fit_params is not None else dict()

        if self.verb > 0:
            print("CV over {} configs: {}".format(len(fit_configs), fit_configs))

        cv = self.get_folding(X, Y, cv)
        if verbose > 0:
            print("CV:", end='')

        scores_cv = [[] for _ in fit_configs]  # list of lists, cv[params]
        for i, (train_idx, valid_idx) in enumerate(cv):
            if verbose > 0:
                print(".", end='', flush=True)
            #print("trn={} val={}".format(train_idx,valid_idx))
            if X[valid_idx, ...].size == 0:
                print("Warning: no-validation trials!!! using all data!")
                valid_idx = slice(X.shape[0])

            for ci, fit_config in enumerate(fit_configs):
                Cxx_tdtd = sum([Cxxs_tdtd[ti] for ti in train_idx])
                if self.compdiagcxx:
                    Cxx_tdtd = Cxx_diag2full(Cxx_tdtd)  # expand to the full version
                Cxy_tde = sum([Cxys_tde[ti] for ti in train_idx])
                # 1) fit with max-rank
                W_etd = self.fit_bwd(Cxx_tdtd, Cxy_tde, **fit_config)

                self.W_, self.R_ = (W_etd, None)
                self.b_ = np.einsum("etd,d->e", self.W_, -muX) if self.center else None

                # predict, forcing removal of copies of  tgt=0 so can score
                Fyi = self.predict(X[valid_idx, ...], Y[valid_idx, ...], dedup0=False)
                score = self.inner_score_audc(Fyi)

                # save the prediction for later
                if i == 0 and ci == 0:  # reshape Fy to include the extra model dim
                    Fy_cv = np.zeros((len(fit_configs), Y.shape[0])+Fyi.shape[1:], dtype=np.float32)
                Fy_cv[ci, valid_idx, ..., :Fyi.shape[-1]] = Fyi

                scores_cv[ci].append(score)

        # 3) get the *best* reg
        avescore = np.mean(np.array(scores_cv), axis=-1)  # (cfgs,folds) -> ranks
        if self.verb > 0:
            for ci, c in enumerate(fit_configs):
                print("Cfg {} = {:4.3f} ".format(c, avescore[ci]))
        maxci = np.argmax(avescore)
        opt_config = fit_configs[maxci]
        for k, v in opt_config.items():
            setattr(self, k, v)  # self.reg = regs[maxci]
        Fy = Fy_cv[maxci, ...]
        scores = scores_cv[maxci]
        print("     -> best={} = {:4.3f}".format(fit_configs[maxci], avescore[maxci]))

        # setup the p-value calibration
        self.sigma0_ = self.estimate_prior_sigma(Fy)
        self.softmaxscale_ = self.calibrate_softmaxscale(Fy) if calibrate_softmax else 3
        #print("Sigma0={:4.3f}  softmaxscale={:4.3f}".format(self.sigma0_, self.softmaxscale_))

        # retrain on all the data
        if retrain_on_all:
            Cxx_tdtd = sum(Cxxs_tdtd)
            if self.compdiagcxx:
                Cxx_tdtd = Cxx_diag2full(Cxx_tdtd)  # expand to the full version
            Cxy_tde = sum(Cxys_tde)
            # 1) fit with max-rank
            W_etd = self.fit_bwd(Cxx_tdtd, Cxy_tde)

            self.W_, self.R_ = (W_etd, None)
            self.b_ = np.einsum("etd,d->e", self.W_, -muX) if self.center else None

        return dict(
            estimator=Fy, score=scores_cv[maxci],
            estimator_cv=Fy_cv, scores_cv=scores_cv, fit_configs=[cf for cf in fit_configs])


# ----------------------------------------------------------
#
# LINEAR SKLEARN
#
# -----------------------------------------------------------

#from sklearn.linear_model import Ridge
class LinearSklearn(BaseSequence2Sequence):
    ''' Wrap a normal sk-learn classifier for sequence to sequence learning '''

    def __init__(
            self, clsfr="lr", clsfr_args: dict = {},
            evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, labelizeY=False,
            ignore_unlabelled=True, chunk_size: int = None, chunk_size_ms: int = None, chunk_labeller: str = None, badEpThresh=None,
            temporal_basis: str = None, score_type:str='audc', **kwargs):
        """  Wrap a normal sk-learn classifier for sequence to sequence learning 

        Args:
            clsfr (str|sklearn.model, optional): The base classifier to use.  If string then one-of; cca,bwd,fwd,ridge,lr,svr,svc,linearsklearn.  Defaults to "lr".
            clsfr_args (dict, optional): arguments to pass to the classifier constructor. Defaults to {}.
            evtlabs (_type_, optional): _description_. Defaults to None.
            evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
            tau (int, optional): the length in samples of the stimulus response. Defaults to None.
            offset (int, optional): offset from the event time for the response window. Defaults to None.
            tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
            offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
            fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
            chunk_size (int, optional): the number of samples between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            chunk_size_ms (float, optional): time in milliseconds between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            labelizeY (bool, optional): flag if whe should labelize-Y such that each unique combination of stimulus-event values is a new class. Defaults to False.
            ignore_unlabelled (bool, optional): flag if we should discard data-windows for which no stimulus events are non-zero. Defaults to True.
            chunk_labeller (_type_, optional): funtion name to use to convert from stimulus-event information to classes. Defaults to None.
            badEpThresh (int, optional): Bad epoch threshold for computing a robustified covariance matrix for the CCA fit.  See `utils.py::zero_outliers` for information on how the threshold is used, and `updateSummaryStatistics.py::updateSummaryStatistics` for usage in computing the summary statistics needed to compute a CCA model. Defaults to 6.
            temporal_basis (_type_, optional): As temporal basis is an array mapping from the raw sample temporal basis to a new parametric basis, such as the amplitude of a particular sinusoidal frequency. See `temporal_basis.py` for more information on the available types of basis. Defaults to None.
        """
        super().__init__(evtlabs=evtlabs, tau=tau, offset=offset, tau_ms=tau_ms, offset_ms=offset_ms, fs=fs, score_type=score_type, **kwargs)
        self.clsfr, self.clsfr_args, self.labelizeY, self.badEpThresh, self.ignore_unlabelled, self.chunk_size, self.chunk_size_ms, self.chunk_labeller, self.temporal_basis = (
            clsfr, clsfr_args, labelizeY, badEpThresh, ignore_unlabelled, chunk_size, chunk_size_ms, chunk_labeller, temporal_basis)

    @staticmethod
    def make_sklearn_dataset(
            X_TSd, Y_TSe, tau, offset, chunk_size=None, labelizeY=False, ignore_unlabelled=True, chunk_labeller=None,
            badEpThresh=None, temporal_basis: str = None, verb=0):
        """convert a sequence 2 sequence learning problem into a conventional classification problem, by treating temporal windows of X and Y as independent training examples.

        Args:
            X_TSd (_type_): The EEG data, with shape (T,S,d) = (number trials, number-samples in a trial, number electrodes)
            Y_TSe (_type_): The brain-trigger stimulus sequence.  with shape (T,S,e) = (number trials, number samples in trial, number stimulus event types per output)
                  when converting to a classification problem, each unique combination of stimulus-event-type values is treated as a different 'class'
            tau (float): response duration in samples
            offset (): offset of the window of X used to predict Y
            chunk_size (int, optional): the number of samples between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            labelizeY (bool, optional): flag if whe should labelize-Y such that each unique combination of stimulus-event values is a new class. Defaults to False.
            ignore_unlabelled (bool, optional): flag if we should discard data-windows for which no stimulus events are non-zero. Defaults to True.
            chunk_labeller (_type_, optional): funtion name to use to convert from stimulus-event information to classes. Defaults to None.
            temporal_basis ():
            badEpThresh (float): threshold for datection of bad epochs, i.e. whole trials, in numbers of standard deviations of excess power above that of the average trial.
            badWinThresh (_type_): threshold for detection of bad temporal-windows, i.e. 3s blocks of data, in numbers of standard deviations of excess power above an average temporal window.
            verb (int, optional): verbosity level. Defaults to 0.

        Raises:
            ValueError: _description_

        Returns:
            X_TS_td : X part of the training data, flattened into a 2d array in sklearn format.  This has shape (TS,td) = (number training examples, number features per example = response length * number-channels)
            Ytrn : Y part of the training data, flattended into sklearn format.  This has shape (TS,e) = (number training examples, number classes assuming hot-one encoding.)
        """
        '''fit 2 multi-dim time series: X = (tr, samp, d), Y = (tr, samp, e) using a given sklearn linear_model method'''
        if chunk_size is None:
            chunk_size = 1  # default to sliding window 1 sample at a time
        # slice X to get block of data per sample (with optional step-size)
        X_TStd = window_axis(X_TSd, winsz=tau, step=chunk_size,
                             axis=1) if tau is not None and tau < X_TSd.shape[1] else X_TSd[:, np.newaxis, ...]
        # TODO[]: include label offset....
        assert offset is None or offset == 0, 'Offset not yet supported'
        if chunk_labeller is None or chunk_labeller=='first':
            # just use the label from the first sample of each chunk
            Y_TSte = window_axis(Y_TSe, winsz=tau, step=chunk_size, axis=1)
            Y_TSe = Y_TSte[:,:,0,:]
        else:
            Y_TSte = window_axis(Y_TSe, winsz=tau, step=chunk_size, axis=1)
            # label is the max value in the chunk
            if chunk_labeller == 'mean':
                Y_TSe = np.mean(Y_TSte, axis=-2)
            elif chunk_labeller == 'max':
                Y_TSe = np.max(Y_TSte, axis=-2)
            elif chunk_labeller == 'median':
                Y_TSe = np.median(Y_TSte, axis=-2)
            # TODO[]: switch to use the median non-zero value
            elif callable(chunk_labeller):
                Y_TSe = chunk_labeller(Y_TSte)
            else:
                raise ValueError('Unknown chunk labeller')

        # clean up the data
        if badEpThresh is not None:
            X_TStd, Y_TSe = zero_outliers(X_TStd, Y_TSe, badEpThresh)

        temporal_basis = get_temporal_basis(tau, temporal_basis, X_TSd.dtype)
        X_TStd = apply_temporal_basis_X_TStd(temporal_basis, X_TStd)

        # convert to a 2d array: (tr*samp, feat) before giving to sklearn
        X_TS_td = np.reshape(X_TStd, (np.prod(X_TStd.shape[:2]), np.prod(X_TStd.shape[2:])))  # ((tr*samp-tau),(tau*d))
        Y_TS_e = np.reshape(Y_TSe, (np.prod(Y_TSe.shape[:2]), np.prod(Y_TSe.shape[2:])))  # ((tr*samp-tau),e)
        if labelizeY:  # convert Y to a label list + with chunk_size
            if Y_TS_e.shape[-1] > 1:  # multiple event types, event type is the label
                Y_TS = np.argmax(Y_TS_e, axis=-1)+1  # 1,2,3... for actual events
                Y_TS[np.all(Y_TS_e == 0, axis=-1)] = 0  # unlabelled samples have class 0
                #import matplotlib.pyplot as plt
                #plt.plot(Y_TS[300:600])
            else:  # level of the single event type is the label
                Y_TS = Y_TS_e[..., 0]
            if ignore_unlabelled:
                if np.any(Y_TS == 0):
                    keep = Y_TS != 0
                    if verb > 0:
                        print(
                            "Warning: removing {} unlabelled examples from {} training".format(
                                np.sum(unlabelled),
                                len(unlabelled)))
                    Y_TS = Y_TS[keep]
                    Y_TS_e = Y_TS_e[keep, ...]
                    X_TS_td = X_TS_td[keep, ...]
            Ytrn = Y_TS
        else:
            Ytrn = Y_TS_e

        if verb > 0:
            print("X={} Y={} Ylab={}".format(X_TS_td.shape, Ytrn.shape, np.unique(Ytrn)))
        return X_TS_td, Ytrn, temporal_basis

    @staticmethod
    def sklearn_fit(X_TSd, Y_TSe, clsfr, tau, offset, chunk_size=1, labelizeY=False, ignore_unlabelled=True,
                    chunk_labeller=None, badEpThresh=None, temporal_basis: str = None, verb=0, **fit_params):
        """convert a sequence 2 sequence learning problem into a conventional classification problem, by treating temporal windows of X and Y as independent training examples.

        Args:
            X_TSd (_type_): The EEG data, with shape (T,S,d) = (number trials, number-samples in a trial, number electrodes)
            Y_TSe (_type_): The brain-trigger stimulus sequence.  with shape (T,S,e) = (number trials, number samples in trial, number stimulus event types per output)
                  when converting to a classification problem, each unique combination of stimulus-event-type values is treated as a different 'class'
            clsfr (str): name of the classifier to use as a string, or a classification class.
            tau (float): response duration in samples
            offset (): offset of the window of X used to predict Y
            chunk_size (int, optional): the number of samples between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            labelizeY (bool, optional): flag if whe should labelize-Y such that each unique combination of stimulus-event values is a new class. Defaults to False.
            ignore_unlabelled (bool, optional): flag if we should discard data-windows for which no stimulus events are non-zero. Defaults to True.
            chunk_labeller (_type_, optional): funtion name to use to convert from stimulus-event information to classes. Defaults to None.
            temporal_basis ():
            badEpThresh (float): threshold for datection of bad epochs, i.e. whole trials, in numbers of standard deviations of excess power above that of the average trial.
            badWinThresh (_type_): threshold for detection of bad temporal-windows, i.e. 3s blocks of data, in numbers of standard deviations of excess power above an average temporal window.
            verb (int, optional): verbosity level. Defaults to 0.

        Raises:
            ValueError: _description_

        Returns:
            clsfr - the fitted classifier
        """
        Xtrn, Ytrn, temporal_basis = LinearSklearn.make_sklearn_dataset(
            X_TSd, Y_TSe, tau, offset, chunk_size, labelizeY, ignore_unlabelled, chunk_labeller, badEpThresh, temporal_basis, verb)
        # BODGE: allow set C at fit time!
        clsfr.set_params(**fit_params)
        clsfr.fit(Xtrn, Ytrn)

        print(clsfr.coef_.shape)
        # log the binary training set performance.
        if verb > 0:
            print("Training score = {}".format(clsfr.score(Xtrn, Ytrn)))
        return clsfr

    def make_clsfr(self, clsfr, clsfr_args, C=1):
        """make a sklear classifer from the given string name and arguments

        Args:
            clsfr (str): Name of the classifer as a string. Current supported classifiers are: ridge, lr, svr, svc
            clsfr_args (dict): dictionary of agruments to pass to the constructed classifier
            C (int, optional): classifier regularization parameter, where higher values mean more regularization.  (Note: this is inverted as needed for 'lambda' type classifiers). Defaults to 1.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            clsfr: initialized  classifier object
        """
        C = clsfr_args.pop('C', C)
        if isinstance(clsfr, str):
            if clsfr.lower() == 'ridge':  # should be equivalent to BwdLinearRegression
                if not 'fit_intercept' in clsfr_args:
                    clsfr_args['fit_intercept'] = True
                clsfr = Ridge(alpha=1/C, **clsfr_args)
            elif clsfr.lower() == 'lr':
                if not 'solver' in clsfr_args:
                    clsfr_args['solver'] = 'lbfgs'
                if not 'max_iter' in clsfr_args:
                    clsfr_args['max_iter'] = 250
                if not 'fit_intercept' in clsfr_args:
                    clsfr_args['fit_intercept'] = True
                clsfr = LogisticRegression(C=C, **clsfr_args)
            elif clsfr.lower() == 'svr':
                clsfr = LinearSVR(C=C, **clsfr_args)
            elif clsfr.lower() == 'svc':
                clsfr = LinearSVC(C=C, **clsfr_args)
            else:
                raise ValueError("Unrecognised sklearn classifier string")
        else:
            if hasattr(clsfr, 'C'):
                clsfr.set_params(C=C)
            elif hasattr(clsfr, 'alpha'):
                clsfr.set_params(alpha=1/C)
            else:
                raise ValueError("Don't know how to set C")
        return clsfr

    def fit(self, X, y, fs=None, C: float = 1, C_scale: float = None, **fit_params):
        """fit a sklearn classifier to the given sequence-2-sequence problem

        Args:
            X_TSd (_type_): The EEG data, with shape (T,S,d) = (number trials, number-samples in a trial, number electrodes)
            Y_TSye (_type_): The brain-trigger stimulus sequence.  with shape (T,S,y,e) = (number trials, number samples in trial, number outputs, number stimulus event types per output)
            fs ([type], optional): sample rate of the data, for plotting. Defaults to None.
            C (float): the regularization strength
            C_scale (float): scaling constant for the regularization strength
            fit_params (dict): additional parameters to pass to the classifier constructor

        Returns:
            [type]: [description]
        """

        if C_scale is None:
            C_scale = self.C_scale_estimate(X)
            #print("C_scale = {}".format(C_scale))

        self.clsfr = self.make_clsfr(self.clsfr, self.clsfr_args, C=C*C_scale)

        self.fit_tau_offset(X, y, fs=fs)
        # map to event sequence
        Y = self.stim2event(y, fit=True)
        # extract the true target to fit to
        Y_true = Y[..., 0, :]  # (tr,samp,e)

        # get the chunksize for the classification windows
        self.chunk_size_ = int(self.chunk_size_ms * self.fs_ // 1000) if self.chunk_size_ms is not None else self.chunk_size
        # transform the data into a sk-learn compatiable version
        Xtrn, Ytrn, temporal_basis = LinearSklearn.make_sklearn_dataset(
            X, Y_true, tau=self.tau_, offset=self.offset_, labelizeY=self.labelizeY,
            ignore_unlabelled=self.ignore_unlabelled, badEpThresh=self.badEpThresh, 
            chunk_size=self.chunk_size_, chunk_labeller=self.chunk_labeller, 
            temporal_basis=self.temporal_basis)

        # fit the model
        # BODGE: allow set C at fit time!
        self.clsfr.set_params(**fit_params)
        self.clsfr.fit(Xtrn, Ytrn)

        #self.clsfr = LinearSklearn.sklearn_fit(X, Y_true, self.clsfr, tau=self.tau_, offset=self.offset_, labelizeY=self.labelizeY, ignore_unlabelled=self.ignore_unlabelled, badEpThresh=self.badEpThresh, chunk_size=self.chunk_size, chunk_labeller=self.chunk_labeller, **fit_params)

        # extract the solution coefficient and store in W_
        W = self.clsfr.coef_  # ( n_class, n_feat ) = (e,(tau*d)))
        b = self.clsfr.intercept_  # (e,1)
        # print("W={}".format(W.shape))
        # print("b={}".format(b.shape))
        W_etd = np.reshape(W, (W.shape[0] if W.ndim > 1 else 1, -1) + X.shape[2:])  # (e,tau,d)
        # print("W={}".format(W.shape))

        # TODO[]: support inversion of the temporal bassis
        W_etd = invert_temporal_basis_mapping_spatiotemporal(temporal_basis, W_etd)

        # ensure the weights vector is the right size ...
        if self.labelizeY and not W_etd.shape[0] == Y.shape[-1]:

            if W_etd.shape[0] > Y.shape[-1]:
                print("Warning: More weights than outputs! Y={} W={} removed".format(Y.shape[-1], W_etd.shape[0]))
                W_etd = W_etd[:Y.shape[-1], ...]
                b = b[:Y.shape[-1]]
            elif W_etd.shape[0] == 1 and Y.shape[-1] == 2:
                if self.verb > 0:
                    print("Fewer weights than outputs! Y={} W={} binary".format(Y.shape[-1], W_etd.shape[0]))
                # Y is binary, so e==0 -> neg-class,  e==1 -> pos-class
                # TODO[]: check?
                W_etd = np.concatenate((-W_etd, W_etd), axis=0)
                b = np.concatenate((-b, b), axis=0)
            else:
                raise ValueError("Don't know what to do with W: W={}  Y={}".format(W_etd.shape, Y.shape))

        # print("W={}".format(W.shape))

        self.W_ = W_etd
        self.R_ = None
        self.b_ = b
        # print("b={}".format(b))

        # set calibration parameters
        # calibration parameters
        self.sigma0_, self.softmaxscale_ = (None, 3)

        # print(self.clsfr.coef_.shape)
        # log the binary training set performance.
        if self.verb > 0:
            print("Training score = {}".format(self.score(Xtrn, Ytrn)))

        return self

    def C_scale_estimate(self, X):
        """estimate a scaling for the given data such that the 'optimal' regularization strength is near 1.

        Args:
            X (_type_): the input data

        Returns:
            _type_: _description_
        """
        if X.ndim < 3:
            X = X.reshape((1,)*(3-X.ndim) + X.shape)
        feat_mu = np.mean(X, axis=(0, 1))
        feat_X2 = np.mean(X**2, axis=(0, 1))
        feat_var = feat_X2 - feat_mu**2
        c_scale = np.sqrt(np.median(feat_var))
        return float(c_scale)  # make simple scalar

    def cv_fit(
            self, X, Y, fs=None, cv=None, fit_params: dict = dict(),
            verbose: bool = 0, return_estimator: bool = True, calibrate_softmax: bool = True, retrain_on_all: bool = True,
            C_scale: float = None):
        ''' override CV-fit to optimize the regularization parameter'''
        if cv is None:
            cv = self.inner_cv_params
        # fast path for cross validation over rank
        cv_in = cv.copy() if hasattr(cv, 'copy') else cv  # save copy of cv info for later
        if fit_params is None:
            # call the base version
            return BaseSequence2Sequence.cv_fit(
                self, X, Y, cv=cv_in, fs=fs, fit_params=fit_params, verbose=verbose, return_estimator=return_estimator,
                calibrate_softmax=calibrate_softmax, retrain_on_all=retrain_on_all)

        cv = self.get_folding(X, Y, cv)
        if verbose > 0:
            print("CV:", end='')

        # estimate C_scaling if not given & combine with the Cs
        if C_scale is None:
            C_scale = self.C_scale_estimate(X)
            if verbose > 0 : print("C_scale = {}".format(C_scale))

        from sklearn.model_selection import ParameterGrid
        fit_configs = ParameterGrid(fit_params) if fit_params is not None else dict()
        if self.verb > 0:
            print("CV over {} configs: {}".format(len(fit_configs), fit_configs))

        # TODO[]: pre-transform the dataset to clsfr training dataset

        scores_cv = [[] for _ in fit_configs]  # list of lists, cv[params]
        for i, (train_idx, valid_idx) in enumerate(cv):
            if verbose > 0:
                print(".", end='', flush=True)
            #print("trn={} val={}".format(train_idx,valid_idx))
            if X[valid_idx, ...].size == 0:
                print("Warning: no-validation trials!!! using all data!")
                valid_idx = slice(X.shape[0])

            for ci, fit_config in enumerate(fit_configs):

                LinearSklearn.fit(self, X[train_idx, ...], Y[train_idx, ...], fs=fs, C_scale=C_scale, **fit_config)

                # predict, forcing removal of copies of  tgt=0 so can score
                Fyi = self.predict(X[valid_idx, ...], Y[valid_idx, ...], dedup0=False)
                # score with the wanted score function
                if self.score_type == 'audc': # Fast-path for audc score
                    score = self.inner_score_audc(Fyi)
                else:
                    score = self.score(X[valid_idx, ...], Y[valid_idx, ...])
                # save the prediction for later
                if i == 0 and ci == 0:  # reshape Fy to include the extra model dim
                    Fy_cv = np.zeros((len(fit_configs), Y.shape[0])+Fyi.shape[1:], dtype=np.float32)
                Fy_cv[ci, valid_idx, ..., :Fyi.shape[-1]] = Fyi

                scores_cv[ci].append(score)

        # 3) get the *best* reg
        avescore = np.mean(np.array(scores_cv), axis=-1)  # (cfgs,folds) -> ranks
        if self.verb > 0:
            for ci, c in enumerate(fit_configs):
                print("Cfg {} = {:4.3f} ".format(c, avescore[ci]))
        maxci = np.argmax(avescore)
        opt_config = fit_configs[maxci]
        for k, v in opt_config.items():
            setattr(self, k, v)  # self.reg = regs[maxci]
        Fy = Fy_cv[maxci, ...]
        self.score_cv_ = scores = scores_cv[maxci]
        if self.verb>=0 :
            print("     -> best={} = {:4.3f}".format(fit_configs[maxci], avescore[maxci]))

        # setup the p-value calibration
        self.sigma0_ = self.estimate_prior_sigma(Fy)
        self.softmaxscale_ = self.calibrate_softmaxscale(Fy) if calibrate_softmax else 3
        #print("Sigma0={:4.3f}  softmaxscale={:4.3f}".format(self.sigma0_, self.softmaxscale_))

        # retrain on all the data
        if retrain_on_all:
            LinearSklearn.fit(self, X, Y)

        return dict(
            estimator=Fy, score=scores_cv[maxci],
            estimator_cv=Fy_cv, scores_cv=scores_cv, fit_configs=[cf for cf in fit_configs])

# ALiases for common configurations


class LinearLogisticRegression(LinearSklearn):
    """Alias for fitting a normal sklearn logistic regression classifier

    Args:
        LinearSklearn (_type_): _description_
    """

    def __init__(
            self, evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, labelizeY=True,
            ignore_unlabelled=True, badEpThresh=None, score_type:str='audc', **kwargs):
        """alias for fitting a normal sklearn logistic regression classifier

        Args:
            evtlabs (_type_, optional): _description_. Defaults to None.
            evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
            tau (int, optional): the length in samples of the stimulus response. Defaults to None.
            offset (int, optional): offset from the event time for the response window. Defaults to None.
            tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
            offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
            fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
            chunk_size (int, optional): the number of samples between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            chunk_size_ms (float, optional): time in milliseconds between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            labelizeY (bool, optional): flag if whe should labelize-Y such that each unique combination of stimulus-event values is a new class. Defaults to False.
            ignore_unlabelled (bool, optional): flag if we should discard data-windows for which no stimulus events are non-zero. Defaults to True.
            chunk_labeller (_type_, optional): funtion name to use to convert from stimulus-event information to classes. Defaults to None.
            badEpThresh (int, optional): Bad epoch threshold for computing a robustified covariance matrix for the CCA fit.  See `utils.py::zero_outliers` for information on how the threshold is used, and `updateSummaryStatistics.py::updateSummaryStatistics` for usage in computing the summary statistics needed to compute a CCA model. Defaults to 6.
            temporal_basis (_type_, optional): As temporal basis is an array mapping from the raw sample temporal basis to a new parametric basis, such as the amplitude of a particular sinusoidal frequency. See `temporal_basis.py` for more information on the available types of basis. Defaults to None.
            score_type (str, optional): Type of scoring function to use. One of: 'audc','corr', 'classifier', callable. Defaults to 'audc'.
            ignore_unlabelled (bool, optional): _description_. Defaults to True.
            badEpThresh (_type_, optional): _description_. Defaults to None.
        """        
        super().__init__(clsfr='lr', evtlabs=evtlabs, tau=tau, offset=offset, tau_ms=tau_ms, offset_ms=offset_ms,
                         fs=fs, labelizeY=labelizeY, ignore_unlabelled=ignore_unlabelled, badEpThresh=badEpThresh, score_type=score_type, **kwargs)


class LinearLogisticRegressionCV(LinearSklearn):
    """alias for fitting a normal sklearn logistic regression classifier with automatic inner cross-valiation to optimize the regularization strength

    Args:
        LinearSklearn (_type_): _description_
    """

    def __init__(
            self, evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, labelizeY=True,
            score_type:str='audc', 
            ignore_unlabelled=True, badEpThresh=None, inner_cv=5,
            inner_cv_params: dict = {"C": [8 ** v for v in [0, 1, 2, 4, 6, 10, 16]]},
            **kwargs):
        """alias for fitting a normal sklearn logistic regression classifier with automatic inner cross-valiation to optimize the regularization strength

        Args:
            evtlabs (_type_, optional): _description_. Defaults to None.
            evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
            tau (int, optional): the length in samples of the stimulus response. Defaults to None.
            offset (int, optional): offset from the event time for the response window. Defaults to None.
            tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
            offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
            fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
            chunk_size (int, optional): the number of samples between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            chunk_size_ms (float, optional): time in milliseconds between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            labelizeY (bool, optional): flag if whe should labelize-Y such that each unique combination of stimulus-event values is a new class. Defaults to False.
            ignore_unlabelled (bool, optional): flag if we should discard data-windows for which no stimulus events are non-zero. Defaults to True.
            chunk_labeller (_type_, optional): funtion name to use to convert from stimulus-event information to classes. Defaults to None.
            badEpThresh (int, optional): Bad epoch threshold for computing a robustified covariance matrix for the CCA fit.  See `utils.py::zero_outliers` for information on how the threshold is used, and `updateSummaryStatistics.py::updateSummaryStatistics` for usage in computing the summary statistics needed to compute a CCA model. Defaults to 6.
            temporal_basis (_type_, optional): As temporal basis is an array mapping from the raw sample temporal basis to a new parametric basis, such as the amplitude of a particular sinusoidal frequency. See `temporal_basis.py` for more information on the available types of basis. Defaults to None.
            score_type (str, optional): Type of scoring function to use. One of: 'audc','corr', 'classifier', callable. Defaults to 'audc'.
            ignore_unlabelled (bool, optional): _description_. Defaults to True.
            badEpThresh (_type_, optional): _description_. Defaults to None.
            inner_cv (int, optional): Number of inner cross validation folds to use. Defaults to 5.
            inner_cv_params (_type_, optional): parameters to evaluate during the inner cross-validation. Defaults to {"C": [8 ** v for v in [0, 1, 2, 4, 6, 10, 16]]}.
        """        
        super().__init__(clsfr='lr', evtlabs=evtlabs, tau=tau, offset=offset, tau_ms=tau_ms, offset_ms=offset_ms,
                         fs=fs, labelizeY=labelizeY, ignore_unlabelled=ignore_unlabelled, badEpThresh=badEpThresh, score_type=score_type, **kwargs)
        self.inner_cv, self.inner_cv_params = (inner_cv, inner_cv_params)

    def fit(self, X, Y, fs=None, inner_cv_params: dict = None, retrain_on_all: bool = True):
        """override fit, to do a cv_fit to optimize the hyper-params if inner_cv_params is given, otherwise a normal fit

        Args:
            X ([type]): [description]
            Y ([type]): [description]
            fs ([type], optional): [description]. Defaults to None.
            inner_cv_params (dict, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if inner_cv_params is None:
            inner_cv_params = self.inner_cv_params
        if inner_cv_params is not None:  # inner CV for fit-params
            super().cv_fit(X, Y, fs=fs, cv=self.inner_cv, fit_params=inner_cv_params, retrain_on_all=retrain_on_all)
        else:
            super().fit(X, Y, fs=fs)
        return self

    def cv_fit(self, X, Y, fit_params: dict = dict(), **kwargs):
        if fit_params is None:
            fit_params = self.inner_cv_params
        else:  # override with input parameters
            tmp = copy.deepcopy(self.inner_cv_params)
            tmp.update(fit_params)
            fit_params = tmp
        return super().cv_fit(X, Y, fit_params=fit_params, **kwargs)


class LinearRidgeRegression(LinearSklearn):
    """Alias for fitting a normal sklearn ridge regression classifier

    Args:
        LinearSklearn (_type_): _description_
    """

    def __init__(
            self, evtlabs=None, tau=None, offset=None, tau_ms=None, offset_ms=0, fs=None, labelizeY=False,
            ignore_unlabelled=False, badEpThresh=None, **kwargs):
        """alias for fitting a normal sklearn ridge regression classifier

        Args:
            evtlabs (_type_, optional): _description_. Defaults to None.
            evtlabs ([ListStr,optional]): the event types to use for model fitting.  See `stim2event.py` for the support event types. Defautls to ('fe','re').
            tau (int, optional): the length in samples of the stimulus response. Defaults to None.
            offset (int, optional): offset from the event time for the response window. Defaults to None.
            tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
            offset_ms (float, optional): the offset from the event time for the start of the response window.  Defaults to None.
            fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
            chunk_size (int, optional): the number of samples between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            chunk_size_ms (float, optional): time in milliseconds between time-windows making a new classification problem. Defaults to 1, which means every sample is a new example.
            labelizeY (bool, optional): flag if whe should labelize-Y such that each unique combination of stimulus-event values is a new class. Defaults to False.
            ignore_unlabelled (bool, optional): flag if we should discard data-windows for which no stimulus events are non-zero. Defaults to True.
            chunk_labeller (_type_, optional): funtion name to use to convert from stimulus-event information to classes. Defaults to None.
            badEpThresh (int, optional): Bad epoch threshold for computing a robustified covariance matrix for the CCA fit.  See `utils.py::zero_outliers` for information on how the threshold is used, and `updateSummaryStatistics.py::updateSummaryStatistics` for usage in computing the summary statistics needed to compute a CCA model. Defaults to 6.
            temporal_basis (_type_, optional): As temporal basis is an array mapping from the raw sample temporal basis to a new parametric basis, such as the amplitude of a particular sinusoidal frequency. See `temporal_basis.py` for more information on the available types of basis. Defaults to None.
            score_type (str, optional): Type of scoring function to use. One of: 'audc','corr', 'classifier', callable. Defaults to 'audc'.
            ignore_unlabelled (bool, optional): _description_. Defaults to True.
            badEpThresh (_type_, optional): _description_. Defaults to None.
        """        
        super().__init__(clsfr='ridge', evtlabs=evtlabs, tau=tau, offset=offset, tau_ms=tau_ms,
                         offset_ms=offset_ms, fs=fs, labelizeY=False, ignore_unlabelled=False, badEpThresh=None,  **kwargs)


def visualize_Fy_Py(Fy, Py):
    """visualize the Fy and Py in a nice plot

    Args:
        Fy (_type_): _description_
        Py (_type_): _description_
    """
    import numpy as np
    import matplotlib.pyplot as plt
    plt.ion()
    for trli in range(Fy.shape[0]):
        print('{})'.format(trli), flush=True)
        plt.clf()
        plt.subplot(211)
        plt.plot(np.cumsum(Fy[trli, :, :], -2))
        plt.subplot(212)
        plt.plot(Py[trli, :, :])
        plt.title("{})".format(trli))
        plt.show(block=False)
        try:
            plt.pause(1)
        except:
            break


def plot_model_weights(model: BaseSequence2Sequence, ch_names=None):
    """plot the weights of a fitted model

    Args:
        model (BaseSequence2Sequence): _description_
        ch_names (_type_, optional): list of channel names used to make a topographic plot of the sptial information if given. Defaults to None.
    """
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_erp
    from mindaffectBCI.decoder.scoreStimulus import factored2full
    plot_erp(factored2full(model.W_, model.R_), ch_names=ch_names, evtlabs=model.evtlabs, offset=model.offset)


def testLeadLag():
    import numpy as np
    from mindaffectBCI.decoder.utils import testSignal
    from mindaffectBCI.decoder.model_fitting import MultiCCA
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
    import matplotlib.pyplot as plt

    irf = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    offset = 0  # X->lag-by-10
    X, Y, st, A, B = testSignal(nTrl=1, nSamp=500, d=10, nE=1, nY=30, isi=10,
                                tau=10, offset=offset, irf=irf, noise2signal=0)

    # reference case lagged response test
    evtlabs = None
    tau = 10
    cca_offset = 0
    cca = MultiCCA(evtlabs=evtlabs, tau=tau, offset=cca_offset)
    scores = cca.cv_fit(X, Y)
    Fy = scores['estimator']
    (_) = decodingCurveSupervised(Fy)

    # leading X
    irf = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    offset = -9  # X leads by 9
    X, Y, st, A, B = testSignal(nTrl=1, nSamp=5000, d=10, nE=1, nY=30, isi=10,
                                tau=10, offset=offset, irf=irf, noise2signal=0)
    plt.figure(0)
    plt.clf()
    plt.plot(X[0, :, 0], label='X')
    plt.plot(Y[0, :, 0, 0], label='Y')
    plt.title("offset={}, irf={}".format(offset, irf))
    plt.legend()

    # no-shift in analysis window
    evtlabs = None
    tau = 10
    cca_offset = 0
    cca = MultiCCA(evtlabs=evtlabs, tau=tau, offset=cca_offset)
    scores = cca.cv_fit(X, Y)
    Fy = scores['estimator']
    (_) = decodingCurveSupervised(Fy)

    # shifted analysis window
    evtlabs = None
    tau = 20
    cca_offset = -9
    cca = MultiCCA(evtlabs=evtlabs, tau=tau, offset=cca_offset)
    scores = cca.cv_fit(X, Y)
    Fy = scores['estimator']
    (_) = decodingCurveSupervised(Fy)

    plt.figure(1)
    plt.clf()
    plot_model_weights(cca)


def testcase(dataset='toy', loader_args=dict()):
    """test function for testing basic operation of the classifiers

    Args:
        dataset (str, optional): _description_. Defaults to 'toy'.
        loader_args (_type_, optional): _description_. Defaults to dict().
    """
    from mindaffectBCI.decoder.model_fitting import MultiCCA, FwdLinearRegression, BwdLinearRegression, LinearSklearn
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
    from mindaffectBCI.decoder.offline.datasets import get_dataset
    from mindaffectBCI.decoder.updateSummaryStatistics import plot_trial
    import matplotlib.pyplot as plt
    import time

    loadfn, filenames, dataroot = get_dataset(dataset)
    if dataset == 'toy':
        loader_args = dict(tau=10, isi=5, noise2signal=.1, nTrl=200, nSamp=200, nE=1, irf=[0, 0, 1, 4, 1, 0, 0])
    X_TSd, Y_TSy, coords = loadfn(filenames[0], **loader_args)
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']
    plot_trial(X_TSd[:2], Y_TSy[:2], fs=fs)

    # raw
    tau = int(.3*fs)
    offset = 0
    evtlabs = ('re')  # ('re', 'ntre')#  ('re', 'ntre', 'rest')#  ('1', '0')#
    # classifiers, need exclusive labels and both positive and negative examples.
    # so use: re, ntre as event codes.
    clsfrevtlabs = ('re', 'ntre')

    def fit_predict_plot(cca, X_TSd, Y_TSy, fs, title: str = None):
        print('clf = {}'.format(cca))
        cca.fit(X_TSd, Y_TSy, fs=fs)
        Fy = cca.predict(X_TSd, Y_TSy, dedup0=True)
        print("score={}".format(cca.score(X_TSd, Y_TSy)))

        plt.figure()
        cca.plot_confusion_matrix(X_TSd, Y_TSy)
        plt.figure()
        cca.plot_decoding_curve(X_TSd, Y_TSy)
        plt.figure()
        cca.plot_model()
        if title:
            plt.title(title)
        plt.show()

    # cca = MultiCCA(tau=tau, offset=offset, evtlabs=evtlabs, reg=None)#(1e-4,1e-4))
    # fit_predict_plot(cca,X_TSd,Y_TSy, fs, title='raw')

    # ccaw = MultiCCA(tau=tau, offset=offset, evtlabs=evtlabs, temporal_basis='resources/cvep_temporal_basis.json' )
    # fit_predict_plot(ccaw,X_TSd,Y_TSy, fs, title='cvep basis')

    # ccaw = MultiCCA(tau=tau, offset=offset, evtlabs=evtlabs, temporal_basis='edge_suppress.25' )
    # fit_predict_plot(ccaw,X_TSd,Y_TSy, fs, title='edge_suppress .25')

    # ccaw = MultiCCACV(tau=tau, offset=offset, evtlabs=evtlabs, temporal_basis="rbf1")
    # fit_predict_plot(ccaw,X_TSd,Y_TSy, fs, title='rbf1')

    # ccaw = MultiCCA(tau=tau, offset=offset, evtlabs=evtlabs, temporal_basis="fourier10")
    # fit_predict_plot(ccaw,X_TSd,Y_TSy, fs, title='fourier5')

    lr = LinearLogisticRegressionCV(tau=tau, offset=offset, evtlabs=evtlabs, labelizeY=True,
                                  ignore_unlabelled=False, temporal_basis='winfourier4', score_type='classifier')
    # lr = LinearLogisticRegressionCV(tau=tau, offset=offset, evtlabs=evtlabs, labelizeY=True,
    #                                 clsfr_args={"max_iter":200,"solver":"lbfgs"}, temporal_basis='winfourier10',
    #                                 inner_cv=5, inner_cv_params={"C":[8**v for v in [-6,-4,-2,-1,0,1,2]]})
    fit_predict_plot(lr, X_TSd, Y_TSy, fs)

    bwd = BwdLinearRegression(tau=tau, evtlabs=evtlabs, verb=2, reg=None, badEpThresh=4,
                              compdiagcxx=True, temporal_basis='winfourier5')
    fit_predict_plot(bwd, X_TSd, Y_TSy, fs, title='bwd')


    plt.show(block=True)

    # ttc=init_clsfr('mindaffectBCI.decoder.temporal_transfer_model.TemporalTransferCCA',tau=tau,offset=offset,evtlabs=evtlabs)
    # ttc.fit(X_TSd,Y_TSy)

    # cca - cv-fit
    print("CV fitted")
    cca = MultiCCA(tau=tau, rank=1, evtlabs=evtlabs)
    cv_res = cca.cv_fit(X_TSd, Y_TSy, ranks=(1, 2, 3, 5), inner_cv_params=dict())
    Fy = cv_res['estimator']
    (_) = decodingCurveSupervised(Fy, priorsigma=(cca.sigma0_, cca.priorweight))

    # cca - cv-fit
    print("CV fitted")
    cca = MultiCCACV(tau=tau, offset_ms=50, rank=1, reg=None, evtlabs=evtlabs, verb=1, nvirt_out=None,
                     inner_cv_params=dict(rank=(1, 2, 3, 5), reg=((1e-4, 1e-4), (1e-3, 1e-3), (1e-2, 1e-2), (1e-1, 1e-1))))
    fit_predict_plot(cca, X_TSd, Y_TSy, fs)

    from mindaffectBCI.decoder.model_fitting import MultiCCA, FwdLinearRegression, BwdLinearRegression
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics
    from mindaffectBCI.decoder.scoreStimulus import factored2full
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
    rank = 1
    evtlabs = ('re', 'fe')  # ('re', 'ntre') # ('re', 'fe')  # ('1', '0') #
    # cca
    cca = MultiCCA(tau=tau, offset=offset, rank=rank, evtlabs=evtlabs, reg=None)
    fit_predict_plot(cca, X_TSd, Y_TSy, fs)

    print('lr-cv')
    lr = LinearLogisticRegressionCV(tau=tau, offset=offset, evtlabs=clsfrevtlabs, labelizeY=True,
                                    clsfr_args={"max_iter": 200, "solver": "lbfgs"}, temporal_basis='winfourier10',
                                    inner_cv=5, inner_cv_params={"C": [8**v for v in [-6, -4, -2, -1, 0, 1, 2]]})
    fit_predict_plot(lr, X_TSd, Y_TSy, fs)

    # fwd-model
    print("Forward Model")
    fwd = FwdLinearRegression(tau=tau, evtlabs=evtlabs, reg=None, badEpThresh=4)
    fit_predict_plot(fwd, X_TSd, Y_TSy, fs)

    # bwd-model
    print("Backward Model")
    t0 = time.time()
    bwd = BwdLinearRegressionCV(
        tau=tau, evtlabs=evtlabs, verb=2, reg=None, badEpThresh=4, compdiagcxx=True,
        inner_cv_params=dict(reg=(0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, .5, 1 - 1e-1, 1 - 1e-2)))
    fit_predict_plot(bwd, X_TSd, Y_TSy, fs)
    print("in {}s".format(time.time()-t0))

    # bwd-model
    print("Backward Model")
    t0 = time.time()
    bwd = BwdLinearRegressionCV(tau=tau, evtlabs=evtlabs, verb=2, reg=None, badEpThresh=4,
                                inner_cv_params=dict(reg=(1e-4, 1e-3, 1e-2, 1e-1, .5, 1-1e-1, 1-1e-2)))
    fit_predict_plot(bwd, X_TSd, Y_TSy, fs)
    print("in {}s".format(time.time()-t0))

    Py = bwd.predict_proba(X_TSd, Y_TSy, dedup0=True)
    Py = bwd.predict_proba(X_TSd, Y_TSy, dedup0=True)
    #visualize_Fy_Py(Fy, Py)

    # sklearn wrapper
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.svm import LinearSVR, LinearSVC
    print("sklearn-ridge")
    ridge = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=Ridge(alpha=0))
    fit_predict_plot(ridge, X_TSd, Y_TSy, fs)

    print("sklear-lr")
    lr = LinearSklearn(tau=tau, offset=offset, evtlabs=clsfrevtlabs, clsfr=LogisticRegression(
        C=1, multi_class='multinomial', solver='sag'), labelizeY=True)
    fit_predict_plot(lr, X_TSd, Y_TSy, fs)

    print("sklear-svc")
    svc = LinearSklearn(tau=tau, offset=offset, evtlabs=clsfrevtlabs,
                        clsfr=LinearSVC(C=1, multi_class='ovr'), labelizeY=True)
    fit_predict_plot(svc, X_TSd, Y_TSy, fs)

    # hyper-parameter optimization with cross-validation
    from sklearn.model_selection import GridSearchCV
    tuned_parameters = {
        'rank': [1, 2, 3, 5],
        'tau': [int(dur * fs) for dur in [.2, .3, .5, .7]],
        'evtlabs': [['re', 'fe'],
                    ['re', 'ntre'],
                    ['0', '1']]}
    cv_cca = GridSearchCV(MultiCCA(), tuned_parameters,
                          n_jobs=1,
                          refit=False)  # scoring=decoding_curve_scorer,
    cv_cca.fit(X_TSd, Y_TSy)
    print("CVOPT:\n\n{} = {}\n".format(cv_cca.best_estimator_, cv_cca.best_score_))
    means = cv_cca.cv_results_['mean_test_score']
    stds = cv_cca.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, cv_cca.cv_results_['params']):
        print("{:5.3f} (+/-{:5.3f}) for {}".format(mean, std * 2, params))
    print()

    # use the best setting to fit a normal model, and get it's cv estimated predictions
    cca = MultiCCA()
    cca.set_params(**cv_cca.best_params_)  # Note: **dict -> k,v argument array
    cv_res = cca.cv_fit(X_TSd, Y_TSy)
    Fy = cv_res['estimator']
    (_) = decodingCurveSupervised(Fy)

    # hyper-parameter optimization with cross-validation
    from sklearn.model_selection import GridSearchCV
    tuned_parameters = {'clsfr': [LogisticRegression(C=1e-2, multi_class='multinomial', solver='lbfgs'),
                                  LogisticRegression(C=1e-1, multi_class='multinomial', solver='lbfgs'),
                                  LogisticRegression(C=1e0, multi_class='multinomial', solver='lbfgs'),
                                  LogisticRegression(C=1e1, multi_class='multinomial', solver='lbfgs'),
                                  LogisticRegression(C=1e2, multi_class='multinomial', solver='lbfgs'),
                                  LogisticRegression(C=1e3, multi_class='multinomial', solver='lbfgs')]}
    lr = LinearSklearn(tau=tau, offset=offset, evtlabs=clsfrevtlabs, clsfr=LogisticRegression(
        C=1, multi_class='multinomial', solver='sag'), labelizeY=True)
    cv_lr = GridSearchCV(lr, tuned_parameters)
    cv_lr.fit(X_TSd, Y_TSy)
    print("CVOPT:\n\n{} = {}\n".format(cv_lr.best_estimator_, cv_lr.best_score_))
    means = cv_lr.cv_results_['mean_test_score']
    stds = cv_lr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, cv_lr.cv_results_['params']):
        print("{:5.3f} (+/-{:5.3f}) for {}".format(mean, std * 2, params))
    print()

    lr.set_params(**cv_lr.best_params_)  # Note: **dict -> k,v argument array
    cv_res = lr.cv_fit(X_TSd, Y_TSy)
    Fy = cv_res['estimator']
    (_) = decodingCurveSupervised(Fy)

    return


if __name__ == "__main__":
    testcase('toy')
    # testcase('askloadsavefile')
