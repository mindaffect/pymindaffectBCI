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

from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
import numpy as np
from mindaffectBCI.decoder.offline.datasets import get_dataset
from mindaffectBCI.decoder.model_fitting import  MultiCCA, LinearSklearn, init_clsfr, BaseSequence2Sequence
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics, plot_factoredmodel, plot_trial
from mindaffectBCI.decoder.scoreStimulus import factored2full, plot_Fe
from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised, print_decoding_curve, plot_decoding_curve, flatten_decoding_curves, score_decoding_curve
from mindaffectBCI.decoder.scoreOutput import plot_Fy
from mindaffectBCI.decoder.preprocess import preprocess, plot_grand_average_spectrum
from mindaffectBCI.decoder.utils import askloadsavefile, block_permute
from mindaffectBCI.decoder.preprocess_transforms import make_preprocess_pipeline
from mindaffectBCI.decoder.stim2event import plot_stim_encoding
import matplotlib.pyplot as plt
import gc
import re
import traceback

try:
    from sklearn.model_selection import GridSearchCV
except:
    pass

def get_train_test_indicators(X,Y,train_idx=None,test_idx=None):
    train_ind=None
    test_ind=None

    # convert idx to indicators
    if test_idx is not None:
        test_ind = np.zeros((X.shape[0],),dtype=bool)
        test_ind[test_idx] = True
    if train_idx is not None:
        train_ind = np.zeros((X.shape[0],),dtype=bool)
        train_ind[train_idx] = True

    # compute missing train/test indicators
    if train_ind is None and test_ind is None:
        # if nothing set use everything
        train_ind = np.ones((X.shape[0],),dtype=bool)
        test_ind  = np.ones((X.shape[0],),dtype=bool)
    elif train_ind is None: # test but no train
        train_ind = np.logical_not(test_ind)
    elif test_ind is None: # train but no test
        test_ind = np.logical_not(train_ind)

    return train_ind, test_ind

def load_and_fit_dataset(loader, filename, model:str='cca', train_idx:slice=None, test_idx:slice=None, loader_args=dict(), preprocess_args=dict(), clsfr_args=dict(), **kwargs):
    from mindaffectBCI.decoder.preprocess import preprocess
    from mindaffectBCI.decoder.analyse_datasets import init_clsfr, get_train_test_indicators
    X, Y, coords = loader(filename, **loader_args)

    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)
    fs = coords[1]['fs'] 
    print("X({})={}, Y={} @{}hz".format([c['name'] for c in coords], X.shape, Y.shape, fs))
    clsfr = init_clsfr(model=model, fs=fs, **clsfr_args)
 
     # do train/test split
    if test_idx is None and train_idx is None:
        X_train = X
        Y_train = Y
    else:
        train_ind, test_ind = get_train_test_indicators(X,Y,train_idx,test_idx)
        print("Training Idx: {}\nTesting Idx :{}\n".format(np.flatnonzero(train_ind),np.flatnonzero(test_ind)))
        X_train = X[train_ind,...]
        Y_train = Y[train_ind,...]

    clsfr.fit(X_train,Y_train)
    return clsfr, filename, X, Y, coords

def load_and_score_dataset(loader, filename, clsfrs:list, train_idx:slice=None, test_idx:slice=None, loader_args=dict(), preprocess_args=dict(), clsfr_args=dict(), **kwargs):
    from mindaffectBCI.decoder.preprocess import preprocess
    from mindaffectBCI.decoder.analyse_datasets import get_train_test_indicators
    X, Y, coords = loader(filename, **loader_args)
    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)

    train_ind, test_ind = get_train_test_indicators(X,Y,train_idx,test_idx)
    X = X[test_ind,...]
    Y = Y[test_ind,...]

    scores = [ None for _ in clsfrs ]
    for i,c in enumerate(clsfrs):
        print('.',end='')
        #try:
        scores[i] = c.score(X,Y) 
        #except:
        #    scores[i] = -1
    print(flush=True)
    return scores, filename


def load_and_decode_dataset(loader, filename, clsfrs:list, loader_args=dict(), preprocess_args=dict(), clsfr_args=dict(), **kwargs):
    from mindaffectBCI.decoder.preprocess import preprocess
    from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised
    X, Y, coords = loader(filename, **loader_args)
    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)
    dcs = [ None for _ in clsfrs ]
    for i,c in enumerate(clsfrs):
        try:
            Fy = c.predict(X,Y)
            (dc) = decodingCurveSupervised(Fy, marginalizedecis=True, minDecisLen=clsfr.minDecisLen, bwdAccumulate=clsfr.bwdAccumulate, priorsigma=(clsfr.sigma0_,clsfr.priorweight), softmaxscale=clsfr.softmaxscale_, nEpochCorrection=clsfr.startup_correction)
            dcs[i] = dc 
        except:
            dcs[i] = None
    return dcs, filename

def print_decoding_curves(decoding_curves):
    int_len, prob_err, prob_err_est, se, st = flatten_decoding_curves(decoding_curves)
    return print_decoding_curve(np.nanmean(int_len,0),np.nanmean(prob_err,0),np.nanmean(prob_err_est,0),np.nanmean(se,0),np.nanmean(st,0))

def plot_decoding_curves(decoding_curves, labels=None):
    int_len, prob_err, prob_err_est, se, st = flatten_decoding_curves(decoding_curves)
    plot_decoding_curve(int_len,prob_err,labels=labels)

def plot_trial_summary(X, Y, Fy, Fe=None, Py=None, fs=None, label=None, evtlabs=None, centerx=True, xspacing=10, sumFy=True, Yerr=None, block=False):
    """generate a plot summarizing the inputs (X,Y) and outputs (Fe,Fe) for every trial in a dataset for debugging purposes

    Args:
        X (nTrl,nSamp,d): The preprocessed EEG data
        Y (nTrl,nSamp,nY): The stimulus information
        Fy (nTrl,nSamp,nY): The output scores obtained by comping the stimulus-scores (Fe) with the stimulus information (Y)
        Fe ((nTrl,nSamp,nE), optional): The stimulus scores, for the different event types, obtained by combining X with the decoding model. Defaults to None.
        Py ((nTrl,nSamp,nY), optional): The target probabilities for each output, derived from Fy. Defaults to None.
        fs (float, optional): sample rate of X, Y, used to set the time-axis. Defaults to None.
        label (str, optional): A textual label for this dataset, used for titles & save-files. Defaults to None.
        centerx (bool, optional): Center (zero-mean over samples) X for plotting. Defaults to True.
        xspacing (int, optional): Gap in X units between different channel lines. Defaults to 10.
        sumFy (bool, optional): accumulate the output scores before plotting. Defaults to True.
        Yerr (bool (nTrl,), optional): indicator for which trials the model made a correct prediction. Defaults to None.
    """    
    times = np.arange(X.shape[1])
    if fs is not None:
        times = times/fs
        xunit='s'
    else:
        xunit='samp'

    if centerx:
        X = X.copy() - np.mean(X,1,keepdims=True)
    if xspacing is None: 
        xspacing=np.median(np.diff(X,axis=-2).ravel())

    if sumFy:
        Fy = np.cumsum(Fy,axis=-2)

    if Fe.ndim>3 and Fe.shape[0]==1:# strip model dim
        Fe = Fe[0,...]

    Xlim = (np.min(X[...,0].ravel()),np.max(X[...,-1].ravel()))

    Fylim = (np.min(Fy.ravel()),np.max(Fy.ravel()))
    if Fe is not None:
        Felim = (np.min(Fe.ravel()),np.max(Fe.ravel()))

    if Py is not None:
        if Py.ndim>3 :
            print("Py: Multiple models? accumulated away")
            Py = np.sum(Py,0)

    if Fy is not None:
        if Fy.ndim>3 :
            print("Fy: Multiple models? accumulated away")
            Fy = np.mean(Fy,0)

    nTrl = X.shape[0]; w = int(np.ceil(np.sqrt(nTrl)*1.8)); h = int(np.ceil(nTrl/w))
    fig=plt.gcf()
    if nTrl>3 : fig.set_size_inches(20,10,forward=True)
    trial_grid = fig.add_gridspec( nrows=h, ncols=w, figure=fig, hspace=.05, wspace=.05) # per-trial grid
    nrows= 5 + (0 if Fe is None else 1) + (0 if Py is None else 1)
    ti=0
    for hi in range(h):
        for wi in range(w):
            if ti>=X.shape[0]:
                break

            gs = trial_grid[ti].subgridspec( nrows=nrows, ncols=1, hspace=0 )

            # pre-make bottom plot
            botax = fig.add_subplot(gs[-1,0])

            # plot X (0-3)
            fig.add_subplot(gs[:3,:], sharex=botax)
            plt.plot(times,X[ti,:,:] + np.arange(X.shape[-1])*xspacing)
            plt.gca().set_xticklabels(())
            plt.grid(True)
            plt.ylim((Xlim[0],Xlim[1]+(X.shape[-1]-1)*xspacing))
            if wi==0: # only left-most-plots
                plt.ylabel('X')
            plt.gca().set_yticklabels(())
            # group 'title'
            plt.text(.5,1,'{}{}'.format(ti,'*' if Yerr is not None and Yerr[ti]==False else ''), ha='center', va='top', fontweight='bold', transform=plt.gca().transAxes)

            # imagesc Y
            fig.add_subplot(gs[3,:], sharex=botax)
            plt.imshow(Y[ti,:,:].T, origin='upper', aspect='auto', cmap='gray', extent=[times[0],times[-1],0,Y.shape[-1]], interpolation=None)
            plt.gca().set_xticklabels(())
            if wi==0: # only left-most-plots
                plt.ylabel('Y')
            plt.gca().set_yticklabels(())

            # Fe (if given)
            if Fe is not None:
                fig.add_subplot(gs[4,:], sharex=botax)
                plt.plot(times,Fe[ti,:,:] + np.arange(Fe.shape[-1])[np.newaxis,:])
                plt.gca().set_xticklabels(())
                plt.grid(True)
                if wi==0: # only left-most-plots
                    plt.ylabel('Fe')
                plt.gca().set_yticklabels(())
                try:
                    plt.ylim((Felim[0],Felim[1]+Fe.shape[-1]-1))
                except:
                    pass

            # Fy
            if Py is None:
                plt.axes(botax) # no Py, Fy is bottom axis
            else:
                row = 4 if Fe is None else 5
                fig.add_subplot(gs[row,:], sharex=botax)
            plt.plot(times,Fy[ti,:,:], color='.5')
            plt.plot(times,Fy[ti,:,0],'k-')
            if hi==h-1 and Py is None: # only bottom plots
                plt.xlabel('time ({})'.format(xunit))
            else:
                plt.gca().set_xticklabels(())
            if wi==0: # only left most plots
                plt.ylabel("Fy")
            plt.grid(True)
            plt.gca().set_yticklabels(())
            try:
                plt.ylim(Fylim)
            except:
                pass

            # Py (if given)
            if Py is not None:
                plt.axes(botax)
                plt.plot(times[:Py.shape[-2]],Py[ti,:,:], color='.5')
                plt.plot(times[:Py.shape[-2]],Py[ti,:,0],'k-')
                if hi==h-1: # only bottom plots
                    plt.xlabel('time ({})'.format(xunit))
                else:
                    plt.gca().set_xticklabels(())
                if wi==0: # only left most plots
                    plt.ylabel("Py")
                plt.grid(True)
                plt.gca().set_yticklabels(())
                plt.ylim((0,1))

            ti=ti+1

    if label is not None:
        if Yerr is not None:
            plt.suptitle("{} {}/{} correct".format(label,sum(np.logical_not(Yerr)),len(Yerr)))
        else:
            plt.suptitle("{}".format(label))
    fig.set_tight_layout(True)
    plt.show(block=block)

def plot_stimseq(Y_TSy,fs=None,show=None):
    if fs is not None:
        plt.plot(np.arange(Y_TSy.shape[1])/fs, Y_TSy[0,...]/np.max(Y_TSy)*.75+np.arange(Y_TSy.shape[-1])[np.newaxis,:],'.')
        plt.xlabel('time (s)')
        plt.ylabel('output + level')
    else:
        plt.plot(Y_TSy[0,...]/np.max(Y_TSy)*.75+np.arange(Y_TSy.shape[-1])[np.newaxis,:],'.')
        plt.xlabel('time (samp)')
        plt.ylabel('output + level')
    plt.grid(True)
    plt.title('Y_TSy')
    if show is not None: plt.show(block=show)


def print_hyperparam_summary(res):
    fn = res[0].get('filenames',None)
    s = "N={}\n".format(len(fn))
    if fn:
        s = s + "fn={}\n".format([f[-30:] if f else None for f in fn])
    for ri in res:
        s += "\n{}\n".format(ri['config'])
        s += print_decoding_curves(ri['decoding_curves'])
    return s


def cv_fit(clsfr, X, Y, cv, fit_params=dict(), verbose:int=0, cv_clsfr_only:bool=False, score_fn=None, **kwargs):
    """cross-validated fit a classifier and compute it's scores on the validation set

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        fit_params ([type]): [description]

    Returns:
        np.ndarray: Fy the predictions on the validation examples
    """
    if fit_params is None: fit_params=dict()

    # FAST PATH use the classifiers optimized CV_fit routine 
    if hasattr(clsfr,'cv_fit'):
        res = clsfr.cv_fit(X, Y, cv=cv, fit_params=fit_params, **kwargs)

    else:
        if cv_clsfr_only and hasattr(clsfr,'stages') and hasattr(clsfr.stages[-1][1],'cv_fit'):
            # BODGE: only cv the final stage
            Xpp, Ypp = clsfr.fit_modify(X, Y, until_stage=-1)
            res = clsfr.stages[-1][1].cv_fit(Xpp, Ypp, cv=cv, fit_params=fit_params, score_fn=score_fn, **kwargs)

        else:
            # manually run the folds and collect the results
            # setup the folding
            if cv == True:  cv = 5
            if isinstance(cv, int):
                if X.shape[0] == 1 or cv <= 1:
                    # single trial, train/test on all...
                    cv = [(slice(X.shape[0]), slice(X.shape[0]))] # N.B. use slice to preserve dims..
                else:
                    from sklearn.model_selection import StratifiedKFold
                    cv = StratifiedKFold(n_splits=min(cv, X.shape[0])).split(np.zeros(X.shape[0]), np.zeros(Y.shape[0]))

            scores=[]
            for i, (train_idx, valid_idx) in enumerate(cv):
                if verbose > 0:
                    print(".", end='', flush=True)
                
                clsfr.fit(X[train_idx, ...].copy(), Y[train_idx, ...].copy(), **fit_params, **kwargs)

                # predict, forcing removal of copies of  tgt=0 so can score
                if X[valid_idx,...].size==0:
                    print("Warning: no-validation trials!!! using all data!")
                    valid_idx = slice(X.shape[0])
                if hasattr(clsfr,'stages'):
                    Xpp, Ypp = clsfr.modify(X[valid_idx,...],Y[valid_idx,...],until_stage=-1) # pre-process
                    if score_fn is not None:
                        score = score_fn(clsfr,Xpp,Ypp)
                    elif hasattr(clsfr.stages[-1][1],'score'):
                        score = clsfr.stages[-1][1].score(Xpp, Ypp) # score
                else:
                    if score_fn is not None:
                        score = score_fn(clsfr,X[valid_idx,...].copy(), Y[valid_idx,...].copy())
                    else:
                        score = clsfr.score(X[valid_idx, ...].copy(), Y[valid_idx, ...].copy())
                scores.append(score)
            res=dict(scores_cv=scores)

    return res


def cv_fit_predict(clsfr, X, Y, cv, fit_params=dict(), verbose:int=0, cv_clsfr_only:bool=False, **kwargs):
    """cross-validated fit a classifier and compute it's predictions on the validation set

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        fit_params ([type]): [description]

    Returns:
        np.ndarray: Fy the predictions on the validation examples
    """
    if fit_params is None: fit_params=dict()

    # FAST PATH use the classifiers optimized CV_fit routine 
    if hasattr(clsfr,'cv_fit'):
        res = clsfr.cv_fit(X, Y, cv=cv, fit_params=fit_params, **kwargs)
        Fy = res['estimator'] if not 'rawestimator' in res else res['rawestimator']

    else:
        if cv_clsfr_only and hasattr(clsfr,'stages') and  hasattr(clsfr.stages[-1][1],'cv_fit'):
            # BODGE: only cv the final stage
            Xpp, Ypp = clsfr.fit_modify(X, Y, until_stage=-1)
            res = clsfr.stages[-1][1].cv_fit(Xpp, Ypp, cv=cv, fit_params=fit_params, **kwargs)
            Fy = res['estimator'] if not 'rawestimator' in res else res['rawestimator']

        else:
            # manually run the folds and collect the results
            # setup the folding
            if cv == True:  cv = 5
            if isinstance(cv, int):
                if X.shape[0] == 1 or cv <= 1:
                    # single trial, train/test on all...
                    cv = [(slice(X.shape[0]), slice(X.shape[0]))] # N.B. use slice to preserve dims..
                else:
                    from sklearn.model_selection import StratifiedKFold
                    cv = StratifiedKFold(n_splits=min(cv, X.shape[0])).split(np.zeros(X.shape[0]), np.zeros(Y.shape[0]))

            for i, (train_idx, valid_idx) in enumerate(cv):
                if verbose > 0:
                    print(".", end='', flush=True)
                
                clsfr.fit(X[train_idx, ...].copy(), Y[train_idx, ...].copy(), **fit_params, **kwargs)

                # predict, forcing removal of copies of  tgt=0 so can score
                if X[valid_idx,...].size==0:
                    print("Warning: no-validation trials!!! using all data!")
                    valid_idx = slice(X.shape[0])
                Fyi = clsfr.predict(X[valid_idx, ...].copy(), Y[valid_idx, ...].copy())

                if i==0: # reshape Fy to include the extra model dim
                    Fy = np.zeros((Y.shape[0],)+Fyi.shape[1:], dtype=X.dtype)       
                Fy[valid_idx,...]=Fyi

    return dict(estimator_cv=Fy)


def decoding_curve_cv(clsfr:BaseSequence2Sequence, X, Y, cv, fit_params=dict(), cv_clsfr_only:bool=False):
    """cross-validated fit a classifier and compute it's decoding curve and associated scores

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        fit_params ([type]): [description]

    Returns:
        [type]: [description]
    """
    if cv is not None:
        res = cv_fit_predict(clsfr, X, Y, cv=cv, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
        Fy = res['estimator_cv'] if isinstance(res,dict) else res
    else: # predict only
        #print("Predict only!")
        Fy = clsfr.predict(X,Y)

    # BODGE: get the proba calibration info!
    try:
        est = clsfr.stages[-1][1] if hasattr(clsfr,'stages') else clsfr
        est.calibrate_proba(Fy)
        softmaxscale, startup_correction, sigma0, priorweight, nvirt_out=\
             (est.softmaxscale_, est.startup_correction, est.sigma0_, est.priorweight, est.nvirt_out)
    except:
        softmaxscale, startup_correction, sigma0, priorweight, nvirt_out = (3,None,None,0, 0)

    nvirt_out = 0

    # extract the predictions and compute the decoding curve and scores
    (dc) = decodingCurveSupervised(Fy, marginalizedecis=True, priorsigma=(sigma0,priorweight), softmaxscale=softmaxscale, nEpochCorrection=startup_correction, nvirt_out=nvirt_out, verb=-1)
    scores = score_decoding_curve(*dc)
    scores['decoding_curve']=dc
    scores['Fy']=Fy
    return scores

def set_params_decoding_curve_cv(clsfr:BaseSequence2Sequence, X, Y, cv, config:dict=dict(), fit_params:dict=dict(), cv_clsfr_only:bool=False, extra_config:dict=None):
    """set parameters on classifier and then cv-fit and compute it's decoding curve

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        config (dict): parameters to set on the estimator with set_params(**config)
        fit_params (dict): additional parameters to pass to cv_fit
        extra_config (dict): extra info about X/Y to store in this runs results

    Returns:
        [type]: [description]
    """    
    from sklearn import clone
    from copy import deepcopy
    clsfr = clone(clsfr) if cv is not None and cv is not False else deepcopy(clsfr)
    clsfr.set_params(**config)
    scores = decoding_curve_cv(clsfr, X, Y, cv=cv, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
    #print("Scores: {}".format(scores['score']))
    scores['config']=config
    scores['clsfr']=clsfr
    if extra_config : 
        scores.update(extra_config)
    return scores

def collate_futures_results(futures:list, verb:int=1):
    """combine the results from `decoding_curve_gridsearchCV` into a single dict

    Args:
        futures (list-of-dicts or list-of-futures): the future results to combine

    Returns:
        [type]: [description]
    """    
    import concurrent.futures
    import time
    list_of_futures = hasattr(futures[0],'result')
    # collect the results as the jobs finish
    res={}
    # get iterator over the futures (or list of results)
    futures_iter = concurrent.futures.as_completed(futures) if list_of_futures else futures
    n_to_collect=len(futures)
    n_collected =0
    t0 = time.time()
    tlog = t0
    for future in futures_iter:
        n_collected = n_collected+1
        if verb>0 and time.time()-tlog > 3:
            elapsed=time.time()-t0
            done = n_collected / n_to_collect
            print("\r{:d}% {:d} / {:d} collected in {:4.1f}s  est {:4.1f}s total {:4.1f}s remaining".format(int(100*done), n_collected,n_to_collect,elapsed,elapsed/done,elapsed/done - elapsed))
            tlog = time.time()
        if list_of_futures:
            try:
                scores = future.result()
            except :
                print("fail in getting result!")
                import traceback
                traceback.print_exc()
                continue
        else: # list of dicts
            scores=future

        # merge into the set of all results
        # work with list of dicts as output
        scores = [scores] if not isinstance(scores,list) else scores
        for score in scores:
            if score is None: continue
            if res :
                for k,v in score.items(): 
                    res[k].append(v.tolist() if isinstance(v,np.ndarray) else v)
            else:
                res = {k:[v.tolist() if isinstance(v,np.ndarray) else v] for k,v in score.items()}
    return res

def get_results_per_config(res):
    # get the unique configurations and the rows for each
    configs = []
    configrows = []
    for rowi, cf in enumerate(res['config']):
        key = str(cf)
        #print("key={}".format(key))
        try:
            fi = configs.index(key)
            configrows[fi].append(rowi)
        except ValueError: # new config
            #print("newkey={}".format(key))
            configs.append(key)
            configrows.append([rowi])

    # for each config, make a new row with the average of this configs values
    configres = dict() # config results in dict with key str config
    for ci,(cf,rows) in enumerate(zip(configs,configrows)):
        configinfo = dict()
        for k,v in res.items():
            vs = [v[r] for r in rows]
            configinfo[k]=vs
        configres[str(cf)] = configinfo
    return configres, configs, configrows


def average_results_per_config(res):

    _, configs, configrows = get_results_per_config(res)

    # for each config, make a new row with the average of this configs values
    newres = { k:[] for k in res.keys() }
    for ci,(cf,rows) in enumerate(zip(configs,configrows)):
        for k,v in res.items():
            vs = [v[r] for r in rows]
            if k.lower() == 'decoding_curve': # BODGE: special reducer for decoding-curves
                newres[k].append( [np.nanmean(v,0) for v in flatten_decoding_curves(vs)] )
            else:
                try:
                    newres[k].append(np.nanmean(vs,0))
                except:
                    newres[k].append(vs)
        # ensure config is right
        newres['config'][ci] = cf
    return newres


def decoding_curve_GridSearchCV(clsfr:BaseSequence2Sequence, X, Y, cv, n_jobs:int=-1, tuned_parameters:dict=dict(), label:str=None, fit_params:dict=dict(), cv_clsfr_only:bool=False, verb:int=1):
    from sklearn.model_selection import ParameterGrid
    import concurrent.futures
    import time

    # TODO[]: get the right number from concurrent_futures
    n_jobs = 8 if n_jobs<0 else n_jobs

    parameter_grid = ParameterGrid(tuned_parameters)
    print("Submitting {} jobs:".format(len(parameter_grid)),end='')
    futures=[]
    t0 = time.time()
    if n_jobs==0 or n_jobs==1: # in main thread
        for ci,fit_config in enumerate(parameter_grid):
            print(".",end='')
            future = set_params_decoding_curve_cv(clsfr, X, Y, cv, config=fit_config, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
            futures.append(future)
        # collect the results as the jobs finish
        res= collate_futures_results(futures)

    else:
        # to job pool
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            print("Running with {} parallel tasks".format(n_jobs))

            # loop over analysis settings
            for ci,fit_config in enumerate(parameter_grid):
                print(".",end='')
                future = executor.submit(set_params_decoding_curve_cv, clsfr, X, Y, cv, config=fit_config, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
                futures.append(future)
            print("submitted. in {}s".format(time.time()-t0))

            # collect the results as the jobs finish
            res= collate_futures_results(futures)

    print("Completed {} jobs in {} s".format(len(parameter_grid),time.time()-t0))
    return res

def load_and_decoding_curve_GridSearchCV(clsfr:BaseSequence2Sequence, filename, loader, cv, 
            n_jobs:int=-1, tuned_parameters:dict=dict(), label:str=None, 
            fit_params:dict=dict(), loader_args:dict=dict(), shortfilename:str=None, **kwargs):
    """ load filename with given loader and then do gridsearch CV

    Args:
        clsfr (BaseSequence2Sequence): the classifier to apply to the datasets
        filenames ([type]): list of filenames to load
        loader ([type]): loader function to load filename
        cv ([type]): cross-validation to do, or CV object
        n_jobs (int, optional): number parallel jobs to run for analysis. Defaults to -1.
        tuned_parameters (dict, optional): description of the different parameter settings to test -- as for `GridSearchCV`. Defaults to dict().
        label (str, optional): descriptive label for this run. Defaults to None.
        fit_params (dict, optional): additional parameters to pass to clsfr.fit . Defaults to dict().
        loader_args (dict, optional): additional parameters to pass to loader(filename). Defaults to dict().

    Returns:
        list-of-dict: list of cvresults dictionary, with keys for different outputs and rows for particular configuration runs (combination of filename and tuned_parameters)
    """
    if tuned_parameters is None : tuned_parameters=dict()
    if fit_params is None : fit_params=dict()
    from sklearn.model_selection import ParameterGrid
    import concurrent.futures
    try:
        X, Y, coords = loader(filename,**loader_args)
    except:
        import traceback
        traceback.print_exc()
        print("Problem loading file: {}.    SKIPPED".format(filename))
        return []
    if shortfilename is None: shortfilename=filename
    
    # BODGE: manually add the meta-info to the pipeline
    #  TODO[]: Make so meta-info is added to X in the loader, and preserved over multi-threading
    try:
        fs = coords[1]['fs']
        ch_names = coords[2]['coords']
        clsfr.set_params(metainfoadder__info=dict(fs=fs, ch_names=ch_names))
    except:
        print("Warning -- cant add meta-info to clsfr pipeline!")

    # record the shortened filename as extra_config
    extra_config = { 'filename':shortfilename }
    if loader_args:
        extra_config['loader_args']=loader_args

    # loop over analysis settings
    futures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for fit_config in ParameterGrid(tuned_parameters):
            if n_jobs==0 or n_jobs==1:
                future = set_params_decoding_curve_cv(clsfr, X, Y, cv, fit_config, fit_params=fit_params, extra_config=extra_config, **kwargs)
            else:
                future = executor.submit(set_params_decoding_curve_cv, clsfr, X, Y, cv, fit_config, fit_params=fit_params, extra_config=extra_config, **kwargs)
            futures.append(future)
    return futures


def datasets_decoding_curve_GridSearchCV(clsfr:BaseSequence2Sequence, filenames, loader, cv, 
            n_jobs:int=-1, tuned_parameters:dict=dict(), label:str=None, 
            fit_params:dict=dict(), cv_clsfr_only:bool=False, loader_args:dict=dict(), job_per_file:bool=True):
    """[summary]

    Args:
        clsfr (BaseSequence2Sequence): the classifier to apply to the datasets
        filenames ([type]): list of filenames to load
        loader ([type]): loader function to load filename
        cv ([type]): cross-validation to do, or CV object
        n_jobs (int, optional): number parallel jobs to run for analysis. Defaults to -1.
        tuned_parameters (dict, optional): description of the different parameter settings to test -- as for `GridSearchCV`. Defaults to dict().
        label (str, optional): descriptive label for this run. Defaults to None.
        fit_params (dict, optional): additional parameters to pass to clsfr.fit . Defaults to dict().
        loader_args (dict, optional): additional parameters to pass to loader(filename). Defaults to dict().

    Returns:
        dict: cvresults dictionary, with keys for different outputs and rows for particular configuration runs (combination of filename and tuned_parameters)
    """    
    import concurrent.futures
    import os.path
    import time

    # TODO[]: get the right number from concurrent_futures
    n_jobs = 8 if n_jobs<0 else n_jobs

    common_path = os.path.commonpath(filenames)

    futures = []
    t0 = time.time()
    tlog = t0
    if n_jobs > 1 and job_per_file: # each file in it's own job
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            print("Running with {} parallel tasks, one-per-filename".format(n_jobs))
            print("Submitting {} jobs:",end='')
            for fi,fn in enumerate(filenames):
                print('.',end='')
                future = executor.submit(load_and_decoding_curve_GridSearchCV, clsfr, fn, loader, cv, n_jobs=1, tuned_parameters=tuned_parameters,label=label,fit_params=fit_params, cv_clsfr_only=cv_clsfr_only, loader_args=loader_args, shortfilename=fn[len(common_path)+1:])
                futures.append(future)
            print("{} jobs submitted in {:4.0f}s. Waiting results.\n".format(len(filenames),time.time()-t0))
        
            # collect the results as the jobs finish
            res = collate_futures_results(futures)
    else: # each config in it's own job
        for fi,fn in enumerate(filenames):
            if time.time()-tlog > 3:
                print("\r{} of {}  in {}s".format(fi,len(filenames),time.time()-t0))
                tlog=time.time()
            future = load_and_decoding_curve_GridSearchCV(clsfr,fn,loader,cv,n_jobs=n_jobs,tuned_parameters=tuned_parameters,label=label,fit_params=fit_params, cv_clsfr_only=cv_clsfr_only, loader_args=loader_args, shortfilename=fn[len(common_path)+1:])
            futures.append(future)
        # collect the results as the jobs finish
        res = collate_futures_results(futures)

    return res


def set_params_cv(clsfr:BaseSequence2Sequence, X, Y, cv, config:dict=dict(), score_fn=None,
                  fit_params:dict=dict(), cv_clsfr_only:bool=False, extra_config:dict=None):
    """set parameters on classifier and then cv-fit and compute it's decoding curve

    Args:
        clsfr (BaseSequence2Sequence): [description]
        X ([type]): [description]
        Y ([type]): [description]
        cv ([type]): [description]
        config (dict): parameters to set on the estimator with set_params(**config)
        fit_params (dict): additional parameters to pass to cv_fit
        extra_config (dict): extra info about X/Y to store in this runs results

    Returns:
        [type]: [description]
    """
    from sklearn import clone
    from copy import deepcopy
    clsfr = clone(clsfr) if cv is not None and cv is not False else deepcopy(clsfr)
    if config is not None:
        clsfr.set_params(**config)
    scores = cv_fit(clsfr, X.copy(), Y.copy(), cv=cv, fit_params=fit_params, cv_clsfr_only=cv_clsfr_only)
    #print("Scores: {}".format(scores['scores_cv']))
    scores['config']=config
    scores['clsfr']=clsfr
    if extra_config : 
        scores.update(extra_config)
    return scores


def load_and_GridSearchCV(clsfr:BaseSequence2Sequence, filename, loader, cv, 
            n_jobs:int=1, tuned_parameters:dict=dict(), label:str=None, 
            fit_params:dict=dict(), loader_args:dict=dict(), shortfilename:str=None, **kwargs):
    """ load filename with given loader and then do gridsearch CV

    Args:
        clsfr (BaseSequence2Sequence): the classifier to apply to the datasets
        filenames ([type]): list of filenames to load
        loader ([type]): loader function to load filename
        cv ([type]): cross-validation to do, or CV object
        n_jobs (int, optional): number parallel jobs to run for analysis. Defaults to -1.
        tuned_parameters (dict, optional): description of the different parameter settings to test -- as for `GridSearchCV`. Defaults to dict().
        label (str, optional): descriptive label for this run. Defaults to None.
        fit_params (dict, optional): additional parameters to pass to clsfr.fit . Defaults to dict().
        loader_args (dict, optional): additional parameters to pass to loader(filename). Defaults to dict().

    Returns:
        list-of-dict: list of cvresults dictionary, with keys for different outputs and rows for particular configuration runs (combination of filename and tuned_parameters)
    """
    if tuned_parameters is None : tuned_parameters=dict()
    if fit_params is None : fit_params=dict()
    from sklearn.model_selection import ParameterGrid
    import concurrent.futures
    try:
        X, Y, coords = loader(filename,**loader_args)
    except:
        import traceback
        traceback.print_exc()
        print("Problem loading file: {}.    SKIPPED".format(filename))
        return []
    if shortfilename is None: shortfilename=filename
    
    # BODGE: manually add the meta-info to the pipeline
    #  TODO[]: Make so meta-info is added to X in the loader, and preserved over multi-threading
    try:
        fs = coords[1]['fs']
        ch_names = coords[2]['coords']
        clsfr.set_params(metainfoadder__info=dict(fs=fs, ch_names=ch_names))
    except:
        print("Warning -- cant add meta-info to clsfr pipeline!")

    # record the shortened filename as extra_config
    extra_config = { 'filename':shortfilename }
    if loader_args:
        extra_config['loader_args']=loader_args

    futures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # loop over analysis settings
        for fit_config in ParameterGrid(tuned_parameters):
            if n_jobs>1: # multi-thread
                future = executor.submit(set_params_cv, clsfr, X, Y, cv, fit_config, fit_params=fit_params, extra_config=extra_config, **kwargs)
            else:# single thread
                future = set_params_cv(clsfr, X, Y, cv, fit_config, fit_params=fit_params, extra_config=extra_config, **kwargs)
            futures.append(future)
    return futures


def datasets_GridSearchCV(clsfr:BaseSequence2Sequence, filenames, loader, cv, 
            n_jobs:int=-1, tuned_parameters:dict=dict(), label:str=None, 
            fit_params:dict=dict(), cv_clsfr_only:bool=False, loader_args:dict=dict(), job_per_file:bool=True):
    """run a complete dataset with different parameter settings expanding the grid of tuned_parameters

    Args:
        clsfr (BaseSequence2Sequence): the classifier to apply to the datasets
        filenames ([type]): list of filenames to load
        loader ([type]): loader function to load filename
        cv ([type]): cross-validation to do, or CV object
        n_jobs (int, optional): number parallel jobs to run for analysis. Defaults to -1.
        tuned_parameters (dict, optional): description of the different parameter settings to test -- as for `GridSearchCV`. Defaults to dict().
        label (str, optional): descriptive label for this run. Defaults to None.
        fit_params (dict, optional): additional parameters to pass to clsfr.fit . Defaults to dict().
        loader_args (dict, optional): additional parameters to pass to loader(filename). Defaults to dict().
    """    
    import concurrent.futures
    import os.path
    import time

    # TODO[]: get the right number from concurrent_futures
    n_jobs = 8 if n_jobs<0 else n_jobs

    common_path = os.path.commonpath(filenames)

    futures = []
    t0 = time.time()
    tlog = t0

    if n_jobs > 1 and job_per_file: # each file in it's own job
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            print("Running with {} parallel tasks, one-per-filename".format(n_jobs))
            print("Submitting {} jobs:",end='')
            for fi,fn in enumerate(filenames):
                print('.',end='')
                future = executor.submit(load_and_GridSearchCV, clsfr, fn, loader, cv, n_jobs=1, tuned_parameters=tuned_parameters,label=label,fit_params=fit_params, cv_clsfr_only=cv_clsfr_only, loader_args=loader_args, shortfilename=fn[len(common_path)+1:])
                futures.append(future)
            print("{} jobs submitted in {:4.0f}s. Waiting results.\n".format(len(filenames),time.time()-t0))
        
            # collect the results as the jobs finish
            res = collate_futures_results(futures)
    else: # each config in it's own job
        for fi,fn in enumerate(filenames):
            if time.time()-tlog > 3:
                print("\r{} of {}  in {}s".format(fi,len(filenames),time.time()-t0))
                tlog=time.time()
            future = load_and_GridSearchCV(clsfr,fn,loader,cv,n_jobs=n_jobs,tuned_parameters=tuned_parameters,label=label,fit_params=fit_params, cv_clsfr_only=cv_clsfr_only, loader_args=loader_args, shortfilename=fn[len(common_path)+1:])
            futures.append(future)

        # collect the results as the jobs finish
        res = collate_futures_results(futures)

    return res


from datetime import datetime
import json
import pickle

if __name__=="__main__":

    #loader_args = dict(fs_out=100,filterband=(3,25,'bandpass'))#((0,3),(45,65)))#,))
    #loader_args = dict()
    loader_args = dict(filterband=((.5,45,'bandpass')), fs_out = 62.5)

    # Noisetag pipeline

    # ask for a directory to lood all datafiles from
    exptdir = askloadsavefile(filetypes='dir')
    loader, filenames, exptdir = get_dataset('mindaffectBCI',exptdir=exptdir)
    print("Got {} files".format(len(filenames)))

    # other testing datasets
    # loader, filenames, exptdir = get_dataset('lowlands')

    # c-VEP pipeline
    fs_out = 60
    tau_ms = 850
    offset_ms = 0
    filterband = None
    temporal_basis = 'wf2,10'
    perlevelweight=True
    evtlabs=('re')
    pipeline=[
        'MetaInfoAdder',
        'BadChannelRemover',
        ['NoiseSubspaceDecorrelator', {'ch_names':['Fp1', 'Fp2'], 'noise_idx':'idOutliers2', 'filterband':(.5,8,'bandpass')}],
        ['ButterFilterAndResampler', {'filterband':filterband, 'fs_out':fs_out}],
        #['TargetEncoder',{'evtlabs':('re')}],
        #['MultiCCA:clsfr',{'tau_ms':950, "rank":1, "temporal_basis":"winfourier10"}],
        #['MultiCCACV:clsfr',{'tau_ms':450, "inner_cv_params":{"rank":(1,3,5),"reg":[(1e-1,1e-1),(1e-3,1e-3),(1e-4,1e-4)]} }],
        #['MultiCCACV:clsfr',{'tau_ms':450, "inner_cv_params":{"rank":(1,3,5)}, "temporal_basis":"winfourier10"}],
        #['FwdLinearRegressionCV:clsfr',{"tau_ms":450,"inner_cv_params":{"reg":[1e-4,1e-3,1e-2,1e-1,.2,.3,.5,.9]}}],
        #['BwdLinearRegressionCV:clsfr',{"tau_ms":450,"inner_cv_params":{"reg":[1e-1,.1,.2,.3,.5]}}]
        #['BwdLinearRegressionCV:clsfr',{"tau_ms":450,"inner_cv_params":{"reg":[1e-1,.1,.2,.3,.5,.8]}}] #"reg":.1}]#
        ["TimeShifter", {"timeshift_ms":offset_ms}],
        ['mindaffectBCI.decoder.model_fitting.MultiCCACV:clsfr', {'evtlabs':evtlabs, 'tau_ms':tau_ms, "rank":1, 'temporal_basis':temporal_basis}]
    ]
    tuned_parameters={'clsfr__tau_ms':[450,650,850]}
    fit_params=None
    #tuned_parameters = dict(clsfr__reg=[1e-4,1e-3,1e-2,1e-1,.3])
    ppp = make_preprocess_pipeline(pipeline)

    res = datasets_GridSearchCV(ppp, filenames[:5], loader, loader_args=loader_args,
                 cv=10, n_jobs=5,
                 job_per_file=True, tuned_parameters=tuned_parameters,
                 cv_clsfr_only=True, fit_params=fit_params)


    # WARNING: cv_clsfr_only is currently broken -- ends up not allowing a classifiers inner-cv to run!
    res = datasets_decoding_curve_GridSearchCV(ppp, filenames, loader, loader_args=loader_args,
                 cv=5, n_jobs=1, tuned_parameters=tuned_parameters, 
                 job_per_file=True, 
                 cv_clsfr_only=False, fit_params=fit_params)

    # plot the x-dataset curves per config:
    label = 'mark_cca'
    configresults, configs, configrows = get_results_per_config(res)
    for config,configres in configresults.items():
        print("\n\n{}\n".format(config))
        plt.figure()
        print(print_decoding_curves(configres['decoding_curve']))
        plot_decoding_curves(configres['decoding_curve'])#, labels=configres['config'])

    # save the per-config results to a json file..
    UNAME = datetime.now().strftime("%y%m%d_%H%M")
    with open('{}_{}'.format(label,UNAME),'wb') as f:
        pickle.dump(res,f)
        #json.dump(res,f)
    plt.show(block=False)

    # get the per-config summary
    res = average_results_per_config(res)
    # report the per-config summary
    for dc,conf in zip(res['decoding_curve'],res['config']):
        print("\n\n{}\n".format(conf))
        print(print_decoding_curve(*dc))
    plt.figure()
    plot_decoding_curves(res['decoding_curve'],labels=res['config'])
    plt.show(block=True)

    quit()

    # # EMG pipeline
    # dataset_args = dict(exptdir='~/Desktop/emg')
    # loader_args = dict(fs_out=100,filterband=((45,55),(95,105),(145,155),(195,205),(2,220,'bandpass')))#((0,3),(45,65)))#,))
    # pipeline = [ 
    #     ["MetaInfoAdder", {"info":{"fs":-1}}],
    #     ["AdaptiveSpatialWhitener:wht1", {"halflife_s":10}],
    #     #"SpatialWhitener:wht1",
    #     ["FFTfilter:filt1", {"filterbank":[[30,35,60,65,"hilbert"],[60,65,110,115,"hilbert"],[110,115,145,145,"hilbert"]], "blksz":100}],
    #     "Log",
    #     #["AdaptiveSpatialWhitener", {"halflife_s":10}],
    #     "SpatialWhitener:wht2",
    #     ["ButterFilterAndResampler:filt2", {"filterband":(8,-1), "fs_out":25}],
    #     ["TargetEncoder", { "evtlabs":"hoton"}],
    #     ["TimeShifter",{"timeshift_ms":0}],
    #     #"FeatureDimCompressor", # make back to 3d Trial,Sample,Features
    #     ["MultiCCA_cv", {"tau_ms":200, "rank":10, "center":True, "startup_correction":0}]
    # ]

    tuned_parameters=dict(#multicca__rank=(5,10), 
    #            multicca__tau_ms=[200, 300, 350, 450], 
    #            multicca__offset_ms=[0, 75, 100],
                        # filt1__filterbank=[[[30,35,60,65,"hilbert"],[60,65,110,115,"hilbert"],[110,115,145,145,"hilbert"]],
                        #                     (30,35,140,145,'hilbert'),
                        #                     (20,25,140,145,'hilbert'),
                        #                     (10,15,140,145,'hilbert'),
                        #                     (20,25,190,195,'hilbert'),
                        #                     (20,25,230,245,'hilbert'),
                        #                     (30,35,120,125,'hilbert'),
                        # ]
                        )


    #fit_params={"rank":(1,3,5),"reg":[(1e-1,1e-1),(1e-4,1e-4)]}

    # pipeline= [ 
    #         ["MetaInfoAdder", {"info":{"fs":-1}, "force":False}],  #N.B. NEED this for GridSearchCV!!
    #         #"SpatialWhitener:wht1",
    #         ["ButterFilterAndResampler:filt2", {"filterband":(45,125,'bandpass'), 'order':6}],
    #         "SpatialWhitener:wht2",
    #         #["FFTfilter:filt1", {"filterbank":[30,35,145,150,"bandpass"], "blksz":100, "squeeze_feature_dim":True}],
    #         ["BlockCovarianceMatrixizer", {"blksz_ms":200, "window":1, "overlap":.5}],
    #         ["ButterFilterAndResampler:filt2", {"filterband":(8,-1)}],
    #         ["TimeShifter", {"timeshift_ms":0}],
    #         ["TargetEncoder", {"evtlabs":"hoton"}],
    #         "DiagonalExtractor",
    #         #"Log",
    #         "FeatureDimCompressor", # make back to 3d Trial,Sample,Features
    #         ["MultiCCA", {"tau_ms":200, "rank":10, "center":True, "startup_correction":0}]
    #     ]

    # tuned_parameters=dict(
    #     diagonalextractor=[None,'skip'],
    #                     )

    ppp = make_preprocess_pipeline(pipeline)
    print(ppp)

    fit_params=dict() #dict(rank=(1,3,5),reg=[(x,y) for x in (1e-3,1e-1) for y in (1e-4,1e-2)])

    loader, filenames, _ = get_dataset(dataset,**dataset_args)
    res = datasets_decoding_curve_GridSearchCV(ppp, filenames, loader, loader_args={"fs_out":500},
                 cv=5, n_jobs=1, tuned_parameters=tuned_parameters, 
                 job_per_file=True, 
                 cv_clsfr_only=False, fit_params=fit_params)
    # get the per-config summary
    res = average_results_per_config(res)
    # report the per-config summary
    for dc,conf in zip(res['decoding_curve'],res['config']):
        print("\n\n{}\n".format(conf))
        print(print_decoding_curve(*dc))
    plt.figure()
    plot_decoding_curves(res['decoding_curve'],labels=res['config'])
    plt.show(block=True)