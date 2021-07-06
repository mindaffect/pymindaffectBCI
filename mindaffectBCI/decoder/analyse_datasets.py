#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
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
from mindaffectBCI.decoder.datasets import get_dataset
from mindaffectBCI.decoder.model_fitting import BaseSequence2Sequence, MultiCCA, FwdLinearRegression, BwdLinearRegression, LinearSklearn
try:
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.svm import LinearSVR, LinearSVC
    from sklearn.model_selection import GridSearchCV
except:
    pass
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics, plot_factoredmodel
from mindaffectBCI.decoder.scoreStimulus import factored2full, plot_Fe
from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised, print_decoding_curve, plot_decoding_curve
from mindaffectBCI.decoder.scoreOutput import plot_Fy
from mindaffectBCI.decoder.preprocess import preprocess, plot_grand_average_spectrum
from mindaffectBCI.decoder.trigger_check import triggerPlot
import matplotlib.pyplot as plt
import gc
import re



def analyse_dataset(X:np.ndarray, Y:np.ndarray, coords, model:str='cca', test_idx=None, cv=True, tau_ms:float=300, fs:float=None,  rank:int=1, evtlabs=None, offset_ms=0, center=True, tuned_parameters=None, ranks=None, retrain_on_all=True, **kwargs):
    """ cross-validated training on a single datasets and decoing curve estimation

    Args:
        X (np.ndarray): the X (EEG) sequence
        Y (np.ndarray): the Y (stimulus) sequence
        coords ([type]): array of dicts of meta-info describing the structure of X and Y
        fs (float): the data sample rate (if coords is not given)
        model (str, optional): The type of model to fit, as in `model_fitting.py`. Defaults to 'cca'.
        cv (bool, optional): flag if we should train with cross-validation using the cv_fit method. Defaults to True.
        test_idx (list-of-int, optional): indexs of test-set trials which are *not* passed to fit/cv_fit. Defaults to True.
        tau_ms (float, optional): length of the stimulus-response in milliseconds. Defaults to 300.
        rank (int, optional): rank of the decomposition in factored models such as cca. Defaults to 1.
        evtlabs ([type], optional): The types of events to used to model the brain response, as used in `stim2event.py`. Defaults to None.
        offset_ms ((2,):float, optional): Offset for analysis window from start/end of the event response. Defaults to 0.

    Raises:
        NotImplementedError: if you use for a model which isn't implemented

    Returns:
        score (float): the cv score for this dataset
        dc (tuple): the information about the decoding curve as returned by `decodingCurveSupervised.py`
        Fy (np.ndarray): the raw cv'd output-scores for this dataset as returned by `decodingCurveSupervised.py` 
        clsfr (BaseSequence2Sequence): the trained classifier
    """
    # extract dataset info
    if coords is not None:
        fs = coords[1]['fs'] 
        print("X({})={}, Y={} @{}hz".format([c['name'] for c in coords], X.shape, Y.shape, fs))
    else:
        print("X={}, Y={} @{}hz".format(X.shape, Y.shape, fs))
    tau = int(tau_ms*fs/1000)
    offset=int(offset_ms*fs/1000)

    Cscale = np.sqrt(np.mean(X.ravel()**2))
    print('Cscale={}'.format(Cscale))
    C = .1/Cscale

    # create the model if not provided
    if isinstance(model,BaseSequence2Sequence):
        clsfr = model
    elif model=='cca' or model is None:
        clsfr = MultiCCA(tau=tau, offset=offset, rank=rank, evtlabs=evtlabs, center=center, **kwargs)
    elif model=='bwd':
        clsfr = BwdLinearRegression(tau=tau, offset=offset, evtlabs=evtlabs, center=center, **kwargs)
    elif model=='fwd':
        clsfr = FwdLinearRegression(tau=tau, offset=offset, evtlabs=evtlabs, center=center, **kwargs)
    elif model == 'ridge': # should be equivalent to BwdLinearRegression
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=Ridge(alpha=0,fit_intercept=center), **kwargs)
    elif model == 'lr':
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=LogisticRegression(C=C,fit_intercept=center), labelizeY=True, **kwargs)
    elif model == 'svr':
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=LinearSVR(C=C), **kwargs)
    elif model == 'svc':
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=LinearSVC(C=C), labelizeY=True, **kwargs)
    elif isinstance(model,sklearn.linear_model) or isinstance(model,sklearn.svm):
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, clsfr=model, labelizeY=True, **kwargs)
    elif model=='linearsklearn':
        clsfr = LinearSklearn(tau=tau, offset=offset, evtlabs=evtlabs, **kwargs)
    else:
        raise NotImplementedError("don't  know this model: {}".format(model))

    # do train/test split
    if test_idx is None:
        X_train = X
        Y_train = Y
    else:
        test_ind = np.zeros((X.shape[0],),dtype=bool)
        test_ind[test_idx] = True
        train_ind = np.logical_not(test_ind)
        X_train = X[train_ind,...]
        Y_train = Y[train_ind,...]
        retrain_on_all = True

    # fit the model
    if cv:
        # hyper-parameter optimization by cross-validation
        if tuned_parameters is not None:
            # hyper-parameter optimization with cross-validation

            cv_clsfr = GridSearchCV(clsfr, tuned_parameters)
            print('HyperParameter search: {}'.format(tuned_parameters))
            cv_clsfr.fit(X_train, Y_train)
            means = cv_clsfr.cv_results_['mean_test_score']
            stds = cv_clsfr.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, cv_clsfr.cv_results_['params']):
                print("{:5.3f} (+/-{:5.3f}) for {}".format(mean, std * 2, params))

            clsfr.set_params(**cv_clsfr.best_params_) # Note: **dict -> k,v argument array
        
        if ranks is not None and isinstance(clsfr,MultiCCA):
            # cross-validated rank optimization
            res = clsfr.cv_fit(X_train, Y_train, cv=cv, ranks=ranks, retrain_on_all=retrain_on_all)
        else:
            # cross-validated performance estimation
            res = clsfr.cv_fit(X_train, Y_train, cv=cv, retrain_on_all=retrain_on_all)

        Fy = res['estimator']

    else:
        print("Warning! overfitting...")
        clsfr.fit(X_train,Y_train)
        Fy = clsfr.predict(X, Y, dedup0=True)

    # use the raw scores, i.e. inc model dim, in computing the decoding curve
    rawFy = res['rawestimator'] if 'rawestimator' in res else Fy

    if test_idx is not None:
        # predict on the hold-out test set
        Fy_test = clsfr.predict(X[test_idx,...],Y[test_idx,...])
        # insert into the full results set
        tmp = list(rawFy.shape); tmp[-3]=X.shape[0]
        allFy = np.zeros(tmp,dtype=Fy.dtype)
        allFy[...,train_ind,:,:] = rawFy
        allFy[...,test_ind,:,:] = Fy_test
        rawFy = allFy
        res['rawestimator']=rawFy

    # assess model performance
    score=clsfr.audc_score(rawFy)
    print(clsfr)
    print("score={}".format(score))

    # compute decoding curve
    (dc) = decodingCurveSupervised(rawFy, marginalizedecis=True, minDecisLen=clsfr.minDecisLen, bwdAccumulate=clsfr.bwdAccumulate, priorsigma=(clsfr.sigma0_,clsfr.priorweight), softmaxscale=clsfr.softmaxscale_, nEpochCorrection=clsfr.startup_correction)

    return score, dc, Fy, clsfr, res


def analyse_datasets(dataset:str, model:str='cca', dataset_args:dict=None, loader_args:dict=None, preprocess_args:dict=None, clsfr_args:dict=None, tuned_parameters:dict=None, **kwargs):
    """analyse a set of datasets (multiple subject) and generate a summary decoding plot.

    Args:
        dataset ([str]): the name of the dataset to load
        model (str, optional): The type of model to fit. Defaults to 'cca'.
        dataset_args ([dict], optional): additional arguments for get_dataset. Defaults to None.
        loader_args ([dict], optional): additional arguments for the dataset loader. Defaults to None.
        clsfr_args ([dict], optional): additional aguments for the model_fitter. Defaults to None.
        tuned_parameters ([dict], optional): sets of hyper-parameters to tune by GridCVSearch
    """    
    if dataset_args is None: dataset_args = dict()
    if loader_args is None: loader_args = dict()
    if clsfr_args is None: clsfr_args = dict()
    loader, filenames, _ = get_dataset(dataset,**dataset_args)
    scores=[]
    decoding_curves=[]
    nout=[]
    for i, fi in enumerate(filenames):
        print("{}) {}".format(i, fi))
        try:
            X, Y, coords = loader(fi, **loader_args)
            if preprocess_args is not None:
                X, Y, coords = preprocess(X, Y, coords, **preprocess_args)
            score, decoding_curve, _, _, _ = analyse_dataset(X, Y, coords, model, tuned_parameters=tuned_parameters, **clsfr_args, **kwargs)
            nout.append(Y.shape[-1] if Y.ndim<=3 else Y.shape[-2])
            scores.append(score)
            decoding_curves.append(decoding_curve)
            del X, Y
            gc.collect()
        except Exception as ex:
            print("Error: {}\nSKIPPED".format(ex))
    avescore=sum(scores)/len(scores)
    avenout=sum(nout)/len(nout)
    print("\n--------\n\n Ave-score={}\n".format(avescore))
    # extract averaged decoding curve info
    int_len, prob_err, prob_err_est, se, st = flatten_decoding_curves(decoding_curves)
    print("Ave-DC\n{}\n".format(print_decoding_curve(np.nanmean(int_len,0),np.nanmean(prob_err,0),np.nanmean(prob_err_est,0),np.nanmean(se,0),np.nanmean(st,0))))
    plot_decoding_curve(int_len,prob_err)
    plt.suptitle("{} ({}) AUDC={:3.2f}(n={} ncls={})\nloader={}\nclsfr={}({})".format(dataset,dataset_args,avescore,len(scores),avenout-1,loader_args,model,clsfr_args))
    plt.savefig("{}_decoding_curve.png".format(dataset))
    plt.show()



def analyse_train_test(X:np.ndarray, Y:np.ndarray, coords, splits=1, label:str='', model:str='cca', tau_ms:float=300, fs:float=None,  rank:int=1, evtlabs=None, preprocess_args=None, clsfr_args:dict=None,  **kwargs):    
    """analyse effect of different train/test splits on performance and generate a summary decoding plot.

    Args:
        splits (): list of list of train-test split pairs.  
        dataset ([str]): the name of the dataset to load
        model (str, optional): The type of model to fit. Defaults to 'cca'.
        dataset_args ([dict], optional): additional arguments for get_dataset. Defaults to None.
        loader_args ([dict], optional): additional arguments for the dataset loader. Defaults to None.
        clsfr_args ([dict], optional): additional aguments for the model_fitter. Defaults to None.
        tuned_parameters ([dict], optional): sets of hyper-parameters to tune by GridCVSearch
    """    
    fs = coords[1]['fs'] if coords is not None else fs

    if isinstance(splits,int): 
        # step size for increasing training data split size
        splitsize=splits
        maxsize = X.shape[0]
        splits=[]
        for tsti in range(splitsize, maxsize, splitsize):
            # N.B. triple nest, so list, of lists of train/test pairs
            splits.append( ( (slice(tsti),slice(tsti,None)), ) )


    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)

    # run the train/test splits
    decoding_curves=[]
    labels=[]
    scores=[]
    Ws=[]
    Rs=[]
    for i, cv in enumerate(splits):
        # label describing the folding
        trnIdx = np.arange(X.shape[0])[cv[0][0]]
        tstIdx = np.arange(X.shape[0])[cv[0][1]]
        lab = "Trn {} ({}) / Tst {} ({})".format(len(trnIdx), (trnIdx[0],trnIdx[-1]), len(tstIdx), (tstIdx[0],tstIdx[-1]))
        labels.append( lab )

        print("{}) {}".format(i, lab))
        score, decoding_curve, Fy, clsfr = analyse_dataset(X, Y, coords, model, cv=cv, retrain_on_all=False, **clsfr_args, **kwargs)
        decoding_curves.append(decoding_curve)
        scores.append(score)
        Ws.append(clsfr.W_)
        Rs.append(clsfr.R_)

    # plot the model for each folding
    plt.figure(1)
    for w,r,l in zip(Ws,Rs,labels):
        plt.subplot(211)
        tmp = w[0,0,:]
        sgn = np.sign(tmp[np.argmax(np.abs(tmp))])
        plt.plot(tmp*sgn,label=l)
        plt.subplot(212)
        tmp=r[0,0,:,:].reshape((-1,))
        plt.plot(np.arange(tmp.size)*1000/fs, tmp*sgn, label=l)
    plt.subplot(211)
    plt.grid()
    plt.title('Spatial filter')
    plt.legend()
    plt.subplot(212)
    plt.grid()
    plt.title('Impulse response')
    plt.xlabel('time (ms)')

    plt.figure(2)
    # collate the results and visualize
    avescore=sum(scores)/len(scores)
    print("\n--------\n\n Ave-score={}\n".format(avescore))
    # extract averaged decoding curve info
    int_len, prob_err, prob_err_est, se, st = flatten_decoding_curves(decoding_curves)
    print("Ave-DC\n{}\n".format(print_decoding_curve(np.mean(int_len,0),np.mean(prob_err,0),np.mean(prob_err_est,0),np.mean(se,0),np.mean(st,0))))
    plot_decoding_curve(int_len,prob_err)
    plt.legend( labels + ['mean'] )
    plt.suptitle("{} AUDC={:3.2f}(n={})\nclsfr={}({})".format(label,avescore,len(scores),model,clsfr_args))
    try:
        plt.savefig("{}_decoding_curve.png".format(label))
    except:
        pass
    plt.show()


def flatten_decoding_curves(decoding_curves):
    ''' take list of (potentially variable length) decoding curves and make into a single array '''
    il = np.zeros((len(decoding_curves),decoding_curves[0][0].size))
    pe = np.zeros(il.shape)
    pee = np.zeros(il.shape)
    se = np.zeros(il.shape)
    st = np.zeros(il.shape)
    # TODO [] : insert according to the actual int-len
    for di,dc in enumerate(decoding_curves):
        il_di = dc[0]
        ll = min(il.shape[1],il_di.size)
        il[di,:ll] = dc[0][:ll]
        pe[di,:ll] = dc[1][:ll]
        pee[di,:ll] = dc[2][:ll]
        se[di,:ll] = dc[3][:ll]
        st[di,:ll] = dc[4][:ll] 
    return il,pe,pee,se,st


def debug_test_dataset(X, Y, coords=None, label=None, tau_ms=300, fs=None, offset_ms=0, evtlabs=None, rank=1, model='cca', preprocess_args:dict=None, clsfr_args:dict=None, plotnormFy=False, triggerPlot=False, **kwargs):
    """Debug a data set, by pre-processing, model-fitting and generating various visualizations

    Args:
        X (nTrl,nSamp,d): The preprocessed EEG data
        Y (nTrl,nSamp,nY): The stimulus information
        coords ([type], optional): meta-info about the dimensions of X and Y. Defaults to None.
        label ([type], optional): textual name for this dataset, used for titles and save-file names. Defaults to None.
        tau_ms (int, optional): stimulus-response length in milliseconds. Defaults to 300.
        fs ([type], optional): sample rate of X and Y. Defaults to None.
        offset_ms (int, optional): offset for start of stimulus response w.r.t. stimulus time. Defaults to 0.
        evtlabs ([type], optional): list of types of stimulus even to fit the model to. Defaults to None.
        rank (int, optional): the rank of the model to fit. Defaults to 1.
        model (str, optional): the type of model to fit. Defaults to 'cca'.
        preprocess_args (dict, optional): additional arguments to send to the data pre-processor. Defaults to None.
        clsfr_args (dict, optional): additional arguments to pass to the model fitting. Defaults to None.

    Returns:
        score (float): the cv score for this dataset
        dc (tuple): the information about the decoding curve as returned by `decodingCurveSupervised.py`
        Fy (np.ndarray): the raw cv'd output-scores for this dataset as returned by `decodingCurveSupervised.py` 
        clsfr (BaseSequence2Sequence): the trained classifier
    """    
    fs = coords[1]['fs'] if coords is not None else fs
    if clsfr_args is not None:
        if 'tau_ms' in clsfr_args and clsfr_args['tau_ms'] is not None:
            tau_ms = clsfr_args['tau_ms']
        if 'offset_ms' in clsfr_args and clsfr_args['offset_ms'] is not None:
            offset_ms = clsfr_args['offset_ms']
        if 'evtlabs' in clsfr_args and clsfr_args['evtlabs'] is not None:
            evtlabs = clsfr_args['evtlabs']
        if 'rank' in clsfr_args and clsfr_args['rank'] is not None:
            rank = clsfr_args['rank']
    if evtlabs is None:
        evtlabs = ('re','fe')

    # work on copy of X,Y just in case
    X = X.copy()
    Y = Y.copy()

    tau = int(fs*tau_ms/1000)
    offset=int(offset_ms*fs/1000)    
    times = np.arange(offset,tau+offset)/fs
    
    if coords is not None:
        print("X({}){}".format([c['name'] for c in coords], X.shape))
    else:
        print("X={}".format(X.shape))
    print("Y={}".format(Y.shape))
    print("fs={}".format(fs))

    if preprocess_args is not None:
        X, Y, coords = preprocess(X, Y, coords, **preprocess_args)

    ch_names = coords[2]['coords'] if coords is not None else None
    ch_pos = None
    if coords is not None and 'pos2d' in coords[2]:
        ch_pos = coords[2]['pos2d']
    elif not ch_names is None and len(ch_names) > 0:
        from mindaffectBCI.decoder.readCapInf import getPosInfo
        cnames, xy, xyz, iseeg =getPosInfo(ch_names)
        ch_pos=xy
    if ch_pos is not None:
        print('ch_pos={}'.format(ch_pos.shape))

    # visualize the dataset
    from mindaffectBCI.decoder.stim2event import stim2event
    from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics, idOutliers
    import matplotlib.pyplot as plt

    print("Plot summary stats")
    if Y.ndim == 4: # already transformed
        Yevt = Y
    else: # convert to event
        Yevt = stim2event(Y, axis=-2, evtypes=evtlabs)
    Cxx, Cxy, Cyy = updateSummaryStatistics(X, Yevt[..., 0:1, :], tau=tau, offset=offset)
    plt.figure(11); plt.clf()
    plot_summary_statistics(Cxx, Cxy, Cyy, evtlabs, times, ch_names)
    plt.show(block=False)

    print('Plot global spectral properties')
    plot_grand_average_spectrum(X,axis=-2,fs=fs, ch_names=ch_names)
    plt.show(block=False)

    print("Plot ERP")
    plt.figure(12);plt.clf()
    plot_erp(Cxy, ch_names=ch_names, evtlabs=evtlabs, times=times)
    plt.suptitle("ERP")
    plt.show(block=False)
    plt.pause(.5)
    plt.savefig("{}_ERP".format(label)+".pdf",format='pdf')
    
    # fit the model
    # override with direct keyword arguments
    if clsfr_args is None: clsfr_args=dict()
    clsfr_args['evtlabs']=evtlabs
    clsfr_args['tau_ms']=tau_ms
    clsfr_args['fs']=fs
    clsfr_args['offset_ms']=offset_ms
    clsfr_args['rank']=rank
    #clsfr_args['retrain_on_all']=False # respect the folding, don't retrain on all at the end
    score, res, Fy, clsfr, cvres = analyse_dataset(X, Y, coords, model, **clsfr_args, **kwargs)
    Fe = clsfr.transform(X)


    # get the prob scores, per-sample
    if 'rawestimator' in cvres:   
        rawFy = cvres['rawestimator'] 
        Py = clsfr.decode_proba(rawFy, marginalizemodels=True, minDecisLen=-1)
    else:
        rawFy = cvres['estimator']
        Py = clsfr.decode_proba(Fy, marginalizemodels=True, minDecisLen=-1)

    Yerr = res[5] # (nTrl,nSamp)
    Perr = res[6] # (nTrl,nSamp)

    plot_trial_summary(X, Y, rawFy, fs=fs, Yerr=Yerr[:,-1], Py=Py, Fe=Fe, label=label)
    plt.show(block=False)
    plt.gcf().set_size_inches((15,9))
    plt.savefig("{}_trial_summary".format(label)+".pdf")
    plt.pause(.5)

    plt.figure(14)
    plot_decoding_curve(*res)
    plt.show(block=False)

    plt.figure(19)
    plt.subplot(211)
    plt.imshow(res[5], origin='lower', aspect='auto',cmap='gray', extent=[0,res[0][-1],0,res[5].shape[0]])
    plt.clim(0,1)
    plt.colorbar()
    plt.title('Yerr - correct-prediction (0=correct, 1=incorrect)?')
    plt.ylabel('Trial#')
    plt.grid()
    plt.subplot(212)
    plt.imshow(res[6], origin='lower', aspect='auto', cmap='gray', extent=[0,res[0][-1],0,res[5].shape[0]])
    plt.clim(0,1)
    plt.colorbar()
    plt.title('Perr - Prob of prediction error (0=correct, 1=incorrect)')
    plt.xlabel('time (samples)')
    plt.ylabel('Trial#')
    plt.grid()
    plt.show(block=False)

    if triggerPlot:
        plt.figure(20)
        triggerPlot(X,Y,fs, clsfr=clsfr, evtlabs=clsfr.evtlabs, tau_ms=tau_ms, offset_ms=offset_ms, max_samp=10000, trntrl=None, plot_model=False, plot_trial=True)
        plt.show(block=False)
        plt.savefig("{}_triggerplot".format(label)+".pdf",format='pdf')

    print("Plot Model")
    plt.figure(15)
    if hasattr(clsfr,'A_'):
        plot_erp(factored2full(clsfr.A_, clsfr.R_), ch_names=ch_names, evtlabs=evtlabs, times=times)
        plt.suptitle("fwd-model")
    else:
        plot_erp(factored2full(clsfr.W_, clsfr.R_), ch_names=ch_names, evtlabs=evtlabs, times=times)
        plt.suptitle("bwd-model")
    plt.show(block=False)

    print("Plot Factored Model")
    plt.figure(18)
    plt.clf()
    clsfr.plot_model(fs=fs,ch_names=ch_names)
    plt.savefig("{}_model".format(label)+".pdf")
    plt.show(block=False)
    
    # print("plot Fe")
    # plt.figure(16);plt.clf()
    # plot_Fe(Fe)
    # plt.suptitle("Fe")
    # plt.show()

    # print("plot Fy")
    # plt.figure(17);plt.clf()
    # plot_Fy(Fy,cumsum=True)
    # plt.suptitle("Fy")
    # plt.show()

    if plotnormFy:
        from mindaffectBCI.decoder.normalizeOutputScores import normalizeOutputScores, plot_normalizedScores
        print("normalized Fy")
        plt.figure(20);plt.clf()
        # normalize every sample
        ssFy, scale_sFy, decisIdx, nEp, nY = normalizeOutputScores(Fy, minDecisLen=-1)
        plot_Fy(ssFy,label=label,cumsum=False)
        plt.show(block=False)

        plt.figure(21)
        plot_normalizedScores(Fy[4,:,:],ssFy[4,:,:],scale_sFy[4,:],decisIdx)
    
    plt.show()

    return score, res, Fy, clsfr, rawFy

def plot_trial_summary(X, Y, Fy, Fe=None, Py=None, fs=None, label=None, evtlabs=None, centerx=True, xspacing=10, sumFy=True, Yerr=None):
    """generate a plot summarizing the inputs (X,Y) and outputs (Fe,Fe) for every trial in a dataset for debugging purposes

    Args:
        X (nTrl,nSamp,d): The preprocessed EEG data
        Y (nTrl,nSamp,nY): The stimulus information
        Fy (nTrl,nSamp,nY): The output scores obtained by comping the stimulus-scores (Fe) with the stimulus information (Y)
        Fe ((nTrl,nSamp,nY,nE), optional): The stimulus scores, for the different event types, obtained by combining X with the decoding model. Defaults to None.
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
    fig = plt.figure(figsize=(20,10))
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
                plt.ylim((Felim[0],Felim[1]+Fe.shape[-1]-1))

            # Fy
            if Py is None:
                plt.axes(botax) # no Py, Fy is bottom axis
            else:
                row = 4 if Fe is None else 5
                fig.add_subplot(gs[row,:], sharex=botax)
            plt.plot(times,Fy[ti,:,:], color='.5')
            plt.plot(times,Fy[ti,:,0],'k-')
            if hi==h-1: # only bottom plots
                plt.xlabel('time ({})'.format(xunit))
            else:
                plt.gca().set_xticklabels(())
            if wi==0: # only left most plots
                plt.ylabel("Fy")
            plt.grid(True)
            plt.gca().set_yticklabels(())
            plt.ylim(Fylim)

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
    plt.show(block=False)




def debug_test_single_dataset(dataset:str,filename:str=None,dataset_args=None, loader_args=None, *args,**kwargs):
    """run the debug_test_dataset for a single subject from dataset

    Args:
        dataset ([str]): the dataset to load with get_dataset from `datasets.py`
        filename ([str], optional): a specific filename regular expression to match to process. Defaults to None.

    Returns:
        clsfr [BaseSeq2seq]: the model fitted during the dataset testing
    """    
    if dataset_args is None: dataset_args=dict()
    if loader_args is None: loader_args=dict()
    l,fs,_=get_dataset(dataset,**dataset_args)
    if filename is not None:
        fs = [f for f in fs if re.search(filename,f)]
    X,Y,coords=l(fs[0],**loader_args)
    return debug_test_dataset(X,Y,coords,*args,**kwargs)




def run_analysis():    
    #analyse_datasets("plos_one",loader_args=dict(fs_out=60,stopband=((0,3),(30,-1))),
    #                 model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=3))
    #"plos_one",loader_args=dict(fs_out=120,stopband=((0,3),(45,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1)): ave-score:67
    #"plos_one",loader_args=dict(fs_out=60,stopband=((0,3),(25,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1)): ave-score:67
    #"plos_one",loader_args=dict(fs_out=60,stopband=((0,3),(25,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=3)): ave-score:67
    #"plos_one",loader_args=dict(fs_out=60,stopband=((0,3),(45,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1)): ave-score:674
    #"plos_one",loader_args=dict(fs_out=60,stopband=((0,2),(25,-1))),model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1)): ave-score:61
    #"plos_one",tau_ms=350,evtlabs=('re','fe'),rank=1 : ave-score=72  -- should be 83!!!
    # C: slightly larger freq range helps. rank doesn't.

    #analyse_datasets("lowlands",loader_args=dict(fs_out=60,stopband=((0,5),(25,-1))),
    #                  model='cca',clsfr_args=dict(tau_ms=350,evtlabs=('re','fe')))#,badEpThresh=6))
    #"lowlands",clsfr_args=dict(tau_ms=550,evtlabs=('re','fe'),rank=1,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=56
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=56
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=3,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=51
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=10,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,3),(25,-1))): ave-score=53
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=10,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=42
    #"lowlands",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=10,badEpThresh=6,rcond=1e-6),loader_args=(stopband=((0,5),(25,-1))): ave-score=45
    #analyse_datasets("lowlands",loader_args=dict(passband=None,stopband=((0,5),(25,-1))),clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=1,badEpThresh=4)): ave-score=.47

    #analyse_datasets("lowlands", clsfr_args=dict(tau_ms=350, evtlabs=('re','fe'), rank=3), loader_args=dict(passband=(4, 25),stopband=None))
    #"lowlands", tau_ms=350, evtlabs=('re','fe'),rank=1,loader_args={'passband':(5,25)} : Ave-score=0.64
    #"lowlands", tau_ms=700, evtlabs=('re'), rank=1, loader_args={'passband':(5, 25)}): ave-scre=.50
    #"lowlands", tau_ms=350, evtlabs=('re','fe'), rank=3, loader_args={'passband':(5, 25)}): score=.65
    #"lowlands", tau_ms=350, evtlabs=('re','fe'), rank=3, loader_args={'passband':(3, 25)}): .49
    # C: 5-25, rank=3, re+fe ~300ms
    # Q: why results so much lower now?

    # N.B. ram limits the  tau size...
    # analyse_datasets("brainsonfire",
    #                 loader_args=dict(fs_out=30, subtriallen=10, stopband=((0,1),(12,-1))),
    #                 model='cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=20))
    #"brainsonfire",loader_args=dict(fs_out=30, subtriallen=10, stopband=((0,1),(12,-1))),model='cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=5)) : score=.46
    #"brainsonfire",loader_args=dict(fs_out=30, subtriallen=10, stopband=((0,1),(12,-1))),model='cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=10)) : score=.53

    #analyse_datasets("twofinger",
    #                 model='cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=5), 
    #                 loader_args=dict(fs_out=60, subtriallen=10, stopband=((0,1),(25,-1))))
    #"twofinger",'cca',clsfr_args=dict(tau_ms=600, offset_ms=-300, evtlabs=None, rank=5),loader_args=dict(fs_out=60, subtriallen=10, stopband=((0,1),(25,-1)))): ave-score=.78
    # "twofinger",tau_ms=600, offset_ms=-300, rank=5,subtriallen=10, stopband=((0,1),(25,-1)))): ave-score: .85
    # C: slight benefit from pre-movement data
    
    # Note: max tau=500 due to memory limitation
    #analyse_datasets("cocktail",
    #                 clsfr_args=dict(tau_ms=500, evtlabs=None, rank=5, rcond=1e-4, center=False),
    #                 loader_args=dict(fs_out=60, subtriallen=10, stopband=((0,1),(25,-1))))
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=4,loader_args={'fs_out':60, 'subtriallen':15,'passband':(5,25)}) : .78
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=4,loader_args={'fs_out':60, 'subtriallen':15,'passband':(1,25)}) : .765
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=4,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,25)}) : .765
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=4,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,12)}) : .77
    #analyse_datasets("cocktail",tau_ms=500,evtlabs=None,rank=8,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,12)}) : .818
    #analyse_datasets("cocktail",tau_ms=700,evtlabs=None,rank=8,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,12)}) : .826
    #analyse_datasets("cocktail",tau_ms=700,evtlabs=None,rank=16,loader_args={'fs_out':30, 'subtriallen':15,'passband':(1,12)}) : .854
    #analyse_datasets("cocktail",tau_ms=500, evtlabs=None, rank=15,fs_out=60, subtriallen=10, stopband=((0,1),(25,-1)) : ave-score:.80 (6-subtrials)
    # C: longer analysis window + higher rank is better.  Sample rate isn't too important

    #analyse_datasets("openBMI_ERP",clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5),loader_args=dict(fs_out=30,stopband=((0,1),(12,-1)),offset_ms=(-500,1000)))
    # "openBMI_ERP",tau_ms=700,evtlabs=('re'),rank=1,loader_args=dict(offset_ms=(-500,1000) Ave-score=0.758
    # "openBMI_ERP",tau_ms=700,evtlabs=('re','ntre'),rank=1,loader_args={'offset_ms':(-500,1000)}) Ave-score=0.822
    # "openBMI_ERP",tau_ms=700,evtlabs=('re','ntre'),rank=5,loader_args={'offset_ms':(-500,1000)}) Ave-score=0.894
    #"openBMI_ERP",clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5),loader_args=dict(offset_ms=(-500,1000))): Ave-score=0.894
    # C: large-window, tgt-vs-ntgt  + rank>1 : gives best fit?

    #analyse_datasets("openBMI_SSVEP",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=6),loader_args=dict(offset_ms=(-500,1000)))
    # "openBMI_SSVEP",tau_ms=700,evtlabs=('re'),rank=1,loader_args={'offset_ms':(-500,1000)} : score=.942
    # "openBMI_SSVEP",tau_ms=700,evtlabs=('re'),rank=6,loader_args={'offset_ms':(-500,1000)} : score=.947
    # "openBMI_SSVEP",tau_ms=700,evtlabs=('re','fe'),rank=1,loader_args={'offset_ms':(-500,1000)} : score= :.745
    # "openBMI_SSVEP",tau_ms=350,evtlabs=('re','fe'),rank=6,loader_args={'offset_ms':(-500,1000)} : score= .916
    #analyse_datasets("openBMI_SSVEP",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=6),loader_args=dict(offset_ms=(-500,1000))) : score=.917
    #analyse_datasets("openBMI_SSVEP",clsfr_args=dict(tau_ms=350,evtlabs=('re','fe'),rank=6),loader_args=dict(offset_ms=(-500,1000))) : score=.92
    # "openBMI",tau_ms=600,evtlabs=('re'),rank=1,loader_args={'offset_ms':(-500,1000)} : score==.940
    # C: large-window, re, rank>1 : gives best fit?
    
    #analyse_datasets("p300_prn",loader_args=dict(fs_out=30,stopband=((0,1),(25,-1)),subtriallen=10),
    #                 model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5))
    #"p300_prn",model='cca',loader_args=dict(fs_out=30,stopband=((0,2),(12,-1)),subtriallen=10),clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=15)) : score=.43
    #"p300_prn",model='cca',loader_args=dict(fs_out=60,stopband=((0,2),(25,-1)),subtriallen=10),clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=15)) : score=.47

    #analyse_datasets("mTRF_audio", tau_ms=600, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)})
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(.5, 15)}) : score=.86
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=2, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(.5, 15)}) : score=.85
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)}) : score = .89
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(.5, 25)}) : score = .86
    #analyse_datasets("mTRF_audio", tau_ms=100, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)}) : score= .85
    #analyse_datasets("mTRF_audio", tau_ms=20, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)}) : score=.88
    #analyse_datasets("mTRF_audio", tau_ms=600, evtlabs=None, rank=5, loader_args={'regressor':'spectrogram', 'fs_out':64, 'passband':(5, 25)}) : score=.91
    
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'envelope', 'fs_out':64, 'passband':(.5, 15)}) : score=.77
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'envelope', 'fs_out':64, 'passband':(5, 25)}) : score=.77
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=5, loader_args={'regressor':'envelope', 'fs_out':128, 'passband':(5, 25)}) : score=.78
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=2, loader_args={'regressor':'envelope', 'fs_out':128, 'passband':(5, 25)}) : score=.76
    #analyse_datasets("mTRF_audio", tau_ms=300, evtlabs=None, rank=1, loader_args={'regressor':'envelope', 'fs_out':128, 'passband':(5, 25)}) : score=.69

    # C: spectrogram (over envelope), rank>3, 5-25Hz, short tau is sufficient ~ 100ms

    #analyse_datasets("tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),
    #                 model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5))
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=10) : ave-score:51
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=5) : ave-score:54
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=3) : ave-score:54
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(12,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=3) : ave-score:52
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(12,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','ntre'),rank=10) : ave-score:49
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','anyre'),rank=5) : ave-score:54
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','anyre'),rank=10) : ave-score:50
    #"tactileP3",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re'),rank=5) : ave-score:44
    # C: above chance for 8/9, low rank~3, slow response
    
    #analyse_datasets("tactile_PatientStudy",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),
    #                 model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','anyre'),rank=5))
    #"tactile_PatientStudy",loader_args=dict(fs_out=60,stopband=((0,1),(25,-1))),model='cca',clsfr_args=dict(tau_ms=700,evtlabs=('re','anyre'),rank=5) : ave-score:44

    #analyse_datasets("ninapro_db2",loader_args=dict(stopband=((0,15), (45,55), (95,105), (250,-1)), fs_out=60, nvirt=20, whiten=True, rectify=True, log=True, plot=False, filterbank=None, zscore_y=True),
    #                 model='cca',clsfr_args=dict(tau_ms=40,evtlabs=None,rank=6))
    #"ninapro_db2",loader_args=dict(subtrllen=10, stopband=((0,15), (45,55), (95,105), (250,-1)), fs_out=60, nvirt=40, whiten=True, rectify=True, log=True, plot=False, filterbank=None, zscore_y=True),model='cca',clsfr_args=dict(tau_ms=40,evtlabs=None,rank=20)): ave-score=65 (but dont' believe it)
    #"ninapro_db2",loader_args=dict(subtrllen=10, stopband=((0,15), (45,55), (95,105), (250,-1)), fs_out=60, nvirt=40, whiten=True, rectify=True, log=True, plot=False, filterbank=None, zscore_y=True),model='ridge',clsfr_args=dict(tau_ms=40,evtlabs=None,rank=20)): ave-score=26 (but dont' believe it)

    #analyse_datasets("openBMI_MI",clsfr_args=dict(tau_ms=350,evtlabs=None,rank=6),loader_args=dict(offset_ms=(-500,1000)))
    pass


if __name__=="__main__":
    from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
    import glob
    import os

    #debug_test_single_dataset('p300_prn',dataset_args=dict(label='rc_5_flash'),
    #              loader_args=dict(fs_out=32,stopband=((0,1),(12,-1)),subtriallen=None),
    #              model='cca',tau_ms=750,evtlabs=('re','anyre'),rank=3,reg=.02)

    savefile = None
    savefile = '~/Desktop/mark/mindaffectBCI_*decoder_off.txt'
    savefile = '~/Desktop/mark/mindaffectBCI_*201020_1148.txt'
    #savefile = '~/Desktop/mark/mindaffectBCI_*201014*0940*.txt'
    savefile = "~/Downloads/mindaffectBCI*.txt"
    if savefile is None:
        savefile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')
    
    # default to last log file if not given
    files = glob.glob(os.path.expanduser(savefile))
    savefile = max(files, key=os.path.getctime)

    X, Y, coords = load_mindaffectBCI(savefile, stopband=((45,65),(5.5,25,'bandpass')), fs_out=100)
    label = os.path.splitext(os.path.basename(savefile))[0]

    #cv=[(slice(0,10),slice(10,None))]
    test_idx = slice(10,None) # hold-out test set

    #analyse_dataset(X, Y, coords, tau_ms=400, evtlabs=('re','fe'), rank=1, model='cca', tuned_parameters=dict(rank=[1,2,3,5]))
    #analyse_dataset(X, Y, coords, tau_ms=450, evtlabs=('re','fe'), 
    #                model='cca', test_idx=test_idx, ranks=(1,2,3,5), startup_correction=1, priorweight=200)#, tuned_parameters=dict(startup_correction=(0,10,100), priorweight=(0,50,500)))

    debug_test_dataset(X, Y, coords, tau_ms=450, evtlabs=('re','fe'), 
                      model='cca', test_idx=test_idx, ranks=(1,2,3,5), startup_correction=1, priorweight=200)#, prediction_offsets=(-1,0,1))

    quit()

    # strip weird trials..
    # keep = np.ones((X.shape[0],),dtype=bool)
    # keep[10:20]=False
    # X = X[keep,...]
    # Y = Y[keep,...]
    # coords[0]['coords']=coords[0]['coords'][keep]

    # set of splits were we train on non-overlapping subsets of trnsize.
    if False:
        trnsize=10
        splits=[]
        for i in range(0,X.shape[0],trnsize):
            trn_ind=np.zeros((X.shape[0]), dtype=bool)
            trn_ind[slice(i,i+trnsize)]=True
            tst_ind= np.logical_not(trn_ind)
            splits.append( ( (trn_ind, tst_ind), ) ) # N.B. ensure list-of-lists-of-trn/tst-splits
        #splits=5
        # compute learning curves
        analyse_train_test(X,Y,coords, label='decoder-on. train-test split', splits=splits, tau_ms=450, evtlabs=('re','fe'), rank=1, model='cca', ranks=(1,2,3,5) )

    else:
        debug_test_dataset(X, Y, coords, label=label, tau_ms=450, evtlabs=('re','fe'), rank=1, model='cca', test_idx=test_idx, ranks=(1,2,3,5), startup_correction=100, priorweight=1e6)#, prediction_offsets=(-2,-1,0,1) )
        #debug_test_dataset(X, Y, coords, label=label, tau_ms=400, evtlabs=('re','fe'), rank=1, model='lr', ignore_unlabelled=True)

    #run_analysis()
