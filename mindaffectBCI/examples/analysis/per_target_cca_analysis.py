import numpy as np
import copy
from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
from mindaffectBCI.decoder.devent2stimsequence import find_tgt_obj, zero_nontarget_stimevents
from mindaffectBCI.decoder.utils import askloadsavefile
from mindaffectBCI.decoder.preprocess_transforms import make_preprocess_pipeline
from mindaffectBCI.decoder.preprocess import plot_grand_average_spectrum, plot_trial
from mindaffectBCI.decoder.analyse_datasets import plot_stimseq
from mindaffectBCI.decoder.stim2event import event2output, output2event
from mindaffectBCI.decoder.updateSummaryStatistics import plot_temporal_components, plot_spatial_components, plot_per_output_temporal_components, plot_factoredmodel
import matplotlib.pyplot as plt
import os
from glob import glob


def intrial_only(Y_TSye, npre=2, npost=2):
    """limit the event stream to only the in-trial events, i.e. discard first/last few events

    Args:
        Y_TSye ([type]): [description]
        npre (int, optional): [description]. Defaults to 2.
        npost (int, optional): [description]. Defaults to 2.
    """
    Y_TSye_re = np.diff(Y_TSye, axis=1, prepend=0)>0 # event starts
    # sum over events/outputs
    Y_TS_re = np.any(Y_TSye_re,axis=(-1,-2) if Y_TSye_re.ndim>3 else (-1))
    if npre is not None and npre > 0 :
        Y_TS_pre = np.cumsum(Y_TS_re,axis=1) # number events before this event for each trial
        # logical mask for samples with fewer than npre events
        pre_mask_TS = Y_TS_pre < npre
        # apply the mask
        Y_TSye[pre_mask_TS, ...] = 0
    if npost is not None and npost > 0:
        # sum time-reversed then reverse back to get count of events after sample
        Y_TS_post = np.cumsum(Y_TS_re[:,::-1,...],axis=1)[:,::-1,...]
        # logical mask for samples with fewer than npost events
        post_mask_TS = Y_TS_post < npost
        # apply the mask
        Y_TSye[post_mask_TS, ...] = 0
    return Y_TSye



def expand_pipeline_template(fs, ch_names,
                    artifact_ch=['Fp1', 'Fp2', 'Fpz', 'F9', 'F10'],
                    filterband=None,
                    channels=None,
                    exclude_channels=['ACC_X', 'ACC_Y', 'ACC_Z'],
                    evtlabs=("re", "fe"), fs_out=100, tau_ms=650, offset_ms=0, rank=1, temporal_basis:str=None,
                    badWinThresh=3.5,
                    badEpThresh=4,
                    levels_basis=None,
                    reg=None, rcond=None,
                    cv:int=10, # cross-validation object
                    n_splits=0, train_size=None, test_size=None,
                    clsfr_args:dict=dict(),
                    **kwargs):
    """make a standard analysis pipeline with the given settings

    Args:
        fs ([type]): [description]
        ch_names ([type]): [description]
        artifact_ch (list, optional): [description]. Defaults to ['Fp1','Fp2','F9','F10'].
        filterband (list, optional): [description]. Defaults to [5,25,'bandpass'].
        evtlabs (tuple, optional): [description]. Defaults to ("re","fe").
        fs_out (int, optional): [description]. Defaults to 100.
        tau_ms (int, optional): [description]. Defaults to 650.
        rank (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    if clsfr_args is None: clsfr_args=dict()
    clsfr_args.update({'evtlabs': None, 'tau_ms': tau_ms, "rank": rank, 
                    'temporal_basis': temporal_basis,
                    "badWinThresh": badWinThresh, "badEpThresh": badEpThresh,
                    "reg":reg, "rcond":rcond
                    })

    # setup folding
    if n_splits > 0:
        from sklearn.model_selection import ShuffleSplit
        cv=ShuffleSplit(n_splits=n_splits, train_size=train_size, test_size=test_size, random_state=42)

    if cv is not None:
        clsfr_args.update({'inner_cv':cv})
        clsfr='mindaffectBCI.decoder.model_fitting.MultiCCACV'
    else:
        clsfr= 'mindaffectBCI.decoder.model_fitting.MultiCCA'


    pipeline = [
        ['MetaInfoAdder', {'info':{'fs':fs, 'ch_names':ch_names}}],
        ['ChannelSlicer', {'channels': channels, 'exclude_channels': exclude_channels}],
        #["TrialPlotter",{"plot_trial_number":[0,1,2], "plot_output_idx":0, "suptitle":"EOG rm"}],
        ['NoiseSubspaceDecorrelator', {'ch_names':artifact_ch, 'noise_idx':'idOutliers2', 'filterband':(.5,8,'bandpass')}],
        ['ButterFilterAndResampler', {'filterband':filterband, 'fs_out':fs_out}],
        #['BadChannelRemover', {'thresh':4, 'mode':'remove', 'verb':1}],
        #['ChannelPowerStandardizer'],
        ["TimeShifter",{"timeshift_ms":offset_ms}],
        #"SpatialWhitener",
        #["TrialPlotter",{"plot_trial_number":[0,1,2], "plot_output_idx":0, "suptitle":"ERP: filt"}],
        ['TargetEncoder', {'evtlabs':evtlabs}],     # onset of each level has it's own response
        #["EventRelatedPotentialPlotter",{"suptitle":"filt", "tau_ms":1200, "offset_ms":-200}],
        [clsfr, clsfr_args],
    ]
    return pipeline


def run_pipeline(X_TSd, Y_TSy, label:str, sessdir:str,  plot_pattern:bool=True, npre:int=2, npost:int=2,
                 fs=250, fs_out=100, ch_names=None, cv=5, per_output_irf:bool=False, **kwargs):

    if npre is not None or npost is not None:
        Y_TSy = intrial_only(Y_TSy, npre=npre, npost=npost)

    # expand the pipeline for this dataset
    pipeline = expand_pipeline_template(fs=fs, fs_out=fs_out, ch_names=ch_names, cv=cv, **kwargs)

    ppp = make_preprocess_pipeline(pipeline)
    Xpp_TSd, Ypp_TSye = ppp.fit_modify(X_TSd, Y_TSy, until_stage=-1)

    # extract the final ch_names and fs
    if hasattr(Xpp_TSd, 'info'):
        fs = Xpp_TSd.info.get('fs', fs)
        ch_names = Xpp_TSd.info.get('ch_names', ch_names)

    if per_output_irf and Ypp_TSye.ndim>3:
        print('per-out-irf: {}'.format(Ypp_TSye.shape))
        outputs= np.arange(Ypp_TSye.shape[2])
        # strip empty outputs
        if not np.any(Ypp_TSye[...,0,:]):
            Ypp_TSye = Ypp_TSye[...,1:,:]
            outputs = outputs[1:]
        evtlabs = np.arange(Ypp_TSye.shape[3]) if Ypp_TSye.ndim>2 else [1]
        Ypp_TSye, _, outputs_levels = output2event(Ypp_TSye)
    # get the classifier and it's parameters
    clf = ppp.stages[-1][1]
    clf.cv_fit(Xpp_TSd, Ypp_TSye, score_type='corr')
    W_kd, R_ket, A_kd, I_ket = clf.W_, clf.R_, clf.A_, clf.I_

    # get cross validated goodness of fit score (or re-used cached if available)
    score = clf.score(Xpp_TSd,Ypp_TSye,cv=cv) if not hasattr(clf,'score_cv_') else np.mean(clf.score_cv_)

    if per_output_irf:
        R_kyet = R_ket.reshape((R_ket.shape[0],len(outputs),len(evtlabs),R_ket.shape[-1]))
        plot_spatial_components(A_kd,ch_names=ch_names,ncols=2,colorbar=False)
        plot_per_output_temporal_components(R_kyet,fs=fs,ncols=2,evtlabs=evtlabs,outputs=outputs)
    else:
        # if thresholds is not None:
        #     plot_3factoredmodel_AM(A_kd if plot_pattern else W_kd, R_ket, S_y, outputs_levels, fs=fs_out, ch_names=ch_names, evtlabs=evtlabs,thresholds=thresholds,n_tones=int(len(thresholds)),n_volumes=int(len(outputs_levels)//len(thresholds)))
        clf.plot_model(fs=fs, ch_names=ch_names, plot_pattern=plot_pattern, colorbar=False)

    plt.suptitle("{} {} (N={})\n Score={:5.3f}".format(label, 'pattern' if plot_pattern else 'filter', X_TSd.shape[0], score))

    fname = os.path.join(sessdir, 'model_' + label +'.png')
    print("Saving fig to: {}".format(fname))
    plt.gcf().set_size_inches(16,8)
    plt.savefig(fname, dpi=360)



def run_pipeline_per_target_object_and_level_set(X_TSd, Y_TSy, targets_levels:list, label:str, sessdir,
                    outputs:list=None, **kwargs):
    if outputs is None:
        outputs =  [ "{}".format(o) for o in range(Y_TSy.shape[-1])]
    levels = np.unique(Y_TSy)

    # get the per-trial target info
    Ytgt_TSy, tgt_idx_T = zero_nontarget_stimevents(Y_TSy)
    if np.all([t < 0 for t in tgt_idx_T]):
        print("Warning: Didn't find any valid target info.  Using non-target object stimuli....")
        Ytgt_TSy = Y_TSy.copy()
    print("Trial Targets:{}".format(tgt_idx_T))

    # special cases for per-target or per-level analysis
    if targets_levels in ('per_target_object', 'per_target', 'per_output'):
        targets_levels = [[[t], None] for t in range(1, Ytgt_TSy.shape[-1]) if t >= 0]

    elif targets_levels == 'per_level':
        targets_levels = [[None, [l]] for l in levels if l > 0]

    elif targets_levels in ('per_target_object_and_level', 'per_object_and_level', 'per_output_and_level', 'per_target_and_level'):
        targets_levels = [[[t], [l]] for t in range(1, Ytgt_TSy.shape[-1]) for l in np.unique(Y_TSy) if l > 0]

    if targets_levels is None:
        return 

    # run for each target,level combination
    for ti, li in targets_levels:
        # make sure these are lists
        if ti is not None and not hasattr(ti, '__iter__'): ti = [ti]
        if li is not None and not hasattr(li, '__iter__'): li = [li]

        # get the target subset
        # label for this target set
        tlabel = ""
        if ti is not None:
            tlabel = "o{}".format(",".join([str(outputs[t]) for t in ti]))

        # get the target objects in this target group
        Ytrn_TSy = Ytgt_TSy[..., ti] if ti is not None else Ytgt_TSy

        # get the level subset
        llabel = ""
        if li is not None:
            llabel = "l{}".format(",".join([str(levels[l]) for l in li]))

            # make a label indicator with only the given levels
            tmp = Ytrn_TSy
            Ytrn_TSy = np.zeros_like(tmp)
            for l in li:
                Ytrn_TSy[tmp == l] = l


        # strip all zero trials for speed
        trn_idx = np.any(Ytrn_TSy > 0, axis=(-1, -2))
        if not np.any(trn_idx):
            print("No data for targets {}. Skipping".format(ti))
            continue
        Xtrn_TSd, Ytrn_TSy = (X_TSd[trn_idx, ...].copy(), Ytrn_TSy[trn_idx, ...].copy())

        # merge multiple objects if needed before training the model
        Ytrn_TSy = np.max(Ytrn_TSy, axis=-1, keepdims=True)

        # run the pipeline
        if tlabel and llabel :
            rlabel = tlabel+"."+llabel+"_"+label
        elif tlabel and not llabel :
            rlabel = tlabel + "_" + label
        elif llabel and not tlabel :
            rlabel = llabel + "_" + label
        else:
            rlabel = label
        run_pipeline(Xtrn_TSd, Ytrn_TSy, rlabel, sessdir, **kwargs)


def load_preprocess_and_epoch(filename:str, sessdir:str=None, label:str=None, 
                        ch_names:list=None, artifact_ch:list=None, 
                        l_freq:float=None, h_freq:float=None, fs:float=None, 
                        subtriallen_ms:float=15000, visualize:bool=True):
    # for saving the plots
    if sessdir is None:
        sessdir = os.path.dirname(filename)

    dataset_args = dict(filterband=[[45 ,65]])#[ [45,55], [95,105], [145,155], [195,205], [245,255] ]) #dict(filterband=[],((45,65),(2,25,'bandpass')), fs_out = 100, zero_before_stimevents=False)
    # inlucde the low/high pass
    if l_freq is not None and h_freq is not None:
        dataset_args['filterband'].append([l_freq,h_freq,'bandpass'])
    elif l_freq is not None:
        dataset_args['filterband'].append([0,l_freq])
    elif h_freq is not None:
        dataset_args['filterband'].append([h_freq,-1])
    if ch_names is not None:
        dataset_args['ch_names'] = ch_names
    if subtriallen_ms is not None:
        dataset_args['subtriallen_ms'] = subtriallen_ms
    if fs is not None:
        dataset_args['fs_out'] = fs

    X,Y,coords = load_mindaffectBCI(filename, **dataset_args)
    
    fs = coords[1]['fs']
    ch_names = coords[2]['coords']
    # output is: X=eeg, Y=stimulus, coords=meta-info about dimensions of X and Y
    print("EEG: X({}){} @{}Hz {}".format([c['name'] for c in coords],X.shape,fs,ch_names))                            
    print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:-1]]+['output'],Y.shape))

    if visualize:
        plt.clf()
        try:
            plot_grand_average_spectrum(X, fs=fs, ch_names=ch_names, log=True, show=None)
            plt.savefig(os.path.join(sessdir, 'psd_{}.png'.format(label)))
        except:
            pass

        try:
            plt.clf()
            plot_trial(X[-2:,...], Y[-2:,...], fs=fs, show=None)
            plt.savefig(os.path.join(sessdir, 'epochs_{}.png'.format(label)))
        except:
            pass

        try:
            plt.clf()
            plot_stimseq(Y[[1],...], fs=fs, show=None)
            plt.savefig(os.path.join(sessdir, 'events_{}.png'.format(label)))
        except:
            pass

        if Y.shape[1] > 4*fs:
            plt.clf()
            plot_stimseq(Y[[1],int(2*fs):int(4*fs),...], fs=fs, show=None)
            plt.savefig(os.path.join(sessdir, 'events_{}_2s.png'.format(label)))

        plt.close('all')

    # TODO: bad channel removal, bad_epoch removal, EOG removal, channel interpolation

    return X,Y,coords

import time
def print_progress_and_timetogo(i,n,t0):
    elapsed=time.time()-t0
    done = i/n
    print("\r{:d}% {:d} / {:d} in {:4.1f}s  est {:4.1f}s total {:4.1f}s remaining".format(int(100*done), i, n,elapsed,elapsed/max(1e-8,done),elapsed/max(1e-8,done) - elapsed))

def run(exptdir:str=None, label:str='cca', filematch:str='mindaffectBCI*.txt', 
        targets_levels:list=None, # default to per-target analysis
        l_freq=1, h_freq=20,  fs:float=None,
        trn_idx=slice(None,None),
        ch_names:list=None, 
        filterband=[3,35,'bandpass'], fs_out=100,
        evtlabs=("re","fe"),tau_ms=650,rank=3, temporal_basis:str=None, cv:int=10, clsfr_args:dict=dict(),
        plot_pattern:bool=True, per_output_irf=False,
        **kwargs
        ):

    # load savefile
    if exptdir is None:
        exptdir = askloadsavefile(initialdir=os.getcwd(),filetypes='dir')
        print(exptdir)

    # get the list of save files to load
    if not os.path.exists(exptdir): # add the data-root prefix
        from mindaffectBCI.decoder.offline.datasets import get_dataroot
        exptdir = os.path.join(get_dataroot(),exptdir)
    filelist = glob(os.path.join(exptdir,'**',filematch),recursive=True)
    # sort to process the newest first, based on modify time
    filelist.sort(key=lambda f:os.path.getmtime(f),reverse=True)
    print("Found {} matching data files\n".format(len(filelist)))
    print(filelist)

    # get the prefix to strip for short names
    commonprefix = os.path.dirname(os.path.commonprefix(filelist))

    # run each file at a time & save per-condition average response for grand-average later
    t0 = time.time()
    subj_condition_erp = []
    for fi,filename in enumerate(filelist):
        sessdir = os.path.dirname(filename)
        flabel = os.path.split(filename[len(commonprefix):])
        flabel = flabel[0] if flabel[0] else flabel[1]
        flabel = flabel.replace('\\','_').replace('/','_')
        flabel = flabel + '_' + label

        # location to save the figures
        savedir = os.path.join(sessdir,label)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        try:
            print("\n\n------------------------------\n{}\n---------------------\n".format(filename))
            print_progress_and_timetogo(fi,len(filelist),t0)
            X,Y,coords = load_preprocess_and_epoch(filename,sessdir=savedir,label=flabel,
                                                ch_names=ch_names, 
                                                l_freq=l_freq, h_freq=h_freq, fs=fs)
            plt.close('all')
        except:
            print("Error loading {}".format(filename))
            import traceback
            traceback.print_exc()
            continue

        # training subset
        if trn_idx is not None: 
            X = X[trn_idx,...]
            Y = Y[trn_idx,...]

        # extract meta-info and fill out the template pipeline
        fs = coords[1]['fs']
        ch_names = coords[2]['coords']
        pipeline = expand_pipeline_template(fs,ch_names,
                                    filterband=filterband,
                                    evtlabs=evtlabs, fs_out=fs_out, tau_ms=tau_ms, rank=rank, temporal_basis=temporal_basis, **kwargs)
        print("Analysis pipeline:\n    " + "\n    ".join([str(p) for p in pipeline]))

        # first run with all the targets
        run_pipeline(X, Y, flabel, savedir,
                    fs=fs,ch_names=ch_names,
                    filterband=filterband,
                    evtlabs=evtlabs, fs_out=fs_out, tau_ms=tau_ms, rank=rank, temporal_basis=temporal_basis,
                    clsfr_args=clsfr_args, cv=cv, 
                    per_output_irf=per_output_irf,
                    **kwargs)

        # now run with per-target set sub-groups
        run_pipeline_per_target_object_and_level_set(X, Y, targets_levels, flabel, savedir, 
                    fs=fs,ch_names=ch_names,
                    filterband=filterband,
                    evtlabs=evtlabs, fs_out=fs_out, tau_ms=tau_ms, rank=rank, temporal_basis=temporal_basis,
                    clsfr_args=clsfr_args, cv=cv, 
                    per_output_irf=per_output_irf,
                    **kwargs)

        plt.close('all')
        del X,Y,coords


def parse_args():
    import argparse
    import json
    parser=argparse.ArgumentParser()
    parser.add_argument('--exptdir', type=str, default=None)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--tau_ms', type=int, help='duration of the analysis window in milliseconds.', default=650)
    parser.add_argument('--offset_ms', type=int, help='offset w.r.t. trigger of analysis window.', default=0)
    parser.add_argument('--evtlabs', type=str, help='comma separated list of stimulus even types to use. Default re,fe', default="hoton_re")
    parser.add_argument('--per_output_irf',type=lambda a:a.lower()=='1',default=True)
    parser.add_argument('--temporal_basis', type=str, default="f2,10")#"4drbf2,4,9,-1,.5,.5,1,1"
    parser.add_argument('--fs_out', type=float, default=None)
    parser.add_argument('--fs', type=float, default=62.5)
    parser.add_argument('--rank', type=int, default=1)
    parser.add_argument('--reg', type=float, default=None)
    parser.add_argument('--rcond', type=float, default=None)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=5)
    parser.add_argument('--n_splits', type=int, default=20)
    parser.add_argument('--trn_idx',type=int,help='max number trials to use for training',default=None)
    parser.add_argument('--l_freq', type=float, default=1.9)
    parser.add_argument('--h_freq', type=float, default=25)
    parser.add_argument('--outputs',type=json.loads,help='list of outputs to fit',default=None)
    parser.add_argument('--plw', type=json.loads, help='perlevelweight', default=True)
    parser.add_argument('--badWinThresh', type=float, default=3.5)
    parser.add_argument('--badEpThresh', type=float, default=None)
    parser.add_argument('--channels', type=json.loads, default=None)

    args=parser.parse_args()

    return args




if __name__=='__main__':
    args=parse_args()
    # get the experiment dir to run on
    exptdir = askloadsavefile(initialdir=os.getcwd(),filetypes='dir') if args.exptdir is None else args.exptdir
    print(exptdir)

    # # run an all-target and individual per-target analysis
    # run(exptdir=exptdir, label='cca', 
    #     targets_levels="per_target_and_level",
    #     l_freq=3, h_freq=45, 
    #     ch_names=None, 
    #     artifact_ch=['Fp2','Fp1','F9','F10'],
    #     filterband=[2,25,'bandpass'],
    #     evtlabs=("re"), fs_out=100, tau_ms=650, rank=1)

    cv=None # don't cross-validate

    # make a summary label to describe this configuration
    label='cca_{}trn_{:d}-{:d}ms_{}-{}hz_fs{:3.1f}'.format(
        args.train_size, args.offset_ms, args.offset_ms + args.tau_ms, 
        args.l_freq if args.l_freq is not None else '',
        args.h_freq if args.h_freq is not None else '', 
        args.fs)
    if args.temporal_basis is not None:
        label = label + '_t{}'.format(args.temporal_basis)
    if args.channels is not None:
        label=label + "_{:d}ch".format(len(args.channels))
    if args.rank > 1:
        label=label + "_k{:d}".format(args.rank)
    if args.outputs is not None:
        label=label + "_o{}-{}".format(args.outputs[0],args.outputs[-1])

    print('poif: {}'.format(args.per_output_irf))

    # Audiometric Testing
    # run an all-target and individual per-target analysis with IRF for each level
    run(exptdir=exptdir, label=label,
        trn_idx=args.trn_idx,
        targets_levels=None, #"per_target",
        per_output_irf=args.per_output_irf, # fit IRF for each output/level combination
        l_freq=args.l_freq, h_freq=args.h_freq, fs=args.fs,
        ch_names=None, 
        channels=args.channels,
        filterband=None, fs_out=None,
        cslfr_args={},
        evtlabs=args.evtlabs, tau_ms=args.tau_ms, offset_ms=args.offset_ms, rank=args.rank,  
        temporal_basis=args.temporal_basis,#'winfourier3,12',
        cv=cv,  n_splits=args.n_splits, train_size=args.train_size, test_size=args.test_size,
        badWinThresh=args.badWinThresh,
        badEpThresh=args.badEpThresh,
        reg=args.reg, rcond=args.rcond, 
        plot_pattern=True)

    # # c-VEP settings
    # run(exptdir=exptdir, label='cca', #trn_idx=slice(0,10),
    #     targets_levels=None, #"per_target",
    #     l_freq=2, h_freq=None, fs=None,
    #     ch_names=None, 
    #     artifact_ch=['Fp2', 'Fp1', 'F9', 'F10'],
    #     filterband=None, fs_out=100,
    #     evtlabs=("re","fe"), tau_ms=450, rank=1, temporal_basis='winfourier10',
    #     plot_pattern=True)

    # # visual-acuity settings
    # run(exptdir=exptdir, label='cca_re12', #trn_idx=slice(0,10),
    #     cv=cv,
    #     targets_levels=None,#"per_target",
    #     per_output_irf=True, npre=2, npost=2, 
    #     l_freq=3, h_freq=None, fs=None,
    #     ch_names=None, 
    #     artifact_ch=['Fp2', 'Fp1'],
    #     filterband=None, fs_out=None,
    #     evtlabs=("re1,2"),
    #     tau_ms=300, rank=1, temporal_basis='winfourier10',
    #     plot_pattern=True)
