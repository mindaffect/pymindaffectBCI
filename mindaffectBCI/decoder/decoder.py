#!/usr/bin/env python3
import numpy as np
from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, butterfilt_and_downsample
from mindaffectBCI.utopiaclient import NewTarget, Selection, ModeChange, PredictedTargetDist, PredictedTargetProb
from mindaffectBCI.decoder.devent2stimsequence import devent2stimSequence, upsample_stimseq
from mindaffectBCI.decoder.model_fitting import BaseSequence2Sequence, MultiCCA
from mindaffectBCI.decoder.decodingSupervised import decodingSupervised
from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised, plot_decoding_curve
from mindaffectBCI.decoder.scoreOutput import dedupY0
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_summary_statistics, plot_erp
import os

PREDICTIONPLOTS = False
CALIBRATIONPLOTS = False

def get_trial_start_end(msgs, start_ts=None):
    '''get the start+end times of the trials in a utopia message stream'''
    trials = []
    keeplast = False
    for mi, m in enumerate(msgs):
        #print("msg={}".format(m))
        # process begin trail messages, N.B. after end-trial!
        if m.msgID == NewTarget.msgID:
            if start_ts is None:
                start_ts = m.timestamp
                print("NT: tr-bgn={}".format(start_ts))
            else: # treat as end of sequence + start next sequence
                trials.append((start_ts, m.timestamp))
                start_ts = m.timestamp
                print("NT: tr-end={}".format(m.timestamp))
                print("NT: tr-bgn={}".format(m.timestamp))
            
        # process the end-trial messages
        if m.msgID == Selection.msgID:
            if start_ts is not None:
                trials.append((start_ts, m.timestamp))
                start_ts = m.timestamp
                print("SL: tr-end={}".format(m.timestamp))
            else:
                print("Selection without start")
        
        if m.msgID == ModeChange.msgID:
            if start_ts is not None:
                trials.append((start_ts,m.timestamp))
                start_ts=m.timestamp
                print("MC: tr-end={}".format(m.timestamp))
            print("mod-chg={}".format(m))
            # mark this message a *not* processed so it gets to the main loop
            keeplast=True
            break
    # make list un-processed messages
    if msgs and keeplast:
        msgs = msgs[mi:]
        #print("unproc messages: {}".format(msgs))
    # return trial start/end + non-processed messages
    # N.B. start_ts is not None if trail start without end..
    return (trials, start_ts, msgs)

def getCalibration_dataset(ui):
    ''' extract a labelled dataset from the utopiaInterface, which are trials between modechange messages '''
    # run until we get a mode change gathering training data in trials
    dataset = []
    start_ts = None  # assume we have just started the first trial?
    isCalibrating = True
    while isCalibrating:

        # get new messages from utopia-hub
        newmsgs, _, _ = ui.update()
        #  print("Extact_msgs:"); print("{}".format(newmsgs))

        # incremental extract trial limits
        trials, start_ts, newmsgs = get_trial_start_end(newmsgs, start_ts)

        # extract any complete trials data/msgs
        for (bgn_ts, end_ts) in trials:
            # N.B. be sure to make a copy so isn't changed outside us..
            data = ui.extract_data_segment(bgn_ts, end_ts)
            stimulus = ui.extract_stimulus_segment(bgn_ts, end_ts)
            print("Extract trl: {}->{}: data={} stim={}".format(bgn_ts, end_ts, data.shape, stimulus.shape))
            dataset.append((data, stimulus))

        # check for end-calibration messages
        for i, m in enumerate(newmsgs):
            if m.msgID == ModeChange.msgID:
                isCalibrating = False
                # return unprocessed messages including the mode change
                #  print('cal pushback: {}'.format(newmsgs[i:]))
                ui.push_back_newmsgs(newmsgs[i:])
                break

    return dataset

def dataset_to_XY_ndarrays(dataset):
    """convert a dataset, consisting of a list of pairs of time-stamped data and stimulus events, to 3-d matrices of X=(trials,samples,channels) and Y=(trials,samples,outputs)

    Args:
        dataset ([type]): list of pairs of time-stamped data and stimulus events

    Returns:
        X (tr,samp,d): the per-trial data
        Y (tr,samp,nY): the per-trial stimulus, with sample rate matched to X
        X_ts (tr,samp): the time-stamps for the smaples in X
        Y_ts (tr,samp): the time-stamps for the stimuli in Y
    """ 
    # get length of each trial
    trlen = [trl[0].shape[0] for trl in dataset]
    trstim = [trl[1].shape[0] for trl in dataset]
    print("Trlen: {}".format(trlen))
    print("Trstim: {}".format(trstim))
    # set array trial length to 90th percential length
    trlen = int(np.percentile(trlen, 75))
    trstim = max(20, int(np.percentile(trstim, 75)))
    # filter the trials to only be the  ones long enough to be worth processing
    dataset = [d for d in dataset if d[0].shape[0] > trlen//2 and d[1].shape[0] > trstim//2]
    if trlen == 0 or len(dataset) == 0:
        return None, None, None, None

    # map to single fixed size matrix + upsample stimulus to he EEG sample rate
    Y = np.zeros((len(dataset), trlen, 256), dtype=dataset[0][1].dtype)
    X = np.zeros((len(dataset), trlen, dataset[0][0].shape[-1]-1), dtype=dataset[0][0].dtype)  # zero-padded data, w/o time-stamps
    X_ts = np.zeros((len(dataset),trlen),dtype=int)
    Y_ts = np.zeros((len(dataset),trlen),dtype=int)
    for ti, (data, stimulus) in enumerate(dataset):
        # extract the data & remove the timestamp channel and insert into the ndarray
        # guard for slightly different sizes..
        if X.shape[1] <= data.shape[0]:
            X[ti, :, :] = data[:X.shape[1], :-1]
            X_ts[ti, :] = data[:X.shape[1], -1]
        else:  # pad end with final value
            X[ti, :data.shape[0], :] = data[:, :-1]
            X[ti, data.shape[0]:, :] = data[-1, :-1]
            X_ts[ti, :data.shape[0]] = data[:, -1]

        # upsample stimulus to the data-sample rate and insert into ndarray
        data_ts = data[:, -1]  # data timestamp per sample
        stimulus_ts = stimulus[:, -1]  # stimulus timestamp per stimulus event
        stimulus, data_i = upsample_stimseq(data_ts, stimulus[:, :-1], stimulus_ts)
        # store -- compensating for any variable trial lengths.
        if Y.shape[1] < stimulus.shape[0]: # long trial
            Y[ti, :, :] = stimulus[:Y.shape[1], :]
        else: # short trial
            Y[ti, :stimulus.shape[0], :] = stimulus
        # record stim-ts @ this data_ts
        tmp = data_i < Y.shape[1]
        Y_ts[ti,data_i[tmp]] = stimulus_ts[tmp] 

    return X, Y, X_ts, Y_ts


def strip_unused(Y):
    """ strip unused outputs from the stimulus info in Y """
    used_y = np.any(Y.reshape((-1, Y.shape[-1])), 0)
    used_y[0] = True # ensure objID=0 is always used..
    Y = Y[..., used_y]
    return Y, used_y


def doCalibrationSupervised(ui: UtopiaDataInterface, clsfr: BaseSequence2Sequence, 
                            cv=2, calfn="calibration_data.pk", fitfn="fit_data.pk", previous_dataset=None, ranks=(1,2,3,5)):
    """
    do a calibration phase = basically just extract  the training data and train a classifier from the utopiaInterface

    Args:
        ui (UtopiaDataInterface): buffered interface to the data and stimulus streams
        clsfr (BaseSequence2Sequence): the classifier to use to fit a model to the calibration data
        cv (int, optional): the number of cross-validation folds to use for model generalization performance estimation. Defaults to 2.
        calfn (str, optional): filename to save the extracted calibration data. Defaults to "calibration_data.pk".
        fitfn (str, optional): filename to save the fitted model. Defaults to "fit_data.pk".
        previous_dataset ([type], optional): data-set from a previous calibration run, used to accumulate data over subsequent calibrations. Defaults to None.
        ranks (tuple, optional): a list of model ranks to optimize as hyperparameters. Defaults to (1,2,3,5).

    Returns:
        dataset [type]: the gathered calibration data
        X : the calibration data as a 3-d array (tr,samp,d)
        Y : the calibration stimulus as a 3-d array (tr,samp,num-outputs)
    """    
    X = None
    Y = None
    dataset = getCalibration_dataset(ui)
    if previous_dataset is not None: # combine with the old calibration data
        dataset.extend(previous_dataset)
    if dataset:
        # convert msgs -> to nd-arrays
        X, Y, X_ts, Y_ts = dataset_to_XY_ndarrays(dataset)

        try:
            import pickle
            pickle.dump(dict(X=X, Y=Y, X_ts=X_ts, Y_ts=Y_ts, fs=ui.fs),
                        open('calibration_data.pk','wb'))
        except:
            print('Error saving cal data')

        # guard against empty training dataset
        if X is None or Y is None :
            return None, None, None
        Y, used_idx = strip_unused(Y)
        
        # now call the clsfr fit method, on the true-target info
        try:
            print("Training dataset = ({},{})".format(X.shape, Y.shape))
            cvscores = clsfr.cv_fit(X, Y, cv=cv, ranks=ranks)
            score = np.mean(cvscores['test_score'])
            print("clsfr={} => {}".format(clsfr, score))
        except:
            import traceback
            traceback.print_exc()
            return None, None, None

        decoding_curve = decodingCurveSupervised(cvscores['estimator'], nInt=(10, 10),
                                      priorsigma=(clsfr.sigma0_, clsfr.priorweight),
                                      softmaxscale=clsfr.softmaxscale_)
        # extract the final estimated performance
        #print("decoding curve {}".format(decoding_curve[1]))
        #print("score {}".format(score))
        perr = decoding_curve[1][-1] if len(decoding_curve)>1 else 1-score
        if CALIBRATIONPLOTS:
            try:
            #if True:
                import matplotlib.pyplot as plt
                plt.figure(1)
                clsfr.plot_model(fs=ui.fs, ncol=3) # use 3 cols, so have: spatial, temporal, decoding-curve
                plt.subplot(1,3,3) # put decoding curve in last sub-plot
                plot_decoding_curve(*decoding_curve)
                plt.suptitle("Model + Decoding Performance")
                #  from analyse_datasets import debug_test_dataset
                #  debug_test_dataset(X,Y,None,fs=ui.fs)
                plt.figure(3) # plot the CCA info
                Y_true = clsfr.stim2event(Y)
                Y_true = Y_true[...,0:1,:]
                Cxx, Cxy, Cyy = updateSummaryStatistics(X,Y_true,tau=clsfr.tau)
                plot_summary_statistics(Cxx,Cxy,Cyy,clsfr.evtlabs,fs=ui.fs)
                plt.suptitle("Summary Statistics")
                try:
                    import pickle
                    pickle.dump(dict(Cxx=Cxx, Cxy=Cxy, Cyy=Cyy, evtlabs=clsfr.evtlabs, fs=ui.fs),
                                open('summary_statistics.pk','wb'))
                except:
                    print('Error saving cal data')
                plt.figure(4)
                plot_erp(Cxy,evtlabs=clsfr.evtlabs,fs=ui.fs)
                plt.suptitle("Event Related Potential (ERP)")
                plt.show(block=False)
                # save figures
                logsdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/')
                plt.figure(1)
                plt.savefig(os.path.join(logsdir,'model_{}.png'.format(uname)))
                #plt.figure(2)
                #plt.savefig(os.path.join(logsdir,'decoding_curve_{}.png'.format(uname)))
                plt.figure(3)
                plt.savefig(os.path.join(logsdir,'summary_statistics_{}.png'.format(uname)))
                plt.figure(4)
                plt.savefig(os.path.join(logsdir,'erp_{}.png'.format(uname)))
            except:
                pass

        # send message with calibration performance score
        ui.sendMessage(PredictedTargetProb(ui.stimulus_timestamp, 0, perr))
        
    return dataset, X, Y


def doPrediction(clsfr: BaseSequence2Sequence, data, stimulus, prev_stimulus=None):
    ''' given the current trials data, apply the classifier and decoder to make target predictions '''
    X = data[:, :-1]
    X_ts = data[:, -1]
    Y = stimulus[:, :-1]
    Y_ts = stimulus[:, -1]
    if X_ts.size == 0 or Y_ts.size == 0: # fast path empty inputs
        return None
    # strip outputs that we don't use, to save compute time
    Y, used_idx = strip_unused(Y)
    # strip the true target info if it's a copy, so it doesn't mess up Py computation
    Y = dedupY0(Y, zerodup=False, yfeatdim=False)
    # up-sample Y to the match the rate of X
    # TODO[]: should this happen in the data-interface?
    Y, _ = upsample_stimseq(X_ts, Y, Y_ts)
    # predict on X,Y without the time-stamp info
    Fy_1 = clsfr.predict(X, Y, prevY=prev_stimulus)
    # map-back to 256
    Fy = np.zeros(Fy_1.shape[:-1]+(256,),dtype=Fy_1.dtype)
    Fy[..., used_idx] = Fy_1
    return Fy


def combine_Ptgt(pvals_objIDs):
    pvals = [p[0] for p in pvals_objIDs] 
    objIDs = [p[1] for p in pvals_objIDs]
    if not all(np.isequal(objIDs[0], oi) for oi in objIDs):
        print("Warning combination only supported for fixed output set currently!")
        return pvals[-1], objIDs[-1]
    pvals = np.hstack(pvals)  # (nBlk,nObj)
    # coorected combination
    Ptgt = softmax(np.sum(np.log(pvals))/np.sqrt(pvals.shape[0]))
    return Ptgt, objIDs


def send_prediction(ui: UtopiaDataInterface, Ptgt, used_idx=None, timestamp=-1):
    """Send the current prediction information to the utopia-hub

    Args:
        ui (UtopiaDataInterface): the interface to the data-hub
        Ptgt ([type]): the current distribution of target probabilities over outputs
        used_idx ([type], optional): a set of output indices currently used. Defaults to None.
        timestamp (int, optional): time stamp for which this prediction applies. Defaults to -1.
    """    
    #print(" Pred= used_idx:{} ptgt:{}".format(used_idx,Ptgt))
    # N.B. for network efficiency, only send for non-zero probability outputs
    nonzero_idx = np.flatnonzero(Ptgt)
    # ensure a least one entry
    if nonzero_idx.size == 0: 
        nonzero_idx = [0]
    Ptgt = Ptgt[nonzero_idx]
    if used_idx is None:
        used_idx = nonzero_idx
    else:
        if np.issubdtype(used_idx.dtype, np.bool): # logical->index
            used_idx = np.flatnonzero(used_idx)
        used_idx = used_idx[nonzero_idx]
    #  print(" Pred= used_idx:{} ptgt:{}".format(used_idx,Ptgt))
    # send the prediction messages, PredictedTargetProb, PredictedTargetDist
    y_est_idx = np.argmax(Ptgt, axis=-1)
    # most likely target and the chance that it is wrong
    if Ptgt[y_est_idx] == 1.0 :
        print("P==1?") 
    else:
        ptp = PredictedTargetProb(timestamp, used_idx[y_est_idx], 1-Ptgt[y_est_idx])
        print(" Pred= {}".format(ptp))
        ui.sendMessage(ptp)
    # distribution over all *non-zero* targets
    ui.sendMessage(PredictedTargetDist(timestamp, used_idx, Ptgt))
    

def doPredictionStatic(ui: UtopiaDataInterface, clsfr: BaseSequence2Sequence, model_apply_type='trial', timeout_ms=None, block_step_ms=100, maxDecisLen_ms=8000):
    """ 
    do the prediction stage = basically extract data/msgs from trial start and generate a prediction from them '''

    Args:
        ui (UtopiaDataInterface): buffered interface to the data and stimulus streams
        clsfr (BaseSequence2Sequence): the trained classification model
        maxDecisLen_ms (float, optional): the maximum amount of data to use to make a prediction, i.e. prediction sliding window size.  Defaults to 8000

    """
    if not clsfr.is_fitted():
        print("Warning: trying to predict without training classifier!")
        return

    # TODO []: Block based prediction is slightly slower?  Why?
    if timeout_ms is None:
        timeout_ms = block_step_ms
    
    # start of the data block to apply the model to
    block_start_ts = ui.data_timestamp
    overlap_samp = clsfr.tau
    overlap_ms = overlap_samp * 1000 / ui.fs
    maxDecisLen_samp = int(maxDecisLen_ms * ui.fs / 1000)
    Fy = None # (1,nSamp,nY):float score for each output for each sample
    trial_start_ts = None
    isPredicting = True
    # run until we get a mode change gathering training data in trials
    while isPredicting:
        # get new messages from utopia-hub
        newmsgs, ndata, nstim = ui.update(timeout_ms=timeout_ms,mintime_ms=timeout_ms//2)

        # TODO[]: Fix to not re-process the same data if no new stim to be processed..
        if len(newmsgs) == 0 and nstim == 0 and ndata == 0:
            continue

        # get the timestamp for the last data which it is valid to apply the model to,
        # that is where have enough data to include a complete response for this stimulus
        # Note: can't just use last data, incase stimuli are lagged w.r.t. data
        # also, prevents processing data for which are not stimulus events to compare with
        valid_end_ts = min(ui.stimulus_timestamp + overlap_ms, ui.data_timestamp)

        # incremental extract trial limits
        otrial_start_ts = trial_start_ts
        trials, trial_start_ts, newmsgs = get_trial_start_end(newmsgs, trial_start_ts)

        # change in trial-start -> end-of-trial / start new trial detected
        if not trial_start_ts == otrial_start_ts:
            print("New trial! tr_start={}".format(trial_start_ts))
            if PREDICTIONPLOTS and Fy is not None:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(10)
                    plt.cla()
                    print("Fy={}".format(Fy.shape))
                    plt.plot(np.cumsum(Fy[0,...],-2))
                    plt.show(block=False)
                except:
                    pass
            Fy = None
            block_start_ts = trial_start_ts

        # compute the start/end of the segement to apply the model to
        if model_apply_type == 'trial':
            # apply the model to all available data from trial start
            block_start_ts = trial_start_ts
            if block_start_ts is not None and block_start_ts + block_step_ms + overlap_ms < valid_end_ts:
                block_end_ts = valid_end_ts
            else:
                block_end_ts = None
            # limit the trial size
            if block_start_ts is not None:
                block_start_ts = max( block_start_ts, valid_end_ts - maxDecisLen_ms)
        else:
            # check if enough data to apply the model
            if block_start_ts is not None and block_start_ts + block_step_ms + overlap_ms < valid_end_ts:
                # got enough data to process this block
                block_end_ts = valid_end_ts
            else:
                # not enough yet -> clear the end to indicate dont apply the model
                block_end_ts = None

        # if we have a valid block to apply the model do
        if block_start_ts is not None and block_end_ts is not None:

            # extract and apply to this block
            print("Extract block: {}->{} = {}ms".format(block_start_ts, block_end_ts, block_end_ts-block_start_ts))
            data = ui.extract_data_segment(block_start_ts, block_end_ts)
            stimulus = ui.extract_stimulus_segment(block_start_ts, block_end_ts)
            # skip if no data/stimulus to process
            if data.size == 0 or stimulus.size == 0:
                continue

            print('got: data {}->{} ({})  stimulus {}->{} ({}>0)'.format(data[0, -1], data[-1, -1], data.shape[0],
                                                              stimulus[0, -1], stimulus[-1, -1], np.sum(stimulus[:,0])))
            if model_apply_type == 'block':
                # update the start point for the next block
                # start next block at overlap before the end of this blocks data
                # so have sample accurate predictions, with no missing data, and no overlaps
                block_start_ts = data[-overlap_samp+1, -1]  # ~= block_end_ts - overlap_ms +1-sample
                bend = block_start_ts + block_step_ms + overlap_ms
                print("next block {}->{}: in {}ms".format(block_start_ts, bend, bend - ui.data_timestamp))

            # get predictions for this data block
            block_Fy = doPrediction(clsfr, data, stimulus)
            # strip predictions from the overlap period
            block_Fy = block_Fy[..., :-overlap_samp, :]

            # if got valid predictions...
            if block_Fy is not None:
                # accumulate or store the predictions
                if model_apply_type == 'trial':
                    Fy = block_Fy
                elif model_apply_type == 'block':  # accumulate blocks in the trial
                    if Fy is None:  # restart accumulation
                        Fy = block_Fy
                    else:
                        Fy = np.append(Fy, block_Fy, -2)
                # limit the trial length
                if maxDecisLen_ms > 0 and Fy.shape[-2] > maxDecisLen_samp:
                    print("limit trial length {} -> {}".format(Fy.shape[-2], maxDecisLen_samp))
                    Fy = Fy[..., -maxDecisLen_samp:, :]

                # send prediction event
                # only process the used-subset
                used_idx = np.any(Fy.reshape((-1, Fy.shape[-1])), 0)
                used_idx[0] = True # force include 0
                # map to probabilities, including the prior over sigma!
                Ptgt = clsfr.decode_proba(Fy[...,used_idx])
                #  _, _, Ptgt, _, _ = decodingSupervised(Fy[...,used_idx])
                # BODGE: only  use the last (most data?) prediction...
                Ptgt = Ptgt[-1, -1, :]
                # send prediction with last recieved stimulus_event timestamp
                print("Fy={} Yest={} Perr={}".format(Fy.shape, np.argmax(Ptgt), 1-np.max(Ptgt)))

                send_prediction(ui, Ptgt, used_idx=used_idx, timestamp=ui.stimulus_timestamp)
            
        # check for end-prediction messages
        for i,m in enumerate(newmsgs):
            if m.msgID == ModeChange.msgID:
                isPredicting = False
                # return unprocessed messages to stack. Q: why i+1?
                ui.push_back_newmsgs(newmsgs[i:])

def run(ui: UtopiaDataInterface=None, clsfr: BaseSequence2Sequence=None, msg_timeout_ms: float=100, 
        host:str=None, datafile:str=None,
        tau_ms:float=450, offset_ms:float=0, out_fs:float=100, evtlabs=None, 
        stopband=((45,65),(5.5,25,'bandpass')), ftype='butter', order:int=6, cv:int=5, 
        calplots:bool=False, predplots:bool=False, label:str=None, **kwargs):
    """ run the main decoder processing loop

    Args:
        ui (UtopiaDataInterface, optional): The utopia data interface class. Defaults to None.
        clsfr (BaseSequence2Sequence, optional): the classifer to use when model fitting. Defaults to None.
        msg_timeout_ms (float, optional): timeout for getting new messages from the data-interface. Defaults to 100.
        host (str, optional): hostname for the utopia hub. Defaults to None.
        tau_ms (float, optional): length of the stimulus response. Defaults to 400.
        offset_ms (float, optiona): offset in ms to shift the analysis window. Use to compensate for response lag.  Defaults to 0.
        stopband (tuple, optional): temporal filter specification for `UtopiaDataInterface.butterfilt_and_downsample`. Defaults to ((45,65),(5.5,25,'bandpass'))
        ftype (str, optional): type of temporal filter to use.  Defaults to 'butter'.
        order (int, optional): order of temporal filter to use.  Defaults to 6.
        out_fs (float, optional): sample rate after the pre-processor. Defaults to 100.
        evtlabs (tuple, optional): the brain event coding to use.  Defaults to None.
        calplots (bool, optional): flag if we make plots after calibration. Defaults to False.
        predplots (bool, optional): flag if we make plots after each prediction trial. Defaults to False.
    """
    global CALIBRATIONPLOTS, PREDICTIONPLOTS
    CALIBRATIONPLOTS = calplots
    PREDICTIONPLOTS = predplots
    try :
        import matplotlib.pyplot as plt
        guiplots=True
    except:
        guiplots=False

    # setup the saving label
    global uname
    from datetime import datetime 
    uname = datetime.now().strftime("%y%m%d_%H%M")
    if label is not None: # include label as prefix
        uname = "{}_{}".format(label,uname)

    # create data interface with bandpass and downsampling pre-processor, running about 10hz updates
    if ui is None:
        try:
            from  scipy.signal import butter
            ppfn = butterfilt_and_downsample(order=order, stopband=stopband, fs_out=out_fs, ftype=ftype)
        except: # load filter from file
            print("Warning: stopband specification *ignored*, using sos_filter_coeff.pk file...")
            ppfn = butterfilt_and_downsample(stopband='sos_filter_coeff.pk', fs_out=out_fs)
        #ppfn = None
        ui = UtopiaDataInterface(data_preprocessor=ppfn,
                                 stimulus_preprocessor=None,
                                 timeout_ms=100, mintime_ms=55, clientid='decoder') # 20hz updates
    ui.connect(host=host, queryifhostnotfound=False)
    # use a multi-cca for the model-fitting
    if clsfr is None:
        if isinstance(evtlabs,str): # decode string coded spec
            evtlabs = evtlabs.split(',')
        clsfr = MultiCCA(tau=int(out_fs*tau_ms/1000), evtlabs=evtlabs, offset=int(out_fs*offset_ms/1000))
        print('clsfr={}'.format(clsfr))
    
    calibration_dataset = None
    current_mode = "idle"
    # clean shutdown when told shutdown
    while current_mode.lower != "shutdown".lower():

        if  current_mode.lower() in ("calibration.supervised","calibrate.supervised"):
            calibration_dataset, _, _ = doCalibrationSupervised(ui, clsfr, cv=cv, previous_dataset=calibration_dataset)
                
        elif current_mode.lower() in ("prediction.static","predict.static"):
            doPredictionStatic(ui, clsfr)

        elif current_mode.lower() in ("reset"):
            calibration_dataset = None
            clsfr.clear()

        # check for new mode-messages
        newmsgs, nsamp, nstim = ui.update()

        # update the system mode
        current_mode = "idle"
        for i, m in enumerate(newmsgs):
            if m.msgID == ModeChange.msgID:
                current_mode = m.newmode
                print("\nNew Mode: {}".format(current_mode))
                ui.push_back_newmsgs(newmsgs[i+1:])
                # stop processing messages
                break
        
        # BODGE: re-draw plots so they are interactive.
        if guiplots:
            for i in plt.get_fignums():
                plt.figure(i).canvas.flush_events()

def mainloop(*args,**kwargs):
    run(*args,**kwargs)

if  __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--out_fs',type=int, help='output sample rate', default=100)
    parser.add_argument('--tau_ms',type=float, help='output sample rate', default=450)
    parser.add_argument('--evtlabs', type=str, help='comma separated list of stimulus even types to use', default='re,fe')
    parser.add_argument('--stopband',type=json.loads, help='set of notch filters to apply to the data before analysis', default=((45,65),(5.5,25,'bandpass')))
    parser.add_argument('--cv',type=int, help='number cross validation folds', default=5)
    parser.add_argument('--predplots', action='store_true', help='flag make decoding plots are prediction time')
    parser.add_argument('--calplots', action='store_false', help='turn OFF model and decoding plots after calibration')
    parser.add_argument('--savefile', type=str, help='run decoder using this file as the proxy data source', default=None)
    parser.add_argument('--savefile_fs', type=float, help='effective sample rate for the save file', default=None)

    args = parser.parse_args()

    if args.savefile is not None or True:#
        #setattr(args,'savefile',"~/utopia/java/messagelib/UtopiaMessages_.log")
        #setattr(args,'savefile',"~/utopia/java/utopia2ft/UtopiaMessages_*1700.log")
        #setattr(args,'savefile',"~/Downloads/jason/UtopiaMessages_200923_1749_*.log")
        setattr(args,'savefile','~/Desktop/mark/mindaffectBCI*.txt')
        #setattr(args,'out_fs',100)
        #setattr(args,'savefile_fs',200)
        #setattr(args,'cv',5)
        setattr(args,'predplots',True) # prediction plots -- useful for prediction perf debugging
        from mindaffectBCI.decoder.FileProxyHub import FileProxyHub
        U = FileProxyHub(args.savefile,use_server_ts=True)
        ppfn = butterfilt_and_downsample(order=6, stopband=args.stopband, fs_out=args.out_fs, ftype='butter')
        ui = UtopiaDataInterface(data_preprocessor=ppfn,
                                 stimulus_preprocessor=None,
                                 timeout_ms=100, mintime_ms=0, U=U, fs=args.savefile_fs, clientid='decoder') # 20hz updates
        # add the file-proxy ui as input argument
        setattr(args,'ui',ui)
    
    running=True
    nCrash = 0
    run(**vars(args))
    while running and nCrash < 10:
        try:
            run(**vars(args))
            # stop restarting if normal terminate
            running=False
        except KeyboardInterrupt:
            # stop running if keyboard interrrupt
            running=False
        except Exception as ex:
            print("Error running mainloop"+ str(ex))
            nCrash = nCrash + 1
            pass
