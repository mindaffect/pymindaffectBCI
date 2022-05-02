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
from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
from mindaffectBCI.decoder.model_fitting import MultiCCA
from mindaffectBCI.decoder.utils import window_axis
from mindaffectBCI.decoder.updateSummaryStatistics import plot_trial
from mindaffectBCI.decoder.analyse_datasets import plot_stimseq
import matplotlib.pyplot as plt
import glob


def load_and_trigger_check(filename=None, filterband=(.1, 45, 'bandpass'), fs_out=250, **kwargs):
    """make a set of visualizations of the stimulus->measurement time-lock

    Args:
        filename ([type], optional): The name of the file to load. Defaults to None.
        evtlabs (tuple, optional): The event coding to use for how stimulus changes map into trigger inputs. Defaults to ('0','1').
        tau_ms (int, optional): The duration of the stimulus response. Defaults to 400.
        offset_ms (int, optional): Offset of the start of the stimulus response w.r.t. the trigger time. Defaults to -50.
        max_samp (int, optional): Limit in the number of samples for each trial. Defaults to 6000.
        filterband (tuple, optional): Temporal filter used to pre-process the EEG. Defaults to (.1,45,'bandpass').
        fs_out (int, optional): Sample rate of the EEG used for analysis. Defaults to 250.
    """
    import glob
    import os
    if filename is None or filename == '-':
        # default to last log file if not given
        # * means all if need specific format then *.csv
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/mindaffectBCI*.txt'))
        filename = max(files, key=os.path.getctime)

    X, Y, coords = load_mindaffectBCI(filename, filterband=filterband, fs_out=fs_out)

    return trigger_check(X, Y, coords, **kwargs)


def trigger_check(X, Y, coords, evtlabs=('re', 'fe'),
                  tau_ms=125, offset_ms=-25, trntrl=slice(10),
                  trnsamp=6000, max_samp=None, 
                  plot_model=True, plot_trials=True, plot_epoch_lines=False, 
                  plot_events:bool=True, plot_erp: bool = True, 
                  **kwargs):
    """make a set of visualizations of the stimulus->measurement time-lock

    Args:
        filename ([type], optional): The name of the file to load. Defaults to None.
        evtlabs (tuple, optional): The event coding to use for how stimulus changes map into trigger inputs. Defaults to ('0','1').
        tau_ms (int, optional): The duration of the stimulus response. Defaults to 400.
        offset_ms (int, optional): Offset of the start of the stimulus response w.r.t. the trigger time. Defaults to -50.
        max_samp (int, optional): Limit in the number of samples for each trial. Defaults to 6000.
        filterband (tuple, optional): Temporal filter used to pre-process the EEG. Defaults to (.1,45,'bandpass').
        fs_out (int, optional): Sample rate of the EEG used for analysis. Defaults to 250.
    """
    # size limit...
    if max_samp is not None and X.shape[1] > max_samp:
        X = X[:, :max_samp, ...]
        Y = Y[:, :max_samp, ...]
    X[..., :-1] = X[..., :-1] - np.mean(X[..., :-1], axis=-2, keepdims=True)  # offset remove
    fs = coords[-2]['fs']
    print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords], X.shape, coords[1]['fs']))
    print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:-1]]+['output'], Y.shape))

    ax, clsfr, wX, wXeY, Y_true = triggerPlot(
        X, Y, fs, evtlabs=evtlabs, tau_ms=tau_ms, offset_ms=offset_ms, max_samp=max_samp, trntrl=trntrl, **kwargs)
    times = (np.arange(wXeY.shape[-1])+offset_ms*fs/1000)*1000/fs

    # over-plot the information on the timestamp errors
    # get a samp#, timestamp dataset
    trl_ts = coords[0]['trl_ts'][:, :Y_true.shape[1]]   # (nTrl, nSamp-tau)
    trl_idx = coords[0]['trl_idx'][:, :Y_true.shape[1]]  # (nTrl, nSamp-tau)
    samp2ms = np.median((np.diff(trl_ts, axis=-1)/np.maximum(1, np.diff(trl_idx, axis=-1))).ravel())
    samp2ms = 1000/fs  # samp2ms*.99
    print("samp2ms={}".format(samp2ms))
    lin_trl_ts = trl_idx*samp2ms  # linear time-stamp assuming constant sample rate
    ts_err = trl_ts - lin_trl_ts  # (tr, nSamp-tau) error btw. linear time-stamp and observed
    ts_err = ts_err - ts_err[0, 0]
    ts_errY = ts_err[Y_true > 0]  # (ep,) slice out the high-stimulus info (in seconds)

    ax2 = ax.twinx()
    plt.plot(ts_errY, 'k-')
    mu2 = np.median(ts_errY.ravel())
    scale2 = np.median(np.abs(ts_errY.ravel()-mu2))
    if scale2*2 < tau_ms*8:  # make line match
        scale2 = tau_ms/2
    elif scale2*2 < tau_ms*.2:  # make image smaller
        ax.set_ylim(min(times[0], mu2-2*scale2), max(times[-1], mu2+2*scale2))
    plt.ylim(mu2-1*scale2, mu2+1*scale2)
    plt.ylabel("Recieved time-stamp error vs. constant rate (ms)")
    plt.show(block=False)

    # allow full re-draw
    plt.pause(.5)

    if plot_events:
        plt.figure(6)
        plt.clf()
        plot_stimseq(Y[:2,...], fs=fs, show=False)
        plt.pause(.1)  # allow full redraw

    if plot_trials:
        plt.figure(2)
        plt.clf()
        plot_trial(X[:2, ...], Y[:2, ...], fs)
        plt.pause(.1)  # allow full redraw

    if plot_model:
        # model in a new figure
        plt.figure(3)
        clsfr.plot_model(fs=fs)
        plt.show(block=False)
        # allow full re-draw
        plt.pause(.5)

    # now plot as lines
    if plot_epoch_lines:
        plt.figure(4)
        tmp = wXeY.reshape((-1, wXeY.shape[-1]))[:100, :]
        tmp = tmp - np.mean(tmp, -1, keepdims=True)
        tmp = tmp / np.std(tmp.ravel())
        plt.plot(times, tmp.T + np.arange(tmp.shape[0])[np.newaxis, :])
        plt.title('1st {} {} evt trigger epochs time-series'.format(tmp.shape[0], evtlabs[0]))


def triggerPlot(
        X, Y, fs, clsfr=None, fig=None, evtlabs=('re', 'fe'),
        tau_ms=125, offset_ms=-25, max_samp=6000, trntrl=None, plot_model=True, plot_trial=True, ax=None, **kwargs):
    if X.ndim < 3:
        X = X[np.newaxis, ...]
    if Y.ndim < 3:
        Y = Y[np.newaxis, ...]
    if isinstance(evtlabs, str):
        evtlabs = [evtlabs]

    # BODGE: for cases where we don't have a 'cued-target' set, just use average over all
    # other targets
    if np.all(Y[..., 0] == 0):
        Y = np.sum(Y, axis=-1, keepdims=True)

    # BODGE: for speed only use first 5 trials!
    # BODGE: reg with 1e-5 so only the strong channels are used..
    if clsfr is None:
        if trntrl is None:
            trntrl = slice(0, 10)
        if isinstance(trntrl, int):
            trntrl = slice(trntrl, None)

        print("training model on {}".format(trntrl))
        clsfr = MultiCCA(evtlabs=evtlabs, tau_ms=tau_ms, rank=1, fs=fs)
        Xtrn, Ytrn = (X[trntrl, -max_samp:, ...], Y[trntrl, -max_samp:, ...]) if max_samp else (X[trntrl],Y[trntrl])
        clsfr.fit(Xtrn,Ytrn)
        print('clsfr={}'.format(clsfr))

    # get the classifier predictions
    Ptgt = clsfr.predict_proba(X, Y)  # (nM,tr,decis,ny) get predicted target probability for each trial
    # Ptgt[Ptgt==0]=1
    Yerr = np.any(Ptgt[..., -1, :1] < Ptgt[..., -1, 1:], axis=-1)  # use last prediction to decide if correct

    # get the event-coded version of Y
    Ye = clsfr.stim2event(Y)

    print("applying spatial filter")
    W = clsfr.W_[..., 0, :].ravel()  # (d,)
    print("W={}".format(W))

    # slice the data w.r.t. the stimulus triggers to generate the visualization
    if offset_ms > 0:
        print('Warning positive offsets not supported yet... ignored!')
        offset_ms = 0
    tau = int(fs*tau_ms/1000.0)
    offset = int(fs*offset_ms/1000.0)
    times = (np.arange(tau)+offset)*1000/fs
    wX = np.einsum("d,Ttd->Tt", W, X)  # (nTrl, nSamp) apply the spatial filter
    # slice out the responses for the trigger stimulus
    print('slicing data')
    wXe = window_axis(wX, winsz=tau, axis=-1)  # (nTrl, nSamp-tau, tau)
    # N.B. negative offset here as negative shift of Y is same as positive shift of X
    Y_true = Ye[..., 0, 0]  # (nTrl, nSamp-tau)
    if offset <= 0:
        Y_true = Ye[..., -offset:wXe.shape[1]-offset, 0, 0]  # (nTrl, nSamp-tau)
    elif offset > 0:
        raise ValueError("Positive offsets not supported yet!")
    wXeY = wXe[Y_true > 0, :]  # [ep,tau]

    print('generating plot')
    mu = np.median(wXeY.ravel())
    scale = np.median(np.abs(wXeY.ravel()-mu))
    if ax is None:
        plt.figure()
        ax = plt.axes()
        newax = True
    else:
        newax = False
    plt.sca(ax)
    plt.imshow(wXeY.T, origin='lower', aspect='auto', extent=[0, wXeY.shape[0], times[0], times[-1]])
    plt.title('{} locked {} trials {:4.1f}min data'.format(evtlabs[0], X.shape[0], (X.shape[0]*X.shape[1])/fs/60))
    plt.clim(mu-scale, mu+scale)
    if newax:
        plt.colorbar()
        plt.set_cmap('jet')  # 'gray')#'nipy_spectral')
        plt.ylabel('time (ms)')
        plt.xlabel('Epoch')
        plt.grid()

    # draw a line at the trial boundary + text?
    if X.ndim > 2 and X.shape[0] > 1:
        # make a trial indicator
        trlIdx = np.tile(np.arange(X.shape[0])[:, np.newaxis], (1, wXe.shape[1]))
        trlIdx = trlIdx[Y_true > 0]
        #plt.plot(trlIdx,'w-',label='trial number')
        trlEndIdx = np.append(0, np.flatnonzero(np.diff(trlIdx) > 0))
        # for i,idx in enumerate(trlEndIdx):
        #    plt.text(idx,0,"{:3d}".format(i+1),ha='center',va='center')
        # N.B. +2 as end-of-trial line and count from 0
        ticklabs = ["{}{}".format(i+1, "" if p else "*") for (i, p) in zip(range(len(trlEndIdx)), Yerr)]
        plt.xticks(trlEndIdx, ticklabs, rotation=-90, size='x-small')
        # ax.set_xticks(trlEndIdx,major=True)
        # ax.set_ticklabels(trlEndIdx)
        plt.grid(True, which='major')
        plt.xlabel('Trial')

    # if X.shape[-1]>8 :
    #     samp_cnt = X[...,-1] # sample count (%255)
    #     samp_miss = samp_cnt[...,1:] - ((samp_cnt[...,:-1]+1) % 256)
    #     samp_miss = samp_miss[...,:X.shape[1]]
    #     samp_miss = samp_miss[Y_true>0]
    #     plt.plot(samp_miss*samp2ms + 100,'k-')
    return ax, clsfr, wX, wXeY, Y_true


def run(hostname='-', filterband=(.1, 45, 'bandpass'), fs_out=250, **kwargs):
    """online continuously updating trigger check plot
    """
    from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, butterfilt_and_downsample
    from mindaffectBCI.decoder.devent2stimsequence import upsample_stimseq
    data_preprocessor = butterfilt_and_downsample(order=6, filterband=filterband, fs_out=fs_out)
    ui = UtopiaDataInterface(data_preprocessor=data_preprocessor, send_signalquality=False)
    ui.connect(hostname)
    try:
        import matplotlib
        matplotlib.use('Qt4Cairo')
    except:
        pass

    fig = plt.gcf()
    ax = None
    data = None
    stimulus = None
    while True:
        # exit if figure is closed..
        if not plt.fignum_exists(1):
            quit()

        # re-draw the display
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(.5)

        # Update the records
        nmsgs, ndata, nstim = ui.update(timeout_ms=500)

        # get the new raw data and append
        rawdata = ui.data_ringbuffer[-ndata:, :]
        if data is None:
            data = rawdata
        else:
            data = np.append(data, rawdata, axis=-2)

        # skip if no stimulus events to process
        if nstim == 0:
            continue

        # get the new raw stimulus
        rawstimulus = ui.stimulus_ringbuffer[-nstim:, :]
        if rawstimulus.size > 0:
            # append to the stimulus info
            if stimulus is None:
                stimulus = rawstimulus
            else:
                if np.any(rawstimulus[0, -1] == stimulus[:, -1]):
                    print('loop')
                stimulus = np.append(stimulus, rawstimulus, axis=-2)

        # update the trigger check plot
        # TODO []: make it nicer / more efficient
        data_ts = data[:, -1]
        stim_ts = stimulus[:, -1]
        # up-sample to the data rate by matching time-stamps
        usstimulus, _ = upsample_stimseq(data_ts, stimulus, stim_ts)
        ax, _, _, _, _ = triggerPlot(data[..., :-1], usstimulus[..., :-1], ui.fs,
                                     new_fig=False, plot_model=False, plot_trial=False, ax=ax, **kwargs)


if __name__ == "__main__":
    filename = None  # on-line trigger check
    # load the most recent matching file
    # filename='~/Desktop/pymindaffectBCI/logs/mindaffectBCI_*.txt'
    # filename='~/Downloads/mindaffectBCI*.txt'
    # filename='~/Desktop/mark/mindaffectBCI_*1239.txt'
    filename = 'askloadsavefile'

    # filename='-'  # take last file save in logs directory
    # filename='~/Desktop/pymindaffectBCI/logs/mindaffectBCI*.txt'

    if filename is None:

        run(evtlabs=('re', 'fe'), tau_ms=125, offset_ms=-25, filterband=(0, .5), fs_out=250)

    else:  # offline

        # trigger/opto-data
        load_and_trigger_check(filename, evtlabs=('re'), tau_ms=225, offset_ms=0, filterband=(0, 2))
        plt.show()

        # brain data, 10-cal trials
        #trigger_check(filename, evtlabs=('fe','re'), tau_ms=450, offset_ms=0, filterband=((45,65),(5.5,25,'bandpass')), fs_out=100, trntrl=slice(10))

        # # do a time-stamp check.
        # from mindaffectBCI.decoder.timestamp_check import timestampPlot
        # plt.figure(6)
        # timestampPlot(filename)
