#!/usr/bin/env python3
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
from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, stim2eventfilt, butterfilt_and_downsample


def run(ui: UtopiaDataInterface, maxruntime_ms: float=np.inf, timeout_ms: float=np.inf, tau_ms: float=500,
              offset_ms=(-15, 0), evtlabs=None, ch_names=None, nstimulus_events: int=600, rank=3, center=True):
    ''' simple sig-viewer using the ring-buffer for testing '''
    import matplotlib.pyplot as plt

    if evtlabs is None:
        evtlabs = ('re', 'fe')

    # initialize the filter from raw stimulus to brain-trigger events
    stimfilt = stim2eventfilt(evtlabs)

    ui.update()

    # compute the size of the erp slice
    irf_range_ms = (offset_ms[0], tau_ms+offset_ms[1])
    irflen_ms = irf_range_ms[1]-irf_range_ms[0]
    irflen_samp = int(np.ceil(irflen_ms * ui.fs / 1000))
    irf_times = np.linspace(irf_range_ms[0], irf_range_ms[1], irflen_samp)

    # initialize the arrays to hold the stimulus responses, and their label info, and a current cursor
    irf = np.zeros((nstimulus_events, irflen_samp, ui.data_ringbuffer.shape[-1] - 1)) # (nERP,irflen,d)
    irf_lab = np.zeros((nstimulus_events, len(evtlabs)), dtype=int)  # (nErp,nstim)
    cursor = 0

    # initialize the plot window
    fig = plt.figure(1)
    fig.clear()

    # main spec, inc titles
    outer_grid = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1,8])

    # right-sub-spec for the ERPs
    plt.figtext(.25,.9,'ERPs',ha='center')
    # get grid-spec to layout the plots, 1 for spatial, 1 for temporal
    gs = outer_grid[-1,0].subgridspec(nrows=len(evtlabs), ncols=1)
    erp_ax = [None for j in range(len(evtlabs))]
    erp_lines = [None for j in range(len(evtlabs))]
    erp_ax[-1] = fig.add_subplot(gs[-1,0])
    for ei,lab in enumerate(evtlabs):
        if ei==len(evtlabs)-1:
            ax = erp_ax[ei]
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Space")
        else:
            ax = fig.add_subplot(gs[ei,0], sharex=erp_ax[-1], sharey=erp_ax[-1])
            ax.tick_params(labelbottom=False)
        erp_lines[ei] = ax.imshow(irf[ei,:,:].T,origin='lower',aspect='auto',extent=(irf_times[0],irf_times[-1],0,irf.shape[-1]))
        ax.set_title(lab)
        erp_ax[ei] = ax

    # left-sub-spec for the decomposition
    plt.figtext(.75,.9,'SVD Decomposition',ha='center')

    # get grid-spec to layout the plots, 1 for temporal, rank for spatial
    gs =  outer_grid[-1,1].subgridspec(nrows=len(evtlabs), ncols=rank+1, width_ratios=(4,)+(1,)*rank)
    spatial_ax = [[ None for i in range(rank)] for j in range(len(evtlabs))]
    spatial_lines = [[ None for i in range(rank)] for j in range(len(evtlabs))]
    temporal_ax = [None for j in range(len(evtlabs))]
    temporal_lines = [None for j in range(len(evtlabs))]
    # make the bottom 2 axs already so can share their limits.
    temporal_ax[-1] = fig.add_subplot(gs[-1, 0])
    spatial_ax[-1][0] = fig.add_subplot(gs[-1, 1])
    for ei, lab in enumerate(evtlabs):
        # single temporal plot for all ranks
        if ei == len(evtlabs)-1 and ri==0: # tick-plot
            ax = temporal_ax[ei]
            ax.set_xlabel("Time (ms)")
        else:
            ax = fig.add_subplot(gs[ei, 0], sharex=temporal_ax[-1], sharey=temporal_ax[-1])
            ax.tick_params(labelbottom=False)
        ax.set_title("{}".format(lab))
        ax.grid(True)
        ax.autoscale(axis='y')
        temporal_ax[ei] = ax
        # component range
        temporal_lines[ei] = ax.plot(irf_times, np.ones((irf.shape[-2], rank)))
        # TODO: add errorbar visualization..
        # component range
        #temporal_lines[ei].extend(ax.plot(irf_times, np.ones((irf.shape[-2], rank)), color=(0, 0, 0)))

        for ri in range(rank):
            # spatial-plot
            if ei == len(evtlabs)-1 and ri==0: # tick-plot
                ax = spatial_ax[ei][ri]
                ax.set_xlabel("Space")
            else:
                ax = fig.add_subplot(gs[ei, 1+ri], sharex=spatial_ax[-1][0], sharey=spatial_ax[-1][0])
                ax.tick_params(labelbottom=False, labelleft=False)

            ax.set_title("{}".format(lab))
            ax.grid(True)
            spatial_ax[ei][ri] = ax
            spatial_lines[ei][ri] = ax.plot(np.zeros((irf.shape[-1])), color=(0, 0, 0))

    # add a reset ERP button
    def reset(event):
        print('reset called')
        irf_lab.fill(0)
        cursor=0 
    from matplotlib.widgets import Button
    butax = fig.add_axes([.05,.85,.1,.1])
    breset = Button(butax,'Reset')
    breset.on_clicked(reset)


    fig.show()
    
    # start the render loop
    t0 = ui.getTimeStamp()
    pending = []  # list of stimulus events for which still waiting for data to complete
    dirty = False # flag if do full canvas redraw
    while ui.getTimeStamp() < t0+maxruntime_ms:

        # exit if figure is closed..
        if not plt.fignum_exists(1):
            quit()

        # re-draw the display
        fig.canvas.flush_events()
        if dirty : 
            fig.canvas.draw()
            dirty=False

        #plt.pause(.001)

        # Update the records
        nmsgs, ndata, nstim = ui.update(timeout_ms=timeout_ms)

        # skip if no stimulus events to process
        if nstim == 0:
            continue

        # get the new raw stimulus
        rawstimulus = ui.stimulus_ringbuffer[-nstim:, :]

        # split the time-stamp and the *true-target* stimulus info
        stimulus_ts = rawstimulus[:, -1]
        # and transform to irf trigger events
        stimulus = stimfilt.transform(rawstimulus)  # (nsamp,nstim)
        # strip to only the objID==0 output
        stimulus = stimulus[:, 0, :]  # (nsamp,nE)

        # identify any new stimulus events and add to the IRF stack
        for ti in np.flatnonzero(np.any(stimulus, -1)):
            # get the trigger time-stamp
            ts = stimulus_ts[ti]
            # get the trigger data range
            data_segment = (ts+irf_range_ms[0], ts+irf_range_ms[1])
            # store in the pending list
            pending.append((stimulus[ti, :], data_segment))

        # process any completed pending stimulus segments
        new_pending = []
        for stimulus_segment in pending:
            stimulus, data_segment = stimulus_segment
            if data_segment[1] < ui.data_timestamp:
                # get the irf window
                data = ui.extract_data_segment(*data_segment)
                if data.size == 0:
                    print("didn't get any valid data!!!: data_seg={}".format(*data_segment))
                    continue
                # insert into the running buffer, w/0 the time-stamp channel
                if data.shape[0] < irf.shape[1]:
                    irf[cursor, :data.shape[0], :] = data[:, :-1]
                    irf[cursor, data.shape[0]:, :] = data[-1, :-1]  # held with last value
                else:
                    irf[cursor, :, :] = data[:irf.shape[1], :-1]
                #print("seglen={}".format(data.shape[0]))
                irf_lab[cursor, ...] = stimulus
                # move the cursor
                cursor = (cursor + 1) % irf.shape[0]
            else:
                new_pending.append(stimulus_segment)
        # update the pending segments list
        pending = new_pending

        # Update the plots for each event type
        dirty=True
        for ei, lab in enumerate(evtlabs):

            # extract this events irfs
            idx = np.flatnonzero(irf_lab[:, ei])
            # skip if no events to show
            if idx.size == 0:
                continue
            data = irf[idx, ...].copy()  #(nevt,nsamp,d)
            
            if center:
                data = data - data[:, 0:1, :]  # np.mean(data,axis=-2,keepdims=True)

            # compute ERP            
            erp = np.mean(data, 0)  # (nsamp,d)
            # plot ERP
            erp_lim = (np.min(erp.ravel()),np.max(erp.ravel()))
            erp_lines[ei].set_data(erp.T)
            erp_lines[ei].set_clim(vmin=erp_lim[0],vmax=erp_lim[1])
            erp_ax[ei].set_title("{} (n={})".format(lab,data.shape[0]))

            #print("erp={}".format(erp))
            # decompose into spatial and temporal components
            if True:
                # TODO: use multiCCA to get the decomp?
                R, s, A = np.linalg.svd(erp, full_matrices=False)
                A = A.T # (d,rank)
                # get the biggest eigenvalue to display
                slmidx = np.argsort(s) # N.B. ascending order
                slmidx = slmidx[::-1]  # N.B. DESCENDING order
                R = R[:, slmidx[:rank]]  #(nsamp,rank)
                s = s[slmidx[:rank]]
                A = A[:,slmidx[:rank]] * s[np.newaxis,:]  #(d,rank)
            else:
                A = np.zeros((erp.shape[-1])) 
                A[-1] = 1  #(1,d)
                R = erp[:, -1:]  # @ A
                
            # apply the spatial filter to the raw data to get pure time-course
            for ri in range(A.shape[-1]):
                # plot the spatial pattern
                spatial_lines[ei][ri][-1].set_ydata(A[:,ri])
                spatial_ax[ei][ri].set_ylim((np.min(A), np.max(A)))
                #temporal_ax[ei].plot(irf_times, R, color=(0,0,0), label='erp ({})'.format(data.shape[0]))

                # plot single-trial data
                #datari = data @ A[:,ri] # (d,1)?
                #for di in range(min(len(temporal_lines[ei][ri])-1, data.shape[0])):
                #    temporal_lines[ei][ri][-1-di-1].set_ydata(datari[-1-di, :])
                    
                # plot the average response, as thick black                
                temporal_lines[ei][ri].set_ydata(R[:,ri])
                if ri==0:
                    temporal_ax[ei].set_title("{} ({})".format(lab, data.shape[0]))
                    temporal_ax[ei].set_ylim( (np.min(R.ravel())*2, np.max(R.ravel()) * 2) )

def parse_args():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--evtlabs', type=str, help='comma separated list of stimulus even types to use', default='re,fe')
    parser.add_argument('--out_fs',type=int, help='output sample rate', default=100)
    parser.add_argument('--stopband',type=json.loads, help='set of notch filters to apply to the data before analysis', default=((45,65),(5.5,25,'bandpass')))
    parser.add_argument('--tau_ms',type=float, help='output sample rate', default=450)
    parser.add_argument('--offset_ms',type=float, help='offset from time 0 for analysis', default=0)
    parser.add_argument('--rank', type=str, help='rank of decomposition to use', default=3)
    parser.add_argument('--ch_names', type=str, help='list of channel names, or capfile', default=None)
    parser.add_argument('--savefile', type=str, help='run decoder using this file as the proxy data source', default=None)
    parser.add_argument('--savefile_fs', type=float, help='effective sample rate for the save file', default=None)
    parser.add_argument('--savefile_speedup', type=float, help='play back the save file with this speedup factor', default=None)
    parser.add_argument('--timeout_ms', type=float, help="timeout for wating for new data from hub, equals min-redraw time.",default=500)
   
    args = parser.parse_args()
    if args.evtlabs: 
        args.evtlabs = [ e.strip() for e in args.evtypes.split(',') ]
    if args.ch_names:
        args.ch_names = [ c.strip() for c in args.ch_names.split(',') ]
        
    return args

                    
if __name__=='__main__':
    args = parse_args()

    if args.savefile is not None:
        from mindaffectBCI.decoder.FileProxyHub import FileProxyHub
        U = FileProxyHub(args.savefile,use_server_ts=True,speedup=args.savefile_speedup)
        ppfn = butterfilt_and_downsample(order=6, stopband=args.stopband, fs_out=args.out_fs, ftype='butter')
        ui = UtopiaDataInterface(data_preprocessor=ppfn,
                                 send_signalquality=False,
                                 timeout_ms=100, mintime_ms=0, U=U, fs=args.savefile_fs, clientid='viewer')
    else:
        data_preprocessor = butterfilt_and_downsample(order=6, stopband=args.stopband, fs_out=args.out_fs)
        ui=UtopiaDataInterface(data_preprocessor=data_preprocessor, send_signalquality=False, clientid='viewer')
        ui.connect(args.host)

    try:
        matplotlib.use('Qt4Cairo')
    except:
        pass

    run(ui, **vars(args))
    