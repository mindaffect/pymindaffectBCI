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
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics
from mindaffectBCI.decoder.multipleCCA import multipleCCA
from mindaffectBCI.decoder.stim2event import stim2event
from mindaffectBCI.decoder.devent2stimsequence import upsample_stimseq

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def ccaViewer(*args, **kwargs):
    run(*args, **kwargs)

def run(ui: UtopiaDataInterface, maxruntime_ms: float=np.inf, timeout_ms:float = 100, tau_ms: float=500,
              offset_ms=(-15, 0), evtlabs=None, ch_names=None, ch_pos=None, nstimulus_events: int=600, 
              rank:int=3, reg=.02, center:bool=True, host:str='-', stopband=None, out_fs=100, **kwargs):
    ''' view the live CCA decomposition.'''

    if ui is None:
        data_preprocessor = butterfilt_and_downsample(order=6, stopband=stopband, fs_out=out_fs)#999)
        #data_preprocessor = butterfilt_and_downsample(order=6, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=60)
        ui=UtopiaDataInterface(data_preprocessor=data_preprocessor, send_signalquality=False)#, sample2timestamp='none')
        ui.connect(host)
    ui.update()

    nCh = ui.data_ringbuffer.shape[-1] - 1

    if evtlabs is None:
        evtlabs = ('re', 'fe')

    if ch_names is None:
        ch_names = np.arange(nCh)

    # extract position info for topo-plots if possible
    if ch_pos is None and len(ch_names) > 0:
        # try to load position info from capfile
        try: 
            print("trying to get pos from cap file!")
            from mindaffectBCI.decoder.readCapInf import getPosInfo
            cnames, xy, xyz, iseeg =getPosInfo(ch_names)
            if all(iseeg):
                ch_pos = xy
        except:
            pass

    # compute the size of the erp slice
    irf_range_ms = (offset_ms[0], tau_ms+offset_ms[1])
    irflen_ms = irf_range_ms[1]-irf_range_ms[0]
    irflen_samp = int(np.ceil(irflen_ms * ui.fs / 1000))
    irf_times = np.linspace(irf_range_ms[0], irf_range_ms[1], irflen_samp)

    # initialize the arrays to hold the stimulus responses, and their label info, and a current cursor
    irf = np.zeros((nstimulus_events, irflen_samp, nCh)) # (nERP,irflen,d)
    irf_lab = np.zeros((nstimulus_events, len(evtlabs)), dtype=int)  # (nErp,nstim)
    nY = np.zeros(len(evtlabs),dtype=int)

    # store for the summary statistics
    Cxx = None
    Cxy = None
    Cyy = None
    
    # initialize the plot window
    fig = plt.figure(1)
    fig.clear()

    # main spec, inc titles; ERP | CCA
    outer_grid = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1,6])

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
        erp_lines[ei] = ax.imshow(irf[ei,:,:].T,aspect='auto',origin='lower',extent=(irf_times[0],irf_times[-1],0,irf.shape[-1]))
        ax.set_title(lab)
        erp_ax[ei] = ax

    # left-sub-spec for the decomposition
    plt.figtext(.75,.9,'CCA Decomposition',ha='center')

    # get grid-spec to layout the plots 1 row per rank, and 1 for spatial, 1 for temporal
    gs = outer_grid[-1,1].subgridspec(nrows=rank, ncols=2, width_ratios=[1,3])
    spatial_ax = [ None for i in range(rank)]
    spatial_lines = [ None for i in range(rank)] 
    temporal_ax = [None for j in range(rank)]
    temporal_lines = [[None for i in range(len(evtlabs))] for j in range(rank)]

    # make the bottom 2 axs already so can share their limits.
    spatial_ax[-1] = fig.add_subplot(gs[-1, 0])
    temporal_ax[-1] = fig.add_subplot(gs[-1, 1])
    for ri in range(rank):
        # spatial-plot
        if ri==rank-1: 
            ax = spatial_ax[ri]
            ax.set_xlabel("Space")
        else:
            ax = fig.add_subplot(gs[ri, 0], sharex=spatial_ax[-1], sharey=spatial_ax[-1])
            spatial_ax[ri] = ax
        ax.set_ylabel("Comp #{}".format(ri))    

        w = np.zeros((irf.shape[-1],))
        if not ch_pos is None: # make as topoplot
            #from scipy.interpolate import griddata
            #xs, ys = np.mgrid[np.min(ch_pos[:,0])-.1:np.min(ch_pos[:,0])-.1:30j,
            #                  np.min(ch_pos[:,1])-.1:np.max(ch_pos[:,1])+.1:30j]
            #img = griddata(ch_pos,w,(xs,ys)) # (30,30)
            #spatial_axis[ri]= ax.imshow(img,extent=(rng_x+rng_y),aspect='auto')

            interp_pos = np.concatenate((ch_pos+ np.array((0,.1)), ch_pos + np.array((-.1,-.05)), ch_pos + np.array((+.1,-.05))),0)
            spatial_lines[ri]=ax.tricontourf(interp_pos[:,0],interp_pos[:,1],np.tile(w,3),cmap='Spectral')

            for i,(x,y) in enumerate(ch_pos):
                ax.text(x,y,ch_names[i],ha='center',va='center') # label
            ax.set_aspect(aspect='equal')
            ax.set_frame_on(False) # no frame
            ax.tick_params(labelbottom=False,labelleft=False,which='both',bottom=False,left=False) # no labels, ticks
            plt.colorbar(spatial_lines[ri],ax=ax)

        else: # line-plot
            ax.grid(True)
            ax.tick_params(labelbottom=False)
            spatial_lines[ri] = ax.plot(ch_names[:irf.shape[-1]], np.zeros((irf.shape[-1])), color=(0, 0, 0))

        # single temporal plot for all events
        if ri==rank-1: # tick-plot
            ax = temporal_ax[-1]
        else:
            ax = fig.add_subplot(gs[ri, 1], sharex=temporal_ax[-1], sharey=temporal_ax[-1])
            ax.tick_params(labelbottom=False)
        # setup the lines
        for ei, lab in enumerate(evtlabs):
            # component range
            temporal_lines[ri][ei] = ax.plot(irf_times, np.zeros(irf_times.shape), label=lab)[0]
            # len(evtlabs))))
            #temporal_lines[ri][ei].set_label(evtlabs[ei])
        ax.grid(True)
        #ax.autoscale(axis='y')
        if ri==rank-1:
            ax.set_xlabel("Time (ms)") 
            ax.legend()
        temporal_ax[ri] = ax


    # add a reset ERP button
    def reset(event):
        print('reset called')
        nonlocal Cxx, Cxy, Cyy
        Cxx=None
        Cxy=None
        Cyy=None
        nY.fill(0)

    from matplotlib.widgets import Button
    butax = fig.add_axes([.05,.85,.1,.1])
    breset = Button(butax,'Reset')
    breset.on_clicked(reset)

    # tidy up the axes locations
    plt.tight_layout()
    fig.show()
    
    # start the render loop
    t0 = ui.getTimeStamp()
    block_start_ts = t0
    oM = None  # last few stimulus states, for correct transformation of later events
    dirty=True # flag if we shoudl redraw the window
    while ui.getTimeStamp() < t0+maxruntime_ms:

        # exit if figure is closed..
        if not plt.fignum_exists(1):
            quit()

        # re-draw the display
        #fig.canvas.draw()
        fig.canvas.flush_events()
        if dirty : 
            fig.canvas.draw()
            dirty=False
        #fig.canvas.start_event_loop(0.1)
        #plt.pause(.001)

        # Update the records
        nmsgs, ndata, nstim = ui.update(timeout_ms=timeout_ms)
        if nstim == 0:
            continue
        
        # last time for which we have full response for a stimulus
        valid_end_ts = min(ui.stimulus_timestamp + irflen_ms, ui.data_timestamp)

        # skip if no new data to process
        if block_start_ts + irflen_ms > valid_end_ts:
            continue 

        block_end_ts = valid_end_ts

        # extract and apply to this block
        print("Extract block: {}->{} = {}ms".format(block_start_ts, block_end_ts, block_end_ts-block_start_ts))
        data = ui.extract_data_segment(block_start_ts, block_end_ts)
        stimulus = ui.extract_stimulus_segment(block_start_ts, block_end_ts)
        block_start_ts = data[-irflen_samp,-1] # block_end_ts - irflen_ms

        # skip if no data/stimulus to process
        if data.size == 0 or stimulus.size == 0:
            continue

        # upsample stimulus to sample rate
        X_ts = data[:, -1]
        X = data[:, :-1]
        M_ts = stimulus[:, -1]
        M = stimulus[:, :-1]
        M, _ = upsample_stimseq(X_ts, M, M_ts)

        # TODO[]: limit to active outputs

        # and transform to irf trigger events
        Y, labs = stim2event(M, evtypes=evtlabs, axis=-2, oM=oM)  # (nsamp,nstim,e)
        # record the last bit before the processed M for next time, i.e. 
        oM = M[:-irflen_samp, :]

        # strip to only the objID==0 output
        Y = Y[:, 0:1, :]  # (nsamp,1,nE)
        # count the numbers each type of event
        nY = nY + np.sum(Y,axis=(0,1))

        # incrementally update the summary statistics
        Cxx, Cxy, Cyy = updateSummaryStatistics(X, Y, None, Cxx, Cxy, Cyy, tau=irflen_samp)

        # debug plot summary statistics.
        erp_lim = (np.min(Cxy.ravel()),np.max(Cxy.ravel()))
        for ei,lab in enumerate(labs):
            erp_lines[ei].set_data(Cxy[0,ei,:,:].T)
            erp_lines[ei].set_clim(vmin=erp_lim[0],vmax=erp_lim[1])
            #erp_ax[ei].imshow(Cxy[0,ei,:,:].T/nY[ei],aspect='auto',extent=(irf_times[0],irf_times[-1],0,Cxy.shape[-1]))
            erp_ax[ei].set_title("{} (n={})".format(lab,nY[ei]))

        # update the cca decomposition
        J, W, R = multipleCCA(Cxx, Cxy, Cyy, reg=reg, rank=rank)
        # strip the model dimension
        J = J[0]
        W = W[0,...]
        R = R[0,...]

        R_lim = (-np.max(np.abs(R.ravel())) * 2, np.max(np.abs(R.ravel())) * 2)
        R_lim = [ d if not np.isnan(d) else 0 for d in R_lim ] # guard

        W_lim = (-np.max(np.abs(W.ravel()))*2, np.max(np.abs(W.ravel()))*2)
        W_lim = [ d if not np.isnan(d) else 0 for d in W_lim ] # guard

        # Update the plots for each event type
        dirty = True
        for ri in range(rank):
            sgn = np.sign(W[ri,np.argmax(np.abs(W[ri,:]))]) # normalize directions

            # plot the temporal responses
            for ei in range(len(evtlabs)):
                temporal_lines[ri][ei].set_ydata(R[ri,ei,:]*sgn)
            temporal_ax[ri].set_ylim( R_lim )

            # plot the spatial patterns
            # TODO[]: as a topographic plot
            if not ch_pos is None: # make as topoplot
                # BODGE: deal with co-linear inputs by replacing each channel with a triangle of points
                interp_pos = np.concatenate((ch_pos+ np.array((0,.1)), ch_pos + np.array((-.1,-.05)), ch_pos + np.array((+.1,-.05))),0)
                spatial_lines[ri]=ax.tricontourf(interp_pos[:,0],interp_pos[:,1],np.tile(W[ri,:]*sgn,3),cmap='Spectral')

                #spatial_lines[ri]=ax.tricontourf(ch_pos[:,0],ch_pos[:,1],np.zeros((irf.shape[-1]),cmap='Spectral')
                #spatial_lines[ri].set_ydata(W[ri,:]*sgn)
                #spatial_ax[ri].set_clim(W_lim)
            else:
                spatial_lines[ri][-1].set_ydata(W[ri,:]*sgn)
                spatial_ax[ri].set_ylim(W_lim)


def parse_args():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--evtlabs', type=str, help='comma separated list of stimulus even types to use', default='re,fe')
    parser.add_argument('--out_fs',type=int, help='output sample rate', default=100)
    parser.add_argument('--stopband',type=json.loads, help='set of notch filters to apply to the data before analysis', default=((45,65),(5.5,25,'bandpass')))
    parser.add_argument('--rank', type=str, help='rank of decomposition to use', default=3)
    parser.add_argument('--ch_names', type=str, help='list of channel names, or capfile', default=None)
    parser.add_argument('--savefile', type=str, help='run decoder using this file as the proxy data source', default=None)
    parser.add_argument('--savefile_fs', type=float, help='effective sample rate for the save file', default=None)
    parser.add_argument('--savefile_speedup', type=float, help='play back the save file with this speedup factor. None means fast as possible', default=None)
    parser.add_argument('--timeout_ms', type=float, help="timeout for wating for new data from hub",default=100)
    args = parser.parse_args()
    if args.evtlabs: 
        args.evtlabs = args.evtlabs.split(',')
    if args.ch_names:
        args.ch_names = args.ch_names.split(',')
        
    return args


if __name__=='__main__':
    args = parse_args()

    if True:
        args.savefile = '~/Desktop/logs/mindaffectBCI*.txt'
        args.ch_names = 'C1,Cz,C2,C3'.split(',') 
        args.savefile_speedup=1
        args.timeout_ms = 1000

    if args.savefile is not None:
        from mindaffectBCI.decoder.FileProxyHub import FileProxyHub
        U = FileProxyHub(args.savefile,use_server_ts=True,speedup=args.savefile_speedup)
        ppfn = butterfilt_and_downsample(order=6, stopband=args.stopband, fs_out=args.out_fs, ftype='butter')
        ui = UtopiaDataInterface(data_preprocessor=ppfn,
                                 send_signalquality=False,
                                 timeout_ms=args.timeout_ms, mintime_ms=0, U=U, fs=args.savefile_fs, clientid='viewer')
    else:
        data_preprocessor = butterfilt_and_downsample(order=6, stopband=args.stopband, fs_out=args.out_fs)
        ui=UtopiaDataInterface(data_preprocessor=data_preprocessor, send_signalquality=False, clientid='viewer')
        ui.connect(args.host)

    try:
        matplotlib.use('Qt4Cairo')
    except:
        pass

    ccaViewer(ui, **vars(args))
