#!/usr/bin/env python3
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
from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, butterfilt_and_downsample
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics
from mindaffectBCI.decoder.multipleCCA import multipleCCA
from mindaffectBCI.decoder.stim2event import stim2event
from mindaffectBCI.decoder.devent2stimsequence import upsample_stimseq

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def run(ui: UtopiaDataInterface=None, maxruntime_ms: float=np.inf, timeout_ms:float = 500, tau_ms: float=500,
              offset_ms=(-15, 0), evtlabs=None, ch_names=None, ch_pos=None, topo_colorbar:bool=False, nstimulus_events: int=600, 
              rank:int=3, reg=.02, center:bool=True, host:str='-', filterband=None, out_fs=100, noutputs:int=1, **kwargs):
    """on-line view of the CCA Forward-Backward model of the data

    Args:
        ui (UtopiaDataInterface): UtopiaDataInterface to get the live data from the hub.  If None then we will make a new data-interface and try to connect to the given hub information.
        host (str, optional): host name where the data hub is running. Defaults to '-'.
        filterband ([type], optional): data filtering parameters to use in the UtopiaDataInterface, if not given. Defaults to None.
        out_fs (int, optional): sampling rate of the output of the UtopiaDataInterface - if not given. Defaults to 100.
        maxruntime_ms (float, optional): terminate automatically after this long. Defaults to np.inf.
        timeout_ms (float, optional): max time to wait for data from hub, equals the max-plot redraw rate. Defaults to 500.
        tau_ms (float, optional): length of the estimated impulse response. Defaults to 500.
        offset_ms (tuple, optional): offset for the start/end of the impulse response w.r.t. the trigger. Defaults to (-15, 0).
        evtlabs (list, optional): the types of events used to model the brain response.  See `stim2event` for more information on the types of transformation allowed. Defaults to None.
        ch_names (list, optional): names of the measurement channels -- used to get channel-positions of ch_pos is not given. Defaults to None.
        ch_pos (list, optional): channel positions. Defaults to None.
        nstimulus_events (int, optional): max number of stimulus events to include in the running Event-Related-Potential. Defaults to 600.
        rank (int, optional): number of CCA components to plot. Defaults to 3.
        reg (float, optional): regularization strength used in the cca decomposition. Defaults to .02.
        center (bool, optional): center the data before computing the cca decomposition. Defaults to True.
    """    
    if ui is None:
        data_preprocessor = butterfilt_and_downsample(order=6, filterband=filterband, fs_out=out_fs)
        ui=UtopiaDataInterface(data_preprocessor=data_preprocessor, send_signalquality=False)
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
    Cxx_dd = None
    Cxy_yetd = None
    Cyy_yetet = None
    
    # initialize the plot window
    fig = plt.figure(1)
    fig.clear()

    # main spec, inc titles; ERP | CCA
    outer_grid = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1,6], width_ratios=[1,2])

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
    gs = outer_grid[-1,1].subgridspec(nrows=rank, ncols=2, width_ratios=[1,2])
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
            spatial_lines[ri]=ax.tricontourf(interp_pos[:,0],interp_pos[:,1],np.tile(w,3),cmap='bwr')

            for i,(x,y) in enumerate(ch_pos):
                ax.text(x,y,ch_names[i],ha='center',va='center') # label
            ax.set_aspect(aspect='equal')
            ax.set_frame_on(False) # no frame
            ax.tick_params(labelbottom=False,labelleft=False,which='both',bottom=False,left=False) # no labels, ticks
            if topo_colorbar:
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
        nonlocal Cxx_dd, Cxy_yetd, Cyy_yetet
        Cxx_dd=None
        Cxy_yetd=None
        Cyy_yetet=None
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
    block_start_ts = None
    oM_Sy = None  # last few stimulus states, for correct transformation of later events
    dirty=True # flag if we shoudl redraw the window
    usedY = np.zeros((256,),dtype=bool)
    if noutputs is not None: usedY[1:noutputs+1]=True  # force min used out
    while ui.getTimeStamp() < t0+maxruntime_ms:

        # exit if figure is closed..
        if not plt.fignum_exists(1):
            quit()

        # re-draw the display
        #fig.canvas.draw()
        fig.canvas.flush_events()
        if dirty : 
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            dirty=False
        #fig.canvas.start_event_loop(0.1)
        #plt.pause(.001)

        # Update the records
        nmsgs, ndata, nstim = ui.update(timeout_ms=timeout_ms)
        if block_start_ts is None and ndata>0: block_start_ts = ui.data_timestamp
        if nstim == 0:
            continue
        
        # last time for which we have full response for a stimulus
        valid_end_ts = min(ui.stimulus_timestamp + irflen_ms, ui.data_timestamp)

        # skip if no new data to process
        if valid_end_ts > block_start_ts and block_start_ts + irflen_ms > valid_end_ts:
            continue 

        block_end_ts = valid_end_ts

        # extract and apply to this block
        print("Extract block: {}->{} = {}ms".format(block_start_ts, block_end_ts, block_end_ts-block_start_ts))
        data = ui.extract_data_segment(block_start_ts, block_end_ts)
        stimulus = ui.extract_stimulus_segment(block_start_ts, block_end_ts)

        # skip if no data/stimulus to process
        if data.size == 0 or stimulus.size == 0 or data.shape[0]<irflen_samp:
            continue


        block_start_ts = data[-irflen_samp,-1] # block_end_ts - irflen_ms


        # upsample stimulus to sample rate
        ts_S = data[:, -1] # sample time-stamps
        X_Sd = data[:, :-1] # (samp,d)
        ts_s = stimulus[:, -1] # (stimulus time-stamps)
        M_sy = stimulus[:, :-1] # (stim,nY)
        M_Sy, _ = upsample_stimseq(ts_S, M_sy, ts_s) # (samp,nY)

        # limit to only the active outputs
        usedY = np.logical_or(usedY,np.any(M_sy,0))

        # and transform to irf trigger events
        Y_Sye, _, labs = stim2event(M_Sy[:,usedY], evtypes=evtlabs, axis=-2, 
                                    oM=oM_Sy[:,usedY] if oM_Sy is not None else None) 
        # record the last bit before the processed M for next time, i.e. 
        oM_Sy = M_Sy[:-irflen_samp, :]

        if np.all(Y_Sye[:-irflen_samp,...]==0): # no trigger events 
            print("No trigger events... skipping..")
            continue

        # strip to only the objID==0 output        
        Y_Sye = Y_Sye[:, 0:1, :]  # (nsamp,1,nE)
        # count the numbers each type of event
        nY = nY + np.sum(Y_Sye[:-irflen_samp,...],axis=(0,1))

        # incrementally update the summary statistics
        Cxx_dd, Cxy_yetd, Cyy_yetet = updateSummaryStatistics(X_Sd, Y_Sye, None, Cxx_dd, Cxy_yetd, Cyy_yetet, tau=irflen_samp)

        # debug plot summary statistics.
        erp_lim = (np.min(Cxy_yetd.ravel()),np.max(Cxy_yetd.ravel()))
        for ei,lab in enumerate(labs):
            erp_lines[ei].set_data(Cxy_yetd[0,ei,:,:].T)
            erp_lines[ei].set_clim(vmin=erp_lim[0],vmax=erp_lim[1])
            #erp_ax[ei].imshow(Cxy[0,ei,:,:].T/nY[ei],aspect='auto',extent=(irf_times[0],irf_times[-1],0,Cxy.shape[-1]))
            erp_ax[ei].set_title("{} (n={})".format(lab,nY[ei]))

        # update the cca decomposition
        J, W_mkd, R_mket, A_mkd, I_mket = multipleCCA(Cxx_dd, Cxy_yetd, Cyy_yetet, reg=reg, rank=rank)
        # strip the model dimension
        J = J[0]
        W_kd = W_mkd[0,...]
        R_ket = R_mket[0,...]

        R_lim = (-np.max(np.abs(R_ket)), np.max(np.abs(R_ket)))
        R_lim = [ d if not np.isnan(d) else 0 for d in R_lim ] # guard

        W_lim = (-np.max(np.abs(W_kd)), np.max(np.abs(W_kd)))
        W_lim = [ d if not np.isnan(d) else 0 for d in W_lim ] # guard

        # Update the plots for each event type
        dirty = True
        for ri in range(rank):
            sgn = np.sign(W_kd[ri,np.argmax(np.abs(W_kd[ri,:]))]) # normalize directions

            # plot the temporal responses
            for ei in range(len(evtlabs)):
                temporal_lines[ri][ei].set_ydata(R_ket[ri,ei,:]*sgn)
            temporal_ax[ri].set_ylim( R_lim )

            # plot the spatial patterns
            # TODO[]: as a topographic plot
            if not ch_pos is None: # make as topoplot
                levels = np.linspace(W_lim[0],W_lim[1],7)
                # BODGE: deal with co-linear inputs by replacing each channel with a triangle of points
                interp_pos = np.concatenate((ch_pos+ np.array((0,.1)), ch_pos + np.array((-.1,-.05)), ch_pos + np.array((+.1,-.05))),0)
                spatial_lines[ri]=spatial_ax[ri].tricontourf(interp_pos[:,0],interp_pos[:,1],np.tile(W_kd[ri,:]*sgn,3),levels=levels,cmap='bwr')

            else:
                spatial_lines[ri][-1].set_ydata(W_kd[ri,:]*sgn)
                spatial_ax[ri].set_ylim(W_lim)


def parse_args():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--evtlabs', type=str, help='comma separated list of stimulus even types to use', default='re,fe')
    parser.add_argument('--out_fs',type=int, help='output sample rate', default=100)
    parser.add_argument('--tau_ms',type=float, help='output sample rate', default=450)
    parser.add_argument('--filterband',type=json.loads, help='set of notch filters to apply to the data before analysis', default=((45,65),(5.5,25,'bandpass')))
    parser.add_argument('--rank', type=str, help='rank of decomposition to use', default=1)
    parser.add_argument('--ch_names', type=str, help='list of channel names, or capfile', default=None)
    parser.add_argument('--savefile', type=str, help='run decoder using this file as the proxy data source', default=None)
    parser.add_argument('--savefile_fs', type=float, help='effective sample rate for the save file', default=None)
    parser.add_argument('--savefile_speedup', type=float, help='play back the save file with this speedup factor. None means fast as possible', default=None)
    parser.add_argument('--timeout_ms', type=float, help="timeout for wating for new data from hub, equals min-redraw time.",default=500)
    args = parser.parse_args()
    if args.evtlabs: 
        args.evtlabs = args.evtlabs.split(',')
    if args.ch_names:
        args.ch_names = args.ch_names.split(',')
        
    return args


if __name__=='__main__':
    args = parse_args()

    if False:
        args.savefile = '~/Desktop/logs/mindaffectBCI*.txt'
        args.ch_names = 'P5,P4,Oz,T6'.split(',') 
        args.savefile_speedup=None
        args.timeout_ms = 1000

    if args.savefile is not None:
        from mindaffectBCI.decoder.FileProxyHub import FileProxyHub
        U = FileProxyHub(args.savefile,use_server_ts=True,speedup=args.savefile_speedup)
        ppfn = butterfilt_and_downsample(order=6, filterband=args.filterband, fs_out=args.out_fs, ftype='butter')
        ui = UtopiaDataInterface(data_preprocessor=ppfn,
                                 send_signalquality=False,
                                 timeout_ms=args.timeout_ms, mintime_ms=0, U=U, fs=args.savefile_fs, clientid='viewer')
    else:
        data_preprocessor = butterfilt_and_downsample(order=6, filterband=args.filterband, fs_out=args.out_fs)
        ui=UtopiaDataInterface(data_preprocessor=data_preprocessor, send_signalquality=False, clientid='viewer')
        ui.connect(args.host)

    run(ui, **vars(args))
