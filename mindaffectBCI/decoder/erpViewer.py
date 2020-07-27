#!/usr/bin/env python3
import numpy as np
from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, stim2eventfilt, butterfilt_and_downsample


def erpViewer(ui: UtopiaDataInterface, timeout_ms: float=np.inf, tau_ms: float=500,
              offset_ms=(-15, 0), evtlabs=None, nstimulus_events: int=600, center=True):
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
    # get grid-spec to layout the plots
    gs = fig.add_gridspec(len(evtlabs),4)
    spatial_ax = [None]*len(evtlabs)
    spatial_lines = [None]*len(evtlabs)
    temporal_ax = [None]*len(evtlabs)
    temporal_lines = [None]*len(evtlabs)
    # make the bottom 2 axs already so can share their limits.
    spatial_ax[-1] = fig.add_subplot(gs[-1, 0])
    temporal_ax[-1] = fig.add_subplot(gs[-1, 1:])
    for ei, lab in enumerate(evtlabs):
        # spatial-plot
        if ei < len(evtlabs)-1:
            ax = fig.add_subplot(gs[ei, 0], sharex=spatial_ax[-1], sharey=spatial_ax[-1])
            ax.tick_params(labelbottom=False, labelleft=False)
        else:
            ax = spatial_ax[-1]
            ax.set_xticklabels('on')
            ax.set_xlabel("Time (ms)")
        ax.set_title("{}".format(lab))
        ax.grid(True)
        spatial_ax[ei] = ax
        spatial_lines[ei] = ax.plot(np.zeros((irf.shape[-1])), color=(0, 0, 0))

        # temporal plot
        if ei < len(evtlabs)-1:
            ax = fig.add_subplot(gs[ei, 1:], sharex=temporal_ax[-1], sharey=temporal_ax[-1])
            ax.tick_params(labelbottom=False, labelleft=False)
        else:
            ax = temporal_ax[-1]
            ax.set_xlabel("Space")
        ax.set_title("{}".format(lab))
        ax.grid(True)
        ax.autoscale(axis='y')
        temporal_ax[ei] = ax
        temporal_lines[ei] = ax.plot(irf_times, np.ones((irf.shape[-2], 10)), color=(.9, .9, .9))
        temporal_lines[ei].extend(ax.plot(irf_times, np.ones((irf.shape[-2], 1)), color=(0, 0, 0)))

    fig.show()
    
    # start the render loop
    t0 = ui.getTimeStamp()
    pending = []  # list of stimulus events for which still waiting for data to complete
    while ui.getTimeStamp() < t0+timeout_ms:

        # exit if figure is closed..
        if not plt.fignum_exists(1):
            exit()

        # re-draw the display
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(.05)

        # Update the records
        nmsgs, ndata, nstim = ui.update(timeout_ms=100)

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
            #print("erp={}".format(erp))
            # decompose into spatial and temporal components
            if False:
                # TODO: use multiCCA to get the decomp?
                R, s, A = np.linalg.svd(erp)
                # get the biggest eigenvalue to display
                si = np.argmax(s)
                R = R[:, si]  #(nsamp,)
                s = s[si]
                A = A[si, :]  #(d,)
            else:
                A = np.zeros((erp.shape[-1])) 
                A[-1] = 1  #(1,d)
                R = erp[:, -1:]  # @ A
                
            # apply the spatial filter to the raw data to get pure time-course
            data = data @ A
            # plot the spatial pattern
            spatial_lines[ei][-1].set_ydata(A)
            spatial_ax[ei].set_ylim((np.min(A), np.max(A)))
            #temporal_ax[ei].plot(irf_times, R, color=(0,0,0), label='erp ({})'.format(data.shape[0]))
            for di in range(min(len(temporal_lines[ei])-1, data.shape[0])):
                temporal_lines[ei][-1-di-1].set_ydata(data[-1-di, :])
                
            # plot the average response, as thick black                
            temporal_lines[ei][-1].set_ydata(R)
            
            temporal_ax[ei].set_title("{} ({})".format(lab, data.shape[0]))
            temporal_ax[ei].set_ylim( (np.min(R.ravel())*2, np.max(R.ravel()) * 2) )

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--evtlabs', type=str, help='comma separated list of stimulus even types to use', default='re,fe')
    args = parser.parse_args()
    hostname = args.host
    evtlabs = args.evtlabs.split(',')
    print('evtlabs={}'.format(evtlabs))

    data_preprocessor = None
    data_preprocessor = butterfilt_and_downsample(order=6, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=60)
    #data_preprocessor = butterfilt_and_downsample(order=4, stopband=((0, 4), (25, -1)), fs_out=60)
    ui=UtopiaDataInterface(data_preprocessor=data_preprocessor)
    ui.connect(hostname)

    try:
        matplotlib.use('Qt4Cairo')
    except:
        pass

    erpViewer(ui, evtlabs=evtlabs)
