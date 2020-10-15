import numpy as np
from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
from mindaffectBCI.utopiaclient import  DataPacket
from mindaffectBCI.decoder.model_fitting import MultiCCA
from mindaffectBCI.decoder.utils import window_axis
import matplotlib.pyplot as plt
import glob


def trigger_check(filename=None, evtlabs=('re','fe'), tau_ms=125, offset_ms=-25, max_samp=6000, stopband=(.1,45,'bandpass'), fs_out=250, **kwargs):
    """make a set of visualizations of the stimulus->measurement time-lock

    Args:
        filename ([type], optional): The name of the file to load. Defaults to None.
        evtlabs (tuple, optional): The event coding to use for how stimulus changes map into trigger inputs. Defaults to ('0','1').
        tau_ms (int, optional): The duration of the stimulus response. Defaults to 400.
        offset_ms (int, optional): Offset of the start of the stimulus response w.r.t. the trigger time. Defaults to -50.
        max_samp (int, optional): Limit in the number of samples for each trial. Defaults to 6000.
        stopband (tuple, optional): Temporal filter used to pre-process the EEG. Defaults to (.1,45,'bandpass').
        fs_out (int, optional): Sample rate of the EEG used for analysis. Defaults to 250.
    """   
    import glob
    import os
    if filename is None or filename == '-':
        # default to last log file if not given
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
        filename = max(files, key=os.path.getctime)
    else:
        files = glob.glob(os.path.expanduser(filename))
        filename = max(files, key=os.path.getctime)    

    X, Y, coords = load_mindaffectBCI(filename, stopband=stopband, fs_out=fs_out, **kwargs)
    # size limit...
    if max_samp is not None and X.shape[1] > max_samp:
        X=X[:,:max_samp,...]
        Y=Y[:,:max_samp,...]
    X[...,:-1] = X[...,:-1] - np.mean(X[...,:-1],axis=-2,keepdims=True) # offset remove
    fs = coords[-2]['fs']
    print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,coords[1]['fs']))
    print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:-1]]+['output'],Y.shape))

    plt.figure(1)
    plt.clf()
    for i in range(min(X.shape[0],3)):
        plt.subplot(3,1,i+1)
        #plt.imshow(X[0,...].T,aspect='auto',label='X',extent=[0,X.shape[-2],0,X.shape[-1]]);
        for c in range(X.shape[-1]):
            tmp = X[i,...,c]
            tmp = (tmp - np.mean(tmp.ravel())) / max(1,np.std(tmp.ravel()))
            plt.plot(tmp+2*c,label='X{}'.format(c))
        plt.plot(Y[i,...,0],'k',label='Y')
        plt.title('Trl {}'.format(i))
    plt.legend()
    plt.suptitle('{}\nFirst trials data vs. stimulus'.format(filename))
    plt.show(block=False)
    plt.pause(.1) # allow full redraw

    wXe, wXeY, Y_true = triggerPlot(X, Y, fs, evtlabs=evtlabs, tau_ms=tau_ms, offset_ms=offset_ms, stopband=stopband, **kwargs)
    times = (np.arange(wXe.shape[-1])+offset_ms*1000/fs)*1000/fs
    plt.show(block=False)

    # over-plot the information on the timestamp errors
    # get a samp#, timestamp dataset
    trl_ts = coords[0]['trl_ts'][:,:wXe.shape[1]]   # (nTrl, nSamp-tau)
    trl_idx = coords[0]['trl_idx'][:,:wXe.shape[1]] # (nTrl, nSamp-tau)
    samp2ms = np.median((np.diff(trl_ts,axis=-1)/np.maximum(1,np.diff(trl_idx,axis=-1))).ravel())
    samp2ms = 1000/fs #samp2ms*.99
    print("samp2ms={}".format(samp2ms))
    lin_trl_ts = trl_idx*samp2ms # linear time-stamp assuming constant sample rate
    ts_err = trl_ts - lin_trl_ts # (tr, nSamp-tau) error btw. linear time-stamp and observed
    ts_err = ts_err - ts_err[0,0]
    ts_errY = ts_err[Y_true>0] # (ep,) slice out the high-stimulus info (in seconds)

    ax2=fig.axes[0].twinx()
    plt.plot(ts_errY,'k-')
    mu2 = np.median(ts_errY.ravel())
    scale2 = np.median( np.abs(ts_errY.ravel()-mu2) )
    if times[-1]*.05 < scale2*2 and scale2*2 < times[-1]*8: # make line match
        scale2 = times[-1]/2
    elif scale2*2 < times[-1]*.2: # make image smaller
        ax.set_ylim(min(times[0],mu2-2*scale2),max(times[-1],mu2+2*scale2))
    plt.ylim(mu2-1*scale2,mu2+1*scale2)
    plt.ylabel("Recieved time-stamp error vs. constant rate (ms)")
    plt.show(block=False)

    # allow full re-draw
    plt.pause(.5)

    # now plot as lines
    plt.figure(5)
    tmp = wXeY.reshape((-1,wXeY.shape[-1]))[:100,:]
    tmp = tmp - np.mean(tmp,-1,keepdims=True)
    tmp = tmp / np.std(tmp.ravel())
    plt.plot(times, tmp.T + np.arange(tmp.shape[0])[np.newaxis,:])
    plt.title('1st {} {} evt trigger epochs time-series'.format(tmp.shape[0],evtlabs[0]))

    # do a time-stamp check.
    from mindaffectBCI.decoder.timestamp_check import timestampPlot
    plt.figure(6)
    timestampPlot(filename)

def triggerPlot(X,Y,fs, evtlabs=('re','fe'), tau_ms=125, offset_ms=-25, max_samp=6000, max_trl=5, stopband=(.1,45,'bandpass'), plot_model=True, plot_trial=True, ax=None, **kwargs):
    if X.ndim < 3 : 
        X = X[np.newaxis, ...]
    if Y.ndim < 3:
        Y = Y[np.newaxis, ...]
    
    print("training model")
    tau = int(fs*tau_ms/1000.0)
    # BODGE: for speed only use first 5 trials!
    # BODGE: reg with 1e-5 so only the strong channels are used..
    clsfr = MultiCCA(evtlabs=evtlabs,tau=tau,rank=1,reg=(1e-2,None)).fit(X[-max_trl:,-max_samp:,...],Y[-max_trl:,-max_samp:,...])

    # get the event-coded version of Y
    Ye = clsfr.stim2event(Y)

    print("applying spatial filter")
    W = clsfr.W_[0,0,...] # (d,)
    print("W={}".format(W))

    # slice the data w.r.t. the stimulus triggers to generate the visualization
    offset = int(fs*offset_ms/1000.0)
    times = (np.arange(tau)+offset)*1000/fs
    wX = np.einsum("d,Ttd->Tt", W, X) # (nTrl, nSamp) apply the spatial filter

    if plot_trial:
        plt.figure(1)
        # add as line to the trl plot:
        for i in range(min(X.shape[0],3)):
            plt.subplot(3,1,i+1)
            tmp = wX[i,...]
            tmp = (tmp - np.mean(tmp.ravel())) / max(.01,np.std(tmp.ravel()))
            plt.plot(tmp+2*(X.shape[-1]+1),label='wX')
        plt.legend()

    if plot_model:
        # model in a new figure
        plt.figure(2)
        clsfr.plot_model(fs=fs)
        plt.show(block=False)
        # allow full re-draw
        plt.pause(.5)

    # slice out the responses for the trigger stimulus
    print('slicing data')
    wXe = window_axis(wX, winsz=tau, axis=-1) # (nTrl, nSamp-tau, tau)
    # N.B. negative offset here as negative shift of Y is same as positive shift of X
    Y_true = Ye[...,-offset:wXe.shape[1]-offset,0,0] # (nTrl, nSamp-tau)
    wXeY = wXe[Y_true>0,:] # [ep,tau]

    print('generating plot')
    mu = np.median(wXeY.ravel())
    scale = np.median( np.abs(wXeY.ravel()-mu) )
    if ax is None:
        ax = plt.axes()
        newax = True
    else:
        newax = False
    plt.sca(ax)
    plt.imshow(wXeY.T,origin='lower',aspect='auto',extent=[0,wXeY.shape[0],times[0],times[-1]])
    plt.title('{} locked {} trials {:4.1f}min data'.format(evtlabs[0],X.shape[0],(X.shape[0]*X.shape[1])/fs/60))
    plt.clim(mu-scale,mu+scale)
    if newax:
        plt.colorbar()
        plt.set_cmap('jet')#'gray')#'nipy_spectral')
        plt.ylabel('time (ms)')
        plt.xlabel('Epoch')
        plt.grid()

    # if X.shape[-1]>8 :
    #     samp_cnt = X[...,-1] # sample count (%255)
    #     samp_miss = samp_cnt[...,1:] - ((samp_cnt[...,:-1]+1) % 256)
    #     samp_miss = samp_miss[...,:X.shape[1]]
    #     samp_miss = samp_miss[Y_true>0]
    #     plt.plot(samp_miss*samp2ms + 100,'k-')
    return ax, wXe, wXeY, Y_true

def run(hostname='-', stopband=(.1,45,'bandpass'), fs_out=250, **kwargs):
    """online continuously updating trigger check plot
    """
    from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, stim2eventfilt, butterfilt_and_downsample
    from mindaffectBCI.decoder.devent2stimsequence import devent2stimSequence, upsample_stimseq
    data_preprocessor = butterfilt_and_downsample(order=6, stopband=stopband, fs_out=fs_out)
    ui=UtopiaDataInterface(data_preprocessor=data_preprocessor, send_signalquality=False)
    ui.connect(hostname)
    try:
        matplotlib.use('Qt4Cairo')
    except:
        pass

    fig = plt.gcf()
    ax = None
    data=None
    stimulus=None
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
            data = np.append(data,rawdata,axis=-2)

        # skip if no stimulus events to process
        if nstim == 0:
            continue

        # get the new raw stimulus 
        rawstimulus = ui.stimulus_ringbuffer[-nstim:, :]
        if rawstimulus.size > 0 :
            # append to the stimulus info
            if stimulus is None:
                stimulus = rawstimulus
            else:
                if np.any(rawstimulus[0,-1] == stimulus[:,-1]):
                    print('loop')
                stimulus = np.append(stimulus,rawstimulus, axis=-2)
            

        # update the trigger check plot
        # TODO []: make it nicer / more efficient
        data_ts = data[:,-1] 
        stim_ts = stimulus[:,-1]
        # up-sample to the data rate by matching time-stamps
        usstimulus, _ = upsample_stimseq(data_ts, stimulus, stim_ts)
        ax, _, _, _ = triggerPlot(data,usstimulus,ui.fs, new_fig=False, plot_model=False, plot_trial=False, ax=ax, **kwargs)


if __name__=="__main__":
    offline = False
    if offline:
        #filename="~/Desktop/trig_check/mindaffectBCI_*brainflow*.txt"
        #filename = '~/Desktop/rpi_trig/mindaffectBCI_*_201001_1859.txt'
        #filename = '~/Desktop/trig_check/mindaffectBCI_*wifi*tschannel*.txt'
        #filename = '~/Desktop/trig_check/mindaffectBCI_*_khash2.txt'
        #filename=None
        filename='~/Desktop/pymindaffectBCI/logs/mindaffectBCI_*.txt'
        trigger_check(filename, evtlabs=('re','fe'), tau_ms=125, offset_ms=-25, stopband=(0,.5), fs_out=250)
    else:
        run(evtlabs=('re','fe'), tau_ms=125, offset_ms=-25, stopband=(0,.5), fs_out=250)

