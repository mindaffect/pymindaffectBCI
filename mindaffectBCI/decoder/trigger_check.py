import numpy as np
from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
from mindaffectBCI.utopiaclient import  DataPacket
from mindaffectBCI.decoder.model_fitting import MultiCCA
from mindaffectBCI.decoder.utils import window_axis
import matplotlib.pyplot as plt
import glob

def triggerPlot(filename=None, evtlabs=('0','1'), tau_ms=400, offset_ms=-50, max_samp=6000, stopband=(.1,45,'bandpass'), fs_out=250):
    import glob
    import os
    if filename is None or filename == '-':
        # default to last log file if not given
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
        filename = max(files, key=os.path.getctime)
    else:
        files = glob.glob(os.path.expanduser(filename))
        filename = max(files, key=os.path.getctime)
    print("Loading : {}\n".format(filename))    

    # TODO[] : correctly use the evtlabs
    X, Y, coords = load_mindaffectBCI(filename, stopband=stopband, fs_out=fs_out)
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

    print("training model")
    tau = int(fs*tau_ms/1000.0)
    # BODGE: for speed only use first 5 trials!
    # BODGE: reg with 1e-5 so only the strong channels are used..
    clsfr = MultiCCA(evtlabs=evtlabs,tau=tau,rank=1,reg=(1e-2,None)).fit(X[:5,:1000,...],Y[:5,:1000,...])

    # get the event-coded version of Y
    Ye = clsfr.stim2event(Y)

    print("applying spatial filter")
    W = clsfr.W_[0,0,...] # (d,)
    print("W={}".format(W))

    # slice the data w.r.t. the stimulus triggers to generate the visualization
    offset = int(fs*offset_ms/1000.0)
    times = (np.arange(tau)+offset)*1000/fs
    wX = np.einsum("d,Ttd->Tt", W, X) # (nTrl, nSamp) apply the spatial filter

    # add as line to the trl plot:
    for i in range(min(X.shape[0],3)):
        plt.subplot(3,1,i+1)
        tmp = wX[i,...]
        tmp = (tmp - np.mean(tmp.ravel())) / max(.01,np.std(tmp.ravel()))
        plt.plot(tmp+2*(X.shape[-1]+1),label='wX')
    plt.legend()

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

    print('generating plot')
    plt.figure(3)
    mu = np.median(wXeY.ravel())
    scale = np.median( np.abs(wXeY.ravel()-mu) )
    fig, ax = plt.subplots()
    plt.imshow(wXeY.T,origin='lower',aspect='auto',extent=[0,wXeY.shape[0],times[0],times[-1]])
    plt.clim(mu-scale,mu+scale)
    plt.set_cmap('nipy_spectral')
    plt.colorbar()
    plt.ylabel('time (ms)')
    plt.xlabel('Epoch')
    plt.title('{}\n{}'.format(evtlabs[0],filename[-50:]))
    plt.grid()
    #plt.colorbar()

    if X.shape[-1]>8 :
        samp_cnt = X[...,-1] # sample count (%255)
        samp_miss = samp_cnt[...,1:] - ((samp_cnt[...,:-1]+1) % 256)
        samp_miss = samp_miss[...,:Xe.shape[1]]
        samp_miss = samp_miss[Y_true>0]
        plt.plot(samp_miss*samp2ms + 100,'k-')

    # over-plot the error
    ax2=ax.twinx()
    plt.plot(ts_errY,'k-')
    mu2 = np.median(ts_errY.ravel())
    scale2 = np.median( np.abs(ts_errY.ravel()-mu) )
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
    plt.plot(tmp.T + np.arange(tmp.shape[0])[np.newaxis,:])
    plt.title('1st {} {} evt trigger epochs time-series'.format(tmp.shape[0],evtlabs[0]))

    # do a time-stamp check.
    from mindaffectBCI.decoder.timestamp_check import timestampPlot
    plt.figure(6)
    timestampPlot(filename)

if __name__=="__main__":
    #filename="~/Desktop/trig_check/mindaffectBCI_*brainflow*.txt"
    #filename = '~/Desktop/rpi_trig/mindaffectBCI_*_201001_1859.txt'
    filename = '~/Desktop/trig_check/mindaffectBCI_*wifi*khash*.txt'
    #filename = '~/Desktop/trig_check/mindaffectBCI_*_khash2.txt'
    #filename=None
    #filename='c:/Users/Developer/Desktop/pymindaffectBCI/logs/mindaffectBCI_*_200928_2004.txt'; #mindaffectBCI_noisetag_bci_201002_1026.txt'
    triggerPlot(filename, evtlabs=('re','fe'), tau_ms=400, offset_ms=-50, stopband=(.5,45,'bandpass'), fs_out=250)
    ##triggerPlot(filename, evtlabs=('re','fe'), tau_ms=400, offset_ms=-50)

