import numpy as np
from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
from mindaffectBCI.utopiaclient import  DataPacket
from mindaffectBCI.decoder.model_fitting import MultiCCA
from mindaffectBCI.decoder.utils import window_axis
import matplotlib.pyplot as plt
import glob

def triggerPlot(filename=None, evtlabs=('0','1'), tau_ms=300):
    import glob
    import os
    if filename is None or filename == '-':
        # default to last log file if not given
        files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
        filename = max(files, key=os.path.getctime)
    else:
        files = glob.glob(filename)
        filename = max(files, key=os.path.getctime)
    print("Loading : {}\n".format(filename))    

    #X, Y, coords = load_mindaffectBCI(filename, stopband=(5.5,45,'bandpass'), ofs=9999)
    X, Y, coords = load_mindaffectBCI(filename, stopband=None, ofs=9999)
    X[...,:-1] = X[...,:-1] - np.mean(X[...,:-1],axis=-2,keepdims=True) # offset remove
    fs = coords[-2]['fs']
    print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,coords[1]['fs']))
    print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:-1]]+['output'],Y.shape))

    print("training model")
    tau = int(fs*tau_ms/1000.0)
    times = np.arange(tau)*1000/fs
    # BODGE: for speed only use first 10 trials!
    clsfr = MultiCCA(evtlabs=evtlabs,tau=tau,rank=1).fit(X[:5,...],Y[:5,...])

    print("applying spatial filter")
    W = clsfr.W_[0,0,...] # (d,)
    #plt.plot(W);plt.title('W');plt.show()
    Xe = window_axis(X, winsz=tau, axis=-2) # (nTrl, nSamp-tau, tau, d)
    wXe = np.einsum("d,TEtd->TEt", W, Xe) # (nTrl, nSamp-tau, tau) apply the spatial filter

    # slice out the responses for the trigger stimulus
    print('slicing data')
    Y_true = Y[...,:Xe.shape[1],0] # (nTrl, nSamp-tau)
    wXeY = wXe[Y_true>0,:] # [ep,tau]

    # get a samp#, timestamp dataset
    trl_ts = coords[0]['trl_ts'][:,:Xe.shape[1]]   # (nTrl, nSamp-tau)
    trl_idx = coords[0]['trl_idx'][:,:Xe.shape[1]] # (nTrl, nSamp-tau)
    samp2ms = np.median((np.diff(trl_ts,axis=-1)/np.maximum(1,np.diff(trl_idx,axis=-1))).ravel())
    samp2ms = 1000/fs #samp2ms*.99
    print("samp2ms={}".format(samp2ms))
    lin_trl_ts = trl_idx*samp2ms # linear time-stamp assuming constant sample rate
    ts_err = trl_ts - lin_trl_ts # (tr, nSamp-tau) error btw. linear time-stamp and observed
    ts_err = ts_err - ts_err[0,0]
    ts_errY = ts_err[Y_true>0] # (ep,) slice out the high-stimulus info (in seconds)

    print('generating plot')
    mu = np.median(wXeY.ravel())
    scale = np.median( np.abs(wXeY.ravel()-mu) )
    ylim_wx = (mu-1.5*scale, mu+1.5*scale)
    fig, ax = plt.subplots()
    plt.imshow(wXeY.T,origin='normal',aspect='auto',extent=[0,wXeY.shape[0],0,times[-1]])
    plt.clim(mu-1.5*scale,mu+1.5*scale)
    plt.set_cmap('jet')
    plt.ylabel('time (ms)')
    plt.xlabel('Epoch')
    plt.title('{}'.format(filename[-50:]))
    #plt.colorbar()

    if X.shape[-1]>8 :
        samp_cnt = X[...,-1] # sample count (%255)
        samp_miss = samp_cnt[...,1:] - ((samp_cnt[...,:-1]+1) % 256)
        samp_miss = samp_miss[...,:Xe.shape[1]]
        samp_miss = samp_miss[Y_true>0]
        plt.plot(samp_miss*samp2ms + 100,'k-')

    # over-plot the error
    ax2=ax.twinx()
    plt.plot(ts_errY)
    mu2 = np.median(ts_errY.ravel())
    scale2 = np.median( np.abs(ts_errY.ravel()-mu) )

    plt.ylim(mu2-2.5*scale2,mu2+2.5*scale2)
    if abs(times[-1]-5*scale2) < 2*times[-1]: # share y-scale if close enough
        ax.set_ylim(0,max(times[-1],5*scale2))
    plt.ylabel("Recieved time-stamp error vs. constant rate (ms)")
    plt.show()

if __name__=="__main__":
    triggerPlot()    

