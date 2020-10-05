import numpy as np
from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_messages
from mindaffectBCI.utopiaclient import StimulusEvent, DataPacket, ModeChange, NewTarget, Selection
import matplotlib.pyplot as plt
from mindaffectBCI.decoder.lower_bound_tracker import lower_bound_tracker

def timestampPlot(filename=None):
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

    print('savefile={}'.format(filename))
    msgs = read_mindaffectBCI_messages(filename,regress=None) # load without time-stamp fixing.
    print("{} msgs".format(len(msgs)))
    dp = [ m for m in msgs if m.msgID == DataPacket.msgID]
    print("{} dp".format(len(dp)))
    samp = np.array([ m.samples.shape[0] for m in dp])
    svr = np.array([ m.sts for m in dp])
    client = np.array([ m.timestamp for m in dp])
    samp = np.cumsum(samp)

    lbt = lower_bound_tracker()
    svr_filt = np.zeros(svr.shape)
    for i in range(samp.shape[0]):
        svr_filt[i] = lbt.transform(samp[i],svr[i])

    samp2ms = np.median(np.diff(svr)/np.diff(samp))
    print("samp2ms {}".format(samp2ms))
    print("{}".format(samp.shape))
    print("Sample : {}".format(samp[:10]))
    print("Server : {}".format(svr[:10]))
    print("Client : {}".format(client[:10]))
    print("svr_flt: {}".format(svr_filt[:10]))
    print("{} samp = {}s".format(samp[-1],samp[-1]*samp2ms/1000))
    svr_err = svr - samp*samp2ms- svr[0]
    client_err = client - samp*samp2ms- client[0]
    svr_filt_err = svr_filt - samp*samp2ms- svr_filt[0]
    cent = np.median(svr_err); scale=np.median(np.abs(svr_err-cent))
    plt.plot(samp*samp2ms,svr_err,label='samp*samp2ms - server')
    plt.plot(samp*samp2ms,client_err,label='samp*samp2ms - client')
    plt.plot(samp*samp2ms,svr_filt_err,label='samp*samp2ms - filt(server)')
    plt.ylim((cent-scale*samp2ms,cent+scale*samp2ms))
    plt.xlabel('Time (ms)')
    plt.ylabel('time stamp error (ms)')
    plt.legend()
    plt.grid()
    plt.title('{}\nserver/client time-stamps w.r.t. fixed sample clock (@{}ms/samp)'.format(filename[-40:],samp2ms))
    plt.show()

if __name__=="__main__":
    #filename = "~/Desktop/trig_check/mindaffectBCI_*brainflow*.txt"
    filename = '~/Desktop/mark/mindaffectBCI*ganglion*1411*.txt'
    #filename=None
    #filename='~/Desktop/pymindaffectBCI/logs/mindaffectBCI_*_200928_2004.txt'; #mindaffectBCI_noisetag_bci_201002_1026.txt'
    timestampPlot(filename)

