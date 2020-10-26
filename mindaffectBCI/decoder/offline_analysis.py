import numpy as np
from mindaffectBCI.decoder.analyse_datasets import debug_test_dataset, analyse_dataset, analyse_datasets
from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
from mindaffectBCI.decoder.timestamp_check import timestampPlot
import matplotlib.pyplot as plt

savefile = '~/Downloads/mindaffectBCI_200701_1617[1].txt'

savefile = '~/Downloads/mindaffectBCI_200701_1624[1].txt'

savefile = '~/Downloads/mindaffectBCI_200701_2025_khash.txt'

savefile = '~/Desktop/mark/mindaffectBCI*ganglion*1411*.txt'
savefile = '~/Desktop/mark/mindaffectBCI*1239.txt'

import glob
import os
files = glob.glob(os.path.expanduser(savefile)); 
#os.path.join(os.path.dirname(os.path.abspath(__file__)),fileregexp)) # * means all if need specific format then *.csv
savefile = max(files, key=os.path.getctime)

# load
X, Y, coords = load_mindaffectBCI(savefile, stopband=((45,65),(5.5,25,'bandpass')), order=6, ftype='butter', fs_out=100)
# output is: X=eeg, Y=stimulus, coords=meta-info about dimensions of X and Y
print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,coords[1]['fs']))
print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:1]]+['output'],Y.shape))

plt.close('all')
# train *only* on 1st 10 trials
debug_test_dataset(X, Y, coords,
                    preprocess_args=dict(decorrelate=1e0),
                    cv=[(slice(10),slice(10,None))], tau_ms=450, evtlabs=('fe','re'), rank=1, model='cca', ranks=(1,2,3,5))


# do a time-stamp check.
timestampPlot(savefile)

# # check the electrode qualities computation
# ppfn= butterfilt_and_downsample(order=6, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=100)
# X=X.reshape((-1,X.shape[-1]))
# ppfn.fit(X[:10,:],fs=200)
# Xp =  ppfn.transform(X)
# plt.clf();plt.subplot(211);plt.plot(X);plt.subplot(212);plt.plot(Xp);

# sigq=testElectrodeQualities(X,fs=200)
# plt.clf();plt.plot(sigq)


