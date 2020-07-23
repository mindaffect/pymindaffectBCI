import numpy as np
from analyse_datasets import debug_test_dataset, analyse_dataset, analyse_datasets
from offline.load_mindaffectBCI  import load_mindaffectBCI
import matplotlib.pyplot as plt

savefile = '~/Downloads/mindaffectBCI_200701_1617[1].txt'

savefile = '~/Downloads/mindaffectBCI_200701_1624[1].txt'

savefile = '~/Downloads/mindaffectBCI_200701_2025_khash.txt'

# get raw
X, Y, coords = load_mindaffectBCI(savefile, stopband=None, ofs=200)

# get normally pre-processed
X, Y, coords = load_mindaffectBCI(savefile, stopband=((0,5),(25,-1)), order=6, ofs=200)
# output is: X=eeg, Y=stimulus, coords=meta-info about dimensions of X and Y
print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,coords[1]['fs']))
print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:1]]+['output'],Y.shape))

plt.close('all')
debug_test_dataset(X, Y, coords, tau_ms=350, evtlabs=('re','fe'), rank=1, model='cca')


# check the electrode qualities computation
ppfn= butterfilt_and_downsample(order=6, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=100)
X=X.reshape((-1,X.shape[-1]))
ppfn.fit(X[:10,:],fs=200)
Xp =  ppfn.transform(X)
plt.clf();plt.subplot(211);plt.plot(X);plt.subplot(212);plt.plot(Xp);

sigq=testElectrodeQualities(X,fs=200)
plt.clf();plt.plot(sigq)


