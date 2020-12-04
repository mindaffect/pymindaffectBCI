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

import glob
import os
import numpy as np
from mindaffectBCI.decoder.analyse_datasets import debug_test_dataset, analyse_dataset, analyse_datasets
from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
from mindaffectBCI.decoder.timestamp_check import timestampPlot
import matplotlib.pyplot as plt

# last file saved to default save location
savefile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')

#savefile = '~/Desktop/mark/mindaffectBCI*1531_linux.txt'
#savefile = '~/Desktop/khash/mindaffectBCI*1531_linux.txt'
savefile = '~/Desktop/rpi/mindaffectBCI*.txt'
#savefile = '~/Desktop/mark/mindaffectBCI_brainflow_android_200916_1148.txt' # p-val bug
#savefile = '~/Desktop/mark/mindaffectBCI_noisetag_bci_*1319_ganglion.txt' # score bug

savefile = '~/Downloads/mindaffectBCI*.txt'

# get the most recent file matching the savefile expression
files = glob.glob(os.path.expanduser(savefile)); 
savefile = max(files, key=os.path.getctime)

# load
X, Y, coords = load_mindaffectBCI(savefile, stopband=((45,65),(5.5,25,'bandpass')), order=6, ftype='butter', fs_out=100)
# output is: X=eeg, Y=stimulus, coords=meta-info about dimensions of X and Y
print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,coords[1]['fs']))
print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:1]]+['output'],Y.shape))

# train *only* on 1st 10 trials
score, dc, Fy, clsfr, cvres = debug_test_dataset(X, Y, coords,
                        test_idx=slice(10,None), tau_ms=450, evtlabs=('fe','re'), rank=1, model='cca', ranks=(1,2,3,5), prediction_offsets=(0))

#score, dc, Fy, clsfr, cvres = analyse_dataset(X, Y, coords,
#                        test_idx=slice(10,None), tau_ms=450, evtlabs=('fe','re'), rank=1, model='cca', ranks=(1,2,3,5))


# test the auto-offset compensation
from mindaffectBCI.decoder.scoreOutput import scoreOutput,  plot_Fy
Fe = clsfr.transform(X)
Ye = clsfr.stim2event(Y)
    
# score all trials with shifts
offsets=[-2,-1,0,1,2] # set offsets to test
prior=np.array([.3,.7,1,.5,.2]) # prior over offsets
Fyo = scoreOutput(Fe,Ye, offset=offsets, dedup0=True)
print("{}".format(Fyo.shape))
for i,o in enumerate(offsets):
    plt.figure()
    plot_Fy(Fyo[i,...],maxplots=50,label="{}\noffset {}".format(savefile,o))
    plt.show(block=False)

from mindaffectBCI.decoder.zscore2Ptgt_softmax import zscore2Ptgt_softmax
from mindaffectBCI.decoder.normalizeOutputScores import normalizeOutputScores
# try auto-model-id in the Pval computation:
ssFyo,scale_sFy,N,_,_=normalizeOutputScores(Fyo.copy(),minDecisLen=-1,nEpochCorrection=100, priorsigma=(clsfr.sigma0_,clsfr.priorweight))
plot_Fy(np.squeeze(ssFyo[:,0,...]),cumsum=False, label="{} Trl={}".format(savefile,0))
plt.show()
Ptgt=zscore2Ptgt_softmax(ssFyo,clsfr.softmaxscale_,prior=prior.reshape((-1,1,1,1)),marginalizemodels=True, marginalizedecis=False) # (nTrl,nEp,nY)
plot_Fy(Ptgt, cumsum=False,maxplots=50,label=savefile)

# do a time-stamp check.
plt.clf()
timestampPlot(savefile)

# # check the electrode qualities computation
# ppfn= butterfilt_and_downsample(order=6, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=100)
# X=X.reshape((-1,X.shape[-1]))
# ppfn.fit(X[:10,:],fs=200)
# Xp =  ppfn.transform(X)
# plt.clf();plt.subplot(211);plt.plot(X);plt.subplot(212);plt.plot(Xp);

# sigq=testElectrodeQualities(X,fs=200)
# plt.clf();plt.plot(sigq)
plt.show()


