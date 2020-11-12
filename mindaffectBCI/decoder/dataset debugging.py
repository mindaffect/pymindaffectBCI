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
import matplotlib.pyplot as plt
from mindaffectBCI.decoder.datasets import get_dataset
from mindaffectBCI.decoder.model_fitting import BaseSequence2Sequence, MultiCCA, FwdLinearRegression, BwdLinearRegression, LinearSklearn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import cross_val_score
from mindaffectBCI.decoder.updateSummaryStatistics import updateSummaryStatistics, plot_erp, plot_summary_statistics
from mindaffectBCI.decoder.scoreStimulus import factored2full, plot_Fe
from mindaffectBCI.decoder.decodingCurveSupervised import decodingCurveSupervised, print_decoding_curve, plot_decoding_curve
from mindaffectBCI.decoder.utils import lab2ind, butter_sosfilt, extract_envelope
from mindaffectBCI.decoder.scoreOutput import plot_Fy
from mindaffectBCI.decoder.multipleCCA import robust_whitener
from mindaffectBCI.decoder.updateSummaryStatistics import updateCxx, idOutliers
from scipy.signal import welch

l,f,_=get_dataset('openBMI_MI')

# get raw MI data, just for the trial duration
#X,Y,coords=l(f[1],stopband=((0,3),(29,-1)),CAR=False,offset_ms=None,ppMI=False)
offset_ms=(0,0);(-6000,12000)
stopband=None #((0,3),(30,-1))
X,Y,coords=l(f[0],stopband=stopband,CAR=True,offset_ms=offset_ms,ppMI=False,fs_out=100)
oX=X.copy()
fs=coords[1]['fs']
lab=coords[0]['lab']

def plot_erp(X,lab,title,fs=None,chnames=None,plotp=True,differp=False):
    lab,key=lab2ind(lab)
    plt.clf();
    plt.subplot(211);
    plt.imshow(X[0,:,:].copy(),aspect='auto');
    erp=np.einsum("Tsd,Ty->ysd",X,lab);
    if differp:
        erp = erp - np.mean(erp,0,keepdims=True) # center erps, so show diffs
    for yi in range(erp.shape[0]):
        plt.subplot(2,erp.shape[0],erp.shape[0]+yi+1);
        if plotp:
            plt.plot(erp[yi,:,:]);
            plt.grid(True)
        else:
            plt.imshow(erp[yi,:,:],aspect='auto');
        plt.xlabel('space (ch*freq)')
        plt.title(key[yi])
    plt.suptitle(title)

X=oX.copy()
plt.figure(100);plot_erp(X,lab,'raw')

# CAR
X = X - np.mean(X,-1,keepdims=True)

# outlier removal
badtr,pow = idOutliers(X,axis=(1,2))
print('Removing {} bad trials'.format(np.sum(badtr.ravel())))
keep = badtr.ravel()==False
X = X[keep,...]
lab=lab[keep]

badch,_ = idOutliers(X,axis=(0,1))
print('Removing {} bad channels'.format(np.sum(badch.ravel())))
keep = badch.ravel()==False
X = X[..., keep]
ch_names=[ch_names[i] for i in range(len(ch_names)) if keep[i]]

# ch-subset
gigasubset=('C5','C3','C1','Cz','C2','C4','C6',\
            'FC5','FC3','FC1','FCz','FC2','FC4','FC6',\
            'CP5','CP3','CP1','CPz','CP2','CP4','CP6')
minsubset =('C3','C4')
chsubset=minsubset
keep=[c in chsubset for c in ch_names]
X = X[...,keep]
ch_names = [ ch_names[i] for i in range(len(ch_names)) if keep[i] ]
plt.figure(100);plot_erp(X,lab,'car')

# hp-lp
X,_,_=butter_sosfilt(X,stopband=((0,8),(16,-1)),fs=fs)
plt.figure(101);plot_erp(X,lab,'hp-lp',plotp=True)

# whiten
Cxx=updateCxx(None,X,None)
W,_=robust_whitener(Cxx)
X  =np.einsum("Tsd,dw->Tsw",X,W)
#Cxxw=updateCxx(None,X,None)

plt.figure(102);plot_erp(X,lab,'wht')

# welch
freqs, X = welch(X,fs,axis=-2,nperseg=int(fs*.5),noverlap=.5,return_onesided=True,scaling='spectrum')
plt.figure(103);plot_erp(X,lab,'welch')

# envelope
X = extract_envelope(X,fs,stopband=None,whiten=None,filterbank=None,log=False,env_stopband=(2,-1),plot=False)
plt.figure(103);plot_erp(X,lab,'env',plotp=True)

X=np.log(np.maximum(X,1e-5))

# baseline
bl_ms=(-1000,0)
bl_idx = slice(int((bl_ms[0]-offset_ms[0])*fs/1000),int((bl_ms[1]-offset_ms[0])*fs/1000))
X = X - np.mean(X[:,bl_idx,:],axis=1,keepdims=True)
plt.figure(103);plot_erp(X,lab,'env',plotp=True)

# classify
clsfr=LogisticRegression(C=1e8)
cross_val_score(clsfr,X.reshape((X.shape[0],-1)),lab,cv=5)
