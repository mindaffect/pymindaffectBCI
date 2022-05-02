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

import glob
import os
import numpy as np
import sys
import mindaffectBCI.decoder.stim2event
from mindaffectBCI.decoder.analyse_datasets import debug_test_dataset, analyse_dataset, analyse_datasets
from mindaffectBCI.decoder.offline.load_mindaffectBCI  import load_mindaffectBCI
from mindaffectBCI.decoder.timestamp_check import timestampPlot
from mindaffectBCI.decoder.utils import block_permute
import matplotlib.pyplot as plt

if __name__=='__main__':

    # last file saved to default save location
    savefile = None
    #savefile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')
    #savefile = '~/Desktop/mark/mindaffectBCI_*.txt'
    #savefile = '~/Desktop/khash/mindaffectBCI*faces*.txt'
    #savefile = '~/Downloads/mindaffectBCI*.txt'

    if savefile is None:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        root = Tk()
        root.withdraw()
        savefile = askopenfilename(initialdir=os.path.dirname(os.path.abspath(__file__)),
                                    title='Chose mindaffectBCI save File',
                                    filetypes=(('mindaffectBCI','mindaffectBCI*.txt'),('All','*.*')))
        root.destroy()

    if savefile is None:
        savefile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')

    # get the most recent file matching the savefile expression
    files = glob.glob(os.path.expanduser(savefile)); 
    savefile = max(files, key=os.path.getctime)

    filterband=((45,65),(5,25,'bandpass'))
    evtlabs=('re','fe')
    tau_ms = 450
    offset_ms = 75
    prediction_offsets=(0)
    startup_correction=5
    priorweight=50
    ranks=(1,2,3,5,10)
    test_idx = slice(10,None)
    if 'rc' in savefile or 'audio' in savefile:
        evtlabs=('re','ntre')
        tau_ms = 500
        offset_ms = 100 # shift in window w.r.t. trigger.
        # final window is  [offset - offset+tau_ms]
        filterband = ((45,65),(5,25,'bandpass'))
        if 'audio' in savefile:
            test_idx=slice(40,None)
    elif 'threshold' in savefile:
        evtlabs='hot-on'
    elif 'acuity' in savefile:
        evtlabs='output2event'
    else:
        evtlabs=('re','fe')

    # load
    X, Y, coords = load_mindaffectBCI(savefile, filterband=filterband, order=6, ftype='butter', fs_out=100)
    # output is: X=eeg, Y=stimulus, coords=meta-info about dimensions of X and Y
    print("EEG: X({}){} @{}Hz".format([c['name'] for c in coords],X.shape,coords[1]['fs']))
    print("STIMULUS: Y({}){}".format([c['name'] for c in coords[:1]]+['output'],Y.shape))

    # for visual-acuity
    if 'visual_acuity' in savefile:
        X = X[10:,...] 
        Y = Y[10:,...,1:]
        evtlabs=('output2event')
    elif 'threshold' in savefile:
        evtlabs=mindaffectBCI.decoder.stim2event.hot_greaterthan
        
    elif 'rc'in savefile:
        evtlabs=('re','ntre')
    else:
        evtlabs=('re','fe')

    test_idx = None # slice(10,None)
    cv= False # True

    if 'central_cap' in savefile:
        coords[2]['coords'] = ['Cp5','Cp1','Cp2','Cp6','P3','P2','P4','POz']

    # train *only* on 1st 10 trials
    #score, dc, Fy, clsfr = analyse_dataset(X, Y, coords,
    #                        test_idx=slice(10,None), tau_ms=450, evtlabs=('fe','re'), rank=1, model='cca',
    #                        ranks=(1,2,3,5), prediction_offsets=(-1,0,1), priorweight=200, startup_correction=0, 
    #                        bwdAccumulate=True, minDecisLen=0)

    score, dc, Fy, clsfr, rawFy = debug_test_dataset(X, Y, coords,
                            test_idx=test_idx, tau_ms=tau_ms, offset_ms=offset_ms, evtlabs=evtlabs, model='cca', 
                            ranks=ranks, prediction_offsets=prediction_offsets, 
                            priorweight=priorweight, startup_correction=startup_correction, 
                            bwdAccumulate=False, minDecisLen=0)

    try:
        import pickle
        pickle.dump(dict(Fy=Fy,rawFy=Fy,X=X,Y=Y,coords=coords),open('stopping.pk','wb'))
    except:
        print("problem saving the scores..")

    quit()


    # score, dc, Fy, clsfr, rawFy = debug_test_dataset(X, Y, coords,
    #                          test_idx=test_idx, cv=cv, tau_ms=450, evtlabs=evtlabs, model='cca', 
    #                          ranks=(1,2,3,5,10), prediction_offsets=(0), priorweight=200, startup_correction=50, 
    #                          bwdAccumulate=False, minDecisLen=0, reg=(1e-8,1e-2))

    ## Manually test different thresholds
    # thresholds = np.unique(Y.ravel())
    # thresholds = [">{}".format(t) for t in thresholds[:-1]] # strip last one
    # dcs=[]
    # gofs=[]
    # for ti,thresh in enumerate(thresholds):
    #     evtlabs = thresh
    #     print("\n\n---------------\n evtlabs={}\n".format(evtlabs))
    #     res = analyse_dataset(X, Y, coords, model='cca', cv=True, tau_ms=450, rank=3, n_virt_out=-30,
    #                     evtlabs=evtlabs)
    #     clsfr_res = res[4]
    #     gofs.append(np.mean(clsfr_res['test_gof']))
    #     print(" Goodness-of-fit : {}".format(gofs[-1]))
    #     dcs.append(res[1])

    # from mindaffectBCI.decoder.decodingCurveSupervised import print_decoding_curve, plot_decoding_curve, flatten_decoding_curves
    # int_len, prob_err, prob_err_est, se, st = flatten_decoding_curves(dcs)
    # print("Ave-DC\n{}\n".format(print_decoding_curve(np.nanmean(int_len,0),np.nanmean(prob_err,0),np.nanmean(prob_err_est,0),np.nanmean(se,0),np.nanmean(st,0))))
    # plot_decoding_curve(int_len,prob_err)
    # plt.legend(["{} gof={:5.3f}".format(t,f) for (t,f) in zip(thresholds,gofs)]+["mean"])

    # plt.show()



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
    # ppfn= butterfilt_and_downsample(order=6, filterband='butter_filterband((0, 5), (25, -1))_fs200.pk', fs_out=100)
    # X=X.reshape((-1,X.shape[-1]))
    # ppfn.fit(X[:10,:],fs=200)
    # Xp =  ppfn.transform(X)
    # plt.clf();plt.subplot(211);plt.plot(X);plt.subplot(212);plt.plot(Xp);

    # sigq=testElectrodeQualities(X,fs=200)
    # plt.clf();plt.plot(sigq)
    plt.show()


