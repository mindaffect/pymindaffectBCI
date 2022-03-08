import numpy as np
import matplotlib.pyplot as plt
import mne

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.decoding import *
from mindaffectBCI.examples.analysis.mne_utils import *
from mindaffectBCI.decoder.alternatingLR import *
import datetime
import json


UNAME = datetime.datetime.now().strftime("%y%m%d_%H%M")

def load_and_epoch_subject(tmin=-1, tmax=4, lfreq=7, hfreq=30, event_id=dict(hands=2,feet=3),runs=[6,10,14], subject=1):
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(lfreq, hfreq, fir_design='firwin', skip_by_annotation='edge')

    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                    exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    return epochs

def epochs2dataset(epochs,event_id):
    epochs_data = epochs.get_data()
    X = epochs.get_data()
    assert len(event_id)==2
    events = epochs.events

    # re-label to start with 0
    Y = np.zeros(events.shape[0],dtype=int)
    for i,(k,v) in enumerate(event_id.items()):
        Y[events[:, -1]==v]=i

    fs = epochs.info['sfreq']
    ch_names = epochs.info['ch_names']

    return X, Y, fs, ch_names


def run_pipeline(pipename,pipeline,X,Y,cv=10,fs=160,ch_names=None, random_state=42, dsname=None):
    # generate a deterministic data split
    cv = ShuffleSplit(cv, test_size=0.2, random_state=random_state)
    cv_split = cv.split(X)

    scores = cross_val_score(pipeline,X,Y,cv=cv_split)
    if dsname is not None:
        print("{:30})  {:30s} = {:5.3f}".format(dsname,pipename,np.mean(scores)))
    else:
        print(" {:30} = {:5.3f}".format(pipename,np.mean(scores)))
    return scores

def run_analysis(pipelines, label:str='', cv=10, tmin=1, tmax=2, lfreq=7, hfreq=30, subjects=None,
                 event_id=dict(hands=2,feet=3),runs=[6,10,14]):
    if subjects is None: subjects = np.arange(1,110)
    scores=dict()
    for subject in subjects:
        # load
        epochs = load_and_epoch_subject(tmin=tmin,tmax=tmax,lfreq=lfreq,hfreq=hfreq,event_id=event_id,runs=runs,subject=subject)

        # get dataset and split
        X, Y, fs, ch_names= epochs2dataset(epochs, event_id)
        # run the pipelines
        for pipename,pipeline in pipelines:
            pscores = run_pipeline(pipename,pipeline,X.copy(),Y.copy(),cv,fs,ch_names,dsname=subject)
            if not pipename in scores:
                scores[pipename]=[]
            scores[pipename].append(pscores.tolist())

        # emergency save results info
        with open('mne_eegbci_raw_{}.json'.format("_".join((label,UNAME))),'w') as f:
            json.dump(scores,f,indent=2)

        # summarize
        score_ave = dict()
        for k,v in scores.items():
            ave = [ np.mean(ps) for ps in v ]
            print("{:30s} : {:5.3f} (N={:d})".format(k,np.mean(ave),len(ave)))
            score_ave[k]=(np.mean(ave),len(ave),ave)
    
        with open('mne_eegbci_summary_{}.json'.format("_".join((label,UNAME))),'w') as f:
            json.dump(score_ave,f,indent=2)

    return scores

# BODGE: set the fs..
fs=160

csp = Pipeline([
                ('CSP', CSP(n_components=4, reg=None, log=True, norm_trace=False)), 
                ('clf', LogisticRegressionCV())
                ])

stdamp = Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # -> TSd 
                ('std',UnitAmp(axis=2)), # TSd, unit ave amp per channel
                ('amp',AverageAmplitude(axis=1,center=True,log=True)), # TSd
                ('vec', Vectorizer()),  # Tf
                ('clf', LogisticRegressionCV())
                ])


whtamp = Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd 
                ('amp',AverageAmplitude(axis=1,center=True,log=True)), # TSd
                ('vec', Vectorizer()),  # Tf
                ('clf', LogisticRegressionCV())
                ])


whtfbamp = Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # -> TSd 
                ('wht',SpatialWhitener(reg=.05)), 
                ('fb',FFTfilter(axis=1,filterbank=((2,4,8,10,'bandpass'),(6,8,12,14,'bandpass'),(10,12,16,18,'bandpass'),(14,16,20,22,'bandpass'),(19,20,24,26,'bandpass'),(22,24,28,30,'bandpass')),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TfSd
                ('amp',AverageAmplitude(axis=2,center=False,log=True)), # Tfd
                ('vec', Vectorizer()), 
                ('clf', LogisticRegressionCV())
                ])

whtfb2hzamp = Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # -> TSd 
                ('wht',SpatialWhitener(reg=.05)), 
                ('fb',FFTfilter(axis=1,filterbank=tuple((f-2,f,f,f+2,'bandpass') for f in range(4,26,2)),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TfSd
                ('amp',AverageAmplitude(axis=2,center=False,log=True)), # Tfd
                ('vec', Vectorizer()), 
                ('clf', LogisticRegressionCV())
                ])


whtcov =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('cov',CovMatrixizer()), # Tbdd
                ('vec', Vectorizer()), 
                ('clf', LogisticRegressionCV())
                ])

whtcovRR =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('cov',CovMatrixizer()), # Tbdd
                ('vec', Vectorizer()), 
                ('clf', RidgeClassifierCV())
                ])

whtfbcov =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('fb',FFTfilter(axis=1,filterbank=((2,4,8,10,'bandpass'),(6,8,12,14,'bandpass'),(10,12,16,18,'bandpass'),(14,16,20,22,'bandpass'),(19,20,24,26,'bandpass'),(22,24,28,30,'bandpass')),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TbSd
                ('cov',CovMatrixizer()), # Tbdd
                ('vec', Vectorizer()), 
                ('clf', LogisticRegressionCV())
                ])

whtfbALR =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('fb',FFTfilter(axis=1,filterbank=((2,4,8,10,'bandpass'),(6,8,12,14,'bandpass'),(10,12,16,18,'bandpass'),(14,16,20,22,'bandpass'),(19,20,24,26,'bandpass'),(22,24,28,30,'bandpass')),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TbSd
                ('amp',AverageAmplitude(axis=2,center=False,log=True)), # Tfd
                ('clf', AlternatingLR(axes=[(1,),(2,)]))
                ])

# more freq for this one
whtfb2hzALR =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('fb',FFTfilter(axis=1,filterbank=tuple((f-2,f,f,f+2,'bandpass') for f in range(4,26,2)),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TbSd
                ('amp',AverageAmplitude(axis=2,center=False,log=True)), # Tfd
                ('clf', AlternatingLR(axes=[(1,),(2,)]))
                ])

whtfb2hzALRCV =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('fb',FFTfilter(axis=1,filterbank=tuple((f-2,f,f,f+2,'bandpass') for f in range(4,26,2)),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TbSd
                ('amp',AverageAmplitude(axis=2,center=False,log=True)), # Tfd
                ('clf', AlternatingLRCV(axes=[(1,),(2,)]))
                ])

whtfb2hzARRCV =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('fb',FFTfilter(axis=1,filterbank=tuple((f-2,f,f,f+2,'bandpass') for f in range(4,26,2)),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TbSd
                ('amp',AverageAmplitude(axis=2,center=False,log=True)), # Tfd
                ('clf', AlternatingRidgeCV(axes=[(1,),(2,)]))
                ])


whtfb1hzALR =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('fb',FFTfilter(axis=1,filterbank=tuple((f-2,f,f,f+2,'bandpass') for f in range(4,26,1)),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TbSd
                ('amp',AverageAmplitude(axis=2,center=False,log=True)), # Tfd
                ('clf', AlternatingLR(axes=[(1,),(2,)]))
                ])

whtfb1hzALRCV =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('fb',FFTfilter(axis=1,filterbank=tuple((f-2,f,f,f+2,'bandpass') for f in range(4,26,1)),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TbSd
                ('amp',AverageAmplitude(axis=2,center=False,log=True)), # Tfd
                ('clf', AlternatingLRCV(axes=[(1,),(2,)]))
                ])

whtfbALRCV =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('fb',FFTfilter(axis=1,filterbank=((2,4,8,10,'bandpass'),(6,8,12,14,'bandpass'),(10,12,16,18,'bandpass'),(14,16,20,22,'bandpass'),(19,20,24,26,'bandpass'),(22,24,28,30,'bandpass')),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TbSd
                ('amp',AverageAmplitude(axis=2,center=False,log=True)), # Tfd
                ('clf', AlternatingLRCV(axes=[(1,),(2,)]))
                ])

whtcovALR =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('cov',CovMatrixizer()), # Tdd
                ('vec', Vectorizer()), 
                ('clf', AlternatingLR(axes=[(1,),(2,)]))
                ])

whtcovARRCV =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('cov',CovMatrixizer()), # Tdd
                ('vec', Vectorizer()), 
                ('clf', AlternatingRidgeCV(axes=[(1,),(2,)]))
                ])

whtfbcovALR =  Pipeline([
                ('t2s',MoveAxis((1,2),(2,1))), # TdS -> TSd 
                ('wht',SpatialWhitener(reg=.05)), # TSd
                ('fb',FFTfilter(axis=1,filterbank=((2,4,8,10,'bandpass'),(6,8,12,14,'bandpass'),(10,12,16,18,'bandpass'),(14,16,20,22,'bandpass'),(19,20,24,26,'bandpass'),(22,24,28,30,'bandpass')),fs=fs,ola_filter=True,prefix_feature_dim=True)), # TbSd
                ('cov',CovMatrixizer()), # Tbdd
                ('clf', AlternatingLR(axes=[(1,),(2,3)]))
                ])


if __name__=='__main__':
    pipelines = [ 
        #('csp',csp),
        #('stdamp',stdamp),
        #('whtamp',whtamp),
        #('whtfbamp',whtfbamp),
        #('whtfb2hzamp',whtfb2hzamp),
        ('whtfbALR',whtfbALR),
        #('whtfbALRCV',whtfbALRCV),
        ('whtfb2hzALR',whtfb2hzALR),
        #('whtfb2hzALRCV',whtfb2hzALRCV),
        #('whtfb2hzARRCV',whtfb2hzARRCV),
        #('whtfb1hzALR',whtfb1hzALR),
        #('whtfb1hzALRCV',whtfb1hzALRCV),
        #('whtcov',whtcov),
        ('whtcovRR',whtcov),
        #('whtfbcov',whtfbcov),
        #('whtcovALR',whtcovALR),
        #('whtcovARRCV',whtcovARRCV),
        #('whtfbcovALR',whtfbcovALR)
    ]
    label='{}'.format("_".join([p[0] for p in pipelines]))
    run_analysis(pipelines,label=label)