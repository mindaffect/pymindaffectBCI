import os
from glob import glob
import re
from mindaffectBCI.decoder.offline.load_mark_EMG import load_mark_EMG
from mindaffectBCI.decoder.offline.load_brainstream import load_brainstream
from mindaffectBCI.decoder.offline.load_p300_prn import load_p300_prn
from mindaffectBCI.decoder.offline.load_openBMI import load_openBMI
from mindaffectBCI.decoder.offline.load_cocktail import load_cocktail
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

from mindaffectBCI.decoder.offline.load_mTRF_audio import load_mTRF_audio
from mindaffectBCI.decoder.offline.load_twofinger import load_twofinger
from mindaffectBCI.decoder.offline.load_brainsonfire import load_brainsonfire
from mindaffectBCI.decoder.offline.load_ninapro_db2 import load_ninapro_db2
from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
from mindaffectBCI.decoder.utils import testSignal

# List of root directories to search for the experiment sub-directory
dataroots = ['~/data/bci',
            'G://Shared drives/Data/experiments',
            'G://Shared drives/Data',
            '/content/drive/Shareddrives/Data',
            '/content/drive/Shareddrives',
            '/home/shared/drive/',
            'D://',
            '.'
            ]

def add_dataroot(dataroot):
    """add a new data root directory to search for dataset sub-directories

    Args:
        dataroot (str): directory to add to the dataroots set
    """    
    global dataroots
    dataroots.append(dataroot)

def set_dataroot(dataroot):
    """set the list of rood directories to search for dataset sub-directories

    Args:
        dataroots (list-of-str): list of directory names to search for data root directories
    """    
    global dataroots
    dataroots = [dataroot] if isinstance(dataroot,str) else dataroot

def get_dataroot(dataroots=None,subdir=None):
    """search through the list of data-roots to find a sub-directory which contains the given experiment sub-directory

    Args:
        dataroots (list-of-str, optional): list of data root directories.  If None then use the system default list. Defaults to None.
        subdir (str, optional): the experiment specific sub-directory to search for.  If None then just the first data root which exists on this machine. Defaults to None.

    Returns:
        str: a data root directory which exists on this machine and contains the desired sub-directory
    """    
    dataroot=None
    if dataroots is None:
        dataroots = globals().get('dataroots')
    if dataroots is not None:
        # check whith dataroots are available
        if subdir is not None:
            for dr in dataroots:
                if os.path.exists(os.path.join(os.path.expanduser(dr),subdir)):
                    dataroot = os.path.expanduser(dr)
                    break
        # if got here, either no subdir or didn't find subdir
        if dataroot is None:
            for dr in dataroots:
                if os.path.exists(os.path.expanduser(dr)):
                    dataroot = os.path.expanduser(dr)
                    break
    return dataroot

def load_plos_one(datadir, ch_names=None, fs_out=None, **kwargs):
    """dataset specific loader for the plos-one dataset

    Args:
        datadir (_type_): _description_
        ch_names (_type_, optional): _description_. Defaults to None.
        fs_out (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    if ch_names is None:
        ch_names = ['Fp1',
            'AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5',
            'T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7',
            'P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','AFz',
            'Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz',
            'Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
    if fs_out is None: fs_out=180
    return load_brainstream(datadir, ch_names=ch_names, fs_out=fs_out, **kwargs)

def plos_one():
    '''generate the directory+filename info for the plos_one noisetagging dataset'''
    loader = load_plos_one # function to call to load the dataset
    expt = 'external_data/plos_one/'
    datadir = get_dataroot(subdir=expt)
    datadir = os.path.join(os.path.expanduser(datadir),expt)
    sessdir = glob(os.path.join(datadir, 's[0-9]*'))
    #sessdir = ['s{:d}'.format(i) for i in range(1, 12)]
    sessfn = 'traindata.mat'
    filenames = [os.path.join(datadir, d, sessfn) for d in sessdir]
    return (loader, filenames, datadir)

def lowlands():
    '''generate the directory+filename info for the lowlands noisetagging dataset'''
    expt = 'external_data/lowlands'
    loader = load_brainstream
    datadir = get_dataroot(subdir=expt)
    datadir = os.path.join(datadir,expt)
    filenames = glob(os.path.join(datadir, '*_tr_train_1.mat'))
    return (loader, filenames, datadir)

def p300_prn(label:str=None):
    '''generate dataset+filename for p300-prn2'''
    expt = 'own_experiments/visual/p300_prn_2'
    loader = load_p300_prn # function to call to load the dataset
    datadir = get_dataroot(subdir=expt)
    filenames = glob(os.path.join(datadir, expt, '*/*/jf_prep/*flash.mat')) + \
                glob(os.path.join(datadir, expt, '*/*/jf_prep/*flip.mat'))
    if label is not None:
        filenames = [f for f in filenames if re.search(label,f)]
    return (loader, filenames, datadir)

def tactileP3():
    '''generate dataset+filename for tactile P3'''
    loader = load_p300_prn # function to call to load the dataset
    expt = 'own_experiments/tactile/selective_parallel_attention/P3speller/Speller'
    datadir = get_dataroot(subdir=expt)
    filenames = glob(os.path.join(datadir, expt, '*/*/jf_prep/*offline.mat'))
    return (loader, filenames, datadir)

def tactile_PatientStudy():
    '''generate dataset+filename for p300-prn2'''
    loader = load_p300_prn # function to call to load the dataset
    datadir = get_dataroot()
    expt = 'own_experiments/tactile/PatientStudy'
    filenames = glob(os.path.join(datadir, expt, '*/*/jf_prep/*offline.mat'))
    return (loader, filenames, datadir)

def openBMI(dstype="SSVEP"):
    loader = load_openBMI
    expt = 'external_data/gigadb/openBMI'
    datadir = get_dataroot(subdir=expt)
    datadir = os.path.join(datadir,expt)
    filenames = glob(os.path.join(datadir, 'sess*/s*/sml_*'+ dstype + '.mat')) + \
                glob(os.path.join(datadir, 'sml_*'+ dstype + '.mat'))
    return (loader, filenames, datadir)

def twofinger():
    loader = load_twofinger
    exptdir = 'external_data/twente/twofinger'
    datadir = get_dataroot(subdir=exptdir)
    filenames =  glob(os.path.join(datadir, exptdir, 'S??.mat'))
    return (loader, filenames, datadir)

def brains_on_fire_online():
    loader = load_brainsonfire
    exptdir = 'own_experiments/motor_imagery/brainsonfire/brains_on_fire_online'
    datadir = get_dataroot(subdir=exptdir)
    filenames =  glob(os.path.join(datadir, exptdir, 'subject*/raw_buffer/0001'))
    return (loader, filenames, datadir)

def brains_on_fire():
    loader = load_brainsonfire
    exptdir = 'own_experiments/motor_imagery/brainsonfire/brains_on_fire'
    datadir = get_dataroot(subdir=exptdir)
    filenames =  glob(os.path.join(datadir, exptdir, 'Subject*/raw_buffer/0001'))
    return (loader, filenames, datadir)

def mTRF_audio():
    loader = load_mTRF_audio
    datadir = get_dataroot()
    filenames = [os.path.join(datadir, 'speech_data.mat')]
    return (loader, filenames, datadir)

def ninapro_db2():
    loader = load_ninapro_db2
    exptdir="external_data/ninapro"
    datadir = get_dataroot(subdir=exptdir)
    filenames = glob(os.path.join(datadir, exptdir, 's*', '*E1*.mat'))
    return (loader, filenames, datadir)

def cocktail():
    loader = load_cocktail
    exptdir="external_data/dryad/Cocktail Party"
    datadir = get_dataroot(subdir=exptdir)
    filenames = glob(os.path.join(datadir, exptdir, 'EEG', 'Subject*'))
    return (loader,filenames,datadir)


def mark_EMG():
    loader = load_mark_EMG
    exptdir="own_experiments/emg/facial"
    datadir = get_dataroot(subdir=exptdir)
    filenames = glob(os.path.join(datadir, exptdir, 'training_data_SV_*.mat'))
    return (loader,filenames,datadir)    


import mne
import numpy as np
def mne_annotation2stimseq(data, event_id=None, regexp=None):
    fs = data.info['sfreq']
    # event per sample, N.B. may be overlapping?
    events, event_id = mne.events_from_annotations(data, regexp=regexp, chunk_duration=1/fs)
    stimSeq_Sy = np.zeros((len(data.times),1), dtype=int)
    stimSeq_Sy[events[:,0]] = events[:,-1:]
    return stimSeq_Sy, event_id

def mne_stimch2stimseq(data):
    raise NotImplementedError

def mne_eegbci2mindaffectBCI(raw):
    """convert mne raw dataset to a mindaffectBCI dataset

    Args:
        raw ([type]): [description]
    """
    X_Sd = raw.get_data().T
    Y_Sy, level_dict = mne_annotation2stimseq(raw)

    X_TSd = X_Sd[np.newaxis, ...]
    Y_TSy = Y_Sy[np.newaxis, ...]

    ch_names = raw.info['ch_names']
    fs       = raw.info['sfreq']
    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X_TSd.ndim
    coords[0] = {'name':'trial'}
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':[i*1000/fs for i in range(X_TSd.shape[1])], \
                 'fs':fs}
    coords[2] = {'name':'channel','coords':raw.info['ch_names'],'level_dict':level_dict, 'info':raw.info}
    return X_TSd, Y_TSy, coords


def mne_eegbci_runlabels2labels(raw):
    # update the annotation description to reflect the run labels
    runclass2class = dict()
    for r in (1,):
        runclass2class[f"{r:x}0"]='eyes_open'
    for r in (2,):
        runclass2class[f"{r:x}0"]='eyes_closed'
    for r in (3,7,11):
        runclass2class[f"{r:x}0"]='rest'
        runclass2class[f"{r:x}1"]='left.ex'
        runclass2class[f"{r:x}2"]='right.ex'
    for r in (4,8,12):
        runclass2class[f"{r:x}0"]='rest'
        runclass2class[f"{r:x}1"]='left.im'
        runclass2class[f"{r:x}2"]='right.im'
    for r in (5,9,13):
        runclass2class[f"{r:x}0"]='rest'
        runclass2class[f"{r:x}1"]='both.ex'
        runclass2class[f"{r:x}2"]='feet.ex'
    for r in (6,10,14):
        runclass2class[f"{r:x}0"]='rest'
        runclass2class[f"{r:x}1"]='both.im'
        runclass2class[f"{r:x}2"]='feet.im'

    for di,d in enumerate(raw.annotations.description):
        raw.annotations.description[di] = runclass2class.get(d,d)
    return raw

def load_mne_eegbci(run_files):
    if isinstance(run_files,str): run_files=[run_files]
    raws = []
    for f in run_files:
        r = mne.io.read_raw_edf(f)
        run = int(f[-6:-4])
        for di,d in enumerate(r.annotations.description):
            r.annotations.description[di] = { f"T{e:d}":f"{run:x}{e:d}" for e in range(3)}.get(d,d)
        raws.append(r)
    raw = mne.concatenate_raws(raws)
    mne_eegbci_runlabels2labels(raw)
    mne.datasets.eegbci.standardize(raw)  # set channel names
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # convert to mindaffectBCI format
    X, Y, coords = mne_eegbci2mindaffectBCI(raw)
    return X,Y,coords


def mne_eegbci():
    from mne.io import concatenate_raws, read_raw_edf
    from mne.datasets import eegbci
    # N.B. subjects/runs index from 1
    filenames = []
    for si in range(1,110):
        try:
            raw_fnames = eegbci.load_data(si,runs=[i for i in range(1,15)])
            filenames.append(raw_fnames)
        except:
            print("Error with subject: {}".format(si))
    loader = load_mne_eegbci
    return loader, filenames, eegbci.data_path

def mindaffectBCI(exptdir, regexp:str=None, exregexp:str=None, **args):
    loader = load_mindaffectBCI
    if not os.path.exists(exptdir):
        exptdir = os.path.join(get_dataroot(subdir=exptdir),exptdir)
    filenames = glob(os.path.join(os.path.expanduser(exptdir), '**', 'mindaffectBCI*.txt'),recursive=True)
    return loader,filenames,exptdir


def kaggle():
    """generate the dataset directory, loader for the mindaffectBCI kaggle dataset

    Returns:
        _type_: _description_
    """    
    return mindaffectBCI('external_data/kaggle')


def testdataset(fn, **kwargs):
    '''  a conforming toy dataset loader'''
    fs=100
    X_TSd,Y_TSye,st,A,B=testSignal(**kwargs)
    if Y_TSye.ndim==4: # re-code event idicators into a class coding
        Y_TSy = Y_TSye.argmax(axis=-1)+1  # 1,2,... for event type
        Y_TSy[~Y_TSye.any(axis=-1)]=0 # 0 for no events at all

    else:
        Y_TSy = Y_TSye

    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X_TSd.ndim
    coords[0] = {'name':'trial'}
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':[i*1000/fs for i in range(X_TSd.shape[1])], \
                 'fs':fs}
    coords[2] = {'name':'channel','coords':None}
    return X_TSd, Y_TSy, coords

def toy():
    ''' make a toy dataset for testing '''
    loader = testdataset
    filenames = [None]
    datadir = None
    return (loader,filenames,datadir)


def dataset_generator(dataset, **kwargs):
    ''' generator for individual datasets from the given set of datasets '''
    loadfn,  filenames, datadir = get_dataset(dataset)
    for fn in filenames:
        X, Y, coords = loadfn(fn, **kwargs)
        yield (X, Y, coords, fn)

def test_loader(loadfn, filenames, dataroot, **kwargs):
    for fn in filenames:
        try:
            X, Y, coords= loadfn(fn, **kwargs)
            print("ds={}\nX({})={}\nY={}\n".format(fn, [c['name'] for c in coords], X.shape, Y.shape))
        except Exception as ex:
            print("ds={}\nFAILED\n{}".format(fn, ex))



def get_dataset(dsname, regexp:str=None, exregexp:str=None, dataroots:list=None, *args, **kwargs):
    if dataroots:
        set_dataroot(dataroots)

    if dsname == 'openBMI_SSVEP':
        loader, filenames, root =  openBMI("SSVEP")
    elif dsname == 'openBMI_ERP':
        loader, filenames, root =  openBMI("ERP")
    elif dsname == 'openBMI_MI':
        loader, filenames, root =  openBMI("MI")
    elif dsname == 'askloadsavefile':
        loader = load_mindaffectBCI
        filenames = ['askloadsavefile']
        root = None
    else:
        try:
            loader, filenames, root =  eval(dsname)(*args,**kwargs)
        except Exception as ex:
            print("Loader error : {}".format(ex))
            raise NotImplementedError("don't know dataset {}".format(dsname))
    if regexp is not None:
        filenames = [f for f in filenames if re.search(regexp,f)]
    if exregexp is not None:
        filenames = [f for f in filenames if not re.search(exregexp,f)]
    return loader, filenames, root


def testcase():
    datasets=["openBMI_MI","tactileP3","toy","mark_EMG","brains_on_fire","twofinger","ninapro_db2","openBMI_MI","openBMI_ERP","openBMI_SSVEP","cocktail","lowlands","plos_one",'p300_prn',"mTRF_audio"]
    for d in datasets:
        #print(d)
        #try:
        loadfn, filenames,  dataroot = get_dataset(d)
        print("{}) {} Files: \n".format(d,len(filenames)))
        #test_loader(loadfn, filenames, dataroot)
        #except:
        #    print("Error with dataset {}".format(d))

if __name__ == '__main__':
    testcase()
