import os
from glob import glob
import re
from mindaffectBCI.decoder.offline.load_mark_EMG import load_mark_EMG
from mindaffectBCI.decoder.offline.load_brainstream import load_brainstream
from mindaffectBCI.decoder.offline.load_p300_prn import load_p300_prn
from mindaffectBCI.decoder.offline.load_openBMI import load_openBMI
from mindaffectBCI.decoder.offline.load_cocktail import load_cocktail
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

from mindaffectBCI.decoder.offline.load_mTRF_audio import load_mTRF_audio
from mindaffectBCI.decoder.offline.load_twofinger import load_twofinger
from mindaffectBCI.decoder.offline.load_brainsonfire import load_brainsonfire
from mindaffectBCI.decoder.offline.load_ninapro_db2 import load_ninapro_db2
from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI
from mindaffectBCI.decoder.utils import testSignal

def plos_one():
    '''generate the directory+filename info for the plos_one noisetagging dataset'''
    loader = load_brainstream # function to call to load the dataset
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot = os.path.expanduser('~/data/bci/own_experiments/noisetagging_v3')
    else:
        dataroot = 'D:/own_experiments/noisetagging_v3/'
    sessdir = glob(os.path.join(dataroot, 's[0-9]*'))
    #sessdir = ['s{:d}'.format(i) for i in range(1, 12)]
    sessfn = 'traindata.mat'
    filenames = [os.path.join(dataroot, d, sessfn) for d in sessdir]
    return (loader, filenames, dataroot)

def lowlands():
    '''generate the directory+filename info for the lowlands noisetagging dataset'''
    loader = load_brainstream
    if os.path.isdir(os.path.expanduser('~/data/bci')):
    	dataroot = os.path.expanduser('~/data/bci/own_experiments/lowlands')
    else:
        dataroot = 'D:/own_experiments/lowlands/'
    filenames = glob(os.path.join(dataroot, '*_tr_train_1.mat'))
    return (loader, filenames, dataroot)

def p300_prn(label:str=None):
    '''generate dataset+filename for p300-prn2'''
    loader = load_p300_prn # function to call to load the dataset
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot = os.path.expanduser('~/data/bci/')
    else:
        dataroot = 'D:/'
    expt = 'own_experiments/visual/p300_prn_2'
    filenames = glob(os.path.join(dataroot, expt, '*/*/jf_prep/*flash.mat')) + \
                glob(os.path.join(dataroot, expt, '*/*/jf_prep/*flip.mat'))
    if label is not None:
        filenames = [f for f in filenames if re.search(label,f)]
    return (loader, filenames, dataroot)

def tactileP3():
    '''generate dataset+filename for tactile P3'''
    loader = load_p300_prn # function to call to load the dataset
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot = os.path.expanduser('~/data/bci/')
    else:
        dataroot = 'D:/'
    expt = 'own_experiments/tactile/P3speller'
    filenames = glob(os.path.join(dataroot, expt, '*/*/jf_prep/*offline.mat'))
    return (loader, filenames, dataroot)

def tactile_PatientStudy():
    '''generate dataset+filename for p300-prn2'''
    loader = load_p300_prn # function to call to load the dataset
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot = os.path.expanduser('~/data/bci/')
    else:
        dataroot = 'D:/'
    expt = 'own_experiments/tactile/PatientStudy'
    filenames = glob(os.path.join(dataroot, expt, '*/*/jf_prep/*offline.mat'))
    return (loader, filenames, dataroot)

def openBMI(dstype="SSVEP"):
    loader = load_openBMI
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot = os.path.expanduser('~/data/bci/external_data/gigadb/openBMI')
    else:
        dataroot= os.path.expanduser('D:/external_data/gigadb/openBMI')
    filenames = glob(os.path.join(dataroot, 'sess*/s*/sml_*'+ dstype + '.mat')) + \
                glob(os.path.join(dataroot, 'sml_*'+ dstype + '.mat'))
    return (loader, filenames, dataroot)

def twofinger():
    loader = load_twofinger
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot = os.path.expanduser('~/data/bci/')
    else:
        dataroot= os.path.expanduser('D:/')
    exptdir = 'external_data/twente/twofinger'
    filenames =  glob(os.path.join(dataroot, exptdir, 'S??.mat'))
    return (loader, filenames, dataroot)

def brains_on_fire_online():
    loader = load_brainsonfire
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot = os.path.expanduser('~/data/bci/')
    else:
        dataroot= os.path.expanduser('D:/')
    exptdir = 'own_experiments/motor_imagery/brainsonfire/brains_on_fire_online'
    filenames =  glob(os.path.join(dataroot, exptdir, 'subject*/raw_buffer/0001'))
    return (loader, filenames, dataroot)


def mTRF_audio():
    loader = load_mTRF_audio
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot= os.path.expanduser('~/data/bci/external_data/mTRF')
    else:
        dataroot= os.path.expanduser('D:/external_data/mTRF')
    filenames = [os.path.join(dataroot, 'speech_data.mat')]
    return (loader, filenames, dataroot)

def ninapro_db2():
    loader = load_ninapro_db2
    if os.path.isdir(os.path.expanduser("~/data")):
        dataroot = os.path.expanduser('~/data/bci/')
    else:
        dataroot = "D:/"
    exptdir="external_data/ninapro"
    filenames = glob(os.path.join(dataroot, exptdir, 's*', '*E1*.mat'))
    return (loader, filenames, dataroot)

def cocktail():
    loader = load_cocktail
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot= os.path.expanduser('~/data/bci/')
    else:
        dataroot= os.path.expanduser('D:/')
    exptdir="external_data/dryad/Cocktail Party"
    filenames = glob(os.path.join(dataroot, exptdir, 'EEG', 'Subject*'))
    return (loader,filenames,dataroot)

def mark_EMG():
    loader = load_mark_EMG
    if os.path.isdir(os.path.expanduser('~/data/bci')):
        dataroot= os.path.expanduser('~/data/bci/')
    else:
        dataroot= os.path.expanduser('D:/')
    exptdir="own_experiments/emg/facial"
    filenames = glob(os.path.join(dataroot, exptdir, 'training_data_SV_*.mat'))
    return (loader,filenames,dataroot)    

def testdataset(fn, **args):
    '''  a conforming toy dataset loader'''
    fs=10
    X,Y,st,A,B=testSignal(**args)
    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = {'name':'trial'}
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':[i*1000/fs for i in range(X.shape[1])], \
                 'fs':fs}
    coords[2] = {'name':'channel','coords':None}
    return (X, Y, coords)

def mindaffectBCI(exptdir, **args):
    loader = load_mindaffectBCI
    filenames = glob(os.path.join(os.path.expanduser(exptdir), 'mindaffectBCI*.txt'))
    return (loader,filenames,exptdir)

def toy():
    ''' make a toy dataset for testing '''
    loader = testdataset
    filenames = [None]
    dataroot = None
    return (loader,filenames,dataroot)

def dataset_generator(dataset, **kwargs):
    ''' generator for individual datasets from the given set of datasets '''
    loadfn,  filenames, dataroot = get_dataset(dataset)
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

def get_dataset(dsname,*args, **kwargs):
    if dsname == 'openBMI_SSVEP':
        return openBMI("SSVEP")
    elif dsname == 'openBMI_ERP':
        return openBMI("ERP")
    elif dsname == 'openBMI_MI':
        return openBMI("MI")
    else:
        try:
            return eval(dsname)(*args,**kwargs)
        except:
            raise NotImplementedError("don't know dataset {}".format(dsname))

def testcase():
    datasets=["openBMI_MI","tactileP3","toy","mark_EMG","brainsonfire","twofinger","ninapro_db2","openBMI_MI","openBMI_ERP","openBMI_SSVEP","cocktail","lowlands","plos_one",'p300_prn',"mTRF_audio"]
    for d in datasets:
        print(d)
        #try:
        loadfn, filenames,  dataroot = get_dataset(d)
        test_loader(loadfn, filenames, dataroot)
        #except:
        #    print("Error with dataset {}".format(d))

if __name__ == '__main__':
    testcase()
