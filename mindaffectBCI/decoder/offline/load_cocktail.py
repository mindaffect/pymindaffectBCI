import os
from glob import glob
import numpy as np
from scipy.io import loadmat, whosmat
from mindaffectBCI.decoder.utils import block_randomize, butter_sosfilt, window_axis


def argsort(l):
    return sorted(range(len(l)), key=l.__getitem__)

def squeeze(v):
    while v.size == 1 and v.ndim > 0:
        v = v[0]
    return v

books = ['journey','20000']
subids_attended_book=[None, #0
                      'journey','journey','20000', 'journey', 'journey', '20000', '20000', '20000', '20000', '20000', # 1-10 
                      'journey','journey','20000', 'journey', '20000', 'journey', '20000', '20000', '20000', 'journey',     # 11-20
                      '20000', 'journey', 'journey', '20000', '20000', 'journey', '20000', 'journey', 'journey', 'journey', # 21-30
                      '20000', '20000', 'journey'] # 31-33
subids_journey=[1,2, 4, 5, 11, 12, 14, 16, 20, 22, 23, 26, 28, 29, 30, 33]
subids_20000=[3, 6, 7, 8, 9, 10, 13, 15, 17, 18, 19, 21, 24, 25, 27, 31, 32]

# 128ch biosemi+10-10
ch_names=('Cz','a2','CPz','a4','a5','a6','a7','a8','a9','PO7','a11','a12','a13','a14','O1','a16','a17','a18','Pz','a20','POz','a22','Oz','a24','Iz','a26','a27','O2','a29','a30','a31','a32','b1','b2','b3','b4','b5','b6','PO8','b8','b9','b10','P8','b12','b13','TP8','b15','b16','b17','b18','b19','C2','b2','C4','b23','C6','b25','T8','FT8','b28','b29','b30','b31','b32','c1','c2','c3','c4','c5','c6','F8','AF8','c9','c10','c11','c12','c13','c14','c15','Fp2','Fpz','c18','AFz','c20','Fz','c22','FCz','c24','c25','c26','c27','c28','Fp1','AF7','c31','c32','d1','d2','d3','d4','d5','d6','F7','FT7','d9','d10','d11','d12','d13','C1','d15','d16','d17','d18','C3','d20','C5','d22','T7','TP7','d25','d26','d27','d28','d29','d30','P7','d32')

def load_cocktail(datadir, sessdir=None, sessfn=None, fs_out=60, stopband=((45,65),(0,5),(25,-1)), verb=0, trlen_ms=None, subtriallen=10):

    # load the data file
    Xfn = os.path.expanduser(datadir)
    if sessdir:
        Xfn = os.path.join(Xfn, sessdir)
    if sessfn:
        Xfn = os.path.join(Xfn, sessfn)
    sessdir = Xfn if os.path.isdir(Xfn) else os.path.dirname(Xfn)
    stimdir = os.path.join(sessdir,'..','..','Stimuli','Envelopes')
    
    runfns = glob(os.path.join(sessdir, '*.mat'))

    # extract subId (to get attended stream)
    subid = int(sessdir.split("Subject")[1].split("_")[0])
    attended_book = subids_attended_book[subid]
    # extract run id
    runid = [int(f.split("Run")[1].split(".mat")[0]) for f in runfns]
    # sort into numeric order
    sorted_id = argsort(runid)
    runfns = [(runid[i], runfns[i]) for i in sorted_id]

    # load the raw EEG data
    data = [None]*len(runid)
    stim = [None]*len(runid)
    print("Run:", end='')
    for i, (ri, rf) in enumerate(runfns):
        print("{} ".format(ri), end='', flush=True)
        data[i] = loadmat(rf)
        # load
        stim[i] = [loadmat(os.path.join(stimdir,book,"{}_{}_env.mat".format(book,ri))) for book in books]
    # make a label list for the trials
    lab = [books.index(attended_book)]*len(data)
    fs = squeeze(data[0]['fs'])

    if not all(squeeze(d['fs']) == fs for d in data):
        raise ValueError("Different sample rates in different runs")
    #if not all(d['fsEnv'] == fs for d in stim):
    #    raise valueError("Different samples rates in between EEG and Envelope")
        
    # make the X and Y arrays
    X0 = data[0]['eegData']
    Y0 = stim[0][0]['envelope']
    nSamp = min(X0.shape[0], Y0.shape[0])
    d = X0.shape[1]
    e = Y0.shape[1]
    X = np.zeros((len(runid), nSamp, d), dtype='float32')
    Y = np.zeros((len(runid), nSamp, 1+len(stim[0]), e), dtype='float32') #  (nTr,nSamp,nY,e)
    for ti,(d,s) in enumerate(zip(data, stim)):
        X[ti, :, :] = d['eegData'][:nSamp, :]
        Y[ti, :, 0, :] = s[lab[ti]]['envelope'][:nSamp, :] # objID==0 is attended
        for si,ss in enumerate(s): # all possible stimuli
            Y[ti, :, si+1, :] = ss['envelope'][:nSamp, :]
    
    print("X={}".format(X.shape), flush=True)

    # preprocess -> spectral filter, in continuous time!
    if stopband is not None:
        if verb > 0:
            print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X,stopband,fs)

    # preprocess -> downsample
    resamprate = int(fs/fs_out)
    if resamprate > 1:
        if verb > 0:
            print("resample: {}->{}hz rsrate={}".format(fs, fs_out, resamprate))
        X = X[:, ::resamprate, :] # decimate X (trl, samp, d)
        Y = Y[:, ::resamprate, ...] # decimate Y (trl, samp, y, e)
        fs = fs/resamprate

    nsubtrials = X.shape[1]/fs/subtriallen
    if nsubtrials > 1:
        winsz = int(X.shape[1]//nsubtrials)
        print('{} subtrials -> winsz={}'.format(nsubtrials,winsz))
        # slice into sub-trials
        X = window_axis(X,axis=1,winsz=winsz,step=winsz) # (trl,win,samp,d)
        Y = window_axis(Y,axis=1,winsz=winsz,step=winsz) # (trl,win,samp,nY)
        # concatenate windows into trial dim
        X = X.reshape((X.shape[0]*X.shape[1],)+X.shape[2:])
        Y = Y.reshape((Y.shape[0]*Y.shape[1],)+Y.shape[2:])
        print("X={}".format(X.shape))

    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = {'name':'trial'}
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':np.arange(X.shape[1])/fs, \
                 'fs':fs}
    coords[2] = {'name':'channel','coords':ch_names}
    # return data + metadata
    return (X, Y, coords)

def testcase():
    datadir= '~/data/bci/external_data/dryad/Cocktail Party'
    datadir= 'D://external_data/dryad/Cocktail Party'
    sessdir= 'EEG/Subject1'
    sessfn = ''
    X,Y,coords=load_cocktail(datadir, sessdir, sessfn, subtriallen=15)

    from analyse_datasets import debug_test_dataset
    debug_test_dataset(X, Y, coords, tau_ms=500, rank=4, evtlabs=None)
    
    
if __name__=='__main__':
    testcase()
