import os
import numpy as np
from scipy.io import loadmat, whosmat
from mindaffectBCI.decoder.utils import block_randomize, butter_sosfilt, upsample_codebook, lab2ind
from mindaffectBCI.decoder.preprocess import extract_envelope

SSVEP_FREQS=(12, 8.57, 6.67, 5.45)
ERP_STIM_DUR = 80/1000 # 80ms
MI_STIM_DUR = 4 # 3s
SSVEP_STIM_DUR = 4

def load_openBMI(datadir, sessdir=None, sessfn=None, fs_out=60, stopband=((45,65),(0,1),(25,-1)), CAR=False, verb=1, trlen_ms=None, offset_ms=(0,0), ppMI=True, ch_names=None):
    """Load and pre-process a openBMI <https://academic.oup.com/gigascience/article/8/5/giz002/5304369> offline save-file and return the EEG data, and stimulus information

    Args:
        datadir (str): root of the data directory tree
        sessdir (str, optional): sub-directory for the session to load. Defaults to None.
        sessfn (str, optional): filename for the session information. Defaults to None.
        fs_out (float, optional): [description]. Defaults to 100.
        stopband (tuple, optional): Specification for a (cascade of) temporal (IIR) filters, in the format used by `mindaffectBCI.decoder.utils.butter_sosfilt`. Defaults to ((45,65),(5.5,25,'bandpass')).
        trlen_ms (float, optional): Trial duration in milli-seconds.  If None then this is deduced from the stimulus information. Defaults to None.
        offset_ms (tuple, (2,) optional): Offset in milliseconds from the trial start/end for the returned data such that X has range [tr_start+offset_ms[0] -> tr_end+offset_ms[0]]. Defaults to (-500,500).
        ch_names (tuple, optional): Names for the channels of the EEG data.
        CAR (bool): flag if we should common-average-reference the raw EEG data

    Returns:
        X (np.ndarray (nTrl,nSamp,nCh)): the pre-processed per-trial EEG data
        Y (np.ndarray (nTrl,nSamp,nY)): the up-sampled stimulus information for each output
        coords (list-of-dicts (3,)): dictionary with meta-info for each dimension of X & Y.  As a minimum this contains
                          "name"- name of the dimension, "unit" - the unit of measurment, "coords" - the 'value' of each element along this dimension
    """    
    
    if offset_ms is None:
        offset_ms = (0, 0)

    # load the data file
    Xfn = datadir
    if sessdir:
        Xfn = os.path.join(Xfn, sessdir)
    if sessfn:
        Xfn = os.path.join(Xfn, sessfn)
    sessdir = os.path.dirname(Xfn)
    fn = os.path.basename(Xfn)
    print("Loading: {}".format(Xfn))
    data = loadmat(Xfn)

    if 'EEG_SSVEP_train' in data:
        # TODO[] : merge training and test
        data = data['EEG_SSVEP_train']
    elif 'EEG_ERP_train' in data:
        data = data['EEG_ERP_train']
    elif 'EEG_MI_train' in data:
        data = data['EEG_MI_train']

    def squeeze(v):
        while v.size == 1 and v.ndim > 0:
            v = v[0]
        return v

    fs = squeeze(data['fs'])
    if ch_names is None:
        ch_names = [d for d in data['chan'][0,:]]
    X = squeeze(data['x']) # (nSamp,d)
    X = np.asarray(X, order='C', dtype='float32')
    print("X={}".format(X.shape),flush=True)
    trl_idx = np.asarray(squeeze(data['t']).ravel(), dtype=int) # (nTrl)
    lab = np.asarray(squeeze(data['y_dec']).ravel(), dtype=int) # (nTrl) label for each trial

    if fs>250: # pre-resample....
        rr = int(fs//250)
        print("pre-downsample : {} -> {}".format(fs,fs/rr))
        X = X [::rr,...]
        trl_idx = trl_idx//rr
        fs = fs/rr
        
    if 'ERP' in fn:
        ep_idx, ep_trl_idx = get_trl_ep_idx(trl_idx, fs*2)
        trl_idx = ep_idx[:,0]
        # make lab the right shape also
        lab0 = lab
        lab = np.zeros(ep_idx.shape,dtype=int)
        for ei, ti_idx in enumerate(ep_trl_idx):
            lab[ei,:len(ti_idx)] = lab0[ti_idx]
        # auto-determine the  right trial length
        if trlen_ms is None:
            trlen_samp = np.max(ep_idx[:,-1]-ep_idx[:,0])
            trlen_ms = trlen_samp * 1000 / fs
    else:
        ep_idx = None
        if trlen_ms is None:
            if 'SSVEP' in Xfn:
                trlen_ms = SSVEP_STIM_DUR * 1000
            elif 'MI' in Xfn:
                trlen_ms = MI_STIM_DUR * 1000
        
    # delete the data variable to free the ram
    del data
    
    # preprocess -> spectral filter, in continuous time!
    if stopband is not None:
        if verb > 0:  print("preFilter: {}Hz".format(stopband))
        X, _, _ = butter_sosfilt(X,stopband,fs)

    if CAR:
        if verb>0 : print("CAR")
        X = X - np.mean(X,-1,keepdims=True)

    # pre-process -> band-power....
    # N.B. *before* slicing to avoid startup artifacts.
    if "MI" in fn and ppMI: # make the target array
        # map X to (log)power in MI relevant frequency bands,
        # N.B. filter-band of **STOPBANDS**
        filterbank=(((0,8),(16,-1)),((0,18),(30,-1))) # relevant frequency bands, mu~=10-14, beta~=20-25 
        env_stopband=(1,-1) # N.B. should be < min(filter-freq)/2 to smooth rectified
        print("map to envelope in bands {}".format(filterbank))
        X = extract_envelope(X,fs,stopband=None,whiten=True,filterbank=filterbank,log=False,env_stopband=env_stopband)
        # update the channel label to include band info
        ch_names = ["{} @{}hz".format(c[0],f[0]) for f in filterbank for c in ch_names]
        if verb>1: print("ch_names={}".format(ch_names))    
    
    # slice X
    trlen_samp = int(trlen_ms * fs / 1000)
    offset_samp = [int(o*fs/1000) for o in offset_ms] # relative to start/end (0,trlen_samp)
    bgnend_samp = (offset_samp[0], trlen_samp+offset_samp[1]) # start end slice window
    xlen_samp = bgnend_samp[1]-bgnend_samp[0]
    print("xslice_samp =[{} - {}] @ {}Hz".format(bgnend_samp[0], bgnend_samp[1],fs))
    Xraw = X
    X = np.zeros((len(trl_idx), xlen_samp, Xraw.shape[-1]), dtype='float32') # (nTrl,nsamp,d)
    for ti, si in enumerate(trl_idx):
        X[ti, :, :] = Xraw[si+bgnend_samp[0]:si+bgnend_samp[1], :]
    del Xraw
        
    # extract the reference signals
    if 'SSVEP' in fn:
        # convert lab+code into Y + samp-times,  where 1st row is always the true label
        if offset_samp[0] > 0 or offset_samp[1] < 0:
            # otherwise more complex to work out how to insert
            raise ValueError("Only offset_ms which padds slice currently supported")
        else:
            cb_idx = slice(-offset_samp[0], -offset_samp[1] if offset_samp[1] > 0 else None)

        # make ssvep codebook 
        cb = make_ssvep_ref_signals(trlen_samp, freqs=SSVEP_FREQS, fs=fs, phases=None) # (nY,nsamp)
        Y  = np.zeros((len(lab), X.shape[1], cb.shape[1]+1), dtype='float32') # (nTr, nSamp, nY)
        for ti, l in enumerate(lab):
            Y[ti, cb_idx, 0]  = cb[:, l-1] # copy in true label
            Y[ti, cb_idx, 1:] = cb # copy in other outputs
        
    elif 'ERP' in fn:
        # permute to make other possible entries
        cb_true = lab == 2 # target=flash (nTrl, nEp)
        cb_true = cb_true[:, :, np.newaxis, np.newaxis] # (nTrl,nEp,nY,e)
        # make 35 virtual targets
        cb = block_randomize(cb_true, 35, axis=-3)
        cb = np.concatenate((cb_true, cb), axis=-2) # [ ..., nY+1]
        # make into a sample rate label set
        Y = upsample_codebook(xlen_samp, cb, ep_idx, fs*ERP_STIM_DUR, offset_samp)
        # BODGE: strip the  feature  dim again..
        Y = Y[...,0]
        
    elif "MI" in fn: # make the target array
        # permute to make other possible entries
        cb_true, lab2class = lab2ind(lab) # (nTrl, e) # feature dim per class
        cb_true = cb_true[:,np.newaxis,:] # (nTrl,1,e)
        cb_all  = np.eye(cb_true.shape[-1]) #(nvirt, e)
        cb_all  = np.tile(cb_all,(cb_true.shape[0], 1, 1)) # (nTrl,nvirt,e)
        cb = np.append(cb_true, cb_all, axis=-2) # (nTrl, nvirt+1, e)
        cb = cb[:,np.newaxis,:,:] # (nTrl,1,nvirt+1,e)
        # make into a sample rate label set
        Y = upsample_codebook(xlen_samp, cb, None, fs*MI_STIM_DUR, offset_samp) #(nTrl,nSamp,nY,e)

        
    # preprocess -> downsample
    resamprate = int(fs/fs_out)
    if resamprate > 1:
        if verb > 0:
            print("resample: {}->{}hz rsrate={}".format(fs, fs/resamprate, resamprate))
        X = X[:, ::resamprate, :] # decimate X (trl, samp, d)
        Y = Y[:, ::resamprate, :] # decimate Y (trl, samp, y)
        fs = fs/resamprate

    stimTimes = None # non-sliced output
    
    # make coords array for the meta-info about the dimensions of X
    coords = [None]*X.ndim
    coords[0] = {'name':'trial','coords':trl_idx, 'lab':lab }
    coords[1] = {'name':'time','unit':'ms', \
                 'coords':np.linspace(offset_ms[0],trlen_ms+offset_ms[1],X.shape[1]), \
                 'fs':fs}
    coords[2] = {'name':'channel','coords':ch_names}
    # return data + metadata
    return (X, Y, coords)


def make_ssvep_ref_signals(nsample, freqs, phases=None, fs=1, binarize=True):
    '''make periodic reference signals'''
    ref = np.zeros((nsample, len(freqs)), dtype='float32')
    for i, freq in enumerate(freqs):
        isi = fs/freq # period in samples
        phase = 0 if phases is None else phases[i]
        ref[:, i] = np.sin((np.arange(nsample)/isi+phase)*2*np.pi)
        if binarize: # put event at phase==0
            ref[:, i] = ref[:, i] > 0
    return ref

def get_trl_ep_idx(trl_idx, iti):
    ''' convert set stimulus times into trials and epochs '''
    # large gap indicates end of this trial
    tr_lims = np.flatnonzero(np.diff(trl_idx) > iti)+1
    tr_lims = np.concatenate(([0], tr_lims, [len(trl_idx)]))
    # insert trial epoch indices into the epoch indices array 
    nEp = np.max(np.diff(tr_lims))
    ep_idx = np.zeros((len(tr_lims)-1, nEp), dtype=int) # (nTrl, nEp)
    ep_trl_idx = [None]*(len(tr_lims)-1) # (nTrl, nEp), list of lists
    for ei in range(0, len(tr_lims)-1):
        tmp = range(tr_lims[ei],tr_lims[ei+1])
        ep_trl_idx[ei] = list(tmp)
        ep_idx[ei, :len(tmp)] = trl_idx[tmp]
    return (ep_idx, ep_trl_idx)


def testcase():
    import sys

    if os.path.isdir('D:\external_data'):
        sessfn = 'D:\external_data\gigadb\openBMI\session1\s1\sml_sess01_subj01_EEG_SSVEP.mat' 
        sessfn = 'D:\external_data\gigadb\openBMI\session1\s1\sml_sess01_subj01_EEG_ERP.mat'
        sessfn = 'D:\external_data\gigadb\openBMI\session1\s2\sml_sess01_subj02_EEG_MI.mat'
    else:
        sessfn = '/home/jadref/data/bci/external_data/gigadb/openBMI/session1/s2/sml_sess01_subj02_EEG_MI.mat'

    # command-line, for testing
    if len(sys.argv) > 1:
        sessfn = sys.argv[1]

    from load_openBMI import load_openBMI
    X, Y, coords = load_openBMI(sessfn, CAR=True, offset_ms=(-400,1000), sessfn=sessfn, fs_out=60, stopband=((0,1),(30,-1)))
    fs = coords[1]['fs']
    # CAR=False,fs_out=60,stopband=((0,3),(29,-1)),rcond=1e-3 : audc=36 Perr[-1]=.30
    # CAR=True,fs_out=60,stopband=((0,3),(29,-1)),rcond=1e-3 : audc=36 Perr[-1]=.30
    # CAR=True,fs_out=60,stopband=((0,3),(29,-1)),rcond=1e-8 : audc=36 Perr[-1]=.30

    
    if 'SSVEP' in sessfn:
        tau_ms = 300
        evtlabs=('re')
        rank=1
    elif 'ERP' in sessfn:
        tau_ms = 700
        evtlabs = ('re','ntre')
        rank=10
    elif 'MI' in sessfn:
        tau_ms = 20
        evtlabs = None
        rank=5

    # visualize the dataset
    from analyse_datasets import debug_test_dataset
    debug_test_dataset(X,Y,coords, rank=rank, evtlabs=evtlabs, tau_ms=tau_ms, offset_ms=0, model='cca', rcond=1e-2)
    
if __name__=="__main__":
    testcase()

