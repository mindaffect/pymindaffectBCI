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
from mindaffectBCI.decoder.utils import window_axis

#@function
def scoreStimulus(X, W, R=None, b=None, offset=0, f=None, isepoched=None):
    '''
    Apply spatio-temporal (possibly factored) model to data 
    Inputs:
      X = (nTrl x nSamp x d) raw response for the current stimulus event
               d=#electrodes tau=#response-samples  nEpoch=#stimulus events to process
         OR
          (nTrl x nEpoch x tau x d) pre-sliced raw data
      W = (nM x nfilt x d) spatial filters for each output
      R = (nM x nfilt x nE x tau) responses for each stimulus event for each output
     OR
      W = (nM x nE x tau x d) spatio-temporal filter per event type and model
      R = None
      b = (nE,1) bias for each stimulus type
      offset = 0 (1,1) offset in X for applying W
    Outputs:
      Fe= (nM x nTrl x nEpoch/nSamp x nE) similarity score for each input epoch for each output
    Copyright (c) MindAffect B.V. 2018
    '''
    tau = W.shape[-2] if R is None else R.shape[-1] # est response length
    if isepoched is None: # guess epoched state from the shapes...
        isepoched = X.ndim > 3
    if isepoched:
        Fe = scoreStimulusEpoch(X, W, R, b)
    else:
        Fe = scoreStimulusCont(X, W, R, b, offset)
    return Fe

def scoreStimulusEpoch(X, W, R=None, b=None):
    '''
    Apply spatio-temporal (possibly factored) model to epoched data 
      X = (nTrl x nEpoch x tau x d) pre-sliced raw data
      W = (nM x nfilt x d) spatial filters for each output
      R = (nM x nfilt x nE x tau) responses for each stimulus event for each output
     OR
      W = (nM x nE x tau x d) spatio-temporal filter per event type and model
      R = None
      b = (nE,1) offset for each stimulus type
    Outputs:
      Fe= (nM x nTrl x nEpoch/nSamp x nE) similarity score for each input epoch for each output
    '''
    if R is not None:
        Fe = scoreStimulusEpoch_factored(X,W,R,b)
    else:
        Fe = scoreStimulusEpoch_full(X,W,b)
    return Fe


def scoreStimulusEpoch_factored(X, W, R, b=None):
    '''
    Apply factored spatio-temporal model to epoched data 
      X = (nTrl x nEpoch x tau x d) pre-sliced raw data
      W = (nM x nfilt x d) spatial filters for each output
      R = (nM x nfilt x nE x tau) responses for each stimulus event for each output
      b = (nE,1) offset for each stimulus type
    Outputs:
      Fe= (nM x nTrl x nEpoch x nE) similarity score for each input epoch for each output
    '''

    # ensure all inputs have the  right  shape, by addig leading singlenton dims
    X = X.reshape((1,)*(4-X.ndim)+X.shape) # (nTrl,nEp,tau,d)
    W = W.reshape(((1,)*(3-W.ndim))+W.shape) # (nM,nfilt,d)
    R = R.reshape(((1,)*(4-R.ndim))+R.shape) # (nM,nfile,nE,tau)

    # apply the factored model, complex product to optimize the path
    Fe = np.einsum("Mkd,TEtd,Mket->MTEe", W, X, R, optimize='optimal') 
    #Fe = np.einsum("Mkd,TEtd->TEMkt",W,X) # manual factored
    #Fe = np.einsum("TEMkt,Mket->MTEe",Fe,R)
    if not b is None: # include the bias, for each stimulus type
        Fe = Fe + b
    return Fe


def scoreStimulusEpoch_full(X, W, b=None):
    '''
    Apply full spatio-temporal model to epoched data 
      X = (nTrl x nEpoch x tau x d) pre-sliced raw data
      W = (nM x nE x tau x d) spatio-temporal filter per event type and model
      b = (nE,1) offset for each stimulus type
    Outputs:
      Fe= (nM x nTrl x nEpoch/nSamp x nE) similarity score for each input epoch for each output
    '''

    # ensure inputs have the  right  shape
    X = X.reshape((1,)*(4-X.ndim)+X.shape)
    W = W.reshape((1,)*(4-W.ndim)+W.shape)

    # apply the model
    Fe = np.einsum("TEtd, metd->mTEe", X, W, optimize='optimal')
    return Fe

def factored2full(W, R):
    ''' convert a factored spatio-temporal model to a full model
    Inputs:
       W (nM,rank,d) spatial filter set (BWD model)
       R (nM,rank,e,tau) temporal filter set (FWD model)
    Output:
       W (nM,e,tau,d) spatio-temporal filter (BWD model) '''
    if R is not None:
        W = W.reshape(((1,)*(3-W.ndim))+W.shape)
        R = R.reshape(((1,)*(4-R.ndim))+R.shape)
        # get to single spatio-temporal filter
        W = np.einsum("mfd, mfet->metd", W, R)
    return W

def scoreStimulusCont(X, W, R=None, b=None, offset=0):
    """ Apply spatio-tempoal (possibly factored) model to raw (non epoched) data

    Args:
        X (np.ndarray (nTr,nSamp,d)): raw per-trial data 
        W (np.ndarray (nM,nfilt,d)): spatial filters for each factor 
        R (np.ndarray (nM,nfilt,nE,tau): responses for each stimulus event for each output
        b (np.ndarray (nE,1)): offset for each stimulus type
    Returns:
        np.ndarray (nM,nTrl,nSamp,nE): similarity score for each input epoch for each output
    """   
    tau = W.shape[-2] if R is None else R.shape[-1] # get response length
    if X.shape[-2] < tau: # X isn't big enough to apply... => zero score
        Fe = np.zeros((W.shape[0],X.shape[0],1,W.shape[1]),dtype=X.dtype)
    
    # slice and apply
    Xe = window_axis(X, winsz=tau, axis=-2) # (nTrl, nSamp-tau, tau, d)
    Fe = scoreStimulusEpoch(Xe, W, R, b) # (nM, nTrl, nSamp-tau, nE)

    # shift for the offset and zero-pad to the input X size
    # N.B. as we are offsetting from X->Y we move in the **OPPOSITTE** direction to
    # how Y is shifted! 
    Feoffset=-offset
    if Feoffset<=0:
        tmp = Fe[..., -Feoffset:, :] # shift-back and shrink
        Fe = np.zeros(Fe.shape[:-2]+(X.shape[-2],)+Fe.shape[-1:],dtype=Fe.dtype)
        Fe[...,:tmp.shape[-2],:] = tmp # insert
    else :
        tmp =  Fe[..., :X.shape[-2]-Feoffset, :] # shrink
        Fe = np.zeros(Fe.shape[:-2]+(X.shape[-2],)+Fe.shape[-1:],dtype=Fe.dtype)
        Fe[...,Feoffset:Feoffset+tmp.shape[-2],:] = tmp # shift + insert

    return Fe

    
def plot_Fe(Fe):
    import matplotlib.pyplot as plt
    '''plot the stimulus score function'''
    #print("Fe={}".format(Fe.shape))
    if Fe.ndim > 3:
        if Fe.shape[0]>1 :
            print('Warning stripping model dimension!')
        Fe = Fe[0,...]
    plt.clf()
    nPlts=min(25,Fe.shape[0])
    if Fe.shape[0]/2 > nPlts:
        tis = np.linspace(0,Fe.shape[0]/2-1,nPlts,dtype=int)
    else:
        tis = np.arange(0,nPlts,dtype=int)
    ncols = int(np.ceil(np.sqrt(nPlts)))
    nrows = int(np.ceil(nPlts/ncols))
    axploti = ncols*(nrows-1)
    ax = plt.subplot(nrows,ncols,axploti+1)
    for ci,ti in enumerate(tis):
        # make the axis
        if ci==axploti: # common axis plot
            pl = ax
        else: # normal plot
            pl = plt.subplot(nrows,ncols,ci+1,sharex=ax, sharey=ax) # share limits
            plt.tick_params(labelbottom=False,labelleft=False) # no labels        
        pl.plot(Fe[ti,:,:])
        pl.set_title("{}".format(ti))
    pl.legend(range(Fe.shape[-1]-1))
    plt.suptitle('Fe')

#@function
def testcase():
    #   X = (nTrl, nEp, tau, d) [d x tau x nEpoch x nTrl ] raw response for the current stimulus event
    #            d=#electrodes tau=#response-samples  nEpoch=#stimulus events to process
    #    w = (nM, nfilt, d) spatial filters for each output
    #    r = (nM, nfilt, nE, tau) responses for each stimulus event for each output
    nE = 2
    d = 8
    tau = 10
    nSamp = 300
    nTrl = 30
    nfilt = 1
    nM = 20
    X = np.random.randn(nTrl, nSamp, d)
    Xe = window_axis(X, winsz = tau, axis = -2)
    W = np.random.randn(nM, nfilt, d)
    R = np.random.randn(nM, nfilt, nE, tau)
    Fe = scoreStimulus(Xe, W, R)
    print("Xe={} -> Fe={}".format(Xe.shape, Fe.shape))
    
    Wf = factored2full(W,R)
    Fef = scoreStimulus(Xe,Wf)
    print("Wf={} -> Fef={}".format(Wf.shape,Fef.shape))

    print("Fe-Fef={}".format(np.max(np.abs(Fe-Fef).ravel())))

    F = scoreStimulusCont(X, W, R)
    print("X={} -> F={}".format(X.shape, F.shape))

if __name__=="__main__":
    testcase()
