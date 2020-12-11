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
from mindaffectBCI.decoder.utils import equals_subarray
def stim2event(M:np.ndarray, evtypes=('re','fe'), axis:int=-1, oM:np.ndarray=None):
    '''
    convert per-sample stimulus sequence into per-sample event sequence (e.g. rising/falling edge, or long/short flash)

    Args:
     M  (...samp) or (...,samp,nY): for and/non-target features
     evnames (tuple (nE), optional): str list of strings.  Defaults to ('re','fe')
        "0", "1", "00", "11", "01" (aka. 're'), "10" (aka, fe), "010" (aka. short), "0110" (aka long)
        "nt"+evtname : non-target event, i.e. evtname occured for any other target
        "any"+evtname: any event, i.e. evtname occured for *any* target
        "first"+evtname: first occurance of event
        "last"+evtname: last occurace of event
        "rest" - not any of the other event types, N.B. must be *last* in event list
        "raw" - unchanged input intensity coding
        "grad" - 1st temporal derivative of the raw intensity
        "inc" - when value is increasing
        "dec" - when value is decreasing
        "diff" - when value is different
        "new" - new value when value has changed
        "ave" - average stim value over all outputs
        "incXXX" - increasing faster than threshold XXX
        "decXXX" - decreasing faster than threshold XXX
        "diffXXX" - difference larger than threshold XXX
        "hot-one" - hot-one (i.e. a unique event for each stimulus level) encoding of the stimulus values
        "hot-on" - hot-one for non-zero levels (i.e. a unique event for each stimulus level) encoding of the stimulus values
        "hotXXX" - a unique event for each level from 0-XXX
        "output2event" - convert each unique output to it's own event type - N.B. assumes time in axis -2, outputs in axis -1
        XXX : int - stimlus level equals XXX
     axis (int,optional) : the axis of M which runs along 'time'.  Defaults to -1
     oM (...osamp) or (...,osamp,nY): prefix stimulus values of M, used to incrementally compute the  stimulus features

    Returns:
     evt (M.shape,nE): the computed event sequence
     evtlabs (list-of-str) : labels for the output event types

    Examples:
      #For a P300 bci, with target vs. non-target response use:
        M = np.array([[1,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0]]).T
        E,_ = stim2event(M,evtypes=['re','ntre'], axis=-2)
      #Or modeling as two responses, target-stim-response and any-stim-response
        E,_ = stim2event(M,evtypes=('re','anyre'), axis=-2)
    '''
    if axis < 0: # ensure axis is positive!
        axis=M.ndim+axis
    if not hasattr(evtypes,'__iter__') or isinstance(evtypes, str):
        evtypes = [evtypes]
    if evtypes is None:
        return M[:,:,:,np.newaxis]
    if oM is not None:
        #  include the prefix
        M = np.append(oM,M,axis)

    # Copyright (c) MindAffect B.V. 2018
    E = np.zeros(M.shape+(len(evtypes), ), M.dtype) # list event types
    #print("E.dtype={}".format(E.dtype))
    if len(M) == 0: # guard empty inputs
        return E
    # single elment padding matrix    
    padshape=list(M.shape); padshape[axis] = 1; pad = np.zeros(padshape, dtype=M.dtype)
    evtlabs=[]
    for ei, etype in enumerate(evtypes):

        elab = etype
        # extract the stimulus modifier
        modifier = None
        for mod in ('nt','any','onset','offset','first','last'):
            if isinstance(etype,str) and etype.startswith(mod):
                modifier = mod
                etype = etype[len(mod):]
                break
        
        if not isinstance(etype,str): # assume it's a value to match:
            F = (M == etype)

        # 1-bit
        elif etype == "flash" or etype == '1':
            F = (M == 1)
        elif etype == '0':
            F = (M == 0)
        # 2-bit
        elif etype == "00":
            F = equals_subarray(M, [0, 0], axis)
        elif etype == "01" or etype == 're':
            F = equals_subarray(M, [0, 1], axis)
        elif etype == "10" or etype == 'fe':
            F = equals_subarray(M, [1, 0], axis)
        elif etype == "11":
            F = equals_subarray(M, [1, 1], axis)
        # 3-bit
        elif etype == "000":
            F = equals_subarray(M, [0, 0, 0], axis)
        elif etype == "001":
            F = equals_subarray(M, [0, 0, 1], axis)
        elif etype == "010" or etype == 'short':
            F = equals_subarray(M, [0, 1, 0], axis)
        elif etype == "011":
            F = equals_subarray(M, [0, 1, 1], axis)
        elif etype == "100":
            F = equals_subarray(M, [1, 0, 0], axis)
        elif etype == "101":
            F = equals_subarray(M, [1, 0, 1], axis)
        elif etype == "110":
            F = equals_subarray(M, [1, 1, 0], axis)
        elif etype == "111":
            F = equals_subarray(M, [1, 1, 1], axis)
        # 4-bit
        elif etype == "0110" or etype == 'long':
            F = equals_subarray(M, [0, 1, 1, 0], axis)

        elif etype == 'output2event':
            F = M.reshape(M.shape[:-1]+(1,M.shape[-1])) # (...,1,nY)
            elab = np.arange(F.shape[-1])+1

        elif etype == "hot-one" or etype == 'hot-on':
            vals = np.unique(M)
            if etype=='hot-on' and vals[0]==0:
                vals = vals[1:]
            F = M[...,np.newaxis] == vals.reshape((1,)*M.ndim+(vals.size,))
            elab = vals  # labs are now the values used

        elif etype.startswith("hot"):
            n = int(etype[3:]) if len(etype)>3 else 0
            vals = np.arange(n,dtype=M.dtype)
            F = M[...,np.newaxis] == vals.reshape((1,)*M.ndim+(vals.size,))

        # continuous values
        elif etype == "ave":
            if not axis == M.ndim-2:
                raise ValueError("ave only for axis==-2")
            F = np.mean(M, axis=axis+1, keepdims=True)[...,np.newaxis].astype(np.float32)

        elif etype.startswith('inc'): # increasing
            thresh = float(etype[3:]) if len(etype)>3 else 0
            F = np.diff(M, axis=axis, append=pad) > thresh

        elif etype.startswith('dec'): # decreasing
            thresh = float(etype[3:]) if len(etype)>3 else 0
            F = np.diff(M, axis=axis, append=pad) < -thresh

        elif etype.startswith("diff"): # changing
            thresh = float(etype[4:]) if len(etype)>4 else 0
            F = np.abs(np.diff(M, axis=axis, append=pad)) >= thresh

        elif etype.startswith("new") or etype=='step': # new value
            thresh = float(etype[3:]) if len(etype)>3 else 0
            tmp = np.abs(np.diff(M, axis=axis, append=pad)) <= thresh
            F = M.copy()
            F[tmp]=0

        elif etype.startswith('cross'):
            thresh = float(etype[5:]) if len(etype)>5 else 0
            if axis==M.ndim-2:
                tmp = np.logical_and(M[...,:-1,:]<=thresh, thresh<M[...,1:,:]) 
            elif axis==M.ndim-1:
                tmp = np.logical_and(M[...,:-1]<=thresh, thresh<M[...,1:])
            else:
                raise ValueError("cross feature only for axis==-2 or axis==-1, not {}".format(axis))
            # ensure is the right size
            F =  np.append(tmp,np.zeros(pad.shape,dtype=tmp.dtype),axis=axis)

        elif etype == 'grad': # gradient of the stimulus
            F = np.diff(M,axis=axis, append=pad)

        elif etype == "rest": # i.e. no stimuli anywhere
            if not axis == M.ndim-2:
                raise ValueError("rest only for axis==-2")
            F = np.logical_not(np.any(M, axis=-1, keepdims=True))

        elif etype == 'raw':
            F = M

        else:
            raise ValueError("Unrecognised evttype:{}".format(etype))

        # apply any modifiers wanted to F
        if modifier == "nt":
            # non-target, means true when OTHER targets are high, i.e. or over other outputs
            if not axis == M.ndim-2:
                raise ValueError("non-target only for axis==-2")
            anyevt = np.any(F > 0, axis=-1) # any stim event type
            for yi in range(F.shape[-1]):
                # non-target if target for *any* Y except this one
                F[..., yi] = np.logical_and(F[..., yi] == 0, anyevt)
                
        elif modifier == "any":
            if not axis == M.ndim-2:
                raise ValueError("any feature only for axis==-2")   
            # any, means true if any target is true, N.B. use logical_or to broadcast
            F = np.any(F > 0, axis=-1, keepdims=True)

        elif modifier in ('onset','first'):
            # first stimulus RE for any output
            F = np.cumsum(F, axis=axis, dtype=M.dtype) # number of stimulus since trial start 
            F[F>1] = 0 # zero out if more than 1 stimulus since trial start

        elif modifier in ('offset','last'):
            # first stimulus RE for any output
            F = np.cumsum(F, axis=axis, dtype=M.dtype) # number of stimulus since trial start 
            F[F>1] = 0 # zero out if more than 1 stimulus since trial start

        if F.shape == M.shape or F.shape[:-1] == M.shape[:-1]: # or can scale up to same size
            E[..., ei] = F
            evtlabs.append(elab)
        elif len(evtypes)==1:
            E = F.astype(E.dtype)
            evtlabs.extend(elab)
        else:
            raise ValueError("Cant (currently) mix direct and indirect encodings")
        

    if oM is not None:
        # strip the prefix
        # build index expression to get the  post-fix along axis
        idx=[slice(None)]*E.ndim
        idx[axis]=slice(oM.shape[axis],E.shape[axis])
        # get  the postfix
        E = E[tuple(idx)]
    #print("E.dtype={}".format(E.dtype))
    return E, evtlabs

def testcase():
    from stim2event import stim2event
    # M = (samp,Y)
    M = np.array([[0, 0, 2, 1, 0, 0, 0, 0, 1, 2, 1, 1, 0],
                  [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]])
    
    print("Raw  :{}".format(M))
    e,_ = stim2event(M, 1, axis=-1);     print("1:{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'flash', axis=-1);     print("flash:{}".format(e[0, ...].T))
    e,_ = stim2event(M, 're', axis=-1);        print("re   :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'fe', axis=-1);        print("fe   :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'diff', axis=-1);      print("diff :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'inc', axis=-1);       print("inc  :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'dec', axis=-1);       print("dec  :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'new', axis=-1);      print("new  :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'grad', axis=-1);      print("grad :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'hot-one', axis=-1);      print("hot-one :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'hot2', axis=-1);      print("hot2 :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'output2event', axis=-1);   print("output2event :{}".format(e[0, ...].T))
    e,_ = stim2event(M*30, 'inc10', axis=-1);       print("inc10 :{}".format(e[0, ...].T))
    e,_ = stim2event(M*30, 'dec10', axis=-1);       print("dec10 :{}".format(e[0, ...].T))
    e,_ = stim2event(M-1, 'cross0', axis=-1);       print("cross0 :{}".format(e[0, ...].T))
    e,_ = stim2event(M, ('re', 'fe'), axis=-1); print("refe :{}".format(e[0, ...].T))
    e,_ = stim2event(M, 'onsetre', axis=-1);     print("onsetre:{}".format(e[0, ...].T))
    e,_ = stim2event(M.T, ('re', 'fe', 'rest'), axis=-2); print("referest :{}".format(e[0, ...].T))
    e,_ = stim2event(M.T, 'ntre', axis=-2);      print("ntre :{}".format(e[0, ...].T))

    # test incremental calling, propogating prefix between calls
    oM= None
    e = []
    for bi,b in enumerate(range(0,M.shape[-1],2)):
        bM = M[:,b:b+2]
        eb,_ = stim2event(bM,('re','fe'),axis=-1,oM=oM)
        e.append(eb)
        oM=bM
    e = np.concatenate(e,-2)
    print("increfe :{}".format(e[0, ...].T))

if __name__ == "__main__":
    testcase()
