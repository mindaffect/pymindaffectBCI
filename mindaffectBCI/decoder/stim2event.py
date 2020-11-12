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
def stim2event(M, evtypes=('re','fe'), axis=-1, oM=None):
    '''
    convert per-sample stimulus sequence into per-sample event sequence (e.g. rising/falling edge, or long/short flash)

    Args:
     M  (...samp) or (...,samp,nY): for and/non-target features
     evnames - [nE]:str list of strings:
        "0", "1", "00", "11", "01" (aka. 're'), "10" (aka, fe), "010" (aka. short), "0110" (aka long)
        "nt"+evtname : non-target event, i.e. evtname occured for any other target
        "any"+evtname: any event, i.e. evtname occured for *any* target
        "rest" - not any of the other event types, N.B. must be *last* in event list
        "raw" - unchanged input intensity coding
        "grad" - 1st temporal derivative of the raw intensity
     axis - (-1) the axis of M which runs along 'time'
     oM - (...osamp) or (...,osamp,nY) prefix stimulus values of M, used to incrementally compute the  stimulus features
    Outputs:
     evt    - (M.shape,nE) event sequence
    Examples:
      #For a P300 bci, with target vs. non-target response use:
        M = np.array([[1,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0]]).T
        E = stim2event(M,evtypes=['re','ntre'], axis=-2)
      #Or modeling as two responses, target-stim-response and any-stim-response
        E = stim2event(M,evtypes=('re','anyre'), axis=-2)
    '''
    if axis < 0: # ensure axis is positive!
        axis=M.ndim+axis
    if isinstance(evtypes, str):
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
    for ei, etype in enumerate(evtypes):
        modifier = None
        if etype.startswith("nt"):
            modifier = "nt"
            etype = etype[len(modifier):]
        if etype.startswith("any"):
            modifier = "any"
            etype = etype[len(modifier):]
            
        # 1-bit
        if etype == "flash" or etype == '1':
            F = (M == 1)
        elif etype == '0':
            F = (M == 0)
        # 2-bit
        elif etype == "00":
            F = equals_subarray(M, [0, 0], axis)
        elif etype == "re" or etype == "01":
            F = equals_subarray(M, [0, 1], axis)
        elif etype == "fe" or etype == "10":
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
        # diff
        elif etype == "diff":
            F = np.diff(M, axis=axis) != 0
            # pad missing entry
            padshape=list(F.shape); padshape[axis] = 1
            F = np.append(np.zeros(padshape, dtype=F.dtype), F, axis)

        elif etype == "rest": # i.e. no stimuli anywhere
            if not axis == M.ndim-2:
                raise ValueError("rest only for axis==-2")
            F = np.logical_not(np.any(M, axis=-1, keepdims=True))

        elif etype == 'raw':
            F = M
        elif etype == 'grad': # i.e. gradient of the stimulus
            F = np.diff(M,axis=axis)

        elif etype == 'onset':
            # first stimulus RE for any output
            F = np.cumsum(M>0, axis=axis, dtype=M.dtype) # number of stimulus since trial start 
            F[F>1] = 0 # zero out if more than 1 stimulus since trial start

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

        E[..., ei] = F

    if oM is not None:
        # strip the prefix
        # build index expression to get the  post-fix along axis
        idx=[slice(None)]*E.ndim
        idx[axis]=slice(oM.shape[axis],E.shape[axis])
        # get  the postfix
        E = E[tuple(idx)]
    #print("E.dtype={}".format(E.dtype))
    return E

def testcase():
    from stim2event import stim2event
    # M = (samp,Y)
    M = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]])
    
    print("Raw  :{}".format(M))
    e = stim2event(M, 'flash', axis=-1);     print("flash:{}".format(e[0, ...].T))
    e = stim2event(M, 're', axis=-1);        print("re   :{}".format(e[0, ...].T))
    e = stim2event(M, 'fe', axis=-1);        print("fe   :{}".format(e[0, ...].T))
    e = stim2event(M, 'diff', axis=-1);      print("diff :{}".format(e[0, ...].T))
    e = stim2event(M, ('re', 'fe'), axis=-1); print("refe :{}".format(e[0, ...].T))
    e = stim2event(M, 'onset', axis=-1);     print("onset:{}".format(e[0, ...].T))
    e = stim2event(M.T, ('re', 'fe', 'rest'), axis=-2); print("referest :{}".format(e[0, ...].T))
    e = stim2event(M.T, 'ntre', axis=-2);      print("ntre :{}".format(e[0, ...].T))

    # test incremental calling, propogating prefix between calls
    oM= None
    e = []
    for bi,b in enumerate(range(0,M.shape[-1],2)):
        bM = M[:,b:b+2]
        eb = stim2event(bM,('re','fe'),axis=-1,oM=oM)
        e.append(eb)
        oM=bM
    e = np.concatenate(e,-2)
    print("increfe :{}".format(e[0, ...].T))

if __name__ == "__main__":
    testcase()
