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

import numpy as np
from mindaffectBCI.decoder.utils import equals_subarray
def stim2event(M:np.ndarray, evtypes=('re','fe'), axis:int=-2, oM:np.ndarray=None, **kwargs):
    '''
    convert per-sample stimulus sequence into per-sample event sequence (e.g. rising/falling edge, or long/short flash)

    Args:
     M  (...samp) or (...,samp,nY): for and/non-target features
     evnames (tuple (nE), optional): str list of strings, or list of values, or list of functions.  Defaults to ('re','fe')
        "0", "1", "00", "11", "01", "10", "010" (aka. short), "0110" (aka long)
        "re", "onset" - rising edge, or onset of this event
        "fe", "offset" - falling edge, or offset from this value
        ">XXX" - greater than XXX
        "<XXX" - less than XXX
        "=XXX" - equal to XXX
        "reX,Y,Z" - rising to a value of X or Y or Z ...
        "feX,Y,Z" - falling from a value of X or Y or Z
        "prX,Y" - pattern reversal from X to Y or Y to X
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
        "hot-yon" - a unique event for each unique combination of outputs and stimulus levels
        "output2event" - convert each unique output to it's own event type - N.B. assumes time in axis -2, outputs in axis -1
        XXX : int - stimlus level equals XXX
        [XXX,YYY,ZZZ]: list - stimulus matches any of the given levels XXX,YYY,ZZZ
     axis (int,optional) : the axis of M which runs along 'time'.  Defaults to -2
     oM (...osamp) or (...,osamp,nY): prefix stimulus values of M, used to incrementally compute the  stimulus features

    Note: you can write your own stim2event mappings with this function call signature:
       s2e(M:ndarray, axis:int=-1, oM:ndarray=None) -> (F:ndarray, s2estate, elab:str)
    where M - (...samp) or (...,samp,nY),  axis='time-axis', F-(M.shape +(nevt,1)), s2estate=internal-state for reproducable calls, elab=list of nevt-str
    For example:
        def s2equal(M,val,axis=-1,oM=None): return M==val, None, val
        F, _, _ = stim2event(M,etypes=(lambda M,axis,oM: s2equal(M,10,axis,oM)))

    Returns:
     evt (M.shape,nE): the computed event sequence
     state (Any) : internal state of the converter, to reproduce the transformation later
     evtlabs (list-of-str) : labels for the output event types

    Examples:
      #For a P300 bci, with target vs. non-target response use:
        M = np.array([[1,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0]]).T
        E,_, _ = stim2event(M,evtypes=['re','ntre'], axis=-2)
      #Or modeling as two responses, target-stim-response and any-stim-response
        E,_, _ = stim2event(M,evtypes=('re','anyre'), axis=-2)
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
    s2estate = None
    E = np.zeros(M.shape+(len(evtypes), ), M.dtype) # list event types
    #print("E.dtype={}".format(E.dtype))
    if len(M) == 0: # guard empty inputs
        return E, evtypes, s2estate
    # single elment padding matrix    
    padshape=list(M.shape); padshape[axis] = 1; pad = np.zeros(padshape, dtype=M.dtype)
    evtlabs=[]
    s2estates=[]
    for ei, etype in enumerate(evtypes):

        elab = etype
        s2estate = etype
        # extract the stimulus modifier
        modifier = None
        for mod in ('nt','any','first','last'):
            if isinstance(etype,str) and etype.startswith(mod):
                modifier = mod
                etype = etype[len(mod):]
                break

        if isinstance(etype,str): # try to convert to callable
            fn = locals().get(etype) or globals().get(etype)
            if not fn is None: etype=fn

        if callable(etype): # function to call, matches our signature
            F, s2estate, elab = etype(M, axis, oM, etype=etype.__name__, **kwargs)

        elif not isinstance(etype,str): # not-function not str -> assume it's a value to match:
            if hasattr(etype,'__iter__'): # set values to match
                F = np.any(M[...,np.newaxis] == np.array(etype, dtype=M.dtype), -1)
            else:
                F = (M == etype)

        # 1-bit
        elif etype == "flash" or etype == '1':
            F = (M == 1)
        elif etype == '0':
            F = (M == 0)
        # 2-bit
        elif etype == "00":
            F = equals_subarray(M, [0, 0], axis)
        elif etype == "01":
            F = equals_subarray(M, [0, 1], axis)
        elif etype == "10":
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

        # continuous values

        elif etype == "rest": # i.e. no stimuli anywhere
            if not axis == M.ndim-2:
                raise ValueError("rest only for axis==-2")
            F = np.logical_not(np.any(M, axis=-1, keepdims=True))

        elif etype == 'raw':
            F = M

        elif etype.startswith("<"):
            n = float(etype[1:]) if len(etype)>1 else 0
            F = M < n

        elif etype.startswith(">"):
            n = float(etype[1:]) if len(etype)>1 else 0
            F = M > n

        elif etype.startswith("="):
            if len(etype)>1:
                val = [float(e) for e in etype[1:].split(',')]
                F = np.any(M[...,np.newaxis]==val,M.ndim)
            else:
                F = (M == 0)

        elif etype.startswith("pr"):
            a,b = [int(e) for e in etype[2:].replace(',', '_').split('_')]
            F = np.logical_or(equals_subarray(M, (a,b), axis),
                              equals_subarray(M, (b,a), axis))

        elif etype.startswith("re") or etype in ("onset",):
            if etype.startswith('re') and len(etype)>2:
                val = [float(e) for e in etype[2:].split(',')]
                F, s2estate, elab = riseto(M, val, axis, oM, etype)
            else:
                F, s2estate, elab = re(M, axis, oM, etype)

        elif etype.startswith("fe") or etype in ("offset",):
            if etype.startswith("fe") and len(etype)>2:
                val = [float(e) for e in etype[2:].split(',')]
                F, s2estate, elab = fallfrom(M, val, axis, oM, etype)
            else:
                F, s2estate, elab = fe(M, axis, oM, etype)

        else:

            if callable(etype):
                F, s2estate, elab = etype(M, axis, oM, etype=fn.__name__, **kwargs)
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
            s2estate = modifier + s2estate

        elif modifier == "any":
            if not axis == M.ndim-2:
                raise ValueError("any feature only for axis==-2")   
            # any, means true if any target is true, N.B. use logical_or to broadcast
            F = np.any(F > 0, axis=-1, keepdims=True)
            #F = np.repeat(F,repeats=M.shape[-1],axis=-1) # blow up to orginal size
            s2estate = modifier + s2estate

        elif modifier in ('onset','first'):
            # first stimulus RE for any output
            F = np.cumsum(F, axis=axis, dtype=M.dtype) # number of stimulus since trial start 
            F[F>1] = 0 # zero out if more than 1 stimulus since trial start
            s2estate = modifier + s2estate

        elif modifier in ('offset','last'):
            # first stimulus RE for any output
            F = np.cumsum(F, axis=axis, dtype=M.dtype) # number of stimulus since trial start 
            F[F>1] = 0 # zero out if more than 1 stimulus since trial start
            s2estate = modifier + s2estate

        # BODGE: use the state as label string if not set
        if elab is None:
            elab = s2estate


        if len(evtypes)==1:
            E = F.astype(E.dtype)
            evtlabs = elab
            s2estates = s2estate
        elif F.shape[:-1] == M.shape[:-1] and (F.shape[-1]==1 or F.shape[-1]==M.shape[-1]) :
            # single output to add
            E[..., ei] = F
            evtlabs.append(elab)
            s2estates.append(s2estate)
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
    if s2estates is None: s2estates = evtlabs
    return E, s2estates, evtlabs

def lessthan(M,val,axis,oM=None): return M<val, "<{}".format(val), None
def greaterthan(M,val,axis,oM=None): return M>val, ">{}".format(val), None

def re(M,axis,oM=None,etype=None):
    """transform stim sequence to rising edge events, which are true when the stimulus level increases

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    tmp = np.diff(M, axis=axis, prepend=0) > 0
    F = np.zeros(M.shape,dtype=M.dtype)
    F[tmp]=M[tmp] # non-zero at new larger value, but retain the old value
    return F, etype, None

def fe(M,axis,oM=None,etype=None,val=None):
    """transform stim sequence to falling edge events, which are true when the stimulus level decreases

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    tmp = np.diff(M, axis=axis, append=0) < 0
    F = np.zeros(M.shape,dtype=M.dtype)
    F[tmp]=M[tmp] # non-zero at old bigger value
    return F, etype, None

def riseto(M,val,axis,oM=None,etype=None):
    """transform stim sequence to rising edge to a given (set of) value, which are true when the stimulus level increases

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        val (list-of-int/float): the set of values to which we will test a rise to
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    if np.ndim(val)==0:
        tmp = np.diff(M==val, axis=axis, prepend=0) > 0 #flag end of rising edge
    else:
        # rise to any value in val
        tmp = np.zeros_like(M,dtype=bool)
        for v in val:
            tmp = np.logical_or(tmp,
                                np.diff(np.any(M[...,np.newaxis]==v,M.ndim), axis=axis, prepend=0) > 0)
    F = np.zeros(M.shape,dtype=bool)
    F[tmp]=True # non-zero at new larger value
    lab = 're' if val is None else 're{}'.format(val)
    return F, etype, lab

def fallfrom(M,val,axis,oM=None,etype=None):
    """transform stim sequence to falling edge from a given (set of) value, which are true when the stimulus level drops from one of the given values

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        val (list-of-int/float): the set of values to which we will test
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    if np.ndim(val)==0:
        tmp = np.diff(M==val, axis=axis, prepend=0) < 0 #flag end of falling edge
    else:
        # fall from any value in val
        tmp = np.zeros_like(M,dtype=bool)
        for v in val:
            tmp = np.logical_or(tmp,
                                np.diff(np.any(M[...,np.newaxis]==v,M.ndim), axis=axis, prepend=0) < 0)
        #tmp = np.diff(np.any(M[...,np.newaxis]==val,M.ndim),axis=axis, prepend=0) < 0
    F = np.zeros(M.shape,dtype=bool)
    F[tmp]=True # non-zero at old bigger value
    lab = 'fe' if val is None else 'fe{}'.format(val)
    return F, etype, lab

def fe2(M,axis,oM=None,etype=None):
    tmp = np.diff(M, axis=axis, prepend=0) < 0 #flag *end* of falling edge
    F = np.zeros(M.shape,dtype=M.dtype)
    F[tmp]=M[tmp] # non-zero at old bigger value
    return F, 'fe', None


def grad(M,axis,oM=None,etype=None):
    """transform stim sequence to it's gradient, i.e. first order difference = s(t)-s(t-1)
    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    F = np.diff(M,axis=axis, append=0)
    return F, 'grad', None

def ave(M,axis,oM=None,etype=None):
    if not axis == M.ndim-2:
        raise ValueError("ave only for axis==-2")
    F = np.mean(M, axis=axis+1, keepdims=True)[...,np.newaxis].astype(np.float32)
    return F, 'ave', None

def maximum(M,axis,oM=None,etype=None):
    if not axis == M.ndim-2:
        raise ValueError("ave only for axis==-2")
    F = np.maximum(M, axis=axis+1, keepdims=True)[...,np.newaxis].astype(np.float32)
    return F, 'max', None

def inc(M,axis,oM=None,thresh=0,etype='inc'):
    """transform stim sequence when the value increases more than the given threshold
    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        thresh (float): threshold for increase
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    if len(etype)>3 : thresh = float(etype[3:])
    F = np.diff(M, axis=axis, append=0) > thresh
    return F, etype, None

def dec(M,axis,oM=None,thresh=0,etype='inc'):
    """transform stim sequence when the value decreases more than the given threshold
    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        thresh (float): threshold for the decrease
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    if len(etype)>3 : thresh = float(etype[3:])
    F = np.diff(M, axis=axis, append=0) < -thresh
    return F, etype, None

def diff(M,axis,oM=None,thresh=0,etype='diff'):
    """transform stim sequence when the value changes more than the given threshold either up or down
    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        thresh (float): threshold for the change
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    if len(etype)>4 : thresh = float(etype[4:])
    F = np.abs(np.diff(M, axis=axis, append=0)) >= thresh
    return F, etype, None

def cross(M,axis,oM=None,thresh=0,etype='cross'):
    """transform stim sequence when the value crosses a given threshold value
    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        thresh (float): threshold to cross to trigger the event
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    if len(etype)>5 : thresh = float(etype[5:])
    if axis==M.ndim-2:
        tmp = np.logical_and(M[...,:-1,:]<=thresh, thresh<M[...,1:,:]) 
    elif axis==M.ndim-1:
        tmp = np.logical_and(M[...,:-1]<=thresh, thresh<M[...,1:])
    else:
        raise ValueError("cross feature only for axis==-2 or axis==-1, not {}".format(axis))
    # ensure is the right size
    padshape=list(M.shape); padshape[axis] = 1
    F =  np.append(tmp,np.zeros(padshape,dtype=tmp.dtype),axis=axis)
    return F, etype, None

def hotone(M,axis,oM=None,etype=None):
    """standard hot-one encoding, i.e. boolean feature per level which is true only when that level is active

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """ 
    vals = np.unique(M)
    F = M[...,np.newaxis] == vals
    return F, vals, None

def hoton(M,axis,oM=None,etype=None):
    """hot-on encoding, i.e. hot-one encoded only for non-zero activations

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    vals = np.unique(M)
    vals = vals[vals!=0] # strip 0
    F = M[...,np.newaxis] == vals
    return F, vals, None

def hot_greaterthan(M,axis,oM=None,etype=None): 
    """event if value is greater than found value

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    vals = np.unique(M)
    if len(vals)>1: vals = vals[:-1] # strip max val with nothing >
    F = M[...,np.newaxis] > vals
    elab = tuple(">{}".format(v) for v in vals)  # labs are now the values used
    return F, elab, None

def hoton_lessthan(M,axis,oM=None,etype=None): 
    """event if value is less than found value

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """    
    vals = np.unique(M)
    if len(vals)>1: vals=vals[1:] # strip min as nothing is <min
    F = M[...,np.newaxis] < vals.reshape((1,)*M.ndim+(vals.size,))
    elab = tuple("<{}".format(v) for v in vals)  # labs are now the values used
    return F, elab, None


def hotyon(M,axis,oM=None,etype=None,vals=None):
    """hot-on encoding, i.e. hot-one encoding over outputs and levels

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """
    assert axis==-2 or axis==M.ndim-2, "Only implemented for axis=-2 currently"
    if vals is None:
        vals = np.unique(M.reshape((-1,M.shape[-1])),axis=-2)
        if np.all(vals[0,:]==0): vals=vals[1:,:] # strip all 0
    F = np.all(M[...,np.newaxis] == vals.T, -2, keepdims=True)
    return F, vals, None

def hotyon_re(M,axis,oM=None,etype=None,vals=None):
    """hot-on encoding, i.e. hot-one encoding over outputs and levels

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """
    F, vals, none = hotyon(M,axis,oM=oM,etype=etype,vals=vals)
    re_ind = np.diff(F, axis=axis, prepend=0) > 0 #flag end of rising edge
    F[~re_ind] = 0  # zero all non-re locations
    return F, vals, None


def oddeven_pattern_reversal(M,axis,oM=None,etype='oddeven_pattern_reversal', vals=None):
    """event if value changes from adjacent odd to even value or vice versa, e.g. 1->2 or 2->1

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """
    if vals is None:
        vals = np.unique(M)
        vals = vals[vals!=0] # strip 0
        maxval = np.max(vals)
        vals = np.arange(1,maxval,2, dtype=M.dtype) # max even value
    F = np.zeros(M.shape+(len(vals),),dtype=M.dtype)
    for i,v in enumerate(vals):
        print('{}) v={}'.format(i,v))
        F[...,i] = np.logical_or(equals_subarray(M,(v,v+1),axis),
                                 equals_subarray(M,(v+1,v),axis))
    return F, vals, None
    
def grad_hoton(M,axis,oM=None, etype='grad_hoton'):
    """hot-on encoded change in the stim-level

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        (F,elab): the encoded stim sequence, and it's label info
    """    
    if not oM is None and axis==-2:
        oM = oM[...,-1,:] 
    else:
        if oM: print("prev value ignored as axis is not -2")
        oM = 0 
    
    # grad along axis
    F = np.diff(M,axis=axis,prepend=oM) #grad along axis

    # hoton encode
    vals = np.unique(F)
    vals = vals[vals !=0 ] # strip 0
    F = F[...,np.newaxis] == vals.reshape((1,)*M.ndim+(vals.size,))
    
    elab = vals # tuple("dt={}".format(v) for v in vals)  # labs are now the values used
    return F, elab, None

def hoton_re(M,axis:int,oM=None,etype:str=None):
    """hot-on encoded at the rising edge of each new value

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (int): the axis along M which we compute the difference, i.e. time
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        (F,elab): the encoded stim sequence, and it's label info
    """    
    vals = np.unique(M)
    vals = vals[vals!=0] # strip 0
    hoton = M[...,np.newaxis] == vals

    tmp = np.diff(hoton, axis=axis, prepend=0) > 0 #flag end of rising edge
    F = np.zeros(hoton.shape,dtype=M.dtype)
    F[tmp]=hoton[tmp] # non-zero at new larger value
    return F, ['re{}'.format(v) for v in vals], None

def hoton_fe(M,axis,oM=None,etype=None):
    """hot-on encoded at the falling edge of each new value

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (np.ndarray): the axis along M which runs 'time'
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        (F,elab): the encoded stim sequence, and it's label info
    """    
    vals = np.unique(M)
    vals = vals[vals!=0] # strip 0
    hoton = M[...,np.newaxis] == vals

    tmp = np.diff(hoton, axis=axis, append=0) < 0 #flag *start* of falling edge
    F = np.zeros(hoton.shape,dtype=M.dtype)
    F[tmp]=hoton[tmp] # non-zero at new larger value
    return F, ['fe{}'.format(v) for v in vals], None

def output2event(M,axis=1,oM=None,etype=None):
    """ convert "outputs" in M axis==2 to "events" in M axis==3

    Args:
        M (np.ndarray): the stimulus sequence array to transform, assumed to be: (trials,samples,outputs,events)
        axis (int): the axis along M which we compute the difference, i.e. time.  Defaults to 1.
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """
    if M.ndim==3:
        labs = ["o{}".format(i) for i in range(M.shape[2])]
    else:
        labs = ["o{}.e{}".format(i,e) for i in range(M.shape[2]) for e in range(M.shape[3])]
    M = M.reshape(M.shape[:2]+(1,)+M.shape[2:]) # shift right by 1
    return M, "output2event", labs

def event2output(M,axis=1,oM=None,etype=None):
    """ convert "events" in M axis==3 to "outputs" in M axis==2

    Args:
        M (np.ndarray): the stimulus sequence array to transform, assumed to be: (trials,samples,outputs,events)
        axis (int): the axis along M which we compute the difference, i.e. time.  Defaults to 1.
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.

    Returns:
        [type]: [description]
    """
    M = M.reshape(M.shape[:2]+(M.shape[2]*M.shape[3],)+ (M.shape[4:] if M.ndim>4 else (1,))) # shift left by 1
    return M, "event2output", None


def peroutput(M,axis,oM=None,etype:str=None, Y0_is_true:bool=True):
    """unique event sequence for each output

    Args:
        M (np.ndarray): the stimulus sequence array to transform
        axis (np.ndarray): the axis along M which runs 'time'
        oM (np.ndarray): previous vales of the stimulus sequence along the time dimension, for incremental calling
        etype (str, optional): the event type string -- ignored. Defaults to None.
        Y0_is_true (bool,optional): treat the 1st entry as the 'true' sequence, and encode in the position of the matching other Y's
    """    
    assert axis==-2 or axis==M.ndim-2, "Only implemented for axis=-2 currently"
    nevent = M.shape[axis+1]-1 if Y0_is_true else M.shape[axis+1] 
    F = np.zeros(M.shape+(nevent,), dtype=M.dtype) # add event type dimension
    labs=[None]*nevent
    for ei in range(nevent):
        yi = ei+1 if Y0_is_true else ei
        F[...,yi,ei] = M[...,yi]
        labs[ei] = "Y{}".format(yi)

    if Y0_is_true: 
        # fill in events for Y0 by matching Y0 and Yi for each trial..
        sad = np.sum(np.abs(M[...,1:].astype(float) - M[...,:1].astype(float)),axis=axis) # sum abs diff for each output w.r.t. output 0
        for ti in range(F.shape[0]):
            yi = np.argmin(sad[ti,...])
            if sad[ti,yi] < M.shape[axis]*.1:
                # got a match, copy in the event info from the appropriate yi
                # N.B. don't forget to correct for 1 less entry as removed idx=0
                F[ti,...,0,:] = F[ti,...,yi+1,:]
            else:
                print("Warning -- no output matched Y0 sufficiently well!")

    return F, ["peroutput"], labs


def perattendedoutput(M,axis,oM=None,etype=None):
    """unique event sequence for each output, with an attended/non-attended distinction

    N.B. it is assumed that the 1st output is the attended output, and any other output sufficiently similar is also attended

    Args:
        M ([type]): [description]
        axis ([type]): the axis along M which runs 'time'
        oM ([type]): [description]
        etype ([type], optional): the event type string -- ignored. Defaults to None.
    """    
    assert axis==-2 or axis==M.ndim-2, "Only implemented for axis=-1 currently"
    nevent = M.shape[axis+1]-1
    F = np.zeros(M.shape+(nevent,2), dtype=M.dtype) # add event type, and attended dimensions

    # start by assuming no output is attended, so just copy into the non-attended stream
    labs1=[None]*nevent
    for ei in range(nevent):
        yi = ei+1
        F[...,yi,:,0] = M[...,1:] # all events are non-attended
        F[...,yi,ei,1] = M[...,yi] # this event is marked as attended, i.e. 1
        F[...,yi,ei,0] = 0
        labs1[ei] = "Y{}".format(yi)
    
    # create a label list:
    labs2 = ('non-att','att')

    # then identify the attended output based on similarity to output 0
    # the move it's info to the attended set, can copy this info back to output 0

    # compute sum-absolute difference, betwee 'true' and other outputs
    sad = np.sum(np.abs(M[...,1:].astype(float) - M[...,:1].astype(float)),axis=axis) # sum abs diff for each output w.r.t. output 0
    # fill in events for Y0 by matching Y0 and Yi for each trial..
    for ti in range(F.shape[0]):
        yi = np.argmin(sad[ti,...])
        if sad[ti,yi] < M.shape[axis]*.1:
            # got a match, copy in the event info from the appropriate yi back into 'true' output 0
            F[ti,...,0,:,:] = F[ti,...,yi+1,:,:]
        else:
            print("Warning -- no output matched Y0 sufficiently well!")

    # flatten and make up the label stream
    F = F.reshape(F.shape[:-2]+(-1,))
    # combine to make the labels info
    labs=[]
    for e in labs1:
        for p in labs2:
            labs.append(e+p)

    return F, ["perattendedoutput"], labs


def rewrite_levels(M,axis,oM=None,etype=None,level_dict:dict=None):
    """rewrite the levels in M to new values in level_dict

    Args:
        M ([type]): [description]
        axis ([type]): [description]
        oM ([type], optional): [description]. Defaults to None.
        etype ([type], optional): [description]. Defaults to None.
        level_dict (dict, optional): dict mapping from old levels (key) to new ones (value).  N.B. if value is not found it is set to 0. Defaults to None.
    """
    F = np.zeros_like(M)
    for k,v in level_dict.items():
        if isinstance(k,str): k=v # bodge: string key, means just use value
        F[M==k]=v
    return F, lambda M,axis,oM,etype: rewrite_levels(M,axis,oM,etype,level_dict=level_dict), None

def slice_outputs(Y_TSye, axis=1, oM=None, etype=None, output_idx=None):
    """select a sub-set of outputs from the full set

    Args:
        M (): the stim-sequence to slice from
        axis (int, optional): the time axis in M. N.B. we assume the outputs axis is axis+1. Defaults to 1.
        output_idx (_type_, optional): the indices along this axis to select. Defaults to None.

    Returns:
        (M, apply_args, evtlabs): 
              M - the stim-seq with the selected outputs
              apply_args - args to apply this transformation to new data
              evtlabs - list-of-str with the names of the new events
    """
    if output_idx is None: return Y_TSye, None, np.arange(Y_TSye.shape[axis+1])
    if not hasattr(output_idx,'__iter__'): output_idx=(output_idx,)
    assert axis in (0,1), 'Only time in 2nd dim is'
    if axis==0:
        Y_TSye=Y_TSye[:,output_idx,...]
    elif axis==1:
        Y_TSye=Y_TSye[:,:,output_idx,...] 
    return Y_TSye, output_idx, ["o{}".format(o) for o in tuple(output_idx)]


def plot_stim_encoding(Y_TSy,Y_TSye=None,evtlabs=None,fs=None,times=None,outputs=None,suptitle:str="stim encoding",plot_all_zero_events:bool=True,block:bool=False):
    """plot a stimulus encoding to debug if the stimulus encoding transformations are correct

    Args:
        Y_TSy ([type]): [description]
        Y_TSye ([type]): [description]
        evtlabs ([type], optional): [description]. Defaults to None.
        fs ([type], optional): [description]. Defaults to None.
        times ([type], optional): [description]. Defaults to None.
        outputs ([type], optional): [description]. Defaults to None.
        suptitle (str, optional): [description]. Defaults to "stim encoding".
        plot_all_zero_events (bool, optional): [description]. Defaults to True.
        block (bool, optional): [description]. Defaults to False.
    """    
    import matplotlib.pyplot as plt
    if evtlabs is None : evtlabs = np.arange(Y_TSye.shape[-1]) if Y_TSye is not None else [0]
    if outputs is None : outputs = np.arange(Y_TSy.shape[-1]) if Y_TSy is not None else []
    if times is None:
        times = np.arange(Y_TSy.shape[1])
        if fs is not None:
            times=times/fs
    yscale = max(1e-6, np.max(np.abs(Y_TSy))) if Y_TSy is not None else None
    yescale = max(1e-6, np.max(np.abs(Y_TSye))) if Y_TSye is not None else None
    ncols = 2 if Y_TSye is not None and Y_TSy is not None else 1

    if plot_all_zero_events is False:
        keep_y = np.any(Y_TSy,axis=(0,1))
        Y_TSy=Y_TSy[...,keep_y]
        keep_ye = np.any(Y_TSye,axis=(0,1,3))
        Y_TSye=Y_TSye[...,keep_ye,:]
    else:
        Y_TSy = Y_TSy.reshape(Y_TSy.shape[:2]+(-1,))
        keep_y = np.ones(Y_TSy.shape[2],dtype=bool)
        if Y_TSye is not None:
            Y_TSye = Y_TSye.reshape(Y_TSye.shape[:3]+(-1,))
            keep_ye = np.ones(Y_TSye.shape[2],dtype=bool)

    fig,ax=plt.subplots(nrows=Y_TSy.shape[0], ncols=ncols, sharex=True, sharey=True, squeeze=False)
    for ti in range(Y_TSy.shape[0]):
        plt.sca(ax[ti][0])
        if Y_TSy is not None:
            plt.plot(times, (Y_TSy[ti,...]/yscale + np.arange(Y_TSy.shape[-1])[np.newaxis,:]),'.-')
            plt.grid(True)
            plt.title('Trial#{} Y-raw (keep={})'.format(ti,np.flatnonzero(keep_y)))
            plt.xlabel('time (seconds)')
            plt.ylabel('Output+level')
        if Y_TSye is not None:
            plt.sca(ax[ti][1])
            Y_TS_ye = np.reshape(Y_TSye,Y_TSye.shape[:-2]+(-1,))
            plt.plot(times, (Y_TS_ye[ti,...]/yescale + np.arange(Y_TS_ye.shape[-1])[np.newaxis,:])/Y_TSye.shape[-1],'.-')
            plt.grid(True)
            plt.title('Trail#{} Yevt {}  (Y={})'.format(ti, evtlabs,np.flatnonzero(keep_ye)))
            plt.xlabel('time (seconds)')
            plt.ylabel('Output+Event+Level')
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.show(block=block)


def testcase():
    from mindaffectBCI.decoder.stim2event import stim2event
    # M = (samp,Y)
    M = np.array([[0, 0, 2, 1, 0, 0, 0, 0, 1, 2, 1, 1, 0],
                  [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]])

    print("Raw  :{}".format(M))

    e,_, l = stim2event(M.T, 'slice_outputs', axis=0, output_idx=[0])
    for s,v in zip(e.T,l): print("{}:{}".format(v,s))

    e,_, l = stim2event(M.T, 'hotyon_re', axis=-2)
    for s,v in zip(e.T,l): print("{}:{}".format(v,s))

    e,_,l = stim2event(M, 'rewrite_levels', level_dict={1:2,2:3})
    e2,_,l2 = stim2event(M,l)
    assert np.all(e==e2)


    e,_, _ = stim2event(M, 'flash', axis=-1);     print("flash:{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 're', axis=-1);        print("re   :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 're1,2,3', axis=-1);        print("re1,2,3   :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'fe', axis=-1);        print("fe   :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 1, axis=-1);     print("1:{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'pr1,2', axis=-1);  print("pr1,2:{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, ((0,1),(1,2)), axis=-1);  print("(0,1):{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'oddeven_pattern_reversal', axis=-1);  print("pr:{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'grad_hoton', axis=-1);  print("grad_hoton:{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'diff', axis=-1);      print("diff :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'inc', axis=-1);       print("inc  :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'dec', axis=-1);       print("dec  :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'grad', axis=-1);      print("grad :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'hotone', axis=-1);      print("hot-one :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'hotyon', axis=-1);      print("hotyon :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'hoton_re', axis=-1);      print("hoton_re :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M, 'hoton_fe', axis=-1);      print("hoton_fe :{}".format(e[0, ...].T))
    #e,_, _ = stim2event(M, 'hot2', axis=-1);      print("hot2 :{}".format(e[0, ...].T))
    #e,_, _ = stim2event(M*30, 'inc10', axis=-1);       print("inc10 :{}".format(e[0, ...].T))
    #e,_, _ = stim2event(M*30, 'dec10', axis=-1);       print("dec10 :{}".format(e[0, ...].T))
    e,_, _ = stim2event(M-1, 'cross', axis=-1);       print("cross :{}".format(e[0, ...].T))
    e,l, _ = stim2event(M, ('re', 'fe'), axis=-1); print("refe {}:{}".format(l,e[0, ...].T))
    e,l, _ = stim2event(M, ('re1', 'fe2'), axis=-1); print("re1fe2 {},:{}".format(l,e[0, ...].T))
    e,l, _ = stim2event(M, 'onsetre', axis=-1);     print("onsetre:{}".format(e[0, ...].T))
    e,l, _ = stim2event(M.T, ('re', 'fe', 'anyre'), axis=-2); print("refeanyer {}:{}".format(l,e[0, ...].T))
    e,l, _ = stim2event(M.T, ('re', 'fe', 'rest'), axis=-2); print("referest {}:{}".format(l,e[0, ...].T))
    e,l, _ = stim2event(M.T, 'ntre', axis=-2);      print("{} :{}".format(l,e[0, ...].T))


    M = (np.maximum(0,np.random.standard_normal((2,10,3)))*2).astype(int)
    # make y0 a copy of the 'true' value
    ytrue = np.random.randint(1,M.shape[-1],size=(M.shape[0],))
    for ti,yi in enumerate(ytrue):
        M[ti,:,0] = M[ti,:,yi]

    e, e2sstate, evtlabs = stim2event(M, "re1,2,3", axis=-2)
    e2, e2sstate, evtlabs2 = stim2event(M, e2sstate, axis=-2) # set re-apply with fitted evt info 
    print("diff={}".format(np.max(np.abs(e.astype(int)-e2.astype(int)))))
    plot_stim_encoding(M,e,evtlabs, plot_all_zero_events=True, block=True,suptitle='{}'.format(evtlabs))    


    e, e2sstate, evtlabs = stim2event(M, "perattendedoutput", axis=-2)
    e2, e2sstate, evtlabs2 = stim2event(M, e2sstate, axis=-2) # set re-apply with fitted evt info
    print("diff={}".format(np.max(np.abs(e.astype(int)-e2.astype(int)))))
    plot_stim_encoding(M,e,evtlabs, plot_all_zero_events=True, block=True,suptitle='{}'.format(evtlabs))    

    e, e2sstate, evtlabs = stim2event(M, "peroutput", axis=-2)
    e2, e2sstate, evtlabs2 = stim2event(M, e2sstate, axis=-2) # set re-apply with fitted evt info
    print("diff={}".format(np.max(np.abs(e.astype(int)-e2.astype(int)))))
    plot_stim_encoding(M,e,evtlabs, plot_all_zero_events=True, block=True,suptitle='{}'.format(evtlabs))    


    # test incremental calling, propogating prefix between calls
    oM= None
    e = []
    for b in range(0,M.shape[-1],2):
        bM = M[:,b:b+2]
        eb,_, _ = stim2event(bM,('re','fe'),axis=-1,oM=oM)
        e.append(eb)
        oM=bM
    e = np.concatenate(e,-2)
    print("increfe :{}".format(e[0, ...].T))

if __name__ == "__main__":
    testcase()
