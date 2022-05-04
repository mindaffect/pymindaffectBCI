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

from mindaffectBCI.utopiaclient import UtopiaMessage
import numpy as np
from mindaffectBCI.utopiaclient import StimulusEvent, NewTarget
from mindaffectBCI.decoder.utils import unwrap

def devent2stimSequence(devents):
    '''
    convert a set of STIMULUSEVENT messages into a stimulus-sequence array with stimulus-times as expected by the utopia RECOGNISER
    
    Args:
     devents (list-of-UtopiaMessage): list of UtopiaMessage messages
            i.e. a decoded utopia STIMULUSEVENT message, should be of type
             { msgID:'E'byte, timeStamp:int, objIDs:(nObj:byte), objState:(nObj:byte) }
    Returns:
     Me    (nEvt,nY, dtype=int): The extract stimulus sequence
     objIDs (nY, dtype=byte): The set of used object IDs
     stimTimes_ms (nEvt, dtype=int): The timestamp of each event in milliseconds
     isistimEvent (nEp dtype=bool): Indicator which input events are stimulus events
      
    Copyright (c) MindAffect B.V. 2018
    '''
    if devents is None:
        return (None,None,None,None)
    Me = np.zeros((len(devents), 256), dtype=int)
    stimTimes_ms = np.zeros(len(devents))
    allobjIDs = np.arange(0, 256)
    usedobjIDs = np.zeros((256), dtype=bool)
    usedobjIDs[0] = True # force objID0 is *always* included
    isstimEvent = np.zeros((len(devents)), dtype=bool)

    # loop over the messages, extracting info from stimEvents and putting into the stim-sequence
    for ei, evt in enumerate(devents):
        if not evt.msgID == StimulusEvent.msgID:
            continue
        isstimEvent[ei] = True
        # extract the stimulusState info
        timeStamp = evt.timestamp
        objIDs = evt.objIDs
        objState = evt.objState
        stimTimes_ms[ei] = timeStamp
        # hold value of used objIDs from previous time stamp
        if ei > 0 and np.any(usedobjIDs):
            Me[ei, usedobjIDs] = Me[ei-1, usedobjIDs]
        # 2) overwrite with updated state
        # N.B. indexing hack with [int,tuple,] to index correctly
        Me[ei, objIDs, ] = objState
        usedobjIDs[objIDs, ] = True
    
    # restrict to only stim-event info
    # Note: horrible integer indexing tricks to get only the elements we want..
    Me = Me[np.flatnonzero(isstimEvent)[:, np.newaxis], usedobjIDs]
    stimTimes_ms = stimTimes_ms[isstimEvent]
    objIDs = allobjIDs[usedobjIDs]
    return Me, stimTimes_ms, objIDs, isstimEvent



def upsample_stimseq(sample_ts, ss, stimulus_ts, objIDs=None, usedobjIDs=None, trlen=None, upsample_type:str='latch'):
    ''' upsample a set of timestamped stimulus states to a sample rate
     WARNING: assumes sample_ts and stimulus_ts are in *ascending* sorted order!
     
    Args:
       sample_ts (ndarray): (nsamp,) time-stamps for the samples
       ss (ndarray): (nevent,nevtype) stimulus sequence for the event types
       stimulus_ts (ndarray): (nevent) time-stamps for the stimulus events
       objIDs (ndaray) : (nevtype,) the object ids for the stimulus events
       upsample_type (str): determing how we should hold the old stim-value until a new one.  one-of:
                    'latch' - keep the previous stim_state value.
                    'latch_and_zero' - keep previous value, until sample before new value -- so always a zero before a new state. 
                    'none'/None - don't hold
    Returns:
       Y (ndarray): (nsamp,nevttype) upsampled stimulus sequence
       stimulus_idx (nevent) index into Y for each stimulus in Y
    '''
    if trlen is None:
        trlen = len(sample_ts)

    if objIDs is None:
        objIDs = list(range(ss.shape[-1]))

    if usedobjIDs is None: # use the whole Y
        usedobjIDs = objIDs
        obj_idx = slice(len(usedobjIDs))

    else: # match objIDs and to get idxs to use
        obj_idx = np.zeros((len(objIDs),), dtype=int)
        for oi, oid in enumerate(objIDs):
            obj_idx[oi] = np.flatnonzero(usedobjIDs == oid)[0]
   
    # convert to sample-rate version
    Y = np.zeros((trlen, len(usedobjIDs)), dtype=ss.dtype)
    data_i = 0
    odata_i = data_i
    stimulus_idx=np.zeros(stimulus_ts.shape,dtype=int)
    for stim_i in range(len(stimulus_ts)):
        #print("{}) ts={} ".format(stim_i,stim_ts),end='')

        # scan along the data-time stamps to find one after the stimulus time-stamp
        while data_i < len(sample_ts) and sample_ts[data_i] < stimulus_ts[stim_i]:
            data_i = data_i + 1

        # top when passed the last data sample
        if data_i >= len(sample_ts):
            break

        # check if sample time-stamps bracket the stimulus time-stamp
        if sample_ts[odata_i] < stimulus_ts[stim_i] and stimulus_ts[stim_i] <= sample_ts[data_i]:
            # stimulus is between the odata_i and data_i samples
            # hold the previous stimulus state until now
            if upsample_type == 'latch_and_zero' and odata_i<data_i-1:
                Y[odata_i:data_i-1, obj_idx] = Y[odata_i, obj_idx]
            elif upsample_type is None or upsample_type == 'none':
                pass
            else: # upsample_type == 'latch':  # fall back on latch up-sampling
                Y[odata_i:data_i, obj_idx] = Y[odata_i, obj_idx]
            # insert the new state
            stimulus_idx[stim_i] = data_i
            Y[data_i, obj_idx] = ss[stim_i, :]
            odata_i = data_i
                
    return (Y, stimulus_idx)

def strip_unused(Y):
    """
    strip unused outputs from the stimulus info in Y

    Args:
        Y (np.ndarray (time,outputs)): the full stimulus information, potentionally with many unused outputs

    Returns:
        (np.ndarray (time,used-outputs)): Y with unused outputs removed
    """    
    if Y is None: return None, np.ones((0,),dtype=bool)
    used_y = np.any(Y.reshape((-1, Y.shape[-1])), 0)
    used_y[0] = True # ensure objID=0 is always used..
    Y = Y[..., used_y]
    return Y, used_y

def devent2stimchannels(sample_ts, stimSeq, stimulus_ts=None, objIDs=None, nt_events=None):
    '''
    convert a set of STIMULUSEVENT messages into a virtual set of stimulus channels as the same sampling rate as the data
    
    Args:
     sample_ts (ndarray - nSamp,nChannels) : the sample time-stamps, or raw data matrix, with sample time-stamps in the last channel
     devents (list-of-UtopiaMessage): list of UtopiaMessage messages
            i.e. a decoded utopia STIMULUSEVENT message, should be of type
             { msgID:'E'byte, timeStamp:int, objIDs:(nObj:byte), objState:(nObj:byte) }
    Returns:
     stim_channel  (nSamp, nObjIds, dtype=int): The stimulus channel
     stim_labels (nY, dtype=byte): The labels for the stimulus channels, either objID, or new_trial
      
    Copyright (c) MindAffect B.V. 2018
    '''
    if stimulus_ts is None or isinstance(stimSeq[0],UtopiaMessage):
        messages = stimSeq
        stimSeq, stimulus_ts, objIDs, is_stimEvent = devent2stimSequence(messages)
        nt_events = [ m for m in messages if m.msgID==NewTarget.msgID ]

    if len(stimSeq)==0: # deal with no stim info case
        return np.zeros((sample_ts.shape[0],0),dtype=int), []
    stimSeq, used_y = strip_unused(stimSeq)
    usedobjIDs = objIDs[used_y]
    if sample_ts.ndim > 1 : sample_ts = sample_ts[...,-1]
    # unwrap both stim-time sequences
    sample_ts = unwrap(sample_ts.astype(np.float64))
    stimulus_ts = unwrap(stimulus_ts.astype(np.float64))
    # upsample to a stim_channel
    stim_channel, stim_idx = upsample_stimseq(sample_ts, stimSeq, stimulus_ts, upsample_type='latch')
    stim_labels = [ 'o{}'.format(i) for i in usedobjIDs ]

    # extract a new one for the new-trial events
    if len(nt_events)>0:
        nt_ts = np.array([ m.timestamp for m in nt_events ],dtype=int) # (nevent,)
        nt_seq = np.array([ 1 for _ in nt_events ], dtype=int)[:,np.newaxis] # (nevent,1)
        nt_channel, nt_idx = upsample_stimseq(sample_ts, nt_seq, nt_ts, upsample_type=None)

        # combine into merged stim-channel set
        stim_channel = np.concatenate((stim_channel,nt_channel),axis=-1)
        stim_labels = stim_labels + ['new_trial']
    
    return stim_channel, stim_labels

def get_trial_bounds(stim_channels,stim_labels,iti_evt=60):
    """given a continuous stimulus sequence try to identify trial boundaries as times with a sufficiently large gap in the stimulus information where all outputs have level 0

    Args:
        stim_channels (_type_): the stimulus seqeuence information, with shape (Trials,Samples,Outputs,...)
        stim_labels (_type_): human readable names for the outputs
        iti_evt (int, optional): minimum number of all zero samples to indicate an inter-trial gap. Defaults to 60.

    Returns:
        list-of-int: sample indices of the trial boundaries
    """    
    nt_channel = [i for i,l in enumerate(stim_labels) if l.startswith('new_target')]
    if False: #len(nt_channel)>0:
        # TODO[]: use the new_taget channel to identify trial boundaries
        pass
    elif 1:
        # fall back on identifying large gaps between stimuli
        stimidx = np.flatnonzero(np.any(stim_channels>0, axis=-1, keepdims=True))
        endtrialidx = np.diff(stimidx) > iti_evt*np.median(np.diff(stimidx))
        endtrialidx[0] = True; endtrialidx=np.concatenate((endtrialidx,[True]))
        trlidx = stimidx[endtrialidx]
    return trlidx

def find_tgt_obj(stim_channels):
    """given a stim-sequence with a set of outputs and trials, identify the target output for each trial as the one which matches best the stimulus info for the cued target, i.e. output with objID==0

    Args:
        stim_channels (ndarray): stimulus sequence with shape (Trials, Samples, Outputs, ...)

    Returns:
        list-of-int: for each trial the identifed 'target' output with non-zero stimulus 
    """    
    # score is normalized hamming distance between true-target and given object.  0=perfect, 1=all-non-zero missed
    score = np.sum(stim_channels[...,[0]] != stim_channels[...,1:],axis=-2)/max(1,np.sum(stim_channels[...,[0]]>0)) 
    score[np.sum(stim_channels[...,0],axis=1)==0] = 1 # don't match if no stimuli 
    tgti = np.argmin(score,axis=-1) # max over outputs
    # threshold and shift objID by 1
    if hasattr(tgti,'__iter__'):
        tgti = [t+1 if score[i,t]<.1 else -1 for i,t in enumerate(tgti)]
    else:
        tgti = tgti+1 if score[tgti]<.1 else -1
    return tgti

def zero_nontarget_stimevents(stim_channels):
    tgti = find_tgt_obj(stim_channels)
    # make a new stim_channels with *only* the tgt bits set to non-zero
    tgt_stim_channels = np.zeros_like(stim_channels)
    # if no target info then leave unchanged
    if np.all([t < 0 for t in tgti]):
        return tgt_stim_channels, tgti
    # if target is set anywwhere
    tgt_stim_channels[..., 0] = stim_channels[..., 0] # objID0 is always set
    for ti in range(stim_channels.shape[0]):
        tgt_stim_channels[ti, ..., tgti[ti]] = stim_channels[ti, ..., tgti[ti]]
    return tgt_stim_channels, tgti

def stimchannels2markerchannels(stim_channels, stim_labels, sep='.'):
    """convert a set of stimulus channels to a single marker channel with a unique integer marker for each stimulus channel and level

    Args:
        stim_channels (ndarray): (nsamp,nstim) the set of stimulus channels 
        stim_labels ([type]): (nstim) the names for each of the stimulus channels
        target_indicator (bool): do we add unique event codes for target/non-target status?

    Returns:
        marker_channels, marker_dict: (nsamp,nstim) integer marker channels and dict mapping markers to stim labels+values
    """
    marker_channels = np.zeros(stim_channels.shape,dtype=int)
    marker_dict = dict()
    used_marker = 0
    # loop over stim_channels
    for si in range(stim_channels.shape[-1]):
        sc = stim_channels[:,si]
        sl = stim_labels[si]
        # insert a unique maker for each unique level of this stim_channel and add to the marker_dict
        lvls = np.unique(sc)
        if len(lvls)>10:
            raise ValueError("More than 10 unique levels!!!")
        for lvl in lvls:
            if lvl==0: continue # skip level==0
            used_marker = used_marker+1
            marker_channels[sc==lvl,si]=used_marker
            lab = "{}{}l{}".format(sl,sep,lvl) if isinstance(sl,str) else "o{}{}l{}".format(sl,sep,lvl)
            marker_dict[lab] = used_marker
        # for human readability we start each new stim_channel at 10's
        used_marker = (used_marker//10) * 10 + 10 

    # identify the non target events and give unique id
    # if there is a true-target channel, with target info
    if stim_labels[0].startswith('o0') and np.any(stim_channels[:,0]>0):
        trlbounds = get_trial_bounds(stim_channels, stim_labels)
        for i in range(len(trlbounds)-1):
            trlidx = range(trlbounds[i],trlbounds[i+1])
            tgtobj = find_tgt_obj(stim_channels[trlidx,:])
            # indicator for non-target objects
            ntgtch = np.ones(marker_channels.shape[-1],dtype=bool)
            if tgtobj:
                ntgtch[tgtobj]=False
                # BODGE: weird indexing approach
            tmp = marker_channels[trlidx,:]
            tmp[:,ntgtch]=tmp[:,ntgtch]+500
            marker_channels[trlidx,:] = tmp

        tmp = dict()
        for k,v in marker_dict.items():
            tmp[k+sep+"tgt"]=v
            tmp[k+sep+"nt"]=v+500
        marker_dict=tmp

    return marker_channels, marker_dict

def devent2markerchannels(sample_ts, messages):
    """encode a message stream into a 'marker-channel' which has a unique integer for each event

    Args:
        sample_ts (ndarray - nSamp,nChannels) : the sample time-stamps, or raw data matrix, with sample time-stamps in the last channel
        messages (list-of-UtopiaMessage): list of UtopiaMessage messages
                i.e. a decoded utopia STIMULUSEVENT message, should be of type
                { msgID:'E'byte, timeStamp:int, objIDs:(nObj:byte), objState:(nObj:byte) }

    Returns:
        marker_channels (...,nmarkers): the upsampled marker channels, one for each objID + new_target
        marker_dict (dict): dict mapping from marker IDs to string names
        marker_labels (list-of-str): labels for each of the marker channels
    """    
    stim_channels, stim_labels = devent2stimchannels(sample_ts, messages)
    marker_channels, marker_dict = stimchannels2markerchannels(stim_channels, stim_labels)
    return marker_channels, marker_dict, stim_labels, stim_channels


if __name__=="__main__":
    from mindaffectBCI.decoder.devent2stimsequence import devent2stimSequence
    from mindaffectBCI.utopiaclient import StimulusEvent
    # make a periodic type stimulus sequence, period, 3,4,5
    se = [StimulusEvent(i*3, (1, 2, 3),
                        (i%3 == 0, i%4 == 0, i%5 == 0)) for i in range(100)]
    
    Me, st, oid, isse = devent2stimSequence(se)

    # print the decoded sequence
    print("oID  {}".format(oid))
    print("---")
    for i in range(Me.shape[0]):
        print("{:3.0f}) {}".format(st[i], Me[i, :]))

    # now check the upsampling function, where we sampled at 1hz
    used_objIDs = np.arange(5)
    samp_ts = np.arange(len(se)*5)

    Y,y_labels = devent2stimchannels(samp_ts, se + [NewTarget(50)])
    print("\n---\n")
    print("{:3d}) {}".format(0,y_labels))
    for i in range(Y.shape[0]):
        print("{:3d}) {}".format(i,Y[i,:]))
    
    M,m_dict = stimchannels2markerchannels(Y,y_labels)
    print("\n---\n")
    print("{}".format(m_dict))
    for i in range(Y.shape[0]):
        print("{:3d}) {}".format(i,M[i]))

    #trlen = int(st[-1]*.6) # make trial shorter than data
    Y,_ = upsample_stimseq(samp_ts, Me, st, oid, used_objIDs)#, trlen)
    print("oID  {}".format(used_objIDs))
    print("\n---\n")
    for i in range(Y.shape[0]):
        print("{:3d}) {}".format(i,Y[i,:]))
    
