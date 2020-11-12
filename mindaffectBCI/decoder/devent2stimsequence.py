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
from mindaffectBCI.utopiaclient import StimulusEvent
def devent2stimSequence(devents):
    '''
    convert a set of STIMULUSEVENT messages into a stimulus-sequence array with stimulus-times as expected by the utopia RECOGNISER
    
    Inputs:
     devents - [nEvt]:UtopiaMessage list of UtopiaMessage messages
            i.e. a decoded utopia STIMULUSEVENT message, should be of type
             { msgID:'E'byte, timeStamp:int, objIDs:(nObj:byte), objState:(nObj:byte) }
    Outputs:
     Me     - (nEvt,nY :int) The extract stimulus sequence
     objIDs - (nY :byte) The set of used object IDs
     stimTimes_ms - (nEvt :int) The timestamp of each event in milliseconds
     isistimEvent - (nEp :bool) Indicator which input events are stimulus events
      
    Copyright (c) MindAffect B.V. 2018
    '''
    if devents is None:
        return (None,None,None,None)
    Me = np.zeros((len(devents), 256), dtype=int)
    stimTimes_ms = np.zeros(len(devents))
    allobjIDs = np.arange(0, 256)
    usedobjIDs = np.zeros((256), dtype=bool)
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



def upsample_stimseq(sample_ts, ss, stimulus_ts, objIDs=None, usedobjIDs=None, trlen=None):
    ''' upsample a set of timestamped stimulus states to a sample rate
     WARNING: assumes sample_ts and stimulus_ts are in *ascending* sorted order! '''
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
    samp_ts_iterator = enumerate(iter(sample_ts))
    data_i=None
    stimulus_idx=np.zeros(stimulus_ts.shape,dtype=int)
    for stim_i, stim_ts in enumerate(stimulus_ts):
        #print("{}) ts={} ".format(stim_i,stim_ts),end='')
        odata_i = data_i # store when old stim sample index
        for data_i, data_ts in samp_ts_iterator:
            #print("{}={} ".format(data_i,data_ts),end='')
            # N.B. assumes samp_ts are already interpolated!
            if data_ts > stim_ts:
                #print("*")
                break
        if data_i is None:
            raise ValueError("Ran out of data!")
        if data_i > Y.shape[0]:
            # events after end of the allowed trial length
            break
        # data_i is one sample too far?
        data_i = max(data_i-1, 0)
        # nearest index for the stim_ts
        if odata_i is not None:
            # hold the previous stimulus state until now
            Y[odata_i:data_i+1, obj_idx] = Y[odata_i, obj_idx]
        # insert the new state
        stimulus_idx[stim_i] = data_i
        Y[data_i, obj_idx] = ss[stim_i, :]
                
    return (Y, stimulus_idx)


if __name__=="__main__":
    from devent2stimsequence import devent2stimSequence
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
    trlen = int(st[-1]*.6) # make trial shorter than data
    Y,_ = upsample_stimseq(samp_ts, Me, st, oid, used_objIDs, trlen)
    print("oID  {}".format(used_objIDs))
    print("\n---\n")
    for i in range(Y.shape[0]):
        print("{:3d}) {}".format(i,Y[i,:]))
        
