#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jadref@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
pydir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(pydir,'stimulus_sequence') if os.path.isdir(os.path.join(pydir,'stimulus_sequence')) else pydir

class StimSeq :
    def __init__(self,st=None,ss=None,es=None):
        """class which holds a stimulus-sequence, i.e. an (nobjects, ntimepoints) sequence describing the state of all objects at each time point.

        Note: In real usage only the `stimSeq` property is used, all othe properties are currently unused.

        Args:
            st (list, optional): (n_timepoints,) the stimulus times in milliseconds.  N.B. *ignored*. Defaults to None.
            ss (list-of-list-of-int, optional): (n_timepoints, n_objects) stimulus code for each object at each time-point. Defaults to None.
            es ([type], optional): (n_timepoints, n_objects). the event sequence, which says what event to send to the BCI at each time point.  N.B. *ignored* Defaults to None.
        """        

        self.stimSeq     = ss if not isinstance(ss,np.ndarray) else ss.tolist()
        # (n_timepoints, n_objects) stimulus code for each time point for each stimulus
        self.stimTime_ms = st
        # (n_timepoints,) **IGNORED** time in milliseconds at which a stimulus event takes place, i.e. stimulus i is on screen from stimTime_ms[i-1]-stimTime_ms[i]
        self.eventSeq    = es
        # (n_timepints,) **IGNORED** additional event to send at each time point.
        

    def __str__(self):
        res = "#stimTimes: ";
        if not self.stimTime_ms is None:
            res += "(1," + str(len(self.stimTime_ms)) + ")\n"
            for i in range(0,len(self.stimTime_ms)-1):
                res += str(self.stimTime_ms[i]) + " "
            res+= str(self.stimTime_ms[-1]) + "\n"
        else:
            res+= "<null>\n"
        res+= "\n\n"
        res += "#stimSeq : "
        if not self.stimSeq is None:
            res += "(" + str(len(self.stimSeq[0])) + "," + str(len(self.stimSeq)) + ")\n"
            for j in range(0,len(self.stimSeq[0])):
                for i in range(len(self.stimSeq)-1):
                    res += str(self.stimSeq[i][j]) + " "                
                res+= str(self.stimSeq[-1][j]) + "\n"
        else:
            res+= "<null>\n"
        res+="\n\n"
        return res

    def plot(self,show=None,title=None, xlim=None, ylim=None):
        """plot the stimulus sequence as a multi-line plot

        Args:
            show (bool, optional): show the plot on the screen. Defaults to False.
            title (str, optional): title for the plot. Defaults to None.

        Returns:
            matplotlib.axes: axes the plot was made in
        """        
        import matplotlib.pyplot as plt
        import numpy as np
        ss = np.array(self.stimSeq)
        if ss.ndim<2 : ss=ss[np.newaxis,:]
        yscale = np.max(np.abs(ss))
        if self.stimTime_ms is not None:
            plt.plot(self.stimTime_ms, ss + yscale*np.arange(len(ss[0]))[np.newaxis,:],'.-')
        else:
            plt.plot(ss + yscale*np.arange(len(ss[0]))[np.newaxis,:],'.-')
        plt.xlabel('time (samples)')
        plt.ylabel('outputs+levels')
        if title:
            plt.title(title)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        if show is not None: 
            plt.show(block=show)
        return plt.gca()

    def is_integer(self):
        """test if the stimulus sequence is all integer states

        Returns:
            bool: integer status
        """        
        return all([ isinstance(s,int) or s.is_integer() for row in self.stimSeq for s in row ])

    def convertstimSeq2int(self,scale:float=1,force:bool=False,minval:float=None,maxval:float=None):
        """convert floating point state stimulus sequence to an integer one

        Args:
            scale (float, optional): multiplier to apply to the floating point values before conversion to int. Defaults to 1.
            force (bool, optional): if true then force conversion even if contain non-integer floating point values. Defaults to False.
            minval (float, optional): minval to convert, i.e. this value == 1. Defaults to None.
            maxval (float, optional): maxvalue to convert, i.e. this value = maxval/scale. Defaults to None.
        """
        if force or self.is_integer():
            self.stimSeq = self.float2int(self.stimSeq, scale=scale, minval=minval, maxval=maxval)

    def convertstimSeq2float(self,scale=1,force=False,minval=None,maxval=None):
        """convert integer state stimulus sequence to an floating point one

        Args:
            scale (float, optional): multiplier to apply to the floating point values before conversion to int. Defaults to 1.
            force (bool, optional): if true then force conversion even if contain non-integer floating point values. Defaults to False.
            minval (float, optional): minval to convert, i.e. this value == 1. Defaults to None.
            maxval (float, optional): maxvalue to convert, i.e. this value = maxval/scale. Defaults to None.
        """
        if force or self.is_integer():
            self.stimSeq = self.int2float(self.stimSeq,scale,minval,maxval)

    @staticmethod
    def float2int(stimSeq,scale=1,minval=None,maxval=None):
        """convert float list of lists to integer

        Args:
            stimSeq (list-of-list-of-float): (n_timepoints, n_object) the raw stimulus sequence
            scale (float, optional): multiplier to apply to the floating point values before conversion to int. Defaults to 1.
            minval (float, optional): minval to convert, i.e. this value == 1. Defaults to None.
            maxval (float, optional): maxvalue to convert, i.e. this value = maxval/scale. Defaults to None.

        Returns:
            (list-of-list-of-int): (n_timepoints, n_object) the integer state value version of the stimulus sequence
        """        
        for i in range(len(stimSeq)):
            for j in range(len(stimSeq[i])):
                v = int(stimSeq[i][j]*scale)
                if minval is not None: v=max(minval,v)
                if maxval is not None: v=min(minval,v)
                stimSeq[i][j] = v
        return stimSeq

    @staticmethod
    def int2float(stimSeq,scale=1,minval=None,maxval=None):
        """convert float list of lists to integer

        Args:
            stimSeq (list-of-list-of-float): (n_timepoints, n_object) the raw stimulus sequence
            scale (float, optional): multiplier to apply to the floating point values before conversion to int. Defaults to 1.
            minval (float, optional): minval to convert, i.e. this value == 1. Defaults to None.
            maxval (float, optional): maxvalue to convert, i.e. this value = maxval/scale. Defaults to None.

        Returns:
            (list-of-list-of-int): (n_timepoints, n_object) the floating point state value version of the stimulus sequence
        """        
        for i in range(len(stimSeq)):
            for j in range(len(stimSeq[i])):
                v = stimSeq[i][j]*scale
                if minval is not None: v=max(minval,v)
                if maxval is not None: v=min(minval,v)
                stimSeq[i][j] = v
        return stimSeq


    def setStimRate(self,rate):
        """rewrite the stimtimes to equal given rate in hz

        Args:
            rate ([type]): [description]
        """        
        setStimRate(self.stimTime_ms,rate)
        
    @staticmethod
    def readArray(f,width=-1):
        """read an array from a character source, such as a file stream

        Args:
            f (char-stream): character source, such as a file stream returned by open(filename,'r')
            width (int, optional): width of the loaded array, i.e. number cols.  Auto determined if <=0. Defaults to -1.

        Raises:
            Exception: if there is a problem loading the file

        Returns:
            (list-of-list-of-float): (n_timepoints, n_objects) array
        """
        array=[]
        nEmpty=0
        for line in f:
            line = line.strip()
            if len(line)==0 :
                nEmpty += 1
                if nEmpty>1 and len(array)>0 : break # double empty means end-of-array
                else: continue 
            elif line[0]=="#" : continue # comment line
            cols = line.split()
            if width<0 : width=len(line)
            elif width>0 and not len(cols) == width : 
                raise ValueError("number of columns unequal {}!={}".format(width,len(cols)))
            cols = [ float(c) for c in cols ] # convert string to numeric
            array.append(cols) # add to the stimSeq
        return array

    @staticmethod
    def fromString(f):
        """read a stimulus-sequence definition from a string

        Args:
            f (str): the string to read the array definition from

        Raises:
            Exception: if there was a conversion error

        Returns:
            (list-of-list-of-float): the loaded array
        """  

        st=StimSeq.readArray(f) # read the stim times
        if len(st) > 1:
            raise Exception
        else:
            st = st[0]  # un-nest
        ss = StimSeq.readArray(f, len(st))  # read stim-seq - check same length
        # transpose ss to have time in the major dimension
        ss = transpose(ss)
        return StimSeq(st, ss)

    @staticmethod
    def fromFile(fname):
        """read a stimulus-sequence from a file on disk

        Args:
            fname (str): the file name to load the array from

        Returns:
            (list-of-list-of-float): the loaded array
        """  

        import os.path
        if not os.path.isfile(fname):
            pydir = os.path.dirname(os.path.abspath(__file__))
            #print('pydir={}'.format(pydir))
            if os.path.isfile(os.path.join(pydir, fname)):
                fname = os.path.join(pydir, fname)  
            elif  os.path.isfile(os.path.join(pydir, 'stimulus_sequence', fname)):
                fname = os.path.join(pydir, 'stimulus_sequence', fname)
        if fname.endswith('.png'):
            try:
                # try to load from the png
                from matplotlib.pyplot import imread
                array = imread(fname)
                if array.ndim > 2:
                    array = array[:, :, :3].mean(-1)
                array = array.tolist()
                ss = StimSeq(None, array, None)
                return ss
            except:
                pass

        f = open(fname, 'r')
        ss = StimSeq.fromString(f)
        return ss

    def toFile(self, fname, comment:str=None):
        """write the current stimulus-sequence to a text file format or a .png graphics file.

        Args:
            fname (str): the file name to save the textual description of the stimulus sequence to
        """ 
        print("Saving to: {}".format(fname))
        if '.png' in fname:
            # write out as a .png file
            # convert to byte with range 0-255
            import numpy as np
            import matplotlib
            array = np.array(self.stimSeq)
            array = array - np.min(array.ravel()) # start at 0
            array = array * 255 / np.max(array.ravel()) # rescale to png range
            array = array.astype(np.uint8) # convert to int
            array = np.tile(array[:,:,np.newaxis],(1,1,3)) # convert to rgb
            print("{}".format(array.shape))
            matplotlib.image.imsave(fname,array,dpi=1)

        else:
            with open(fname,'w') as f:
                # ensure has stimTimes
                if self.stimTime_ms is None:
                    self.stimTime_ms = list(range(len(self.stimSeq)))
                # add header comment
                if comment is not None:
                    f.write("# {}\n".format(comment))
                f.write(str(self))

def transpose(M):
    """swap the rows and colums of the input array

    Args:
        M (list-of-list-of-?): the input array of size (i,j)

    Returns:
        (list-of-list-of-?): the transposed array of size (j,i)
    """    
    return [[row[i] for row in M] for i in range(len(M[0]))]


def setStimRate(stimTime_ms,framerate):
    """rewrite the stim-times to a new frequency

    Args:
        stimTime_ms ([type]): total stimulus duration in milliseconds
        framerate ([type]): the framerate to use

    Returns:
        (list-of-float): the stimulus time of the events
    """
    for i in range(len(stimTime_ms)):
        stimTime_ms[i] = i*1000/framerate
    return stimTime_ms

def mkRandPerm(width=5, nEvents=None, repeats=10, level:int=1):
    """make a random permutation sequence, where only 1 object is active at a time in random order

    Args:
        width (int, optional): number of objects. Defaults to 5.
        repeats (int, optional): number of repeats of the full set objects. Defaults to 10.
    """
    return mkRowCol(width=width, height=1, nEvents=nEvents, repeats=repeats, level=level)


def mkRowCol(width=5,height=5, repeats=10, nEvents=None, level:int=1):
    """Make a row-column stimulus sequence

    Args:
        width (int, optional): width of the matrix. Defaults to 5.
        height (int, optional): height of the matrix. Defaults to 5.
        repeats (int, optional): number of random row->col repeats. Defaults to 10.

    Returns:
        [StimSeq]: The generated stimulus sequence
    """    
    import numpy as np
    replen = (width if width>1 else 0) + (height if height>1 else 0)
    if nEvents is not None:
        repeats = nEvents // replen + 1
    array = np.zeros((repeats,replen,height,width))
    for ri in range(repeats):
        ei=0
        if width>1:
            for r in np.random.permutation(width):
                array[ri,ei,:,r]=level
                ei=ei+1
        if height>1:
            for c in np.random.permutation(height):
                array[ri,ei,c,:]=level
                ei=ei+1
    array = array.reshape((repeats*replen,width*height)) # (e,(w*h)) = (nEvent,nStim)
    if nEvents:
        array = array[:nEvents,:]
    return StimSeq(None,array.tolist(),None)


def mkRandLevel(ncodes=36, nEvent=400, soa=3, jitter=1, minval=0, maxval=1, nlevels=10):
    """make a random levels stimulus -- where rand level every soa frames

    Args:
        ncodes (int, optional): number different sequences to make. Defaults to 36.
        nEvent (int, optional): number of events in each sequence. Defaults to 400.
        soa (int, optional): stimulus onset asycnroncy - number of 'null' events between stim events. Defaults to 3.
        jitter (int, optional): jitter in soa between stimulus events. Defaults to 1.
        minval (int, optional): minimum stimulus event level. Defaults to 0.
        maxval (int, optional): max stimulus event level. Defaults to 1.
        nlevels (int, optional): number of levels beween min and max. Defaults to 10.

    Returns:
        [StimSeq]: The generated stimulus sequence
    """
    # compute set of levels to use
    b = minval
    a = (maxval-minval)/(nlevels-1)
    levels = np.arange(nlevels)*a + b
    # make the levels set
    return mkRandLevelSet(ncodes, nEvent, soa, jitter, levels)

def mkRandLevelSet(ncodes=36, nEvent=400, soa=3, jitter=1, levels:list=None):
    """make a random levels stimulus -- where rand level every soa frames

    Args:
        ncodes (int, optional): number different sequences to make. Defaults to 36.
        nEvent (int, optional): number of events in each sequence. Defaults to 400.
        soa (int, optional): stimulus onset asycnroncy - number of 'null' events between stim events. Defaults to 3.
        jitter (int, optional): jitter in soa between stimulus events. Defaults to 1.
        levels (ist): the set of levels to use in the generation.

    Returns:
        [StimSeq]: The generated stimulus sequence
    """    
    import numpy as np
    array = np.zeros((nEvent,ncodes),dtype=float)

    nStim = len(range(0,nEvent,soa))
    e = np.random.randint(0,len(levels),size=(nStim,ncodes))
    # map to the levels set
    e = np.array(levels)[e]

    if jitter is None or jitter==0:
        array[::soa,:] = e
    else: # jitter the soa
        idx = list(range(0,nEvent,soa))
        for ei in range(ncodes):
            jit_idx = idx + np.random.randint(0,jitter+1,size=(nStim,)) - jitter//2
            jit_idx = np.maximum(0,np.minimum(jit_idx,array.shape[0]-1))
            array[jit_idx,ei] = e[:,ei]
    return StimSeq(None,array.tolist(),None)


def mkFreqTag(period_phase=((4,0),(5,0),(6,0),(7,0),(8,0),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),(3,2),(4,2),(5,2),(6,2),(7,2),(8,2),(4,3),(5,3),(6,3),(7,3),(8,3),(5,4),(6,4),(7,4),(8,4),(6,5),(7,5),(8,5),(7,6),(8,6),(8,7)),
            nEvent=840, isbinary=True, duty_cycle=.5):
    """Generate a frequency tagging stimulus sequence

    Args:
        period_phase (list, optional): List of stimulus periods and phase offsets expressed in events. Defaults to all phases from 2-5 long.
        nEvent (int, optional): total length of the sequence to generate. (To avoid steps should be LCM of periods) Defaults to 120.
        isbinary (bool, optional): flag if we generate a binary sequence or continuous
        duty_cycle (float, optional): fraction of the cycle which the stimulus is  'on' if is binary.  Use -1 to set only to fire at phase=0. Defaults to .5
    """
    import numpy as np
    array = np.zeros((nEvent,len(period_phase)),dtype=np.float)
    times = np.arange(array.shape[0])
    for i,e in enumerate(period_phase):
        # extract desired length and phase
        l,o = e if hasattr(e,'__iter__') else (e,0)
        # generate the sequence
        if isbinary:
            s = (times + o) % l
            if duty_cycle >=0:
                s = s + np.random.uniform(-1e-3,1e-3,size=s.shape) # add some tie-breaking noise
                array[:,i] = s > (l-1)*duty_cycle
            elif duty_cycle == -1:
                array[:,i] = s == 0
        else:
            s = np.sin( 2*np.pi* ( times.astype(np.float32)+o+1e-6)/l )
            s = (s + 1) / 2  # convert to 0-1 range
            array[:,i] = s
    return StimSeq(None,array.tolist(),None)



def mkBlockRandPatternReversal(ncodes=1, nEvent=None, nSweep=1, soa=0, blockLen=10, blockLevels:list()=None, randblk=True):
    """make a block random pattern reversal stimulus
    Args:
        blockLen (int): length of each block
        blockLevels (list): list of pairs of levels (for the pattern reversal) to use in each block

    Returns:
        [StimSeq]: The generated stimulus sequence
    """    
    import numpy as np
    if nEvent is None:
        nEvent = nSweep * len(blockLevels) * ( blockLen + soa )
    array = np.zeros((nEvent,ncodes),dtype=int)
    blkSeq = np.arange(len(blockLevels))
    for c in range(array.shape[1]):
        bi = 0
        cursor = 0 # current position in this sequence
        if randblk :
            blkSeq = np.random.permutation(blkSeq)
        while cursor < array.shape[0]:
            # generate random block sequence
            blk_ss = np.zeros((blockLen,),dtype=array.dtype)
            nlvl = len(blockLevels[blkSeq[bi]])
            for li in range(nlvl):
                blk_ss[li::nlvl] = blockLevels[blkSeq[bi]][li]
            array[cursor:cursor+blk_ss.shape[0],c] = blk_ss
            cursor = cursor + blk_ss.shape[0] + soa # move the cursor on

            bi = bi + 1 
            if randblk and bi >= len(blkSeq):
                blkSeq = np.random.permutation(blkSeq)
            bi = bi % len(blkSeq) # wrap around

    return StimSeq(None,array.tolist(),None)

def mkBlockSweepPatternReversal(**kwargs):
    return mkBlockRandPatternReversal(randblk=False, **kwargs)


def mkModulatedFreqTag(soa=3, jitter=1, levels:list=[2], base_level:int=1, # modulator parameters
                      period_phase=((4,0),(5,0),(6,0),(3,1),(4,1),(5,1),(6,1),(3,2),(4,2),(5,2),(6,2),(4,3),(5,3),(6,3),(5,4),(6,4),(6,5)), nEvent=840 ): # freq-params

    """make a modulated periodic stimulus

    Args:
        soa (int, optional): stimulus onset asynchrony, i.e. number events between modulator applications. Defaults to 3.
        jitter (int, optional): jitter in the soa. Defaults to 1.
        levels (list, optional): list of levels for the modulated stimulus. Defaults to [2].
        base_level (int,optional): level for a non-modulated stimulus.  Defaults to 1.
        nEvent (int, optional): [description]. Defaults to 840.

    Returns:
        [type]: [description]
    """    
    array = np.zeros((nEvent, len(period_phase) ),dtype=int)
    for ci,(step,offset) in enumerate(period_phase):    
        idx = slice(offset,nEvent,step) # indices where freq fires
        # make the AM sequence, with same number of events
        am = mkRandLevelSet(ncodes=1, nEvent=len(array[idx,ci]), soa=soa, jitter=jitter, levels=levels).stimSeq
        am = np.array(am).ravel()
        # set the non-modulated stimuli to the background level
        am[am==0] = base_level
        # insert into the array
        array[idx,ci] = am
    return StimSeq(None,array.tolist(),None)


def mk_m_sequence(taps:list, state:list=None, nlevels:int=2, seqlen=None):
    """make an m-sequence

    Args:
        taps (list): set of taps, [a_0, a_1, ... a_n-1]
        state (list, optional): [description]. current state [r_0, r_1, ...r_n-1]. Defaults to None.
        nlevels (int, optional): number of levels in the sequence. Defaults to 2.

    Returns:
        [type]: [description]
    """
    import numpy as np
    taps = np.array(taps,dtype=int)
    if state is None:
        state = np.zeros((len(taps)),dtype=int)
        state[-1] = 1
    state = np.array(state,dtype=int)
    if len(state) < len(taps):
        state = np.concatenate((np.zeros((len(taps)-len(state)),dtype=int), state),0)

    n = max(len(taps),len(state))
    seqlen = nlevels**(n-1) if seqlen is None else min(seqlen,nlevels**(n-1))
    #print("n={} taps={} state={} seqlen={}".format(n,taps,state,seqlen))
    mseq = np.zeros((seqlen),dtype=int)
    for t in range(seqlen):
        # compute the current output
        o = sum([s*t for s,t in zip(state,taps)]) % nlevels
        mseq[t] = state[0]
        state[:-1] = state[1:] # shift back
        state[-1] = o # insert
    return mseq

def mkMSequence(taps:list, state:list=None, nlevels:int=2, seqlen=None):
    """make an m-sequence and return a stimulus sequence object

    Args:
        taps (list): _description_
        state (list, optional): _description_. Defaults to None.
        nlevels (int, optional): _description_. Defaults to 2.
        seqlen (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    ss = mk_m_sequence(taps,state,nlevels,seqlen)
    if ss.ndim<2 : ss=ss[:,np.newaxis] # ensure 2d
    return StimSeq(None,ss.tolist(),None)

def mktaps(tappos):
    """make a list with given taps at the given locations

    Args:
        tappos (list-of-int): the list of tap positions

    Returns:
        (list-of-float): a 0-1 list at the taps locations
    """
    return [1 if i in tappos else 0 for i in range(max(tappos)+1)]


def mk_gold_code(as1, as2, ncodes=64, nlevels:int=2, seqlen=400):
    """make a set of gold codes (low-cross-correlation) from a set of m-sequence (low auto-correlation) specifications 

    Args:
        ncodes (int, optional): number of codes to generate. Defaults to 64.
        as1 ([type]): taps and state for 1st m-seq. 
        as2 ([type]): taps and state for 2nd m-seq.
        nlevels ([type]): the m-level of the sequences
        seqlen ([type]): the length of the output sequence to generate
    """
    from random import randint    
    s1 = mk_m_sequence(as1[0], as1[1], nlevels, seqlen)
    s2 = mk_m_sequence(as2[0], as2[1], nlevels, seqlen)
    seq = np.zeros((seqlen, ncodes))
    for ci in range(seq.shape[-1]):
        shift = randint(0,len(s2)-1)
        for t in range(seq.shape[0]):
            seq[t,ci] = (s1[ t % len(s1)-1] + s2[ (t+shift) % len(s2)-1]) % nlevels
    return seq


def stripbadpairs(seq,thresh=.2):
    """strip pairs of objects stimulus sequence, where bad pairs are pairs of object sequences which have higher than average cross-correlation.

    Args:
        seq (list-of-list-of-int): (n_timepoints, n_objects)  the sequence to optimize
        thresh (float, optional): threshold in standard deviations for removal. Defaults to .2.

    Returns:
        list-of-list-of-int: (n_timepoints, n_objects2) the revised stimulus sequence with bad pairs removed
    """
    if hasattr(seq,'stimSeq') : seq=np.array(seq.stimSeq)
    cc = crosscorr(seq)
    print(cc)
    ccb = np.abs(cc.copy())
    np.fill_diagonal(ccb,0)
    goodidx = np.ones(cc.shape[0],dtype=bool)
    while np.max(ccb) > thresh*np.max(cc):
        badidx = np.argmax(np.max(ccb,0))
        goodidx[badidx]=False
        ccb[badidx,:]=0
        ccb[:,badidx]=0
        #print( 'badidx={}'.format(badidx) )
        #print(" cc=\n{}".format(ccb))
    ac = np.sum(np.abs(autocorr(seq,seq.shape[0]//2)),0)
    goodidx[ac>3*np.median(ac)]=False
    print("#good= {} / {}".format(np.sum(goodidx), len(goodidx)))
    gcc = cc[goodidx,:][:,goodidx]
    return seq[:,goodidx]

def id_unbalanced(seq, zero_is_level:bool=True, thresh:float=1):
    """identify unbalanced object sequences in the stimulus sequence, where unbalanced objects have an anamalouly uneven distribution of events over the object levels.

    Args:
        seq (list-of-list-of-int): (n_timepoints, n_objects)  the sequence to optimize
        thresh (float, optional): threshold in standard deviations for removal. Defaults to .2.
        zero_is_level (bool, optional): if true then treat level 0 as a level for level distribution calculation.  Otherwise level 0 events are ignored. Defaults to True.

    Returns:
        list-of-list-of-int: (n_timepoints, n_objects2) the revised stimulus sequence with bad pairs removed
    """    
    # look at the distribution over levels in each entry
    if hasattr(seq,'stimSeq') : seq=np.array(seq.stimSeq)
    levels = np.unique(seq)
    if not zero_is_level and levels[0]==0:
        levels=levels[1:]
    #print((seq[...,0][:,np.newaxis]==levels).shape)
    ent=[]
    for oi,o in enumerate(seq.T):
        hist = np.sum(o[:,np.newaxis]==levels,0)
        ent.append(hist/np.sum(hist) @ np.log(hist/np.sum(hist)).T)
        print("{:3} {} ent={}".format(oi,hist,ent[-1]))
    ent = np.array(ent)
    mu, sigma = np.nanmean(ent), np.nanstd(ent)
    print(mu,sigma)
    badidx =  np.logical_or(ent >= mu+thresh*sigma, np.isnan(ent))
    print("Unbalanced sequences {} / {}".format(np.sum(badidx),len(badidx)))
    return badidx 


def mkGoldCode(as1,as2,ncodes=8,nlevels=2,seqlen=400,stripbad:float=None):
    """make a gold code from 2 input m-sequences

    Args:
        as1 (_type_): (2-tuple of lists-of-int) taps+state for the first m-sequence generator
        as2 (_type_): (2-tuple of lists-of-int) taps+state for the first m-sequence generator
        ncodes (int, optional): number of codes (or objects) to generate. Defaults to 8.
        nlevels (int, optional): number of levels in the resulting code. Defaults to 2.
        seqlen (int, optional): the length of stimulus sequence to generate. Defaults to 400.
        stripbad (float, optional): threshold for removal of 'bad' code pairs, which have excessively high cross-correlation. Defaults to None.

    Returns:
        list-of-list-of-int: (n_timepoints, n_objects2) the revised stimulus sequence with bad pairs removed
    """    
    s = mk_gold_code(as1, as2, ncodes=ncodes, nlevels=nlevels, seqlen=seqlen)
    print(s.shape)
    if stripbad is not None:
        s = stripbadpairs(s,thresh=stripbad)
    return StimSeq(None,s,None)

# compute auto-corr properties
import numpy as np
from mindaffectBCI.decoder.utils import window_axis
def autocorr(s,l,mu=None):
    """compute the auto-correlation of a stim-sequence with itself at different time shifts

    Args:
        s (list-of-list-of-float): (n_timepoints, n_objects) the stimulus sequence
        l (int): the length of the shifted version to compare
        mu (ndarray, optional): (n_objects,) a mean value to subtract before computing the cross-correlation.  Sequence mean if None. Defaults to None.

    Returns:
        list-of-list-of-int: (n_timepoints, n_objects2) the revised stimulus sequence with bad pairs removed
    """    
    s=np.array(s)
    if s.ndim<2: s=s[:,np.newaxis]
    if mu is None:
        mu = np.mean(s,0,keepdims=True)
    s = s - mu
    s_Ttd = window_axis(s,l,axis=0)
    return np.einsum("Ttd,td->Td",s_Ttd,s[:s_Ttd.shape[1],:])

def crosscorr(s,mu=None):
    """compute the cross correlation of a object stimulus sequence with the other objects stim-sequences in the code

    Args:
        s (list-of-list-of-float): (n_timepoints, n_objects) the stimulus sequence
        mu (ndarray, optional): (n_objects,) a mean value to subtract before computing the cross-correlation.  Sequence mean if None. Defaults to None.

    Returns:
        _type_: _description_
    """    
    s = np.array(s)
    if s.ndim<2: s=s[:,np.newaxis]
    if mu is None:
        mu = np.mean(s,0,keepdims=True)
    s = s - mu
    Css = s.T @ s
    return Css

def testcase_mseq():
    """run a test of the msequence generator
    """    
    import matplotlib.pyplot as plt

    s2 = mk_m_sequence([1,0,0,0,0,1],state=[1,0,1,0,1,1],nlevels=2)
    ac2 = autocorr(s2,len(s2)//2)
    print("s2: {}".format(s2[:50]))
    print("2-level autocorr: {}".format(ac2[:30]))
    plt.subplot(321)
    plt.plot(s2)
    plt.subplot(322)
    plt.plot(ac2)
    plt.title('s2')

    a,s=[3, 0, 4],[1, 4, 4]  #[3,2,0],[0,3,0] #[1, 1, 1, 3],[3, 2, 3, 3, 4] #
    s5 = mk_m_sequence(a,state=s,nlevels=5)
    ac5 = autocorr(s5, len(s5)//2)
    print("s5: {}".format(s5[:50]))
    print("5-level autocorr: {}".format(ac5[:30]))

    plt.subplot(323)
    plt.plot(s5)
    plt.subplot(324)
    plt.plot(ac5)
    plt.title('s5')

    a,s =[3, 1, 1, 7],[4, 4, 4, 3, 6, 1, 0, 6]#[5, 1, 5, 3],[4, 2, 3, 5, 5, 3, 1, 3]#[3, 2, 4, 1],[1, 2, 7, 0, 2, 1, 7, 0] #[5, 6, 7, 5, 3, 4, 6], [5, 7, 6, 0, 5, 0, 5]# [3, 7, 5, 7], [2, 4, 3, 4, 3, 6, 1, 3]
    s8 = mk_m_sequence(a,state=s, nlevels=8, seqlen=2000)
    ac8 = autocorr(s8, len(s8)//2)
    print("s8: {}".format(s8[:50]))
    print("8-level autocorr: {}".format(ac8[:30]))

    plt.subplot(325)
    plt.plot(s8)
    plt.subplot(326)
    plt.plot(ac8)
    plt.title('s8')

    # properties between levels
    s8 = np.array(s8)
    s8b = s8[:,np.newaxis] == np.arange(8)[np.newaxis,:]  # get seq for each level
    ac = autocorr(s8b,s8b.shape[0]//2)
    cc = crosscorr(s8b)
    # raw properties
    plt.figure()
    plt.subplot(211)
    plt.plot(ac)
    plt.subplot(212)
    plt.imshow(cc)
    plt.title('levels')

    plt.show()


def plot_level_counts(Y,zero_is_level:bool=False,label:str='',show:bool=False,re:bool=False):
    """generate a plot summarizing for each object in the stimulus sequence number of times it used a particular level.

    Args:
        Y (list-of-list-of-float): (n_timepoints, n_objects) the stimulus sequence
        zero_is_level (bool, optional): treat level=0 as a valid level in computing the counts. Defaults to False.
        label (str, optional): label for the plot. Defaults to ''.
        show (bool, optional): show the plot?. Defaults to False.
    """    
    import matplotlib.pyplot as plt
    if hasattr(Y,'stimSeq'): Y=np.array(Y.stimSeq)
    Y=np.array(Y)
    if re:
        from mindaffectBCI.decoder.stim2event import stim2event
        Y_re,_,_ = stim2event(Y,evtypes='onset')
        Y=Y_re[...,0] # remove the event-dim
    levels = np.unique(Y)
    if not zero_is_level == True and levels[0]==0: levels=levels[1:]
    hist_ol = np.sum(Y.reshape((-1,Y.shape[-1]))[...,np.newaxis]==levels,0)
    print("lvl {}".format(levels))
    print(hist_ol)
    plt.imshow(hist_ol,aspect='auto')
    plt.xticks(np.arange(hist_ol.shape[-1]),labels=levels)
    plt.ylabel('Object ID')
    plt.xlabel("Level")
    plt.colorbar()
    plt.title("Event Counts: {}".format(label))
    if show is not None: plt.show(block=show)


def testcase_gold():
    """test for the gold code generator
    """    
    import matplotlib.pyplot as plt
    if True:
        as1=([6,5],[0,0,0,0,0,1])
        as2=([6,5,3,2],[0,0,0,0,0,1])
        s8 = mk_gold_code(as1=as1,as2=as2,nlevels=2,seqlen=2**(6+3),ncodes=64)
    else:
        as1=([3, 1, 1, 7],[4, 4, 4, 3, 6, 1, 0, 6])
        as2=([5, 1, 5, 3],[4, 2, 3, 5, 5, 3, 1, 3])
        s8 = mk_gold_code(as1,as2,ncodes=64,nlevels=8,seqlen=800)

    ac = autocorr(s8,s8.shape[0]//2)
    cc = crosscorr(s8)

    # raw properties
    plt.figure()
    plt.subplot(211)
    plt.plot(ac)
    plt.subplot(212)
    plt.imshow(cc)
    plt.suptitle('gold')

    # binarize and check again
    s8b = s8[:,:,np.newaxis] == np.arange(8)[np.newaxis,np.newaxis,:]  # get seq for each level
    s8b = s8b.reshape((s8b.shape[0],-1))
    ac = autocorr(s8b,s8b.shape[0]//2)
    cc = crosscorr(s8b)
    # raw properties
    plt.figure()
    plt.subplot(211)
    plt.plot(ac)
    plt.subplot(212)
    plt.imshow(cc)
    plt.suptitle('gold levels')

    plt.show()


def test_m_seq(taps,state,nlevels,seqlen,worstcasecost=.05):
    """score a m-sequence for it's auto-correlation properties / cycle length

    Args:
        taps (list-of-int): list with the length of sequence 1 to make
        state (list-of-int): list with the length of the 2nd sequence to make
        nlevels (int): the number of levels in the generated sequences
        seqlen (int): length of the sequence to generate
        worsecasecost (float, optional): additional cost penalty for the worst-case autocorrelation. Defaults to .01

    Returns:
        _type_: _description_
    """    
    seq = mk_m_sequence(taps,state=state,nlevels=nlevels,seqlen=seqlen)
    ac = autocorr(seq, len(seq)//2)
    # normalize so ac at time-point 0==1
    N = ac[0]; ac = [ a/N for a in ac]
    mu=sum(seq)/len(seq)
    if sum([abs(si-mu) for si in seq])/len(seq) < nlevels*.2:
        v = 99999999
    else:
        # score is sum squared auto-corr, so don't like large values
        v = sum([c**2 for c in ac[1:]]) # max value = len(seq)
        # worse case auto-correlation penalty
        v = v + worstcasecost*len(seq)*max([abs(c) for c in ac[1:]]) #sum([c*largevalcost*len(seq) for c in ac[1:] if c>largevalthresh])
    # penalize not using all the levels (roughly) equally
    h = np.sum(seq == np.arange(nlevels)[:,np.newaxis],1)
    n_opt = len(seq)/nlevels
    v_bal = np.sum( np.abs(h - n_opt) / n_opt ) * 30
    #print("h={} v_bal={}".format(h,v_bal))
    #import matplotlib.pyplot as plt; plt.plot(ac8); plt.title("{}".format(v)); plt.show()
    return float(v+v_bal),ac,seq


def get_rand_seq(a,s,p):
    """ generate a random multi-level sequence, using the python randint function. N.B. No guarantees on auto/cross correlation properties.

    This function is mostly used to generate the taps+state info for a m-sequence

    Args:
        a (list-of-int): list with the length of sequence 1 to make
        s (list-of-int): list with the length of the 2nd sequence to make
        p (int): the number of levels in the generated sequences

    Returns:
        (a,s): the generated random sequences
    """    
    from random import randint
    for j in range(len(a)):
        a[j] = randint(0,p-1)
    for j in range(len(s)):
        s[j] = randint(0,p-1)
    return a,s

def inc_seq(a,s,p):
    """generate the next consequetive sequence from the input sequences, i.e. by increasing (with overflow+carry) the level by 1

    This is mostly use for exhaustive search through the possible taps+state for m-sequence generators

    Args:
        a (list-of-int): list with the length of sequence 1 to make
        s (list-of-int): list with the length of the 2nd sequence to make
        p (int): the number of levels in the generated sequences

    Raises:
        StopIteration: _description_

    Returns:
        _type_: _description_
    """    
    c=1
    for i in range(len(s)):
        ic = s[i] + c
        s[i] = ic % p
        c = ic - (p - 1)
        if c<=0:
            break
    if c>0:
        for i in range(len(a)):
            ic = a[i] + c
            a[i] = ic % p
            c = ic - (p - 1)
            if c<=0:
                break
    if c>0:
        raise StopIteration
    return (a,s)

def test_rand_m_seq(a,s,p,seqlen=3000):
    """ generate the settings for a random m-sequence and score the properties of the result

    Args:
        a (list-of-int): list with the length of sequence 1 to make
        s (list-of-int): list with the length of the 2nd sequence to make
        p (int): the number of levels in the generated sequences
        seqlen (int, optional): the lenght of the sequence generated used to test it's performance. Defaults to 3000.

    Returns:
        _type_: _description_
    """    
    a,s=get_rand_seq(a,s,p)
    return test_m_seq(a,s,p,seqlen)

def optimize_m_sequence_taps(p:int, a=[0,0,0], s=None, taps=None, state=None, seqlen=None, max_iter=10000, randsearch=False, nbest:int=10, plot:bool=True):
    """find taps and state to generate pseudo-random sequences with good auto-correlation properties

    Args:
        p (int): number of levels in the m-seq
        a (list, optional): taps for the noise generator. Defaults to [0,0,0].
        s ([type], optional): state for the noise generator. Defaults to None.
        taps ([type], optional): taps for the noise generator. Defaults to None.
        state ([type], optional): state for the noise generator. Defaults to None.
        seqlen ([type], optional): length of the sequence to generate and test. Defaults to None.
        max_iter (int, optional): max number of noise states to test. Defaults to 10000.
        randsearch (bool, optional): randomly sample noise generator configurations. Defaults to False.
        nbest (int, optional): number of 'best' generators to return. Defaults to 10.
        plot (bool, optional): flag if we plot the seq auto-correlations as they are found. Defaults to True.

    Returns:
        list-of-3-tuple: list of the nbest noise generators as a 3-tuple (score,taps,state)
    """    
    import matplotlib.pyplot as plt
    #import concurrent.futures
    # search for good parameter settings:
    if plot: 
        plt.figure()
    a = list(a) if a is not None else taps
    if s is None and state is not None:  s=state
    if s is None:
        s = [0 for i in range(len(a))]
    s = list(s)

    # futures=[]
    # executor = concurrent.futures.ProcessPoolExecutor()
    # print("Submitting jobs:")
    # for i in range(max_iter):
    #     # generate random taps
    #     futures.append(executor.submit(test_rand_m_seq,a,s,p))
    #     print(".",end='',flush=True)
    # print("Done!\n")
    
    a_star = [(999999,[],[]),] * nbest # keep the best 10
    #for future in concurrent.futures.as_completed(futures):
    for i in range(max_iter):
        if randsearch:
            a,s = get_rand_seq(a,s,p)
        else:
            try:
                a,s = inc_seq(a,s,p)
            except StopIteration:
                break
        v,ac,seq = test_m_seq(a,s,p,seqlen)
        #print("\n {},{} = {}".format(a,s,v))
        #v,a,s = future.result()
        # process the results as they come in
        beti = [ i for i,(val,_,_) in enumerate(a_star) if v<=val ]
        if beti :
            seq = mk_m_sequence(a,state=s,nlevels=p,seqlen=seqlen)
            ac = autocorr(seq, len(seq)//2)
            if v < 999999:
                #print("\n{}) {},{} = {} **".format(i,a,s,v))
                if plot and i > plot and v <= min([a[0] for a in a_star]):
                    plt.clf()
                    plt.plot([a/ac[0] for a in ac],label="{}".format(a))
                    plt.title("{}) {},{} = {}".format(i,a,s,v))
                    plt.show(block=False)
                    v,ac,seq = test_m_seq(a,s,p,seqlen)

            a_star[beti[0]] = (v,a.copy(),s.copy())
        else:
            if i%100==0: print('.',end='',flush=True)
            if i%1000==0: print("{}) {},{}".format(i,a,s),end='')
            if plot and i % 100: plt.pause(.001)
    print("-----\n {}) searched".format(i))
    print("{}".format(a_star))
    return a_star

def upsample_with_jitter(seq,soa:int=4,jitter:int=0,min_soa:int=None, max_soa:int=None, latch:bool=False):
    """upsample a stim-sequence to a higher sampling rate, padding between new locations with 0

    Args:
        seq ([type]): the orginal sequence
        soa (int, optional): stimulus-onset-asynchrony, i.e. the upsampling factor. Defaults to 4.
        jitter (int, optional): jitter in the new stimuli locations. Defaults to 0.
        min_soa (int, optional): min number stim events between stimuli
        max_soa (int, optional): max number between stim events
        latch (bool,optional): if true the hold previous value until new event, if False the pad with zeros.  Defaults to False.
    """    
    stimSeq = np.array(seq.stimSeq)
    # compute the new indices
    # interval between events
    dsample = soa*np.ones((stimSeq.shape[0],),dtype=stimSeq.dtype)
    if jitter and not jitter == 0:
        dsample = dsample + np.random.randint(low=-jitter,high=jitter,size=(stimSeq.shape[0],))
    if min_soa:
        dsample = np.maximum(dsample,min_soa)
    if max_soa:
        dsample = np.minimum(dsample,max_soa)
    # event indices
    event_idx = np.cumsum(np.maximum(dsample,1)).astype(int)
    # insert in place
    ss = np.zeros((event_idx[-1]+soa,stimSeq.shape[-1]),dtype=stimSeq.dtype)
    if latch:
        for ei,(es,ee) in enumerate(zip(event_idx[:-1],event_idx[1:])):
            ss[es:ee,:] = stimSeq[ei,:]
    else:
        ss[event_idx,:] = stimSeq
    seq.stimSeq=ss.tolist()
    seq.stimTime_ms=None
    return seq


def rewrite_levels(seq,new_levels:dict):
    """rewrite the levels values in the given stim-sequence, such that stimSeq[stimSeq==old_level[i]]=new_level[i]

    Args:
        seq ([type]): [description]
        new_levels (dict): the new level values for the levels in old_levels
    """
    seq.stimSeq = [ [new_levels.get(s,s) for s in ss] for ss in seq.stimSeq ] # subistuting the matching old-levels
    return seq

def repeat_sequence(seq:StimSeq,nrepeats=4):
    """repeat the input sequence a number of times

    Args:
        seq (StimSeq): the stimulus sequence to be repeated
        nrepeats (int, optional): number of times to repeat it temporally. Defaults to 4.

    Returns:
        StimSeq: _description_
    """    
    seq.stimSeq = [s for i in range(nrepeats) for s in seq.stimSeq ]
    seq.stimTime_ms=None #TODO[]: better stimtime bits
    return seq

def interleave_objects(seq:StimSeq,nobj=4,block_size:int=1, nactive:int=1):
    """make an interleaved sequence (i.e. only 1 is active at a time) with nobj objects

    Args:
        seq (StimSeq): the input stimulus sequence
        nobj (int, optional): [description]. Defaults to 4.
        block_size (int, optional): size of blocks of single obj to do at a time. Defaults to 1.
        nactive (int,optional): number of active objects at any time.  Defaults to 1.
    """
    if not hasattr(nobj,'__iter__'):
        nobj = [i for i in range(nobj)]
    nblk = len(seq.stimSeq)//block_size
    # pre-generate the permutations of the objects
    perm = [i for _ in range(nblk) for i in np.random.permutation(len(nobj))]
    ss, act_j = [], 0
    for i in range(nblk):
        blk = seq.stimSeq[i*block_size:(i+1)*block_size]
        # get number active outputs in this block
        nact=nactive if not hasattr(nactive, '__iter__') else np.random.choice(nactive)
        active = [perm[j]%len(nobj) for j in range(act_j,act_j+nact)] # active objects
        act_j = act_j + len(active)
        for t in blk:
            sti = [0]*len(nobj) # all off
            for o in active:
                sti[o] = t[nobj[o]]  # except o is on with correct value
            ss.append( sti )
    return StimSeq(None,ss,None)


def concatenate_objects(seq1:StimSeq,seq2:StimSeq):
    """concatenate 2 sequences temporally

    Args:
        seq1 (StimSeq): the first stimulus sequence
        seq2 (StimSeq): the second stimulus sequence to concatenate behind the first one

    Returns:
        StimSeq: the temporally concatenated result
    """    
    a=seq1.stimSeq if hasattr(seq1,'stimSeq') else seq1
    b=seq2.stimSeq if hasattr(seq2,'stimSeq') else seq2
    stimSeq = [ a[i%len(a)] + b[i%len(b)] for i in range(max(len(a),len(b)))]
    return StimSeq(None,stimSeq,None)


def make_audio(nobj=4,soa=6):
    """make a stimulus sequence for audio experiments, which uses levels 1..., only has a single object active at any time, and has a given SOA

    Args:
        nobj (int, optional): number of objects in the final sequence. Defaults to 4.
        soa (int, optional): the stimulus-onset-asynchrony, i.e. the time (in events) between stimulus events. Defaults to 6.

    Returns:
        StimSeq: the final stimulus sequence
    """    
    a = StimSeq.fromFile('level9_gold.txt')
    a = rewrite_levels(a,{k:k+1 for k in range(9)}) # level 0,1..8 -> 1,2..9
    a = interleave_objects(a, nobj=nobj) # 4-obj interleave
    a = upsample_with_jitter(a, soa=soa, jitter=2) # upsample by 6 -> 10hz stimuluation
    a.plot(show=True)
    a.toFile(os.path.join(savedir,'level9_gold_{:d}obj_interleaved_soa{:d}.txt'.format(nobj,soa)))
    return a

def make_visual(nobj=8,block_size=60):
    """make a stimulus sequence for visual experiments, which uses levels 1,2, only has a single object active at any time, and interleaves the objective activation in blocks of multiple seconds

    Args:
        nobj (int, optional): number of objects in the final sequence. Defaults to 4.
        block_size (int, optional): the active object is flicking for this number of events in a row, before switching to the next active object

    Returns:
        StimSeq: the final stimulus sequence
    """    
    b = StimSeq.fromFile('mgold_0000011_0011011.txt')
    #b.plot(show=False);plt.figure()
    b = rewrite_levels(b,{0:1,1:2}) # level 0,1 -> 1,2
    #b.plot(show=False);plt.figure()
    b = repeat_sequence(b, nrepeats=10) # loop it to make it longer
    #b.plot(show=False);plt.figure()
    b = interleave_objects(b,nobj=nobj,block_size=block_size) # interleave 8 obj in blocks of 30 stimuli
    b.plot(show=True)
    b.toFile(os.path.join(savedir,'mgold_0000011_0011011_level12_{:d}obj_interleaved{:d}.txt'.format(nobj,block_size)),
             comment='{:d}obj/2levels interleaved {:d}blocks'.format(nobj,block_size))
    return b


def make_audio_visual(naudio=4,audio_soa=6,nvisual=8,visual_block_size=60):
    """make and plot a combined audio-visual stimulus sequence

    Args:
        naudio (int, optional): number of audio objects in the final sequence. Defaults to 4.
        audio_soa (int, optional): the stimulus-onset-asynchrony, i.e. the time (in events) between stimulus events. Defaults to 6.
        nvisual (int, optional): number of visual objects in the final sequence. Defaults to 4.
        visual_block_size (int, optional): the active object is flicking for this number of events in a row, before switching to the next active object

    Returns:
        StimSeq: the final stimulus sequence
    """    
    import matplotlib.pyplot as plt

    a = StimSeq.fromFile('level9_gold_{:d}obj_interleaved_soa{:d}.txt'.format(naudio,audio_soa))
    b = StimSeq.fromFile('mgold_0000011_0011011_level12_{:d}obj_interleaved{:d}.txt'.format(nvisual,visual_block_size))
    ab = concatenate_objects(a,b)
    ab.plot(show=True)
    ab.toFile(os.path.join(savedir,'{:d}obj_8lvl_{:d}x+{:d}obj_2lvl_{:d}int.txt'.format(naudio,audio_soa,nvisual,visual_block_size)),
              comment='concatenate 4obj/8level/6x-upsampled with 8obj/2level interleaved sequences')


def mkCodes():  
    """make a load of standard stimulus sequences, e.g. visual, audio, m-sequence, gold, etc.
    """
    make_visual(nobj=3,block_size=60)

    make_audio(nobj=4,soa=6)
    make_visual(nobj=8,block_size=60)
    make_audio_visual(naudio=4,audio_soa=6,nvisual=8,visual_block_size=60)

    as1=(mktaps([6,5]),[0,0,0,0,0,1])
    as2=(mktaps([6,5,3,2]),[0,0,0,0,0,1])
    gold = mkGoldCode(as1=as1,as2=as2,nlevels=2,seqlen=2**6,ncodes=64)
    lab = '{}_{}'.format("".join(str(a) for a in as1[0]),"".join(str(a) for a in as2[0])) 
    gold.plot(show=True,title='bin_gold {}'.format(lab))
    gold.toFile(os.path.join(savedir,'mgold_{}.txt'.format(lab)))

    g = interleave_objects(gold,nobj=8)
    g.plot(show=True, title='bin_gold {} interleaved 8'.format(lab))
    g.toFile(os.path.join(savedir,'mgold_{}_8obj_interleaved.txt'.format(lab)),
                comment='gold code randomly interleaved so only 1 object is on at any time')

    g = rewrite_levels(gold,{0:1,1:2})
    g.plot(show=True,title='bin gold level12 {}'.format(lab))
    g.toFile(os.path.join(savedir,'mgold_{}_level12.txt'.format(lab)),comment="binary gold code using levels 1,2 for the coding")

    g = rewrite_levels(gold,{0:1,1:2})
    g = interleave_objects(g,nobj=8)
    g.plot(show=True, title='bin gold level12 {} interleaved8'.format(lab))
    g.toFile(os.path.join(savedir,'mgold_{}_level12_8obj_interleaved.txt'.format(lab)),
                comment='level 1,2 gold code randomly interleaved so only 1 object is on at any time')

    as1=(mktaps([6,1]),[0,0,0,0,0,1])
    as2=(mktaps([6,5,2,1]),[0,0,0,0,0,1])
    gold = mkGoldCode(as1=as1,as2=as2,nlevels=2,seqlen=2**6,ncodes=64)
    lab = '{}_{}'.format("".join(str(a) for a in as1[0]),"".join(str(a) for a in as2[0]))
    gold.plot(show=True,title='bin_gold {}'.format(lab))
    gold.toFile(os.path.join(savedir,'mgold_{}.txt'.format(lab)))

    seq = mkRandPerm()
    seq = upsample_with_jitter(seq,jitter=0)
    seq.plot(show=True)

    # mft = mkModulatedFreqTag(nEvent=800, period_phase=((23,0),(29,0)), soa=4, levels=[2], base_level=1)
    # mft.plot(show=True, title='mod-freq-tag')
    # mft.toFile(os.path.join(savedir,'mod_freq_tag_1_2.txt')))


    # seq = mkRandLevelSet(ncodes=1, soa=1, jitter=0, levels=(1,2,3,4,5,6,7,8,9,10))
    # seq.plot(show=True,title='block rand pattern reversal')
    # seq.toFile(os.path.join(savedir,'rand_10level.txt',comment="random order with 10 levels"))

    # rc=mkRowCol(width=5,height=1, repeats=50)
    # rc.plot(show=True,title='row')
    # rc.toFile(os.path.join(savedir,'5randrow.txt'))

    # rc=mkRowCol(width=10,height=1, repeats=50)
    # rc.plot(show=True,title='row')
    # rc.toFile(os.path.join(savedir,'10randrow.txt'))

    # rc=mkRowCol(width=20,height=1, repeats=50)
    # rc.plot(show=True,title='row')
    # rc.toFile(os.path.join(savedir,'20randrow.txt'))


    as1=([3, 1, 1, 7],[4, 4, 4, 3, 6, 1, 0, 6])
    as2=([5, 1, 5, 3],[4, 2, 3, 5, 5, 3, 1, 3])
    seq = mkGoldCode(as1=as1,as2=as2,nlevels=8,seqlen=800,ncodes=64)
    # insert 0 before each stim with fancy list comprenhsion
    seq.stimSeq = [ i for j in seq.stimSeq for i in ([jj+1 for jj in j],[0]*len(j))]
    seq.plot(show=True, title='8level_expdist')
    seq.toFile(os.path.join(savedir,'level8_gold_0before.txt'),
                comment='8 levels gold code with 0 before each level')

    # 4 outputs / 8 levels interleaved
    seq = mkGoldCode(as1=as1,as2=as2,nlevels=8,seqlen=800,ncodes=64)
    seq = interleave_objects(seq,nobj=4)
    seq.plot(show=True, title='8level_expdist')
    seq.toFile(os.path.join(savedir,'level8_gold_4obj_interleaved.txt'),
                comment='8 levels gold code randomly interleaved so only 1 object is on at any time')


    odd = mkModulatedFreqTag(nEvent=1600, period_phase=((5,0),(5,0)), soa=6, jitter=0, levels=[2,3],base_level=1)
    odd.plot(show=True, title='oddball')
    odd.toFile(os.path.join(savedir,'nojitter_standard_target_distractor_oddball.txt'))

    # oddball sequence
    odd = mkModulatedFreqTag(nEvent=1600, period_phase=((5,0),(5,0)), soa=6, jitter=4, levels=[2,3],base_level=1)
    odd.plot(show=True, title='oddball')
    odd.toFile(os.path.join(savedir,'standard_target_distractor_oddball.txt'))


    # oddball sequence
    odd = mkModulatedFreqTag(nEvent=1600, period_phase=((5,0),(5,0)), soa=6, jitter=4, levels=[2],base_level=1)
    odd.plot(show=True, title='oddball')
    odd.toFile(os.path.join(savedir,'standard_deviant_oddball.txt'))

    quit()

    # modulated frequency tags
    for base_level in (20,):
        levels=[20]
        mft = mkModulatedFreqTag(nEvent=800, period_phase=((23,0),(29,0)), soa=4, levels=levels, base_level=base_level)
        mft.plot(show=True, title='mod-freq-tag')
        mft.toFile(os.path.join(savedir,'mod_freq_tag_{}_{}.txt'.format(base_level,levels)))

    quit()



    # block pat rev
    bpr = mkRandLevelSet(ncodes=1, soa=1, jitter=0, levels=(1,3,5,7,9,11))
    bpr.plot(show=True,title='block rand pattern reversal')
    bpr.toFile(os.path.join(savedir,'6blk_rand.txt'))

    bpr = mkBlockRandPatternReversal(ncodes=1, nSweep=1, blockLevels=[(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)])
    bpr.plot(show=True,title='block rand pattern reversal')
    bpr.toFile(os.path.join(savedir,'6blk_rand_pr.txt'))

    bpr = mkBlockSweepPatternReversal(ncodes=1, nSweep=1, blockLevels=[(1,2),(3,4),(5,6),(7,8),(9,10),(11,12)])
    bpr.plot(show=True,title='block sweep pattern reversal')
    bpr.toFile(os.path.join(savedir,'6blk_sweep_pr.txt'))
    
    levels=[]
    for i in range(1,120,2):
        levels.append((i,i+1))
    bpr = mkBlockSweepPatternReversal(ncodes=1, nSweep=1, blockLevels=levels)
    bpr.plot(show=True,title='multifocal sweep pattern reversal')
    bpr.toFile(os.path.join(savedir,'multifocal_sweep_pr.txt'))

    # test generators
    rc=mkRowCol(width=5,height=5, repeats=10)
    rc.plot(show=True,title='rc')
    rc.toFile(os.path.join(savedir,'rc5x5.png'))
    rc.toFile(os.path.join(savedir,'rc5x5.txt'))

    ssvep=mkFreqTag()
    ssvep.toFile(os.path.join(savedir,'ssvep.png'))
    ssvep.toFile(os.path.join(savedir,'ssvep.txt'))

    ssvep_cont = mkFreqTag(isbinary=False)
    ssvep.plot(show=True,title='ssvep')
    ssvep_cont.toFile(os.path.join(savedir,'ssvep_cont.png'))
    ssvep_cont.toFile(os.path.join(savedir,'ssvep_cont.txt'))

    # random integers 0-9
    level10 = mkRandLevel(maxval=9, nlevels=10)
    level10.plot(show=True,title='level10')
    level10.toFile(os.path.join(savedir,'level10.png'))
    level10.toFile(os.path.join(savedir,'level10.txt'))

    # random levels 0-1
    level11_cont = mkRandLevel(nlevels=11)
    level11_cont.plot(show=True,title='level11_cont')
    level11_cont.toFile(os.path.join(savedir,'level11_cont.png'))
    level11_cont.toFile(os.path.join(savedir,'level11_cont.txt'))

    # m-seq max-uncorr 8 levels
    level5_mseq = mkMSequence(taps=[3,2,0],state=[0,3,0],p=5,seqlen=800)
    level5_mseq.plot(show=True,title='level5_mseq')
    level5_mseq.toFile(os.path.join(savedir,'level5_mseq.txt'))

    level8_mseq = mkMSequence(taps=[3, 2, 4, 1],state=[1, 2, 7, 0, 2, 1, 7, 0],p=8,seqlen=800)
    level8_mseq.plot(show=True,title='level8_mseq')
    level8_mseq.toFile(os.path.join(savedir,'level8_mseq.txt'))


    as1=([3, 1, 1, 7],[4, 4, 4, 3, 6, 1, 0, 6])
    as2=([5, 1, 5, 3],[4, 2, 3, 5, 5, 3, 1, 3])
    seq = mkGoldCode(as1=as1,as2=as2,nlevels=8,seqlen=800,ncodes=64)
    seq.plot(show=True,title='level8_gold')
    seq.toFile(os.path.join(savedir,'level8_gold.txt'))

    seq = mkGoldCode(as1=as1,as2=as2,nlevels=8,seqlen=800,ncodes=64)
    seq.convertstimSeq2float(scale=1/8,force=True) # convert to 0-1
    seq.toFile(os.path.join(savedir,'level8_gold_01.txt'))

    as1=([3, 1, 1, 7],[4, 4, 4, 3, 6, 1, 0, 6])
    as2=([5, 1, 5, 3],[4, 2, 3, 5, 5, 3, 1, 3])
    seq = mkGoldCode(as1=as1,as2=as2,nlevels=8,seqlen=800,ncodes=64)
    idx2level = [0,3,7,15,31,63,127,249]
    seq.stimSeq = [[idx2level[i] for i in j] for j in seq.stimSeq]
    seq.toFile(os.path.join(savedir,'level8_gold_expdist.txt'))




def load_n_plotstimSeq(savefile=None):
    """load and plot a stimulus sequence from file

    Args:
        savefile (str, optional): stimulus sequence file to load. Defaults to None.
    """    
    if savefile is None:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        import os
        root = Tk()
        root.withdraw()
        savefile = askopenfilename(initialdir=os.path.dirname(os.path.abspath(__file__)),
                                    title='Chose mindaffectBCI STIMULUS SEQUENCE',
                                    filetypes=(('stimseq','*.txt'),('All','*.*')))
        root.destroy()
    ss = StimSeq.fromFile(savefile)

    ss.plot(title=savefile,show=True)

    import matplotlib.pyplot as plt
    # summary info
    ssa = np.array(ss.stimSeq)
    ac = autocorr(ssa,ssa.shape[0]//2)
    cc = crosscorr(ssa)
    # raw properties
    plt.figure()
    plt.subplot(211)
    plt.plot(ac)
    plt.subplot(212)
    plt.imshow(cc)

# testcase code
if __name__ == "__main__":

    nobj=8
    nactive=2
    nlevel=9
    a = StimSeq.fromFile('level{}_gold.txt'.format(nlevel))
    g = interleave_objects(a,nobj=8,nactive=2)
    plot_level_counts(g.stimSeq)
    g.plot(show=True, title='level{} {}obj {}interleaved'.format(nlevel,nobj,nactive))
    g.toFile(os.path.join(savedir,'level{}_{}obj_{}interleaved.txt'.format(nlevel,nobj,nactive)),
                comment='gold code randomly interleaved so only 2 object is on at any time')

    import json
    nlevels=9
    seqlen=1500
    a_star = optimize_m_sequence_taps(nlevels,a=(1,1,0,0),s=(1,0,0,0,0,0,0,0),max_iter=10000,seqlen=seqlen,randsearch=True)
    # save best 5 to text file
    with open(os.path.join(savedir,'level{}_best_mseqs.json'.format(nlevels)),'w') as f:
        json.dump(a_star,f)
    # make gold code with the best 2
    s = mk_gold_code(a_star[-1],a_star[-2],ncodes=128,nlevels=nlevels,seqlen=seqlen)
    s = stripbadpairs(s)
    s.plot(label="level{}_gold.txt".format(nlevels))
    s.toFile(os.path.join(savedir,'level{}_gold.txt'))

    quit()

    s = mkMSequence(taps=a_star[-1][0],state=a_star[-1][1],p=nlevels,seqlen=seqlen)
    s.plot(label="level{}_mseq.txt".format(nlevels))
    s.toFile(os.path.join(savedir,'level{}_mseq.txt'))

    quit()



    # import sys
    #load_n_plotstimSeq(sys.argv[1] if len(sys.argv)>1 else None)

    make_audio(nobj=8,soa=6)

    quit()


    #testcase_gold()

    # make codebooks
    #mkCodes()



    #testcase_mseq()


    ss = StimSeq.fromFile("vep_threshold.txt")
    # loading text version
    print(("gold(txt): " + str(StimSeq.fromFile("mgold_65_6532_psk_60hz.txt"))))

    # loading png version
    print(("gold(png): " + str(StimSeq.fromFile("mgold_61_6521_psk_60hz.png"))))
    
    # test writing to file
    ss = StimSeq.fromFile("mgold_65_6532_psk_60hz.txt")
    ss.toFile("mgold_65_6532_psk_60hz.png")

