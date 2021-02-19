#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
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



class StimSeq :
    """[summary]

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        [type]: [description]
    """    

    stimSeq     = None # [ nEvent x nSymb ] stimulus code for each time point for each stimulus
    stimTime_ms = None # time stim i ends, i.e. stimulus i is on screen from stimTime_ms[i-1]-stimTime_ms[i]
    eventSeq    = None # events to send at each stimulus point

    def __init__(self,st=None,ss=None,es=None):
        """[summary]

        Args:
            st ([type], optional): [description]. Defaults to None.
            ss ([type], optional): [description]. Defaults to None.
            es ([type], optional): [description]. Defaults to None.
        """        

        self.stimSeq     = ss if not isinstance(ss,np.ndarray) else ss.tolist()
        self.stimTime_ms = st
        self.eventSeq    = es

    def __str__(self):
        """[summary]

        Returns:
            [type]: [description]
        """        

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

    def plot(self,show=False,title=None):
        import matplotlib.pyplot as plt
        import numpy as np
        ss = np.array(self.stimSeq)
        if ss.ndim<2 : ss=ss[np.newaxis,:]
        yscale = np.max(np.abs(ss))
        if self.stimTime_ms is not None:
            plt.plot(self.stimTime_ms, ss + yscale*np.arange(len(ss[0]))[np.newaxis,:],'.-')
        else:
            plt.plot(ss + yscale*np.arange(len(ss[0]))[np.newaxis,:],'.-')
        if title:
            plt.title(title)
        if show: 
            plt.show()

    def is_integer(self):
        return all([ s.is_integer() for row in self.stimSeq for s in row ])

    def convertstimSeq2int(self,scale=1,force=False,minval=None,maxval=None):
        """[summary]

        Args:
            scale (int, optional): [description]. Defaults to 1.
        """        
        if force or self.is_integer():
            self.stimSeq = self.float2int(self.stimSeq)

    def convertstimSeq2float(self,scale=1,force=False,minval=None,maxval=None):
        """[summary]

        Args:
            scale (int, optional): [description]. Defaults to 1.
        """        
        if force or self.is_integer():
            self.stimSeq = self.int2float(self.stimSeq,scale,minval,maxval)

    @staticmethod
    def float2int(stimSeq,scale=1,minval=None,maxval=None):
        """convert float list of lists to integer

        Args:
            stimSeq ([type]): [description]
            scale (int, optional): [description]. Defaults to 1.
            minval ([type], optional): [description]. Defaults to None.
            maxval ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
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
            stimSeq ([type]): [description]
            scale (int, optional): [description]. Defaults to 1.
            minval ([type], optional): [description]. Defaults to None.
            maxval ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
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
        """[summary]

        Args:
            f ([type]): [description]
            width (int, optional): [description]. Defaults to -1.

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
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
            f ([type]): [description]

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
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
            fname ([type]): [description]

        Returns:
            [type]: [description]
        """  

        import os.path
        if not os.path.isfile(fname):
            pydir = os.path.dirname(os.path.abspath(__file__))
            print('pydir={}'.format(pydir))
            fname = os.path.join(pydir, fname) if os.path.isfile(os.path.join(pydir, fname)) else fname
        if '.png' in fname:
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

    def toFile(self, fname):
        """[summary]

        Args:
            fname ([type]): [description]
        """ 

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
                f.write(str(self))

def transpose(M):
    """[summary]

    Args:
        M ([type]): [description]

    Returns:
        [type]: [description]
    """    

    return [[row[i] for row in M] for i in range(len(M[0]))]

def setStimRate(stimTime_ms,framerate):
    """rewrite the stim-times to a new frequency

    Args:
        stimTime_ms ([type]): [description]
        framerate ([type]): [description]

    Returns:
        [type]: [description]
    """    

    for i in range(len(stimTime_ms)):
        stimTime_ms[i] = i*1000/framerate
    return stimTime_ms


def mkRowCol(width=5,height=5, repeats=10):
    """Make a row-column stimulus sequence

    Args:
        width (int, optional): width of the matrix. Defaults to 5.
        height (int, optional): height of the matrix. Defaults to 5.
        repeats (int, optional): number of random row->col repeats. Defaults to 10.

    Returns:
        [StimSeq]: The generated stimulus sequence
    """    
    import numpy as np
    array = np.zeros((repeats,width+height,width,height))
    for ri in range(repeats):
        ei=0
        for r in np.random.permutation(width):
            array[ri,ei,:,r]=1
            ei=ei+1
        for c in np.random.permutation(height):
            array[ri,ei,c,:]=1
            ei=ei+1
    array = array.reshape((repeats*(width+height),width*height)) # (e,(w*h)) = (nEvent,nStim)
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
    import numpy as np
    array = np.zeros((nEvent,ncodes),dtype=float)

    b = minval
    a = (maxval-minval)/(nlevels-1)
    nStim = len(range(0,nEvent,soa))
    e = np.random.randint(0,nlevels,size=(nStim,ncodes)) * a + b
    if jitter is None or jitter==0:
        array[::soa,:] = e
    else: # jitter the soa
        idx = list(range(0,nEvent,soa))
        for ei in range(ncodes):
            jit_idx = idx + np.random.randint(0,jitter+1,size=(nStim,)) - jitter//2
            jit_idx = np.maximum(0,np.minimum(jit_idx,array.shape[0]-1))
            array[jit_idx,ei] = e[:,ei]
    return StimSeq(None,array.tolist(),None)

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


def mkFreqTag(period_phase=((4,0),(5,0),(6,0),(7,0),(8,0),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),(3,2),(4,2),(5,2),(6,2),(7,2),(8,2),(4,3),(5,3),(6,3),(7,3),(8,3),(5,4),(6,4),(7,4),(8,4),(6,5),(7,5),(8,5),(7,6),(8,6),(8,7)),nEvent=840, isbinary=True):
    """Generate a frequency tagging stimulus sequence

    Args:
        period_phase (list, optional): List of stimulus periods and phase offsets expressed in events. Defaults to all phases from 2-5 long.
        nEvent (int, optional): total length of the sequence to generate. (To avoid steps should be LCM of periods) Defaults to 120.
        isbinary (bool, optional): flag if we generate a binary sequence or continuous
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
            s = s + np.random.uniform(-1e-3,1e-3,size=s.shape)
            array[:,i] = s > (l-1)/2
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


def mk_m_sequence(taps:list, state:list=None, p:int=2, seqlen=None):
    """make an m-sequence

    Args:
        taps (list): set of taps, [a_0, a_1, ... a_n-1]
        state (list, optional): [description]. current state [r_0, r_1, ...r_n-1]. Defaults to None.
        p (int, optional): [description]. Defaults to 2.

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
    if seqlen is None:
        seqlen = p**n-1
    else:
        seqlen = min(p**n-1,seqlen)
    mseq = np.zeros((seqlen),dtype=int)
    for t in range(seqlen):
        # compute the current output
        o = sum([s*t for s,t in zip(state,taps)]) % p
        mseq[t] = state[0]
        state[:-1] = state[1:] # shift back
        state[-1] = o # insert
    return mseq

def mkMSequence(taps:list, state:list=None, p:int=2, seqlen=None):
    ss = mk_m_sequence(taps,state,p,seqlen)
    if ss.ndim<2 : ss=ss[:,np.newaxis] # ensure 2d
    return StimSeq(None,ss.tolist(),None)

def mk_gold_code(as1, as2, ncodes=4, p=2, seqlen=800):
    """make a set of gold codes (low-cross-correlation) from a set of m-sequence (low auto-correlation) specifications 

    Args:
        ncodes (int, optional): number of codes to generate. Defaults to 8.
        as1 ([type]): taps and state for 1st m-seq. 
        as2 ([type]): taps and state for 2nd m-seq.
        p ([type]): the m-level of the sequences
        seqlen ([type]): the length of the output sequence to generate
    """
    from random import randint    
    s1 = mk_m_sequence(as1[0], as1[1], p, seqlen)
    s2 = mk_m_sequence(as2[0], as2[1], p, seqlen)
    seq = np.zeros((seqlen, ncodes))
    print(seq.shape)
    for ci in range(seq.shape[-1]):
        shift = randint(0,len(s2)-1)
        for t in range(seq.shape[0]):
            seq[t,ci] = (s1[ t % len(s1)-1] + s2[ (t+shift) % len(s2)-1]) % p
    #print(seq)
    print(seq.shape)
    ei=0
    array=seq
    shape=array.shape
    chunk_size=int(shape[0]/ncodes)
    for j in range (shape[1]):
         for i in range (shape[0]):
            array[0:(ei*(chunk_size))+60,j]=0
            array[(ei*(chunk_size)):((ei+1)*(chunk_size)),j] = array[(ei*(chunk_size)):((ei+1)*(chunk_size)),j]
            array[((ei+1)*(chunk_size)):shape[0],j]=0
         ei=ei+1
    return seq

def mkGoldCode(as1,as2,ncodes=8,p=2,seqlen=400):
    s = mk_gold_code(as1, as2, ncodes=ncodes, p=p, seqlen=seqlen)
    return StimSeq(None,s,None)

def mkGoldCodeAudio(as1,as2,ncodes=8,p=2,seqlen=400):
    s = mk_gold_code(as1, as2, ncodes=ncodes, p=p, seqlen=seqlen)
    return StimSeq(None,s,None)

# compute auto-corr properties
import numpy as np
from mindaffectBCI.decoder.utils import window_axis
def autocorr(s,l,mu=None):
    s=np.array(s)
    if s.ndim<2: s=s[:,np.newaxis]
    if mu is None:
        mu = np.mean(s,0,keepdims=True)
    s = s - mu
    s_Ttd = window_axis(s,l,axis=0)
    return np.einsum("Ttd,td->Td",s_Ttd,s[:s_Ttd.shape[1],:])

def crosscorr(s,mu=None):
    s = np.array(s)
    if s.ndim<2: s=s[:,np.newaxis]
    if mu is None:
        mu = np.mean(s,0,keepdims=True)
    s = s - mu
    Css = s.T @ s
    return Css

def testcase_mseq():
    import matplotlib.pyplot as plt

    s2 = mk_m_sequence([1,0,0,0,0,1],state=[1,0,1,0,1,1],p=2)
    ac2 = autocorr(s2,len(s2)//2)
    print("s2: {}".format(s2[:50]))
    print("2-level autocorr: {}".format(ac2[:30]))
    plt.subplot(321)
    plt.plot(s2)
    plt.subplot(322)
    plt.plot(ac2)
    plt.title('s2')

    a,s=[3, 0, 4],[1, 4, 4]  #[3,2,0],[0,3,0] #[1, 1, 1, 3],[3, 2, 3, 3, 4] #
    s5 = mk_m_sequence(a,state=s,p=5)
    ac5 = autocorr(s5, len(s5)//2)
    print("s5: {}".format(s5[:50]))
    print("5-level autocorr: {}".format(ac5[:30]))

    plt.subplot(323)
    plt.plot(s5)
    plt.subplot(324)
    plt.plot(ac5)
    plt.title('s5')

    a,s =[3, 1, 1, 7],[4, 4, 4, 3, 6, 1, 0, 6]#[5, 1, 5, 3],[4, 2, 3, 5, 5, 3, 1, 3]#[3, 2, 4, 1],[1, 2, 7, 0, 2, 1, 7, 0] #[5, 6, 7, 5, 3, 4, 6], [5, 7, 6, 0, 5, 0, 5]# [3, 7, 5, 7], [2, 4, 3, 4, 3, 6, 1, 3]
    s8 = mk_m_sequence(a,state=s, p=8, seqlen=2000)
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


def testcase_gold():
    import matplotlib.pyplot as plt
    as1=([3, 1, 1, 7],[4, 4, 4, 3, 6, 1, 0, 6])
    as2=([5, 1, 5, 3],[4, 2, 3, 5, 5, 3, 1, 3])
    s8 = mk_gold_code(as1,as2,ncodes=64,p=8,seqlen=800)
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


def test_m_seq(a,s,p,seqlen):
    s8 = mk_m_sequence(a,state=s,p=p,seqlen=seqlen)
    ac8 = autocorr(s8, len(s8)//2)
    mu=sum(s8)/len(s8)
    if sum([abs(s-mu) for s in s8])/len(s8) < p*.2:
        v = 99999999
    else:
        v = sum([abs(s) for s in ac8[1:]])/ac8[0]
    #import matplotlib.pyplot as plt; plt.plot(ac8); plt.title("{}".format(v)); plt.show()
    return v,a,s

def get_rand_seq(a,s,p):
    from random import randint
    for j in range(len(a)):
        a[j] = randint(0,p-1)
    for j in range(len(s)):
        s[j] = randint(0,p-1)
    return a,s

def inc_seq(a,s,p):
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
    a,s=get_rand_seq(a,s,p)
    return test_m_seq(a,s,p,seqlen)

def optimize_m_sequence_taps(p,a=[0,0,0],s=None, seqlen=None,max_iter=10000, randsearch=False):
    import matplotlib.pyplot as plt
    import concurrent.futures
    # search for good parameter settings:
    plt.figure()
    a = list(a)
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
    
    a_star = []
    v_star = 99999999
    #for future in concurrent.futures.as_completed(futures):
    for i in range(max_iter):
        if randsearch:
            a,s = get_rand_seq(a,s,p)
        else:
            a,s = inc_seq(a,s,p)
        v,a,s = test_m_seq(a,s,p,seqlen)
        #print("\n {},{} = {}".format(a,s,v))
        #v,a,s = future.result()
        # process the results as they come in
        if v < v_star :
            s8 = mk_m_sequence(a,state=s,p=8,seqlen=seqlen)
            ac8 = autocorr(s8, len(s8)//2)
            print("\n{}) {},{} = {} **".format(i,a,s,v))
            #plt.plot([a/ac8[0] for a in ac8],label="{}".format(a))
            #plt.title("{} = {}".format(a,v))
            #plt.pause(.001)
            a_star = (a.copy(),s.copy())
            v_star = v
        else:
            if i%100==0: print('.',end='',flush=True)
            if i%1000==0: print("{},{}".format(a,s),end='')
            #if i % 100: plt.pause(.001)
    print("-----")
    print("{} = {}".format(a_star,v_star))
    return a_star[0],a_star[1]



def mkCodes():
    """[summary]
    """    
 

    as1=([3, 1, 1, 7],[4, 4, 4, 3, 6, 1, 0, 6])
    as2=([5, 1, 5, 3],[4, 2, 3, 5, 5, 3, 1, 3])
#    level8_mseq = mkGoldCode(as1=as1,as2=as2,p=8,seqlen=800,ncodes=64)
#    level8_mseq.plot(show=True,title='level8_gold')
#    level8_mseq.toFile('level8_gold.txt')

    level8_mseq = mkGoldCode(as1=as1,as2=as2,p=8,seqlen=1200,ncodes=4)
    level8_mseq.convertstimSeq2float(scale=1/8,force=True) # convert to 0-1
    level8_mseq.toFile('AudioMultFreqGold.txt')


# testcase code
if __name__ == "__main__":
    # make codebooks
    mkCodes()

    #testcase_gold()

    #testcase_mseq()

    #[3,2,0],[0,3,0]
    #a_star, res = optimize_m_sequence_taps(8,a=(1,1,0,0),s=(1,0,0,0,0,0,0,0),max_iter=1000000,seqlen=1500,randsearch=True)

    #ss = StimSeq.fromFile("vep_threshold.txt")
    # loading text version
    #print(("gold(txt): " + str(StimSeq.fromFile("mgold_65_6532_psk_60hz.txt"))))

    # loading png version
    #print(("gold(png): " + str(StimSeq.fromFile("mgold_61_6521_psk_60hz.png"))))
    
    # test writing to file
    #ss = StimSeq.fromFile("mgold_65_6532_psk_60hz.txt")
    #ss.toFile("mgold_65_6532_psk_60hz.png")

