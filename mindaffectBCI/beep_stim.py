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

import matplotlib.image

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

        self.stimSeq     = ss
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

    def is_integer(self):
        return all([ s.is_integer() for row in self.stimSeq for s in row ])

    def convertstimSeq2int(self,scale=1,force=False):
        """[summary]

        Args:
            scale (int, optional): [description]. Defaults to 1.
        """        
        if force or self.is_integer():
            self.stimSeq = self.float2int(self.stimSeq)

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

        if type(stimSeq[0][0]) is float :
            for i in range(len(stimSeq)):
                for j in range(len(stimSeq[i])):
                    v = int(stimSeq[i][j]*scale)
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

from random import randrange
 
 
# Function to generate random input from a list according to
# given probabilities
def randomprob(input, probability):
 
    n = len(input)
    if n != len(probability):
        return -1                   # error
 
    # construct a sum list from given probabilities
    prob_sum = [None] * n
 
    # prob_sum[i] holds sum of all probability[j] for 0 <= j <=i
    prob_sum[0] = probability[0]
    for i in range(1, n):
        prob_sum[i] = prob_sum[i - 1] + probability[i]
 
    # generate a random integer from 1 to 100
    # and check where it lies in prob_sum
    r = randrange(1, 100)
 
    # based on the comparison result, return corresponding
    # element from the input list
 
    if r <= prob_sum[0]:  # handle 0'th index separately
        return input[0]
 
    for i in range(1, n):
        if prob_sum[i - 1] < r <= prob_sum[i]:
            return input[i]
 
    return -1
 

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
            array[ri,ei,r,:]=1
            ei=ei+1
        for c in np.random.permutation(height):
            array[ri,ei,:,c]=1
            ei=ei+1
    array = array.reshape((repeats*(width+height),width*height)) # (e,(w*h)) = (nEvent,nStim)
    shape=array.shape
    print("array shape is :",shape)
    for i in range (shape[0]):
        array[i,:]=array[i,0]
    for j in range (shape[1]):
	    array[:,j]=array[:,j]/(pow(j+1,3))
    ei=0
    for j in range (shape[1]):
        for i in range (shape[0]):
           array[0:(ei*(width*height)),j]=0
           array[(ei*(width*height)):((ei+10)*(width*height)),j] = array[(ei*(width*height)):((ei+10)*(width*height)),j]
           array[((ei+10)*(width*height)):shape[0],j]=0
        ei=ei+10
        #array[:,-1]=0
    return StimSeq(None,array.tolist(),None)

def mkSingleBeep(width=1,height=1, repeats=100):
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
            array[ri,ei,r,:]=1
            ei=ei+1
        for c in np.random.permutation(height):
            array[ri,ei,:,c]=1
            ei=ei+1
    array = array.reshape((repeats*(width+height),width*height)) # (e,(w*h)) = (nEvent,nStim)
    shape=array.shape
    print("array shape is :",shape)
    for i in range (shape[0]):
        array[i,:]=array[i,0]
    for j in range (shape[1]):
	    array[:,j]=array[:,j]/(pow(j+1,3))
    ei=0
    for j in range (shape[1]):
        for i in range (shape[0]):
           array[0:(ei*(width*height)),j]=0
           array[(ei*(width*height)):((ei+10)*(width*height)),j] = array[(ei*(width*height)):((ei+10)*(width*height)),j]
           array[((ei+10)*(width*height)):shape[0],j]=0
        ei=ei+10
        #array[:,-1]=0
        array=random.shuffle(array)
    return StimSeq(None,array.tolist(),None)
	
def mkRandLevel(ncodes=36, nEvent=400, soa=3, jitter=1, minval=0, maxval=1, nlevels=10):
    """make a random levels stimulus -- where rand level every soa frames

    Args:
        width (int, optional): width of the matrix. Defaults to 5.
        height (int, optional): height of the matrix. Defaults to 5.
        repeats (int, optional): number of random row->col repeats. Defaults to 10.

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

def mkRandLevelAudio(ncodes=36, nEvent=400, soa=10, jitter=1, minval=0, maxval=1, nlevels=10):
    """make a random levels stimulus -- where rand level every soa frames

    Args:
        width (int, optional): width of the matrix. Defaults to 5.
        height (int, optional): height of the matrix. Defaults to 5.
        repeats (int, optional): number of random row->col repeats. Defaults to 10.

    Returns:
        [StimSeq]: The generated stimulus sequence
    """    
    import numpy as np
    array = np.zeros((nEvent,ncodes),dtype=float)

    b = minval
    a = (maxval-minval)/(nlevels-1)
    nStim = len(range(0,nEvent,soa))
    e = np.random.randint(0,nlevels,size=(nStim,ncodes))/(nlevels-1)
	
    for i in range (ncodes):
        for j in range (nStim):
            e[j,i] = e[j,i] * (pow(e[j,i],3))
	
    #print(e)
    #e = np.random.randint(0,nlevels,size=(nStim,ncodes)) * a + b
    e = e +b
    print(e)
    if jitter is None or jitter==0:
        array[::soa,:] = e
    else: # jitter the soa
        idx = list(range(0,nEvent,soa))
        for ei in range(ncodes):
            jit_idx = idx + np.random.randint(0,jitter+1,size=(nStim,)) - jitter//2
            jit_idx = np.maximum(0,np.minimum(jit_idx,array.shape[0]-1))
            array[jit_idx,ei] = e[:,ei]
    return StimSeq(None,array.tolist(),None)

def mkLinLevelAudio(ncodes=36, nEvent=400, soa=2, jitter=2, minval=0, maxval=1, nlevels=60):
    """make a random levels stimulus -- where rand level every soa frames

    Args:
        width (int, optional): width of the matrix. Defaults to 5.
        height (int, optional): height of the matrix. Defaults to 5.
        repeats (int, optional): number of random row->col repeats. Defaults to 10.

    Returns:
        [StimSeq]: The generated stimulus sequence
    """    
    import numpy as np
    array = np.zeros((nEvent,ncodes),dtype=float)

    b = minval
    a = (maxval-minval)/(nlevels-1)
    #print(a)
    nStim = len(range(0,nEvent,soa))
    #e = np.random.randint(0,nlevels,size=(nStim,ncodes))/(nlevels-1)
	
    #for i in range (ncodes):
        #for j in range (nStim):
            #e[j,i] = e[j,i] * (pow(e[j,i],3))
	
    #print(e)
    e = np.random.randint(0,nlevels,size=(nStim,ncodes)) * a + b
    print(e.shape)
    elin = np.linspace(0,255,60)
    elog = np.logspace(np.log2(1),np.log2(256),num=60,endpoint=True, base =2)
    print(elog)
    probs = np.ones(elog.shape)
    probs=100*probs/(len(elog))
    print(probs)
    for j in range (ncodes):
        for i in range (nStim):		
            e[i,j] = randomprob(elin,probs)/255
            #print(randomprob(elin,probs))
    #print(e)
    import matplotlib.pyplot as plt 
    #plt.plot(e)
    #plt.plot(elog)
    #plt.show()	
    #print(elin/255)
    #e = e +b
    #print(e)
    if jitter is None or jitter==0:
        array[::soa,:] = e
    else: # jitter the soa
        idx = list(range(0,nEvent,soa))
        for ei in range(ncodes):
            jit_idx = idx + np.random.randint(0,jitter+1,size=(nStim,)) - jitter//2
            jit_idx = np.maximum(0,np.minimum(jit_idx,array.shape[0]-1))
            array[jit_idx,ei] = e[:,ei]
    #print(array)
    plt.plot(array)
    plt.show()
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


def mkCodes():
    """[summary]
    """    
    # test generators
    rc=mkLinLevelAudio(ncodes=1, nEvent=400, soa=3, jitter=3, minval=0, maxval=1, nlevels=60)
	#rc mkSingleBeep()
    rc.toFile('BeepLinear.png')
    rc.toFile('BeepLinear.txt')

# testcase code
if __name__ == "__main__":
    # make codebooks
    mkCodes()
