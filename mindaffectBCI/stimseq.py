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
    stimSeq     = None # [ nEvent x nSymb ] stimulus code for each time point for each stimulus
    stimTime_ms = None # time stim i ends, i.e. stimulus i is on screen from stimTime_ms[i-1]-stimTime_ms[i]
    eventSeq    = None # events to send at each stimulus point

    def __init__(self,st=None,ss=None,es=None):
        self.stimSeq     = ss
        self.stimTime_ms = st
        self.eventSeq    = es

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

    def convertstimSeq2int(self,scale=1):
        self.stimSeq = self.float2int(self.stimSeq)

    @staticmethod
    def float2int(stimSeq,scale=1,minval=None,maxval=None):
        '''convert float list of lists to integer'''
        if type(stimSeq[0][0]) is float :
            for i in range(len(stimSeq)):
                for j in range(len(stimSeq[i])):
                    v = int(stimSeq[i][j]*scale)
                    if minval is not None: v=max(minval,v)
                    if maxval is not None: v=min(minval,v)
                    stimSeq[i][j] = v
        return stimSeq

    def setStimRate(self,rate):
        '''rewrite the stimtimes to equal given rate in hz'''
        setStimRate(self.stimTime_ms,rate)
        
    @staticmethod
    def readArray(f,width=-1):
        array=[]
        nEmpty=0
        for line in f:
            line = line.strip();
            if len(line)==0 :
                nEmpty += 1
                if nEmpty>1 and len(array)>0 : break # double empty means end-of-array
                else: continue 
            elif line[0]=="#" : continue # comment line
            cols = line.split();
            if width<0 : width=len(line)
            elif width>0 and not len(cols) == width : 
                raise Exception
            cols = [ float(c) for c in cols ] # convert string to numeric
            array.append(cols) # add to the stimSeq
        return array

    @staticmethod
    def fromString(f):
        """read a stimulus-sequence definition from a string"""
        st=StimSeq.readArray(f) # read the stim times
        if len(st) > 1:
            raise Exception
        else:
            st=st[0] # un-nest
        ss=StimSeq.readArray(f,len(st)) # read stim-seq - check same length
        # transpose ss to have time in the major dimension
        ss=transpose(ss)
        return StimSeq(st,ss)

    @staticmethod
    def fromFile(fname):
        """read a stimulus-sequence from a file on disk"""
        if '.png' in fname:
            try:
                # try to load from the png
                from matplotlib.pyplot import imread
                array = imread(fname)
                if array.ndim > 2:
                    array = array[:,:,:3].mean(-1)
                array = array.tolist()
                ss = StimSeq(None,array,None)
                return ss
            except:
                pass
        
        f = open(fname,'r')        
        ss = StimSeq.fromString(f)
        return ss

    def toFile(self, fname):
        if '.png' in fname:
            # write out as a .png file
            # convert to byte with range 0-255
            import numpy as np
            import matplotlib
            array = np.array(self.stimSeq)
            array = array * 255 / np.max(array.ravel())
            array = array.astype(np.uint8)
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
    return [[row[i] for row in M] for i in range(len(M[0]))]

def setStimRate(stimTime_ms,framerate):
    # rewrite the stim-times to a new frequency
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


def mkFreqTag(period_phase=((2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(3,2),(4,2),(5,2),(6,2),(7,2),(4,3),(5,3),(6,3),(7,3),(5,4),(6,4),(7,4),(6,5),(7,5),(7,6)),nEvent=120, isbinary=True):
    """Generate a frequency tagging stimulus sequence

    Args:
        period_phase (list, optional): List of stimulus periods and phase offsets expressed in events. Defaults to all phases from 2-5 long.
        nEvent (int, optional): total length of the sequence to generate. (To avoid steps should be LCM of periods) Defaults to 120.
        isbinary (bool, optional): flag if we generate a binary sequence or continuous
    """
    import numpy as np
    array = np.zeros((nEvent,len(period_phase)))
    for i,e in enumerate(period_phase):
        # extract desired length and phase
        l,o = e if hasattr(e,'__iter__') else (e,0)
        # generate the sequence
        s = np.cos( (np.arange(array.shape[0])+o)/l*2*np.pi )
        if isbinary:
            array[s>0,i] = 1
        else:
            array[:,i] = s
    return StimSeq(None,array.tolist(),None)

# testcase code
if __name__ == "__main__":
    # loading text version
    print(("gold(txt): " + str(StimSeq.fromFile("mgold_65_6532_psk_60hz.txt"))))

    # loading png version
    print(("gold(png): " + str(StimSeq.fromFile("mgold_61_6521_psk_60hz.png"))))
    
    # test writing to file
    ss = StimSeq.fromFile("mgold_65_6532_psk_60hz.txt")
    ss.toFile("mgold_65_6532_psk_60hz.png")

    # test generators
    rc=mkRowCol(width=5,height=5, repeats=10)
    rc.toFile('rc5x5.png')
    rc.toFile('rc5x5.txt')

    ssvep=mkFreqTag()
    ssvep.toFile('ssvep.png')
    ssvep.toFile('ssvep.txt')
