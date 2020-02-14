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

    def convertstimSeq2int(self):
        '''convert float stimstates to integer'''
        if type(self.stimSeq[0][0]) is float :
            for i in range(len(self.stimSeq)):
                for j in range(len(self.stimSeq[i])):
                    self.stimSeq[i][j]=int(self.stimSeq[i][j])
                    
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
        if len(st)>1 : raise Exception
        else: st=st[0] # un-nest
        ss=StimSeq.readArray(f,len(st)) # read stim-seq - check same length
        # transpose ss to have time in the major dimension
        ss=transpose(ss)
        return StimSeq(st,ss)

    @staticmethod
    def fromFile(fname):
        """read a stimulus-sequence from a file on disk"""
        f=open(fname,'r')
        return StimSeq.fromString(f)

def transpose(M):
    return [[row[i] for row in M] for i in range(len(M[0]))]

def setStimRate(stimTime_ms,framerate):
    # rewrite the stim-times to a new frequency
    for i in range(len(stimTime_ms)):
        stimTime_ms[i] = i*1000/framerate
    return stimTime_ms

# testcase code
if __name__ == "__main__":
    print(("gold: " + str(StimSeq.fromFile("../../resources/codebooks/mgold_65_6532_60hz.txt"))))
    
