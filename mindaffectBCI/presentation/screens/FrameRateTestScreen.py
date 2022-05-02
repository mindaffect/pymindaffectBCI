#!/usr/bin/env python3
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
from mindaffectBCI.presentation.screens.basic_screens import InstructionScreen

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------------------------------------------------------
class FrameRateTestScreen(InstructionScreen):
    ''' screen from testing the frame rate of the display under pyglet control '''

    testing_text = "Checking display framerate\nPlease wait"
    results_text = "Frame rate: "
    failure_text = "WARNING:\nhigh variability in frame timing detected\nyour performance may suffer\n"
    success_text = "SUCCESS:\nsufficient accuracy frame timing detected\n"
    statistics_text = "\n{:3.0f} +/-{:3.1f} [{:2.0f},{:2.0f}]\n mean +/-std [min,max]"
    closing_text = "\n Press <space> to continue."
    
    def __init__(self, window, testduration=2000, warmup_duration=1000, duration=20000, waitKey=False):
        super().__init__(window, self.testing_text, duration, waitKey)
        self.testduration = testduration
        self.warmup_duration = warmup_duration
        self.ftimes = []
        self.logtime = None
        self.log_interval = 2000
        
    def draw(self, t):
        if not self.isRunning:
            self.ftimes = []
            self.logtime = 0
        # call parent draw method
        super().draw(t)
                    
        # record the  flip timing info
        # TODO[]: use a deque to make sliding window...
        # TODO[]: plot the histogram of frame-times?
        if self.elapsed_ms() > self.warmup_duration:
            self.ftimes.append(self.window.lastfliptime)

        if self.elapsed_ms() > self.warmup_duration + self.testduration:
            if self.elapsed_ms() > self.logtime:
                self.logtime=self.elapsed_ms() + self.log_interval
                log=True
            else:
                log=False
            (medt,madt,mint,maxt) = self.analyse_ftimes(self.ftimes,log)
            # show warning if timing is too poor
            if madt > 1:
                msg=self.failure_text
            else:
                msg=self.success_text
            msg += self.statistics_text.format(medt,madt,mint,maxt)
            msg += self.closing_text
            self.set_text(msg)
            self.waitKey = True

    @staticmethod
    def analyse_ftimes(ftimes, verb=0):
        # convert to inter-frame time
        fdur = [ ftimes[i+1]-ftimes[i] for i in range(len(ftimes)-1) ]
        #print(["%d"%(int(f)) for f in fdur])
        # analyse the frame durations, in outlier robust way
        from statistics import median    
        medt=median(fdur) # median (mode?)
        madt=0; mint=999; maxt=-999; N=0;
        for dt in fdur:
            if dt > 200 : continue # skip outliers
            N=N+1
            madt += (dt-medt) if dt>medt else (medt-dt)
            mint = dt if dt<mint else mint
            maxt = dt if dt>maxt else maxt
        madt = madt/len(fdur)

        if verb>0 :
            print("Statistics: %f(%f) [%f,%f]"%(medt,madt,mint,maxt))
            try:    
                from numpy import histogram
                [hist,bins]=histogram(fdur,range(8,34,2))
                # report summary statistics to the user
                print("Histogram:",
                      "\nDuration:","\t".join("%6.4f"%((bins[i]+bins[i+1])/2) for i in range(len(bins)-1)),
                      "\nCount   :","\t".join("%6d"%t for t in hist))
            except:
                pass
        return (medt,madt,mint,maxt)



if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    window = initPyglet(width=640, height=480)
    screen = FrameRateTestScreen(window)
    run_screen(window, screen)
