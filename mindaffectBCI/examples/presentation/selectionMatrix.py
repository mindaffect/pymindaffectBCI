#!/usr/bin/env python3

# Copyright (c) 2019 MindAffect B.V.
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


# get the general noisetagging framework
import os
from mindaffectBCI.noisetag import Noisetag, PredictionPhase
from mindaffectBCI.utopiaclient import DataPacket
from mindaffectBCI.decoder.utils import search_directories_for_file

# graphic library
import pyglet
window = None
ss = None
nframe = None
isi = 1/60
drawrate = 0  # rate at which draw is called

class Screen:

    '''Screen abstract-class which draws stuff on the screen until finished'''
    def __init__(self, window):
        self.window = window

    def reset(self):
        '''reset this screen to clean state'''
        pass

    def draw(self, t):
        '''draw the display, N.B. NOT including flip!'''
        pass

    def is_done(self):
        '''test if this screen wants to quit'''
        return False






#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class InstructionScreen(Screen):
    '''Screen which shows a textual instruction for duration or until key-pressed'''
    def __init__(self, window, text, duration=5000, waitKey=True, logo="MindAffect_Logo.png"):
        super().__init__(window)
        self.t0 = None # timer for the duration
        self.duration = duration
        self.waitKey = waitKey
        self.isRunning = False
        self.isDone = False
        self.clearScreen = True
        # initialize the instructions screen
        self.instructLabel = pyglet.text.Label(x=self.window.width//2,
                                               y=self.window.height//2,
                                               anchor_x='center',
                                               anchor_y='center',
                                               font_size=24,
                                               color=(255, 255, 255, 255),
                                               multiline=True,
                                               width=int(self.window.width*.8))
        self.set_text(text)

        # add the framerate box
        self.framerate=pyglet.text.Label("", font_size=12, x=self.window.width, y=self.window.height,
                                        color=(255, 255, 255, 255),
                                        anchor_x='right', anchor_y='top')
        
        self.logo = None
        if isinstance(logo,str): # filename to load
            logo = search_directories_for_file(logo,
                                               os.path.dirname(os.path.abspath(__file__)),
                                               os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
            try:
                logo = pyglet.image.load(logo)
            except:
                logo = None
        if logo:
            logo.anchor_x, logo.anchor_y  = (logo.width,logo.height) # anchor top-right 
            self.logo = pyglet.sprite.Sprite(logo,self.window.width,self.window.height-16)
            self.logo.update(scale_x=self.window.width*.1/logo.width, 
                            scale_y=self.window.height*.1/logo.height)

    def reset(self):
        self.isRunning = False
        self.isDone = False

    def set_text(self, text):
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        self.instructLabel.begin_update()
        self.instructLabel.text=text
        self.instructLabel.end_update()

    def is_done(self):
        # check termination conditions
        if not self.isRunning:
            self.isDone = False
            return self.isDone
        if self.waitKey:
            #global last_key_press
            if self.window.last_key_press:
                self.key_press = self.window.last_key_press
                self.isDone = True
                self.window.last_key_press = None
        if self.elapsed_ms() > self.duration:
            self.isDone = True

        return self.isDone

    def elapsed_ms(self):
        return getTimeStamp()-self.t0

    def draw(self, t):
        '''Show a block of text to the user for a given duration on a blank screen'''
        if not self.isRunning:
            self.isRunning = True  # mark that we're running
            self.t0 = getTimeStamp()
        if self.clearScreen:
            self.window.clear()
        self.instructLabel.draw()

        # check if should update display
        # TODO[]: only update screen 1x / second
        global flipstats
        flipstats.update_statistics()
        self.framerate.begin_update()
        self.framerate.text = "{:4.1f} +/-{:4.1f}ms".format(flipstats.median,flipstats.sigma)
        self.framerate.end_update()
        self.framerate.draw()

        if self.logo: self.logo.draw()





#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class MenuScreen(InstructionScreen):
    '''Screen which shows a textual instruction for duration or until key-pressed'''
    def __init__(self, window, text, valid_keys):
        super().__init__(window, text, 99999999, True)
        self.menu_text = text
        self.valid_keys = valid_keys
        self.key_press = None
        #print("Menu")

    def set_message(self,message:str):
        self.set_text(self.menu_text+'\n\n\n'+message)

    def is_done(self):
        # check termination conditions
        if not self.isRunning:
            self.isDone = False
            return self.isDone

        # valid key is pressed
        global last_key_press
        if self.window.last_key_press:
            self.key_press = self.window.last_key_press
            if self.key_press in self.valid_keys:
                self.isDone = True
            self.window.last_key_press = None

        # time-out
        if self.elapsed_ms() > self.duration:
            self.isDone = True
        return self.isDone




#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class ResultsScreen(InstructionScreen):
    '''Modified instruction screen with waits for and presents calibration results'''

    waiting_text = "Waiting for performance results from decoder\n\nPlease wait"
    results_text = "Calibration Performance: %3.0f%% Correct\n\nKey to continue"
    def __init__(self, window, noisetag, duration=20000, waitKey=False):
        super().__init__(window, self.waiting_text, duration, waitKey)
        self.noisetag = noisetag
        self.pred = None

    def reset(self):
        self.noisetag.clearLastPrediction()
        self.pred = None
        super().reset()

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        if not self.isRunning:
            self.reset()
        # check for new predictions
        pred = self.noisetag.getLastPrediction()
        # update text if got predicted performance
        if pred is not None and (self.pred is None or pred.timestamp > self.pred.timestamp) :
            self.pred = pred
            print("Prediction:{}".format(self.pred))
            self.waitKey = True
            self.set_text(self.results_text%((1.0-self.pred.Perr)*100.0))
        super().draw(t)






#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class ConnectingScreen(InstructionScreen):
    '''Modified instruction screen with waits for the noisetag to connect to the decoder'''

    prefix_text = "Welcome to the mindaffectBCI\n\n"
    searching_text = "Searching for the mindaffect decoder\n\nPlease wait"
    trying_text = "Trying to connect to: %s\n Please wait"
    connected_text = "Success!\nconnected to: %s"
    query_text = "Couldnt auto-discover mindaffect decoder\n\nPlease enter decoder address: %s"
    drawconnect_timeout_ms = 50
    autoconnect_timeout_ms = 5000

    def __init__(self, window, noisetag, duration=150000):
        super().__init__(window, self.prefix_text + self.searching_text, duration, False)
        self.noisetag = noisetag
        self.host = None
        self.port = -1
        self.usertext = ''
        self.stage = 0

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        global last_text, last_key_press
        if not self.isRunning:
            super().draw(t)
            return

        if not self.noisetag.isConnected():
            if self.stage == 0: # try-connection
                print('Not connected yet!!')
                self.noisetag.connect(self.host, self.port,
                                      queryifhostnotfound=False,
                                      timeout_ms=self.drawconnect_timeout_ms)
                if self.noisetag.isConnected():
                    self.set_text(self.prefix_text + self.connected_text%(self.noisetag.gethostport()))
                    self.t0 = getTimeStamp()
                    self.duration = 1000
                    self.noisetag.subscribe("MSPQ")
                elif self.elapsed_ms() > self.autoconnect_timeout_ms:
                    # waited too long, giveup and ask user
                    self.stage = 1
                    # ensure old key-presses are gone
                    self.window.last_text = None
                    self.window.last_key_press = None

            elif self.stage == 1:  # query hostname
                # query the user for host/port
                # accumulate user inputs
                if self.window.last_key_press:
                    if self.window.last_key_press == pyglet.window.key.BACKSPACE:
                        # remove last character
                        self.usertext = self.usertext[:-1]
                    self.window.last_key_press = None
                if self.window.last_text:
                    print(self.window.last_text + ":" + str(ord(self.window.last_text)))
                    if self.window.last_text == '\n' or self.window.last_text == '\r':
                        # set as new host to try
                        self.host = self.usertext
                        self.usertext = ''
                        self.set_text(self.prefix_text + self.trying_text%(self.host))
                        self.stage = 0 # back to try-connection stage
                    elif self.window.last_text:
                        # add to the host string
                        self.usertext += last_text
                    self.window.last_text = None
                if self.stage == 1: # in same stage
                    # update display with user input
                    self.set_text(self.prefix_text + self.query_text%(self.usertext))
        super().draw(t)






#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class SettingsScreen(InstructionScreen):
    '''Modified instruction screen to change various settings - selection threshold'''

    prefix_text = "Configuration Settings\n\n"
    threshold_text = "New Selection Threshold: %s\n"

    def __init__(self, window, settings_class, duration=150000):
        super().__init__(window, self.prefix_text + self.threshold_text%(settings_class.selectionThreshold), duration, False)
        self.settings_class = settings_class
        self.usertext = ''

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        global last_text, last_key_press
        if not self.isRunning:
            super().draw(t)
            return

        # query the user for host/port
        # accumulate user inputs
        if self.window.last_key_press:
            if self.window.last_key_press == pyglet.window.key.BACKSPACE:
                # remove last character
                self.usertext = self.usertext[:-1]
            self.window.last_key_press = None
            if self.window.last_text:
                print(self.window.last_text + ":" + str(ord(self.window.last_text)))
            if self.window.last_text == '\n' or self.window.last_text == '\r':
                # set as new host to try
                try:
                    self.threshold = float(self.usertext)
                    self.settings_class.selectionThreshold=self.threshold
                    self.isDone = True
                except ValueError:
                    # todo: flash to indicate invalid..
                    pass
            elif self.window.last_text and self.window.last_text in "0123456789.":
                # add to the host string
                self.usertext += window.last_text
            self.window.last_text = None
            self.set_text(self.prefix_text + self.threshold_text%(self.usertext))
        super().draw(t)


#-----------------------------------------------------------------
class QueryDialogScreen(InstructionScreen):
    '''Modified instruction screen queries the user for textual input'''

    def __init__(self, window, text, duration=50000, waitKey=True):
        super().__init__(window, text, 50000, False)
        self.query = text
        self.usertext = ''

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        # query the user for host/port
        # accumulate user inputs
        global last_key_press, last_text

        if self.window.last_key_press:
            if self.window.last_key_press == pyglet.window.key.BACKSPACE:
                self.usertext = self.usertext[:-1]
                self.set_text(self.query +self.usertext)
            self.window.last_key_press = None
        if self.window.last_text:
            if self.window.last_text == '\r' or self.window.last_text == '\n':
                self.isDone = True
            elif self.window.last_text:
                # add to the host string
                self.usertext += self.window.last_text
            self.window.last_text=None
            # update display with user input
            self.set_text(self.query +self.usertext)
        super().draw(t)




#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
from math import log10
from collections import deque
class ElectrodequalityScreen(Screen):
    '''Screen which shows the electrode signal quality information'''

    instruct = "Electrode Quality\n\nAdjust headset until all electrodes are green\n(or noise to signal ratio < 5)"
    def __init__(self, window, noisetag, nch=4, duration=3600*1000, waitKey=True):
        super().__init__(window)
        self.noisetag = noisetag
        self.t0 = None  # timer for the duration
        self.duration = duration
        self.waitKey = waitKey
        self.clearScreen = True
        self.isRunning = False
        self.update_nch(nch)
        self.dataringbuffer = deque()  # deque so efficient sliding data window
        self.datawindow_ms = 4000  # 5seconds data plotted
        self.datascale_uv = 20  # scale of gap between ch plots
        print("Electrode Quality (%dms)"%(duration))

    def update_nch(self, nch):
        self.batch      = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)
        winw, winh = self.window.get_size()
        r = (winh*.8)/(nch+1)
        # TODO[X] use bounding box
        self.chrect = (int(winw*.1), 0, r, r) # bbox for each signal, (x, y, w, h)
        # make a sprite to draw the electrode qualities
        img = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(2, 2)
        # anchor in the center to make drawing easier
        img.anchor_x = 1
        img.anchor_y = 1
        self.sprite = [None]*nch
        self.label  = [None]*nch
        self.linebbox = [None]*nch # bounding box for the channel line
        for i in range(nch):
            x = self.chrect[0]
            y = self.chrect[1]+(i+1)*self.chrect[3]
            # convert to a sprite and make the right size
            self.sprite[i] = pyglet.sprite.Sprite(img, x=x, y=y,
                                                batch=self.batch,
                                                group=self.background)
            # make the desired size
            self.sprite[i].update(scale_x=r*.6/img.width, scale_y=r*.6/img.height)
            # and a text label object
            self.label[i] = pyglet.text.Label("%d"%(i), font_size=32,
                                            x=x, y=y,
                                            color=(255, 255, 255, 255),
                                            anchor_x='center',
                                            anchor_y='center',
                                            batch=self.batch,
                                            group=self.foreground)
            # bounding box for the datalines
            self.linebbox[i] = (x+r, y, winw-(x+r)-.5*r, self.chrect[3])
        # title for the screen
        self.title=pyglet.text.Label(self.instruct, font_size=32,
                                     x=winw*.1, y=winh, color=(255, 255, 255, 255),
                                     anchor_y="top",
                                     width=int(winw*.9),
                                     multiline=True,
                                     batch=self.batch,
                                     group=self.foreground)

    def reset(self):
        self.isRunning = False

    def is_done(self):
        # check termination conditions
        isDone=False
        if not self.isRunning:
            return False
        if self.waitKey:
            global last_key_press
            if self.window.last_key_press:
                self.key_press = self.window.last_key_press
                isDone = True
                self.window.last_key_press = None
        if getTimeStamp() > self.t0+self.duration:
            isDone=True
        if isDone:
            self.noisetag.removeSubscription("D")
            self.noisetag.modeChange("idle")
        return isDone

    def draw(self, t):
        '''Show a set of colored circles based on the lastSigQuality'''
        if not self.isRunning:
            self.isRunning = True # mark that we're running
            self.t0 = getTimeStamp()
            self.noisetag.addSubscription("D") # subscribe to "DataPacket" messages
            self.noisetag.modeChange("ElectrodeQuality")
            self.dataringbuffer.clear()
        if self.clearScreen:
            self.window.clear()
        # get the sig qualities
        electrodeQualities = self.noisetag.getLastSignalQuality()
        if not electrodeQualities: # default to 4 off qualities
            electrodeQualities = [.5]*len(self.sprite)

        if len(electrodeQualities) != len(self.sprite):
            self.update_nch(len(electrodeQualities))

        issig2noise = True #any([s>1.5 for s in electrodeQualities])
        # update the colors
        #print("Qual:", end='')
        for i, qual in enumerate(electrodeQualities):
            self.label[i].text = "%d: %3.1f"%(i+1, qual)
            #print(self.label[i].text + " ", end='')
            if issig2noise:
                qual = log10(qual)/1 # n2s=50->1 n2s=10->.5 n2s=1->0
            qual = max(0, min(1, qual))
            qualcolor = (int(255*qual), int(255*(1-qual)), 0) #red=bad, green=good
            self.sprite[i].color=qualcolor
        #print("")
        # draw the updated batch
        self.batch.draw()

        # get the raw signals
        msgs=self.noisetag.getNewMessages()
        for m in msgs:
            if m.msgID == DataPacket.msgID:
                print('D', end='', flush=True)
                self.dataringbuffer.extend(m.samples)
                if getTimeStamp() > self.t0+self.datawindow_ms: # slide buffer
                    # remove same number of samples we've just added
                    for i in range(len(m.samples)):
                        self.dataringbuffer.popleft()


        if self.dataringbuffer:
            if len(self.dataringbuffer[0]) != len(self.sprite):
                self.update_nch(len(self.dataringbuffer[0]))

            # transpose and flatten the data
            # and estimate it's summary statistics
            from statistics import median

            # CAR
            dataringbuffer =[]
            for t in self.dataringbuffer:
                mu = median(t)
                dataringbuffer.append([c-mu for c in t])
            
            # other pre-processing
            data = []
            mu = [] # mean
            mad = [] # mean-absolute-difference
            nch=len(self.linebbox)
            for ci in range(nch):
                d = [ t[ci] for t in dataringbuffer ]
                # mean last samples
                tmp = d[-int(len(d)*.2):]
                mui = sum(tmp)/len(tmp)
                # center (in time)
                d = [ t-mui for t in d ]
                # scale estimate
                madi = sum([abs(t-mui) for t in tmp])/len(tmp)
                data.append(d)
                mu.append(mui)
                mad.append(madi)

            
            datascale_uv = max(5,median(mad)*4)

            for ci in range(nch):
                d = data[ci]
                # map to screen coordinates
                bbox=self.linebbox[ci]

                # downsample if needed to avoid visual aliasing
                #if len(d) > (bbox[2]-bbox[1])*2:
                #    subsampratio = int(len(d)//(bbox[2]-bbox[1]))
                #    d = [d[i] for i in range(0,len(d),subsampratio)]

                # map to screen coordinates
                xscale = bbox[2]/len(d)
                yscale = bbox[3]/datascale_uv #datascale_uv # 10 uV between lines
                y = [ bbox[1] + s*yscale for s in d ]
                x = [ bbox[0] + i*xscale for i in range(len(d)) ]
                # interleave x, y to make gl happy
                coords = tuple( c for xy in zip(x, y) for c in xy )
                # draw this line
                col = [0,0,0]; col[ci%3]=1
                pyglet.graphics.glColor3d(*col)
                pyglet.gl.glLineWidth(1)
                pyglet.graphics.draw(len(d), pyglet.gl.GL_LINE_STRIP, ('v2f', coords))

                # axes scale
                x = bbox[0]+bbox[2]+20 # at *right* side of the line box
                y = bbox[1]
                pyglet.graphics.glColor3f(1,1,1)
                pyglet.gl.glLineWidth(10)
                pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                                        ('v2f', (x,y-10/2*yscale, x,y+10/2*yscale)))





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
    closing_text = "\n Press key to continue."
    
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





#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------------------------------------------------------
class SelectionGridScreen(Screen):
    '''Screen which shows a grid of symbols which will be flickered with the noisecode
    and which can be selected from by the mindaffect decoder Brain Computer Interface'''

    LOGLEVEL=0

    def __init__(self, window, symbols, noisetag, objIDs=None,
                 bgFraction:float=.2, instruct:str="", 
                 clearScreen:bool=True, sendEvents:bool=True, liveFeedback:bool=True, optosensor:bool=True, 
                 target_only:bool=False, show_correct:bool=True,
                 waitKey:bool=True, stimulus_callback=None, framerate_display:bool=True,
                 logo:str='MindAffect_Logo.png'):
        '''Intialize the stimulus display with the grid of strings in the
        shape given by symbols.
        Store the grid object in the fakepresentation.objects list so can
        use directly with the fakepresentation BCI presentation wrapper.'''
        self.window=window
        # create set of sprites and add to render batch
        self.clearScreen= True
        self.isRunning=False
        self.isDone=False
        self.sendEvents=sendEvents
        self.liveFeedback=liveFeedback
        self.framestart = getTimeStamp()
        self.frameend = getTimeStamp()
        self.symbols = symbols
        self.objIDs = objIDs
        self.optosensor = optosensor
        self.framerate_display = framerate_display
        self.logo = logo
        # N.B. noisetag does the whole stimulus sequence
        self.set_noisetag(noisetag)
        self.set_grid(symbols, objIDs, bgFraction, sentence=instruct, logo=logo)
        self.liveSelections = None
        self.feedbackThreshold = .4
        self.waitKey=waitKey
        self.stimulus_callback = stimulus_callback
        self.last_target_idx = -1
        self.show_correct = show_correct

    def reset(self):
        self.isRunning=False
        self.isDone=False
        self.nframe=0
        self.last_target_idx=-1
        self.set_grid()

    def set_noisetag(self, noisetag):
        self.noisetag=noisetag

    def setliveFeedback(self, value):
        self.liveFeedback=value

    def setliveSelections(self, value):
        if self.liveSelections is None :
            self.noisetag.addSelectionHandler(self.doSelection)
        self.liveSelections = value

    def get_idx(self,idx):
        ii=0 # linear index
        for i in range(len(self.symbols)):
            for j in range(len(self.symbols[i])):
                if self.symbols[i][j] is None: continue
                if idx==(i,j) or idx==ii :
                    return ii
                ii = ii + 1
        return None

    def getLabel(self,idx):
        ii = self.get_idx(idx)
        return self.labels[ii] if ii is not None else None

    def setLabel(self,idx,val):
        ii = self.get_idx(idx)
        # update the label object to the new value
        if ii is not None and self.labels[ii]:
            self.labels[ii].text=val

    def setObj(self,idx,val):
        ii = self.get_idx(idx)
        if ii is not None and self.objects[ii]:
            self.objects[ii]=val

    def doSelection(self, objID):
        if self.liveSelections == True:
            if objID in self.objIDs:
                print("doSelection: {}".format(objID))
                symbIdx = self.objIDs.index(objID)
                sel = self.getLabel(symbIdx)
                sel = sel.text if sel is not None else ''
                text = self.update_text(self.sentence.text, sel)
                if self.show_correct and self.last_target_idx>=0:
                    text += "*" if symbIdx==self.last_target_idx else "_"
                self.set_sentence( text )

    def update_text(self,text:str,sel:str):
        # process special codes
        if sel in ('<-','<bkspc>','<backspace>'):
            text = text[:-1]
        elif sel in ('spc','<space>','<spc>'):
            text = text + '😀'
        elif sel in ('home','quit'):
            pass
        elif sel == ':)':
            text = text + ""
        else:
            text = text + sel
        return text

    def set_sentence(self, text):
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        self.sentence.begin_update()
        self.sentence.text=text
        self.sentence.end_update()
    
    def set_framerate(self, text):
        '''set/update the text to show in the frame rate box'''
        if type(text) is list:
            text = "\n".join(text)
        self.framerate.begin_update()
        self.framerate.text=text
        self.framerate.end_update()

    def set_grid(self, symbols=None, objIDs=None, bgFraction=.3, sentence="What you type goes here", logo=None):
        '''set/update the grid of symbols to be selected from'''
        winw, winh=self.window.get_size()
        # tell noisetag which objIDs we are using
        if symbols is None:
            symbols = self.symbols

        if isinstance(symbols, str):
            symbols = load_symbols(symbols)

        self.symbols=symbols

        # Number of non-None symbols
        nsymb      = sum([sum([(s is not None and not s == '') for s in x ]) for x in symbols])

        if objIDs is not None:
            self.objIDs = objIDs
        else:
            self.objIDs = list(range(1,nsymb+1))
            objIDs = self.objIDs
        if logo is None:
            logo = self.logo
        # get size of the matrix
        gridheight  = len(symbols) + 1 # extra row for sentence
        gridwidth = max([len(s) for s in symbols])
        self.ngrid      = gridwidth * gridheight

        self.noisetag.setActiveObjIDs(self.objIDs)

        # add a background sprite with the right color
        self.objects=[None]*nsymb
        self.labels=[None]*nsymb
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)

        # now create the display objects
        w=winw/gridwidth # cell-width
        bgoffsetx = w*bgFraction
        h=winh/gridheight # cell-height
        bgoffsety = h*bgFraction
        idx=-1
        for i in range(len(symbols)): # rows
            y = (gridheight-1-i-1)/gridheight*winh # top-edge cell
            for j in range(len(symbols[i])): # cols
                # skip unused positions
                if symbols[i][j] is None or symbols[i][j]=="": continue
                idx = idx+1
                symb = symbols[i][j]
                x = j/gridwidth*winw # left-edge cell
                try : # symb is image to use for this button
                    img = search_directories_for_file(symb,os.path.dirname(__file__))
                    img = pyglet.image.load(img)
                    symb = '.' # symb is a fixation dot
                except :
                    # create a 1x1 white image for this grid cell
                    img = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(2, 2)
                # convert to a sprite (for fast re-draw) and store in objects list
                # and add to the drawing batch (as background)
                self.objects[idx]=pyglet.sprite.Sprite(img, x=x+bgoffsetx, y=y+bgoffsety,
                                                    batch=self.batch, group=self.background)
                # re-scale (on GPU) to the size of this grid cell
                self.objects[idx].update(scale_x=int(w-bgoffsetx*2)/img.width,
                                        scale_y=int(h-bgoffsety*2)/img.height)

                # add the foreground label for this cell, and add to drawing batch
                self.labels[idx]=pyglet.text.Label(symb, font_size=32, x=x+w/2, y=y+h/2,
                                                color=(255, 255, 255, 255),
                                                anchor_x='center', anchor_y='center',
                                                batch=self.batch, group=self.foreground)

        # add opto-sensor block
        img = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(1, 1)
        self.opto_sprite=pyglet.sprite.Sprite(img, x=0, y=winh*.9,
                                              batch=self.batch, group=self.background)
        self.opto_sprite.update(scale_x=int(winw*.1), scale_y=int(winh*.1))
        self.opto_sprite.visible=False

        # add the sentence box
        y = winh # top-edge cell
        x = winw*.15 # left-edge cell
        self.sentence=pyglet.text.Label(sentence, font_size=32, 
                                        x=x, y=y, 
                                        width=winw-x-winw*.1, height=(gridheight-1)/gridheight*winh,
                                        color=(255, 255, 255, 255),
                                        anchor_x='left', anchor_y='top',
                                        multiline=True,
                                        batch=self.batch, group=self.foreground)

        # add the framerate box
        self.framerate=pyglet.text.Label("", font_size=12, x=winw, y=winh,
                                        color=(255, 255, 255, 255),
                                        anchor_x='right', anchor_y='top',
                                        batch=self.batch, group=self.foreground)
        
        # add a logo box
        if isinstance(logo,str): # filename to load
            logo = search_directories_for_file(logo,os.path.dirname(__file__),
                                                os.path.join(os.path.dirname(__file__),'..','..'))
            try :
                logo = pyglet.image.load(logo)
                logo.anchor_x, logo.anchor_y  = (logo.width,logo.height) # anchor top-right 
                self.logo = pyglet.sprite.Sprite(logo, self.window.width, self.window.height-16) # sprite a window top-right
            except :
                self.logo = None
        if self.logo:
            self.logo.batch = self.batch
            self.logo.group = self.foreground
            self.logo.update(x=self.window.width,  y=self.window.height-16,
                            scale_x=self.window.width*.1/logo.width, 
                            scale_y=self.window.height*.1/logo.height)


    def is_done(self):
        if self.isDone:
            self.noisetag.modeChange('idle')
        return self.isDone

    # mapping from bci-stimulus-states to display color
    state2color={0:(5, 5, 5),   # off=grey
                 1:(255, 255, 255), # on=white
                 2:(0, 255, 0),    # cue=green
                 3:(0, 0, 255)}    # feedback=blue
    def draw(self, t):
        """draw the letter-grid with given stimulus state for each object.
        Note: To maximise timing accuracy we send the info on the grid-stimulus state
        at the start of the *next* frame, as this happens as soon as possible after
        the screen 'flip'. """
        if not self.isRunning:
            self.isRunning=True
        self.framestart=self.noisetag.getTimeStamp()
        winflip = self.window.lastfliptime
        if winflip > self.framestart or winflip < self.frameend:
            print("Error: frameend={} winflip={} framestart={}".format(self.frameend,winflip,self.framestart))
        self.nframe = self.nframe+1
        if self.sendEvents:
            self.noisetag.sendStimulusState(timestamp=winflip)#self.frameend)#window.lastfliptime)

        # get the current stimulus state to show
        try:
            self.noisetag.updateStimulusState()
            stimulus_state, target_idx, objIDs, sendEvents=self.noisetag.getStimulusState()
            target_state = stimulus_state[target_idx] if target_idx>=0 else -1
            if target_idx >= 0 : self.last_target_idx = target_idx
        except StopIteration:
            self.isDone=True
            return

        if self.waitKey:
            global last_key_press
            if self.window.last_key_press:
                self.key_press = self.window.last_key_press
                #self.noisetag.reset()
                self.isDone = True
                self.window.last_key_press = None

        # turn all off if no stim-state
        if stimulus_state is None:
            stimulus_state = [0]*len(self.objects)

        # do the stimulus callback if wanted
        if self.stimulus_callback is not None:
            self.stimulus_callback(stimulus_state, target_state)

        # draw the white background onto the surface
        if self.clearScreen:
            self.window.clear()
        # update the state
        # TODO[]: iterate over objectIDs and match with those from the
        #         stimulus state!
        for idx in range(min(len(self.objects), len(stimulus_state))):
            # set background color based on the stimulus state (if set)
            try:
                ssi = stimulus_state[idx]
                if self.target_only and not target_idx == idx :
                    ssi = 0
                if self.objects[idx]:
                    self.objects[idx].color=self.state2color[ssi]
                if self.labels[idx]:
                    self.labels[idx].color=(255,255,255,255) # reset labels
            except KeyError:
                pass


        # show live-feedback (if wanted)
        if self.liveFeedback:
            # get prediction info if any
            predMessage=self.noisetag.getLastPrediction()
            if predMessage and predMessage.Yest in objIDs and predMessage.Perr < self.feedbackThreshold:
                predidx=objIDs.index(predMessage.Yest) # convert from objID -> objects index
                prederr=predMessage.Perr
                # BODGE: manually mix in the feedback color as blue tint on the label
                if self.labels[predidx]:
                    fbcol=list(self.labels[predidx].color) 
                    fbcol[0]=fbcol[0]*.4 
                    if fbcol[0]>1: fbcol[0]=int(fbcol[0])
                    fbcol[1]=fbcol[1]*.4; 
                    if fbcol[1]>1: fbcol[1]=int(fbcol[1])
                    fbcol[2]=fbcol[2]*.4+255*(1-prederr)*.6; 
                    if fbcol[2]>1: fbcol[2]=int(fbcol[2])
                    self.labels[predidx].color = fbcol
                
        # disp opto-sensor if targetState is set
        if self.optosensor :
            if self.opto_sprite is not None:
                self.opto_sprite.visible=False  # default to opto-off
            if target_state is not None and target_state in (0, 1):
                print("*" if target_state==1 else '.', end='', flush=True)
                if self.opto_sprite is not None:
                    self.opto_sprite.visible=True
                    self.opto_sprite.color = (0, 0, 0) if target_state==0 else (255, 255, 255)

        # do the draw
        self.batch.draw()
        if self.logo: self.logo.draw()
        self.frameend=self.noisetag.getTimeStamp()

        # frame flip time logging info
        if self.LOGLEVEL > 0 and self.noisetag.isConnected():
            opto = target_state if target_state is not None else 0
            logstr="FrameIdx:%d FlipTime:%d FlipLB:%d FlipUB:%d Opto:%d"%(nframe, self.framestart, self.framestart, self.frameend, opto)
            self.noisetag.log(logstr)

        # add the frame rate info
        # TODO[]: limit update rate..
        if self.framerate_display:
            global flipstats
            flipstats.update_statistics()
            self.set_framerate("{:4.1f} +/-{:4.1f}ms".format(flipstats.median,flipstats.sigma))



#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#---------------------------------------------------------
from enum import IntEnum
class ExptScreenManager(Screen):
    '''class to manage a whole application, with main menu, checks, etc.'''

    class ExptPhases(IntEnum):
        ''' enumeration for the different phases of an experiment/BCI application '''
        MainMenu=0
        Connecting=1
        SignalQuality=2
        CalInstruct=3
        Calibration=4
        CalResults=5
        CuedPredInstruct=6
        CuedPrediction=7
        PredInstruct=8
        Prediction=9
        Closing=10
        Quit=100
        Welcome=99
        Minimize=101
        Settings=105
        FrameRateCheck=200
        Reset=110
        ExtraSymbols=300

    welcomeInstruct="Welcome to the mindaffectBCI\n\nkey to continue"
    calibrationInstruct="Calibration\n\nThe next stage is CALIBRATION\nlook at the indicated green target\n\nkey to continue"
    cuedpredictionInstruct="Prediction\n\nThe next stage is CUED PREDICTION\nLook at the green cued letter\n\nLive BCI feedback in blue\n\nkey to continue"
    predictionInstruct="Prediction\n\nThe next stage is free PREDICTION\nLook at the letter you want to select\nLive BCI feedback in blue\n\nkey to continue"
    closingInstruct="Closing\nThankyou\n\nPress to exit"
    resetInstruct="Reset\n\nThe decoder model has been reset.\nYou will need to run calibration again to use the BCI\n\nkey to continue"

    main_menu_header ="Welcome to the mindaffectBCI" +"\n"+ \
               "\n"+ \
               "Press the number of the option you want:" +"\n"+ \
               "\n"
    main_menu_numbered = "0) Electrode Quality" +"\n"+ \
               "1) Calibration" +"\n"+ \
               "2) Cued Prediction" +"\n"+ \
               "3) Free Typing" +"\n"
    main_menu_footer = "\n\n\n" + "Q) Quit" + "\n" + \
               "f) frame-rate-check" + "\n" + \
               "s) settings\n" + \
               "r) reset calibration model"
               
    menu_keys = {pyglet.window.key._0:ExptPhases.SignalQuality,
                 pyglet.window.key._1:ExptPhases.CalInstruct,
                 pyglet.window.key._2:ExptPhases.CuedPredInstruct,
                 pyglet.window.key._3:ExptPhases.PredInstruct,
                 pyglet.window.key.F:ExptPhases.FrameRateCheck,
                 pyglet.window.key.S:ExptPhases.Settings,
                 pyglet.window.key.R:ExptPhases.Reset,
                 pyglet.window.key.Q:ExptPhases.Quit}

    def __init__(self, window, noisetag, symbols, nCal:int=1, nPred:int=1, 
                 calibration_trialduration=4.2, prediction_trialduration=10,  waitduration=1, feedbackduration=2,
                 framesperbit:int=None, fullscreen_stimulus:bool=True, 
                 selectionThreshold:float=.1, optosensor:bool=True,
                 simple_calibration:bool=False, calibration_symbols=None, extra_symbols=None, bgFraction=.1,
                 calibration_args:dict=None, prediction_args:dict=None):
        self.window = window
        self.noisetag = noisetag
        self.symbols = symbols
        self.calibration_symbols = calibration_symbols if calibration_symbols is not None else symbols
        self.bgFraction = bgFraction

        # auto-generate menu items for each prediction symbols set
        self.extra_symbols = extra_symbols
        if extra_symbols:
            for i,ps in enumerate(extra_symbols):
                keyi = i + 4
                self.main_menu_numbered = self.main_menu_numbered + "\n" + \
                        "{}) Free Typing: {}".format(keyi,ps)
                self.menu_keys[getattr(pyglet.window.key,"_{:d}".format(keyi))] = self.ExptPhases.ExtraSymbols

        self.menu = MenuScreen(window, self.main_menu_header+self.main_menu_numbered+self.main_menu_footer, self.menu_keys.keys())
        self.instruct = InstructionScreen(window, '', duration = 50000)
        self.connecting = ConnectingScreen(window, noisetag)
        self.query  =  QueryDialogScreen(window, 'Query Test:')
        self.electquality = ElectrodequalityScreen(window, noisetag)
        self.results = ResultsScreen(window, noisetag)
        self.selectionGrid = SelectionGridScreen(window, symbols, noisetag, optosensor=optosensor)
        self.stage = self.ExptPhases.Connecting
        self.next_stage = self.ExptPhases.Connecting

        self.nCal = nCal
        self.nPred = nPred
        self.framesperbit = framesperbit
        self.calibration_trialduration = calibration_trialduration
        self.prediction_trialduration = prediction_trialduration
        self.waitduration = waitduration
        self.feedbackduration = feedbackduration
        self.calibration_args = calibration_args if calibration_args else dict()
        self.prediction_args = prediction_args if prediction_args else dict()
        self.calibration_args['nTrials']=self.nCal
        self.prediction_args['nTrials']=self.nPred
        self.calibration_args['framesperbit'] = self.framesperbit
        self.prediction_args['framesperbit'] = self.framesperbit
        self.calibration_args['numframes'] = self.calibration_trialduration / isi
        self.prediction_args['numframes'] = self.prediction_trialduration / isi
        self.calibration_args['waitframes'] = self.waitduration / isi
        self.prediction_args['waitframes'] = self.waitduration / isi
        self.calibration_args['feedbackframes'] = self.feedbackduration / isi
        self.prediction_args['feedbackframes'] = self.feedbackduration / isi

        self.fullscreen_stimulus = fullscreen_stimulus
        self.selectionThreshold = selectionThreshold
        self.simple_calibration = simple_calibration
        self.screen = None
        self.transitionNextPhase()

    def draw(self, t):
        if self.screen is None:
            return
        self.screen.draw(t)
        if self.screen.is_done():
            self.transitionNextPhase()

    def is_done(self):
        return self.screen is None

    def transitionNextPhase(self):
        print("stage transition")

        # move to the next stage
        if self.next_stage is not None:
            self.stage = self.next_stage
            self.next_stage = None
        else: # assume it's from the menu
            self.stage = self.menu_keys.get(self.menu.key_press,self.ExptPhases.MainMenu)
            self.next_stage = None

        if self.stage==self.ExptPhases.MainMenu: # main menu
            if self.fullscreen_stimulus==True :
                self.window.set_fullscreen(fullscreen=False)

            print("main menu")
            self.menu.reset()
            self.screen = self.menu
            self.noisetag.modeChange('idle')
            self.next_stage = None
            

        elif self.stage==self.ExptPhases.Welcome: # welcome instruct
            print("welcome instruct")
            self.instruct.set_text(self.welcomeInstruct)
            self.instruct.reset()
            self.screen = self.instruct
            self.next_stage = self.ExptPhases.Connecting

        elif self.stage==self.ExptPhases.Reset: # reset the decoder
            print("reset")
            self.instruct.set_text(self.resetInstruct)
            self.instruct.reset()
            self.screen = self.instruct
            self.noisetag.modeChange("reset")
            self.next_stage = self.ExptPhases.MainMenu

        elif self.stage==self.ExptPhases.Connecting: # connecting instruct
            print("connecting screen")
            self.connecting.reset()
            self.screen = self.connecting
            self.next_stage = self.ExptPhases.MainMenu

        elif self.stage==self.ExptPhases.SignalQuality: # electrode quality
            print("signal quality")
            self.electquality.reset()
            self.screen=self.electquality
            self.next_stage = self.ExptPhases.MainMenu

        elif self.stage==self.ExptPhases.CalInstruct: # calibration instruct
            print("Calibration instruct")
            if self.fullscreen_stimulus==True :
                self.window.set_fullscreen(fullscreen=True)
            self.instruct.set_text(self.calibrationInstruct)
            self.instruct.reset()
            self.screen=self.instruct
            self.next_stage = self.ExptPhases.Calibration

        elif self.stage==self.ExptPhases.Calibration: # calibration
            print("calibration")
            self.selectionGrid.reset()
            self.selectionGrid.set_grid(symbols=self.calibration_symbols, bgFraction=self.bgFraction)
            self.selectionGrid.liveFeedback=False
            self.selectionGrid.target_only=self.simple_calibration
            self.selectionGrid.set_sentence('Calibration: look at the green cue.')

            self.calibration_args['framesperbit'] = self.framesperbit
            self.calibration_args['numframes'] = self.calibration_trialduration / isi
            self.calibration_args['selectionThreshold']=self.selectionThreshold

            self.selectionGrid.noisetag.startCalibration(**self.calibration_args)
            self.screen = self.selectionGrid
            self.next_stage = self.ExptPhases.CalResults

        elif self.stage==self.ExptPhases.CalResults: # Calibration Results
            print("Calibration Results")
            if self.fullscreen_stimulus==True :
                self.window.set_fullscreen(fullscreen=True)
            self.results.reset()
            self.screen=self.results
            self.next_stage = self.ExptPhases.MainMenu

        elif self.stage==self.ExptPhases.CuedPredInstruct: # pred instruct
            print("cued pred instruct")
            if self.fullscreen_stimulus==True :
                self.window.set_fullscreen(fullscreen=True)
            self.instruct.set_text(self.cuedpredictionInstruct)
            self.instruct.reset()
            self.screen=self.instruct
            self.next_stage = self.ExptPhases.CuedPrediction

        elif self.stage==self.ExptPhases.CuedPrediction: # pred
            print("cued prediction")
            self.selectionGrid.reset()
            self.selectionGrid.set_grid(symbols=self.symbols, bgFraction=self.bgFraction)
            self.selectionGrid.liveFeedback=True
            self.selectionGrid.setliveSelections(True)
            self.selectionGrid.target_only=False
            self.selectionGrid.show_correct=True
            self.selectionGrid.set_sentence('CuedPrediction: look at the green cue.\n')

            self.prediction_args['framesperbit'] = self.framesperbit
            self.prediction_args['numframes'] = self.prediction_trialduration / isi
            self.prediction_args['selectionThreshold']=self.selectionThreshold

            self.selectionGrid.noisetag.startPrediction(cuedprediction=True, **self.prediction_args)
            self.screen = self.selectionGrid
            self.next_stage = self.ExptPhases.MainMenu

        elif self.stage==self.ExptPhases.PredInstruct: # pred instruct
            print("pred instruct")
            if self.fullscreen_stimulus==True :
                self.window.set_fullscreen(fullscreen=True)
            self.instruct.set_text(self.predictionInstruct)
            self.instruct.reset()
            self.screen=self.instruct
            self.next_stage = self.ExptPhases.Prediction

        elif self.stage==self.ExptPhases.Prediction: # pred
            print("prediction")
            self.selectionGrid.reset()
            self.selectionGrid.set_grid(symbols=self.symbols, bgFraction=.05)
            self.selectionGrid.liveFeedback=True
            self.selectionGrid.target_only=False
            self.selectionGrid.show_correct=False
            self.selectionGrid.set_sentence('')
            self.selectionGrid.setliveSelections(True)

            self.prediction_args['framesperbit'] = self.framesperbit
            self.prediction_args['numframes'] = self.prediction_trialduration / isi
            self.prediction_args['selectionThreshold']=self.selectionThreshold
            
            self.selectionGrid.noisetag.startPrediction(**self.prediction_args)
            self.screen = self.selectionGrid
            self.next_stage = self.ExptPhases.MainMenu

        elif self.stage==self.ExptPhases.ExtraSymbols: # pred
            print("Extra Prediction")
            key2i = {pyglet.window.key._4:0,pyglet.window.key._5:1, pyglet.window.key._6:2}
            extrai = key2i.get(self.menu.key_press,None)
            if extrai is not None:
                self.selectionGrid.reset()
                self.selectionGrid.set_grid(symbols=self.extra_symbols[extrai], bgFraction=.05)
                self.selectionGrid.liveFeedback=True
                self.selectionGrid.target_only=False
                self.selectionGrid.show_correct=False
                self.selectionGrid.set_sentence('')
                self.selectionGrid.setliveSelections(True)

                self.prediction_args['framesperbit'] = self.framesperbit
                self.prediction_args['numframes'] = self.prediction_trialduration / isi
                self.prediction_args['selectionThreshold']=self.selectionThreshold
                
                self.selectionGrid.noisetag.startPrediction(**self.prediction_args)
                self.screen = self.selectionGrid
            self.next_stage = self.ExptPhases.MainMenu

        elif self.stage==self.ExptPhases.Closing: # closing instruct
            print("closing instruct")
            self.instruct.set_text(self.closingInstruct)
            self.instruct.reset()
            self.screen=self.instruct
            self.next_stage = self.ExptPhases.Quit

        elif self.stage==self.ExptPhases.Minimize: # closing instruct
            print("minimize")
            self.window.minimize()
            self.next_stage = self.ExptPhases.MainMenu

        elif self.stage==None: # testing stages..
            #print("flicker with selection")
            #self.selectionGrid.noisetag.startFlickerWithSelection(numframes=10/isi)
            print("single trial")
            self.selectionGrid.set_grid([[None, 'up', None],
                                         ['left', 'fire', 'right']])
            self.selectionGrid.noisetag.startSingleTrial(numframes=10/isi)
            # N.B. ensure decoder is in prediction mode!
            self.selectionGrid.noisetag.modeChange('Prediction.static')
            self.selectionGrid.reset()
            self.screen = self.selectionGrid

        elif self.stage==self.ExptPhases.FrameRateCheck: # frame-rate-check
            print("frame-rate-check")
            #from mindaffectBCI.examples.presentation.framerate_check import FrameRateTestScreen
            self.screen=FrameRateTestScreen(self.window,waitKey=True)
            self.next_stage = self.ExptPhases.MainMenu

        elif self.stage==self.ExptPhases.Settings: # config settings
            print("settings")
            self.screen = SettingsScreen(self.window, self)
            self.next_stage = self.ExptPhases.MainMenu

        else: # end
            print('quit')
            self.screen=None


#------------------------------------------------------------------------
# Initialization: display, utopia-connection
# use noisetag object as time-stamp provider
def getTimeStamp():
    if 'nt' in globals():
        global nt
        return nt.getTimeStamp()
    else: # fall back if not connected to utopia client
        import time
        return (int(time.perf_counter()*1000) % (1<<31))

import types
from mindaffectBCI.noisetag import sumstats
flipstats=sumstats(60)
fliplogtime=0
def timedflip(self):
    '''pseudo method type which records the timestamp for window flips'''
    global flipstats, fliplogtime
    type(self).flip(self)
    olft=self.lastfliptime
    self.lastfliptime=getTimeStamp()
    if flipstats is not None:
        flipstats.addpoint(self.lastfliptime-olft)
        #if self.lastfliptime > fliplogtime:
        #    fliplogtime=fliplogtime+5000
        #    print("\nFlipTimes:"+str(flipstats))
        #    print("Hist:\n"+flipstats.hist())

def on_key_press(symbols, modifiers):
    '''main key-press handler, which stores the last key in a global variable'''
    global window
    window.last_key_press=symbols

def on_text(text):
    global window
    window.last_text = text

def initPyglet(fullscreen=False):
    '''intialize the pyglet window, keyhandler'''
    global window
    # set up the window
    if fullscreen:
        print('Fullscreen mode!')
        # N.B. accurate video timing only guaranteed with fullscreen
        # N.B. over-sampling seems to introduce frame lagging on windows+Intell
        config = pyglet.gl.Config(double_buffer=True) #double_buffer=False,sample_buffers=1, samples=4)
        window = pyglet.window.Window(fullscreen=True, vsync=True, resizable=False, config=config)
    else:
        config = pyglet.gl.Config(double_buffer=True)#,sample_buffers=1, samples=4)
        window = pyglet.window.Window(width=1024, height=768, vsync=True, resizable=True, config=config)

    # setup a key press handler, just store key-press in global variable
    window.push_handlers(on_key_press, on_text)
    window.last_key_press=None
    window.last_text=None

    # override window's flip method to record the exact *time* the
    # flip happended
    window.flip = types.MethodType(timedflip, window)
    window.lastfliptime=getTimeStamp()
    global fliplogtime; fliplogtime=window.lastfliptime

    # minimize on tab away, when in fullscreen mode:
    @window.event
    def on_deactivate():
        # TODO []: stop minimise when switch to/from full-screen mode
        if fullscreen:
            window.minimize()

    # TODO[]: handle resize events correctly.
    return window

def draw(dt):
    '''main window draw function, which redirects to the screen stack'''
    global ss, nframe
    nframe=nframe+1
    ss.draw(dt)
    # check for termination
    if ss.is_done():
        print('app exit')
        pyglet.app.exit()
    #print('.', end='', flush=True)

def run_screen(screen:Screen, drawrate:float=-1, win:pyglet.window=None):
    global ss, window, nframe
    nframe = 0
    ss = screen
    if win is not None:
        window = win 
    # set per-frame callback to the draw function
    if drawrate > 0:
        # slow down for debugging
        pyglet.clock.schedule_interval(draw, drawrate)
    else:
        # call the draw method as fast as possible, i.e. at video frame rate!
        pyglet.clock.schedule(draw)
    # mainloop
    pyglet.app.run()
    pyglet.app.EventLoop().exit()
    window.set_visible(False)

def load_symbols(fn):
    """load a screen layout from a text file

    Args:
        fn (str): file name to load from

    Returns:
        symbols [list of lists of str]: list of list of the symbols strings
    """
    symbols = []

    # search in likely directories for the file, cwd, pydir, projectroot 
    fn = search_directories_for_file(fn,os.path.dirname(os.path.abspath(__file__)),
                                        os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))

    with open(fn,'r', encoding='utf8') as f:
        for line in f:
            # skip comment lines
            if line.startswith('#'): continue
            # delim is ,
            line = line.split(',')
            # strip whitespace
            line = [ l.strip() for l in line if l is not None ]
            # None for empty strings
            line = [ l if not l == "" else None for l in line ]
            # strip quotes
            line = [ l.strip('\"') if l is not None else l for l in line ]
            # add
            symbols.append(line)

    return symbols

def run(symbols=None, ncal:int=10, npred:int=10, calibration_trialduration=4.2,  prediction_trialduration=20, feedbackduration:float=2, stimfile=None, selectionThreshold:float=.1,
        framesperbit:int=1, optosensor:bool=True, fullscreen:bool=False, windowed:bool=None, 
        fullscreen_stimulus:bool=True, simple_calibration=False, host=None, calibration_symbols=None, bgFraction=.1,
        extra_symbols=None, calibration_args:dict=None, prediction_args:dict=None):
    """ run the selection Matrix with default settings

    Args:
        nCal (int, optional): number of calibration trials. Defaults to 10.
        nPred (int, optional): number of prediction trials at a time. Defaults to 10.
        simple_calibration (bool, optional): flag if we show only a single target during calibration, Defaults to False.
        stimFile ([type], optional): the stimulus file to use for the codes. Defaults to None.
        framesperbit (int, optional): number of video frames per stimulus codebit. Defaults to 1.
        fullscreen (bool, optional): flag if should runn full-screen. Defaults to False.
        fullscreen_stimulus (bool, optional): flag if should run the stimulus (i.e. flicker) in fullscreen mode. Defaults to True.
        simple_calibration (bool, optional): flag if we only show the *target* during calibration.  Defaults to False
        calibration_trialduration (float, optional): flicker duration for the calibration trials. Defaults to 4.2.
        prediction_trialduration (float, optional): flicker duration for the prediction trials.  Defaults to 10.
        calibration_args (dict, optional): additional keyword arguments to pass to `noisetag.startCalibration`. Defaults to None.
        prediction_args (dict, optional): additional keyword arguments to pass to `noisetag.startPrediction`. Defaults to None.
    """
    global nt, ss, window
    # N.B. init the noise-tag first, so asks for the IP
    if stimfile is None:
        stimfile = 'mgold_61_6521_psk_60hz.txt'
    if fullscreen is None and windowed is not None:
        fullscreen = not windowed
    if windowed == True or fullscreen == True:
        fullscreen_stimulus = False
    nt=Noisetag(stimFile=stimfile,clientid='Presentation:selectionMatrix')
    if host is not None and not host in ('','-'):
        nt.connect(host, queryifhostnotfound=False)

    # init the graphics system
    window = initPyglet(fullscreen=fullscreen)

    # the logical arrangement of the display matrix
    if symbols is None:
        symbols=[['a', 'b', 'c', 'd', 'e'],
                 ['f', 'g', 'h', 'i', 'j'],
                 ['k', 'l', 'm', 'n', 'o'],
                 ['p', 'q', 'r', 's', 't'],
                 ['u', 'v', 'w', 'x', '<-']]

    # different calibration symbols if wanted
    if calibration_symbols is None:
        calibration_symbols = symbols

    # make the screen manager object which manages the app state
    ss = ExptScreenManager(window, nt, symbols, nCal=ncal, nPred=npred, framesperbit=framesperbit, 
                        fullscreen_stimulus=fullscreen_stimulus, selectionThreshold=selectionThreshold, 
                        optosensor=optosensor, simple_calibration=True, calibration_symbols=calibration_symbols, 
                        extra_symbols=extra_symbols,
                        bgFraction=bgFraction, 
                        calibration_args=calibration_args, calibration_trialduration=calibration_trialduration, 
                        prediction_args=prediction_args, prediction_trialduration=prediction_trialduration, feedbackduration=feedbackduration)

    run_screen(ss,drawrate)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ncal',type=int, help='number calibration trials', nargs='?', default=10)
    parser.add_argument('npred',type=int, help='number prediction trials', nargs='?', default=10)
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--stimfile',type=str, help='stimulus file to use', default=None)
    parser.add_argument('--framesperbit',type=int, help='number of video frames per stimulus bit', default=1)
    parser.add_argument('--fullscreen_stimulus',action='store_true',help='run with stimuli in fullscreen mode')
    parser.add_argument('--windowed',action='store_true',help='run in fullscreen mode')
    parser.add_argument('--selectionThreshold',type=float,help='target error threshold for selection to occur',default=.1)
    parser.add_argument('--simple_calibration',action='store_true',help='flag to only show a single target during calibration')
    parser.add_argument('--symbols',type=str,help='file name for the symbols grid to display',default=None)
    parser.add_argument('--calibration_symbols',type=str,help='file name for the symbols grid to use for calibration',default=None)
    parser.add_argument('--extra_symbols',type=str,help='comma separated list of extra symbol files to show',default=None)
    #parser.add_option('-m','--matrix',action='store',dest='symbols',help='file with the set of symbols to display',default=None)
    args = parser.parse_args()

    if args.extra_symbols:
        args.extra_symbols = args.extra_symbols.split(',')

    return args

if __name__ == "__main__":
    args = parse_args()

    setattr(args,'symbols',[['yes','no','<-']])
    setattr(args,'extra_symbols',['3x3.txt','robot_control.txt'])

    run(**vars(args))

