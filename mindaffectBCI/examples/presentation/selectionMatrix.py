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
from mindaffectBCI.noisetag import Noisetag, getTimeStamp
from mindaffectBCI.utopiaclient import DataPacket

# graphic library
import pyglet
isi=1/60
drawrate=0 # rate at which draw is called
last_key_press=None
last_text=None

class Screen:
    '''Screen abstract-class which draws stuff on the screen until finished'''
    def __init__(self,window):
        self.window=window
    def reset(self):
        '''reset this screen to clean state'''
        pass
    def draw(self,t):
        '''draw the display, N.B. NOT including flip!'''
        pass
    def is_done(self):
        '''test if this screen wants to quit'''
        return False
            
class InstructionScreen(Screen):
    '''Screen which shows a textual instruction for duration or until key-pressed'''
    def __init__(self,window, text, duration=5000, waitKey=True):
        super().__init__(window)
        self.t0=None # timer for the duration
        self.duration=duration
        self.waitKey=waitKey
        self.isRunning=False
        self.isDone=False
        self.clearScreen=True
        # initialize the instructions screen
        self.instructLabel=pyglet.text.Label(x=window.width//2, 
                                             y=window.height//2, 
                                             anchor_x='center', 
                                             anchor_y='center', 
                                             font_size=24, 
                                             color=(255, 255, 255, 255), 
                                             multiline=True, 
                                             width=int(window.width*.8))
        self.set_text(text)
        print("Instruct (%dms): %s"%(duration, text))

    def reset(self):
        self.isRunning=False
        self.isDone=False
        
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
            self.isDone=False
            return self.isDone
        if self.waitKey:
            global last_key_press
            if last_key_press:
                self.key_press=last_key_press
                self.isDone=True
                last_key_press=None
        if self.elapsed_ms() > self.duration:
            self.isDone=True
        return self.isDone

    def elapsed_ms(self):
        return getTimeStamp(self.t0)    
    
    def draw(self, t):
        '''Show a block of text to the user for a given duration on a blank screen'''
        if not self.isRunning:
            self.isRunning=True # mark that we're running
            self.t0=getTimeStamp()
        if self.clearScreen:
            self.window.clear()
        self.instructLabel.draw()

#-----------------------------------------------------------------
class ResultsScreen(InstructionScreen):
    '''Modified instruction screen with waits for and presents calibration results'''

    waiting_text = "Waiting for performance results from decoder\n\nPlease wait"
    results_text = "Calibration Performance: %3.0f%% Correct\n\nKey to continue"
    def __init__(self, window, noisetag, duration=20000, waitKey=False):
        super().__init__(window, self.waiting_text, duration, waitKey)
        self.noisetag=noisetag
        self.pred=None

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        if not self.isRunning:
            self.noisetag.clearLastPrediction()
        # check for predictions
        pred=self.noisetag.getLastPrediction()
        # update text if got predicted performance
        if pred is not None:
            print("Prediction:{}".format(pred))
            self.waitKey=True
            self.set_text(self.results_text%((1.0-pred.Perr)*100.0))
        super().draw(t)


#-----------------------------------------------------------------
class ConnectingScreen(InstructionScreen):
    '''Modified instruction screen with waits for the noisetag to connect to the decoder'''

    searching_text  = "Searching for the mindaffect decoder\n\nPlease wait"
    trying_text   = "Trying to connect to: %s\n Please wait"
    connected_text= "Success!\nconnected to: %s"
    query_text    = "Couldnt auto-discover mindaffect decoder\n\nPlease enter decoder address: %s"
    drawconnect_timeout_ms=200
    autoconnect_timeout_ms=5000
    
    def __init__(self, window, noisetag, duration=150000):
        super().__init__(window, self.searching_text, duration, False)
        self.noisetag=noisetag
        self.host=None
        self.port=-1
        self.usertext=''
        self.stage=0

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        global last_text, last_key_press
        if not self.isRunning:
            super().draw(t)
            return
        
        if not self.noisetag.isConnected():
            if self.stage==0: # try-connection
                print('Not connected yet!!')
                self.noisetag.connect(self.host, self.port, 
                                      queryifhostnotfound=False, 
                                      timeout_ms=self.drawconnect_timeout_ms)
                if self.noisetag.isConnected():
                    self.set_text(self.connected_text%(self.noisetag.gethostport()))
                    self.t0=getTimeStamp()
                    self.duration=1000
                    self.noisetag.subscribe("MSPQ")
                elif self.elapsed_ms() > self.autoconnect_timeout_ms:
                    # waited too long, giveup and ask user
                    self.stage=1
                    # ensure old key-presses are gone
                    last_text=None
                    last_key_press=None
                    
            elif self.stage==1: # query hostname
                # query the user for host/port
                # accumulate user inputs
                if last_key_press:
                    if last_key_press == pyglet.window.key.BACKSPACE:
                        # remove last character
                        self.usertext = self.usertext[:-1]
                    last_key_press=None
                if last_text:
                    print(last_text + ":" + str(ord(last_text)))
                    if last_text == '\n' or last_text=='\r':
                        # set as new host to try
                        self.host=self.usertext
                        self.usertext=''
                        self.set_text(self.trying_text%(self.host))
                        self.stage=0 # back to try-connection stage
                    elif last_text: 
                        # add to the host string
                        self.usertext +=last_text
                    last_text=None
                if self.stage==1: # in same stage
                    # update display with user input
                    self.set_text(self.query_text%(self.usertext))
        super().draw(t)


#-----------------------------------------------------------------
class QueryDialogScreen(InstructionScreen):
    '''Modified instruction screen queries the user for textual input'''

    def __init__(self, window, text, duration=50000, waitKey=True):
        super().__init__(window, text, 50000, False)
        self.query=text
        self.usertext=''

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        # query the user for host/port
        # accumulate user inputs
        global last_key_press, last_text

        if last_key_press:
            if last_key_press == pyglet.window.key.BACKSPACE:
                self.usertext = self.usertext[:-1]
                self.set_text(self.query +self.usertext)
            last_key_press=None
        if last_text:
            if last_text == '\r' or last_text=='\n':
                self.isDone=True
            elif last_text: 
                # add to the host string
                self.usertext += last_text
            last_text=None
            # update display with user input
            self.set_text(self.query +self.usertext)
        super().draw(t)
        
#-----------------------------------------------------------------
from math import log10
from collections import deque
class ElectrodequalityScreen(Screen):
    '''Screen which shows the electrode signal quality information'''

    instruct="Electrode Quality\n\nAdjust headset until all\nelectrodes are green\n(or noise to signal ratio < 5)"
    def __init__(self, window, noisetag, nch=4, duration=200000, waitKey=True):
        super().__init__(window)
        self.noisetag=noisetag
        self.t0=None # timer for the duration
        self.duration=duration
        self.waitKey=waitKey
        self.clearScreen=True
        self.isRunning=False
        self.update_nch(nch)
        self.dataringbuffer=deque() # deque so efficient sliding data window
        self.datawindow_ms=5000 # 5seconds data plotted
        self.datascale_uv=20 # scale of gap between ch plots
        print("Electrode Quality (%dms)"%(duration))

    def update_nch(self, nch):
        self.batch      = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)
        winw, winh=window.get_size()
        r=(winh*.6)/(nch+1)
        # TODO[] use bounding box
        self.chrect=(int(winw*.25), 0, r, r) # bbox for each signal, (x, y, w, h)
        # make a sprite to draw the electrode qualities
        img = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(2, 2)
        # anchor in the center to make drawing easier
        img.anchor_x=1
        img.anchor_y=1        
        self.sprite=[None]*nch
        self.label =[None]*nch
        self.linebbox=[None]*nch # bounding box for the channel line
        for i in range(nch):
            x=self.chrect[0]
            y=self.chrect[1]+(i+1)*self.chrect[3]
            # convert to a sprite and make the right size
            self.sprite[i]=pyglet.sprite.Sprite(img, x=x, y=y, 
                                                batch=self.batch, 
                                                group=self.background)
            # make the desired size
            self.sprite[i].update(scale_x=r*.6/img.width, scale_y=r*.6/img.height)
            # and a text label object
            self.label[i]=pyglet.text.Label("%d"%(i), font_size=32, 
                                            x=x, y=y, 
                                            color=(255, 255, 255, 255), 
                                            anchor_x='center', 
                                            anchor_y='center', 
                                            batch=self.batch, 
                                            group=self.foreground)
            # bounding box for the datalines
            self.linebbox[i]= (x+r, y, winh*.9-x+r, self.chrect[3])
        # title for the screen
        self.title=pyglet.text.Label(self.instruct, font_size=32, 
                                     x=0, y=winh, color=(255, 255, 255, 255), 
                                     anchor_y="top", 
                                     width=int(window.width*.7), 
                                     multiline=True, 
                                     batch=self.batch, 
                                     group=self.foreground)

    def reset(self):
        self.isRunning=False
        
    def is_done(self):
        # check termination conditions
        isDone=False
        if not self.isRunning:
            return False
        if self.waitKey:
            global last_key_press
            if last_key_press:
                self.key_press=last_key_press
                isDone=True
                last_key_press=None
        if getTimeStamp(self.t0) > self.duration:
            isDone=True
        if isDone:
            self.noisetag.removeSubscription("D")
            self.noisetag.modeChange("idle")
        return isDone
        
    def draw(self, t):
        '''Show a set of colored circles based on the lastSigQuality'''
        if not self.isRunning:
            self.isRunning=True # mark that we're running
            self.t0=getTimeStamp()
            self.noisetag.addSubscription("D") # subscribe to "DataPacket" messages
            self.noisetag.modeChange("ElectrodeQuality")
        if self.clearScreen:
            self.window.clear()
        # get the sig qualities
        electrodeQualities = self.noisetag.getLastSignalQuality()
        if not electrodeQualities: # default to 4 off qualities
            electrodeQualities=[.5]*len(self.sprite)

        if len(electrodeQualities) != len(self.sprite):
            self.update_nch(len(electrodeQualities))

        issig2noise = True #any([s>1.5 for s in electrodeQualities])
        # update the colors
        print("Qual:", end='')
        for i, qual in enumerate(electrodeQualities):
            self.label[i].text = "%d: %3.1f"%(i, qual)
            print(self.label[i].text + " ", end='')
            if issig2noise:
                qual = log10(qual)/2 # n2s=50->1 n2s=10->.5 n2s=1->0
            qual=max(0, min(1, qual))
            qualcolor = (int(255*qual), int(255*(1-qual)), 0) #red=bad, green=good
            self.sprite[i].color=qualcolor
        print("")
        # draw the updated batch
        self.batch.draw()

        # get the raw signals
        msgs=self.noisetag.getNewMessages()
        for m in msgs:
            if m.msgID==DataPacket.msgID:
                print('D', end='', flush=True)
                if getTimeStamp() > self.t0+self.datawindow_ms: # slide buffer
                    self.dataringbuffer.popleft()
                self.dataringbuffer.append(m.samples)
        # draw the lines
        if self.dataringbuffer:
            nch=len(self.linebbox)
            for ci in range(nch):
                # flatten into list samples for each channel
                d = [ t[ci] for m in self.dataringbuffer for t in m ]
                # map to screen coordinates
                bbox=self.linebbox[ci]
                yscale = bbox[3]/self.datascale_uv # 10 uV between lines
                mu = sum(d[int(len(d)*.8):])/(len(d)*.8) # center last samples
                y = [ bbox[1] + (s-mu)*yscale for s in d ]
                x = [ bbox[0] + bbox[2]*i/len(y) for i in range(len(d)) ]
                # interleave x, y to make gl happy
                coords = tuple( c for xy in zip(x, y) for c in xy )
                # draw this line
                pyglet.graphics.draw(len(d), pyglet.gl.GL_LINES, ('v2f', coords))


#-------------------------------------------------------------
class SelectionGridScreen(Screen):
    '''Screen which shows a grid of symbols which will be flickered with the noisecode
    and which can be selected from by the mindaffect decoder Brain Computer Interface'''

    LOGLEVEL=1
    
    def __init__(self, window, symbols, noisetag, objIDs=None, 
                 bgFraction=.2, clearScreen=True, sendEvents=True, liveFeedback=True):
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
        # N.B. noisetag does the whole stimulus sequence
        self.set_noisetag(noisetag)
        self.set_grid(symbols, objIDs, bgFraction)

    def reset(self):
        self.isRunning=False
        self.isDone=False

    def set_noisetag(self, noisetag):
        self.noisetag=noisetag

    def setliveFeedback(self, value):
        self.liveFeedback=value
        
    def set_grid(self, symbols, objIDs=None, bgFraction=.3):
        '''set/update the grid of symbols to be selected from'''
        winw, winh=window.get_size()
        # get size of the matrix
        # Number of non-None symbols
        nsymb      = sum([sum([(s is not None) for s in x ]) for x in symbols])
        gridheight  = len(symbols)
        gridwidth = max([len(s) for s in symbols])
        ngrid      = gridwidth * gridheight     
        # tell noisetag how many symbols we have
        if objIDs is None: 
            self.noisetag.setnumActiveObjIDs(nsymb)
        else: # use give set objIDs
            self.noisetag.setActiveObjIDs(objIDs)
    
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
        for i in range(gridheight): # rows
            y = (gridheight-1-i)/gridheight*winh # top-edge cell
            for j in range(gridwidth): # cols
                # skip unused positions
                if symbols[i][j] is None: continue
                idx = idx+1
                x = j/gridwidth*winw # left-edge cell
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
                self.labels[idx]=pyglet.text.Label(symbols[i][j], font_size=32, x=x+w/2, y=y+h/2, 
                                                   color=(255, 255, 255, 255), 
                                                   anchor_x='center', anchor_y='center', 
                                                   batch=self.batch, group=self.foreground)

        # add opto-sensor block - as last object
        img = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(1, 1)
        self.opto_sprite=pyglet.sprite.Sprite(img, x=0, y=winh-h/2, 
                                              batch=self.batch, group=self.background)
        self.opto_sprite.update(scale_x=int(w/2), scale_y=int(h/2))
        self.opto_sprite.visible=False

    def is_done(self):
        return self.isDone

    # mapping from bci-stimulus-states to display color
    state2color={0:(10, 10, 10),   # off=grey
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
        if self.sendEvents:
            self.noisetag.sendStimulusState(timestamp=window.lastfliptime)

        # get the current stimulus state to show
        try:
            self.noisetag.updateStimulusState()
            stimulus_state, target_state, objIDs, sendEvents=self.noisetag.getStimulusState()
        except StopIteration:
            print("end noisetag sequence")
            self.isDone=True
            return
        
        # turn all off if no stim-state
        if stimulus_state is None:
            stimulus_state = [0]*len(self.objects)

        # draw the white background onto the surface
        if self.clearScreen:
            window.clear()
        # update the state
        # TODO[]: iterate over objectIDs and match with those from the
        #         stimulus state!
        for idx in range(min(len(self.objects), len(stimulus_state))): 
            # set background color based on the stimulus state (if set)
            try:
                self.objects[idx].color=self.state2color[stimulus_state[idx]]
            except KeyError:
                pass
            
                
        # show live-feedback (if wanted)
        if self.liveFeedback:
            # get prediction info if any
            predMessage=self.noisetag.getLastPrediction()
            if predMessage and predMessage.Yest in objIDs:
                predidx=objIDs.index(predMessage.Yest) # convert from objID -> objects index
                # BODGE: manually mix in the feedback color as blue tint.
                fbcol=self.objects[predidx].color
                fbcol=(fbcol[0]*.6, fbcol[1]*.6, fbcol[2]*.6+255*(1-predMessage.Perr))
                self.objects[predidx].color=fbcol

        # disp opto-sensor if targetState is set
        if self.opto_sprite is not None:
            self.opto_sprite.visible=False  # default to opto-off
        if target_state is not None and target_state in (0, 1):
            print("*" if target_state==1 else '.', end='', flush=True)
            if self.opto_sprite is not None:
                self.opto_sprite.visible=True
                self.opto_sprite.color = (0, 0, 0) if target_state==0 else (255, 255, 255)

        # do the draw                
        self.batch.draw()
        self.frameend=self.noisetag.getTimeStamp()
    
        # frame flip time logging info
        if self.LOGLEVEL > 0 and self.noisetag.isConnected():
            opto = target_state if target_state is not None else 0
            logstr="FrameIdx:%d FlipTime:%d FlipLB:%d FlipUB:%d Opto:%d"%(nframe, self.framestart, self.framestart, self.frameend, opto)
            self.noisetag.log(logstr)


#---------------------------------------------------------
class ExptScreenManager(Screen):
    '''class to manage a whole experiment:
       instruct->cal->instruct->predict->instruct'''

    welcomeInstruct="Welcome to the mindaffectBCI\n\nkey to continue"
    calibrationInstruct="Calibration\n\nThe next stage is CALIBRATION\nlook at the indicated green target\n\nkey to continue"
    predictionInstruct="Prediction\n\nThe next stage is PREDICTION\nLook at the letter you want to select\nLive BCI feedback in blue\n\nkey to continue"
    closingInstruct="Closing\nThankyou\n\nkey to continue"
    def __init__(self, window, noisetag, symbols, nCal=1, nPred=1):
        self.window=window
        self.noisetag=noisetag
        self.symbols=symbols
        self.instruct=InstructionScreen(window, '', duration=50000)
        self.connecting=ConnectingScreen(window, noisetag)
        self.query = QueryDialogScreen(window, 'Query Test:')
        self.electquality=ElectrodequalityScreen(window, noisetag)
        self.results=ResultsScreen(window, noisetag)
        self.selectionGrid=SelectionGridScreen(window, symbols, noisetag)
        self.stage=0
        self.nCal=nCal
        self.nPred=nPred
        self.screen=None
        self.transitionNextPhase()
        
    def draw(self, t):
        self.screen.draw(t)
        if self.screen.is_done():
            self.transitionNextPhase()

    def is_done(self):
        return self.screen is None

    def transitionNextPhase(self):
        print("stage transition")
        if self.stage==0: # welcome instruct
            print("welcome instruct")
            self.instruct.set_text(self.welcomeInstruct)
            self.instruct.reset()
            self.screen=self.instruct
            
        elif self.stage==1: # connecting instruct
            print("connecting screen")
            self.connecting.reset()
            self.screen=self.connecting

        elif self.stage==2: # electrode quality
            print("signal quality")
            self.electquality.reset()
            self.screen=self.electquality
            
        elif self.stage==3: # calibration instruct
            print("Calibration instruct")
            self.instruct.set_text(self.calibrationInstruct)
            self.instruct.reset()
            self.screen=self.instruct

        elif self.stage==4: # calibration
            print("calibration")
            self.selectionGrid.noisetag.startCalibration(nTrials=self.nCal, numframes=4.2/isi, waitduration=1)
            self.selectionGrid.reset()
            self.selectionGrid.liveFeedback=False
            self.screen = self.selectionGrid
            
        elif self.stage==5: # Calibration Results
            print("Calibration Results")
            self.results.reset()
            self.screen=self.results

        elif self.stage==6: # pred instruct
            print("pred instruct")
            self.instruct.set_text(self.predictionInstruct)
            self.instruct.reset()
            self.screen=self.instruct
            
        elif self.stage==7: # pred
            print("prediction")
            self.selectionGrid.noisetag.startPrediction(nTrials=self.nPred, numframes=10/isi, cuedprediction=True, waitduration=1)
            self.selectionGrid.reset()
            self.selectionGrid.liveFeedback=True
            self.screen = self.selectionGrid
            
        elif self.stage==8: # closing instruct
            print("closing instruct")
            self.instruct.set_text(self.closingInstruct)
            self.instruct.reset()
            self.screen=self.instruct

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

        else: # end
            self.screen=None
        self.stage=self.stage+1        
            
#------------------------------------------------------------------------
# Initialization: display, utopia-connection
import types
from mindaffectBCI.noisetag import sumstats
flipstats=sumstats(); fliplogtime=0
def timedflip(self):
    '''pseudo method type which records the timestamp for window flips'''
    type(self).flip(self)
    olft=self.lastfliptime
    self.lastfliptime=getTimeStamp()
    flipstats.addpoint(self.lastfliptime-olft)
    global fliplogtime
    if self.lastfliptime > fliplogtime:
        fliplogtime=fliplogtime+5000    
        print("\nFlipTimes:"+str(flipstats))
        print("Hist:\n"+flipstats.hist())

def initPyglet(fullscreen=False):
    '''intialize the pyglet window, keyhandler'''
    global window
    # set up the window
    if fullscreen: 
        # N.B. accurate video timing only guaranteed with fullscreen
        config = pyglet.gl.Config(double_buffer=True)
        window = pyglet.window.Window(vsync=True, config=config)
    else:
        config = pyglet.gl.Config(double_buffer=True)
        window = pyglet.window.Window(width=1024, height=768, vsync=True, config=config)

    # setup a key press handler, just store key-press in global variable
    window.push_handlers(on_key_press, on_text)

    global nframe; nframe=0
    # override window's flip method to record the exact *time* the
    # flip happended
    window.flip = types.MethodType(timedflip, window)
    window.lastfliptime=getTimeStamp()
    global fliplogtime; fliplogtime=window.lastfliptime
    return window

def draw(dt):
    '''main window draw function, which redirects to the screen stack'''
    global ss, nframe
    nframe=nframe+1
    ss.draw(dt)
    # check for termination
    if ss.is_done():
        pyglet.app.exit()
    #print('.', end='', flush=True)

def on_key_press(symbols, modifiers):
    '''main key-press handler, which stores the last key in a global variable'''
    global last_key_press
    last_key_press=symbols

def on_text(text):
    global last_text
    last_text=text
    
if __name__ == "__main__":
    import sys
    argv = sys.argv
    nCal = int(argv[1]) if len(argv) > 1 else 10
    nPred = int(argv[2]) if len(argv) > 2 else 10

    # N.B. init the noise-tag first, so asks for the IP
    nt=Noisetag()
        
    # init the graphics system
    initPyglet()

    # the logical arrangement of the display matrix
    symbols=[['a', 'b', 'c', 'd', 'e'], 
             ['f', 'g', 'h', 'i', 'j'], 
             ['k', 'l', 'm', 'n', 'o'], 
             ['p', 'q', 'r', 's', 't'], 
             ['u', 'v', 'w', 'x', 'y']]
    # make the screen manager object which manages the app state
    ss = ExptScreenManager(window, nt, symbols, nCal=nCal, nPred=nPred)

    # set per-frame callback to the draw function    
    if drawrate > 0:
        # slow down for debugging
        pyglet.clock.schedule_interval(draw, drawrate)
    else:
        # call the draw method as fast as possible, i.e. at video frame rate!
        pyglet.clock.schedule(draw)
    # mainloop
    pyglet.app.run()
