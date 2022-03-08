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
from mindaffectBCI.utopiaclient import PredictedTargetProb, PredictedTargetDist
import os
from tkinter.messagebox import askyesno
from mindaffectBCI.noisetag import Noisetag, PredictionPhase
from mindaffectBCI.utopiaclient import DataPacket
from mindaffectBCI.decoder.utils import search_directories_for_file, import_and_make_class
from mindaffectBCI.config_file import load_config, set_args_from_dict, askloadconfigfile

# graphic library
import pyglet
nt = None
window = None
ss = None
nframe = None
isi = 1/60
drawrate = 0  # rate at which draw is called
configmsg = None
last_key_press = None
# global user state for communication between screens etc. 
# N.B. to use in child classes you *must* import this variable directly with : 
# import mindaffectBCI.examples.presentation.selectionMatrix as selectionMatrix
# You can then access this variable and it's state with:
#  selectionMatrix.user_state['my_state'] = 'hello'
user_state = dict()


class Screen:

    '''Screen abstract-class which draws stuff on the screen until finished'''
    def __init__(self, window, label:str=None):
        self.window, self.label = window, label
        if self.label is None: self.label = self.__class__.__name__

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
from enum import IntEnum
class ScreenList(Screen):
    '''screen which iterates through a list of sub-screens'''
    def __init__(self, window:pyglet.window, noisetag:Noisetag, symbols, label:str=None, instruct:str="This is the default start-screen.  Press <space> to continue", **kwargs):
        self.window, self.noisetag, self.symbols, self.label, self.instruct = (window, noisetag, symbols, label, instruct)
        if self.label is None: self.label = self.__class__.__name__

        instruct_screen = InstructionScreen(window, self.instruct, duration = 50000)

        # make a list to store the screens in the order you want to go through them
        self.sub_screens = [instruct_screen]

        self.current_screen_idx = None 
        self.screen = None

    def reset(self):
        super().reset()
        self.current_screen_idx = None
        self.transitionNextPhase()

    def draw(self, t):
        if self.screen is None:
            return
        self.screen.draw(t)
        if self.screen.is_done():
            self.transitionNextPhase()

    def is_done(self):
        """test if this screen is finished, we are done when the
        last sub-screen is done, and we've got no such screens to 
        play

        Returns:
            bool: true if we are finished
        """        
        return self.screen is None

    def transitionNextPhase(self):
        """function to manage the transition between sub-screens.  Override to implement
           your desired screen transition logic.
        """        
        print("stage transition")
        self.current_screen_idx = self.current_screen_idx + 1 if self.current_screen_idx is not None else 0
        if self.current_screen_idx < len(self.sub_screens):
            self.screen = self.sub_screens[self.current_screen_idx]
            self.screen.reset()
        else:
            self.screen = None




from enum import IntEnum
class ScreenSequence(Screen):
    '''screen which manages transitions between sub-screens'''

    class SubScreens(IntEnum):
        ''' enumeration for the different phases of an experiment/BCI application '''
        StartScreen=0

    def __init__(self, window:pyglet.window, noisetag:Noisetag, symbols, label:str=None, **kwargs):
        self.window, self.noisetag, self.symbols = (window, noisetag, symbols)
        if self.label is None: self.label = self.__class__.__name__

        self.instruct_screen = InstructionScreen(window, 'This is a default start screen...\nPress <space> to continue to continue', duration = 50000)

        self.current_screen = self.SubScreens.StartScreen
        self.next_screen = self.SubScreens.StartScreen
        self.screen = None
        self.transitionNextPhase()

    def draw(self, t):
        if self.screen is None:
            return
        self.screen.draw(t)
        if self.screen.is_done():
            self.transitionNextPhase()

    def is_done(self):
        """test if this screen is finished, we are done when the
        last sub-screen is done, and we've got no such screens to 
        play

        Returns:
            bool: true if we are finished
        """        
        return self.screen is None

    def transitionNextPhase(self):
        """function to manage the transition between sub-screens.  Override to implement
           your desired screen transition logic.
        """        
        print("stage transition")

        # move to the next stage
        if self.next_screen is not None:
            self.current_screen = self.next_screen
            self.next_screen = None
        else: # ask the current screen what to do
            self.current_screen = self.current_screen.next_screen
            self.next_screen = None

        if self.current_screen==self.SubScreens.StartScreen: # main menu
            print("start screen")
            self.instruct_screen.reset()
            self.screen = self.instruct_screen
            self.next_screen = None

        else: # end
            print('quit')
            self.screen=None



#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class WaitScreen(Screen):
    '''Screen which shows a blank screen for duration or until key-pressed'''
    def __init__(self, window, duration=5000, waitKey=True, logo="MindAffect_Logo.png", fixation:bool=False, batch=None, group=None):
        super().__init__(window)
        self.t0 = None # timer for the duration
        self.duration, self.waitKey, self.fixation, self.batch, self.group= (duration, waitKey, fixation, batch, group)
        self.isRunning = False
        self.isDone = False
        self.clearScreen = True

        # add the framerate box
        self.framerate=pyglet.text.Label("", font_size=12, x=self.window.width, y=self.window.height,
                                        color=(255, 255, 255, 255),
                                        anchor_x='right', anchor_y='top',
                                        batch=self.batch, group=self.group)
        
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
            self.logo = pyglet.sprite.Sprite(logo,self.window.width,self.window.height-16,
                                        batch=self.batch, group=self.group)
            self.logo.update(scale_x=self.window.width*.1/logo.width, 
                            scale_y=self.window.height*.1/logo.height)

        # make a cross character, with size given by self.fixation
        font_size = self.fixation if self.fixation>1 else 40
        self.fixation_obj = pyglet.text.Label("+", font_size=font_size, 
                                            x=self.window.width//2, y=self.window.height//2,
                                            color=(255, 0, 0, 255),
                                            anchor_x='center', anchor_y='center',
                                        batch=self.batch, group=self.group)

    def reset(self):
        self.isRunning = False
        self.isDone = False

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
        if not self.duration is None and self.elapsed_ms() > self.duration:
            self.isDone = True

        return self.isDone

    def elapsed_ms(self):
        return getTimeStamp()-self.t0 if self.t0 else -1

    def draw(self, t):
        '''Show a block of text to the user for a given duration on a blank screen'''
        if not self.isRunning:
            self.isRunning = True  # mark that we're running
            self.t0 = getTimeStamp()
        if self.clearScreen:
            self.window.clear()

        # check if should update display
        # TODO[]: only update screen 1x / second
        from mindaffectBCI.examples.presentation.selectionMatrix import flipstats, fliplogtime
        flipstats.update_statistics()
        self.framerate.begin_update()
        self.framerate.text = "{:4.1f} +/-{:4.1f}ms".format(flipstats.median,flipstats.sigma)
        self.framerate.end_update()
        self.framerate.draw()

        if self.logo: self.logo.draw()
        if self.fixation and self.fixation_obj: self.fixation_obj.draw()


#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class UserInputScreen(WaitScreen):
    '''Modified screen to enter text'''

    def __init__(self, window, callback, title_text=None, text=None, valid_text=None, duration=150000, logo="Mindaffect_Logo.png"):
        super().__init__(window, duration, False, logo)
        self.valid_text = valid_text
        self.usertext = ''
        self.callback = callback

        # initialize the instructions screen
        self.titleLabel = pyglet.text.Label(x=int(self.window.width*.1),
                                            y=self.window.height,
                                            anchor_x='left',
                                            anchor_y='top',
                                            font_size=24,
                                            color=(255, 255, 255, 255),
                                            multiline=True,
                                            width=int(self.window.width*.8))
        self.set_title(title_text)

        self.inputLabel = pyglet.text.Label(x=self.window.width//2,
                                            y=self.window.height//2,
                                            anchor_x='center',
                                            anchor_y='center',
                                            font_size=24,
                                            color=(255, 255, 255, 255),
                                            multiline=True,
                                            width=int(self.window.width*.8))
        self.set_text(text)

    def set_text(self, text):
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        elif text is None: 
            text=""
        self.inputLabel.begin_update()
        self.inputLabel.text=text
        self.inputLabel.end_update()

    def set_title(self, text):
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        elif text is None: 
            text=""
        self.titleLabel.begin_update()
        self.titleLabel.text=text
        self.titleLabel.end_update()

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        global last_text, last_key_press
        if not self.isRunning:
            WaitScreen.draw(self,t)
            return
        WaitScreen.draw(self,t)

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
                self.isDone = True
                if self.callback is not None:
                    self.callback(self.usertext)
            elif self.window.last_text:
                if self.valid_text is None or self.window.last_text in self.valid_text:
                    # add to the host string
                    self.usertext += window.last_text
            self.window.last_text = None
            self.set_text(self.usertext)

        # draw the screen bits..
        self.titleLabel.draw()
        self.inputLabel.draw()



#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class InstructionScreen(WaitScreen):
    '''Screen which shows a textual instruction for duration or until key-pressed'''
    def __init__(self, window, text, duration=5000, waitKey=True, logo="MindAffect_Logo.png",fixation:bool=False):
        super().__init__(window, duration, waitKey, logo,fixation)
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

    def set_text(self, text):
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        self.instructLabel.begin_update()
        self.instructLabel.text=text
        self.instructLabel.end_update()


    def draw(self, t):
        '''Show a block of text to the user for a given duration on a blank screen'''
        if not self.isRunning:
            self.isRunning = True  # mark that we're running
            self.t0 = getTimeStamp()
        WaitScreen.draw(self,t)
        self.instructLabel.draw()



#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class MenuScreen(InstructionScreen):
    '''Screen which shows a textual instruction for duration or until key-pressed'''
    def __init__(self, window, text, valid_keys):
        super().__init__(window, text, 99999999, True)
        self.valid_keys = valid_keys
        self.key_press = None
        #print("Menu")

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
            self.window.last_text = None

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
    results_text = "Calibration Performance: %3.0f%% Correct\n\n<space> to continue"
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
                    # # P=Target Prediction F=Target Dist S=selection N=new-target M=mode-change E=stimulus-event Q=signal-quality
                    # # self.noisetag.subscribe("MSPQFT")
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
                        self.usertext += self.window.last_text
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

    def __init__(self, window, text, duration=50000, waitKey=True, input_callback=None):
        """simple screen for asking for input from the user

        Args:
            window ([type]): pyglet window
            text ([type]): the text to show
            duration (int, optional): max length of time to show the window. Defaults to 50000.
            waitKey (bool, optional): finish on key-press. Defaults to True.
            input_callback ([type], optional): function to validate user input as string, raise ValueError if invalid input. Defaults to None.
        """        
        super().__init__(window, text, duration, False)
        self.query = text
        self.usertext = ''
        self.input_callback = input_callback

    def reset(self):
        super().reset()
        # clear text on reset
        self.usertext = ''
        self.set_text(self.query)

    def draw(self, t):
        '''grab key presses, process, validate, finish is good.'''
        if self.window.last_key_press:
            if self.window.last_key_press == pyglet.window.key.BACKSPACE:
                self.usertext = self.usertext[:-1]
                self.set_text(self.query +self.usertext)
            self.window.last_key_press = None
        if self.window.last_text:
            # if this is end of input
            if self.window.last_text == '\r' or self.window.last_text == '\n':
                if self.input_callback is None:
                    # No validation so we are done.
                    self.isDone = True
                else:
                    # validate first
                    try:
                        self.input_callback(self.usertext)
                        self.isDone = True
                    except ValueError:
                        # todo []: flash to indicate invalid input
                        self.usertext = ''
                        pass

            elif self.window.last_text:
                # add to the host string
                self.usertext += self.window.last_text
            self.window.last_text=None
            # update display with user input
            self.set_text(self.query + self.usertext)
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
        if nch > 12: # 2-col mode
            col2ch = nch//2
            r = min((winh*.8)/(col2ch+1),winw*.2/2)
        else:
            col2ch = nch+1
            r = min((winh*.8)/(nch+1), winw*.2)
        # make a sprite to draw the electrode qualities
        img = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(2, 2)
        # anchor in the center to make drawing easier
        img.anchor_x = 1
        img.anchor_y = 1
        self.label  = [None]*nch
        self.sprite = [None]*nch
        self.quallabel=[None]*nch
        self.linebbox = [None]*nch # bounding box for the channel line
        for i in range(nch):
            # get the rectangle this channel should remain inside
            if nch<col2ch:
                chrect = [.05*winw,(i+1)*r,.9*winw,r] # xywh
            else:
                if i < col2ch: # left-col
                    chrect = [.025*winw,(i+1)*r,.45*winw,r] # xywh
                else: # right col
                    chrect = [.525*winw,(i-col2ch+1)*r,.45*winw,r] # xywh

            x,y,w,h = chrect
            # channel label name
            self.label[i] = pyglet.text.Label("{:2d}".format(i), font_size=26,
                                            x=x+r*.5, y=y,
                                            color=self.get_linecol(i),
                                            anchor_x='right',
                                            anchor_y='center',
                                            batch=self.batch,
                                            group=self.foreground)

            # signal quality box
            # convert to a sprite and make the right size, N.B. anchor'd to the image center!
            self.sprite[i] = pyglet.sprite.Sprite(img, x=x+r*.8, y=y,
                                                batch=self.batch,
                                                group=self.background)
            # make the desired size
            self.sprite[i].update(scale_x=r*.4/img.width, scale_y=r*.8/img.height)
            # and a text label object
            self.quallabel[i] = pyglet.text.Label("{:2d}".format(i), font_size=15,
                                            x=x+r*.8, y=y,
                                            color=(255,255,255,255),
                                            anchor_x='center',
                                            anchor_y='center',
                                            batch=self.batch,
                                            group=self.foreground)
            # bounding box for the datalines
            self.linebbox[i] = (x+r, y, w-.5*r, h)
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

    def get_linecol(self,i):
        col = [0,0,0,255]; col[i%3]=255
        return col

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
            self.quallabel[i].text = "%2.0f"%(qual)
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
                col = self.get_linecol(ci)
                pyglet.graphics.glColor4d(*col)
                pyglet.gl.glLineWidth(1)
                pyglet.graphics.draw(len(d), pyglet.gl.GL_LINE_STRIP, ('v2f', coords))

                # axes scale
                x = bbox[0]+bbox[2]+20 # at *right* side of the line box
                y = bbox[1]
                pyglet.graphics.glColor4d(255,255,255,255)
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





#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------------------------------------------------------
class SelectionGridScreen(Screen):
    '''Screen which shows a grid of symbols which will be flickered with the noisecode
    and which can be selected from by the mindaffect decoder Brain Computer Interface'''

    LOGLEVEL=0
    def __init__(self, window, noisetag, symbols=None, objIDs=None,
                 bgFraction:float=.2, instruct:str="", label:str=None,
                 clearScreen:bool=True, sendEvents:bool=True, liveFeedback:bool=True, 
                 optosensor:bool=True, fixation:bool=False, state2color:dict=None,
                 target_only:bool=False, show_correct:bool=True, show_newtarget_count:bool=None,
                 waitKey:bool=True, stimulus_callback=None, framerate_display:bool=True,
                 logo:str='MindAffect_Logo.png', media_directories:list=[], font_size:int=None, only_update_on_change:bool=True,
                 self_paced_intertrial:int=-1, self_paced_sentence:str='\n\n\n\nPause.  Press <space> to continue'):
        '''Intialize the stimulus display with the grid of strings in the
            shape given by symbols.
        '''
        self.window, self.symbols, self.noisetag, self.objIDs = ( window, symbols, noisetag, objIDs)
        self.sendEvents, self.liveFeedback, self.optosensor, self.framerate_display, self.logo, self.font_size, self.waitKey, self.stimulus_callback, self.target_only, self.show_correct, self.fixation, self.only_update_on_change, self.label, self.self_paced_intertrial, self.self_paced_sentence, self.media_directories = \
            (sendEvents, liveFeedback, optosensor, framerate_display, logo, font_size, waitKey, stimulus_callback, target_only, show_correct, fixation, only_update_on_change, label, self_paced_intertrial, self_paced_sentence, media_directories)
        # ensure media directories has right format
        if self.media_directories is None: self.media_directories = []
        elif not hasattr(self.media_directories,'__iter__'): self.media_directories=[self.media_directories]
        # create set of sprites and add to render batch
        self.clearScreen= True
        self.isRunning=False
        self.isDone=False
        self.nframe=0
        self.last_target_idx=-1
        self.framestart = getTimeStamp()
        self.frameend = getTimeStamp()
        #self.stimulus_state = None
        self.prev_stimulus_state = None # used to track state changes for display optimization / trigger injection
        self.target_idx = None
        # N.B. noisetag does the whole stimulus sequence
        self.set_noisetag(noisetag)
        # register event handlers
        self.noisetag.addSelectionHandler(self.doSelection)
        self.noisetag.addNewTargetHandler(self.doNewTarget)
        self.noisetag.addPredictionHandler(self.doPrediction)
        self.noisetag.addPredictionDistributionHandler(self.doPredictionDistribution)
        if self.symbols is not None:
            self.set_grid(self.symbols, objIDs, bgFraction, sentence=instruct, logo=logo)
        self.liveSelections = None
        self.feedbackThreshold = .4
        self.last_target_idx = -1
        self.injectSignal = None
        self.show_newtarget_count=show_newtarget_count

        # add new state to color mappings
        if state2color is not None:
            for k,v in state2color.items():
                self.state2color[int(k)]=v

    def reset(self):
        """reset the stimulus state
        """        
        self.isRunning=False
        self.isDone=False
        self.nframe=0
        self.t0=-1
        self.pause=False
        self.self_paced_time_ms=0
        self.last_target_idx=-1
        #self.stimulus_state = None
        if self.show_newtarget_count is not None and not self.show_newtarget_count==False:
            self.show_newtarget_count=0 
        self.prev_stimulus_state = None #self.stimulus_state
        self.set_grid()

    def elapsed_ms(self):
        return self.getTimeStamp()-self.t0 if self.t0 else -1

    def getTimeStamp(self):
        return self.noisetag.getTimeStamp()

    def set_noisetag(self, noisetag:Noisetag):
        """set the noisetag object to use to get/send the stimulus state info

        Args:
            noisetag (Noisetag): the noise tag object
        """        
        self.noisetag=noisetag

    def setliveFeedback(self, value:bool):
        """set flag if we show live prediction feedback (blue-objects) during stimulus

        Args:
            value (bool): live feedback value
        """        
        self.liveFeedback=value

    def setliveSelections(self, value):
        self.liveSelections = value

    def setshowNewTarget(self, value):
        self.show_newtarget_count=value

    def get_idx(self,idx):
        """get a linear index into the self.objects or self.labels arrays from a (i,j) index into self.symbols

        Args:
            idx ([type]): index into symbols

        Returns:
            int: linear index
        """        
        ii=0 # linear index
        for i in range(len(self.symbols)):
            for j in range(len(self.symbols[i])):
                if self.symbols[i][j] is None: continue
                if idx==(i,j) or idx==ii :
                    return ii
                ii = ii + 1
        return None

    def getLabel(self,idx):
        """get the label object at given index

        Args:
            idx ([type]): the index

        Returns:
            pyglet.label: the label object
        """        
        ii = self.get_idx(idx)
        return self.labels[ii] if ii is not None else None

    def setLabel(self,idx,val:str):
        """set the text value of the label at idx

        Args:
            idx ([type]): [description]
            val (str): the label string
        """        
        ii = self.get_idx(idx)
        # update the label object to the new value
        if ii is not None and self.labels[ii]:
            self.labels[ii].text=val

    def setObj(self,idx,val):
        """set the object at the given index

        Args:
            idx ([type]): [description]
            val ([type]): the new object at this index
        """        
        ii = self.get_idx(idx)
        if ii is not None and self.objects[ii]:
            self.objects[ii]=val

    def doSelection(self, objID:int):
        """do processing triggered by selection of an object, e.g. add letter to sentence

        Args:
            objID (int): the objId of the selected letter
        """        
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

    def doNewTarget(self):
        """do processing triggered by starting a new-target sequence, e.g. update the trial counter
        """        
        if not self.show_newtarget_count is None and self.show_newtarget_count:
            self.show_newtarget_count = self.show_newtarget_count+1
            bits = self.sentence.text.split("\n")
            text = "\n".join(bits[:-1]) if len(bits)>1 else bits[0]
            self.set_sentence( "{}\n{}".format(text,self.show_newtarget_count-1) )
        
        # insert a user based wait if wanted
        if self.self_paced_intertrial>0 and self.elapsed_ms() > self.self_paced_intertrial*1000 + self.self_paced_time_ms :
            self.set_sentence( self.sentence.text + self.self_paced_sentence )
            self.pause = True

    def doPredictionDistribution(self, ptd:PredictedTargetDist):
        pass
    def doPrediction(self, pred:PredictedTargetProb):
        pass

    def update_text(self,text:str,sel:str):
        """add 'sel' to the current text with some intelligent replacements

        Args:
            text (str): current text
            sel (str): text to add.  Replacements: <bkspc> -> remove char, <space> = " " etc.

        Returns:
            str: updated text
        """        
        # process special codes
        if sel in ('<-','<bkspc>','<backspace>'):
            text = text[:-1]
        elif sel in ('spc','<space>','<spc>'):
            text = text + ' '
        elif sel == '<comma>':
            text = text + ','
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

    def set_grid(self, symbols=None, objIDs=None, bgFraction:float=.3, sentence:str="What you type goes here", logo=None):
        """set/update the grid of symbols to be selected from

        Args:
            symbols ([type], optional): list-of-list-of-strings for the objects to show. Should be a 2-d array, can use None, for cells with no symbol in. Defaults to None.
            objIDs ([type], optional): list-of-int the BCI object IDs for each of the displayed symbols. Defaults to None.
            bgFraction (float, optional): background fraction -- fraction of empty space between selectable objects. Defaults to .3.
            sentence (str, optional): starting sentence for the top of the screen. Defaults to "What you type goes here".
            logo (str, optional): logo to display at the top of the screen. Defaults to None.
        """        
        winw, winh=self.window.get_size()
        # tell noisetag which objIDs we are using
        if symbols is None:
            symbols = self.symbols

        if isinstance(symbols, str):
            symbols = load_symbols(symbols)

        self.symbols=symbols
        # Number of non-None symbols
        self.nsymb  = sum([sum([(s is not None and not s == '') for s in x ]) for x in symbols])

        if objIDs is not None:
            self.objIDs = objIDs
        else:
            self.objIDs = list(range(1,self.nsymb+1))
            objIDs = self.objIDs
        if logo is None:
            logo = self.logo
        # get size of the matrix
        self.gridheight  = len(symbols) # extra row for sentence
        self.gridwidth = max([len(s) for s in symbols])
        self.ngrid      = self.gridwidth * self.gridheight

        self.noisetag.setActiveObjIDs(self.objIDs)

        # add a background sprite with the right color
        self.objects=[None]*self.nsymb
        self.labels=[None]*self.nsymb
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)

        # init the symbols list -- using the bottom 90% of the screen
        self.init_symbols(symbols, 0, 0, winw, winh*.9, bgFraction )
        # add the other bits
        self.init_opto()
        # sentence in top 10% of screen
        self.init_sentence(sentence, winw*.15, winh, winw*.7, winh*.1 )
        self.init_framerate()
        self.init_logo(logo)
        self.init_fixation(0, 0, winw, winh*.9)

    def init_symbols(self, symbols, x, y, w, h, bgFraction:float=.1, font_size:int=None):
        """setup the display for the given symbols set

        Args:
            symbols ([type]): the symbols to show, inside the given display box
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            bgFraction (float, optional): fraction of empty space between objects. Defaults to .1.
            font_size (int, optional): label font size. Defaults to None.
        """        
        # now create the display objects
        sw = int(w/self.gridwidth) # cell-width
        bgoffsetx = int(sw*bgFraction) # offset within cell for the button
        sh = int(h/self.gridheight) # cell-height
        bgoffsety = int(sh*bgFraction) # offset within cell for the button
        idx=-1
        for i in range(len(symbols)): # rows
            sy = y + (self.gridheight-i-1)*sh # top-edge symbol
            for j in range(len(symbols[i])): # cols
                # skip unused positions
                if symbols[i][j] is None or symbols[i][j]=="": continue
                idx = idx+1
                symb = symbols[i][j]
                sx = x + j*sw # left-edge symbol
                self.objects[idx], self.labels[idx] = self.init_target(symb, 
                                                       sx+bgoffsetx, sy+bgoffsety, 
                                                       int(sw-bgoffsetx*2), int(sh-bgoffsety*2),
                                                       i,j,font_size=font_size)

    def init_target(self, symb:str, x, y, w, h, i=None,j=None, font_size:int=None):
        """ setup the display of a single target 'button' inside the given display box 

        Args:
            symb (str): the symbol to show at this target
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            i,j (int): row/col of this target in the grid
            font_size (int, optional): [description]. Defaults to None.

        Returns:
            (sprite, label): background sprite, and foreground text label
        """        
        sprite, symb = self.init_sprite(symb, x, y, w, h)
        label = self.init_label(symb, x, y, w, h, font_size)
        return sprite, label

    def init_label(self, symb, x, y, w, h, font_size=None):
        """ setup the display of a single target foreground label inside the given display box 

        Args:
            symb (str): the symbol to show at this target
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            font_size (int, optional): [description]. Defaults to None.

        Returns:
            (sprite, label): background sprite, and foreground text label
        """        
        # add the foreground label for this cell, and add to drawing batch
        symb = str(symb)
        if font_size is None: font_size = self.font_size
        if font_size is None or font_size == 'auto':
            font_size = int(min(w,h)*.75*72/96/max(1,len(symb)))
        label=pyglet.text.Label(symb, font_size=font_size, x=x+w/2, y=y+h/2,
                                color=(255, 255, 255, 255),
                                anchor_x='center', anchor_y='center',
                                batch=self.batch, group=self.foreground)
        return label

    def init_sprite(self, symb, x, y, w, h, scale_to_fit=True):
        """setup the background image for this target 'button', inside the given display box

        Args:
            symb ([type]): [description]
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            scale_to_fit (bool, optional): do we scale the image to fix in the display box. Defaults to True.

        Returns:
            sprite: the background image sprite
        """        
        try : # symb is image to use for this button
            img = search_directories_for_file(symb,os.path.dirname(__file__),
                                              os.path.join(os.path.dirname(__file__),'images'),
                                              *self.media_directories)
            img = pyglet.image.load(img)
            symb = '.' # symb is a fixation dot
        except :
            # create a 1x1 white image for this grid cell
            img = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(2, 2)
        img.anchor_x = img.width//2
        img.anchor_y = img.height//2
        # convert to a sprite (for fast re-draw) and store in objects list
        # and add to the drawing batch (as background)
        sprite=pyglet.sprite.Sprite(img, x=x+w/2, y=y+h/2,
                                    batch=self.batch, group=self.background)
        sprite.w = w
        sprite.h = h
        # re-scale (on GPU) to the size of this grid cell
        if scale_to_fit:
            sprite.update(scale_x=int(sprite.w)/sprite.image.width, scale_y=int(sprite.h)/sprite.image.height)
        return sprite, symb


    def init_framerate(self):
        """create the display label for the framerate feedback box
        """        
        winw, winh=self.window.get_size()
        # add the framerate box
        self.framerate=pyglet.text.Label("", font_size=12, x=winw, y=winh,
                                        color=(255, 255, 255, 255),
                                        anchor_x='right', anchor_y='top',
                                        batch=self.batch, group=self.foreground)

    def init_sentence(self,sentence, x, y, w, h, font_size=32):
        """create the display lable for the current spelled sentence, inside the display box

        Args:
            sentence ([type]): the starting sentence to show
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            font_size (int, optional): [description]. Defaults to 32.
        """        
        self.sentence=pyglet.text.Label(sentence, font_size=font_size, 
                                        x=x, y=y, width=w, height=h,
                                        color=(255, 255, 255, 255),
                                        anchor_x='left', anchor_y='top',
                                        multiline=True,
                                        batch=self.batch, group=self.foreground)

    def init_logo(self,logo):
        """init the sprite for the logo display

        Args:
            logo ([type]): [description]
        """        
        # add a logo box
        if isinstance(logo,str): # filename to load
            logo = search_directories_for_file(logo,os.path.dirname(__file__),
                                                os.path.join(os.path.dirname(__file__),'..','..'))
            try :
                logo = pyglet.image.load(logo)
                logo.anchor_x, logo.anchor_y  = (logo.width,logo.height) # anchor top-right 
                self.logo = pyglet.sprite.Sprite(logo, self.window.width, self.window.height-16,
                                                 batch=self.batch, group=self.background) # sprite a window top-right
            except :
                self.logo = None
        if self.logo:
            self.logo.batch = self.batch
            self.logo.group = self.foreground
            self.logo.update(x=self.window.width,  y=self.window.height-16,
                            scale_x=self.window.width*.1/self.logo.image.width, 
                            scale_y=self.window.height*.1/self.logo.image.height)

    def init_opto(self):
        """initialize the sprite for the opto-sensor display
        """        
        winw, winh=self.window.get_size()
        self.opto_sprite, _ = self.init_sprite(1,0,winh*.9,winw*.1,winh*.1)
        self.opto_sprite.visible=False

    def init_fixation(self,x,y,w,h):
        self.fixation_obj = None
        # add a fixation point in the middle of the grid
        if isinstance(self.fixation,str):
            # load the given image
            self.fixation_obj = self.init_sprite(self.fixation,x,y,w,h)[0]

        elif self.fixation is not None and not self.fixation == 0:
            # make a cross character, with size given by self.fixation
            font_size = 40 if self.fixation == True else self.fixation
            self.fixation_obj = self.init_label("+",x=x,y=y,w=w,h=h,font_size=font_size)
            self.fixation_obj.color = (255,0,0,255)


    def is_done(self):
        if self.isDone:
            self.noisetag.modeChange('idle')
            self.injectSignal=None
        return self.isDone

    # mapping from bci-stimulus-states to display color
    state2color={0:(5, 5, 5),       # off=grey
                 1:(255, 255, 255), # on=white
                 #2:(0, 255, 0),     # cue=green
                 #3:(0, 0, 255),     # feedback=blue
                 254:(0,255,0),     # cue=green
                 255:(0,0,255),     # feedback=blue
                 None:(100,0,0)}    # red(ish)=task     
    def update_object_state(self, idx:int, state:int):
        """update the idx'th object to stimulus state state

            N.B. override this method to implement different state dependent stimulus changes, e.g. size changes, background images, rotations

            This version changes the sprite color based on the requested state.  By either;
               1) getting the color to use from the state2color class dictionary, i.e color=self.state2color[state]
               2) setting the background lumaniance color to state (for floating point states), i.e. color= color*state
        Args:
            idx (int): index of the object to update
            state (int or float): the new desired object state
        """        
        if self.objects[idx]:
            if isinstance(state,int): # integer state, map to color lookup table
                self.objects[idx].color = self.state2color.get(state,self.state2color[None])
            elif isinstance(state,float): # float state, map to intensity
                self.objects[idx].color = tuple(int(c*state) for c in self.state2color[1])                
        if self.labels[idx]:
            self.labels[idx].color=(255,255,255,255) # reset labels

    def get_stimtime(self):
        """get the timestamp of the start of the last stimulus event, i.e. the last window flip for visual stimuli 

        Returns:
            int: time-stamp for the last stimulus start
        """        
        winflip = self.window.lastfliptime
        if winflip > self.framestart or winflip < self.frameend:
            print("Error: frameend={} winflip={} framestart={}".format(self.frameend,winflip,self.framestart))
        return winflip


    def draw(self, t):
        """draw the letter-grid with given stimulus state for each object.
        Note: To maximise timing accuracy we send the info on the grid-stimulus state
        at the start of the *next* frame, as this happens as soon as possible after
        the screen 'flip'. """
        if not self.isRunning:
            self.isRunning=True
            self.t0 = self.getTimeStamp()
        self.framestart=self.getTimeStamp()
        if self.pause:
            #start again if key is pressed
            if self.window.last_key_press:
                self.window.last_key_press = None
                self.pause=False
                # remove the cue text from the sentence
                sentence = self.sentence.text
                sentence = sentence[:-len(self.self_paced_sentence)]
                self.set_sentence( sentence )
                # record the time the user ended the cue
                self.self_paced_time_ms = self.elapsed_ms()
            else:
                self.do_draw()
                return

        stimtime = self.get_stimtime()
        self.nframe = self.nframe+1
        if self.sendEvents:
            self.noisetag.sendStimulusState(timestamp=stimtime,injectSignal=self.injectSignal)

        # get the current stimulus state to show
        try:
            self.noisetag.updateStimulusState()
            stimulus_state, target_idx, objIDs, sendEvents=self.noisetag.getStimulusState()
            if stimulus_state:
                stimulus_state = [ s for s in stimulus_state ] # BODGE: deep copy the stim-state to avoid changing noisetag
            target_state = stimulus_state[target_idx] if target_idx>=0 and len(stimulus_state)>0 else -1
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
                if self.target_only and not target_idx == idx :
                    stimulus_state[idx]=0
                # Optimization: only update if the state has changed
                if not self.only_update_on_change: # default to always update
                    self.update_object_state(idx,stimulus_state[idx])
                else:
                    if not hasattr(self,'prev_stimulus_state') or self.prev_stimulus_state is None or not self.prev_stimulus_state[idx] == stimulus_state[idx]:
                        self.update_object_state(idx,stimulus_state[idx])
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
                    fbcol[0]=int(fbcol[0]*.4) 
                    fbcol[1]=int(fbcol[1]*.4) 
                    fbcol[2]=int(fbcol[2]*.4+255*(1-prederr)*.6) 
                    self.labels[predidx].color = fbcol

        # disp opto-sensor if targetState is set
        if self.optosensor is not None and self.optosensor is not False and self.opto_sprite is not None:
            self.opto_sprite.visible=False  # default to opto-off
            
            opto_state = None
            if self.optosensor is True:
                # use the target state
                if target_state is not None and target_state >=0 and target_state <=1:
                    print("*" if target_state>.5 else '.', end='', flush=True)
                    opto_state = target_state
                else:
                    opto_state = sum(stimulus_state)

            elif isinstance(self.optosensor,int):
                # guarded array acess
                opto_state = stimulus_state[self.optosensor] if len(stimulus_state)>self.optosensor else None

            if opto_state is not None:
                self.opto_sprite.visible=True
                if opto_state <= 1.0: # pure on-off
                    self.opto_sprite.color = tuple(int(c*opto_state) for c in (255, 255, 255))
                else: # level -> intensity
                    self.opto_sprite.color = tuple(min(255,int(c*opto_state/100)) for c in (255,255,255))


        # record the previous and current stimulus state
        self.prev_stimulus_state = [s for s in stimulus_state] if stimulus_state else None # copy! old stimulus-state
        #self.stimulus_state = stimulus_state # current stimulus-state
        self.target_idx = target_idx

        # do the draw
        self.do_draw()


    def do_draw(self):
        """do the actual drawing of the current display state
        """        
        self.batch.draw()
        #if self.logo: self.logo.draw()
        #if self.fixation_obj: self.fixation_obj.draw()
        self.frameend=self.getTimeStamp()
        # add the frame rate info
        # TODO[]: limit update rate..
        if self.framerate_display:
            from mindaffectBCI.examples.presentation.selectionMatrix import flipstats
            flipstats.update_statistics()
            self.set_framerate("{:4.1f} +/-{:4.1f}ms".format(flipstats.median,flipstats.sigma))                



#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#---------------------------------------------------------
from enum import IntEnum, auto
class ExptScreenManager(ScreenSequence):
    '''class to manage a whole application, with main menu, checks, etc.'''

    class SubScreens(IntEnum):
        ''' enumeration for the different phases of an experiment/BCI application '''
        MainMenu=auto()
        Connecting=auto()
        SignalQuality=auto()
        CalInstruct=auto()
        Calibration=auto()
        CalResults=auto()
        CuedPredInstruct=auto()
        CuedPrediction=auto()
        PredInstruct=auto()
        Prediction=auto()
        PracticeInstruct=auto()
        Practice=auto()
        Closing=auto()
        Quit=auto()
        Welcome=auto()
        Minimize=auto()
        Settings=auto()
        LogMessage=auto()
        FrameRateCheck=auto()
        Reset=auto()
        ExtraSymbols=auto()
        ExtraMenuScreen=auto()

    welcomeInstruct="Welcome to the mindaffectBCI\n\n<space> to continue"
    calibrationInstruct="Calibration\n\nThe next stage is CALIBRATION\nlook at the indicated green target\n\n<space> to continue"
    cuedpredictionInstruct="Prediction\n\nThe next stage is CUED PREDICTION\nLook at the green cued letter\n\nLive BCI feedback in blue\n\n<space> to continue"
    predictionInstruct="Prediction\n\nThe next stage is free PREDICTION\nLook at the letter you want to select\nLive BCI feedback in blue\n\n<space> to continue"
    closingInstruct="Closing\nThankyou\n\nPress to exit"
    resetInstruct="Reset\n\nThe decoder model has been reset.\nYou will need to run calibration again to use the BCI\n\n<space> to continue"
    calibrationSentence='Calibration: look at the green cue.\n'
    cuedpredictionSentence='CuedPrediction: look at the green cue.\n'
    predictionSentence='Your Sentence:\n'

    main_menu_header =["Welcome to the mindaffectBCI", "", "", "Press the number of the option you want:", ""]
    main_menu_footer = ["", 
                "", 
                "",
                "Q) Quit",
                "f) frame-rate-check",
                "s) settings",
                "r) reset calibration model",
                "L) send log message"]
    menu_keys_footer = {pyglet.window.key.F:SubScreens.FrameRateCheck,
                pyglet.window.key.S:SubScreens.Settings,
                pyglet.window.key.R:SubScreens.Reset,
                pyglet.window.key.Q:SubScreens.Quit,
                pyglet.window.key.L:SubScreens.LogMessage}

    def __init__(self, window:pyglet.window, noisetag:Noisetag, symbols, nCal:int=None, ncal:int=1, npred:int=1, nPred:int=None, 
                 calibration_trialduration:float=4.2, prediction_trialduration:float=10,  
                 feedbackduration:float=None, cueduration:float=None, intertrialduration:float=None,
                 framesperbit:int=None, fullscreen_stimulus:bool=True, 
                 start_screen:int=None,
                 selectionThreshold:float=.1, optosensor:bool=True,  
                 calibration_screen:Screen="mindaffectBCI.examples.presentation.selectionMatrix.SelectionGridScreen", calibration_screen_args:dict=dict(), calibration_sentence:str=None, calibration_instruct:str=None,
                 prediction_screen:Screen="mindaffectBCI.examples.presentation.selectionMatrix.SelectionGridScreen", prediction_screen_args:dict=dict(),
                 cued_prediction_screen:Screen=True,
                 stimseq:str=None, stimfile:str=None, calibration_stimseq:str=None, prediction_stimseq:str=None,
                 simple_calibration:bool=None, calibration_symbols=None, 
                 extra_symbols=None, extra_screens=None, extra_stimseqs=None, extra_labels=None,  extra_screen_args=None,
                 bgFraction=.1,
                 calibration_args:dict=None, prediction_args:dict=None,
                 calibrationInstruct:str=None, cuedpredictionInstruct:str=None, predictionInstruct:str=None,
                 calibrationSentence:str=None, cuedpredictionSentence:str=None, predictionSentence:str=None,
                 config_file:str=None):
        self.window = window
        self.noisetag = noisetag
        self.symbols = symbols
        self.optosensor = optosensor
        self.calibration_symbols = calibration_symbols if calibration_symbols is not None else symbols

        # get and instantiate the various stimulus sequences
        self.stimseq = stimseq if not stimseq is None else stimfile
        self.calibration_stimseq = calibration_stimseq
        self.prediction_stimseq = prediction_stimseq

        self.bgFraction = bgFraction
        self.start_screen = start_screen


        if calibrationInstruct is not None:  self.calibrationInstruct=calibrationInstruct
        if cuedpredictionInstruct is not None:  self.cuedpredictionInstruct=cuedpredictionInstruct
        if predictionInstruct is not None:  self.predictionInstruct=predictionInstruct
        if calibrationSentence is not None:  self.calibrationSentence=calibrationSentence
        if cuedpredictionSentence is not None:  self.cuedpredictionSentence=cuedpredictionSentence
        if predictionSentence is not None:  self.predictionSentence=predictionSentence

        # initialize any extra screens we want
        self.extra_symbols = extra_symbols
        self.extra_stimseqs = extra_stimseqs
        if extra_symbols:
            if not extra_screens is None:
                extra_screen_cls = extra_screens 
            else:
                extra_screen_cls = ['mindaffectBCI.examples.presentation.selectionMatrix.SelectionGridScreen']
            # convert to a set of screens 
            extra_screens = []
            for i,symbols in enumerate(extra_symbols):
                cls = extra_screen_cls[i] if i < len(extra_screen_cls) else extra_screen_cls[-1]
                if extra_screen_args : 
                    args = extra_screen_args[i] if i < len(extra_screen_args) else extra_screen_args[-1]
                else:
                    args = dict()
                # default to general values is not set
                if not 'symbols' in args: args['symbols']=symbols
                if not 'optosensor' in args: args['optosensor']=self.optosensor                
                scr = import_and_make_class(cls, window=window, noisetag=noisetag, **args)
                scr.label = symbols if extra_labels is None else extra_labels[i]
                extra_screens.append(scr)

        # Make the extra screens
        if extra_screens is not None:
            for i,s in enumerate(extra_screens):
                print(" Extra-screen {} = {}\n\n".format(i,s))
                if isinstance(s,str):
                    if extra_screen_args : 
                        args = extra_screen_args[i] if i < len(extra_screen_args) else extra_screen_args[-1]
                    else:
                        args = dict()
                    # remove duplicates.
                    if not 'symbols' in args: args['symbols']=self.symbols
                    if not 'optosensor' in args: args['optosensor']=self.optosensor
                    print("Making screen: {} ({})".format(s,args))
                    extra_screens[i] = import_and_make_class(s, window=window, noisetag=noisetag, **args)
                    if extra_labels is not None:
                        extra_screens[i].label = extra_labels[i]
        self.extra_screens = extra_screens

        # Make the calibration screen
        if not 'symbols' in calibration_screen_args: calibration_screen_args['symbols']=self.symbols
        if not 'optosensor' in calibration_screen_args: calibration_screen_args['optosensor']=self.optosensor
        if calibration_screen is not None:
            if isinstance(calibration_screen,str):
                calibration_screen = import_and_make_class(calibration_screen,window=window,noisetag=noisetag,**calibration_screen_args)
        else:
            # no calibration option
            self.calibration_screen = None
        self.calibration_screen = calibration_screen
        
        # Make the prediction screen
        if prediction_screen is None and prediction_screen_args is not None:
            prediction_screen = calibration_screen
        if prediction_screen is not None:
            if not 'symbols' in prediction_screen_args: prediction_screen_args['symbols']=self.symbols
            if not 'optosensor' in prediction_screen_args: prediction_screen_args['optosensor']=self.optosensor
            if isinstance(prediction_screen,str):
                prediction_screen = import_and_make_class(prediction_screen,window=window,noisetag=noisetag,**prediction_screen_args)
        self.prediction_screen = prediction_screen
        if cued_prediction_screen == True:
            cued_prediction_screen = self.prediction_screen
        elif isinstance(cued_prediction_screen,str):
            cued_prediction_screen = import_and_make_class(prediction_screen,window=window,noisetag=noisetag,**prediction_screen_args)
        self.cued_prediction_screen = cued_prediction_screen

        self.init_main_menu_numbered()

        self.menu = MenuScreen(window, 
                                self.main_menu_header+self.main_menu_numbered+self.main_menu_footer, 
                                tuple(self.menu_keys.keys()) + tuple(self.menu_keys_footer.keys()))
        self.instruct = InstructionScreen(window, '', duration = 50000)
        self.connecting = ConnectingScreen(window, noisetag)
        self.electquality = ElectrodequalityScreen(window, noisetag)
        self.results = ResultsScreen(window, noisetag)
        if self.extra_menu_numbered is not None:
            self.extra_menu = MenuScreen(window, self.extra_menu_numbered, self.extra_menu_keys.keys())
        else:
            self.extra_menu = None

        self.ncal = ncal if nCal is None else nCal
        self.npred = npred if nPred is None else nPred
        self.framesperbit = framesperbit
        self.calibration_trialduration = calibration_trialduration
        self.prediction_trialduration = prediction_trialduration
        self.feedbackduration = feedbackduration
        self.cueduration = cueduration
        self.intertrialduration = intertrialduration
        self.calibration_args = calibration_args if calibration_args else dict()
        self.prediction_args = prediction_args if prediction_args else dict()
        self.calibration_args['nTrials']=self.ncal
        self.calibration_args['stimSeq']=self.calibration_stimseq if self.calibration_stimseq is not None else self.stimseq

        self.prediction_args['nTrials']=self.npred        
        self.prediction_args['stimSeq']=self.prediction_stimseq if self.prediction_stimseq is not None else self.stimseq
        if self.framesperbit is not None:
            self.calibration_args['framesperbit'] = self.framesperbit
            self.prediction_args['framesperbit'] = self.framesperbit
        if self.calibration_trialduration is not None:
            self.calibration_args['numframes'] = self.calibration_trialduration / isi
        if self.prediction_trialduration is not None:
            self.prediction_args['numframes'] = self.prediction_trialduration / isi
        if self.feedbackduration is not None:
            self.prediction_args['feedbackframes'] = self.feedbackduration / isi
        if self.cueduration is not None:
            self.calibration_args['cueframes'] = self.cueduration / isi
            self.prediction_args['cueframes'] = self.cueduration / isi
        if self.intertrialduration is not None:
            self.calibration_args['intertrialframes'] = self.intertrialduration / isi
            self.prediction_args['intertrialframes'] = self.intertrialduration / isi
        if selectionThreshold is not None:
            self.calibration_args['selectionThreshold'] = selectionThreshold
            self.prediction_args['selectionThreshold'] = selectionThreshold

        self.confirm_quit = None
        self.fullscreen_stimulus = fullscreen_stimulus
        self.selectionThreshold = selectionThreshold
        self.simple_calibration = simple_calibration
        self.screen = None

        # start with the connecting screen
        self.current_screen = self.SubScreens.Connecting
        self.next_screen = self.SubScreens.Connecting

        self.transitionNextPhase()

    def init_main_menu_numbered(self):
        main_menu_numbered = ["0) Electrode Quality",
            "1) Calibration",
            "2) Cued Prediction" ,
            "3) Free Typing",
            "4) Practice"]
        menu_keys = {pyglet.window.key._0:self.SubScreens.SignalQuality,
                    pyglet.window.key.NUM_0:self.SubScreens.SignalQuality}
        extra_menu_numbered = []
        extra_menu_keys = dict()

        # build the main menu options
        if self.calibration_screen is not None: 
            if hasattr(self.calibration_screen,'label') and not self.calibration_screen.label is None:
                main_menu_numbered[1] = "{}) {}".format(1,self.calibration_screen.label)
            # add to the key-menu
            menu_keys.update({pyglet.window.key._1:self.SubScreens.CalInstruct,
                             pyglet.window.key.NUM_1:self.SubScreens.CalInstruct,
                             pyglet.window.key._4:self.SubScreens.PracticeInstruct,
                             pyglet.window.key.NUM_4:self.SubScreens.PracticeInstruct})
        else: # No calibration mode
            for i,s in enumerate(main_menu_numbered):
                if 'calibration' in s.lower():
                    main_menu_numbered[i]=""
                if 'practice' in s.lower():
                    main_menu_numbered[i]=""

        if self.prediction_screen is not None:
            if hasattr(self.prediction_screen,'label') and not self.prediction_screen.label is None:
                main_menu_numbered[3] = "{}) {}".format(3,self.prediction_screen.label)
            menu_keys.update({pyglet.window.key._3:self.SubScreens.PredInstruct,
                              pyglet.window.key.NUM_3:self.SubScreens.PredInstruct})
        else: # No prediction mode
            for i,s in enumerate(main_menu_numbered):
                if 'prediction' in s.lower():
                    main_menu_numbered[i]=""
            # TODO[]: remove from menu-keys?
        if self.cued_prediction_screen is not None: # remove the cued-prediction option
            if hasattr(self.cued_prediction_screen,'label') and not self.cued_prediction_screen.label is None:
                main_menu_numbered[2] = "{}) {} (Cued)".format(2,self.cued_prediction_screen.label)
            menu_keys.update({pyglet.window.key._2:self.SubScreens.CuedPredInstruct,
                             pyglet.window.key.NUM_2:self.SubScreens.CuedPredInstruct})
        else: # no cued prediction
            for i,s in enumerate(main_menu_numbered):
                if 'cued' in s.lower():
                    main_menu_numbered[i]=""

        if self.extra_screens is not None:
            if len(self.extra_screens)<=5: # few enough for direct selection
                extra_menu_numbered = None  # don't use extra menu
                for i,ps in enumerate(self.extra_screens):
                    keyi = i + 5
                    label = getattr(ps,'label',type(ps).__name__) # use label slot, fallback on class name
                    main_menu_numbered.append("{}) {}".format(keyi,label))
                    menu_keys[getattr(pyglet.window.key,"_{:d}".format(keyi))] = self.SubScreens.ExtraSymbols
                    menu_keys[getattr(pyglet.window.key,"NUM_{:d}".format(keyi))] = self.SubScreens.ExtraSymbols
                    extra_menu_keys[getattr(pyglet.window.key,"_{:d}".format(keyi))] = i # map key-symb to srcreen to run
                    extra_menu_keys[getattr(pyglet.window.key,"NUM_{:d}".format(keyi))] = i
            else: # add to sub-menu
                main_menu_numbered.append("{}) Extra options".format('9'))
                menu_keys[pyglet.window.key._9] = self.SubScreens.ExtraMenuScreen
                menu_keys[pyglet.window.key.NUM_9] = self.SubScreens.ExtraMenuScreen

                for i,ps in enumerate(self.extra_screens):
                    keyi = i
                    label = getattr(ps,'label',type(ps).__name__) # use label slot, fallback on class name
                    extra_menu_numbered.append("{}) {}".format(keyi,label))
                    extra_menu_keys[getattr(pyglet.window.key,"_{:d}".format(keyi))] = keyi
                    extra_menu_keys[getattr(pyglet.window.key,"NUM_{:d}".format(keyi))] = keyi
                extra_menu_numbered.append("")
                extra_menu_numbered.append('<bkspc> to return to main menu')
                extra_menu_keys[pyglet.window.key.BACKSPACE]=None

        self.extra_menu_numbered = extra_menu_numbered
        self.extra_menu_keys = extra_menu_keys
        # set the updated numbered menu sequence
        self.main_menu_numbered = main_menu_numbered
        self.menu_keys = menu_keys
        self.extra_menu_numbered = extra_menu_numbered
        self.extra_menu_keys = extra_menu_keys


    def draw(self, t):
        if self.screen is None:
            return
        if self.screen.is_done():
            self.transitionNextPhase()
        self.screen.draw(t)

    def is_done(self):
        return self.screen is None

    def transitionNextPhase(self):
        print("stage transition")

        # move to the next stage
        if self.next_screen is not None:
            self.current_screen = self.next_screen
            self.next_screen = None
        else: # assume it's from the menu
            self.current_screen = self.menu_keys.get(self.menu.key_press,self.menu_keys_footer.get(self.menu.key_press,self.SubScreens.MainMenu))
            self.next_screen = None

        if self.current_screen==self.SubScreens.MainMenu: # main menu
            if self.fullscreen_stimulus==True :
                self.window.set_fullscreen(fullscreen=False)
            global configmsg
            if configmsg is not None:
                self.noisetag.log(configmsg)
                configmsg = None

            print("main menu")
            self.menu.reset()
            self.screen = self.menu
            self.noisetag.modeChange('idle')
            self.next_screen = None
            

        elif self.current_screen==self.SubScreens.Welcome: # welcome instruct
            print("welcome instruct")
            self.instruct.set_text(self.welcomeInstruct)
            self.instruct.reset()
            self.screen = self.instruct
            self.next_screen = self.SubScreens.Connecting

        elif self.current_screen==self.SubScreens.Reset: # reset the decoder
            print("reset")
            self.instruct.set_text(self.resetInstruct)
            self.instruct.reset()
            self.screen = self.instruct
            self.noisetag.modeChange("reset")
            self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==self.SubScreens.Connecting: # connecting instruct
            print("connecting screen")
            self.connecting.reset()
            self.screen = self.connecting

            if self.start_screen is not None:
                if isinstance(self.start_screen, str):
                    if self.start_screen.lower() == 'signal_quality':
                        self.next_screen = self.SubScreens.SignalQuality
                    elif self.start_screen.lower() == 'main_menu':
                        self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==self.SubScreens.SignalQuality: # electrode quality
            print("signal quality")
            self.electquality.reset()
            self.screen=self.electquality
            self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen in (self.SubScreens.CalInstruct,self.SubScreens.PracticeInstruct) : # calibration instruct
            print("Calibration instruct")
            if self.calibration_screen is None:
                self.next_screen = self.SubScreens.MainMenu

            else:
                if self.fullscreen_stimulus==True :
                    self.window.set_fullscreen(fullscreen=True)
                self.instruct.set_text(self.calibrationInstruct)
                self.instruct.reset()
                self.screen=self.instruct
                if self.current_screen == self.SubScreens.CalInstruct:
                    self.next_screen = self.SubScreens.Calibration
                else:
                    self.next_screen = self.SubScreens.Practice

        elif self.current_screen in (self.SubScreens.Calibration, self.SubScreens.Practice): # calibration
            print("calibration")
            screen = self.calibration_screen
            screen.reset()
            screen.set_grid(symbols=self.calibration_symbols, bgFraction=self.bgFraction)
            screen.setliveFeedback(False)
            screen.setshowNewTarget(True)
            if self.simple_calibration is not None:
                screen.target_only=self.simple_calibration
            screen.set_sentence(self.calibrationSentence)
            if self.current_screen == self.SubScreens.Calibration:
                screen.sendEvents=True
            else:
                screen.sendEvents=False

            screen.noisetag.startCalibration(**self.calibration_args)
            self.screen = screen
            if self.current_screen == self.SubScreens.Calibration:
                self.next_screen = self.SubScreens.CalResults
            else:
                self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==self.SubScreens.CalResults: # Calibration Results
            print("Calibration Results")
            if self.fullscreen_stimulus==True :
                self.window.set_fullscreen(fullscreen=True)
            self.results.reset()
            self.results.waitKey=True
            self.screen=self.results
            self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==self.SubScreens.CuedPredInstruct: # pred instruct
            print("cued pred instruct")
            if self.fullscreen_stimulus==True :
                self.window.set_fullscreen(fullscreen=True)
            self.instruct.set_text(self.cuedpredictionInstruct)
            self.instruct.reset()
            self.screen=self.instruct
            self.next_screen = self.SubScreens.CuedPrediction

        elif self.current_screen==self.SubScreens.CuedPrediction: # pred
            print("cued prediction")
            screen = self.prediction_screen
            screen.reset()
            screen.set_grid(symbols=self.symbols, bgFraction=self.bgFraction)
            screen.liveFeedback=True
            screen.setliveSelections(True)
            #screen.setshowNewTarget(False)
            screen.target_only=False
            screen.show_correct=True
            screen.set_sentence(self.cuedpredictionSentence)

            screen.noisetag.startPrediction(cuedprediction=True, **self.prediction_args)
            self.screen = screen
            self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==self.SubScreens.PredInstruct: # pred instruct
            print("pred instruct")
            if self.fullscreen_stimulus==True :
                self.window.set_fullscreen(fullscreen=True)
            self.instruct.set_text(self.predictionInstruct)
            self.instruct.reset()
            self.screen=self.instruct
            self.next_screen = self.SubScreens.Prediction

        elif self.current_screen==self.SubScreens.Prediction: # pred
            print("prediction")
            screen = self.prediction_screen
            screen.reset()
            screen.set_grid(symbols=self.symbols, bgFraction=self.bgFraction)
            screen.liveFeedback=True
            screen.target_only=False
            screen.show_correct=False
            screen.set_sentence(self.predictionSentence)
            screen.setliveSelections(True)
            #screen.setshowNewTarget(False)

            screen.noisetag.startPrediction(**self.prediction_args)
            self.screen = screen
            self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==self.SubScreens.ExtraMenuScreen:
            # indirect call to extrasymbols 
            self.screen = self.extra_menu
            self.screen.reset()
            self.next_screen = self.SubScreens.ExtraSymbols

        elif self.current_screen==self.SubScreens.ExtraSymbols: # pred
            print("Extra Prediction")
            key_press = self.extra_menu.key_press if self.extra_menu is not None else self.menu.key_press
            extrai = self.extra_menu_keys.get(key_press,None)
            if extrai is not None:
                screen = self.extra_screens[extrai]
                screen.reset()
                
                screen.liveFeedback=True
                screen.target_only=False
                screen.show_correct=False
                screen.set_sentence('')
                screen.setliveSelections(True)

                extra_args = {k:v for k,v in self.prediction_args.items()}
                if self.extra_stimseqs and len(self.extra_stimseqs)>extrai:
                    extra_args['stimSeq']=self.extra_stimseqs[extrai]
                screen.noisetag.startPrediction(**extra_args)

                if self.fullscreen_stimulus==True :
                    self.window.set_fullscreen(fullscreen=True)
                    
                self.screen = screen
            self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==self.SubScreens.Closing: # closing instruct
            print("closing instruct")
            self.instruct.set_text(self.closingInstruct)
            self.instruct.reset()
            self.screen=self.instruct
            self.next_screen = self.SubScreens.Quit

        elif self.current_screen==self.SubScreens.Minimize: # closing instruct
            print("minimize")
            self.window.minimize()
            self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==None: # testing stages..
            #print("flicker with selection")
            #self.selectionGrid.noisetag.startFlickerWithSelection(numframes=10/isi)
            print("single trial")
            screen = self.prediction_screen
            screen.set_grid([[None, 'up', None],
                                         ['left', 'fire', 'right']])
            screen.noisetag.startSingleTrial(numframes=10/isi)
            # N.B. ensure decoder is in prediction mode!
            screen.noisetag.modeChange('Prediction.static')
            screen.reset()
            self.screen = screen

        elif self.current_screen==self.SubScreens.FrameRateCheck: # frame-rate-check
            print("frame-rate-check")
            #from mindaffectBCI.examples.presentation.framerate_check import FrameRateTestScreen
            self.screen=FrameRateTestScreen(self.window,waitKey=True)
            self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==self.SubScreens.Settings: # config settings
            print("settings")
            self.screen = SettingsScreen(self.window, self)
            self.next_screen = self.SubScreens.MainMenu

        elif self.current_screen==self.SubScreens.LogMessage: # log message
            print("settings")
            doLogMessage = lambda x: self.noisetag.log(x)
            self.screen = UserInputScreen(self.window, doLogMessage, "Enter your log message\n\nEnter to send")
            self.next_screen = self.SubScreens.MainMenu

        else: # end
            if self.confirm_quit is None:
                from tkinter import Tk
                from tkinter.messagebox import askyesno
                root = Tk()
                root.withdraw()
                self.confirm_quit = askyesno(title='Confirmation',
                                            message='Are you sure that you want to quit?')
                if self.confirm_quit:
                    print('quit')
                    self.screen=None
                    self.confirm_quit = None
                else: 
                    self.next_screen = self.SubScreens.MainMenu


#------------------------------------------------------------------------
# Initialization: display, utopia-connection
# use noisetag object as time-stamp provider
def getTimeStamp():
    global nt
    if nt and nt.isConnected():
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
    from mindaffectBCI.examples.presentation.selectionMatrix import flipstats, fliplogtime
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
    try:
        config = pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4)
        if fullscreen:
            # N.B. accurate video timing only guaranteed with fullscreen
            # N.B. over-sampling seems to introduce frame lagging on windows+Intell
            window = pyglet.window.Window(fullscreen=True, vsync=True, resizable=False, config=config)
        else:
            window = pyglet.window.Window(width=1920, height=1080, vsync=True, resizable=True, config=config)
    except:
        print('Warning: anti-aliasing disabled')
        config = pyglet.gl.Config(double_buffer=True) 
        if fullscreen:
            print('Fullscreen mode!')
            # N.B. accurate video timing only guaranteed with fullscreen
            # N.B. over-sampling seems to introduce frame lagging on windows+Intell
            window = pyglet.window.Window(fullscreen=True, vsync=True, resizable=False, config=config)
            #width=1280, height=720, 
        else:
            window = pyglet.window.Window(width=1920, height=1080, vsync=True, resizable=True, config=config)

    # setup alpha blending
    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    # setup anti-aliasing on lines
    pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH)                                                     
    pyglet.gl.glHint(pyglet.gl.GL_LINE_SMOOTH_HINT, pyglet.gl.GL_DONT_CARE)

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

def load_symbols(fn, replacements:dict={"<comma>":",", "<space>":"_"}):
    """load a screen layout from a text file

    Args:
        fn (str): file name to load from
        replacements (dict): dictionary of string replacements to apply

    Returns:
        symbols [list of lists of str]: list of list of the symbols strings
    """
    symbols = []

    # search in likely directories for the file, cwd, pydir, projectroot 
    fn = search_directories_for_file(fn,os.path.dirname(os.path.abspath(__file__)),
                                        os.path.join(os.path.dirname(os.path.abspath(__file__)),'symbols'),
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
            # BODGE: replace the <comma> string with the , symbol'
            line = [ replacements.get(l,l) for l in line ]

            # add
            symbols.append(line)

    return symbols


def init_noisetag_and_window(stimseq=None,host:str=None,fullscreen:bool=False):
    global nt, window    
    nt=Noisetag(stimSeq=stimseq, clientid='Presentation:selectionMatrix')
    if host is not None and not host in ('','-'):
        nt.connect(host, queryifhostnotfound=False)

    # init the graphics system
    window = initPyglet(fullscreen=fullscreen)


def run(symbols=None, #ncal:int=10, npred:int=10, 
        #calibration_trialduration:float=4.2,  prediction_trialduration:float=20, 
        #feedbackduration:float=2, cueduration:float=1, intertrialduration:float=1, 
        stimfile:str=None, stimseq:str=None, state2color:dict=None,
        fullscreen:bool=None, windowed:bool=None, fullscreen_stimulus:bool=True, host:str=None,       
        #calibration_screen:Screen=None, calibration_screen_args:dict=dict(),
        #prediction_screen:Screen=None, prediction_screen_args:dict=dict(),        
        # simple_calibration=None, host=None, 
        calibration_symbols=None, calibration_stimseq:str=None, calibration_screen_args:dict=dict(), #bgFraction=.1,
        #extra_symbols=None, extra_screens=None, extra_stimseqs=None, extra_labels=None, extra_screen_args=None,
        #calibration_args:dict=None, prediction_args:dict=None, 
        **kwargs):
    """ run the selection Matrix with default settings

    Args:
        ncal (int, optional): number of calibration trials. Defaults to 10.
        npred (int, optional): number of prediction trials at a time. Defaults to 10.
        simple_calibration (bool, optional): flag if we show only a single target during calibration, Defaults to False.
        stimseq ([type], optional): the stimulus file to use for the codes. Defaults to None.
        framesperbit (int, optional): number of video frames per stimulus codebit. Defaults to 1.
        fullscreen (bool, optional): flag if should runn full-screen. Defaults to False.
        fullscreen_stimulus (bool, optional): flag if should run the stimulus (i.e. flicker) in fullscreen mode. Defaults to True.
        simple_calibration (bool, optional): flag if we only show the *target* during calibration.  Defaults to False
        calibration_trialduration (float, optional): flicker duration for the calibration trials. Defaults to 4.2.
        prediction_trialduration (float, optional): flicker duration for the prediction trials.  Defaults to 10.
        calibration_args (dict, optional): additional keyword arguments to pass to `noisetag.startCalibration`. Defaults to None.
        prediction_args (dict, optional): additional keyword arguments to pass to `noisetag.startPrediction`. Defaults to None.
    """
    # configuration message for logging what presentation is used
    global configmsg
    configmsg = "{}".format(dict(component=__file__, args=locals()))

    global nt, window
    # N.B. init the noise-tag first, so asks for the IP
    if stimfile is None:
        stimfile = 'mgold_61_6521_psk_60hz.txt'
    if stimseq is None:
        stimseq = stimfile
    if calibration_stimseq is None:
        calibration_stimseq = stimseq
    if fullscreen is None and windowed is not None:
        fullscreen = not windowed
    if windowed == True or fullscreen == True:
        fullscreen_stimulus = False

    init_noisetag_and_window(stimseq,host,fullscreen)

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
    if state2color is not None: # put in the cal-screen args
        calibration_screen_args['stat2color']=state2color
    # make the screen manager object which manages the app state
    ss = ExptScreenManager(window, nt, symbols=symbols,
                        calibration_screen_args=calibration_screen_args,
                        fullscreen_stimulus=fullscreen_stimulus, 
                        calibration_symbols=calibration_symbols, 
                        stimseq=stimseq, calibration_stimseq=calibration_stimseq,
                        **kwargs)

    run_screen(ss,drawrate)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncal',type=int, help='number calibration trials', default=argparse.SUPPRESS)
    parser.add_argument('--npred',type=int, help='number prediction trials', default=argparse.SUPPRESS)
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--stimseq',type=str, help='stimulus file to use', default=argparse.SUPPRESS)
    parser.add_argument('--framesperbit',type=int, help='number of video frames per stimulus bit. Default 1', default=argparse.SUPPRESS)
    parser.add_argument('--fullscreen_stimulus',action='store_true',help='run with stimuli in fullscreen mode')
    parser.add_argument('--windowed',action='store_true',help='run in fullscreen mode')
    parser.add_argument('--selectionThreshold',type=float,help='target error threshold for selection to occur. Default .1',default=argparse.SUPPRESS)
    parser.add_argument('--simple_calibration',action='store_true',help='flag to only show a single target during calibration',default=argparse.SUPPRESS)
    parser.add_argument('--symbols',type=str,help='file name for the symbols grid to display',default=argparse.SUPPRESS)
    parser.add_argument('--calibration_symbols',type=str,help='file name for the symbols grid to use for calibration',default=argparse.SUPPRESS)
    parser.add_argument('--extra_symbols',type=str,help='comma separated list of extra symbol files to show',default=argparse.SUPPRESS)
    parser.add_argument('--config_file', type=str, help='JSON file with default configuration for the on-line BCI', default=None)#'debug')#'online_bci.json')
    #parser.add_argument('--artificial_deficit_id',type=int, help='objID for introducing stimulus deficit', default=argparse.SUPPRESS)
    #parser.add_option('-m','--matrix',action='store',dest='symbols',help='file with the set of symbols to display',default=argparse.SUPPRESS)
    args = parser.parse_args()
    if hasattr(args,'extra_symbols') and args.extra_symbols is not None:
        args.extra_symbols = args.extra_symbols.split(',')

    if args.config_file is None:
        config_file = askloadconfigfile()
        setattr(args,'config_file',config_file)

    if args.config_file is not None:
        config = load_config(args.config_file)
        # get the presentation_args
        if 'presentation_args' in config:
            config = config['presentation_args']
        # set them
        args = set_args_from_dict(args,config)

    return args

if __name__ == "__main__":
    args = parse_args()
    # setattr(args,'symbols',[['yes','no','<-']]) #"eog.txt")#
    # #setattr(args,'extra_symbols',['3x3.txt','robot_control.txt'])
    # setattr(args,'stimfile','level8_gold_01.txt')
    # setattr(args,'calibration_stimseq','rc5x5.txt')
    # #setattr(args,'extra_symbols',['prva.txt'])
    # setattr(args,"symbols",[["+|visualacuity/grating1.jpg|visualacuity/grating1_neg.jpg|visualacuity/grating2.jpg|visualacuity/grating2_neg.jpg|visualacuity/grating3.jpg|visualacuity/grating3_neg.jpg|visualacuity/grating4.jpg|visualacuity/grating4_neg.jpg|visualacuity/grating7.jpg|visualacuity/grating7_neg.jpg|visualacuity/grating10.jpg|visualacuity/grating10_neg.jpg"]])
    # setattr(args,"stimfile","6blk_rand_pr.txt")
    # setattr(args,'extra_screens',["mindaffectBCI.examples.presentation.image_flash.ImageFlashScreen",
    #                       "mindaffectBCI.examples.presentation.image_flash.ImageFlashScreen",
    #                       "mindaffectBCI.examples.presentation.image_flash.ImageFlashScreen"])
    # setattr(args,"extra_labels",["rand pr", "sweep pr", "rand"])
    # setattr(args,"extra_stimseqs",["6blk_rand_pr.txt","6blk_sweep_pr.txt","6blk_rand.txt"])

    # setattr(args,"symbols","keyboard.txt")
    # setattr(args,    "calibration_symbols", "3x3.txt")
    # setattr(args,    "extra_symbols", ["emojis.txt","robot_control.txt"])
    # setattr(args,    "stimfile","mgold_65_6532_psk_60hz.png")



    # setattr(args,'fullscreen',False)
    # setattr(args,'calibration_args',{"startframe":"random"})
    # setattr(args,'optosensor',-1)
    # setattr(args,'framesperbit',30)
    run(**vars(args))

