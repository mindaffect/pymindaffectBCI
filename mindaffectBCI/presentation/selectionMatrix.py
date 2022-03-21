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
from tkinter.messagebox import askyesno
from mindaffectBCI.noisetag import Noisetag, PredictionPhase
from mindaffectBCI.decoder.utils import search_directories_for_file, import_and_make_class
from mindaffectBCI.config_file import load_config, set_args_from_dict, askloadconfigfile
from mindaffectBCI.presentation.screens.basic_screens import Screen, ScreenList, ScreenSequence, WaitScreen, InstructionScreen
from mindaffectBCI.presentation.screens.UserInputScreen import UserInputScreen
from mindaffectBCI.presentation.screens.FrameRateTestScreen import FrameRateTestScreen
from mindaffectBCI.presentation.screens.ElectrodeQualityScreen import ElectrodeQualityScreen
from mindaffectBCI.presentation.screens.MenuScreen import MenuScreen
from mindaffectBCI.presentation.screens.UtopiaConnectingScreen import ConnectingScreen
from mindaffectBCI.presentation.screens.SelectionGridScreen import SelectionGridScreen


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
# import mindaffectBCI.presentation.selectionMatrix as selectionMatrix
# You can then access this variable and it's state with:
#  selectionMatrix.user_state['my_state'] = 'hello'
user_state = dict()





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
        FullScreenToggle=auto()
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
                "f) Toggle Full-Screen",
                "s) settings",
                "r) reset calibration model",
                "L) send log message"]
    menu_keys_footer = {pyglet.window.key.F:SubScreens.FullScreenToggle, 'f':SubScreens.FullScreenToggle,
                pyglet.window.key.S:SubScreens.Settings, 's':SubScreens.Settings,
                pyglet.window.key.R:SubScreens.Reset, 'r':SubScreens.Reset,
                pyglet.window.key.Q:SubScreens.Quit, 'Q':SubScreens.Quit,
                pyglet.window.key.L:SubScreens.LogMessage, 'L':SubScreens.LogMessage}

    def __init__(self, window:pyglet.window, noisetag:Noisetag, symbols, nCal:int=None, ncal:int=1, npred:int=1, nPred:int=None, 
                 calibration_trialduration:float=4.2, prediction_trialduration:float=10,  
                 feedbackduration:float=None, cueduration:float=None, intertrialduration:float=None,
                 framesperbit:int=None, fullscreen_stimulus:bool=True, 
                 start_screen:int=None,
                 selectionThreshold:float=.1, optosensor:bool=True,  
                 calibration_screen:Screen="mindaffectBCI.presentation.selectionMatrix.SelectionGridScreen", calibration_screen_args:dict=dict(), calibration_sentence:str=None, calibration_instruct:str=None,
                 prediction_screen:Screen="mindaffectBCI.presentation.selectionMatrix.SelectionGridScreen", prediction_screen_args:dict=dict(),
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
                extra_screen_cls = ['mindaffectBCI.presentation.selectionMatrix.SelectionGridScreen']
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
        self.electquality = ElectrodeQualityScreen(window, noisetag)
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
        """setup the menu with numbered options to press
        """
        main_menu_numbered = ["0) Electrode Quality",
            "1) Calibration",
            "2) Cued Prediction" ,
            "3) Free Typing",
            "4) Practice"]
        menu_keys = {'0':self.SubScreens.SignalQuality,
                    pyglet.window.key._0:self.SubScreens.SignalQuality,
                    pyglet.window.key.NUM_0:self.SubScreens.SignalQuality}
        extra_menu_numbered = []
        extra_menu_keys = dict()

        # build the main menu options
        if self.calibration_screen is not None: 
            if hasattr(self.calibration_screen,'label') and not self.calibration_screen.label is None:
                main_menu_numbered[1] = "{}) {}".format(1,self.calibration_screen.label)
            # add to the key-menu
            menu_keys.update({'1':self.SubScreens.CalInstruct,
                             pyglet.window.key._1:self.SubScreens.CalInstruct, 
                             pyglet.window.key.NUM_1:self.SubScreens.CalInstruct,
                             '4':self.SubScreens.PracticeInstruct,
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
            menu_keys.update({'3':self.SubScreens.PredInstruct,
                              pyglet.window.key._3:self.SubScreens.PredInstruct,
                              pyglet.window.key.NUM_3:self.SubScreens.PredInstruct})
        else: # No prediction mode
            for i,s in enumerate(main_menu_numbered):
                if 'prediction' in s.lower():
                    main_menu_numbered[i]=""
            # TODO[]: remove from menu-keys?
        if self.cued_prediction_screen is not None: # remove the cued-prediction option
            if hasattr(self.cued_prediction_screen,'label') and not self.cued_prediction_screen.label is None:
                main_menu_numbered[2] = "{}) {} (Cued)".format(2,self.cued_prediction_screen.label)
            menu_keys.update({'2':self.SubScreens.CuedPredInstruct,
                             pyglet.window.key._2:self.SubScreens.CuedPredInstruct,
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
                    menu_keys["{:d}".format(keyi)] = self.SubScreens.ExtraSymbols
                    menu_keys[getattr(pyglet.window.key,"_{:d}".format(keyi))] = self.SubScreens.ExtraSymbols
                    menu_keys[getattr(pyglet.window.key,"NUM_{:d}".format(keyi))] = self.SubScreens.ExtraSymbols
                    extra_menu_keys["_{:d}".format(keyi)] = i # map key-symb to srcreen to run
                    extra_menu_keys[getattr(pyglet.window.key,"_{:d}".format(keyi))] = i # map key-symb to srcreen to run
                    extra_menu_keys[getattr(pyglet.window.key,"NUM_{:d}".format(keyi))] = i
            else: # add to sub-menu
                main_menu_numbered.append("{}) Extra options".format('9'))
                menu_keys['9'] = self.SubScreens.ExtraMenuScreen
                menu_keys[pyglet.window.key._9] = self.SubScreens.ExtraMenuScreen
                menu_keys[pyglet.window.key.NUM_9] = self.SubScreens.ExtraMenuScreen

                for i,ps in enumerate(self.extra_screens):
                    keyi = i
                    label = getattr(ps,'label',type(ps).__name__) # use label slot, fallback on class name
                    extra_menu_numbered.append("{}) {}".format(keyi,label))
                    extra_menu_keys["{:d}".format(keyi)] = keyi
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
        """draw the screen

        Args:
            t (float): draw time (ignored)
        """
        if self.screen is None:
            return
        if self.screen.is_done():
            self.transitionNextPhase()
        self.screen.draw(t)

    def is_done(self):
        """test if the screen is finished displaying

        Returns:
            bool: running state of this screen.  If True then screen is finished
        """
        return self.screen is None

    def transitionNextPhase(self):
        """move to the next phase in the screen sequence.

        This method is called when a screen has finished to determine which is the next screen to run.  
        It then initialized the new screen and sets it to run.

        Most of the actually application logic is in this function which uses a really big switch based on the 
        `SubScreens` enumeration to decide which new screen to run.

        Two key variables are: self.current_screen : which holds the currently running screen,
        and self.next_screen which holds the screen to transition to when the current screen is_done
        """        
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

        elif self.current_screen==self.SubScreens.FullScreenToggle: # fullscreen
            print("fullscreen toggle")
            self.window.set_fullscreen(fullscreen= not self.window.fullscreen)
            #from mindaffectBCI.examples.presentation.framerate_check import FrameRateTestScreen
            self.next_screen = self.SubScreens.MainMenu

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
                else: 
                    self.confirm_quit = None
                    self.next_screen = self.SubScreens.MainMenu


#------------------------------------------------------------------------
# Initialization: display, utopia-connection
def getTimeStamp():
    """get a valid time-stamp -- either from the utopia connection or the python real-time-clock

    Returns:
        _type_: _description_
    """    
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

    from mindaffectBCI.presentation.selectionMatrix import flipstats, fliplogtime
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
    """handler for text input

    Args:
        text (str): text entered into the window
    """    
    global window
    window.last_text = text

def on_mouse_press(x, y, button, modifiers):
    """handler for mouse presses

    Args:
        x (_type_): _description_
        y (_type_): _description_
        button (_type_): _description_
        modifiers (_type_): _description_
    """    
    global window
    print("on_mouse_press: {}".format((x,y,button,modifiers)))
    window.last_mouse_press = (x,y,button,modifiers)

def on_mouse_release(x, y, button, modifiers):
    """handler for mouse release

    Args:
        x (_type_): _description_
        y (_type_): _description_
        button (_type_): _description_
        modifiers (_type_): _description_
    """
    global window
    print("on_mouse_release: {}".format((x,y,button,modifiers)))
    window.last_mouse_release = (x,y,button,modifiers)


def initPyglet(fullscreen=False):
    '''intialize the pyglet window, keyhandlers, resize handlers etc.'''
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
    window.push_handlers(on_key_press, on_text, on_mouse_press, on_mouse_release)
    window.last_key_press=None
    window.last_text=None
    window.last_mouse_press=None
    window.last_mouse_release=None

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
    '''main window draw function, which redirects to the screen stack stored in the global variable `ss`'''
    global ss, nframe
    nframe=nframe+1
    ss.draw(dt)
    # check for termination
    if ss.is_done():
        print('app exit')
        pyglet.app.exit()
    #print('.', end='', flush=True)

def run_screen(screen:Screen, drawrate:float=-1, win:pyglet.window=None):
    """run the given screen in the given window with the given frame-rate

    Args:
        screen (Screen): main screen to run, probably a screen-stack
        drawrate (float, optional): frame rate for drawing the screen. Defaults to -1.
        win (pyglet.window, optional): window to use for drawing. Defaults to None.
    """    
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


def init_noisetag_and_window(stimseq=None,host:str=None,fullscreen:bool=False):
    """setup the connection to the utopia-hub, and initialize the pyglet display stack

    Args:
        stimseq (_type_, optional): _description_. Defaults to None.
        host (str, optional): _description_. Defaults to None.
        fullscreen (bool, optional): _description_. Defaults to False.
    """    
    global nt, window    
    nt=Noisetag(stimSeq=stimseq, clientid='Presentation:selectionMatrix')
    if host is not None and not host in ('','-'):
        nt.connect(host, queryifhostnotfound=False)

    # init the graphics system
    window = initPyglet(fullscreen=fullscreen)


def run(symbols=None, 
        stimfile:str=None, stimseq:str=None, state2color:dict=None,
        fullscreen:bool=None, windowed:bool=None, fullscreen_stimulus:bool=True, host:str=None,       
        calibration_symbols=None, calibration_stimseq:str=None, calibration_screen_args:dict=dict(), #bgFraction=.1,
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
    """parse the command line arguments -- if running from the command line

    Returns:
        _type_: _description_
    """
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

