import pyglet
import os
import time
from mindaffectBCI.noisetag import Noisetag
from mindaffectBCI.decoder.utils import search_directories_for_file

class Screen:

    '''Screen abstract-class which draws stuff on the screen until finished'''
    def __init__(self, window, label:str=None):
        """Abstract screen for drawing content on the screen

        Args:
            window (_type_): the window to draw into
            label (str, optional): human readable name for this screen. Defaults to None.
        """
        self.window, self.label, = window, label
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
class WaitScreen(Screen):
    '''Screen which shows a blank screen for duration or until key-pressed'''
    def __init__(self, window, duration=5000, waitKey=True, waitMouse=True, logo="MindAffect_Logo.png", fixation:bool=False):
        super().__init__(window)
        self.t0 = None # timer for the duration
        self.duration, self.waitKey, self.waitMouse, self.fixation= (duration, waitKey, waitMouse, fixation)
        self.isRunning = False
        self.isDone = False
        self.clearScreen = True

        self.batch = pyglet.graphics.Batch()
        self.group = pyglet.graphics.OrderedGroup(0)

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
        if self.fixation:
            font_size = self.fixation if self.fixation>1 else 40
            self.fixation_obj = pyglet.text.Label("+", font_size=font_size, 
                                                x=self.window.width//2, y=self.window.height//2,
                                                color=(255, 0, 0, 255),
                                                anchor_x='center', anchor_y='center',
                                                batch=self.batch, group=self.group)
        
        self.reset()

    def reset(self):
        self.isRunning = False
        self.isDone = False
        if self.waitKey: self.window.last_key_press = None
        if self.waitMouse: self.window.last_mouse_release = None

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
        if self.waitMouse:
            if self.window.last_mouse_release:
                self.mouse_release = self.window.last_mouse_release
                self.isDone = True
                self.window.last_mouse_release = None
        if not self.duration is None and self.elapsed_ms() > self.duration:
            self.isDone = True

        return self.isDone

    def getTimeStamp(self):
        return (int(time.perf_counter()*1000) % (1<<31))

    def elapsed_ms(self):
        return self.getTimeStamp()-self.t0 if self.t0 else -1

    def draw(self, t):
        '''Show a block of text to the user for a given duration on a blank screen'''
        if not self.isRunning:
            self.isRunning = True  # mark that we're running
            self.t0 = self.getTimeStamp()
        if self.clearScreen:
            self.window.clear()

        # check if should update display
        # TODO[]: only update screen 1x / second
        from mindaffectBCI.presentation.selectionMatrix import flipstats, fliplogtime
        flipstats.update_statistics()
        self.framerate.begin_update()
        self.framerate.text = "{:4.1f} +/-{:4.1f}ms".format(flipstats.median,flipstats.sigma)
        self.framerate.end_update()

        # draw the batch
        self.batch.draw()


#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
# TODO[]: use batch for efficient draws
class InstructionScreen(WaitScreen):
    '''Screen which shows a textual instruction for duration or until key-pressed'''
    def __init__(self, window, text, duration=5000, waitKey=True, waitMouse=True, logo="MindAffect_Logo.png",fixation:bool=False):
        super().__init__(window, duration, waitKey, waitMouse, logo, fixation)
        # initialize the instructions screen --- and add to the parent screen's batch
        self.instructLabel = pyglet.text.Label(x=self.window.width//2,
                                               y=self.window.height//2,
                                               anchor_x='center',
                                               anchor_y='center',
                                               font_size=24,
                                               color=(255, 255, 255, 255),
                                               multiline=True,
                                               width=int(self.window.width*.8),
                                               batch=self.batch,
                                               group=self.group)
        self.set_text(text)

    def set_text(self, text):
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        self.instructLabel.begin_update()
        self.instructLabel.text=text
        self.instructLabel.end_update()

    # def draw(self, t):
    #     '''Show a block of text to the user for a given duration on a blank screen'''
    #     super().draw(t)
    #     if self.batch is None:
    #         self.instructLabel.draw()


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

        self.instruct_screen = InstructionScreen(window, 'This is a default start screen...\nPress <space> to continue to continue', duration = 50000, batch=batch, group=group)

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

