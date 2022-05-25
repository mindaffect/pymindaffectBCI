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
import pyglet
import os
import time
from mindaffectBCI.noisetag import Noisetag
from mindaffectBCI.decoder.utils import search_directories_for_file, import_and_make_class

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
    def __init__(self, window, duration=5000, waitKey=True, waitMouse=True, logo="MindAffect_Logo.png", fixation:bool=False, label:str=None, noisetag=None):
        super().__init__(window)
        self.t0 = None # timer for the duration
        self.duration, self.waitKey, self.waitMouse, self.fixation, self.label= (duration, waitKey, waitMouse, fixation, label)
        self.label = label if label is not None else self.__class__.__name__
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
        font_size = self.fixation if self.fixation>1 else 40
        self.fixation_obj = pyglet.text.Label("+", font_size=font_size, 
                                            x=self.window.width//2, y=self.window.height//2,
                                            color=(255, 0, 0, 255),
                                            anchor_x='center', anchor_y='center',
                                            batch=self.batch, group=self.group)
        self.set_fixation( self.fixation is not None and self.fixation>0 )

    def set_fixation(self,visible:bool):
        self.fixation_obj.color = (255,0,0,255) if visible else (255,0,0,0)

    def reset(self):
        self.isRunning = False
        self.isDone = False
        if self.waitKey:
            self.window.last_key_press= None
        if self.waitMouse:
            self.window.last_mouse_release = None

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
        if hasattr(self.window,'flipstats'):
            self.window.flipstats.update_statistics()
            self.framerate.begin_update()
            self.framerate.text = "{:4.1f} +/-{:4.1f}ms".format(self.window.flipstats.median,self.window.flipstats.sigma)
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
    def __init__(self, window, text:str, x:float=.5, y:float=.5, anchor_x:str='center', anchor_y:str='center', width:int=.8, duration=5000, waitKey=True, waitMouse=True, font_size:int=25, logo="MindAffect_Logo.png", title:str=None, fixation:bool=False, **kwargs):
        """Screen which shows a textual instruction for duration or until key-pressed

        Args:
            window (_type_): pyglet window to draw into
            text (str): the instruction text to show
            x,y (float, optional): x,y position for the text box. If <1 then fraction of screen size. Defaults to .5.
            y (float, optional): _description_. Defaults to .5.
            anchor_x (str, optional): anchor of the x,y position in the box. Defaults to 'center'.
            anchor_y (str, optional): anchor of the x,y position in the box. Defaults to 'center'.
            width (int, optional): text box width.  If <1 then in fraction of screen width. Defaults to .8.
            duration (int, optional): Duration to show the instruction for. Defaults to 5000.
            waitKey (bool, optional): If True the key-press finishes the screen. Defaults to True.
            waitMouse (bool, optional): If True then mouse-click finishes the screen. Defaults to True.
            logo (str, optional): logo image to show in top-right of the screen. Defaults to "MindAffect_Logo.png".
            title (str, optional): title text to show at the top of the screen. Defaults to None.
            fixation (bool, optional): If True then show fixation cross in center of the screen. Defaults to False.
        """
        super().__init__(window, duration, waitKey, waitMouse, logo, fixation, **kwargs)
        # initialize the instructions screen --- and add to the parent screen's batch
        if 0 < x and x < 1: x = int(self.window.width * x)
        if 0 < y and y < 1: y = int(self.window.height * y)
        if 0 < width and width < 1: width=int(self.window.width*width) 
        self.instructLabel = pyglet.text.Label(x=x,
                                               y=y,
                                               anchor_x=anchor_x,
                                               anchor_y=anchor_y,
                                               font_size=font_size,
                                               color=(255, 255, 255, 255),
                                               multiline=True,
                                               width=width,
                                               batch=self.batch,
                                               group=self.group)
        self.titleLabel = pyglet.text.Label(x=self.window.width//2,
                                               y=self.window.height,
                                               anchor_x='center',
                                               anchor_y='top',
                                               font_size=24,
                                               color=(255, 255, 255, 255),
                                               multiline=False,
                                               align='center',
                                               width=int(self.window.width*.8),
                                               batch=self.batch,
                                               group=self.group)
        self.set_text(text)
        self.set_title(title)

    def set_text(self, text):
        '''set/update the text to show in the instruction screen'''
        self.text = text.split("\n") if isinstance(text,str) else text
        text = "\n".join(self.text)
        self.instructLabel.begin_update()
        self.instructLabel.text=text
        self.instructLabel.end_update()

    def set_title(self, text):
        '''set/update the text to show in the instruction screen'''
        if text is None: return
        self.title = text
        if not isinstance(text,str):
            text = "\n".join(text)
        self.titleLabel.begin_update()
        self.titleLabel.text=text
        self.titleLabel.end_update()


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
    def __init__(self, window:pyglet.window, label:str=None, instruct:str="This is the default start-screen.  Press <space> to continue", **kwargs):
        self.window, self.label, self.instruct = (window, label, instruct)
        if self.label is None: self.label = self.__class__.__name__

        instruct_screen = InstructionScreen(window, self.instruct, duration = 50000)

        # make a list to store the screens in the order you want to go through them
        self.subscreens = [instruct_screen]

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
        if self.current_screen_idx < len(self.subscreens):
            self.screen = self.subscreens[self.current_screen_idx]
            self.screen.reset()
        else:
            self.screen = None



#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class ScreenGraph(Screen):
    '''screen which manages transitions between sub-screens'''
    def __init__(self, window:pyglet.window, label:str=None, 
                 subscreens:dict={'ins1':('InstructionScreen',{'text':'This is a default start screen....\nPress <space> to continue', "waitKey":True, "duration":1000}),
                                  'ins2':('InstructionScreen',{'text':'And this is a second default screen to show transitions', "waitKey":True, "duration":1000})},
                 subscreen_transitions:dict=dict(),
                 subscreen_args:dict=None,
                 start_screen:str=None, default_screen:str=None, exit_screen:str=None, noisetag=None):
        """A generic meta-screen which has a set of named sub-screens and a transition graph to specify the transitions between subscreens

        Args:
            window (pyglet.window): the pyglet window to draw into
            label (str, optional): Human readable name for this screen, also used in menu entries. Defaults to None.
            subscreens (dict, optional): Dictionary of named sub-screens.  Key is the screen name, Value is either a created screen, or a 2-tuple with the fully-qualified screen class name and the arguments to pass to the screen. Defaults to {'ins1':('InstructionScreen',{'text':'This is a default start screen....\nPress <space> to continue'}), 'ins2':('InstructionScreen',{'text':'And this is a second default screen to show transitions'})}.
            subscreen_transitions (dict, optional): Dictionary of transitions between screens.  Key the current-screen name, value is the screen to move to, or function to call to get the screen to transition to. Defaults to {"ins1":"ins2", "ins2":"end"}.
            subscreen_args (dict, optional): Dictionary of extra args to pass to sub-screen constructors, e.g. the noisetag object.  Defaults to None.
            start_screen (str, optional): Name of the screen to start with.  If None then the first screen in subscreens. Defaults to None.
            default_screen (str, optional): Default screen to transition to if no valid transition is found.  If None then the first screen in subscreens. Defaults to None.
            exit_screen (str, optional): Screen to exit the graph from.  If None then the first screen in subscreens. Defaults to None.
        """
        self.window, self.label, self.noisetag = window, label, noisetag
        if self.label is None: self.label = self.__class__.__name__

        self.subscreens = self.init_subscreens(subscreens, subscreen_args)
        self.subscreen_transitions = subscreen_transitions if subscreen_transitions else self.sequential_subscreen_transitions(subscreens)
        self.default_screen, self.exit_screen = default_screen, exit_screen
        self.start_screen = start_screen if start_screen else tuple(subscreens.keys())[0]
        self.reset()

    def reset(self):
        super().reset()
        self.current_screen = self.start_screen
        self.screen = self.subscreens.get(self.current_screen,None)
        if self.screen:
            self.screen.reset()

    def init_subscreens(self, subscreens:dict, subscreen_args:dict=None):
        """setup the set of sub-screens, creating the screen classes as needd

        Args:
            subscreens (dict, optional): Dictionary of named sub-screens.  Key is the screen name, Value is either a created screen, or a 2-tuple with the fully-qualified screen class name and the arguments to pass to the screen. Defaults to {'ins1':('InstructionScreen',{'text':'This is a default start screen....\nPress <space> to continue'}), 'ins2':('InstructionScreen',{'text':'And this is a second default screen to show transitions'})}.

        Returns:
            (dict): dictionary of named sub-screens with instaintated screen classes as values.
        """
        inited_subscreens = dict()
        for k,screen in subscreens.items():
            if not isinstance(screen,Screen):
                screenclass = screen[0]
                # add prefix to make fully qualified class name
                if not '.' in screenclass: 
                    if screenclass in locals():
                        screenclass = 'mindaffectBCI.presentation.screens.basic_screens' + "." + screenclass
                    else:
                        screenclass = 'mindaffectBCI.presentation.screens.' + screenclass + '.' + screenclass
                # create the screen
                screen_args = screen[1] if len(screen)>1 else dict()
                if subscreen_args is not None: # include extra args
                    screen_args.update(subscreen_args)
                if self.noisetag:
                    screen_args['noisetag']=self.noisetag
                screen=import_and_make_class(screenclass,window=self.window,**screen_args)
            inited_subscreens[k] = screen
        return inited_subscreens

    def sequential_subscreen_transitions(self,subscreens):
        """ make a set of transitions to sequentially transition through the given dict/list of subscreens

        Returns:
            dict: a dict of from:to screen names
        """
        # init as a sequential scan through sub-screens in given order
        # get the list of screens
        scrns = tuple(subscreens.keys()) if isinstance(subscreens,dict) else subscreens
        # make the dict of transitions for one screen to the next
        subscreen_transitions = { k:v for k,v in zip(scrns[:-1],scrns[1:]) }
        return subscreen_transitions

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
        # get the next screen to move to from the screen_transitions dict
        if self.current_screen == self.exit_screen: # done if exit_screen is done
            self.current_screen = None
        else:
            self.current_screen = self.subscreen_transitions.get(self.current_screen,self.default_screen)
        if callable(self.current_screen):  
            self.current_screen = self.current_screen(self.screen)
        # get the screen from the subscreens dict, or no-screen if the screen is not found
        self.screen = self.subscreens.get(self.current_screen,None)
        if self.screen:
            self.screen.reset()



#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class LoopedScreenGraph(ScreenGraph):
    '''screen which iterates through a list of sub-screens a given number of times'''
    def __init__(self, window:pyglet.window, label:str=None, 
                 subscreens:dict={'ins1':('InstructionScreen',{'text':'This is a default start screen....\nPress <space> to continue', "waitKey":True, "duration":1000}),
                                  'ins2':('InstructionScreen',{'text':'And this is a second default screen to show transitions', "waitKey":True, "duration":1000}),
                                  'exit':('InstructionScreen',{'text':'This is the exit screen', "waitKey":True, "duration":1000})
                                  },
                 subscreen_transitions:dict={"ins1":"ins2", "ins2":"exit"},
                 subscreen_args:dict=None,
                 start_screen:str=None, default_screen:str=None, exit_screen:str=None, noisetag=None,
                 n_loop:int=5):
        """screen which loops the inner screen graph a number of times

        Args:
            window (pyglet.window): the pyglet window to draw into
            label (str, optional): Human readable name for this screen, also used in menu entries. Defaults to None.
            subscreens (dict, optional): Dictionary of named sub-screens.  Key is the screen name, Value is either a created screen, or a 2-tuple with the fully-qualified screen class name and the arguments to pass to the screen. Defaults to {'ins1':('InstructionScreen',{'text':'This is a default start screen....\nPress <space> to continue'}), 'ins2':('InstructionScreen',{'text':'And this is a second default screen to show transitions'})}.
               N.B. the key 'looper' is used to indicate the loop screen which is shown after the end of an iteration through the screen graph.  Provide a screen construction phase to setup this screen to something special.
            subscreen_transitions (dict, optional): Dictionary of transitions between screens.  Key the current-screen name, value is the screen to move to, or function to call to get the screen to transition to. Defaults to {"ins1":"ins2", "ins2":"end"}.
            subscreen_args (dict, optional): Dictionary of extra args to pass to sub-screen constructors, e.g. the noisetag object.  Defaults to None.
            start_screen (str, optional): Name of the screen to start with.  If None then the first screen in subscreens. Defaults to None.
            default_screen (str, optional): Default screen to transition to if no valid transition is found.  If None then the first screen in subscreens. Defaults to None.
            exit_screen (str, optional): Screen to exit the graph from.  If None then the first screen in subscreens. Defaults to None.
            n_loop (int, optional): number of iterations round the screen graph to do. Defaults to 1.
        """
        super().__init__(window=window,label=label,subscreens=subscreens, subscreen_transitions=subscreen_transitions, subscreen_args=subscreen_args, start_screen=start_screen, default_screen=default_screen, exit_screen=None, noisetag=noisetag)
        self.inner_start_screen = start_screen
        self.inner_exit_screen = exit_screen
        self.n_loop = n_loop
        self.loop_i = 0
 
        # add a looper screen to the set of screen transitions
        if self.subscreens.get('looper',None) is None:
            self.subscreens['looper'] = InstructionScreen(window,label='loop screen', text='0/{}'.format(self.n_loop), duration=100, waitKey=False, waitMouse=False)
        self.subscreen_transitions['looper']=self.loop_test
        # attach into the transition graph
        if exit_screen is not None:
            self.subscreen_transitions[exit_screen]='looper'

    def loop_test(self, screen):
        """conditional transition method which loops when needed and exits when done

        Returns:
            str: the next screen to run
        """
        if self.loop_i < self.n_loop :
            self.loop_i = self.loop_i + 1
            print("trial={}".format(self.loop_i))
            # update the looper screen
            if hasattr(self.subscreens['looper'],'set_text'):
                text = self.subscreens['looper'].text
                # replace the final line with the counter
                text[-1] = "{}/{}".format(self.loop_i,self.n_loop)
                self.subscreens['looper'].set_text(text)
            return self.start_screen
        else:
            return self.exit_screen

    def reset(self):
        super().reset()
        self.loop_i = 0



if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    window = initPyglet(width=640, height=480)
    ins= InstructionScreen(window=window,text='hello there pre-made screen')
    subscreens = {
        'ins1':('InstructionScreen',{'text':'This is a default start screen....\nPress <space> to continue', "waitKey":True, "duration":1000}),
        #'ins2':('InstructionScreen',{'text':'And this is a second default screen to show transitions', "waitKey":True, "duration":1000}),
        "test":["ScreenGraph",{}],
        'ins2':ins,
        'exit':('InstructionScreen',{'text':'This is the exit screen', "waitKey":True, "duration":1000}),
        "looper":["InstructionScreen",{"text":"Loop Screen\n\n","waitKey":True,"duration":1000}],    
    }
    subscreen_transitions = {'ins1':'ins2', 'ins2':'exit'}
    screen = ScreenGraph(window, subscreens=subscreens, subscreen_transitions=subscreen_transitions, exit_screen='exit')
    #screen = LoopedScreenGraph(window, n_loop=5, subscreens=subscreens, subscreen_transitions=subscreen_transitions, exit_screen='exit')
    run_screen(window, screen)
