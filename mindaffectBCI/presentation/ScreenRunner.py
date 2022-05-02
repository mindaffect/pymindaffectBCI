import pyglet
import types
import time
from mindaffectBCI.noisetag import sumstats
from mindaffectBCI.decoder.utils import import_and_make_class
from mindaffectBCI.presentation.screens.basic_screens import Screen

user_state = dict()

def initPyglet(fullscreen=False, on_flip_callback=None, width=None, height=None):
    """intialize the pyglet window, keyhandlers, resize handlers etc.

    Note: one important specific thing done here is to insert frame-timeing code into the window's `flip` method
    which records precise display flip time information into the windows `lastfliptime` property.
    
    The handlers installed add the following properties to the window instances:
     window.last_key_press - contains the last key press in this window
     window.last_text - contains the last text entered in this window, as a string
     window.last_mouse_press - contains the last mouse press as (x,y,button,modifiers)
     window.last_mouse_release - contains the last mouse release info as (x,y,button,modifiers)

    # TODO[]: handle resize events correctly.

    Args:
        fullscreen (bool, optional): _description_. Defaults to False.
        on_flip_callback (lambda window:pass, optional): function to callback every time the windows flip method is called. If None then add a flip-time callback which records the lastfliptime in milliseconds in the widow property  window.lastfliptime. Defaults to None.
        width (int): screen width
        height (int): screen height

    Returns:
        pyglet.window: the configured pyglet window
    """
    if not fullscreen:
        width = width if width else 1920
        height = height if height else 1024
    # set up the window
    try:
        config = pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4)
        if fullscreen:
            # N.B. accurate video timing only guaranteed with fullscreen
            # N.B. over-sampling seems to introduce frame lagging on windows+Intell
            window = pyglet.window.Window(fullscreen=True, vsync=True, resizable=False, config=config)
        else:
            window = pyglet.window.Window(width=width, height=height, vsync=True, resizable=True, config=config)
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
            window = pyglet.window.Window(width=width, height=height, vsync=True, resizable=True, config=config)

    # setup alpha blending
    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    # setup anti-aliasing on lines
    pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH)                                                     
    pyglet.gl.glHint(pyglet.gl.GL_LINE_SMOOTH_HINT, pyglet.gl.GL_DONT_CARE)


    #-----------------------------------------
    # setup handlers for window events -- basically to store state in the window object itself
    def on_deactivate():
        if fullscreen:
            window.minimize()

    window.last_key_press=None
    def on_key_press(symbols, modifiers):
        '''main key-press handler, which stores the last key as a window property'''
        window.last_key_press=symbols

    window.last_text=None
    def on_text(text):
        """store last text entered as window property"""
        window.last_text = text

    window.last_mouse_press=None
    def on_mouse_press(x, y, button, modifiers):
        """store mouse press info as a window property"""
        #print("on_mouse_press: {}".format((x,y,button,modifiers)))
        window.last_mouse_press = (x,y,button,modifiers)

    window.last_mouse_release=None
    def on_mouse_release(x, y, button, modifiers):
        """store mouse release info as window property"""
        #print("on_mouse_release: {}".format((x,y,button,modifiers)))
        window.last_mouse_release = (x,y,button,modifiers)

    if on_flip_callback is None:
        def on_flip(window):
            '''callback function when window flip happens'''
            window.lastfliptime=(int(time.perf_counter()*1000) % (1<<31))
        on_flip_callback = on_flip

    def flip_with_callback(self):
        '''pseudo method type which executes callback after fliping the window'''
        type(self).flip(self)
        on_flip_callback(self)

    # register the various event handlers
    window.push_handlers(on_key_press, on_text, on_mouse_press, on_mouse_release, on_deactivate)

    # override window's flip method to record the exact *time* the
    # flip happended
    window.flip = types.MethodType(flip_with_callback, window)
    window.lastfliptime=-1

    # holder for general user information
    window.user_state=dict()

    return window


def draw_screen(screen,dt):
    '''main window draw function, which redirects to the draw function'''
    screen.draw(dt)
    # check for termination
    if screen.is_done():
        print('app exit')
        pyglet.app.exit()


def run_screen(window:pyglet.window, screen:Screen, drawrate:float=-1):
    """run the given screen in the given window with the given frame-rate

    Args:
        window (pyglet.window, optional): window to use for drawing.
        screen (Screen): main screen to run.
        drawrate (float, optional): frame rate for drawing the screen. Defaults to -1.
    """
    # set per-frame callback to the draw function
    if drawrate > 0:
        # slow down for debugging
        pyglet.clock.schedule_interval(lambda dt: draw_screen(screen,dt), drawrate)
    else:
        # call the draw method as fast as possible, i.e. at video frame rate!
        pyglet.clock.schedule(lambda dt: draw_screen(screen,dt))
    # mainloop
    window.set_visible(True)
    screen.reset()
    pyglet.app.run()
    pyglet.app.EventLoop().exit()
    window.set_visible(False)


def run(screen:Screen=None, noisetag:dict=dict(), fullscreen:bool=False, width:int=None, height:int=None, config_file:str=None):
    """setup and run the given screen

    Args:
        screen (Screen): the screen to run
        fullscreen (bool, optional): if true then run the window fullscreen. Defaults to True.
        config_file (dict|str, optional): config file to load screen, or config dictionary to use, to run from.  Defaults to None
    """
    if screen is None and config_file is not None:
        if isinstance(config_file,str):
            from mindaffectBCI.config_file import load_config
            config = load_config(config_file)
        else:
            config = config_file
        if 'presentation_args' in config:
            config = config['presentation_args']
        screen, noisetag, fullscreen, width, height = config.get('screen',None), config.get('noisetag',None), config.get('fullscreen',None), config.get('width',None), config.get('height',None)

    window = initPyglet(fullscreen=fullscreen, width=width, height=height)
    if not isinstance(screen,Screen):
        if isinstance(screen,str): screen=[screen,{}]
        screenclass, args = screen
        if noisetag is not None: # BODGE: as noisetag object to the screen args
            from mindaffectBCI.noisetag import Noisetag
            args['noisetag']=Noisetag(**noisetag)
        if not '.' in screenclass: 
            screenclass = 'mindaffectBCI.presentation.screens.' + screenclass + '.' + screenclass
        screen = import_and_make_class(screenclass, window=window, **args)
    run_screen(window, screen)


def parse_args():
    """parse the command line arguments -- if running from the command line

    Returns:
        _type_: _description_
    """
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--fullscreen',action='store_true',help='run in fullscreen mode')
    parser.add_argument('--screen', type=json.loads, help='specify the screen to run, as class,arguments',default=None)
    parser.add_argument('--config_file', type=str, help='JSON file with default configuration for the on-line BCI', default=None)
    args = parser.parse_args()

    # run from the presentation part of a normal online_bci config if wanted
    if args.config_file is None and args.screen is None:
        from mindaffectBCI.config_file import askloadconfigfile
        config_file = askloadconfigfile()
        setattr(args,'config_file',config_file)

    return args

if __name__=='__main__':
    args = parse_args()
    run(**vars(args))
