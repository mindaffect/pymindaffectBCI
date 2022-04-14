"""This module is used as a facade between the bci keyboard and the pyglet framework.

This module contains a single class ``PygletFacade``.

"""

#  Copyright (c) 2021,
#  Authors: Thomas de Lange, Thomas Jurriaans, Damy Hillen, Joost Vossers, Jort Gutter, Florian Handke, Stijn Boosman
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pyglet
from pyglet import shapes
from mindaffectBCI.examples.presentation.smart_keyboard.app_exceptions import Logger
from mindaffectBCI.examples.presentation.smart_keyboard.framework_facade import FrameworkFacade
import os

def flip_callback(self, application=None):
    '''pseudo method type which inserts a call-back after window flip for time-stamp recording'''
    type(self).flip(self)
    if application: application.set_flip_time()

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

def on_mouse_motion(x, y, dx, dy):
    global window
    #print("on_mouse_motion: {}".format((x,y,dx,dy)))
    window.last_mouse = (x,y,dx,dy)

def on_mouse_press(x, y, button, modifiers):
    global window
    print("on_mouse_press: {}".format((x,y,button,modifiers)))
    window.last_mouse_press = (x,y,button,modifiers)

def on_mouse_release(x, y, button, modifiers):
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
            window = pyglet.window.Window(width=1024, height=786, vsync=True, resizable=True, config=config)
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

    return window


class PygletFacade(FrameworkFacade):
    """Implements the psychopy functionality into the facade.

    Args:
        size ((int,int)): Size of window for the keyboard to run in.
        full_screen (bool): Whether or not to run the application in full screen mode

    Attributes:
        window (visual.Window): The psychopy window object
        mouse (event.Mouse): The psychopy mouse object
    """

    def __init__(self, size, wait_blanking=True, full_screen=False):
        self.window = initPyglet(fullscreen=full_screen)
        self.mouse = self.window
        self.click_feedback_buttons = []
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)
        self.application = None
        self.nframe = 0

    def get_window(self):
        return self.window

    def create_button(self, size, pos, button_color, line_color, label_color, label, has_icon):
        """Creates and returns a PsychoPy Rect and a TextStim or ImageStim object, representing the button and its label or icon.

        Args:
            size ((float, float)): The width and height of the button as a fraction of the entire window size
            pos ((float, float)): The position of the centre of the button on the screen, expressed as two fractions
            of the window sizes
            button_color: ((int, int, int)): The color of the button, in RGB format
            line_color ((int, int, int)): The color of the button border, in RGB format
            label_color ((int, int, int)): The color of the button label, in RGB format
            label (string): The text to be displayed on the button
            has_icon (string): Path to a png file from which the icon is created.

        Returns:
            ((visual.Rect, visual.TextStim/visual.ImageStim)): A rectangle object and a text object or image object,
            representing the button and its label or icon respectively.

        """
        button = self.create_rect(size, pos, button_color, line_color)
        if not has_icon:
            button_label = self.create_text(label, label_color, pos, size=size)
        else:
            button_label = self.create_icon(label, label_color, size, pos)
        return button, button_label

    def create_rect(self, size, pos, color, line_color):
        """Creates and returns a PsychoPy Rect(angle) object according to the passed arguments

        Args:
            size ((float, float)): The width and height of the rectangle as a fraction of the window size. 
               N.B. Rect is **symetric** about the pos!
            pos ((float, float)): The position of the centre of the rectangle, expressed as fractions of the window sizes
            color: ((int, int, int)): The color of the rectangle, in RGB format
            line_color ((int, int, int)): The color of the rectangle border, in RGB format

        Returns:
            (visual.Rect): A psychopy rectangle object
        """
        x,y = self.convert_pos(pos)
        w,h = self.convert_size(size)
        rect = shapes.Rectangle(x,y,width=w,height=h,
                                color=self.convert_color(color),group= self.background, batch=self.batch)
        rect.anchor_x, rect.anchor_y = w//2, h//2
        rect.visible = False
        #rect = shapes.BorderedRectangle(x,y,w,h,border=3,color=self.convert_color(color),border_color=self.convert_color(line_color),batch=self.batch)
        return rect

    def create_text(self, text='', col=(255, 255, 255), pos=(0, 0), size=(1,1), text_size=20, align_hor='center', align_vert='center', wrap_width=1):
        """Creates and returns a psychopy Text object, centered at the passed position

        Args:
            text (string): The text
            col: ((int, int, int)): text color in RGB format
            pos ((float, float)): Position of the text on the screen, expressed as two fractions relative to window dimensions
            size (())
            text_size (int): Text size
            align_hor (string): Horizontal alignment of the text ('left', 'right' or 'center')
            align_vert (string): Vertical alignment of the text ('top', 'bottom' or 'center')
            wrap_width (float): the maximal horizontal width of the text; it will wrap to a newline at this point

        Returns:
            (visual.TextStim): A psychopy text object

        """
        x,y=self.convert_pos(pos)
        w,h=self.convert_size(size)
        text = pyglet.text.Label(str(text), font_size=text_size, 
                                x=x, y=y, #width=w, height=h,
                                color=self.convert_text_color(col,0),
                                #anchor_x=align_hor, #anchor_y=align_vert, 
                                anchor_x='center', anchor_y='center',
                                batch=self.batch, group=self.foreground)
        text.visible = False
        # text = visual.TextStim(self.window, text=text, pos=self.convert_pos(pos), color=self.convert_color(col),
        #                        depth=-1, height=size/100, alignText=align_hor, anchorVert=align_vert,
        #                        wrapWidth=wrap_width)
        # text.setAutoDraw(False)
        return text

    def add_click_feedback(self, key):
        self.click_feedback_buttons.append(key)

    def create_text_field(self, size, pos, text_size, text_col, align_hor='left', align_vert='top', text='', languageStyle:str=None):
        """Creates and returns a psychopy Text object, the upper left corner aligned to the given position
        
        Args:
            size ((float, float)): The wrap width of the text, as a fraction of the window width
            pos ((float, float)): position of the upper left corner of the textfield, as two fractions of the window sizes
            text_col ((int, int, int)): The text color in RGB format
            text_size (int): The size of the text.
            align_hor (string): Horizontal alignment of the textfield ('left', 'right' or 'center')
            align_vert (string): Vertical alignment of the text ('top', 'bottom' or 'center')
            text (string): The text to be displayed
            languageStyle (): ????

        Returns:
            (visual.TextStim): A psychopy text object

        """
        # textfield = visual.TextStim(
        #     self.window,
        #     text=text,
        #     pos=self.convert_pos(pos),
        #     color=self.convert_color(text_col), depth=-1, height=text_size/100,
        #     alignText=align_hor, anchorVert=align_vert,
        #     wrapWidth=self.convert_size((size[0], 0))[0]
        # )
        # textfield.autoDraw = False
        # return textfield
        x,y=self.convert_pos(pos)
        w,h=self.convert_size(size)
        # TODO[]: use the language style info
        text = pyglet.text.Label(text, font_size=text_size, 
                                x=x, y=y, width=w, height=h,
                                color=self.convert_text_color(text_col,0),
                                anchor_x='center', anchor_y='center',
                                multiline=True,
                                #anchor_x=align_hor, #anchor_y=align_vert,
                                batch=self.batch, group=self.foreground)
        text.visible = False
        return text

    def create_icon(self, file, label_col, size, pos):
        """Creates and returns a psychopy image object.

        Args:
            file (str): Path to png file used as icon.
            label_col (tuple): The color to be used for the icon in RGB format.
            size (tuple): Size of the icon.  N.B. size is anchored on the center of the square
            pos (tuple): Position of the icon (centered).

        Returns
            (visual.ImageStim): A psychopy image object

        """
        x,y=self.convert_pos(pos)
        w,h=self.convert_size(size)
        if not os.path.exists(file):
            file = os.path.join(os.path.dirname(__file__),file)
        img = pyglet.image.load(file)
        img.anchor_x, img.anchor_y  = (img.width//2,img.height//2)
        icon = pyglet.sprite.Sprite(img,x,y,
                                    batch=self.batch, group=self.foreground)
        # N.B. Sprites set color and alpha separately
        icon.color = self.convert_color(label_col)
        icon.update(scale_x=w/icon.image.width, 
                    scale_y=h/icon.image.height)
        icon.visible = False
        return icon

    def create_line(self, vertices, color):
        """
        Creates and returns a psychopy line object.

        Args:
            vertices (list): List of x and y coordinates.
            color (string): Color of the line.

        Returns:
            (visual.Shapestim): A psychopy line.

        """
        vertices = [(self.convert_pos(v)) for v in vertices]
        line=[]
        for vi in range(len(vertices)-1):
            x,y,x2,y2 = *vertices[vi], *vertices[vi+1]
            line.append(shapes.Line(x,y,x2,y2,color=self.convert_color(color)))#,batch=self.batch)
        #line.visible = False
        return line

    def set_text(self, text_object, new_text):
        """Changes the displayed text of a psychopy text object

        Args:
            text_object (visual.TextStim): The text object to change
            new_text (str): The new text

        """
        text_object.begin_update()
        text_object.text=str(new_text)
        text_object.end_update()

    def set_text_size(self, text_object, text_size):
        """Sets the text size of the passed text_object.

        Args:
            text_object (Object): A reference to the to be changed text object
            text_size (int): The size of the text.
        """
        text_object.begin_update()
        text_object.h = text_size/100
        text_object.end_update()

    def change_button_color(self, shape, new_color):
        """Changes the color of the ShapeStim"""
        shape.color = self.convert_color(new_color)[:3]

    def set_line_color(self, shape, line_color):
        """Sets the border color of the specified shape to the provided line_color"""
        shape.border_color = self.convert_color(line_color)[:3]

    def toggle_text_render(self, text_object, mode):
        """Toggles the rendering of a psychopy text object

        Args:
            text_object (object): The text object to be toggled
            mode (bool): The bool that determines if the text object will be rendered or not
        """
        if isinstance(text_object,pyglet.text.Label):
            if not mode and text_object.visible:
                text_object.color = tuple(text_object.color[:3]) + (0,)
            elif mode and not text_object.visible:
                text_object.color = tuple(text_object.color[:3]) + (255,)
        text_object.visible = mode

    def toggle_shape_render(self, shape_object, mode):
        """Toggles the rendering of a psychopy shape object

        Args:
            shape_object (visual.ShapeStim): The shape object to be toggled
            mode (bool): The bool that determines if the text object will be rendered or not
        """
        shape_object.visible = mode

    def toggle_image_render(self, image_object, mode):
        """Toggles the rendering of a psychopy shape object

        Args:
            shape_object (visual.ShapeStim): The shape object to be toggled
            mode (bool): The bool that determines if the text object will be rendered or not
        """
        image_object.visible = mode


    def isposinrect(self,x,y,rx,ry,rw,rh):
        return rx-rw//2 < x and x < rx+rw//2 and ry-rh//2 < y and y < ry + rh//2

    def button_mouse_event(self, button, mouse_buttons=[0, 1, 2]):
        """Returns `True` if the mouse is currently inside the button and
        one of the mouse buttons is pressed. The default is that any of
        the 3 buttons can indicate a click; for only a left-click,
        specify `mouse_buttons=[0]`

        Args:
            button (visual.ShapeStim): A psychopy shape object to check for a mouse click
            mouse_buttons (list(int)): A list of mouse buttons (left button: 0. right button: 1, middle button: 2)

        Returns:
            (bool): Whether there was a mouse press with any of the given buttons on the passed object

        """
        # TODO[]: filter on the button pressed
        if self.window.last_mouse_release is None: return False
        mx,my,mb,mm = self.window.last_mouse_release
        return self.isposinrect(mx,my,button.x,button.y,button.width,button.height)

    def mouse_event(self, mouse_buttons):
        """Returns `True` if all passed buttons are currently pressed

        Args:
            mouse_buttons (list(int)): A list of mouse buttons (left button: 0. right button: 1, middle button: 2)

        Returns:
            (bool): `True` if all passed mouse_buttons are pressed, `False` otherwise

        """
        # TODO[]: Huh?
        if self.window.last_mouse_release is None: return False
        mx,my,button,modifiers = self.window.last_mouse_release
        return button in mouse_buttons

    def key_event(self, key_list=None):
        """Returns a list of keys that are pressed

        Args:
            key_list: Specify a list of keys to check. If None, will check all keys.

        Returns:
            list: a list of pressed keys

        """
        return self.window.last_key_press

    def convert_pos(self, pos):
        """Converts a position expressed in fractions of the window dimensions (between 0 and 1) into
        the pyglet coordinate system (pixels)

        Args:
            pos ((float, float)): A position passed as a tuple of two fractions

        Returns:
            ((float, float)): The converted positions

        """
        w,h=self.window.width,self.window.height
        return int(pos[0] * w), int(pos[1] * h)

    def convert_size(self, size):
        """Converts a size tuple expressed in fractions of the window dimensions (between 0 and 1) into
        the pyglet coordinate system pixels

        Args:
            size ((float, float)): A size passed as a tuple of two fractions

        Returns:
            ((float, float)): The converted size

        """
        return self.convert_pos(size)

    def convert_to_icon_size(self, size, scale=1.0):
        """Converts given size to the size used for creating the icon.

        The converted size is expressed in the coordinate system (-1,1)
        but ensures that the icon has the same height as width in pixels.
        The side length of this square is determined by the size of the
        smaller side of the key.

        Args:
            size ((float, float)): A size passed as a tuple of two fractions
            scale (float, optional): Scale of the icon
        Returns
            (float): The icon size

        """
        screen_ratio = self.window.size[0] / self.window.size[1]
        min_size = scale * min(size[0], size[1])
        icon_size = (min_size * (1 / screen_ratio), min_size) if screen_ratio > 1 else (min_size, min_size * screen_ratio)
        return self.convert_size(icon_size)

    def convert_color(self, col):
        """Converts a n RGB color (an int of 0 to 255 for each channel) into the pyglet format, which must have an alpha channel

        Args:
            col ((int, int, int): A color passed in the RGB format

        Returns:
            ((float, float, float)): The converted color

        """
        # return col[0]/127.5 - 1, col[1]/127.5 - 1, col[2]/127.5 - 1
        #if len(col)==3: col= tuple(col)+(255,)
        return col

    def convert_text_color(self, col, alpha=255):
        """Converts a n RGB color (an int of 0 to 255 for each channel) into the pyglet format, which must have an alpha channel

        Args:
            col ((int, int, int): A color passed in the RGB format

        Returns:
            ((float, float, float)): The converted color

        """
        # return col[0]/127.5 - 1, col[1]/127.5 - 1, col[2]/127.5 - 1
        if len(col)==3: col= tuple(col)+(alpha,)
        return col


    def draw(self, t, *args):
        """draws all active shapes and objects to the frame buffer

        Args:
            application: Not used for psychopy

        """
        for key in self.click_feedback_buttons:
            # handling timing of color feedback on clicking buttons:
            if key.pressed_frames == 0:
                key.reset_color()
            else:
                key.pressed_frames -= 1
        self.nframe = self.nframe + 1
        self.window.clear()
        self.application.handle_mouse_events()
        self.application.draw()
        # finally draw the actual batch
        self.batch.draw()
        # clear the UI state
        self.window.last_key_press=None
        self.window.last_mouse_press=None
        self.window.last_mouse_release=None
        self.window.last_mouse_motion=None

    def start(self, application, exit_keys):
        """Starts and runs the event loop

        Args:
            application (bci_keyboard.Application): The application to run
            exit_keys (list): A list of key names, which, if pressed will terminate the application
        """

        self.application = application
        self.nframe = 0

        # setup a key press handler, just store key-press in global variable
        self.window.push_handlers(on_key_press, on_text, on_mouse_press, on_mouse_release, on_mouse_motion)
        self.window.last_key_press=None
        self.window.last_text=None
        self.window.last_mouse_press=None
        self.window.last_mouse_release=None
        self.window.last_mouse_motion=None

        # override window's flip method to record the exact *time* the
        # flip happended
        import types
        self.window.flip = types.MethodType(lambda self: flip_callback(self,application), self.window)

        # call the draw method as fast as possible, i.e. at video frame rate!
        pyglet.clock.schedule(self.draw)
        # mainloop
        pyglet.app.run()
        pyglet.app.EventLoop().exit()
        self.window.set_visible(False)

    def flip(self):
        """Draws the frame buffer to the screen"""
        self.window.flip()

    def quit(self, keys):
        """Quits the application when one of the passed keys is being pressed

        Args:
            keys (list): A list of key names

        Returns:
            (bool): `True` if one of the keys is being pressed

        """
        keys = event.getKeys(keyList=keys)
        if keys:
            core.quit()
        return keys
