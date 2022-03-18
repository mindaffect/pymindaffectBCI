"""This module is used as a facade between the bci keyboard and the psychopy framework.

This module contains a single class ``PsychopyFacade``.

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

from psychopy import core, visual, event
from mindaffectBCI.examples.presentation.smart_keyboard.app_exceptions import Logger
from mindaffectBCI.examples.presentation.smart_keyboard.framework_facade import FrameworkFacade
from pathlib import Path
import cv2
import os
import re as regex


class PsychopyFacade(FrameworkFacade):
    """Implements the psychopy functionality into the facade.

    Args:
        size ((int,int)): Size of window for the keyboard to run in.
        full_screen (bool): Whether or not to run the application in full screen mode

    Attributes:
        window (visual.Window): The psychopy window object
        mouse (event.Mouse): The psychopy mouse object
    """

    def __init__(self, size, wait_blanking=True, full_screen=False):
        self.window = visual.Window(
            size=size,
            color=(-1, -1, -1),
            fullscr=full_screen,
            checkTiming=True,
            waitBlanking=wait_blanking,
            winType='pyglet'
        )
        self.window.recordFrameIntervals = True
        # set threshold for frame duration:
        self.window.refreshThreshold = 1 / 60 + 0.004
        self.mouse = event.Mouse(win=self.window)
        self.click_feedback_buttons = []

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
            button_label = self.create_text(label, label_color, pos)
        else:
            button_label = self.create_icon(label, label_color, size, pos)
        return button, button_label

    def create_rect(self, size, pos, color, line_color):
        """Creates and returns a PsychoPy Rect(angle) object according to the passed arguments

        Args:
            size ((float, float)): The width and height of the rectangle as a fraction of the window size
            pos ((float, float)): The position of the centre of the rectangle, expressed as fractions of the window sizes
            color: ((int, int, int)): The color of the rectangle, in RGB format
            line_color ((int, int, int)): The color of the rectangle border, in RGB format

        Returns:
            (visual.Rect): A psychopy rectangle object
        """
        rect = visual.Rect(self.window, size=self.convert_size(size), pos=self.convert_pos(pos),
                           fillColor=self.convert_color(color), lineColor=self.convert_color(line_color),
                           lineWidth=3)
        return rect

    def create_text(self, text='', col=(255, 255, 255), pos=(0, 0), size=10, align_hor='center', align_vert='center', wrap_width=1):
        """Creates and returns a psychopy Text object, centered at the passed position

        Args:
            text (string): The text
            col: ((int, int, int)): text color in RGB format
            pos ((float, float)): Position of the text on the screen, expressed as two fractions relative to window dimensions
            size (int): Text size
            align_hor (string): Horizontal alignment of the text ('left', 'right' or 'center')
            align_vert (string): Vertical alignment of the text ('top', 'bottom' or 'center')
            wrap_width (float): the maximal horizontal width of the text; it will wrap to a newline at this point

        Returns:
            (visual.TextStim): A psychopy text object

        """
        text = visual.TextStim(self.window, text=text, pos=self.convert_pos(pos), color=self.convert_color(col),
                               depth=-1, height=size/100, alignText=align_hor, anchorVert=align_vert,
                               wrapWidth=wrap_width)
        text.setAutoDraw(False)
        return text

    def add_click_feedback(self, key):
        self.click_feedback_buttons.append(key)

    def create_text_field(self, size, pos, text_size, text_col, align_hor='left', align_vert='top', text='',languageStyle='LTR'):
        """Creates and returns a psychopy Text object, the upper left corner aligned to the given position
        
        Args:
            size ((float, float)): The wrap width of the text, as a fraction of the window width
            pos ((float, float)): position of the upper left corner of the textfield, as two fractions of the window sizes
            text_col ((int, int, int)): The text color in RGB format
            text_size (int): The size of the text.
            align_hor (string): Horizontal alignment of the textfield ('left', 'right' or 'center')
            align_vert (string): Vertical alignment of the text ('top', 'bottom' or 'center')
            text (string): The text to be displayed

        Returns:
            (visual.TextStim): A psychopy text object

        """
        textfield = visual.TextStim(
            self.window,
            text=text,
            pos=self.convert_pos(pos),
            color=self.convert_color(text_col), depth=-1, height=text_size/100,
            alignText=align_hor, anchorVert=align_vert,
            wrapWidth=self.convert_size((size[0], 0))[0],
            languageStyle=languageStyle
        )
        textfield.autoDraw = False
        return textfield

    def create_icon(self, file, label_col, size, pos):
        """Creates and returns a psychopy image object.

        Args:
            file (str): Path to png file used as icon.
            label_col (tuple): The color to be used for the icon in RGB format.
            size (tuple): Size of the icon.
            pos (tuple): Position of the icon (centered).

        Returns
            (visual.ImageStim): A psychopy image object

        """
        try:
            # reverse RGB to BGR for use in cv2:
            fill_color = [label_col[2], label_col[1], label_col[0]]
            # Fix incorrect path separator for non-Windows systems:
            file = os.path.normpath(file)
            if not os.path.exists(file):
                file = os.path.join(os.path.dirname(os.path.abspath(__file__)),file)
            # read original image:
            img = cv2.imread(file, -1)
            # split into separate channels:
            B, G, R, A = cv2.split(img)
            # merge color channels back into a single image:
            edited_no_alpha = cv2.merge((B, G, R))
            # set all pixels to the specified color:
            edited_no_alpha[True] = fill_color
            # split into separate channels again:
            B, G, R, = cv2.split(edited_no_alpha)
            # merge with the alpha channel back into a single image:
            edited_with_alpha = cv2.merge((B, G, R, A))
            # split the file extension from path/file name:
            path_parts = file.split('.')
            # add '_recolored' to the file name:
            path_parts[-2] += '_recolored'
            # join everything back together:
            updated_file = '.'.join(path_parts)
            print("Writing file to: {}".format(updated_file))
            # save the edited image as copy under the new file name:
            cv2.imwrite(updated_file, edited_with_alpha)
            # load recolored copy as icon:
            icon = visual.ImageStim(self.window, image=updated_file, size=self.convert_to_icon_size(size, 0.5), pos=self.convert_pos(pos), depth=-1)
            # delete recolored file:
            if os.path.exists(updated_file):
                os.remove(updated_file)
        except IOError:
            Logger.log_unknown_icon(file)
            icon = self.create_text(file, label_col, pos)
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
        return visual.ShapeStim(self.window, closeShape=False, vertices=vertices, autoDraw=False, lineColor=color)

    def set_text(self, text_object, new_text):
        """Changes the displayed text of a psychopy text object

        Args:
            text_object (visual.TextStim): The text object to change
            new_text (str): The new text

        """
        text_object.text = new_text

    def set_text_size(self, text_object, text_size):
        """Sets the text size of the passed text_object.

        Args:
            text_object (Object): A reference to the to be changed text object
            text_size (int): The size of the text.
        """
        text_object.setHeight(text_size/100)

    def change_button_color(self, shape, new_color):
        """Changes the color of the ShapeStim"""
        shape.fillColor = self.convert_color(new_color)

    def set_line_color(self, shape, line_color):
        """Sets the border color of the specified shape to the provided line_color"""
        shape.lineColor = self.convert_color(line_color)

    def toggle_text_render(self, text_object, mode):
        """Toggles the rendering of a psychopy text object

        Args:
            text_object (object): The text object to be toggled
            mode (bool): The bool that determines if the text object will be rendered or not
        """
        text_object.setAutoDraw(mode)
        
    def toggle_image_render(self, image_object, mode):
        """Toggles the rendering of a psychopy image object
        Args:
            image_object (object): The image object to be toggled
            mode (bool): The bool that determines if the text object will be rendered or not
        """
        image_object.setAutoDraw(mode)          

    def toggle_shape_render(self, shape_object, mode):
        """Toggles the rendering of a psychopy shape object

        Args:
            shape_object (visual.ShapeStim): The shape object to be toggled
            mode (bool): The bool that determines if the text object will be rendered or not
        """
        shape_object.autoDraw = mode

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
        return self.mouse.isPressedIn(button, buttons=mouse_buttons)

    def mouse_event(self, mouse_buttons):
        """Returns `True` if all passed buttons are currently pressed

        Args:
            mouse_buttons (list(int)): A list of mouse buttons (left button: 0. right button: 1, middle button: 2)

        Returns:
            (bool): `True` if all passed mouse_buttons are pressed, `False` otherwise

        """
        buttons = self.mouse.getPressed()
        for button in mouse_buttons:
            if not buttons[button]:
                return False
        return True

    def key_event(self, key_list=None):
        """Returns a list of keys that are pressed

        Args:
            key_list: Specify a list of keys to check. If None, will check all keys.

        Returns:
            list: a list of pressed keys

        """
        keys = event.getKeys(keyList=key_list)
        return keys

    def convert_pos(self, pos):
        """Converts a position expressed in fractions of the window dimensions (between 0 and 1) into
        the psychopy coordinate system (between -1 and 1)

        Args:
            pos ((float, float)): A position passed as a tuple of two fractions

        Returns:
            ((float, float)): The converted positions

        """
        return pos[0] * 2 - 1, pos[1] * 2 - 1

    def convert_size(self, size):
        """Converts a size tuple expressed in fractions of the window dimensions (between 0 and 1) into
        the psychopy coordinate system (between -1 and 1)

        Args:
            size ((float, float)): A size passed as a tuple of two fractions

        Returns:
            ((float, float)): The converted size

        """
        return size[0] * 2, size[1] * 2

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
        """Converts a n RGB color (an int of 0 to 255 for each channel) into the psychopy format
        (a float between -1 and 1 for each channel)

        Args:
            col ((int, int, int): A color passed in the RGB format

        Returns:
            ((float, float, float)): The converted color

        """
        return col[0]/127.5 - 1, col[1]/127.5 - 1, col[2]/127.5 - 1

    def draw(self, application):
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
        application.draw()

    def start(self, application, exit_keys):
        """Starts and runs the event loop

        Args:
            application (bci_keyboard.Application): The application to run
            exit_keys (list): A list of key names, which, if pressed will terminate the application
        """
        while True:
            self.draw(application)
            application.handle_mouse_events()
            self.flip()
            application.set_flip_time() # flip-time recording callback
            if self.quit(exit_keys):  # press 'Q'(uit) or Escape to close the application
                break

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
