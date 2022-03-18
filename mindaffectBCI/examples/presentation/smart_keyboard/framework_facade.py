"""This module serves as an interface between a selected GUI and the ``Keyboard`` class.

**Usage**

Set up a class which implements (inherits) this interface. Preferably name it ``(some GUI)Facade``.
The class must contain the methods below and must override them by providing the described
GUI specific functionality.

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

class FrameworkFacade:

    def get_window(self):
        """Gets the window of the application.

        Returns:
            The window.
        """
        pass

    def create_button(self, size, pos, col, line_col, label_col, label, icon):
        """Creates a button and a label on that button in the specific GUI.

        This method is called upon creation of a ``Key`` instance. It has to
        set up some form of a geometric shape and a textfield centered on it.
        For now, only those geometric shapes are supported whose size can be
        described by ``size ((float, float))``.

        Args:
            size ((float, float)): Size of the button relative to the window size.
            pos ((float, float)): Position of the button in window with lower left corner (0,0), upper right corner (1,1).
            col ((int, int, int)): RGB value of the color used for filling.
            line_col ((int, int, int)): RGB value of the color used for surrounding line.
            label_col ((int, int, int)): RGB value of the color used for the label.
            label (str): The label to be displayed on the button.
            icon (str): Path to a png file from which the icon is created.

        Returns:
            The button

        """
        pass

    def create_text(self, text, col, pos, size, align_hor, align_vert, wrap_width):
        """Creates a text object in the specific GUI.

        Args:
            text (str): The text to be created.
            col ((int, int, int)): RGB value of the color used for filling.
            pos ((float, float)): Position of the text in window.
            size (int): Size of the text.
            align_hor (str): Horizontal alignment.
            align_vert (str): Vertical alignment.
            wrap_width (float): Width of the textbox.
        """
        pass

    def create_text_field(self, size, pos, text_size, text_col, align_text, anchor_vert, text):
        """Creates a textfield in the specific GUI.

        This method is called upon creation of a ``Key`` instance. It has to
        set up the output textfield of the keyboard.

        Args:
            size ((float, float)): The size of the textfield relative to the window size.
            pos ((float, float)): Position of the center of the textfield in window with lower left corner (0,0), upper right corner (1,1).
            text_size (int): The size of the text.
            text_col ((int, int, int)): Color of the text to be displayed in the textfield.

        Returns:
            The textfield

        """
        pass

    def create_icon(self, file, label_col, pos, size):
        """Creates an icon in the specific GUI.

        Args:
            file (string): Path to a png file from which the icon is created.
            size (float, float): Icon size fitted on the size of the shorter side of the key.
            pos ((float, float)): Position of the icon on the screen, expressed as two fractions relative to window dimensions.
            label_col ((int, int, int)): The color of the button label which is replaced by the icon, in RGB format.

        Returns:
            The icon.

        """
        pass

    def create_line(self, vertices, color):
        """Creates and returns a line object.

        Args:
            vertices (list): List of x and y coordinates.
            color (string): Color of the line.

        Returns:
            A line.

        """
        pass

    def set_text(self, text_object, new_text):
        """Accesses the textfield of the specific GUI and updates it.

        Args:
            text_object (object): The textfield to be accessed.
            new_text (str): The string to update the textfield.

        """
        pass

    def set_text_size(self, text_object, text_size):
        """Sets the text size of the textfield.

        Args:
            text_object (Object): A framework dependent reference to the to be changed text object
            text_size (int): The size of the text.
        """
        pass

    def set_line_color(self, shape, line_color):
        """

        Args:
            shape: A reference to the framework specific shape object to change the border color of
            line_color: The color to set the border color to

        """
        pass

    def change_button_color(self, shape, new_color):
        """

        Args:
            shape: A reference to the framework specific shape object to change the color of
            new_color: The to be used color

        """
        pass

    def toggle_text_render(self, text_object, mode):
        """Toggles the rendering of a text_object of the specific GUI

        Args:
            text_object (object): The text object to be toggled
            mode (bool): The bool that determines if the text object will be rendered or not
        """
        pass

    def toggle_shape_render(self, shape_object, mode):
        """Toggles the rendering of a shape_object of the specific GUI

        Args:
            shape_object (object): The text object to be toggled
            mode (bool): The bool that determines if the text object will be rendered or not
        """
        pass

    def button_mouse_event(self, button, mouse_buttons):
        """Listens for mouse clicks on a received button.

        Args:
            button (object): A button in the specific GUI.
            mouse_buttons (optional): The type of mouse clicks to listen for.

        Return:
            True if there is a click, False if there is no click

        """
        pass

    def convert_pos(self, pos):
        """Converts general position into GUI specific position for buttons and textfields.

        Converts the position from a window with lower left corner (0,0), upper right corner (1,1) to the GUI
        specific position. The position provided marks the center of a textfield or button.

        Args:
            pos ((float, float)): Position of the center of the textfield in window with lower left corner (0,0), upper right corner (1,1).

        Returns:
            Converted position

        """
        pass

    def convert_size(self, size):
        """Converts general size into GUI specific size for buttons and textfields.

        The provided size represents the size of a button or textfield relative to the window
        (0.5,0.5) = (50%,50%).

        Args:
            size ((float, float)): Size relative to the window.

        Returns:
            Converted size

        """
        pass

    def convert_color(self, col):
        """Converts RGB value into GUI specific color definition.

        Args:
            col ((int, int, int)): RGB value of a color.

        Returns:
            Converted color

        """
        pass

    def draw(self, application):
        """Updates the GUI specific window with new content."""

        pass

    def flip(self):
        """Draws the frame buffer to the screen"""

        pass

    def start(self, application, exit_keys):
        """Starts the GUI specific event loop.

        There should be an option for exiting the application (e.g. press 'Q'(uit)).

        """
        pass

    def quit(self, keys):
        """Called in start() to exit the application."""

        pass
