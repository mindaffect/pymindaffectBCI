"""This module is for text field management.

A ``TextField`` has the following functionality:
 * Graphical text field for display
 * Separation of the live and editable logical text line, and the previously written (historic) lines.
 * Getters and setters for the logical text line's state
 * Add and undo function to edit logical text line
 * Replace last word function for logical text line
 * A boolean function is_empty() to check if the logical text line is empty
 * A function to update the word suggestions
 * A function to clear the TextField

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

from mindaffectBCI.examples.presentation.smart_keyboard.settings_manager import SettingsManager


class TextField:
    """Manages logical and graphical text lines

    Args:
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        style_sheet (dict): Style instructions for the keyboard.
        update_suggestions (func): Outside function to update the suggestion keys.
        max_history_store (int): The max amount of historic lines to store.
        max_history_show (int): The max amount of historic lines to show. Keep low to not render out-of-screen text.

    Attributes:
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        style_sheet (dict): Style instructions for the keyboard.
        text_object (psychopy.visual.text.TextStim): A wrapper for the graphical text field.
        update_suggestions (func): Outside function to update the suggestion keys.
        history (list): A list of the previously typed lines.
        previous_state (str): Previous state of the logical text line.
        current_state (str): Current state of the logical text line.
    """
    def __init__(self, facade, style_sheet, update_suggestions, max_history_store=999, max_history_show=10,
                 predictor=None):
        self.facade = facade
        self.style_sheet = style_sheet
        self.settings_manager = SettingsManager.get_instance()
        self.text_size = self.settings_manager.get_text_size()
        self.languageStyle = self.settings_manager.get_language_style()
        self.text_object = self.facade.create_text_field(size=(.9, 1 - self.style_sheet["keyboard_size"]),
                                                         pos=(.5, 0.02 + self.style_sheet["keyboard_size"]),
                                                         text_size=self.text_size,
                                                         text_col=self.style_sheet["text_color"], align_vert='bottom',
                                                         languageStyle=self.languageStyle)
        self.update_suggestions = update_suggestions
        self.predictor = predictor

        # Graphical text
        self.facade.set_text(self.text_object, '_')
        self.max_history_show = max_history_show

        # Logical text
        self.max_history_store = max_history_store
        self.history = []
        self.previous_state = ''
        self.current_state = ''

        # Attaches this class to settings_manager:
        self.settings_manager.attach(self)

    def update(self):
        """Updates the text size at use if it has been changed."""
        new_size = self.settings_manager.get_text_size()

        if self.text_size == new_size:
            return

        self.text_size = new_size
        self.facade.set_text_size(text_object=self.text_object, text_size=self.text_size)

    def clear_text_field(self):
        """Clears the text field and the text field history."""
        self.history = []
        self.previous_state = ''
        self.current_state = ''
        self.update_display_text()

    def new_line(self):
        """"Moves the previous line into history and starts a new line."""
        if len(self.history) >= self.max_history_store:
            self.history.pop(0)  # Removes oldest line from history

        self.history.append(self.current_state)
        self.current_state = ''
        if self.predictor:
            self.predictor.update_frequencies(self.history[-1])
        self.update_display_text()

    def add(self, string):
        """Intelligently adds string to the current logical text line.

        Args:
            string (str): The string to be added to the logical text line.
        """
        self.previous_state = self.current_state

        # Autocomplete adds spaces after word suggestions. If the user wants to add punctuation
        # after a suggestion, the space is be removed.
        if len(self.current_state) > 0 and string[0] in self.style_sheet['punctuation'] and self.current_state[-1] == ' ':
            self.current_state = self.current_state[:-1] + string + ' '

        else:
            self.current_state += string
        self.update_display_text()

    def backspace(self):
        """Implements backspace functionality as one would expect

        If the current line is empty, brings back the most recent historic line to be the current line.

        Intelligently determines what the previous state should/would be if going back to historic line."""

        # If current line is empty, go back to previous line if it exists.
        if self.is_empty() and len(self.history) > 0:
            self.current_state = self.history.pop()

            # 'Synthetically' set the previous state to be current state -1 character.
            if not self.is_empty():
                self.previous_state = self.current_state[:-1]
            else:
                self.previous_state = ''

        elif self.current_state == self.previous_state:
            self.current_state = self.current_state[:-1]
            self.previous_state = self.current_state
        else:
            self.current_state = self.previous_state
        self.update_display_text()

    def replace_last_word(self, string):
        """Replaces last word of the current logical text line with a string.

        Args:
            string (str): The string to replace the last word with.
        """
        line = self.current_state
        while not len(line) == 0 and not line[-1] == ' ':
            line = line[:len(line) - 1]

        self.previous_state = self.current_state
        if string in self.style_sheet["punctuation"]:
            self.current_state = line = line[:-1]
        self.current_state = line + string + " "
        self.update_display_text()

    def set_state(self, string):
        """Setter for the logical text line.

        Args:
            string (str): New state for the logical text line.
        """
        self.previous_state = self.current_state
        self.current_state = string
        self.update_display_text()

    def get_state(self):
        """Getter for the current logical text line.

        Returns:
            The current state of the logical text line.
        """
        return self.current_state

    def get_text_history(self):
        """Returns a string containing the text as it appears on screen.

        This function can be used to print the entire text history, or save it to a file.

        Returns:
            String containing entire text history.
        """
        # Joins all lines with empty lines in between
        return "\n\n".join(self.history) + "\n\n" + self.current_state

    def is_empty(self):
        """Boolean function to check whether the current logical text line is empty.

        Returns:
            True if logical text line is empty, False otherwise.
        """
        return self.current_state == ''

    def update_display_text(self):
        """
        Updates the graphical text field and calls the outside function update_suggestions()

        The history and current line are seperated by a blank line to clearly indicate what the currently active
        line is.
        """
        # Show the last max_history_show nr of lines and current state seperated by blank line.
        self.facade.set_text(self.text_object,
                             "\n\n".join(self.history[-self.max_history_show:]) + "\n\n" + self.current_state + "_")
        self.update_suggestions()

    def activate(self):
        """Activate rendering of the text field."""
        self.facade.toggle_text_render(self.text_object, True)

    def deactivate(self):
        """Deactivate rendering of the text field."""
        self.facade.toggle_text_render(self.text_object, False)
