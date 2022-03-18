"""
This module contains a single class representing a keypad.

It contains the following visual and functional elements:
 * keys used for typing
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

from mindaffectBCI.examples.presentation.smart_keyboard.windows.window import Window
from mindaffectBCI.examples.presentation.smart_keyboard.key import Key
from mindaffectBCI.examples.presentation.smart_keyboard.key_type import KeyType


class KeypadWindow(Window):
    """
    A Window representing a keypad.

    Args:
        parent (windows.window.Window): The parent of this window.
        facade (framework_facade.FrameworkFacade): Contains the GUI-specific functionality.
        style (dict): A dictionary containing configurable style settings.
        keypad_layout (list): Layout of the keypad.
        use_flickering (bool): A boolean indicating whether or not to use flickering.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
    """

    def __init__(self, parent, facade, style, keypad_layout, use_flickering, text_field, noisetag=None):
        super().__init__(
            parent=parent,
            facade=facade,
            style=style,
            use_flickering=use_flickering,
            noisetag=noisetag
        )

        self.text_field = text_field
        self.windows = None
        self.active_window = None
        self.switched_windows = False

        self.key_padding = self.style["key_padding"]
        self.vertical_ratio = self.style["keypad_size"]
        self.state2color = style["state2color"]

        self.keys = self.build_keypad(keypad_layout)

    def get_keys(self):
        """Returns the keys of this Window."""
        return self.keys

    def activate(self):
        """Activates the visual and functional elements of this Window."""
        for key in self.keys:
            key.toggle_render(True)

    def deactivate(self):
        """Deactivates the visual and functional elements of this Window."""
        for key in self.keys:
            key.toggle_render(False)

    def switch_window(self, window_name):
        """
        Switches to Window specified by window_name.

        Args:
            window_name (str): The name of the Window to switch to.
        """
        if self.windows:
            if window_name in self.windows:
                self.windows[self.active_window].deactivate()
                self.windows[window_name].activate()
                self.active_window = window_name
                self.switched_windows = True
            else:
                self.parent.switch_window(window_name)
                self.switched_windows = True
        else:
            self.parent.switch_window(window_name)
            self.switched_windows = True

    def handle_mouse_events(self):
        """Handles mouse events within this Window."""
        if not (self.switched_windows or self.parent.switched_windows):
            for key in self.keys:
                key.handle_mouse_events(self)
            if self.windows:
                self.windows[self.active_window].handle_mouse_events()
        else:
            if not self.facade.mouse_event([0]):
                self.parent.switched_windows = False
                self.switched_windows = False

    def get_window(self, window_name):
        """
        Gives access to Window specified by window_name.

        Args:
            window_name (str): The name of the Window to get.
        """
        if self.windows:
            if window_name in self.windows:
                return self.windows[window_name]
            else:
                return self.parent.get_window(window_name)
        else:
            return self.parent.get_window(window_name)

    def draw(self, offset, stimulus_state, target_idx=-1):
        """Colors all buttons of the keypad with with their respective stimulus state's color."""
        # Drawing with flickering to be implemented
        for i, key in enumerate(self.keys):
            if target_idx == -1 or i == target_idx:
                key.change_color(self.state2color[str(stimulus_state[i + offset])])

    def text_to_speech(self):
        self.parent.text_to_speech()

    def build_keypad(self, keypad_layout):
        """
        Builds keypad.

        Args:
            keypad_layout (list): The layout used for building the keypad.

        Returns:
             keys (list): A list of keys
        """
        n_rows = len(keypad_layout)
        key_height = self.vertical_ratio / n_rows

        keys = []

        for i, row in enumerate(keypad_layout):
            n_keys = len(row)
            key_width = 1 / n_keys

            for j, key in enumerate(row):
                x_pos = (j + 0.5) * key_width
                y_pos = self.vertical_ratio - (key_height * (i + 0.5))

                pos = (x_pos, y_pos)
                size = (key_width - self.key_padding, key_height - self.key_padding)

                key_type = KeyType[key[0]]
                key_label = key[1]

                keys.append(Key(self.facade, key_type, size, pos, key_label, self.button_color,
                                self.line_color, self.label_color))

        return keys
