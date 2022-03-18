"""
This module contains a single class representing a submenu.

It contains the following visual and functional elements:
 * keys for settings accessible from this window
 * information texts for these settings
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


class SubMenuWindow(Window):
    """
    Subclass of Window, designed for displaying settings of a certain type within the menu.

    Args:
        parent (windows.window.Window): The parent of this window.
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        submenu_layout (dict): The key and info layout of the submenu.
        submenu_style (dict): Style instructions for building the submenu.
        style (dict): General style instructions used for coloring.
        use_flickering (bool): Activates or deactivates flickering.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
    """

    def __init__(self, parent, facade, submenu_layout, submenu_style, style, use_flickering, noisetag=None):
        super().__init__(
            parent=parent,
            facade=facade,
            style=style,
            use_flickering=use_flickering,
            noisetag=noisetag
        )

        self.windows = None
        self.active_window = None
        self.switched_windows = False

        # Layout settings:
        self.vertical_padding = submenu_style["vertical_padding"]
        self.horizontal_padding = submenu_style["horizontal_padding"]
        self.in_between_padding = submenu_style["in_between_padding"]
        self.vertical_ratio = submenu_style["vertical_ratio"]
        self.horizontal_ratio = submenu_style["horizontal_ratio"]

        self.keys, self.setting_info = self.build(submenu_layout)

    def get_keys(self):
        """Return keys of this Window."""
        return self.keys

    def activate(self):
        """Activates the visual and functional elements of this Window."""
        for key in self.keys:
            key.toggle_render(True)
        for text in self.setting_info:
            self.facade.toggle_text_render(text, True)

    def deactivate(self):
        """Deactivates the visual and functional elements of this Window."""
        for key in self.keys:
            key.toggle_render(False)
        for text in self.setting_info:
            self.facade.toggle_text_render(text, False)

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
        for key in self.keys:
            key.handle_mouse_events(self)

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

    def build(self, submenu_layout):
        """
        Builds the submenu.

        Args:
            submenu_layout (dict): The key and info layout of the submenu.
        """
        n_settings = len(submenu_layout)

        keys = []
        setting_info = []

        # Loop over settings within the submenu:
        for i, setting in enumerate(submenu_layout):
            n_keys = len(submenu_layout[setting])

            # Key height and width are calculations of the from: available space / number of settings, number of keys:
            key_height = (self.vertical_ratio - self.vertical_padding * 2 -
                          ((n_settings - 1) * self.in_between_padding)) / n_settings
            key_width = (self.horizontal_ratio - (2 * self.horizontal_padding)) / n_keys

            y_pos = self.vertical_ratio - (self.vertical_padding + key_height * (i + 0.5) + self.in_between_padding * i)

            # Loop over keys used for that setting:
            for j, key in enumerate(submenu_layout[setting]):
                x_pos = self.horizontal_padding + key_width * (j + 0.5)

                key_type = KeyType[key[0]]
                key_label = key[1]

                keys.append(Key(self.facade, key_type, (key_width, key_height), (x_pos, y_pos),
                                key_label, self.button_color, self.line_color, self.label_color))

            x_pos = self.horizontal_ratio + ((1 - self.horizontal_ratio) / 2) + self.horizontal_padding

            # Add info text to every setting:
            setting_info.append(self.facade.create_text(setting, self.label_color, (x_pos, y_pos), align_hor="left"))

        return keys, setting_info
