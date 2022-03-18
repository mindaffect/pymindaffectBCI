"""
This module contains a single class representing a menu.

It contains the following visual and functional elements:
 * four submenu windows
 * navigation bar
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
from mindaffectBCI.examples.presentation.smart_keyboard.windows.submenu_window import SubMenuWindow
from mindaffectBCI.examples.presentation.smart_keyboard.key import Key
from mindaffectBCI.examples.presentation.smart_keyboard.key_type import KeyType


class MenuWindow(Window):
    """
    A Window representing a menu.

    Args:
        parent (windows.window.Window): The parent of this window.
        facade (framework_facade.FrameworkFacade): Contains the GUI-specific functionality.
        style (dict): A dictionary containing configurable style settings.
        menu_config (dict): Configuration for the menu.
        use_flickering (bool): A boolean indicating whether or not to use flickering.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
    """
    def __init__(self, parent, facade, style, menu_config, use_flickering, noisetag=None):
        super().__init__(
            parent=parent,
            facade=facade,
            style=style,
            use_flickering=use_flickering,
            noisetag=noisetag
        )

        self.vertical_ratio = style["Navigation"]["vertical_ratio"]
        self.horizontal_ratio = style["Navigation"]["horizontal_ratio"]

        self.windows = {
            "GeneralMenu": SubMenuWindow(
                parent=self,
                facade=facade,
                submenu_layout=menu_config["SubMenus"]["General"],
                submenu_style=style["General"],
                style=style,
                use_flickering=use_flickering,
                noisetag=noisetag
            ),

            "SaveMenu": SubMenuWindow(
                parent=self,
                facade=facade,
                submenu_layout=menu_config["SubMenus"]["Save"],
                submenu_style=style["Save"],
                style=style,
                use_flickering=use_flickering,
                noisetag=noisetag
            ),

            "ConfigurationMenu": SubMenuWindow(
                parent=self,
                facade=facade,
                submenu_layout=menu_config["SubMenus"]["Configuration"],
                submenu_style=style["Configuration"],
                style=style,
                use_flickering=use_flickering,
                noisetag=noisetag
            )
        }
        
        self.logo = self.facade.create_icon(
            "key_icons\MindAffect_Logo.png",
            label_col=self.style.get("MA_orange",(128,128,0)),
            size=(.1,.1),
            pos=(.95,.05),
        )
        
        self.active_window = "GeneralMenu"
        self.switched_windows = False
        self.use_flickering = use_flickering

        self.keys = self.build_navigation_bar(menu_config["Navigation_bar"])

    def get_keys(self):
        """Returns the keys of this Window."""
        return self.keys

    def activate(self):
        """Activates all visual and functional elements of this Window."""
        for key in self.keys:
            key.toggle_render(True)
        self.windows[self.active_window].activate()
        self.facade.toggle_image_render(self.logo, True)
        if not self.use_flickering:
            self.noisetag.modeChange("idle")

    def deactivate(self):
        """Deactivates all visual and functional elements of this Window."""
        for key in self.keys:
            key.toggle_render(False)
        self.windows[self.active_window].deactivate()
        self.facade.toggle_image_render(self.logo, False)

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

    # The draw method is not used yet (no flickering available here).
    def draw(self, noisetag, last_flip_time, target_idx=-1):
        """
        Changes the state of the buttons/keys within this Window.
        It draws the display with the colors given by the Noisetag module if flickering is active.

        Args:
            noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
            last_flip_time (int): Timestamp of last screen update, i.e. buffer flip.
            target_idx (int): (Optional) index of the target stimulus.
        """
        pass

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

    # The draw method is not used yet (no flickering available here).
    def select_key(self, objID):
        """Cannot have functionality in Calibration mode."""
        pass

    def build_navigation_bar(self, navigation_layout):
        """
        Builds navigation keys.

        Args:
            navigation_layout (list): Layout of the navigation keys.
        """
        n_keys = len(navigation_layout)
        key_width = self.horizontal_ratio/(n_keys-1)
        key_height = self.vertical_ratio

        keys = []

        for i, key in enumerate(navigation_layout):
            x_pos = (i + 0.5) * key_width

            if i == (n_keys-1):
                x_pos -= 0.5 * key_width
                key_width = 1 - self.horizontal_ratio
                x_pos += 0.5 * key_width

            y_pos = 1 - (0.5 * key_height)

            size = (key_width, key_height)
            pos = (x_pos, y_pos)

            key_type = KeyType[key[0]]
            key_label = key[1]

            keys.append(Key(self.facade, key_type, size, pos, key_label, self.button_color,
                            self.line_color, self.label_color))
        return keys
