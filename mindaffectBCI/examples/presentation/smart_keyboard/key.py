"""This module contains a single class, Key."""


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


class Key:
    """
    This class represents a key.

    Args:
        facade (framework_facade.FrameworkFacade): Provides this class with GUI specific functionality.
        key_type (KeyType): The type of the key.
        size (tuple): The size of the key proportionally to the coordinate system.
        position (tuple): The position of the key in the coordinate system.
        label_text (str): The text to be displayed on the label or the path to the file used as an icon.
        button_color (tuple): The color of the button in the RGB format.
        line_color (tuple): The color of the line in the RGB format.
        text_color (tuple): The color of the label text in the RGB format.
    """

    def __init__(self, facade, key_type, size, position, label_text, button_color, line_color,
                 text_color):
        self.facade = facade
        self.settings_manager = SettingsManager.get_instance()
        self.key_type = key_type
        self.position = position
        self.label_text = label_text
        self.button_color = button_color
        self.line_color = line_color
        self.was_pressed = False
        self.changed_feedback_color = False
        self.pressed_frames = 0
        self.button, self.label = self.build_key(size, position, label_text, button_color, line_color, text_color)

        self.settings_manager.attach(self)
        self.key_type.func_provider.retrieve(self)

    def build_key(self, size, position, label_text, button_color, line_color, text_color):
        """Builds key."""
        has_icon = self.key_type.has_icon

        return self.facade.create_button(size, position, button_color, line_color, text_color, label_text, has_icon)

    def toggle_render(self, mode):
        """Sets the rendering mode of the key."""
        self.facade.toggle_shape_render(self.button, mode)
        self.facade.toggle_text_render(self.label, mode)

    def handle_mouse_events(self, window):
        """Handles the selection event of the key."""
        currently_pressed = self.facade.button_mouse_event(self.button)

        if currently_pressed:
            if not self.was_pressed:
                window.reset_key_colors()
                self.pressed_frames = 15
                self.facade.add_click_feedback(self)
                self.change_color((34, 160, 34))
                self.key_type.func_provider.apply(self, window)
                self.was_pressed = True
                window.reset_flicker()
        else:
            self.was_pressed = False

    def set_label_text(self, new_label_text):
        """ Sets the label text of the key."""
        self.label_text = new_label_text
        self.facade.set_text(self.label, self.label_text)

    def change_color(self, new_button_color):
        """Changes the color of a key's button object."""
        self.facade.change_button_color(self.button, new_button_color)

    def set_line_color(self, new_color=None):
        """Sets the color of border line of the key to the given color"""
        # by default, use the default line color to reset the border:
        col = self.line_color
        # if a new color was specified, use that one:
        if new_color:
            col = new_color
        self.facade.set_line_color(self.button, col)

    def reset_color(self):
        """Resets the color of a key's button object."""
        self.change_color(self.button_color)

    def update(self):
        """Notifies the key with changes in settings."""
        self.key_type.func_provider.update(self)
