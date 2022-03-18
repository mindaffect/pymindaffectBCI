"""This module defines the functionalities a key can be equipped with."""

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

from mindaffectBCI.examples.presentation.smart_keyboard.app_exceptions import Logger
from datetime import datetime
import os


class FunctionalProvider:
    """Interface to create a key functionality."""

    def apply(self, key, window):
        """To be overridden by classes inheriting this interface.

        Is called in handle_mouse_events of Key and applies this
        functionality when the equipped key is selected.
        """
        pass

    def update(self, key):
        """To be overridden by classes inheriting this interface.

        Is called in update of Key and updates the equipped key
        with changes in settings.
        """
        pass

    def retrieve(self, key):
        """To be overridden by classes inheriting this interface.

        Is called upon creation of a Key and provides the FunctionalProvider
        with necessary information.
        """
        pass


class BasicFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to add its label text to the text field."""

    def apply(self, key, window):
        window.text_field.add(key.label_text)


class BackspaceFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to delete the last character of the text field."""

    def apply(self, key, window):
        window.text_field.backspace()


class SpaceBarFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to add an empty space to the text field."""

    def apply(self, key, window):
        window.text_field.add(" ")


class EnterFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to add a new line to the text field."""

    def apply(self, key, window):
        window.text_field.new_line()


class SuggestionFunction(FunctionalProvider):
    """
    Provides the equipped key with the functionality
    to replace the last word in text field with its label text
    """
    def apply(self, key, window):
        window.text_field.replace_last_word(key.label_text)


class ClearFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to clear the text field."""

    def apply(self, key, window):
        window.text_field.clear_text_field()


class TTSFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to transform the text of text field into spoken output."""

    def apply(self, key, window):
        window.text_to_speech()


class SwitchWindowFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to activate the linked window.

    Args:
        link (str): The link to the to be activated window.
    """
    def __init__(self, link):
        self.link = link

    def apply(self, key, window):
        window.switch_window(self.link)


class PlusMinusFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to add or subtract 1 from a numerical setting.

    Args:
        link (str): Link to the setting name used in settings_manager.py.
        mode (bool): Regulatory boolean to decide on subtraction or addition (True = '+').
    """
    def __init__(self, link, mode):
        self.link = link
        self.mode = mode
        self.setting_size = None
        self.min_setting_size = 1
        self.max_setting_size = 80

    def apply(self, key, window):
        if not self.mode and self.setting_size > self.min_setting_size:
            key.settings_manager.setting_to_function[self.link](self.setting_size - 1)
        elif self.mode and self.setting_size < self.max_setting_size:
            key.settings_manager.setting_to_function[self.link](self.setting_size + 1)

    def update(self, key):
        self.setting_size = key.settings_manager.setting_from_function[self.link]()

    def retrieve(self, key):
        self.setting_size = key.settings_manager.setting_from_function[self.link]()


class SaveTextFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to save the text of the text field to a file."""

    def apply(self, key, window):
        try:
            # If the output folder does not exist yet, make it.
            path = "output"
            if not os.path.isdir(path):
                print(os.path.join("Directory \'.", path) + " does not exist yet. Creating it now.")
                os.mkdir(path)

            # write without overwriting existing files ("x"), using current datetime to identify file.
            file = open("output/output-%s.txt" % datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "x")
            file.write(window.get_window("Keyboard").text_field.get_text_history())
            file.close()
        except IOError as ioe:
            Logger.log_write_file_error(ioe)


class DefaultSettingsFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to reset all settings to their default value."""

    def apply(self, key, window):
        key.settings_manager.set_default()


class ToggleFunction(FunctionalProvider):
    """Provides the equipped key with functionality to toggle a setting to the value specified by mode.

    Args:
        link (str): Link to the setting name used in settings_manager.py.
        mode (bool, str, int): The value to be toggled.
    """
    def __init__(self, link, mode):
        self.link = link
        self.mode = mode
        self.toggled = False

    def apply(self, key, window):
        if not self.toggled:
            key.button_color = (34, 139, 34)
            key.change_color(key.button_color)
            key.settings_manager.setting_to_function[self.link](self.mode)
            self.toggled = True

    def update(self, key):
        setting = key.settings_manager.setting_from_function[self.link]()
        if setting != self.mode:
            key.button_color = (30, 30, 30)
            key.change_color(key.button_color)
            self.toggled = False
        else:
            key.button_color = (34, 139, 34)
            key.change_color(key.button_color)
            self.toggled = True     # Technically, this line is not necessary.

    def retrieve(self, key):
        setting = key.settings_manager.setting_from_function[self.link]()
        if setting == self.mode:
            key.button_color = (34, 139, 34)
            key.change_color(key.button_color)
            self.toggled = True


class DisplayFunction(FunctionalProvider):
    """Provides the equipped key with the functionality to display a setting's current value.

    Args:
        link (str): Link to the setting name used in setting_manager.py.
    """
    def __init__(self, link):
        self.link = link

    def update(self, key):
        display = key.settings_manager.setting_from_function[self.link]()
        key.facade.set_text(key.label, display)

    def retrieve(self, key):
        display = key.settings_manager.setting_from_function[self.link]()
        key.facade.set_text(key.label, display)

    def apply(self, key, window):
        pass
