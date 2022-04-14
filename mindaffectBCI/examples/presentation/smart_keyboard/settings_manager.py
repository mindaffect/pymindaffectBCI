"""
This module manages settings provided by ``user_config.json``.

It contains a single class SettingsManager.

A SettingsManager receives the already loaded in ``user_config.json`` and notifies
every attached class when changes are made to the file.

The class can be extended by any setting. In order to do so one must:
* Add the setting to ``user_config.json``.
* Add an 'attribute' to this class.
* Add a ``get_'attribute'`` and a ``set_'attribute'`` method to this class.
* Add the ``set_'attribute'`` method to the ``setting_to_function`` attribute.
* Pass the SettingsManager to the class that wants to be notified when changes are made to ``user_config.json``.
* Attach 'that class' to the SettingsManager by calling ``SettingsManager.attach('that class')``.
* Implement an 'update' method in 'that class' to be notified when (specific/used) settings are changed.
* For an example see ``text_to_speech.py``.
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

import os
import json
import mindaffectBCI.examples.presentation.smart_keyboard.dictionary_retrieval as dictionary_retrieval

class SettingsManager:
    """
    Manages the changes in user settings.

    Args:
        file (dict): The already loaded in JSON file containing the user settings.
    """

    __instance = None

    @staticmethod
    def get_instance(file=None):
        """ Static access method. """
        if SettingsManager.__instance is None:
            SettingsManager(file)
        return SettingsManager.__instance

    def __init__(self, file=None):
        if SettingsManager.__instance is not None:
            raise Exception("An instance of this class already exists")
        else:
            SettingsManager.__instance = self

            self.file = file

            self.language = self.file['settings']['language']
            self.language_style = self.file['settings']['language_style']
            self.text_size = self.file["settings"]["text_size"]
            self.word_prediction_correction = self.file["settings"]["word_prediction_correction"]
            self.basedirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)))

            if self.word_prediction_correction:
                # Make sure that the dictionary for word correction is downloaded:
                dictionary_retrieval.check_dictionary(self.language)

            self.observers = []

            self.setting_to_function = {
                "language": self.set_language,
                "language_style": self.set_language_style,
                "text_size": self.set_text_size,
                "word_prediction_correction": self.set_word_prediction_correction,
            }
            self.setting_from_function = {
                "language": self.get_language,
                "text_size": self.get_text_size,
                "language_style": self.get_language_style,
                "word_prediction_correction": self.get_word_prediction_correction,
            }

    def write(self):
        """Writes updates to the user_config file."""
        path = os.path.join(self.basedirectory,"configs","user_config.json")
        with open(path, "w", encoding="utf-16") as jsonFile:
            json.dump(self.file, jsonFile, indent=2, ensure_ascii=False)

        self.notify_all_observers()

    def set_text_size(self, text_size):
        """
        Sets text size, writes to file and notifies observers.

        Args:
            text_size (int): The new text size.
        """
        self.text_size = text_size
        self.file["settings"]["text_size"] = self.text_size
        self.write()

    def set_language(self, language):
        """
        Sets language, writes to file and notifies observers.

        Args:
            language (str): The new language as an IETF tag.
        """
        self.language = language
        self.file["settings"]["language"] = self.language

        # Make sure that the dictionary for word correction is downloaded:
        dictionary_retrieval.check_dictionary(self.language)

        self.write()

    def set_language_style(self, language_style):
        """
        Sets language style, writes to file and notifies observers.
        Args:
            language_style (str): The language style LTR or RTL
        """
        self.language_style = language_style
        self.file["settings"]["language_style"] = self.language_style
        self.write()    

    def set_word_prediction_correction(self, mode):
        """
        Sets word prediction and correction to mode, writes to file and notifies observers.

        Args:
            mode (bool): The new mode of word prediction.
        """
        self.word_prediction_correction = mode
        self.file["settings"]["word_prediction_correction"] = self.word_prediction_correction
        self.write()

    def get_language(self, lang_tag=False):
        """
        Returns the language being used.

        Return:
            (str): Language being used.
        """
        if lang_tag:
            return self.language.lower()
        return self.language

    def get_language_style(self):
        """
        Returns the language style being used.
        Return:
            (str): Language style being used.
        """    
        return self.language_style    
        
    def get_text_size(self):
        """
        Returns the text size being used.

         Return:
             (int): Text size being used.
        """
        return self.text_size

    def get_word_prediction_correction(self):
        """
        Returns the current mode of word prediction and correction.

         Return:
             (bool): Mode of word prediction.
        """
        return self.word_prediction_correction

    def notify_all_observers(self):
        """Notifies observers attached by calling their update methods."""
        for observer in self.observers:
            observer.update()

    def attach(self, observer):
        """
        Attaches observer class.

        Args:
            observer (Object): The class to attach.
        """
        self.observers.append(observer)

    def set_default(self):
        """Loads default settings."""
        for setting in self.file["settings"]:
            self.setting_to_function[setting](self.file["standard_settings"][setting])
            self.file["settings"][setting] = self.file["standard_settings"][setting]
