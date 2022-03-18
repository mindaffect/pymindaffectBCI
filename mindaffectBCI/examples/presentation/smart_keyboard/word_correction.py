"""This module uses a frequency list for word correction. It supports different languages.

The module contains a single class ``WordCorrection``.
Make an instance of this class in order to use it.

**Usage example**::

        import WordCorrection
        self.correction = WordCorrection(edit_distance=2)
        correction.correct("Testing")
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

from symspellpy import SymSpell, Verbosity
import os.path

from mindaffectBCI.examples.presentation.smart_keyboard.settings_manager import SettingsManager


class WordCorrection:
    """
    Provides the ability to correct input.
    """

    __instance = None

    @staticmethod
    def get_instance(edit_distance=None):
        """ Static access method. """
        if WordCorrection.__instance is None:
            WordCorrection(edit_distance)
        return WordCorrection.__instance

    def __init__(self, edit_distance=None):
        """
            Creates the SpellChecker object to be used for word correction.
            Language frequency lists are downloaded from a Git dictionary.

            Args:
                edit_distance (int): The maximum edit distance used for the correction (default is 2).
        """
        if WordCorrection.__instance is not None:
            raise Exception("An instance of this class already exists")
        else:
            WordCorrection.__instance = self

            self.settings_manager = SettingsManager.get_instance()
            self.settings_manager.attach(self)  # attach the object so it gets observed when the language changes

            self.language = self.settings_manager.get_language()

            self.spell = SymSpell(max_dictionary_edit_distance=edit_distance)
            self.spell.load_dictionary(os.path.join(os.path.dirname(os.path.abspath(__file__)),"dictionaries", "frequency_lists", self.language + ".txt"),
                                       0, 1, encoding="utf-8")

    def update(self):
        """Updates the language at use if it has been changed."""

        new_lang = self.settings_manager.get_language()
        if self.language == new_lang:
            return

        self.language = new_lang
        self.spell.load_dictionary(os.path.join(os.path.dirname(os.path.abspath(__file__)),"dictionaries", "frequency_lists", self.language + ".txt"),
                                   0, 1, encoding="utf-8")

    def correct(self, word):
        """
        Returns a correction of the word.

         Args:
             word (string): The word to be corrected.

        Returns:
            string: Word as predicted by the module.
        """

        suggestions = self.spell.lookup(word,
                                        verbosity=Verbosity.CLOSEST,
                                        include_unknown=True,
                                        transfer_casing=True)

        suggestion = str(suggestions[0]).split(", ")[0]

        return suggestion
