"""
This module is for loading in keyboards / keypads layouts from JSON files.

It contains a single class ``KeyboardLoader``.

A ``KeyboardLoader`` has the following functionality:
 * Static ``load_keyboard`` function which returns a list of objects that have been read
   from a JSON file. It contains all buttons the user wants to have on their keyboard.
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

import json
import os
from mindaffectBCI.examples.presentation.smart_keyboard.app_exceptions import Logger, Handler


class KeyboardLoader:
    """Facilitates the load_keyboard function to load in keyboards / keypad layouts from JSON files."""

    @staticmethod
    def load_json(file: str):
        """
        Reads a JSON file specified from an input string and returns it.

        Args:
            file (str): specified JSON file to read our keyboard layout from.

        Returns:
            A list of lists which contains strings for the button labels.
        """

        try:
            if not os.path.exists(file):
                file = os.path.join(os.path.dirname(os.path.abspath(__file__)),file)
            return json.load(open(file, "r", encoding="utf-16"))

        except OSError as exc:
            # Handles OSError, which includes FileNotFound errors among others, return mockup keyboard:
            Logger.log_layout_OS_error()

            # Could also load in our test_keyboard but don't do so in case the file was deleted:
            return Handler.mockup_keyboard()

        except Exception as exc:
            # Other exceptions for the generic case:
            Logger.handle_generic()
            return Handler.mockup_keyboard()
