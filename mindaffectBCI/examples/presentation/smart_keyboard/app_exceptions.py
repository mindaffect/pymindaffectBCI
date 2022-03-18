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

"""
This module is for error and exception handling and logging. It serves as a an easy way to access log information
for certain kind of procedures that might go wrong when running the program. Most stuff that can go wrong is on the
side of user input when tinkering with the software. While all tinkering is encouraged as it is open source,
the most likely errors to occur are when changing settings in the dedicated json files for configuration.
Therefore these are caught, logged and handled when possible.

A ``Logger`` has the following functionality:
 * Functions that log for information and errors for individual exceptions in the code

A ``Handler`` has the following functionality:
 * Functions that handle errors and exceptions at certain points in the code to allow the program to
   continue running.

"""
import os
import sys
import logging
import datetime
import re as regex

# Ensure logging directory exists
if not os.path.isdir("logs"):
    os.mkdir("logs")
try:
    logging.basicConfig(filename=regex.sub(":", "", os.path.join("logs", "{}.log".format(datetime.datetime.now().isoformat()[:-7]))),
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        force=True)
except:
    logging.basicConfig(filename=regex.sub(":", "", os.path.join("logs", "{}.log".format(datetime.datetime.now().isoformat()[:-7]))),
                        format='%(asctime)s %(message)s',
                        filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Logger:
    @staticmethod
    def __console_message():
        """Informs the user in the console that an issue occurred and the exception can be found in the log."""
        print("!!!Something went wrong, to figure out the problem consult the log file!!!", file=sys.stderr)

    @staticmethod
    def log_generic():
        """Most generic function for logging some runtime error."""
        Logger.__console_message()
        logger.error("The program ran into something unexpected, see the Traceback for details.")

    @staticmethod
    def log_layout_OS_error():
        """Logs an error whenever the loading of json files throws an OS Error, which often means that
        the file can't be found.
        """
        Logger.__console_message()
        logger.exception("Something went wrong reading the key layouts files, see Traceback for details.\n"
                         "Please make sure the config uses the correct naming, for example: 'keypad_layouts/test_keyboard.json'.\n"
                         "This can be changed in the keyboard_config.json.")

    @staticmethod
    def log_layout_generic_error():
        """Logs when something unexpected went wrong when loading keyboard layout files."""
        Logger.__console_message()
        logger.error("Something went wrong loading the key layouts and we're unsure what\n"
                     "See the Traceback for details.")

    @staticmethod
    def log_config_key_error():
        """Logs when there is a key error for loading our config files, e.g.
        when there are typos in the config file.
        """
        Logger.__console_message()
        logger.exception("Something went wrong loading the keyboard configuration.\n"
                         "The specified key to load in could not be found in the config file.\n"
                         "See the Traceback for details")

    @staticmethod
    def log_config_error():
        """Logs when something unexpected happened when loading in the config"""
        Logger.__console_message()
        logger.exception("Something went wrong loading the keyboard configuration.\n"
                         "Please make sure there are no typos in the file.\n"
                         "See the Traceback for details.")
        sys.exit()

    @staticmethod
    def log_threading_error():
        """Logs whenever an error occurs when creating a new Thread."""
        Logger.__console_message()
        logger.error("Something went wrong creating a new Thread.\n"
                     "Continuing without multithreading.\n"
                     "The program might be slower as a result.")

    @staticmethod
    def log_google_auth_error():
        """Logs when a google authentication error occurs in tts.py."""
        Logger.__console_message()
        logger.error("Something went wrong while authenticating with Google, using local TTS.\n"
                     "See the Traceback for details.")

    @staticmethod
    def log_gTTS_value_error(ve):
        """Logs when a ValueError occurs in tts.py when using gTTS."""
        Logger.__console_message()
        logger.exception("Something went wrong in gTTS: %s" % str(ve),
                         "\nThis error seems to occur every once in a while.")

    @staticmethod
    def log_open_file_error():
        """Logs when a ValueError occurs when opening a file in tts.py when using paid services."""
        Logger.__console_message()
        logger.error("Error creating a temporary audiofile for TTS, unable to play sound.\n"
                     "It is possible you don't have writing or reading rights in this directory.\n"
                     "See the Traceback for details.")

    @staticmethod
    def log_write_file_error(ioe):
        """Logs when an IOError occurs when writing to a file in key.py when saving text."""
        Logger.__console_message()
        logger.error("Error writing text to file.\n"
                     + str(ioe))

    @staticmethod
    def log_download_frequency_list_error():
        """Logs when a ConnectionError occurs when trying to download a frequency list."""
        Logger.__console_message()
        logger.exception("Something went wrong while trying to download a frequency list.\n"
                         "See the Traceback for details.")

    @staticmethod
    def log_unknown_icon(file):
        Logger.__console_message()
        logger.exception("The specified icon '{}' could not be found.\n".format(file) +
                         "Using the text description instead.\n"
                         "See the Traceback for details.")

    @staticmethod
    def log_no_tts_input():
        Logger.__console_message()
        logger.exception("The text-to-speech service cannot be used without any input text.\n"
                         "See the Traceback for details.")


class Handler:

    @staticmethod
    def mockup_keyboard():
        """Returns a mockup keyboard so the program can continue running."""
        print("Loading in temporary keyboard")
        return [[["BASIC_KEY", "E"], ["BASIC_KEY", "R"], ["BASIC_KEY", "R"], ["BASIC_KEY", "O"], ["BASIC_KEY", "R"]]]
