"""This module is used to convert text into speech.

The module contains a single class ``TextToSpeech``.

**Usage example**::

        import text_to_speech
        tts = keyboard.text_to_speech.TextToSpeech(connection_timeout=0.3, lang_tag='en', service='free_online')
        tts.speak("testing testing 123")

``TextToSpeech`` supports 3 different types of TTS services:

**Free offline (pyttsx3)**
 * `+` Very low latency
 * `+` Does not require internet connection
 * `-` Language support depends on OS, generally only English
 * `-` Bad voice quality

**Free online (gTTS)**
 * `+` Decent audio quality
 * `+` Supports almost all languages
 * `-` Requires internet connection
 * `-` Potential instability & high latency; is based off of undocumented & unofficial Google Translate api

**Paid online (Google Cloud TTS)**
 * `+` Stable and relatively low latency (~200ms)
 * `+` Best voice quality available on the market
 * `+` Supports almost all languages
 * `-` Requires internet connection
 * `-` Costs money after first 1 million characters/month.
 * `-` Requires account & credentials.

Follow `these instructions <https://cloud.google.com/text-to-speech/docs/quickstart-protocol>`_ to get the required Google Cloud TTS credentials.
The JSON file containing the credentials should be named ``TTS-gcloud-credentials.json`` and placed in the same folder as this module.

**Language tags**

The language the TTS service should use is indicated with an IETF language tag.
A list of these tags can be found `here. <https://gtts.readthedocs.io/en/latest/_modules/gtts/lang.html>`_

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

import pyttsx3
from gtts import gTTS
import gtts.lang
from playsound import playsound
import os
import requests
from google.cloud import texttospeech
import google.auth.exceptions
from mindaffectBCI.examples.presentation.smart_keyboard.app_exceptions import Logger
import threading

from mindaffectBCI.examples.presentation.smart_keyboard.settings_manager import SettingsManager


class TextToSpeech(object):
    """Implements different TTS services to turn text into speech.

    The argument ``service`` determines which TTS provider is used.
    If the service fails, the offline provider is used as a backup.

    Args:
        connection_timeout (float, optional): slowest allowed connection speed in seconds.
        service (str, optional): which TTS service to prefer. options: ``paid_online``, ``free_online``, ``offline``
    """
    def __init__(self, connection_timeout=0.3, service='free_online'):
        self.name_to_function = {
            'paid_online': self.say_paid_online,
            'free_online': self.say_free_online,
            'offline': self.say_offline
        }
        self.settings_manager = SettingsManager.get_instance()
        lang_tag = self.settings_manager.get_language(lang_tag=True)

        if service not in self.name_to_function:
            print("Invalid TTS service name, defaulting to free_online")
            self.service = 'free_online'
        else:
            self.service = service

        if lang_tag not in gtts.lang.tts_langs():
            print(lang_tag, " is an invalid TTS language tag, defaulting to 'en'")
            self.lang_tag = 'en'
        else:
            self.lang_tag = lang_tag

        self.tsx3engine = pyttsx3.init()
        self.connection_timeout = connection_timeout
        # attaches this class to settings_manager
        self.settings_manager.attach(self)
        self.basedirectory = os.path.dirname(os.path.abspath(__file__))

        if service == 'paid_online':
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'TTS-gcloud-credentials.json'
            try:
                self.client = texttospeech.TextToSpeechClient()
                self.audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
                self.voice = texttospeech.VoiceSelectionParams(language_code=self.lang_tag, ssml_gender=texttospeech.SsmlVoiceGender.MALE)
            except google.auth.exceptions.GoogleAuthError as err:
                Logger.log_google_auth_error()
                self.service = 'offline'

    def update(self):
        """Updates the language at use."""

        lang_tag = self.settings_manager.get_language(lang_tag=True)
        if lang_tag not in gtts.lang.tts_langs():
            print(lang_tag, " is an invalid TTS language tag, defaulting to 'en'")
            self.lang_tag = 'en'
        else:
            self.lang_tag = lang_tag

    def determine_service(self):
        """
        Determines which TTS service is used, bases on preference and whether or not an internet connection is
        available or not.
        """
        # First check if preference is local to prevent overuse of slow is_online()
        if self.service == 'offline':
            return 'offline'
        elif self.is_online():
            return self.service
        else:
            return 'offline'

    def speak_threaded(self, message):
        """Calls the `speak` function with a seperate thread.

        Args:
            message (str): The message.
        """
        thread = threading.Thread(target=self.speak, args=(message, ), daemon=True)
        thread.start()

    def speak(self, message):
        """Decides on which TTS service to use, which language to speak, and plays the audio.

        Args:
            message (str): The message.
        """
        service = self.determine_service()
        # TODO[]: fall-back service if this fails!
        offline_fallback = True
        try:
            self.name_to_function[service](message)     # Uses a dictionary to determine which function to use
            offline_fallback = False
        except ValueError as ve:
            # This error occurs every once in a while, seemingly randomly.
            Logger.log_gTTS_value_error(ve)
        except PermissionError as e:
            pass
        if offline_fallback:
            print("Using OFFLINE Fallback!")
            self.say_offline(message)

    def say_paid_online(self, message):
        """Uses Google Cloud WaveNet voice to speak the message.

        Args:
            message (str): The message.
        """
        synthesis_input = texttospeech.SynthesisInput(text=message)

        response = self.client.synthesize_speech(input=synthesis_input, voice=self.voice, audio_config=self.audio_config)
        try:
            ttsfile = os.path.join(self.basedirectory,'ttsmessage.mp3')
            with open(ttsfile, "wb") as out:
                # Write the response to the output file.
                out.write(response.audio_content)
            playsound(ttsfile)
            os.remove(ttsfile)
        except ValueError as ve:
            Logger.log_open_file_error()

    def say_free_online(self, message):
        """Uses ``gTTS`` to speak a given message.

        Args:
            message (str): The message.
        """
        try:
            tts = gTTS(message, lang=self.lang_tag)
            ttsfile = os.path.join(self.basedirectory,'ttsmessage.mp3')
            tts.save(ttsfile)
            playsound(ttsfile)
            os.remove(ttsfile)
        except AssertionError as ae:
            # This happens when a TTS request is made without any input.
            Logger.log_no_tts_input()

    def say_offline(self, message):
        """Uses ``pyttsx3``  to speak a given message.

        Args:
            message (str): The message.
        """
        try:
            self.tsx3engine.say(message)
            self.tsx3engine.runAndWait()
        except RuntimeError as re:
            # this happens when the user presses the TTS button again while the audio is still running
            pass

    def is_online(self):
        """Checks if there is an internet connection available with sufficiently low ping.

            Returns:
                bool: ``True`` if successful, ``False`` otherwise.
        """

        url = "https://translate.google.com"
        try:
            request = requests.get(url, timeout=self.connection_timeout)
            print("TTS: Connected to the Internet")
            return True
        except (requests.ConnectionError, requests.Timeout) as exception:
            print("TTS: No internet connection.")
            return False
