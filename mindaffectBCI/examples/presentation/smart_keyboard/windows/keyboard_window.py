"""
This module contains a single class representing a keyboard.

It contains the following visual and functional elements:
 * three keypad windows (upper, lower, symbols)
 * suggestion keys
 * textfield
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

from mindaffectBCI.examples.presentation.smart_keyboard.settings_manager import SettingsManager
from mindaffectBCI.examples.presentation.smart_keyboard.windows.window import Window
from mindaffectBCI.examples.presentation.smart_keyboard.windows.keypad_window import KeypadWindow
from mindaffectBCI.examples.presentation.smart_keyboard.key import Key
from mindaffectBCI.examples.presentation.smart_keyboard.key_type import KeyType
from mindaffectBCI.examples.presentation.smart_keyboard.word_prediction import WordPrediction
from mindaffectBCI.examples.presentation.smart_keyboard.word_correction import WordCorrection
from mindaffectBCI.examples.presentation.smart_keyboard.text_to_speech import TextToSpeech
from mindaffectBCI.examples.presentation.smart_keyboard.text_field import TextField
from mindaffectBCI.examples.presentation.smart_keyboard.keyboard_loader import KeyboardLoader
import re as regex
import _thread as thread
import sys


class KeyboardWindow(Window):
    """
    A Window representing a keyboard.

    Args:
        parent (windows.window.Window): The parent of this window.
        facade (framework_facade.FrameworkFacade): The GUI-specific functionality.
        style (dict): Style configurations for objects contained by a window.
        use_flickering (bool): A boolean indicating whether or not to use flickering in the window.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
        ai_settings (dict): Instructions on whether or not to use AI modules.
        config (dict): Configurations (to be replaced).
    """

    def __init__(self, parent, facade, style, use_flickering, noisetag, ai_settings, config):
        super().__init__(
            parent=parent,
            facade=facade,
            style=style,
            use_flickering=use_flickering,
            noisetag=noisetag
        )

        self.keys = []

        # Keyboard layout:
        self.key_padding = style["key_padding"]
        self.keyboard_size = style["keyboard_size"]
        self.keypad_size = style["keypad_size"]

        # Setup AI modules:
        self.load_ai = ai_settings["load_ai"]

        self.config = config
        keypad_layouts = config["keypads"]
        self.feedback_threshold = self.config["feedback_threshold"]
        self.predictor = None
        self.state2color = style["state2color"]

        if self.load_ai:
            # Initialize text-to-speech engine:
            self.tts = TextToSpeech(
                service=config['text_to_speech']["tts_service"]
            )

            # Initialize word prediction module:
            self.predictor = WordPrediction.get_instance(ai_settings["ngram_depth"])

            # Initialize word correction module:
            self.correction = WordCorrection.get_instance(ai_settings["correction_distance"])

            # Generate suggestion keys:
            self.keys = self.build_suggestion_keys()

        self.text_field = TextField(
            facade=self.facade,
            style_sheet=self.style,
            update_suggestions=self.update_suggestions,
            predictor=self.predictor
        )

        # Initialize KeypadWindows:
        self.windows = {
            "Upper": KeypadWindow(
                parent=self,
                facade=facade,
                style=style,
                keypad_layout=KeyboardLoader.load_json(keypad_layouts["Upper"]),
                use_flickering=use_flickering,
                text_field=self.text_field,
                noisetag=noisetag
            ),

            "Lower": KeypadWindow(
                parent=self,
                facade=facade,
                style=style,
                keypad_layout=KeyboardLoader.load_json(keypad_layouts["Lower"]),
                use_flickering=use_flickering,
                text_field=self.text_field,
                noisetag=noisetag
            ),

            "Symbols": KeypadWindow(
                parent=self,
                facade=facade,
                style=style,
                keypad_layout=KeyboardLoader.load_json(keypad_layouts["Symbols"]),
                use_flickering=use_flickering,
                text_field=self.text_field,
                noisetag=noisetag
            ),

            "AdditionalSymbols": KeypadWindow(
                parent=self,
                facade=facade,
                style=style,
                keypad_layout=KeyboardLoader.load_json(keypad_layouts["AdditionalSymbols"]),
                use_flickering=use_flickering,
                text_field=self.text_field,
                noisetag=noisetag
            )
        }
        
        #MA logo
        self.logo = self.facade.create_icon(
            "key_icons\MindAffect_Logo.png",
            label_col=self.style.get("MA_orange",(128,128,0)), 
            size=(.1,.1),
            pos=(.95,.95),
        )
        
        # Construct optosensor square (in upper-left corner of the screen):
        self.opto = facade.create_rect(
            size=(0.1, 0.1),
            pos=(0.05, 0.95),
            color=self.style["button_color"],
            line_color=self.style["line_color"]
        )

        self.active_window = "Lower"
        self.switched_windows = False
        self.is_active = False
        self.active_trial = False

    def get_keys(self):
        """Returns the keys of this Window. """
        return self.keys

    def activate(self):
        """Activates all visual and functional elements of this Window."""
        self.text_field.activate()
        self.facade.toggle_image_render(self.logo, True)
        if self.keys:
            for key in self.keys:
                key.toggle_render(True)
        self.windows[self.active_window].activate()
        if self.use_flickering:
            self.noisetag.setnumActiveObjIDs(len(self.get_keys()) +
                                             len(self.windows[self.active_window].get_keys()))
            self.start_new_trial()

        # renders optometer square when turned on specifically, or when cuing is done
        if self.config["use_cued"]:
            self.facade.toggle_shape_render(self.opto, True)
        self.is_active = True

    def deactivate(self):
        """Deactivates all visual and functional elements of this Window."""
        self.is_active = False
        self.text_field.deactivate()
        self.facade.toggle_image_render(self.logo, False)
        self.facade.toggle_shape_render(self.opto, False)
        for key in self.keys:
            key.toggle_render(False)
        self.windows[self.active_window].deactivate()

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

                # When switching keypads, reset_flicker() is called to update the number of objectIDs in the Noisetag
                # to match the new key count:
                self.reset_flicker()
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

    def draw(self, noisetag, last_flip_time, target_idx=-1):
        """
        Changes the state of the buttons/keys within the Window.
        It draws the display with the colors given by the Noisetag module if flickering is active.

        Args:
            noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
            last_flip_time (int): Timestamp of last screen update, i.e. buffer flip.
            target_idx (int): (Optional) index of the target stimulus.
        """
        if self.use_flickering:
            # Send info of the previous stimulus state, with the recorded vsync time (if available):
            flip_time = last_flip_time if last_flip_time is not None else noisetag.getTimeStamp()
            noisetag.sendStimulusState(timestamp=flip_time)

            # Update and get the new stimulus state to display:
            try:
                noisetag.updateStimulusState()
                stimulus_state, target_idx, obj_ids, send_vents = noisetag.getStimulusState()
                target_state = stimulus_state[target_idx] if target_idx >= 0 else -1
            except StopIteration:
                # Start a new trial each time the last one ends in order to be able to keep typing:
                self.start_new_trial()
                return

            # Draw the display with the instructed colors:
            if stimulus_state:
                for i, key in enumerate(self.keys):
                    key.change_color(self.state2color[str(stimulus_state[i])])

                # Pass the stimulus states to the keypad for drawing the flickering:
                self.windows[self.active_window].draw(len(self.keys), stimulus_state)

                # Handle prediction feedback:
                self.feedback(self.feedback_threshold)

            if self.config["use_cued"]:
                if target_state is not None and target_state in (0, 1):
                    # Draw optometer square with the colors of the target stimulus state:
                    self.facade.change_button_color(self.opto, self.style["state2color"][
                        str(target_state)])

    def select_key(self, obj_id):
        """
        Selects and activates the key (like a mouseclick) if selected using prediction trials from the Noisetag.

        Args:
            obj_id (int): ID of the object that is selected.
        """
        if self.is_active:
            # Suggestion keys:
            if obj_id <= len(self.keys):
                # Convert objID to the corresponding key. 1 is subtracted from the objID since the objIDs start at 1:
                key = self.keys[obj_id - 1]

                # Apply the functionality of that key:
                key.key_type.func_provider.apply(key, self)

            # Keys on the keypad:
            else:
                # Convert the objID to the corresponding key:
                keys = self.windows[self.active_window].get_keys()
                key = keys[obj_id - len(self.keys) - 1]

                if key.key_type != KeyType.DEFAULT_SETTINGS_KEY:
                    # apply the functionality of that key, as long as it's not the settings key:
                    # this is to prevent the cued prediction from switching to the settings menu when actually selecting
                    # this would abort cued prediction immediately obviously
                    key.key_type.func_provider.apply(key, self)

    def reset_flicker(self):
        """Resets the number of stimuli for the Noisetag when changing keypads."""
        if self.use_flickering:
            self.noisetag.setnumActiveObjIDs(len(self.keys) + len(self.windows[self.active_window].get_keys()))
            self.start_new_trial()

    # KeyboardWindow-specific functions:
    def reset_key_colors(self):
        """Resets all the keys to their original color"""
        for key in self.keys:
            key.reset_color()
        for key in self.windows[self.active_window].get_keys():
            key.reset_color()

        # Manually triggers a window flip so the default colors show up before the key function is executed:
        self.facade.flip()

    def start_new_trial(self):
        """Starts a new EEG trial."""
        # if cuing is on, we use a cued trial otherwise normal single trials for typing
        self.noisetag.startPrediction(
                cuedprediction=self.config["use_cued"],
                nTrials=1,
                waitduration=1,
                duration=self.config["trial_duration"],
                framesperbit=self.style["framesperbit"]
            )
        # to clear feedback of the last trial:
        self.feedback(self.feedback_threshold)

    def text_to_speech(self):
        """Converts the text in the text field to speech and reads it out loud."""
        if self.load_ai:
            if not self.text_field.is_empty():
                self.tts.speak_threaded(self.text_field.get_state())

    def update_suggestions(self):
        """Updates the suggestions of the keyboard; correction, autocompletion, and next word prediction."""
        if self.load_ai:
            settings_manager = SettingsManager.get_instance()
            if settings_manager.get_word_prediction_correction():
                correct = False  # Regulatory boolean

                # Split sentences with ., !, and ? as delimiters:
                sentences = regex.split(r"[.!?]", self.text_field.get_state())

                # Check if anything is typed in last sentence:
                if len(sentences[-1]) > 0 and regex.match(r".*[a-zA-Z].*", sentences[-1]):
                    # Only base prediction and autocompletion on last sentence:
                    sentence = sentences[-1]
                    if regex.match(r"\s", sentence[-1]):  # Predictive text
                        # Get suggestion_keys:
                        suggestions = self.predictor.predict(sentence, len(self.keys))
                    else:  # Autocompletion and correction
                        # Get autocomplete suggestions:
                        suggestions = self.predictor.autocomplete(sentence, len(self.keys))
                        # Find possible correction for last word:
                        correct = self.correction.correct(regex.split(' ', sentence)[-1])
                else:
                    # Standard suggestions for beginning of sentence:
                    suggestions = self.predictor.predict("", len(self.keys))

                offset = -1 if correct else 0
                for index, prediction in enumerate(self.keys):
                    if index == 0 and correct:
                        # If there is a correction, suggest it. If not, give prediction instead:
                        prediction.set_label_text(correct)
                    else:
                        # Fill in the other prediction labels:
                        prediction.set_label_text(suggestions[index + offset])
            else:
                for index, prediction in enumerate(self.keys):
                    # set labels to blank if module is disabled
                    prediction.set_label_text("")

    def build_suggestion_keys(self):
        """
        Builds keys for prediction suggestions.

        Returns:
            keys (list): A list of keys.
        """
        n_pred = self.style["n_prediction"]

        key_width = 1 / n_pred
        key_height = self.keyboard_size - self.keypad_size

        size = (key_width - self.key_padding, key_height - self.key_padding)

        keys = []
        for i in range(n_pred):

            x_pos = (i + 0.5) * key_width
            y_pos = self.keyboard_size - 0.5 * key_height
            pos = (x_pos, y_pos)

            keys.append(Key(self.facade, KeyType.SUGGESTION_KEY, size, pos, "", self.button_color,
                            self.line_color, self.label_color))

        return keys
