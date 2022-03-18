"""
This module contains a single class displaying cued prediction.

It contains the following visual and functional elements:
 * keypad window consisting of number 1-9
 * textfield displaying instructions
 * optometer square
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
from mindaffectBCI.examples.presentation.smart_keyboard.windows.keypad_window import KeypadWindow
from mindaffectBCI.examples.presentation.smart_keyboard.keyboard_loader import KeyboardLoader
from mindaffectBCI.examples.presentation.smart_keyboard.text_field import TextField


class CuedPredictionWindow(Window):
    """
    Subclass of Window, designed for testing the calibration for EEG caps.
    Consists of 3 phases which can be traversed by any key on the physical keyboard:
    *  instruction:  Shows a small instructional text
    *  trials: The actual cued prediction trials happen during this phase
    *  end: Will immediately return to the menu window

    Args:
        parent (windows.window.Window): The parent of this window.
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        style (dict): Style instructions for the keyboard.
        cued_prediction_config (dict): Contains the calibration configurations.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.

    Attributes:
        parent (windows.window.Window): The parent of this window.
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        style (dict): Style instructions for the keyboard.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
        cued_prediction_config (dict): Contains the cued prediction configurations.
        use_optometer (bool): Indicates whether to use the optometer.
        use_free_typing (bool): Indicates whether to use freetyping in the Noisetag, for testing
        state2color (dict): Mapping from states to RGB values.
        windows (dict): dictionary holding one KeypadWindow object containing all calibration keys.
        opto (Object): Framework dependent object for the optometer square.
        text_field (text_field.TextField): The TextField to display the instructions.
        instructions (dict): Mapping of phase names to instruction texts
        current_phase (str): Name of the current phase.
        phase_links (dict): Mapping from phases to their next phase.
        instruction (Object): Framework dependent text object which holds the instruction.
        active_window (str): Name of the current active Window.
        switched_windows (bool): Indicates whether windows have been switched.
        is_active (bool): Regulatory boolean to indicate if a window is active or not.
    """
    def __init__(self, parent, facade, style, cued_prediction_config, use_flickering, noisetag=None):
        super().__init__(
            parent=parent,
            facade=facade,
            style=style,
            use_flickering=use_flickering,
            noisetag=noisetag
        )

        self.cued_prediction_config = cued_prediction_config
        self.use_optometer = cued_prediction_config["use_optometer"]
        self.use_free_typing = cued_prediction_config["use_free_typing"]
        self.state2color = style["state2color"]

        # Prediction feedback attributes:
        self.show_live_feedback = cued_prediction_config['show_live_feedback']
        self.feedback_threshold = cued_prediction_config['feedback_threshold']

        self.windows = {
            "Numbers": KeypadWindow(
                parent=self,
                facade=facade,
                style=style,
                keypad_layout=KeyboardLoader.load_json(cued_prediction_config["file"]),
                use_flickering=use_flickering,
                text_field=None,
                noisetag=noisetag
            )
        }
        self.logo = self.facade.create_icon(
            "key_icons\MindAffect_Logo.png",
            label_col=self.style.get("MA_orange",(128,128,0)),
            size=(.1,.1),
            pos=(.95,.95),
        )        

        # Opto sensor square (upper-left corner of the screen):
        self.opto = facade.create_rect(
            size=(0.1, 0.1),
            pos=(0.05, .95),
            color=self.style["button_color"],
            line_color=self.style["line_color"]
        )

        self.text_field = TextField(
            facade=self.facade,
            style_sheet=self.style,
            update_suggestions=self.update_suggestions,
        )

        self.instructions = {
            "instruction": "Cued Prediction\n\nPress any key to start 10 cued trials",
            "trials": "Cued Prediction\n\nLook at the cued key",
            "error": "Cued Prediction\n\nFlickering is not enabled\nEnable flickering in the configuration"
                     "\n\nPress any key to return to the menu"
        }

        self.current_phase = "instruction"
        if not self.use_flickering:
            self.current_phase = "error"

        self.phase_links = {
            "instruction": "trials",
            "trials": "end",
            "error": "end"
        }

        self.instruction = self.facade.create_text(
            text=self.instructions[self.current_phase],
            pos=(0.5, 0.95),
            text_size=self.style['text_size'],
            col=self.style['text_color'],
            align_hor='center',
            align_vert='top'
        )

        self.active_window = "Numbers"
        self.switched_windows = False
        self.is_active = False

    def update_suggestions(self):
        """
        Dummy function that gets passed to the textfield.

        This function has no functionality as there are no suggestion keys for Cued Prediction
        """
        pass

    def next_phase(self):
        """Handles the phase changing of cued prediction"""
        self.current_phase = self.phase_links[self.current_phase]

        if self.current_phase == "end":
            self.noisetag.modeChange("idle")
            # Reset the phase for next time cued prediction is required:
            if self.use_flickering:
                self.current_phase = "instruction"
            else:
                self.current_phase = "error"

            # Return to the menu:
            self.parent.switch_window("Menu")

        if self.current_phase == "trials":
            self.start_trials()

        # Set the appropriate instruction:
        self.facade.set_text(self.instruction, self.instructions[self.current_phase])

    def start_trials(self):
        """Starts a series of cued prediction trials."""
        n_keys = len(self.windows[self.active_window].get_keys())
        self.noisetag.setnumActiveObjIDs(n_keys)

        if self.use_free_typing:
            # when freetyping is on from keyboard_config, then use this
            self.noisetag.startPrediction(
                cuedprediction=False,
                nTrials=1,
                waitduration=1,
                duration=self.cued_prediction_config["trial_duration"],
                framesperbit=self.style["framesperbit"]
            )
        else:
            # if free typing, exists for testing purposes isn't set to true in the config, use normal prediction
            self.noisetag.startPrediction(
                cuedprediction=True,
                nTrials=self.cued_prediction_config["trials"],
                waitduration=1,
                duration=self.cued_prediction_config["trial_duration"],
                framesperbit=self.style["framesperbit"]
            )

    def activate(self):
        """Activates all visual and functional elements of this Window."""
        self.text_field.activate()
        self.text_field.clear_text_field()
        self.facade.toggle_text_render(self.instruction, True)
        self.facade.toggle_image_render(self.logo, True)

        if self.use_optometer:
            self.facade.toggle_shape_render(self.opto, True)
        self.windows[self.active_window].activate()

        # Clear buffer of pressed keys:
        self.facade.key_event()
        self.is_active = True

    def deactivate(self):
        """Deactivates all visual and functional elements of this Window."""
        self.is_active = False
        self.text_field.deactivate()
        self.facade.toggle_image_render(self.logo, False)        
        self.facade.toggle_text_render(self.instruction, False)
        self.facade.toggle_shape_render(self.opto, False)
        self.windows[self.active_window].deactivate()

    def handle_mouse_events(self):
        """No functionality in this class"""
        pass

    def select_key(self, objID):
        """
        Selects and activates the key (as if a mouseclick) if selected using prediction trials from the
        Noisetag.

        Args:
            objID: (int) ID of the object that is selected
        """

        if self.is_active:
            # Convert the objID to the corresponding key:
            key = self.windows[self.active_window].get_keys()[objID - 1]

            # Apply the functionality of that key:
            key.key_type.func_provider.apply(key, self)

    def draw(self, noisetag, last_flip_time, target_idx=-1):
        """
        Decides per frame if a phase change should occur and when to render flickering states.

        Args:
            noisetag (Noisetag): Reference to the noisetag object
            last_flip_time (int): time of the last flip
            target_idx (int): (Optional) index of the target stimulus.
        """
        if self.facade.key_event():
            self.next_phase()
        if self.current_phase == "trials":
            if self.use_flickering:
                self.draw_flicker_frame(noisetag, last_flip_time)

    def draw_flicker_frame(self, noisetag, last_flip_time):
        """
        Handles the rendering of the flickering when in trial phase.

        Args:
            noisetag: Noisetag object which handles flicker states and communication with EEG
            last_flip_time: Timestamp of the last window flip
        """
        # Send info on the *previous* stimulus state, with the recorded vsync time (if available):
        flip_time = last_flip_time if last_flip_time is not None else noisetag.getTimeStamp()
        noisetag.sendStimulusState(timestamp=flip_time)

        # Update and get the new stimulus state to display:
        try:
            noisetag.updateStimulusState()
            stimulus_state, target_idx, objIDs, sendEvents = noisetag.getStimulusState()
            target_state = stimulus_state[target_idx] if target_idx >= 0 else -1
        except StopIteration:
            if self.use_free_typing:
                self.start_trials()
            else:
                # Go to results phase:
                self.next_phase()
            return

        # Draw the buttons with the right stimulus state colors:
        if stimulus_state:
            self.windows[self.active_window].draw(0, stimulus_state)
            prediction_message = self.noisetag.getLastPrediction()
            # give prediction feedback to the user:
            if prediction_message:
                self.feedback(self.feedback_threshold)

        if target_state is not None and target_state in (0, 1):
            # Draw optometer square with the colors of the target stimulus state:
            self.facade.change_button_color(self.opto, self.style["state2color"][
                str(target_state)])
