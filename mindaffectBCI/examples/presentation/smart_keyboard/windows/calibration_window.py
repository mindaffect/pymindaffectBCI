"""
This module contains a single class representing a calibration display.

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
from mindaffectBCI.examples.presentation.smart_keyboard.windows.keyboard_window import KeypadWindow
from mindaffectBCI.examples.presentation.smart_keyboard.keyboard_loader import KeyboardLoader


class CalibrationWindow(Window):
    """
    Subclass of Window, designed for setting up the calibration for EEG caps.
    Consists of a number of phases which can be traversed by any key on the physical keyboard:
    *  instruction:  Shows a small instructional text.
    *  trials: The actual calibration trials happen during this phase.
    *  waiting: Displays a message while waiting for the calibration results, and eventually shows those results.
    *  error: When flickering is off, calibration is not possible and tis phase will be active.
    *  abort: when interrupting the calibration this phase will become active, instructing the user how to return to the settings.
    *  end: Will immediately return to the MenuWindow.

    Args:
        parent (windows.window.Window): The parent of this Window.
        facade (framework_facade.FrameworkFacade): Contains the GUI-specific functionality.
        style (dict): A dictionary containing configurable style settings.
        calibration_config (dict): Contains the calibration configurations.
        use_flickering (bool): A boolean indicating whether or not to use flickering.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.

    Attributes:
        parent (windows.window.Window): The parent of this Window.
        facade (framework_facade.FrameworkFacade): Contains the GUI-specific functionality.
        style (dict): A dictionary containing configurable style settings.
        use_flickering (bool): A boolean indicating whether or not to use flickering.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.

        calibration_config (dict): Contains the calibration configurations.
        use_optometer (bool): Indicates whether to use the optometer.
        state2color (dict): Mapping from states to RGB values.
        opto (Object): Framework-dependent rectangle object for the optometer square.
        instructions (dict): Mapping of phase names to instruction texts.
        windows (dict): Dictionary holding one KeypadWindow object containing all calibration keys.
        phase_links (dict): Mapping from phases to their next phase.
        current_phase (str): Name of the current phase.
        instruction (Object): Framework dependent text object which holds the instruction.
        pred (Object): Holds calibration results.
        active_window (str): Name of the current active Window.
    """

    def __init__(self, parent, facade, style, calibration_config, use_flickering, noisetag=None):
        super().__init__(
            parent=parent,
            facade=facade,
            style=style,
            use_flickering=use_flickering,
            noisetag=noisetag
        )

        # CalibrationWindow-specific attributes:
        self.calibration_config = calibration_config
        self.use_optometer = calibration_config["use_optometer"]
        self.state2color = style["state2color"]

        # Construct optosensor square (in upper-left corner of the screen):
        self.opto = facade.create_rect(
            size=(0.1, 0.1),
            pos=(0.05, 0.95),
            color=self.style["button_color"],
            line_color=self.style["line_color"]
        )

        # Dictionary of instruction messages:
        self.instructions = {
            'instruction': "Calibration\n\nPress any key on your real keyboard to start the calibration",
            'trials': 'Calibration\n\nlook at the indicated green target key',
            'waiting': 'Calibration\n\nWaiting for performance results from decoder\n\nPlease wait',
            'results': 'Calibration\n\nCalibration Performance: %3.0f%% Correct\n\n'
                       'Press any key on your real keyboard to return to the menu',
            'abort': 'Calibration\n\nCalibration was aborted.\n'
                     'Press any key on your real keyboard to return to the menu',
            'error': 'Calibration\n\nThe keyboard is not using flickering\n'
                     'Enable flickering in the config\n\n'
                     'Press any key on your real keyboard to return to the menu'
        }

        # The keypad used for calibration of the EEG:
        self.windows = {
            "Numbers": KeypadWindow(
                parent=self,
                facade=facade,
                style=style,
                keypad_layout=KeyboardLoader.load_json(calibration_config["file"]),
                use_flickering=use_flickering,
                text_field=None,
                noisetag=noisetag
            )
        }

        # For handling the different phases of calibration:
        self.phase_links = {
            'instruction': 'trials',
            'trials': 'waiting',
            'waiting': 'end',
            'pre_abort': 'abort',
            'abort': 'end',
            'error': 'end'
        }

        self.current_phase = 'instruction'
        if not self.use_flickering:
            self.current_phase = 'error'

        # Instruction text object:
        self.instruction = self.facade.create_text(
            text=self.instructions[self.current_phase],
            pos=(0.5, 0.95),
            text_size=self.style['text_size'],
            col=self.style['text_color'],
            align_hor='center',
            align_vert='top'
        )
        
        self.logo = self.facade.create_icon(
            "key_icons\MindAffect_Logo.png",
            label_col=self.style.get("MA_orange",(128,128,0)),
            size=(.1,.2),
            pos=(.95,.9),
        )
        # Last calibration results:
        self.pred = None
        self.active_window = "Numbers"

    def activate(self):
        """Activates all visual and functional elements of this Window."""
        self.facade.toggle_text_render(self.instruction, True)
        self.facade.toggle_image_render(self.logo, True)
        
        if self.use_optometer:
            self.facade.toggle_shape_render(self.opto, True)

        self.windows[self.active_window].activate()

        # Clear buffer of pressed keys:
        self.facade.key_event()

    def deactivate(self):
        """Deactivates all visual and functional elements of this Window."""
        self.facade.toggle_image_render(self.logo, False)
        self.facade.toggle_text_render(self.instruction, False)
        self.facade.toggle_shape_render(self.opto, False)
        self.windows[self.active_window].deactivate()

    def switch_window(self, window_name):
        """Switches to Window specified by window_name."""
        if self.windows:
            if window_name in self.windows:
                self.windows[self.active_window].deactivate()
                self.windows[window_name].activate()
                self.active_window = window_name
                self.switched_windows = True
        else:
            self.parent.switch_window(window_name)
            self.switched_windows = True

    def draw(self, noisetag, last_flip_time, target_idx=-1):
        """
        Changes the state of the buttons/keys within the Window.
        It draws the display with the colors given by the Noisetag module if flickering is active.

        Args:
            noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
            last_flip_time (int): Timestamp of last screen update, i.e. buffer flip.
            target_idx (int): (Optional) index of the target stimulus.
        """
        if self.facade.key_event():
            if self.current_phase == "trials":
                self.current_phase = "pre_abort"
                self.noisetag.modeChange("idle")
            self.next_phase()

        if self.current_phase == "trials":
            if self.use_flickering:
                self.draw_flicker_frame(noisetag, last_flip_time)

        if self.current_phase == 'waiting':
            pred = self.noisetag.getLastPrediction()

            if pred is not None and (self.pred is None or pred.timestamp > self.pred.timestamp):
                self.pred = pred
                # Set results text:
                self.facade.set_text(self.instruction, self.instructions['results'] % ((1.0 - self.pred.Perr) * 100.0))

    # CalibrationWindow-specific functions:
    def draw_flicker_frame(self, noisetag, last_flip_time):
        """
        Handles the rendering of the flickering when in calibration phase.

        Args:
            noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
            last_flip_time (int): Timestamp of last screen update, i.e. buffer flip.
        """
        # Send info of the previous stimulus state, with the recorded vsync time (if available):
        flip_time = last_flip_time if last_flip_time is not None else noisetag.getTimeStamp()
        noisetag.sendStimulusState(timestamp=flip_time)

        # Update and get the new stimulus state to display:
        try:
            noisetag.updateStimulusState()
            stimulus_state, target_idx, objIDs, sendEvents = noisetag.getStimulusState()
            target_state = stimulus_state[target_idx] if target_idx >= 0 else -1
        except StopIteration:
            # Get prediction from Noisetag at the end of the calibration:
            self.pred = self.noisetag.getLastPrediction()
            # Move to results phase:
            self.next_phase()
            return

        # Draw the display with the instructed colors:
        if stimulus_state:
            flicker_key = -1
            if self.calibration_config["only_flicker_cued"]:
                flicker_key = target_idx
            self.windows[self.active_window].draw(0, stimulus_state, flicker_key)
        if target_state is not None and target_state in (0, 1):
            # Draw optometer square with the colors of the target stimulus state:
            self.facade.change_button_color(self.opto, self.style["state2color"][
                str(target_state)])

    def next_phase(self):
        """Handles the phase switching of the calibration."""
        self.current_phase = self.phase_links[self.current_phase]
        if self.current_phase == "end":
            if self.use_flickering:
                self.current_phase = "instruction"
            else:
                self.current_phase = "error"
            self.parent.switch_window("Menu")

        if self.current_phase == "trials":
            self.start_trials()

        # Set the appropriate instruction:
        self.facade.set_text(self.instruction, self.instructions[self.current_phase])

    def start_trials(self):
        """Starts a series of calibration trials."""
        n_keys = len(self.windows[self.active_window].get_keys())
        self.noisetag.setnumActiveObjIDs(n_keys)
        self.noisetag.setActiveObjIDs(list(range(1, n_keys + 1)))
        self.noisetag.startCalibration(
            nTrials=self.calibration_config["trials"],
            waitduration=1,
            duration=self.calibration_config["trial_duration"],
            framesperbit=self.style["framesperbit"]
        )
