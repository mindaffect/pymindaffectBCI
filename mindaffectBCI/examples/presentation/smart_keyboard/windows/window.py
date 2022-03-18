"""
This module contains a single super class representing a window.

A Window must have:
 *  ``activate()``              : Method to activate objects to be rendered.
 *  ``deactivate()``            : Method to deactivate objects to not be rendered.
 *  ``switch_window()``         : Method to switch from it to another instance of a window.
 *  ``handle_mouse_events()``   : Method to handle process selection of its objects.
 *  ``draw()``                  : Method to change the state of its objects (Noisetag).
 *  ``select_key()``            : Method to activate the key when using prediction trials instead of a mouseclick.
 *  ``reset_flicker()``         : Method to reset the flickering.
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


class Window:
    """
    Superclass to represent a Window.

    Args:
        parent (windows.window.Window): The parent of a window if existing.
        facade (framework_facade.FrameworkFacade): The GUI-specific functionality.
        style (dict): Style configurations for objects contained by a window.
        use_flickering (bool): A boolean indicating whether or not to use flickering in the Window.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
    """

    def __init__(self, parent, facade, style, use_flickering, noisetag=None):
        self.parent = parent
        self.facade = facade
        self.style = style
        self.use_flickering = use_flickering
        self.noisetag = noisetag
        self.keys = []
        self.windows = None
        self.active_window = None
        self.switched_windows = False

        # Buttons layout:
        self.button_color = style["button_color"]
        self.line_color = style["line_color"]
        self.label_color = style["text_color"]

    def activate(self):
        """Activates all visual and functional elements of the Window."""
        pass

    def deactivate(self):
        """Deactivates all visual and functional elements of the Window."""
        pass

    def reset_key_colors(self):
        """Resets all keys to their original color"""
        pass

    def prediction_feedback(self, prediction_idx, prediction_err, feedback_threshold):
        """Provides visual feedback by changing the border color of the key which is being looked at"""
        for i, key in enumerate(self.keys):
            # If key has changed border color but is not the predicted key, reset the line color:
            if key.changed_feedback_color and i != prediction_idx:
                key.set_line_color()
                # Indicate for the given key that its border was reset to it original color:
                key.changed_feedback_color = False
            # If key is the predicted key, change border color:
            if i == prediction_idx and prediction_err < feedback_threshold:
                old_color = key.line_color
                new_color = (
                    int(old_color[0]*.4),
                    int(old_color[1]*.4),
                    int(old_color[2]*.4 + 255*(1-prediction_err)*.6)
                )
                key.set_line_color(new_color=new_color)
                # indicate for the given key that its border was changed:
                key.changed_feedback_color = True

    def feedback(self, feedback_threshold):
        """"""
        prediction_message = self.noisetag.getLastPrediction()
        pred_key_idx = -1
        keypad_key_idx = -1
        prediction_err = 0

        if prediction_message:
            # prediction id:
            prediction_idx = prediction_message.Yest - 1
            # prediction error:
            prediction_err = prediction_message.Perr
            # calculate corresponding key index:

            if prediction_idx < len(self.keys):
                pred_key_idx = prediction_idx
            else:
                keypad_key_idx = prediction_idx - len(self.keys)
        # prediction feedback over suggestion keys:
        self.prediction_feedback(
            pred_key_idx,
            prediction_err,
            feedback_threshold
        )
        # prediction feedback over keypad keys:
        if self.windows:
            self.windows[self.active_window].prediction_feedback(
                keypad_key_idx,
                prediction_err,
                feedback_threshold
            )

    def switch_window(self, window_name):
        """
        Switches to window specified by window_name.

        Args:
            window_name (str): The link to the window to switch to.
        """
        pass

    def handle_mouse_events(self):
        """Handles mouse events within the Window."""
        pass

    def draw(self, noisetag, last_flip_time, target_idx=-1):
        """
        Changes the state of the buttons/keys within the Window.
        It draws the display with the colors given by the Noisetag module if flickering is active.

        Args:
            noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
            last_flip_time (int): Timestamp of last screen update, i.e. buffer flip.
            target_idx (int): (Optional) index of the target stimulus.
        """
        pass

    def select_key(self, objID):
        """
        Selects and activates the key (like a mouseclick) if selected using prediction trials from the Noisetag.

        Args:
            objID (int): ID of the object that is selected.
        """
        pass

    def reset_flicker(self):
        """Resets the number of stimuli for the Noisetag when changing keypads."""
        pass

    def get_keys(self):
        """Returns all keys used in the Window."""
        pass
