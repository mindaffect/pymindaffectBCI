"""
This module sets up a keyboard provided any GUI.

It contains a single class ``Application``.

An ``Application`` has the following functionality:
 * A clickable functional keyboard including textfield and prediction keys.
 * A clickable menu.

An ``Application`` has the following two layouts:
 * A Keyboard which handles all keyboard functionality.
 * A Menu which handles all settings and bci_calibration.
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

from mindaffectBCI.noisetag import Noisetag
#from mindaffectBCI.examples.presentation.smart_keyboard.psychopy_facade import PsychopyFacade
from mindaffectBCI.examples.presentation.smart_keyboard.pyglet_facade import PygletFacade
from mindaffectBCI.examples.presentation.smart_keyboard.windows.connection_window import ConnectionWindow
from mindaffectBCI.examples.presentation.smart_keyboard.windows.menu_window import MenuWindow
from mindaffectBCI.examples.presentation.smart_keyboard.windows.cued_prediction_window import CuedPredictionWindow
from mindaffectBCI.examples.presentation.smart_keyboard.windows.calibration_window import CalibrationWindow
from mindaffectBCI.examples.presentation.smart_keyboard.windows.keyboard_window import KeyboardWindow
from mindaffectBCI.examples.presentation.smart_keyboard.windows.electrode_quality_window import ElectrodeQualityWindow
from mindaffectBCI.examples.presentation.smart_keyboard.windows.calibration_reset_window import CalibrationResetWindow
from mindaffectBCI.examples.presentation.smart_keyboard.app_exceptions import Logger
from mindaffectBCI.examples.presentation.smart_keyboard.settings_manager import SettingsManager
import json

# Import for a rare error, we don't quite understand:
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Application:
    """
    A container that handles switching between the different Keyboard and Menu Windows.

    Args:
        facade: (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        keyboard_config (dict): Mapping of all configurable keyboard settings loaded from a JSON file.
        user_config (dict): Mapping of the user settings loaded from a JSON file.

    Attributes:
        parent (windows.window.Window): The parent of a window if existing.
        facade (framework_facade.FrameworkFacade): Contains the GUI specific functionality.
        keyboard_config (dict): Mapping of all configurable settings loaded from a JSON file.
        user_config (dict): Mapping of the user settings loaded from a JSON file.
        use_flickering (bool): A boolean indicating whether or not to use flickering.
        style (dict): Style configurations for the keyboard.
        noisetag (noisetag.Noisetag): Reference to the Noisetag module from MindAffect.
        last_flip_time (int): Timestamp of last screen update.
        switched_windows (bool) Indicates if a window changed has just occurred.
        windows (dict): Mapping from Window names to Window objects.
        active_window (string): The name of the currently active Window.
    """

    def __init__(self, facade, keyboard_config, user_config):
        self.parent = None
        self.facade = facade
        self.keyboard_config = keyboard_config
        self.user_config = user_config

        self.use_flickering = user_config["use_flickering"]
        self.style = keyboard_config["style"]

        self.noisetag = self.setup_noisetag()
        self.last_flip_time = 0
        self.switched_windows = False

        # Creates the settings manager object so it can be referenced later
        SettingsManager.get_instance(user_config)

        # Initially only the connection window is loaded which will attempt to connect the Noisetag to the Utopia Hub:
        self.windows = {
            "Connection": ConnectionWindow(
                parent=self,
                facade=self.facade,
                style=self.style,
                use_flickering=self.use_flickering,
                noisetag=self.noisetag,
                host= user_config.get('host',None)
            )
        }

        self.active_window = "Connection"
        self.windows[self.active_window].activate()

    def build_windows(self, connection_window, use_flickering, start_window="Menu"):
        """
        This function is called from an instance of ConnectionWindow and builds all required windows for the application
        to run.

        Args:
            connection_window (window.ConnectionWindow): A reference to the calling instance of ConnectionWindow.
            use_flickering (bool): The result of the connection attempt.
        """
        # Set use_flickering to the result of the connection attempt at all locations:
        self.use_flickering = use_flickering

        # Clears the reference to the Noisetag if flickering is not enabled:
        if not use_flickering:
            self.noisetag = None
        else:
            # Add selection_handler to the noisetag when a successful connection was made:
            self.noisetag.addSelectionHandler(self.selection_handler)

        # Building of the required Windows:
        #try:
        if 1:
            self.windows = {
                "Menu": MenuWindow(
                    parent=self,
                    facade=self.facade,
                    menu_config=self.keyboard_config["Menu"],
                    style=self.style,
                    use_flickering=use_flickering,
                    noisetag=self.noisetag
                ),
                "Keyboard": KeyboardWindow(
                    parent=self,
                    facade=self.facade,
                    ai_settings=self.user_config["ai_settings"],
                    use_flickering=use_flickering,
                    config=self.keyboard_config["Keyboard"],
                    style=self.style,
                    noisetag=self.noisetag
                ),
                "Calibration": CalibrationWindow(
                    parent=self,
                    facade=self.facade,
                    calibration_config=self.keyboard_config["Calibration"],
                    style=self.style,
                    use_flickering=use_flickering,
                    noisetag=self.noisetag
                ),
                "CuedPrediction": CuedPredictionWindow(
                    parent=self,
                    facade=self.facade,
                    cued_prediction_config=self.keyboard_config["CuedPrediction"],
                    style=self.style,
                    use_flickering=use_flickering,
                    noisetag=self.noisetag
                ),
                "ElectrodeQuality": ElectrodeQualityWindow(
                    parent=self,
                    facade=self.facade,
                    style=self.style,
                    use_flickering=self.use_flickering,
                    noisetag=self.noisetag,
                ),
                "CalibrationReset": CalibrationResetWindow(
                    parent=self,
                    facade=self.facade,
                    style=self.style,
                    use_flickering=self.use_flickering,
                    noisetag=self.noisetag,
                )
            }
        # except KeyError as exc:
        #     # Triggers if a key in the config file is spelled incorrectly, e.g. 'keyboard1' won't be found by 'keyboard'
        #     Logger.log_config_key_error()
        # except Exception as exc:
        #     Logger.log_config_error()
        if 1:
        # else:
            # Triggers if no exception is triggered

            # Deactivates the ConnectionWindow:
            connection_window.deactivate()

            # start_window='Calibration'

            self.windows[start_window].activate()
            self.active_window = start_window

    def setup_noisetag(self):
        """Sets up a Noisetag object to communicate with the Utopia server."""
        noisetag = Noisetag(stimFile=self.user_config.get('stimfile',None), clientid='smart_keyboard')

        return noisetag

    def selection_handler(self, objID):
        """Passes the predicted object id to the active Window."""
        self.windows[self.active_window].select_key(objID)

    def switch_window(self, window_name):
        """
        Switches focus to the specified Window.

        Args:
            window_name (str): The name of the Window to switch focus to.
        """
        if self.windows:
            if window_name in self.windows:
                self.windows[self.active_window].deactivate()
                self.windows[window_name].activate()
                self.active_window = window_name
            else:
                self.parent.switch_window(window_name)
                self.switched_windows = True
        else:
            self.parent.switch_window(window_name)
            self.switched_windows = True

    def handle_mouse_events(self):
        """Calls the handle_mouse_events function of the active Window."""
        self.windows[self.active_window].handle_mouse_events()

    def draw(self):
        """Calls the draw method of the active Window."""
        self.windows[self.active_window].draw(self.noisetag, self.last_flip_time)

    def set_flip_time(self):
        """call-back immeadiately after flip to record the flip-time."""
        if self.use_flickering:
            self.last_flip_time = self.noisetag.getTimeStamp()

    def get_window(self, window_name):
        """Gives access to Window specified by window_name."""
        if self.windows:
            if window_name in self.windows:
                return self.windows[window_name]
            else:
                return self.parent.get_window(window_name)
        else:
            return self.parent.get_window(window_name)


def load_json(file):
    """
    Loads JSON file and returns it as dictionary.

    Args:
        file (str): Path of JSON file to be loaded.
    """
    try:
        return json.load(open(file, "r", encoding="utf-16"))
    except OSError as exc:
        Logger.log_layout_OS_error()
    except Exception as exc:
        Logger.log_layout_generic_error()


def run(symbols=None, keyboard_config:str="keyboard_config.json", user_config:str="user_config.json", ncal:int=10, npred:int=10, stimfile=None, selectionThreshold:float=None,
        framesperbit:int=1, optosensor:bool=True, fullscreen:bool=None, windowed:bool=None,
        fullscreen_stimulus:bool=True, simple_calibration=False, host=None, calibration_symbols=None, bgFraction=.1):
    """Starts the keyboard."""

    # TODO[]: actually use the run options to configure the keyboard!

    # Load the config files:
    # TODO[]: search for the config file location
    basedirectory=os.path.dirname(os.path.abspath(__file__))
    if isinstance(keyboard_config,str):
        if not os.path.exists(keyboard_config):
            keyboard_config = os.path.join(basedirectory,"configs",keyboard_config)
        keyboard_config = load_json(keyboard_config)
    if isinstance(user_config,str):
        if not os.path.exists(user_config):
            user_config = os.path.join(basedirectory,"configs",user_config)
        user_config = load_json(user_config)

    if fullscreen is None: # override with argument
        fullscreen = user_config.get('full_screen',False)
    if not stimfile is None:
        user_config['stimfile']=stimfile
    if not framesperbit is None:
        user_config['framesperbit']=framesperbit
    if not optosensor is None:
        user_config['optosensor']=optosensor
    if not simple_calibration is None:
        user_config['simple_calibration']=simple_calibration
    if not selectionThreshold is None:
        user_config['selectionThreshold']=selectionThreshold
    if not ncal is None:
        user_config['ncal']=ncal

    if not host is None:
        user_config['host']=host

    #try:
    if 1:
        # facade = PsychopyFacade(
        #     size=user_config["screen_dimensions"],
        #     full_screen=fullscreen,
        #     wait_blanking=keyboard_config["wait_blanking"]
        # )

        facade = PygletFacade(
            size=user_config["screen_dimensions"],
            full_screen=fullscreen,
            wait_blanking=keyboard_config["wait_blanking"]
        )

        application = Application(
            facade=facade,
            keyboard_config=keyboard_config,
            user_config=user_config
        )

        # Event-loop:
        facade.start(application, keyboard_config["exit_keys"])

    # except KeyError as exc:
    #     Logger.log_config_key_error()
    # except Exception as exc:
    #     Logger.log_config_error()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyboard_config',type=str,help='the config file name to use for the keyboard',default='keyboard_config.json')
    parser.add_argument('--user_config',type=str,help='the config file name to use for the keyboard',default='user_config.json')
    parser.add_argument('ncal',type=int, help='number calibration trials', nargs='?', default=10)
    parser.add_argument('npred',type=int, help='number prediction trials', nargs='?', default=10)
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--stimfile',type=str, help='stimulus file to use', default=None)
    parser.add_argument('--framesperbit',type=int, help='number of video frames per stimulus bit', default=1)
    parser.add_argument('--fullscreen',type=bool, help='set fullscreen state',default=False)
    parser.add_argument('--selectionThreshold',type=float,help='target error threshold for selection to occur',default=.1)
    parser.add_argument('--simple_calibration',action='store_true',help='flag to only show a single target during calibration')
    #parser.add_argument('--symbols',type=str,help='file name for the symbols grid to display',default=None)
    parser.add_argument('--calibration_symbols',type=str,help='file name for the symbols grid to use for calibration',default=None)
    #parser.add_option('-m','--matrix',action='store',dest='symbols',help='file with the set of symbols to display',default=None)
    args = parser.parse_args()

    run(**vars(args))
