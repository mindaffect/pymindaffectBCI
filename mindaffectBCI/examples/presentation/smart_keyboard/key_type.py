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

import enum
from mindaffectBCI.examples.presentation.smart_keyboard.functional_provider import *


class KeyType(enum.Enum):
    """This class defines the key types currently used.

    Args:
        func_provider (FunctionalProvider): The functionality of a key type.
        has_icon (bool): Whether a key is loaded in with an icon instead of a label text.

    """
    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        return obj

    def __init__(self, func_provider, has_icon):
        self.func_provider = func_provider
        self.has_icon = has_icon

    # Keys used in the keyboard
    BASIC_KEY = (BasicFunction(), False)
    ENTER_KEY = (EnterFunction(), True)
    SPACE_BAR_KEY = (SpaceBarFunction(), False)
    BACKSPACE_KEY = (BackspaceFunction(), True)
    TTS_KEY = (TTSFunction(), True)
    CLEAR_KEY = (ClearFunction(), True)
    SUGGESTION_KEY = (SuggestionFunction(), False)

    # Keys used to link keypads of the keyboard
    SWITCH_TO_UPPER_KEY = (SwitchWindowFunction(link="Upper"), False)
    SWITCH_TO_LOWER_KEY = (SwitchWindowFunction(link="Lower"), False)
    SWITCH_TO_SYMBOLS_KEY = (SwitchWindowFunction(link="Symbols"), False)
    SWITCH_TO_ADDITIONAL_SYMBOLS_KEY = (SwitchWindowFunction(link="AdditionalSymbols"), False)

    # Keys used to link keyboard and menu
    SWITCH_TO_MENU_KEY = (SwitchWindowFunction(link="Menu"), True)
    SWITCH_TO_KEYBOARD_KEY = (SwitchWindowFunction(link="Keyboard"), True)

    # Keys used for navigation in the menu
    SWITCH_TO_CONFIGURATION_MENU_KEY = (SwitchWindowFunction(link="ConfigurationMenu"), False)
    SWITCH_TO_GENERAL_MENU_KEY = (SwitchWindowFunction(link="GeneralMenu"), False)
    SWITCH_TO_SAVE_LOAD_TEXT_MENU_KEY = (SwitchWindowFunction(link="SaveMenu"), False)
    SWITCH_TO_ELECTRODE_QUALITY_KEY = (SwitchWindowFunction(link="ElectrodeQuality"), False)

    TOGGLE_ON_WORD_PREDICTION_CORRECTION_KEY = (ToggleFunction(link="word_prediction_correction", mode=True), False)
    TOGGLE_OFF_WORD_PREDICTION_CORRECTION_KEY = (ToggleFunction(link="word_prediction_correction", mode=False), False)

    TOGGLE_ON_GERMAN_KEY = (ToggleFunction(link="language", mode="DE"), False)
    TOGGLE_ON_ENGLISH_KEY = (ToggleFunction(link="language", mode="EN"), False)
    TOGGLE_ON_DUTCH_KEY = (ToggleFunction(link="language", mode="NL"), False)

    SAVE_TEXT_KEY = (SaveTextFunction(), False)

    PLUS_KEY = (PlusMinusFunction(mode=True, link="text_size"), False)
    MINUS_KEY = (PlusMinusFunction(mode=False, link="text_size"), False)
    DISPLAY_TEXT_SIZE_KEY = (DisplayFunction(link="text_size"), False)

    DEFAULT_SETTINGS_KEY = (DefaultSettingsFunction(), False)

    # Keys used in calibration menu
    SWITCH_TO_CALIBRATION_KEY = (SwitchWindowFunction(link="Calibration"), False)
    RESET_CALIBRATION = (SwitchWindowFunction(link="CalibrationReset"), False)

    # Keys used in cued prediction menu
    SWITCH_TO_CUED_PREDICTION_KEY = (SwitchWindowFunction(link="CuedPrediction"), False)
