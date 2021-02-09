.. _lower-case-layout:

lower_case_layout
=================

This is the standard keyboard that appears when the application is opened.
The JSON file for this keyboard looks like this:

.. code-block:: json

    [
        [
            ["BASIC_KEY","q"], ["BASIC_KEY","w"], ["BASIC_KEY","e"], ["BASIC_KEY","r"], ["BASIC_KEY","t"],
            ["BASIC_KEY","y"], ["BASIC_KEY","u"], ["BASIC_KEY","i"], ["BASIC_KEY","o"], ["BASIC_KEY","p"]
        ],
        [
            ["BASIC_KEY","a"], ["BASIC_KEY","s"], ["BASIC_KEY","d"], ["BASIC_KEY","f"], ["BASIC_KEY","g"],
            ["BASIC_KEY","h"], ["BASIC_KEY","j"], ["BASIC_KEY","k"], ["BASIC_KEY","l"]
        ],
        [
            ["BASIC_KEY","z"], ["BASIC_KEY","x"], ["BASIC_KEY","c"], ["BASIC_KEY","v"], ["BASIC_KEY","b"],
            ["BASIC_KEY","n"], ["BASIC_KEY","m"], ["CLEAR_KEY","key_icons\\clear_default.png"],
            ["BACKSPACE_KEY","key_icons\\delete_space_default.png"]
        ],
        [
            ["SWITCH_TO_MENU_KEY","key_icons\\settings_default.png"], ["TTS_KEY","key_icons\\speech_default.png"],
            ["SWITCH_TO_SYMBOLS_KEY","?123"], ["SWITCH_TO_UPPER_KEY","ABC"], ["BASIC_KEY",","], ["SPACE_BAR_KEY","Space"],
            ["BASIC_KEY","."], ["ENTER_KEY","key_icons\\enter_default.png"]
        ]
    ]

This results in the following keyboard:

.. image:: /../images/lower_case_layout.png
   :width: 1920
   :height: 540
   :scale: 40
   :alt: lower_case_layout image