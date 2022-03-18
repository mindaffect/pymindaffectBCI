.. _upper-case-layout:

upper_case_layout
=================

This is the standard upper case keyboard.
The JSON file for this keyboard looks like this:

.. code-block:: json

    [
      [
        ["BASIC_KEY","Q"], ["BASIC_KEY","W"], ["BASIC_KEY","E"], ["BASIC_KEY","R"], ["BASIC_KEY","T"],
        ["BASIC_KEY","Y"], ["BASIC_KEY","U"], ["BASIC_KEY","I"], ["BASIC_KEY","O"], ["BASIC_KEY","P"]
      ],
      [
        ["BASIC_KEY","A"], ["BASIC_KEY","S"], ["BASIC_KEY","D"], ["BASIC_KEY","F"], ["BASIC_KEY","G"],
        ["BASIC_KEY","H"], ["BASIC_KEY","J"], ["BASIC_KEY","K"], ["BASIC_KEY","L"]
      ],
      [
        ["BASIC_KEY","Z"], ["BASIC_KEY","X"], ["BASIC_KEY","C"], ["BASIC_KEY","V"], ["BASIC_KEY","B"],
        ["BASIC_KEY","N"], ["BASIC_KEY","M"], ["CLEAR_KEY","key_icons\\clear_default.png"],
        ["BACKSPACE_KEY","key_icons\\delete_space_default.png"]
      ],
      [
        ["SWITCH_TO_MENU_KEY","key_icons\\settings_default.png"], ["TTS_KEY","key_icons\\speech_default.png"],
        ["SWITCH_TO_SYMBOLS_KEY","?123"], ["SWITCH_TO_LOWER_KEY","abc"], ["BASIC_KEY",","], ["SPACE_BAR_KEY","Space"],
        ["BASIC_KEY","."], ["ENTER_KEY","key_icons\\enter_default.png"]
      ]
    ]

This results in the following keyboard:

.. image:: /../images/upper_case_layout.png
   :width: 1920
   :height: 540
   :scale: 40
   :alt: upper_case_layout image