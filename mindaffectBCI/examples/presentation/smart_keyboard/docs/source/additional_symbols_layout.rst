.. _additional-symbols-layout:

additional_symbols_layout
=========================

This is the standard second sub keyboard which contains additional special characters.
The JSON file for this keyboard looks like this:

.. code-block:: json

    [
      [
        ["BASIC_KEY","~"], ["BASIC_KEY","`"], ["BASIC_KEY","|"], ["BASIC_KEY","•"], ["BASIC_KEY","º"],
        ["BASIC_KEY","π"], ["BASIC_KEY","÷"], ["BASIC_KEY","×"], ["BASIC_KEY","√"], ["BASIC_KEY","¶"]
      ],
      [
        ["BASIC_KEY","€"], ["BASIC_KEY","¥"], ["BASIC_KEY","£"], ["BASIC_KEY","ȼ"], ["BASIC_KEY","^"],
        ["BASIC_KEY","="], ["BASIC_KEY","{"], ["BASIC_KEY","}"], ["BASIC_KEY","<"], ["BASIC_KEY", ">"]
      ],
      [
        ["SWITCH_TO_SYMBOLS_KEY","?123"], ["BASIC_KEY","%"], ["BASIC_KEY","©"], ["BASIC_KEY","®"], ["BASIC_KEY","™"],
        ["BASIC_KEY","℅"], ["BASIC_KEY","["], ["BASIC_KEY","]"], ["CLEAR_KEY","key_icons\\clear_default.png"],
        ["BACKSPACE_KEY","key_icons\\delete_space_default.png"]
      ],
      [
        ["SWITCH_TO_MENU_KEY","key_icons\\settings_default.png"], ["TTS_KEY","key_icons\\speech_default.png"],
        ["SWITCH_TO_LOWER_KEY","abc"], ["SWITCH_TO_UPPER_KEY","ABC"],["BASIC_KEY",","], ["SPACE_BAR_KEY","Space"],
        ["BASIC_KEY","."], ["ENTER_KEY","key_icons\\enter_default.png"]
      ]
    ]

This results in the following keyboard:

.. image:: /../images/additional_symbols_layout.png
   :width: 1920
   :height: 540
   :scale: 40
   :alt: additional_symbols_layout image