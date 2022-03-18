.. _adding-new-languages-guide:

Guide to adding new languages
=============================

New languages can be added to be used in the keyboard. For examples, you can refer to the pre-existing code.

Replace LANGUAGE KEY with the code of your language e.g. 'EN' and LANGUAGE with the language e.g. ENGLISH

Buttons
-------
1. In :ref:`key-type` add a toggle key named  as ``TOGGLE_ON_LANGUAGE_KEY = (ToggleFunction(link="language", mode="LANGUAGE KEY"), False)``
2. In `keyboard_config.json` add the button under "Set language" as ``["TOGGLE_ON_LANGUAGE_KEY","LANGUAGE KEY"]``

Correction
----------
If there is an internet connection and online frequency list available for the chosen language, the program will automatically download it. 

Prediction
----------
1. The program will make an n-gram file that builds prediction based on what the user has previously written.
2. If you want to have prediction from the start you will need to train your own n-gram model. To do this you will have to download .txt files in your language that will be used to train on. A good source for this is the `Gutenberg project <https://www.gutenberg.org>`_.
3. When you have the files, put them in a folder and run :ref:`create-ngram` in your command line. Look at the documentation for the possible arguments.
