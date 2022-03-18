Smart BCI keyboard
============================


Introduction
------------
This module implements a keyboard for the BCI system developed by MindAffect.
   * It supports adding your own configurable keyboard from a JSON file.
   * Multiple keyboards (sub-keyboards) can be used.
   * It is to be used cross-platform (Windows, MacOS, Linux).
   * Predictive text: given the first few letters of a word,
     the user will be given the most likely options as 'keys' on the keyboard.
   * Text to speech: allows the user to say text out loud (with a 'say-out-loud' key).
   * Word correction: adds automatic correction of misspelled words with easy ability to
     undo.

Installation
~~~~~~~~~~~~
Our keyboard implementation is based on the `mindaffectbci module
<https://pypi.org/project/mindaffectBCI/>`_ developed by `MindAffect
<https://www.mindaffect.nl/>`_ so this should be installed.  (Or it will be automatically installed when installing this package.)

The full list of required packages can be found in `requirements.txt`

We recommend using Python version at least 3.8

To install:

 1.  Download or checkout a copy of this repository, from `mindaffect/smart-keyboard <https://github.com/mindaffect/smart-keyboard>`_
 2.  Install this package and all it's dependencies into your python environment, by running the following command _in the director where this repository was downloaded_ ::

     python3 -m pip install -e .

If everything is correctly installed, then you can quickly test the system by running the mindaffectBCI with a 'fakedata' stream and the smart-keyboard configuration with ::

    python3 -m mindaffectBCI.online_bci --config_file smart_keyboard.json --acquisition fakedata

Note: If this does not work because the `smart_keyboard.json` file cannot be found.  Then try ::

    python3 -m mindaffectBCI.online_bci --acquisition fakedata

You will then be prompted to select a configuration file.  Navigate to the directory where `smart-keyboard` is installed and select the `smart_keyboard.json` file.

If all is installed correctly, the BCI should start and you should see the `smart_keyboard` configuration screen, which looks like this.

.. image:: docs/images/configuration_screen.png
   :width: 795
   :height: 630
   :scale: 50
   :alt: Configuration screen of the keyboard

Quickstart
~~~~~~~~~~

If the installation test was successfull, and you have a supported EEG amplifier.  You can now try a test with brain signals.  To do this:

 1. Modify the `smart_keyboard.json` configuration file to match your amplifier configuration.  See the mindaffectBCI docs for more information `here <https://mindaffect-bci.readthedocs.io/en/latest/goingfurther.html>`_ on how to configure for differnt types of amplifier.  (For an openBCI cyton, the current config should work 'out-of-the-box').
 2. Run the BCI with::

    python3 -m mindaffectBCI.online_bci
  3. In the file-selection window which opens select the `smart_keyboard.json` file which you configured in step 1.


You should now be able to run a complete BCI system following the instructions outlined in the `user guide <https://github.com/mindaffect/smart-keyboard/blob/main/User%20Guide%20-%20BCI%20keyboard.pdf>`_


Adding a configurable JSON keyboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using a `keyboard editor <http://www.keyboard-layout-editor.com/#/>`_ we can generate
custom .json files that can be read into the program. Unlike normal keyboards we only allow
one character per key, much like mobile keyboards, so no usage of shift to select a different
thing on the same key.

More information on this can be found in guide_to_creating_custom_json_keyboards.txt

.. image:: docs/images/keyboard.png
   :width: 795
   :height: 630
   :scale: 50
   :alt: Wow such a cool keyboard!

Word prediction and autocompletion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Our word prediction and autocompletion module ues a n-gram prediction engine, which has been trained on free to use e-books.  Standard (pretrained) languages that are provided in this module are English, Dutch and German.  It is posible to add custom languages via creating new N-gram files by training them on texts in the desired language.

While typing the n-gram files are updated based on what the user types.  This means that frequently used words, names, or sentences will appear more often in the word prediction keys, with better predictions the more the keyboard is used.

Word correction
~~~~~~~~~~~~~~~
Our word correction module uses the `SpellChecker <https://pypi.org/project/pyspellchecker/>`_
module which is based on Peter Norvig's `blog post <https://norvig.com/spell-correct.html>`_
on setting up a simple spell checking algorithm.

English, Spanish, German, French and Portugese were already supported by this module. Our
module allows the user to add their own language if they have a word frequency file for
this language. We also implemented the option to download such a frequency list from a 
`Git repository. <https://github.com/hermitdave/FrequencyWords>`_

Text to speech
~~~~~~~~~~~~~~
Our text-to-speech module uses the `pyttsx3 module <https://pypi.org/project/pyttsx3/>`_
which works offline and is compatible with both Python 2 and 3.
If there is a connection to the internet, Google's `gTTs <https://pypi.org/project/gTTS/>`_
will be used.

Problems
~~~~~~~~
First check if your Python version is not outdated and if all the required packages have
been installed correctly (see Requirements).

Authors and acknowledgment
--------------------------
*The base keyboard module plus word completion was developed by the following group of students:*

Thomas de Lange,
Thomas Jurriaans,
Damy Hillen,
Joost Vossers,
Jort Gutter,
Florian Handke,
Stijn Boosman

*Developed in close collaboration with* `MindAffect <https://www.mindaffect.nl/>`_

License
-------
MIT License (MIT)


Project status
--------------
Project is in development
