mindaffectBCI
=============
This repository contains the python SDK code for the Brain Computer Interface (BCI) developed by the company `Mindaffect <https://mindaffect.nl>`_.

When installed, with the right hardware you can do things like shown `here <https://youtu.be/MVuQzaqDkKI>`_


Online Documentation and Tutorials
----------------------------------
Available at: `https://mindaffect-bci.readthedocs.io/ <https://mindaffect-bci.readthedocs.io/en/latest/tutorials.html>`_

Installation
------------

To install from **source** (currently the recommended method):
  1. Clone or download this repository::

       git clone https://github.com/mindaffect/pymindaffectBCI

  2. Install the necessary bits to your local python path:

    1. change to the directory where you cloned the repository.

    #. Add this module to the python path, and install dependencies::
   
         pip install -e .
  
  3. Install a JAVA JVM, such as `this one <https://adoptopenjdk.net/index.html?variant=openjdk15&jvm>`_

To install as a python library::

    pip install --upgrade mindaffectBCI

Try the off-line analysis on-line on binder.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/mindaffect/pymindaffectBCI/pip_test

Try off-line multiple datasets analysis on `kaggle <https://www.kaggle.com/mindaffect/mindaffectbci>`_ 



Installation Test
-----------------

You can run a quick test if the installation without any additional hardware by running::

  python3 -m mindaffectBCI.online_bci --acquisition fakedata

Essentially, this run the SDK test code which simulates a *fake* EEG source and then runs the full BCI sequence, with decoder discovery, calibration and prediction.

If all is successfully installed then you should see a window like this open up.

<img src='docs/source/images/mainmenu.png' width=300>

If you now press 2 you should see a flickering grid of "buttons" like below.  You should see a random one briefly flash green (it's the target) then rapidly flicker and eventually turn blue (to indicate it's selected.)

<img src='docs/source/images/selectionmatrix.png' width=300>

If all this works then you have successfully installed the mindaffectBCI python software. You should now ensure your hardware (display, amplifier) is correctly configured before jumping into BCI control.


Important: FrameRate Check
--------------------------

For rapid visual stimulation BCI (like the noisetagging BCI), it is *very* important that the visual flicker be displayed *accurately*.  However, as the graphics performance of computers varies widely it is hard to know in advance if a particular configuration is accurate enough.  To help with this we also provide a graphics performance checker, which will validate that your graphics system is correctly configured.  You can run this with::

  python3 -m mindaffectBCI.examples.presentation.framerate_check

As this runs it will show in a window your current graphics frame-rate and, more importantly, the variability in the frame times.  For good BCI performance this jitter should be <1ms.  If you see jitter greater than this you should probably adjust your graphics card settings.  The most important setting to consider is to be sure that you  have `_vsync_ <https://en.wikipedia.org/wiki/Screen_tearing#Vertical_synchronization>` *turned-on*.  Many graphics cards turn this off by default, as it (in theory) gives higher frame rates for gaming.  However, for our system, frame-rate is less important than *exact*  timing, hence always turn vsync on for visual Brain-Compuber-Interfaces!


Brain Computer Interface Test
-----------------------------

If you have:
  1. installed `pyglet <https://pyglet.org>`_ , e.g. using `pip3 install pyglet`
  #. installed `brainflow <https://brainflow.org>`_ , e.g. using `pip3 install brainflow`
  #. have connected an `openBCI ganglion <https://shop.openbci.com>`_ ,
  #. have followed `MindAffect headset layout.pdf <https://github.com/mindaffect/Headset/blob/master/MindAffect%20headset%20layout.pdf>`_ to attach the electrodes to the back of your head.

Then you can jump directly to trying a fully functional simple letter matrix BCI using::

  python3 -m mindaffectBCI.online_bci

Note: For more information on how to run an on-line BCI, *including using other supported amplifiers*, see our complete `on-line documentation <mindaffect-bci.readthedocs.io>`_ and in particular our `tutorials section <https://mindaffect-bci.readthedocs.io/en/latest/tutorials.html>`_.

Getting Support
---------------

If you run into and issue you can either directly raise an issue on the projects `github page <https://github.com/mindaffect/pymindaffectBCI>`_ 

..
    or directly contact the developers on `gitter <https://gitter.im/mindaffect>`_ -- to complain, complement, or just chat:

    .. image:: https://badges.gitter.im/mindaffect/unitymindaffectBCI.svg
      :target: https://gitter.im/mindaffect/pymindaffectBCI?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge


File Structure
--------------
This repository is organized roughly as follows:

- `mindaffectBCI <mindaffectBCI>`_ - contains the python package containing the mindaffectBCI SDK.  Important modules within this package are: 
  - `noisetag.py <mindaffectBCI/noisetag.py>`_ - This module contains the main API for developing User Interfaces with BCI control
  - `utopiaController.py <minaffectBCI/utopiaController.py>`_ - This module contains the application level APIs for interacting with the MindAffect Decoder.
  - `utopiaclient.py <mindaffectBCI/utopiaclient.py>`_ - This module contains the low-level networking functions for communicating with the MindAffect Decoder - which is normally a separate computer running the eeg analysis software.
  - stimseq.py -- This module contains the low-level functions for loading and codebooks - which define how the presented stimuli will look.

- `decoder <mindaffectBCI/decoder>`_ - contains our open source python based Brain Computer Interface decoder, for both on-line and off-line analysis of neuro-imaging data. Important modules within this package are:
  - `decoder.py <mindaffectBCI/decoder/decoder.py>`_ - This module contains the code for the on-line decoder.
  - `offline_analysis.ipynb <mindaffectBCI/decoder/offline_analysis.ipynb>`_ - This `juypter <https://jupyter.org/>`_ notebook contains to run an off-line analysis of previously saved data from the mindaffectBCI or other publically available BCI datasets. 
   
- `examples <mindaffectBCI/examples/>`_ - contains python based examples for Presentation and Output parts of the BCI. Important sub-directories
   - `output <mindaffectBCI/examples/output/>`_ - Example output modules.  An output module translates BCI based selections into actions.
   - `presentation <mindaffectBCI/examples/presentation/>`_ - Example presentation modules.  A presentation module, presents the BCI stimulus to the user, and is normally the main UI.  In particular here we have:
     - `framerate_check.py <mindaffectBCI/examples/presentation/framerate_check.py>`_ - Which you can run to test if your display settings (particularly vsync) are correct for accurate flicker presentation.
     - `selectionMatrix.py <mindaffectBCI/examples/presentation/selectionMatrix.py>`_ - Which you can run as a simple example of using the mindaffectBCI to select letters from an on-screen grid.

   - `utilities <mindaffectBCI/examples/utilities/>`_ - Useful utilities, such as a simple *raw* signal viewer
   - `acquisition <mindaffectBCI/examples/acquisition/>`_ - Example data acquisition modules.  An acquisition module interfaces with the EEG measurment hardware and streams time-stamped data to the hub.

- `docs <docs/>`_ -- contains the documentation.

  - `source <docs/source>`_ -- contains the source for the documentation, in particular this directory contains the juypter notebooks for tutorials on how to use the mindaffectBCI.
  
    - `online_bci.ipynb <docs/source/quickstart.ipynb>`_ - This `juypter <https://jupyter.org/>`_ notebook contains the code to run a complete on-line noise-tagging BCI
