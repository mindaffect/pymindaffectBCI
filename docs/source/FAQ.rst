Frequently Asked Questions
==========================



How do I improve my calibration accuracy?
--------------------------------------------------
First, open the signal viewer by pressing 0 in the main menu to check the quality of the EEG signal. If the reported noise to signal ratio is high, try to reduce the noise as follows:  

- Make sure your headset is properly set up and fitted. See our instructions here: :ref:`fittingRef`.	
- Move away from wall outlets, plugged in electronic devices, and other potential sources of line-noise.  
- If possible, place the amplifier behind you so your body is between the machine running the mindaffcetBCI and the amplifier.
- Place the EEG hardware close to or on your body and run the electrode cables over your back.  
- Check all the connections between the electrodes in your headset and the amplifier. 

Second, make sure that you have followed our OS Optimization instructions: :ref:`osoptRef`, and you are running the BCI in fullscreen mode: :ref:`fullscreenRef`.   
If the issue still remains you may have to dive deeper in the timing stability of your system. 
To do this we provide tutorials on:

- :ref:`optobuildRef`
- Building a trigger circuit
- :ref:`triggercheckRef` Analysing your opto and trigger data

My calibration accuracy is fine but prediction mode does not work.
-------------------------------------------------------------
- Add more calibration trials to the model by running the calibration sequence multiple times, or change the :code:`ncal` argument in the :code:`online_bci.json` configuration file (or the config file you are currently using). 
- Run the BCI in always full-screen mode: :ref:`fullscreenRef`
- Inconsistent frame timing between calibration and prediction can cause this issue. To check the frametime stability of your system follow our tutorials on:

	- :ref:`optobuildRef`
	- Building a trigger circuit
	- Analysing your opto and trigger data

I'm getting an  :code:`Acq did not start correctly` and/or a :code:`brainflow.board_shim.BrainFlowError` message.
--------------------------------------------------------------------------------------------------------------
- Check if your amplifier is turned on and --if needed-- the usb-dongle is plugged in.  
- When using an OpenBCI amplifier with the WIFI-shield, make sure it is connected to the same wireless network as the machine running the mindaffcetBCI.  
- Check you serial port settings as described in the :ref:`COMref` section of the installation instructions.

.. _fullscreenRef:

How do I run the BCI in full-screen mode?
-----------------------------------------
To run the BCI in full-screen mode set the :code:`fullscreen` parameter in the :code:`onlin_bci.json` configuration file --or in any other :code:`.json` config file you are currently using-- to :code:`true`. 

Can I use the mindaffectBCI without an EEG acquisition device?
--------------------------------------------------------------
In some scenarios it is useful to run the BCI without having to connect an amplifier that's streaming real brain data (e.g. debugging/developing other components of the BCI). 
To run the full BCI stack (i.e. hub, acquisition, decoder and presentation) with a fake data stream, launch it with the :code:`debug.json` config file::

	python -m mindaffectBCI.online_bci --config_file debug.json

Alternatively, do run full decoder stack (i.e. hub, acquisation and decoder) *without* presenation use::

	python -m mindaffectBCI.online_bci --config_file fake_recogniser.json

When using the fake data stream, calibration and cued prediction performance will be 100%. In Free Typing mode selections are made randomly. 


I'm getting a `framework not found` error on Mac OS
---------------------------------------------------

On mac-os Big-Sur there is a known issue with older versions of pyglet see `here <https://github.com/pyglet/pyglet/issues/274>`_.  The solution is to ensure you are running pyglet 1.5.11 or higher.  You can directly update your pyglet install with: :code:`pip3 install --upgrade pyglet`

I'm getting poor performance on linux
-------------------------------------

As we are a small team, we have decided to focus our development and testing effort mainly on Windows PCs.  We have tested the BCI on linux and mac-os, and it technically works.  However, as mentioned here :ref:`triggercheckRef` it is also important that the screen redraws be accuratly time-stamped.  In our testing on linux this time-locking was less robust than with an optimized windows installation.  We believe this is can be addressed by a correct graphics system configuration, but have not identified it as yet. We would welcome feedback from the community about how to setup linux better.


 
