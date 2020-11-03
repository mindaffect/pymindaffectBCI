Frequently Asked Questions
==========================



How do I improve my reported calibration accuracy?
--------------------------------------------------
First, open the signal viewer by pressing 0 in the main menu to check the quality of the EEG signal. If the reported noise to signal ratio is high, try to reduce the noise as follows:  

- Make sure your headset is properly set up and fitted. See our instructions here: :ref:`fittingRef`.	
- Move away from wall outlets, plugged in electronic devices, and other potential sources of 50Hz noise.  
- If possible, place the amplifier behind you so your body is between the machine running the mindaffcetBCI and the amplifier. 
- Check all the connections between the electrodes in your headset and the amplifier. 

Second, make sure that you have followed our OS Optimization instructions: :ref:`osoptRef`, and you are running the BCI in fullscreen mode: :ref:`fullscreenRef`.   
If the issue still remains you may have to dive deeper in the timing stability of your system. 
To do this we provide tutorials on:

- Building an Opto-sensor
- Building a trigger circuit
- Analysing your opto and trigger data

My calibration accuracy is fine but prediction mode does not work.
-------------------------------------------------------------
- Add more calibration trials to the model by running the calibration sequence multiple times, or change the :code:`ncal` argument in the :code:`online_bci.json` configuration file (or the config file you are currently using). 
- Run the BCI in always full-screen mode: :ref:`fullscreenRef`
- Inconsistent frame timing between calibration and prediction can cause this issue. To check the frametime stability of your system follow our tutorials on:

	- Building an Opto-sensor
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
To run the BCI in full-screen mode set the :code:`fullscreen` parameter in the :code:`onlin_bci.json` configuration file --or any other :code:`.json` config file you are currently using-- to :code:`true`. 
