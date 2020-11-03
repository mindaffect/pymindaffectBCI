Installation
=========================

Requirements
------------
- Python (3.x) installation (Suggested: Anaconda_ ) 
- EEG amplifier (e.g. OpenBCI Cyton, OpenBCI Ganglion )

.. _Anaconda: https://docs.anaconda.com/anaconda/install/


Recommended Setup
*****************
The mindaffcetBCI can be used with various EEG acquisition and presentation devices and is therefore not limited to our recommended setup. 
It is however tested on the following supported hardware:  
  
- Amplifier: OpenBCI Cyton 
- Headset:  The Mindaffect Headset (Find the files to 3d print your own on our `Github <https://github.com/mindaffect/Headset>`_)
- Windows 10 machine

OS Optimization
***************************
For rapid visual stimulation BCI (like the MindAffect BCI), it is very important that the visual flicker is displayed accurately.
From our testing we found that the following things help in improving timing accuracy:  

- For windows 10: Disable full-screen optimization for your Python executable as explained `here <https://www.tenforums.com/tutorials/104080-enable-disable-fullscreen-optimizations-windows-10-a.html>`_.
- For laptop users: make sure that your charger is plugged in and your machine is in *power mode: Best performance* (or a similar). `How to change power mode <https://support.microsoft.com/en-us/windows/change-the-power-mode-for-your-windows-10-pc-c2aff038-22c9-f46d-5ca0-78696fdf2de8>`_.
- Set you screen to maximum brightness and disable *Night Light*, *f.lux*, or other applications that change the colour temperature of your screen. 
 

Installing the package
----------------------

To install the code:
  1. Clone or download the pymindaffectBCI repository::

       git clone https://github.com/mindaffect/pymindaffectBCI
  
  2. Switch to the open_source branch::
		
		git checkout open_source
			
  3. Install the necessary bits to your local python path:

    1. change to the directory where you cloned the repository.
    2. Add this module to the python path, and install dependencies::
   
         pip install -e .

COM port
--------
When using either the OpenBCI Ganglion or Cyton *with an USB-dongle* we have to pass the serial_port argument, to find the serial port in use by your amplifier follow the following instructions:


On Windows
**********
1. Open Device Manager and unfold Ports(COM&LPT), the com port number is shown behind your used bluetooth adapter. 

    .. image:: images/comport.jpg

Then, in the online_bci file your configuration file (mindaffectBCI/online_bci.json) you should have: :code:`"serial_port":"COM_X_"`


On Mac
*******
1. Open a Terminal session
2. Type: :code:`ls /dev/cu.*`, and look for something like :code:`/dev/cu.usbmodem1` (or similar)::

	$ ls /dev/cu.*
    /dev/cu.Bluetooth-Modem		/dev/cu.iPhone-WirelessiAP
    /dev/cu.Bluetooth-PDA-Sync	/dev/cu.usbserial
    /dev/cu.usbmodem1
	
Then, in the online_bci configuration file (mindaffectBCI/online_bci.json) you should define as  :code:`"serial_port":"dev/cu.your_com_name"`


On Linux
*********
1. Open a Terminal session
2. Plug in your USB dongle
3. Type :code:`dmesg`, and look for something like :code:`ttyACM0` or :code:`ttyUSB0` (or similar)::
	
	[43.364199] usb 2-1: New USB device found, idVendor=2458, idProduct=0001, bcdDevice= 0.01
	[43.364206] usb 2-1: New USB device strings: Mfr=1, Product=2, SerialNumber=3
	[43.364209] usb 2-1: Product: Low Energy Dongle
	[43.364213] usb 2-1: Manufacturer: Bluegiga
	[43.364215] usb 2-1: SerialNumber: 1
	[43.394168] cdc_acm 2-1:1.0: ttyACM0: USB ACM device
	
Then, in the online_bci configuration file (mindaffectBCI/online_bci.json) you should define as  :code:`"serial_port":"dev/ttyXXXX"`	

Linux Serial Port Permissions
******************************
As explained in the `OpenBCI Docs <https://docs.openbci.com/docs/06Software/01-OpenBCISoftware/GUIDocs>`_, on Linux you need to have permission to acces the serial ports of your machine.
Otherwise, you will get the error Failed to connect using :code:`/dev/ttyUSB0` or similar.  
To fix this follow their instructions:  

1. First, verify if the user does belong to the *dialout* group using the :code:`id` command.

    - Type :code:`id -Gn <username>` in terminal and check if it prints dialout as one of the options.  
    - Replace with your Linux username. Example: :code:`id -Gn susieQ`  
2. Next, add the user to the *dialout* supplementary group.

    - Type :code:`sudo usermod -a -G dialout <username>` in terminal.  
    - Example: :code:`sudo usermod -a -G dialout susieQ`  
3. Restart Ubuntu
4. Try :code:`id` command again

    - Repeat step one
	
OpenBCI Cyton Latency Fix
------------------------
If you are using the OpenBCI Cyton with the included USB dongle, the default COM config has to be changed to fix latency issues.   
The default config for the dongle driver sends very big data-packets relatively slowly. The fix is pretty simple, just drop the packet size.    
To do so:  

1. Open device-manager
2. Find the dongle driver under the ports dropdown
3. Go to properties for this com port
4. Go to port-settings
5. Select Advanced
6. Reduce the recieve buffer to 1024 Bytes
7. Reduce the latency timer to 6ms
8. Apply and reboot

FrameRate Check
---------------
For rapid visual stimulation BCI (like the noisetagging BCI), it is very important that the visual flicker be displayed accurately.
However, as the graphics performance of computers varies widely it is hard to know in advance if a particular configuration is accurate enough. 
To help with this we also provide a graphics performance checker, which will validate that your graphics system is correctly configured. 
You can run this with::

	python3 -m mindaffectBCI.examples.presentation.framerate_check
	
As this runs it will show in a window your current graphics frame-rate and, more importantly, the variability in the frame times.
For good BCI performance this jitter should be <1ms. If you see jitter greater than this you should probably adjust your graphics card settings. 
The most important setting to consider is to be sure that you have `vsync <https://en.wikipedia.org/wiki/Screen_tearing#Vertical_synchronization>`_ turned-on. 
Many graphics cards turn this off by default, as it (in theory) gives higher frame rates for gaming.
However, for our system, frame-rate is less important than exact timing, hence always turn vsync on for visual Brain-Compuber-Interfaces!