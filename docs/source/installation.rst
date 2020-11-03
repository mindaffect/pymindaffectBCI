Installation
=========================
Requirements
------------
- Python (3.x) installation (Suggested: Anaconda_ ) 
- OpenBCI Cyton / Ganglion (supported) amplifier

.. _Anaconda: https://docs.anaconda.com/anaconda/install/

Installing the package
----------------------

To install the code:
  1. Clone or download the pymindaffectBCI repository::

       git clone https://github.com/mindaffect/pymindaffectBCI

  2. Install the necessary bits to your local python path:

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


On Mac:
*******
1. Open a Terminal session
2. Type: :code:`ls /dev/cu.*`, and look for something like :code:`/dev/cu.usbmodem1` (or similar)::

	$ ls /dev/cu.*
    /dev/cu.Bluetooth-Modem		/dev/cu.iPhone-WirelessiAP
    /dev/cu.Bluetooth-PDA-Sync	/dev/cu.usbserial
    /dev/cu.usbmodem1

Then, in the online_bci configuration file (mindaffectBCI/online_bci.json) you should define as  :code:`"serial_port":"dev/cu.your_com_name"`

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