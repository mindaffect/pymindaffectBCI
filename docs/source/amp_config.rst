.. _ampref:

Amplifier Setup: openBCI Cyton and Ganglion
===================================================

.. _COMref:
COM port
--------
When using either the OpenBCI Ganglion or Cyton *with an USB-dongle* we have to pass the serial_port argument, to find the serial port in use by your amplifier follow the following instructions:
 
 
On Windows
**********
1. Open Device Manager and unfold Ports(COM&LPT), the com port number is shown behind your used bluetooth adapter.
 
    .. image:: images/comport.jpg
 
Then, in the online_bci configuration file (mindaffectBCI/online_bci.json) you should have: :code:`"serial_port":"COM_X_"`.
Also make sure to set :code:`"board_id":0` to the value that corresponds with your amplifier as specified in the `BrainFlow Docs <https://brainflow.readthedocs.io/en/stable/SupportedBoards.html>`_. 
 
On Mac
*******
1. Open a Terminal session
2. Type: :code:`ls /dev/cu.*`, and look for something like :code:`/dev/cu.usbmodem1` (or similar)::
 
           	$ ls /dev/cu.*
    /dev/cu.Bluetooth-Modem                 	/dev/cu.iPhone-WirelessiAP
    /dev/cu.Bluetooth-PDA-Sync   /dev/cu.usbserial
    /dev/cu.usbmodem1
           	
Then, in the online_bci configuration file (mindaffectBCI/config/online_bci.json) you should define as  :code:`"serial_port":"dev/cu.your_com_name"`.
Also make sure to set :code:`"board_id":0` to the value that corresponds with your amplifier as specified in the `BrainFlow Docs <https://brainflow.readthedocs.io/en/stable/SupportedBoards.html>`_. 
 
 
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
           	
Then, in the online_bci configuration file (mindaffectBCI/online_bci.json) you should define as  :code:`"serial_port":"dev/ttyXXXX"`.	
Also make sure to set :code:`"board_id":0` to the value that corresponds with your amplifier as specified in the `BrainFlow Docs <https://brainflow.readthedocs.io/en/stable/SupportedBoards.html>`_. 
 
Linux Serial Port Permissions
******************************
As explained in the `OpenBCI Docs <https://docs.openbci.com/docs/06Software/01-OpenBCISoftware/GUIDocs>`_, on Linux you need to have permission to access the serial ports of your machine.
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
6. Reduce the receive buffer to 1024 Bytes
7. Reduce the latency timer to 6ms
8. Apply and reboot
