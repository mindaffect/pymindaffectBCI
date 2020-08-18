# MindAffect BCI: 1st Run Quickstart


# Installation

Requirements:

*   Python installation (Suggested:[Installation — Anaconda documentation](https://docs.anaconda.com/anaconda/install/) - Version 3.6.9) 
*   Bluetooth 4.0 dongle - (BLED112)
 

Before installing the mindaffectBCI make sure you have the following packages installed in your python environment:


*   Pyglet (Version 1.3.2)
*   Scipy (Version 1.4.1)
*   Numpy (Version 1.18.1)
*   Matplotlib (Version 3.2.1)
*   Brainflow (Version 3.3.1)

If you are missing one or more of the packages listed above, install them as follows:


1. Open the (anaconda) command prompt
2. `pip install "package name"`

Then, to install the MindAffect BCI: (For now we use a local pip install)


1. Place the supplied pymindaffectBCI folder in a directory of your choice. 
2. Open (or navigate to) the pymindaffectBCI folder in the (anaconda) command prompt.
3. Install the pymindaffectBCI package by typing (including the dot!):

    For anaconda users: `pip install -e .`

    For others: `pip3 install -e .`


# Running the MindAffect BCI software

The system consists of 4 main components as illustrated here:

![mindaffect BCI system architecture](https://github.com/mindaffect/pymindaffectBCI/blob/doc/doc/SystemArchitecture.png "mindaffectBCI system architecture")


To actually run the BCI we need to start each of these components:

*   UtopiaHub: This component is the central server which coordinates all the other pieces, and saves the data for offline analysis
*   Acquisition: This component talks to the *EEG Headset* and streams the data to the Hub
*   Decoder: This component analysis the EEG data to fit the subject specific model and generate predictions
*   Presentation: This component presents the User-Interface to the user, including any BCI specific stimuli which need to be presented. It also selects outputs when the BCI is sufficiently confident and generates the appropriate output. This 1st run tutorial will use the python based Selection Matrix as its presentation component

To launch all these components at once:

1.  Power on the  OpenBCI Ganglion. (toggle on/off button)
2.  Open a command prompt / your Anaconda virtual python environment
3.  Run the BCI by typing: `python -m mindaffectBCI.online_bci`

If all is installed it should start the selection matrix with all the other components in the background.

When the blue light on the ganglion does not turn solid after starting the BCI you most likely have to change the com port for your bluetooth dongle. You can specify the correct serial port to use in the acq_args section of the [`online_bci.json`](mindaffectBCI/online_bci.json) system configuration file.  You can find the used com port as follows:

## On Mac:

1. Open a Terminal session
2. Type: `ls /dev/cu.*`, and look for something like `/dev/cu.usbmodem1` (or similar):

    ```
    $ ls /dev/cu.*
    /dev/cu.Bluetooth-Modem		/dev/cu.iPhone-WirelessiAP
    /dev/cu.Bluetooth-PDA-Sync	/dev/cu.usbserial
    /dev/cu.usbmodem1
    ```

    Then, in the online_bci configuration file [`online_bci.json`](mindaffectBCI/online_bci.json) you should be defined as  `"serial_port":"dev/cu.your_com_name"`


## On Windows:

1. Open Device Manager and unfold Ports(COM&LPT), the com port number is shown behind your used bluetooth adapter. 

    ![alt_text](images/image2.png "image_tooltip")

    Then, in the online_bci file your configuration file [`online_bci.json`](mindaffectBCI/online_bci.json) you should have: `"serial_port":"COM_X_"`



# Using the MindAffect BCI:

Now that the system is up and running, you can go through the following steps to use the BCI!



1. EEG headset Setup

    Prepare a headset such that it follows the [MindAffect headset layout.pdf](https://github.com/mindaffect/Headset/blob/master/MindAffect%20headset%20layout.pdf)) in our Headset repository or prepare the headset delivered with your kit by following [MindAffect headset setup.pdf](https://github.com/mindaffect/Headset/raw/master/MindAffect%20Headset%20Set%20up%20instructions.pdf))

2. Signal Quality

    Check the signal quality by pressing 0 in the main menu. Try to adjust the headset until all electrodes are green, or noise to signal ratio is below 5. 


    You can try to improve the signal for an electrode by pressing it firmly into your head. After releasing pressure, wait a few seconds to see if the signal improves. If not, remove the electrode, and apply more water to the sponge. The sponges should feel wet on your scalp.


    If  the noise to signal ratio does not improve by adjusting the headset, try to distance yourself from power outlets and other electronics.

3. Calibration

    Start calibration by pressing 1 in the main menu. Continue to follow the on-screen instructions.

4. Feedback

    You are now ready to try out the BCI by either selecting Copy-spelling (2) or Free-spelling (1)!

# Going Further

You can run the BCI in different modes by specifying different arguments on the command line.  Or by modifying the basic configuration file  [online_bci.json](mindaffectBCI/online_bci.json)

## Alternative Amplifiers

This online_bci uses [brainflow](http://brainflow.org) by default for interfacing with the EEG amplifier.  Specificially the file in [examples\acquisation\utopia_brainflow.py](mindaffectBCI/examples/acquisation/utopia_brainflow.py) is used to setup the brainflow connection.  You can check in this file to see what options are available to configure different amplifiers.   In particular you should setup the `board_id` and and additional parameters as discussed in the [brainflow documentation](https://brainflow.readthedocs.io/en/stable/SupportedBoards.html).

You can specify the configuration for your amplifer in the `acq_args` section of the configuration file [online_bci.json](mindaffectBCI/online_bci.json).  For example to specify to use a simulated board use:

```
   "acq_args":{ "board_id":-1}
```

Or to use the openBCI Cyton on com-port 4:
```
   "acq_args":{ 
       "board_id":0,
       "serial_port":"COM4"
    }
```

## Alternative BCI types / Stimulus

By default we use the mindaffect NoiseTagging style stimulus with a 25-symbol letter matrix for presentation.  You can easily try different types of stimulus and selection matrices by modifying the `symbols` and `stimfile` in `presentation_args` section of the configuration file [`online_bci.json`](mindaffectBCI/online_bci.json).  Where:
 * _symbols_ : can either by a list-of-lists of the actual text to show, for example:

    ```
    symbols=[['one','two'],['three','four']]
    ```

    or a file from which to load the set of symbols as a *comma-separated* list of strings like the file [symbols.txt](mindaffectBCI/examples/presentation/symbols.txt).

* _stimfile_ : is a file which contains the stimulus-code to display.  This can either be a text-file with a matrix specified with a white-space separated line per output or a png with the stimulus with outputs in 'x' and time in 'y' like: 

![rc5x5.png](mindaffectBCI/rc5x5.png)