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
 * _symbols_ : can either by a list-of-lists of the actual text to show, for example for a 2x2 grid of sentences:

    ```
    "presentation_args":{
        "symbols":[ ["I'm happy","I'm sad"], ["I want to play","I want to sleep"] ],
        "stimfile":"mgold_65_6532_psk_60hz.png",
        "framesperbit":1
    }
    ```

    or a file from which to load the set of symbols as a *comma-separated* list of strings like the file [symbols.txt](mindaffectBCI/examples/presentation/symbols.txt).

* _stimfile_ : is a file which contains the stimulus-code to display.  This can either be a text-file with a matrix specified with a white-space separated line per output or a png with the stimulus with outputs in 'x' and time in 'y' like: 

![rc5x5.png](mindaffectBCI/rc5x5.png)

## Change Decoder parameters

The decoder is the core of the BCI at it takes in the raw EEG and stimulus information and generates predictions about which stimulus the user is attending to.  Generating these predictions relies on signal processing and machine learning techniques to learn the best decoding parameters for each user.   However, ensuring best performance means the settings for the decoder should be appropriate for the particular BCI being used.  The default decoder parameters are found in the configuration file [`online_bci.json`](mindaffectBCI/online_bci.json) in the `decoder_args` section, and are setup for a noisetagging BCI.

The default settings for noisetagging are:
```
    "decoder_args":{
        "stopband" : [[0,3],[25,-1]],
        "out_fs" : 80,
        "evtlabs" : ["re","fe"],
        "tau_ms" : 450,
        "calplots" : true,
        "predplots" : false
    },
```

The key parameters here are:
  * `stopband`: this is a [temporal filter](https://en.wikipedia.org/wiki/Filter_(signal_processing)) which is applied as a pre-processing step to the incomming data.  This is important to remove external noise so the decoder can focus on the target brain signals.   Here the filter is specified as a list of [band stop](https://en.wikipedia.org/wiki/Band-stop_filter) filters, which specify which signal frequencies should be suppressed, (where, in classic python fashion -1 indicates the max-possible frequency).  Thus, in this example all frequencies below 3Hz and above 25Hz are removed.
  * `out_fs`: this specifies the post-filtering sampling rate of the data.  This reduces the amount of data which will be processed by the rest of the decoder.  Thus, in this example after filtering the data is re-sampled to 80Hz.  (Note: to avoid []() out_fs should be greater than 2x the maximum frequency passed by the stop-band).
  * `evtlabs`: this specifies the stimulus properties (or event labels) the decoder will try to predict from the brain responses.  The input to the decoder (and the brain) is the raw-stimulus intensity (i.e. it's brightness, or loudness).  However, depending on the task the user is performing, the brain may *not* respond directly to the brightness, but some other property of the stimulus.  For example, in the classic [P300 'odd-ball' BCI](https://en.wikipedia.org/wiki/P300_(neuroscience)#Applications), the brain responds not to the raw intensity, but to the start of *surprising* stimuli.  The design of the P300 matrix-speller BCI means this response happens when the users choosen output 'flashes', or gets bright.  Thus, in the P300 BCI the brain responses to the [rising-edge](https://en.wikipedia.org/wiki/Signal_edge) of the stimulus intensity.   Knowing, exactly what stimulus property the brain is responding to is a well studied neuroscientific research question, with examples including, stimulus-onset (a.k.a. rising-edge, or 're'), stimulus-offset (a.k.a. falling-edge, or 'fe'), stimulus intensity ('flash'), stimulus-duration etc.  Getting the right stimulus-coding is critical for BCI peformance, see [`stim2event.py`](mindaffectBCI/decoder/stim2event.py) for more information on supported event types.
  * `tau_ms`: this specifies the maximum duration of the expected brain response to a triggering event in *milliseconds*.  As with the trigger type, the length of the brian response to a triggering event depends on the type of response expected.  For example for the P300 the response is between 300 and 600 ms after the trigger, whereas for a VEP the response is between 100 and 400 ms.   Ideally, the response window should be as small as possible, so the learning system only gets the brain response, and not a lot of non-response containing noise which could lead the machine learning component to [overfitt](https://en.wikipedia.org/wiki/Overfitting).

# Other BCI types.

We provide exmaple configuration files for 3 basic types of visual BCIs:
 * [`noisetag.json`](mindaffectBCI/noisetag_bci.json) : example for a [noise-tagging](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0133797) (or c-VEP) BCI
 * [`rc5x5.json`](mindaffectBCI/rc5x5_bci.json) : example for a classic visual P300 odd-ball type BCI with row-column stimulus.
 * [`ssvep.json`](mindaffectBCI/ssvep_bci.json) : example for a classic [steady-state-visual-response](https://arxiv.org/abs/2002.01171) BCI.

