.. _goingfurtherRef:

Going Further : BCI-types, Decoder Config
=============

You can run the BCI in different modes by specifying different arguments on the command line.  Or by modifying the basic configuration file  `online_bci.json <https://github.com/mindaffect/pymindaffectBCI/blob/open_source/mindaffectBCI/online_bci.json>`_.


Alternative BCI types / Stimulus
--------------------------------

By default we use the mindaffect NoiseTagging style stimulus with a 25-symbol letter matrix for presentation.  You can easily try different types of stimulus and selection matrices by modifying the `symbols` and `stimfile` in `presentation_args` section of the configuration file `online_bci.json <https://github.com/mindaffect/pymindaffectBCI/blob/open_source/mindaffectBCI/online_bci.json>`_  Where:
 * _symbols_ : can either by a list-of-lists of the actual text to show, for example for a 2x2 grid of sentences.

 .. code-block:: JSON

    "presentation_args":{
        "symbols":[ ["I'm happy","I'm sad"], ["I want to play","I want to sleep"] ],
        "stimfile":"mgold_65_6532_psk_60hz.png",
        "framesperbit":1
    }

 or a file from which to load the set of symbols as a *comma-separated* list of strings like the file `symbols.txt <https://github.com/mindaffect/pymindaffectBCI/blob/open_source/mindaffectBCI/examples/presentation/symbols.txt>`_.

* _stimfile_ : is a file which contains the stimulus-code to display.  This can either be a text-file with a matrix specified with a white-space separated line per output or a png with the stimulus with outputs in 'x' and time in 'y' like.

 You can clearly see the difference between the two main types of BCI stimulus file used here when viewed as an image.   Firstly, this is the stimulus file for the noisecodes.

 .. image :: images/mgold_61_6521_psk_60hz.png

 which clearly shows the noise-like character of this code.   

 By contrast the, classical P300 row-column speller stimulus sequence looks like.

 .. image :: images/rc5x5.png

 which shows the more structured row-column structure, and that only a few outputs are 'on' at any time.
 


Change Decoder parameters
-------------------------

The decoder is the core of the BCI at it takes in the raw EEG and stimulus information and generates predictions about which stimulus the user is attending to.  Generating these predictions relies on signal processing and machine learning techniques to learn the best decoding parameters for each user.   However, ensuring best performance means the settings for the decoder should be appropriate for the particular BCI being used.  The default decoder parameters are found in the configuration file `online_bci.json <mindaffectBCI/online_bci.json>`_ in the `decoder_args` section, and are setup for a noisetagging BCI.

The default settings for noisetagging are

.. code-block:: JSON

    "decoder_args":{
        "stopband" : [3,25,"bandpass"],
        "out_fs" : 80,
        "evtlabs" : ["re","fe"],
        "tau_ms" : 450,
        "calplots" : true,
        "predplots" : false
    }

The key parameters here are:

  * `stopband`: this is a `temporal filter <https://en.wikipedia.org/wiki/Filter_(signal_processing)>`_ which is applied as a pre-processing step to the incomming data.  This is important to remove external noise so the decoder can focus on the target brain signals.   Here the filter is specified as a list of bandpass or `band stop <https://en.wikipedia.org/wiki/Band-stop_filter>`_ filters, which specify which signal frequencies should be suppressed, (where, in classic python fashion -1 indicates the max-possible frequency).  Thus, in this example only frequencies between 3 and 25Hz remain after filtering.

  * `out_fs`: this specifies the post-filtering sampling rate of the data.  This reduces the amount of data which will be processed by the rest of the decoder.  Thus, in this example after filtering the data is re-sampled to 80Hz.  (Note: to avoid []() out_fs should be greater than 2x the maximum frequency passed by the stop-band).

  * `evtlabs`: this specifies the stimulus properties (or event labels) the decoder will try to predict from the brain responses.  The input to the decoder (and the brain) is the raw-stimulus intensity (i.e. it's brightness, or loudness).  However, depending on the task the user is performing, the brain may *not* respond directly to the brightness, but some other property of the stimulus.  For example, in the classic `P300 'odd-ball' BCI <https://en.wikipedia.org/wiki/P300_(neuroscience)#Applications>`_, the brain responds not to the raw intensity, but to the start of *surprising* stimuli.  The design of the P300 matrix-speller BCI means this response happens when the users choosen output 'flashes', or gets bright.  Thus, in the P300 BCI the brain responses to the `rising-edge <https://en.wikipedia.org/wiki/Signal_edge>`_ of the stimulus intensity.   Knowing, exactly what stimulus property the brain is responding to is a well studied neuroscientific research question, with examples including, stimulus-onset (a.k.a. rising-edge, or 're'), stimulus-offset (a.k.a. falling-edge, or 'fe'), stimulus intensity ('flash'), stimulus-duration etc.  Getting the right stimulus-coding is critical for BCI peformance, see `stim2event.py <mindaffectBCI/decoder/stim2event.py>`_ for more information on supported event types.

  * `tau_ms`: this specifies the maximum duration of the expected brain response to a triggering event in *milliseconds*.  As with the trigger type, the length of the brian response to a triggering event depends on the type of response expected.  For example for the P300 the response is between 300 and 600 ms after the trigger, whereas for a VEP the response is between 100 and 400 ms.   Ideally, the response window should be as small as possible, so the learning system only gets the brain response, and not a lot of non-response containing noise which could lead the machine learning component to `overfit <https://en.wikipedia.org/wiki/Overfitting>`_.
