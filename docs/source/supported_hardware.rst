Supported EEG hardware
======================

The mindaffectBCI can supports different ways to connect to amplifiers allowing us to work with many different amplifiers.   Here we give an overview of the main connection types and the supported amps. 


BrainFlow Connections
+++++++++++++++++++++

The primary amplifier connection type is provided by `brainflow <https://brainflow.org/>`_.  Thus any amplifier supported by brainflow can be used by the mindaffect BCI with a simple configuration file change.  

Internally, we have most experience with:
  * openBCI `Ganglion <https://shop.openbci.com/products/ganglion-board?>`_ with 4 water-based EEG electrodes over the occiptial cortex using our 3d printed `headband <https://mindaffect-bci.readthedocs.io/en/latest/printing_guide.html>`_.
  * openBCI `Cyton <https://shop.openbci.com/products/cyton-biosensing-board-8-channel?variant=38958638542>`_ with 6 to 8 water based EEG electrodes over the occiptial cortex using our 3d printed `headband <https://mindaffect-bci.readthedocs.io/en/latest/printing_guide.html>`_


.. _alternativeAmpRef:

Alternative Amplifiers
----------------------

Brainflow supported
+++++++++++++++++++

This online_bci uses `brainflow <http://brainflow.org>`_ by default for interfacing with the EEG amplifier.  Specificially the file in `examples\\acquisition\\utopia_brainflow.py <https://github.com/mindaffect/pymindaffectBCI/blob/open_source/mindaffectBCI/examples/acquisition/utopia_brainflow.py>`_ is used to setup the brainflow connection.  You can check in this file to see what options are available to configure different amplifiers.   In particular you should setup the `board_id` and and additional parameters as discussed in the `brainflow documentation <https://brainflow.readthedocs.io/en/stable/SupportedBoards.html>`_.

You can specify the configuration for your amplifer in the `acq_args` section of the configuration file `online_bci.json <https://github.com/mindaffect/pymindaffectBCI/blob/open_source/mindaffectBCI/online_bci.json>`_.  For example to specify to use the brainflow simulated board use

.. code-block:: JSON

   "acq_args":{ "board_id":-1}

Or to use the openBCI Cyton on com-port 4 

.. code-block:: JSON

   "acq_args":{ 
       "board_id":0,
       "serial_port":"COM4"
    }


LSL supported
+++++++++++++

If your amplifier supports streaming with the `Lab-Streaming-Layer <https://labstreaminglayer.readthedocs.io/index.html>`_ then directly use this as an acquisition device.  Specificially the file in `examples\\acquisition\\utopia_lsl.py <https://github.com/mindaffect/pymindaffectBCI/blob/open_source/mindaffectBCI/examples/acquisition/utopia_lsl.py>`_ is used to setup the LSL connection.  You can check in this file for detailed (and up-to-date) information on what options are available for the LSL connection.  

You can specify the configuration for your amplifer in the `acq_args` section of the configuration file.  For example to use connect to the 1st LSL device with the EEG datatype use:

.. code-block:: JSON
   "acquisition": "lsl",
   "acq_args":{ "streamtype":"EEG"}

BrainProducts LiveAmp
+++++++++++++++++++++

Thanks to valuable support from BrainProducts, including loan equipment for testing, MindAffectBCI includes 'out-of-the-box' basic support for the BrainProducts `LiveAmp <https://www.brainproducts.com/products_by_type.php?tid=1>`_.  Specificially the file in `examples\\acquisition\\utopia_brainproducts.py <https://github.com/mindaffect/pymindaffectBCI/blob/open_source/mindaffectBCI/examples/acquisition/utopia_brainproducts.py>`_ is used to connect to the amplifier.  You can check in this file for detailed (and up-to-date) information on what options are available for this amplifier.  

Note: To use this amplifier you must:
 1. *first* install the amplifier driver, which you should have recieved along with your amplifier) *and*
 2. have attached your 'key-dongle' to an available USB port on your computer.

You can specify the configuration for the LiveAmp in the `acq_args` section of the configuration file.  For example to use this amp with default configuration use:

.. code-block:: JSON
   "acquisition": "bp"

AntNeuro eego
+++++++++++++

Thanks to valuable support from AntNeuro, including loan equipment for testing, MindAffectBCI includes 'out-of-the-box' basic support for the ANT-NEURO `EEGO <https://www.ant-neuro.com/products/eego_product_family>`_.
Specificially the file in `examples\\acquisition\\utopia_eego.py <https://github.com/mindaffect/pymindaffectBCI/blob/open_source/mindaffectBCI/examples/acquisition/utopia_eego.py>`_ is used to connect to the amplifier.  You can check in this file for detailed (and up-to-date) information on what options are available for this amplifier.  

Note: To use this amplifier you must *first* install the amplifier driver, which you should have recieved along with your amplifier.

You can specify the configuration for the eego in the `acq_args` section of the configuration file.  To use this driver with default config use

.. code-block:: JSON
   "acquisition": "eego"


Other Amplifiers
++++++++++++++++

Alternatively, thanks to valuable support from their developers, we support some non-brainflow amplifiers 'out-of-the-box', specifically;
 * TMSi `Mobita <https://shop.tmsi.com/product-tag/mobita>`_: using `--acquisition mobita`, see `examples\\acquisition\\utopia_mobita.py <https://github.com/mindaffect/pymindaffectBCI/blob/open_source/mindaffectBCI/examples/acquisition/utopia_mobita.py>`_ for the configuration options.

We are also happy to add support for additional amplifiers if EEG makers request it and are willing to provide open-source SDKs and test hardware.

Add your own AMP support
++++++++++++++++++++++++

If you have an amp which is not currently supported, and you have a way of getting raw samples out of it, then you can easily (7 lines of Python!) add support for your device as described in the `Add a new Amplifier <https://mindaffect-bci.readthedocs.io/en/latest/add_a_new_amplifier.html>`_ tutorial.