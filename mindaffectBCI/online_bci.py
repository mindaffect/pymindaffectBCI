import os
import signal
from multiprocessing import Process
from time import sleep

def run(label='', acquisation=None, acq_args=None, decoder='decoder', decoder_args=None, presentation='selectionMatrix', presentation_args=None):
    """[summary]

    Args:
        label (str, optional): string label for the saved data file. Defaults to ''.
        acquisation (str, optional): the name of the acquisation driver to use. Defaults to None.
        acq_args (dict, optional): dictionary of optoins to pass to the acquisation driver. Defaults to None.
        decoder (str, optional): the name of the decoder function to use.  Defaults to 'decoder'.
        decoder_args (dict, optional): dictinoary of options to pass to the mindaffectBCI.decoder.run(). Defaults to None.
        presentation (str, optional): the name of the presentation function to use.  Defaults to: 'selectionMatrix'
        presentation_args (dict, optional): dictionary of options to pass to mindaffectBCI.examples.presentation.selectionMatrix.run(). Defaults to None.

    Raises:
        ValueError: invalid options, e.g. unrecognised acq_driver
    """    
    if acquisation is None: 
        acquisation = 'brainflow'

    #--------------------------- HUB ------------------------------
    # start the utopia-hub process
    from mindaffectBCI.decoder import startUtopiaHub
    hub = Process(target=startUtopiaHub.run, kwargs=dict(label=label), daemon=True)
    hub.start()
    sleep(1)

    #---------------------------ACQUISATION ------------------------------
    # start the ganglion acquisation process
    # Using brainflow for the acquisation driver.  
    #  the brainflowargs are kwargs passed to BrainFlowInputParams
    #  so change the board_id and other args to use other boards
    if acquisation == 'none':
        # don't run acq driver here, user will start it manually
        acquisation = None
    elif acquisation == 'fakedata':
        print('Starting fakedata')
        from mindaffectBCI.examples.acquisation import utopia_fakedata
        acq_args=dict(host='localhost', nch=4, fs=200)
        acquisation = Process(target=utopia_fakedata.run, kwargs=acq_args, daemon=True)
        acquisation.start()
    elif acquisation == 'brainflow':
        from mindaffectBCI.examples.acquisation import utopia_brainflow
        if acq_args is None:
            acq_args = dict(board_id=1, serial_port='com3') # connect to the ganglion
        acquisation = Process(target=utopia_brainflow.run, kwargs=acq_args, daemon=True)
        acquisation.start()

        # give it some time to startup successfully
        sleep(5)
    elif acquisation == 'ganglion': # pyOpenBCI ganglion driver
        from mindaffectBCI.examples.acquisation import utopia_ganglion
        acquisation = Process(target=utopia_ganglion.run, kwargs=acq_args, daemon=True)
        acquisation.start()
    elif acquisation == 'eego': # ANT-neuro EEGO
        from mindaffectBCI.examples.acquisation import utopia_eego
        acquisation = Process(target=utopia_eego.run, kwargs=acq_args, daemon=True)
        acquisation.start()
    else:
        raise ValueError("Unrecognised acquisation driver! {}".format(acquisation))

    if not acquisation.is_alive():
        raise ValueError("Acquisation didn't start correctly!")

    #---------------------------DECODER ------------------------------
    # start the decoder process - with default settings for a noise-tag
    if decoder == 'decoder' or decoder == 'mindaffectBCI.decoder.decoder':
        from mindaffectBCI.decoder import decoder
        if decoder_args is None:
            decoder_args = dict(calplots=True)
        decoder = Process(target=decoder.run, kwargs=decoder_args, daemon=True)
        decoder.start()
        # allow time for the decoder to startup
        sleep(4)
    elif decoder == 'none':
        decoder = None

    # check all started up and running..
    if hub is not None and not hub.is_alive():
        raise ValueError("Hub didn't start correctly!")
    if acquisation is not None and not acquisation.is_alive():
        print("Acq didn't start correctly!")
        raise ValueError("Acquisation didn't start correctly!")
    if decoder is not None and not decoder.is_alive():
        raise ValueError("Decoder didn't start correctly!")

    #--------------------------- PRESENTATION ------------------------------
    # run the stimulus, with our matrix and default parameters for a noise tag
    #  Make a custom matrix to show:
    if presentation == 'selectionMatrix' or presentation == 'mindaffectBCI.examples.presentation.selectionMatrix':
        if presentation_args is None:
            presentation_args = dict(symbols= [['Hello', 'Good bye'], 
                                               ['Yes',   'No']])
        from mindaffectBCI.examples.presentation import selectionMatrix
        selectionMatrix.run(**presentation_args)
    elif presentation == 'none':
        from mindaffectBCI.decoder.sigViewer import sigViewer
        sigViewer()

    # TODO []: pop-up a monitoring object / dashboard!

    #--------------------------- SHUTDOWN ------------------------------
    # shutdown the background processes
    try: 
        decoder.terminate()
    except:
        pass
    try:
        acquisation.terminate()
    except:
        pass
    
    print('killing hub')
    os.kill(hub.pid, signal.SIGTERM)
    print('exit online_bci')

def parse_args():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, help='user label for the data savefile', default=None)
    parser.add_argument('--config_file', type=str, help='JSON file with default configuration for the on-line BCI', default='online_bci.json')
    parser.add_argument('--acquisation', type=str, help='set the acquisation driver type: one-of: "none","brainflow","fakedata","ganglion","eego"', default=None)
    parser.add_argument('--acq_args', type=json.loads, help='a JSON dictionary of keyword arguments to pass to the acquisation system', default=None)
    parser.add_argument('--decoder', type=str, help='set eeg decoder function to use. one-of: "none", "decoder"', default=None)
    parser.add_argument('--decoder_args', type=json.loads, help='set JSON dictionary of keyword arguments to pass to the decoder. Note: need to doublequote the keywords!', default=None)
    parser.add_argument('--presentation', type=str, help='set stimulus presentation function to use: one-of: "none","selectionMatrix"', default=None)
    parser.add_argument('--presentation_args', type=json.loads, help='set JSON dictionary of keyword arguments to pass to the presentation system', default=None)

    args = parser.parse_args()

    if args.config_file is not None:
        config_file = args.config_file
        # search for the file in py-dir if not in CWD
        import os.path
        if not os.path.isfile(config_file):
            pydir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(pydir, config_file) if os.path.isfile(os.path.join(pydir, config_file)) else config_file

        with open(config_file,'r') as f:
            config = json.load(f)

        # MERGE the config-file parameters with the command-line overrides
        for name in config: # config file parameter
            val = config[name]
            if name in args: # command line override available
                newval = getattr(args, name)
                if newval is None: # ignore not set
                    pass
                elif isinstance(val,dict): # dict, merge with config-file version
                    val.update(newval)
                else: # otherwise just override
                    val = newval
            setattr(args,name,val)
        
        # set label to config file name if label is not set
        if args.label is None:
            # get filename without path or ext
            tmp = os.path.splitext(os.path.basename(config_file))[0]
            setattr(args,'label',tmp)

    return args

# N.B. we need this guard for multiprocessing on Windows!
if __name__ == '__main__':
    args = parse_args()
    run(label=args.label, acquisation=args.acquisation, acq_args=args.acq_args, 
                          decoder=args.decoder, decoder_args=args.decoder_args, 
                          presentation=args.presentation, presentation_args=args.presentation_args)
