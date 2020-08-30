from multiprocessing import Process
from time import sleep

def run(acq_driver=None, acq_args=None, decoder_args=None, presentation_args=None):
    if acq_driver is None: 
        acq_driver = 'brainflow'

    #--------------------------- HUB ------------------------------
    # start the utopia-hub process
    from mindaffectBCI.decoder import startUtopiaHub
    hub = Process(target=startUtopiaHub.run, daemon=True)
    hub.start()

    #---------------------------ACQUISATION ------------------------------
    # start the ganglion acquisation process
    # Using brainflow for the acquisation driver.  
    #  the brainflowargs are kwargs passed to BrainFlowInputParams
    #  so change the board_id and other args to use other boards
    if acq_driver == 'fakedata':
        from mindaffectBCI.examples.acquisation import utopia_fakedata
        acq_args=dict(host='localhost', nch=4, fs=200)
        acquisation = Process(target=utopia_fakedata.run, kwargs=acq_args, daemon=True)
        acquisation.start()
    elif acq_driver == 'brainflow':
        from mindaffectBCI.examples.acquisation import utopia_brainflow
        if acq_args is None:
            acq_args = dict(board_id=1, serial_port='com3') # connect to the ganglion
        acquisation = Process(target=utopia_brainflow.run, kwargs=acq_args, daemon=True)
        acquisation.start()
        # wait for driver to startup -- N.B. NEEDED!!
        sleep(1)
    elif acq_driver == 'ganglion': # pyOpenBCI ganglion driver
        from mindaffectBCI.examples.acquisation import utopia_ganglion
        acquisation = Process(target=utopia_ganglion.run, kwargs=acq_args, daemon=True)
        acquisation.start()
    elif acq_driver == 'eego': # ANT-neuro EEGO
        from mindaffectBCI.examples.acquisation import utopia_eego
        acquisation = Process(target=utopia_eego.run, kwargs=acq_args, daemon=True)
        acquisation.start()


    #---------------------------DECODER ------------------------------
    # start the decoder process - with default settings for a noise-tag
    from mindaffectBCI.decoder import decoder
    if decoder_args is None:
        decoder_args = dict(calplots=True)
    decoder = Process(target=decoder.mainloop, kwargs=decoder_args, daemon=True)
    decoder.start()

    #--------------------------- PRESENTATION ------------------------------
    # run the stimulus, with our matrix and default parameters for a noise tag
    #  Make a custom matrix to show:
    if presentation_args is None:
        presentation_args = dict(symbols= [['Hello', 'Good bye'], 
                                            ['Yes',   'No']])
    from mindaffectBCI.examples.presentation import selectionMatrix
    selectionMatrix.run(**presentation_args)

    #--------------------------- SHUTDOWN ------------------------------
    # shutdown the background processes
    decoder.terminate()
    hub.terminate()
    acquisation.terminate()

def noisetag(acq_driver=None, acq_args=None, decoder_args=None, presentation_args=None):
    run(acq_driver=acq_driver, acq_args=acq_args, decoder_args=decoder_args, presentation_args=presentation_args)

def p300(acq_driver=None, acq_args=None, decoder_args=None, presentation_args=None):
    #---------------------------DECODER ------------------------------
    # start the decoder process - with settings for p300 data.

    if decoder_args is None:
        # Pre-processing:
        #  filter = 1-12Hz -> stop band 0-1, 12-inf
        #  downsample = 30hz
        from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, butterfilt_and_downsample
        stopband = ((0, 1), (12, -1))
        fs_out = 30
        ppfn = butterfilt_and_downsample(order=6, stopband=stopband, fs_out=fs_out)
        ui = UtopiaDataInterface(data_preprocessor=ppfn) 

        # Classifier:
        #   * response length 700ms (as the p300 is from 300-600ms)
        tau_ms = 700
        #   * start of target stimulus vs. start of any stimuls
        #       -> 'rising-edge' and 'non-target rising edge'
        evtlabs = ('re', 'ntre')
        #   * rank-3 decomposition, as we tend to get multiple component, e.g. perceptual, P3a, P3b
        rank = 3
        #  CCA classifier
        from mindaffectBCI.decoder.model_fitting import MultiCCA
        clsfr = MultiCCA(tau=int(fs_out*tau_ms/1000), rank=rank, evtlabs=evtlabs)

        # setup the decoder args
        decoder_args = dict( ui=ui, clsfr=clsfr)

    #--------------------------- PRESENTATION ------------------------------
    if presentation_args is None:
        # with the standard 4x4 letter matrix as the symbol file
        symbol_file = 'symbols.txt'
        # with the row-column stimulus sequence for a 5x5 matrix
        stimulus_file = 'rc5x5.png'
        # and with 4 frames / bit to slow the stimulus down to 60/4 = 15hz
        framesperbit = 4
        presentation_args = dict(symbol_file=symbol_file, stimulus_file=stimulus_file, frameperbits=framesperbit)

    run(acq_driver=acq_driver, acq_args=acq_args, decoder_args=decoder_args, presentation_args=presentation_args)

def ssvep(acq_driver=None, acq_args=None, decoder_args=None, presentation_args=None):
    #---------------------------DECODER ------------------------------
    if decoder_args is None:
        # Pre-processing:
        #  filter = 1-50Hz -> stop band 0-1, 12-inf
        #  downsample = 90hz
        from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, butterfilt_and_downsample
        stopband = ((0, 1), (50, -1))
        fs_out = 90
        ppfn = butterfilt_and_downsample(order=6, stopband=stopband, fs_out=fs_out)
        ui = UtopiaDataInterface(data_preprocessor=ppfn) 

        # Classifier:
        #   * response length 30ms (as the VEP is short)
        tau_ms = 300
        #   * simple stimulus intensity based coding
        evtlabs = ('0', '1')
        #   * rank-1 decomposition, as we tend to get multiple component, e.g. perceptual, P3a, P3b
        rank = 3
        #  CCA classifier
        from mindaffectBCI.decoder.model_fitting import MultiCCA
        clsfr = MultiCCA(tau=int(fs_out*tau_ms/1000), rank=rank, evtlabs=evtlabs)

        # setup the args
        decoder_args = dict(ui=ui, clsfr=clsfr, calplots=True)

    #--------------------------- PRESENTATION ------------------------------
    if presentation_args is None:
        # run the presentation, and the row-column 5x5 stimuls file = p300
        from mindaffectBCI.examples.presentation import selectionMatrix
        # with the standard 4x4 letter matrix as the symbol file
        symbol_file = 'symbols.txt'
        # with the ssvep stimulus for a 5x5 matrix -- all possible frequencies and phases for a 60hz display
        stimulus_file = 'ssvep.txt'
        # and with 1 frames / bit to allow up to 30hz ssvep
        framesperbit = 1

        # setup the runtime arguments
        presentation_args = dict(symbols=symbol_file, stimfile=stimulus_file, framesperbit=framesperbit)

    run(acq_driver=acq_driver, acq_args=acq_args, decoder_args=decoder_args, presentation_args=presentation_args)

def parse_args():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--acq_driver', type=str, help='set the acquisation driver type: one-of: "brainflow","fakedata"', default=None)
    parser.add_argument('--bcitype', type=str, help='set the type of BCI to run, one-of: noisetag, p300, ssvep', default='noisetag')
    parser.add_argument('--config_file', type=str, help='JSON file with default configuration for the on-line BCI', default='online_bci.json')
    parser.add_argument('--acq_args', type=json.loads, help='a JSON dictionary of keyword arguments to pass to the acquisation system', default=None)
    parser.add_argument('--decoder_args', type=json.loads, help='set JSON dictionary of keyword arguments to pass to the decoder. Note: need to doublequote the keywords!', default=None)
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

        # insert into the parser args -- overriding with command line bits if needed
        if 'acq_driver' in config:
            if args.acq_driver is None:
                args.acq_driver = config['acq_driver']                
        if 'acq_args' in config:
            if args.acq_args is not None:
                config['acq_args'].update(args.acq_args)
            args.acq_args = config['acq_args']
        if 'decoder_args' in config:
            if args.decoder_args :
                config['decoder_args'].update(args.decoder_args)
            args.decoder_args = config['decoder_args']
        if 'presentation_args' in config:
            if args.presentation_args:
                config['presentation_args'].update(args.presentation_args)
            args.presentation_args = config['presentation_args']
    return args

# N.B. we need this guard for multiprocessing on Windows!
if __name__ == '__main__':
    args = parse_args()
    if args.bcitype == 'noisetag':
        noisetag(acq_driver=args.acq_driver, acq_args=args.acq_args, decoder_args=args.decoder_args, presentation_args=args.presentation_args )
    elif args.bcitype == 'p300':
        p300(acq_driver=args.acq_driver, acq_args=args.acq_args, decoder_args=args.decoder_args, presentation_args=args.presentation_args)
    elif args.bcitype == 'ssvep':
        ssvep(acq_driver=args.acq_driver, acq_args=args.acq_args, decoder_args=args.decoder_args, presentation_args=args.presentation_args)
    else:
        raise ValueError("Unknown BCItype")