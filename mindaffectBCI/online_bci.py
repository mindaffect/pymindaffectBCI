import os
import signal
from multiprocessing import Process
import subprocess 
from time import sleep
import json
import argparse
import traceback

def startHubProcess(label='online_bci', logdir=None):
    """Start the process to manage the central utopia-hub

    Args:
        label (str): a textual name for this process

    Raises:
        ValueError: unrecognised arguments, e.g. acquisition type.

    Returns:
        hub (Process): sub-process for managing the started acquisition driver
    """    
    from mindaffectBCI.decoder import startUtopiaHub
    hub = startUtopiaHub.run(label=label, logdir=logdir)
    #hub = Process(target=startUtopiaHub.run, kwargs=dict(label=label), daemon=True)
    #hub.start()
    sleep(1)
    return hub


def startacquisitionProcess(acquisition, acq_args, label='online_bci', logdir=None):
    """Start the process to manage the acquisition of data from the amplifier

    Args:
        label (str): a textual name for this process
        acquisition (str): the name for the acquisition device to start.  One-of:
                  'none' - do nothing,  
                  'brainflow' - use the mindaffectBCI.examples.acquisition.utopia_brainflow driver
                  'fakedata'- start a fake-data streamer
                  'eego' - start the ANT-neuro eego driver
                  'lsl' - start the lsl EEG sync driver
        acq_args (dict): dictionary of additional arguments to pass to the acquisition device

    Raises:
        ValueError: unrecognised arguments, e.g. acquisition type.

    Returns:
        Process: sub-process for managing the started acquisition driver
    """    
    # start the ganglion acquisition process
    # Using brainflow for the acquisition driver.  
    #  the brainflowargs are kwargs passed to BrainFlowInputParams
    #  so change the board_id and other args to use other boards
    if acquisition == 'none':
        # don't run acq driver here, user will start it manually
        acquisition = None
    elif acquisition == 'fakedata':
        print('Starting fakedata')
        from mindaffectBCI.examples.acquisition import utopia_fakedata
        acq_args=dict(host='localhost', nch=4, fs=200)
        acquisition = Process(target=utopia_fakedata.run, kwargs=acq_args, daemon=True)
        acquisition.start()
    elif acquisition == 'brainflow':
        from mindaffectBCI.examples.acquisition import utopia_brainflow
        if acq_args is None:
            acq_args = dict(board_id=1, serial_port='com3', log=1) # connect to the ganglion
        acquisition = Process(target=utopia_brainflow.run, kwargs=acq_args, daemon=True)
        acquisition.start()

        # give it some time to startup successfully
        sleep(5)
    elif acquisition == 'ganglion': # pyOpenBCI ganglion driver
        from mindaffectBCI.examples.acquisition import utopia_ganglion
        acquisition = Process(target=utopia_ganglion.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition == 'cyton': # pyOpenBCI ganglion driver
        from mindaffectBCI.examples.acquisition import utopia_cyton
        acquisition = Process(target=utopia_cyton.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition == 'javacyton': # java cyton driver
        from mindaffectBCI.examples.acquisition import startJavaCyton
        acquisition = Process(target=startJavaCyton.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition == 'eego': # ANT-neuro EEGO
        from mindaffectBCI.examples.acquisition import utopia_eego
        acquisition = Process(target=utopia_eego.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition == 'lsl': # lsl eeg input stream
        from mindaffectBCI.examples.acquisition import utopia_lsl
        acquisition = Process(target=utopia_lsl.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    elif acquisition == 'brainproducts': # brainproducts eeg input stream
        from mindaffectBCI.examples.acquisition import utopia_brainproducts
        acquisition = Process(target=utopia_brainproducts.run, kwargs=acq_args, daemon=True)
        acquisition.start()

    else:
        raise ValueError("Unrecognised acquisition driver! {}".format(acquisition))
    
    return acquisition


def startDecoderProcess(decoder, decoder_args, label='online_bci', logdir=None):
    """start the EEG decoder process

    Args:
        label (str): a textual name for this process
        decoder (str): the name for the acquisition device to start.  One-of:
                  'decoder' - use the mindaffectBCI.decoder.decoder
                  'none' - don't start a decoder
        decoder_args (dict): dictionary of additional arguments to pass to the decoder
        logdir (str, optional): directory to save log/save files.

    Raises:
        ValueError: unrecognised arguments, e.g. acquisition type.

    Returns:
        Process: sub-process for managing the started decoder
    """    
    if decoder == 'decoder' or decoder == 'mindaffectBCI.decoder.decoder':
        from mindaffectBCI.decoder import decoder
        if decoder_args is None:
            decoder_args = dict(calplots=True)
        if not 'logdir' in decoder_args or decoder_args['logdir']==None: 
            decoder_args['logdir']=logdir
        print('Starting: {}'.format('mindaffectBCI.decoder.decoder'))
        decoder = Process(target=decoder.run, kwargs=decoder_args, daemon=True)
        decoder.start()
        # allow time for the decoder to startup
        sleep(4)
    elif decoder == 'none':
        decoder = None
    return decoder

def run(label='', logdir=None, acquisition=None, acq_args=None, decoder='decoder', decoder_args=None, presentation='selectionMatrix', presentation_args=None):
    """[summary]

    Args:
        label (str, optional): string label for the saved data file. Defaults to ''.
        logdir (str, optional): directory to save log files / data files.  Defaults to None = $installdir$/logs.
        acquisition (str, optional): the name of the acquisition driver to use. Defaults to None.
        acq_args (dict, optional): dictionary of optoins to pass to the acquisition driver. Defaults to None.
        decoder (str, optional): the name of the decoder function to use.  Defaults to 'decoder'.
        decoder_args (dict, optional): dictinoary of options to pass to the mindaffectBCI.decoder.run(). Defaults to None.
        presentation (str, optional): the name of the presentation function to use.  Defaults to: 'selectionMatrix'
        presentation_args (dict, optional): dictionary of options to pass to mindaffectBCI.examples.presentation.selectionMatrix.run(). Defaults to None.

    Raises:
        ValueError: invalid options, e.g. unrecognised acq_driver
    """    
    global hub_proc, acquisition_proc, decoder_proc
    if acquisition is None: 
        acquisition = 'brainflow'

    hub_proc = None
    acquisition_proc = None
    decoder_proc = None
    for retries in range(10):
        #--------------------------- HUB ------------------------------
        # start the utopia-hub process
        if hub_proc is None or not hub_proc.poll() is None:
            try:
                hub_proc = startHubProcess(label=label, logdir=logdir)
            except:
                hub_proc = None
                traceback.print_exc()

        #---------------------------acquisition ------------------------------
        if acquisition_proc is None or not acquisition_proc.is_alive():
            try:
                acquisition_proc = startacquisitionProcess(acquisition, acq_args, label=label, logdir=logdir)
            except:
                acquisition_proc = None
                traceback.print_exc()

        #---------------------------DECODER ------------------------------
        # start the decoder process - with default settings for a noise-tag
        if decoder_proc is None or not decoder_proc.is_alive():
            try:
                decoder_proc = startDecoderProcess(decoder, decoder_args, label=label, logdir=logdir)
            except:
                decoder_proc = None
                traceback.print_exc()

        # terminate if all started successfully
        # check all started up and running..
        component_failed=False
        if hub_proc is None or hub_proc.poll() is not None:
            print("Hub didn't start correctly!")
            component_failed=True
        if acquisition_proc is None or not acquisition_proc.is_alive():
            print("Acq didn't start correctly!")
            component_failed=True
        if decoder_proc is None or not decoder_proc.is_alive():
            print("Decoder didn't start correctly!")
            component_failed=True

        # stop re-starting if all are running fine
        if not component_failed:
            break
        else:
            sleep(1)

    if hub_proc is None or not hub_proc.poll() is None:
        print("Hub didn't start correctly!")
        shutdown(hub_proc,acquisition_proc,decoder_proc)
        raise ValueError("Hub didn't start correctly!")
    if acquisition_proc is None or not acquisition_proc.is_alive():
        print("Acq didn't start correctly!")
        shutdown(hub_proc,acquisition_proc,decoder_proc)
        raise ValueError("acquisition didn't start correctly!")
    if decoder_proc is None or not decoder_proc.is_alive():
        shutdown(hub_proc,acquisition_proc,decoder_proc)
        raise ValueError("Decoder didn't start correctly!")

    #--------------------------- PRESENTATION ------------------------------
    # run the stimulus, with our matrix and default parameters for a noise tag
    #  Make a custom matrix to show:
    if presentation == 'selectionMatrix' or presentation == 'mindaffectBCI.examples.presentation.selectionMatrix':
        if presentation_args is None:
            presentation_args = dict(symbols= [['Hello', 'Good bye'], 
                                               ['Yes',   'No']])
        from mindaffectBCI.examples.presentation import selectionMatrix
        try:
            selectionMatrix.run(**presentation_args)
        except:
            traceback.print_exc()

    elif presentation == 'sigviewer':
        from mindaffectBCI.decoder.sigViewer import sigViewer
        try:
            sigViewer()
        except:
            traceback.print_exc()

    elif isinstance(presentation,str):
        try:
            import importlib
            pres = importlib.import_module(presentation)
            pres.run(**presentation_args)
        except:
            print("Error: could not run the presentation method")
            traceback.print_exc()
    
    elif presentation is None or presentation is False:
        print('No presentation specified.  Running in background!  Be sure to terminate with `mindaffectBCI.online_bci.shutdown()`')
        return

    # TODO []: pop-up a monitoring object / dashboard!

    #--------------------------- SHUTDOWN ------------------------------
    # shutdown the background processes
    shutdown(hub_proc, acquisition_proc, decoder_proc)


def check_is_running(hub=None, acquisition=None, decoder=None):
    """check if the background processes are still running

    Args:
        hub_proc ([type], optional): the hub subprocess. Defaults to hub_proc.
        acquisition_proc ([type], optional): the acquisation subprocess. Defaults to acquisition_proc.
        decoder_proc ([type], optional): the decoder subprocess. Defaults to decoder_proc.

    Returns:
        bool: true if all are running else false
    """
    # use module globals if not given?
    if hub is None: 
        global hub_proc
        hub = hub_proc
    if acquisition is None:
        global acquisition_proc
        acquisition = acquisition_proc
    if decoder is None:
        global decoder_proc
        decoder = decoder_proc

    isrunning=True
    if hub is None or not hub.poll() is None:
        isrunning=False
        print("Hub is dead!")
    if acquisition is None or not acquisition.is_alive():
        print("Acq is dead!")
        isrunning=False
    if decoder is None or not decoder.is_alive():
        print("Decoder is dead!")
        isrunning=False
    return isrunning

def shutdown(hub=None, acquisition=None, decoder=None):    
    """shutdown any background processes started for the BCI

    Args:
        hub (subprocess, optional): handle to the hub subprocess object. Defaults to hub_proc.
        acquisition (subprocess, optional): the acquisatin subprocess object. Defaults to acquisition_proc.
        decoder (subprocess, optional): the decoder subprocess object. Defaults to decoder_proc.
    """    
    # use module globals if not given?
    if hub is None: 
        global hub_proc
        hub = hub_proc
    if acquisition is None:
        global acquisition_proc
        acquisition = acquisition_proc
    if decoder is None:
        global decoder_proc
        decoder = decoder_proc

    hub.terminate()

    try: 
        decoder.terminate()
        decoder.join()
    except:
        pass
    try:
        acquisition.terminate()
        acquisition.join()
    except:
        pass
    
    hub.wait()
#    if os.name == 'nt': # hard kill
#        subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=hub_proc.pid))
#    else: # hard kill
#        os.kill(hub_proc.pid, signal.SIGTERM)
    #print('exit online_bci')


def load_config(config_file):
    """load an online-bci configuration from a json file

    Args:
        config_file ([str, file-like]): the file to load the configuration from. 
    """    
    from mindaffectBCI.decoder.utils import search_directories_for_file
    if isinstance(config_file,str):
        # search for the file in py-dir if not in CWD
        if not os.path.splitext(config_file)[1] == '.json':
            config_file = config_file + '.json'
        config_file = search_directories_for_file(config_file,os.path.dirname(os.path.abspath(__file__)))
        print("Loading config from: {}".format(config_file))
        with open(config_file,'r') as f:
            config = json.load(f)
    else:
        print("Loading config from: {}".format(f))
        config = json.load(f)

    # set the label from the config file
    if 'label' not in config or config['label'] is None:
        # get filename without path or ext
        config['label'] = os.path.splitext(os.path.basename(config_file))[0]
        
    return config


def parse_args():
    """ load settings from the json config-file, parse command line arguments, and merge the two sets of settings.

    Returns:
        NameSpace: the combined arguments name-space
    """    
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, help='user label for the data savefile', default=None)
    parser.add_argument('--config_file', type=str, help='JSON file with default configuration for the on-line BCI', default='debug')#'online_bci.json')
    parser.add_argument('--acquisition', type=str, help='set the acquisition driver type: one-of: "none","brainflow","fakedata","ganglion","eego"', default=None)
    parser.add_argument('--acq_args', type=json.loads, help='a JSON dictionary of keyword arguments to pass to the acquisition system', default=None)
    parser.add_argument('--decoder', type=str, help='set eeg decoder function to use. one-of: "none", "decoder"', default=None)
    parser.add_argument('--decoder_args', type=json.loads, help='set JSON dictionary of keyword arguments to pass to the decoder. Note: need to doublequote the keywords!', default=None)
    parser.add_argument('--presentation', type=str, help='set stimulus presentation function to use: one-of: "none","selectionMatrix"', default=None)
    parser.add_argument('--presentation_args', type=json.loads, help='set JSON dictionary of keyword arguments to pass to the presentation system', default=None)
    parser.add_argument('--logdir', type=str, help='directory where the BCI output files will be saved. Uses $installdir$/logs if None.', default=None)

    args = parser.parse_args()

    # load config-file
    if args.config_file is not None:
        config = load_config(args.config_file)

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

    return args

# N.B. we need this guard for multiprocessing on Windows!
if __name__ == '__main__':
    args = parse_args()
    run(label=args.label, logdir=args.logdir, acquisition=args.acquisition, acq_args=args.acq_args, 
                          decoder=args.decoder, decoder_args=args.decoder_args, 
                          presentation=args.presentation, presentation_args=args.presentation_args)
