#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
# This file is part of pymindaffectBCI <https://github.com/mindaffect/pymindaffectBCI>.
#
# pymindaffectBCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pymindaffectBCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pymindaffectBCI.  If not, see <http://www.gnu.org/licenses/>

import os
import signal
from multiprocessing import Process
import subprocess 
from time import sleep
import json
import argparse
import traceback

def startHubProcess(label, logdir=None):
    """Start the process to manage the central utopia-hub

    Args:
        label (str): a textual name for this process

    Raises:
        ValueError: unrecognised arguments, e.g. acquisation type.

    Returns:
        hub (Process): sub-process for managing the started acquisation driver
    """    
    from mindaffectBCI.decoder import startUtopiaHub
    hub = startUtopiaHub.run(label=label, logdir=logdir)
    #hub = Process(target=startUtopiaHub.run, kwargs=dict(label=label), daemon=True)
    #hub.start()
    sleep(1)
    return hub



def startAcquisationProcess(label, acquisation, acq_args, logdir=None):
    """Start the process to manage the acquisation of data from the amplifier

    Args:
        label (str): a textual name for this process
        acquisation (str): the name for the acquisation device to start.  One-of:
                  'none' - do nothing,  
                  'brainflow' - use the mindaffectBCI.examples.acquisation.utopia_brainflow driver
                  'fakedata'- start a fake-data streamer
                  'eego' - start the ANT-neuro eego driver
                  'lsl' - start the lsl EEG sync driver
        acq_args (dict): dictionary of additional arguments to pass to the acquisation device

    Raises:
        ValueError: unrecognised arguments, e.g. acquisation type.

    Returns:
        Process: sub-process for managing the started acquisation driver
    """    
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
            acq_args = dict(board_id=1, serial_port='com3', log=1) # connect to the ganglion
        acquisation = Process(target=utopia_brainflow.run, kwargs=acq_args, daemon=True)
        acquisation.start()

        # give it some time to startup successfully
        sleep(5)
    elif acquisation == 'ganglion': # pyOpenBCI ganglion driver
        from mindaffectBCI.examples.acquisation import utopia_ganglion
        acquisation = Process(target=utopia_ganglion.run, kwargs=acq_args, daemon=True)
        acquisation.start()

    elif acquisation == 'cyton': # pyOpenBCI ganglion driver
        from mindaffectBCI.examples.acquisation import utopia_cyton
        acquisation = Process(target=utopia_cyton.run, kwargs=acq_args, daemon=True)
        acquisation.start()

    elif acquisation == 'javacyton': # java cyton driver
        from mindaffectBCI.examples.acquisation import startJavaCyton
        acquisation = Process(target=startJavaCyton.run, kwargs=acq_args, daemon=True)
        acquisation.start()

    elif acquisation == 'eego': # ANT-neuro EEGO
        from mindaffectBCI.examples.acquisation import utopia_eego
        acquisation = Process(target=utopia_eego.run, kwargs=acq_args, daemon=True)
        acquisation.start()

    elif acquisation == 'lsl': # lsl eeg input stream
        from mindaffectBCI.examples.acquisation import utopia_lsl
        acquisation = Process(target=utopia_lsl.run, kwargs=acq_args, daemon=True)
        acquisation.start()

    elif acquisation == 'brainproducts': # brainproducts eeg input stream
        from mindaffectBCI.examples.acquisation import utopia_brainproducts
        acquisation = Process(target=utopia_brainproducts.run, kwargs=acq_args, daemon=True)
        acquisation.start()

    else:
        raise ValueError("Unrecognised acquisation driver! {}".format(acquisation))
    
    return acquisation

def startDecoderProcess(label,decoder,decoder_args, logdir=None):
    """start the EEG decoder process

    Args:
        label (str): a textual name for this process
        decoder (str): the name for the acquisation device to start.  One-of:
                  'decoder' - use the mindaffectBCI.decoder.decoder
                  'none' - don't start a decoder
        decoder_args (dict): dictionary of additional arguments to pass to the decoder
        logdir (str, optional): directory to save log/save files.

    Raises:
        ValueError: unrecognised arguments, e.g. acquisation type.

    Returns:
        Process: sub-process for managing the started decoder
    """    
    if decoder == 'decoder' or decoder == 'mindaffectBCI.decoder.decoder':
        from mindaffectBCI.decoder import decoder
        if decoder_args is None:
            decoder_args = dict(calplots=True)
        if not 'logdir' in decoder_args or decoder_args['logdir']==None: 
            decoder_args['logdir']=logdir
        decoder = Process(target=decoder.run, kwargs=decoder_args, daemon=True)
        decoder.start()
        # allow time for the decoder to startup
        sleep(4)
    elif decoder == 'none':
        decoder = None
    return decoder

def run(label='', logdir=None, acquisation=None, acq_args=None, decoder='decoder', decoder_args=None, presentation='selectionMatrix', presentation_args=None):
    """[summary]

    Args:
        label (str, optional): string label for the saved data file. Defaults to ''.
        logdir (str, optional): directory to save log files / data files.  Defaults to None = $installdir$/logs.
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

    hub_proc = None
    acquisation_proc = None
    decoder_proc = None
    for retries in range(10):
        #--------------------------- HUB ------------------------------
        # start the utopia-hub process
        if hub_proc is None or not hub_proc.poll() is None:
            try:
                hub_proc = startHubProcess(label, logdir=logdir)
            except:
                hub_proc = None
                traceback.print_exc()

        #---------------------------ACQUISATION ------------------------------
        if acquisation_proc is None or not acquisation_proc.is_alive():
            try:
                acquisation_proc = startAcquisationProcess(label, acquisation, acq_args, logdir=logdir)
            except:
                acquisation_proc = None
                traceback.print_exc()

        #---------------------------DECODER ------------------------------
        # start the decoder process - with default settings for a noise-tag
        if decoder_proc is None or not decoder_proc.is_alive():
            try:
                decoder_proc = startDecoderProcess(label, decoder, decoder_args, logdir=logdir)
            except:
                decoder_proc = None
                traceback.print_exc()

        # terminate if all started successfully
        # check all started up and running..
        component_failed=False
        if hub_proc is None or hub_proc is not None and not hub_proc.poll() is None:
            print("Hub didn't start correctly!")
            component_failed=True
        if acquisation_proc is None or acquisation_proc is not None and not acquisation_proc.is_alive():
            print("Acq didn't start correctly!")
            component_failed=True
        if decoder_proc is None or decoder_proc is not None and not decoder_proc.is_alive():
            print("Decoder didn't start correctly!")
            component_failed=True

        # stop re-starting if all are running fine
        if not component_failed:
            break
        else:
            sleep(1)

    if hub_proc is None or hub_proc is not None and not hub_proc.poll() is None:
        print("Hub didn't start correctly!")
        shutdown(hub_proc,acquisation_proc,decoder_proc)
        raise ValueError("Hub didn't start correctly!")
    if acquisation_proc is None or acquisation_proc is not None and not acquisation_proc.is_alive():
        print("Acq didn't start correctly!")
        shutdown(hub_proc,acquisation_proc,decoder_proc)
        raise ValueError("Acquisation didn't start correctly!")
    if decoder_proc is None or decoder_proc is not None and not decoder_proc.is_alive():
        shutdown(hub_proc,acquisation_proc,decoder_proc)
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

    elif presentation == 'none':
        from mindaffectBCI.decoder.sigViewer import sigViewer
        try:
            sigViewer()
        except:
            traceback.print_exc()

    else:
        try:
            import importlib
            pres = importlib.import_module(presentation)
            pres.run(**presentation_args)
        except:
            print("Error: could not run the presentation method")
            traceback.print_exc()

    # TODO []: pop-up a monitoring object / dashboard!

    #--------------------------- SHUTDOWN ------------------------------
    # shutdown the background processes
    shutdown(hub_proc, acquisation_proc, decoder_proc)


def shutdown(hub,acquisation,decoder):    
    try: 
        decoder.terminate()
        decoder.join()
    except:
        pass
    try:
        acquisation.terminate()
        acquisation.join()
    except:
        pass
    
    hub.terminate()
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
        with open(config_file,'r') as f:
            config = json.load(f)
    else:
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
    parser.add_argument('--acquisation', type=str, help='set the acquisation driver type: one-of: "none","brainflow","fakedata","ganglion","eego"', default=None)
    parser.add_argument('--acq_args', type=json.loads, help='a JSON dictionary of keyword arguments to pass to the acquisation system', default=None)
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
    run(label=args.label, logdir=args.logdir, acquisation=args.acquisation, acq_args=args.acq_args, 
                          decoder=args.decoder, decoder_args=args.decoder_args, 
                          presentation=args.presentation, presentation_args=args.presentation_args)
