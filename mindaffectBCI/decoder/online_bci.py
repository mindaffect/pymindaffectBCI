import mindaffectBCI.decoder
from multiprocessing import Process
from time import sleep


def noisetag():
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
    from mindaffectBCI.examples.acquisation import utopia_brainflow
    acq_args =dict(board_id=1, serial_port='com3') # connect to the ganglion
    acquisation = Process(target=utopia_brainflow.run, kwargs=acq_args, daemon=True)
    acquisation.start()
    # wait for driver to startup -- N.B. NEEDED!!
    sleep(1)

    #---------------------------DECODER ------------------------------
    # start the decoder process - with default settings for a noise-tag
    from mindaffectBCI.decoder import decoder
    decoder = Process(target=decoder.mainloop, kwargs=dict(calplots=True), daemon=True)
    decoder.start()

    #--------------------------- PRESENTATION ------------------------------
    # run the stimulus, with our matrix and default parameters for a noise tag
    #  Make a custom matrix to show:
    symbols = [['Hello', 'Good bye'], 
               ['Yes',   'No']]
    from mindaffectBCI.examples.presentation import selectionMatrix
    selectionMatrix.run(symbols=symbols)

    #--------------------------- SHUTDOWN ------------------------------
    # shutdown the background processes
    decoder.kill()
    hub.kill()
    acquisation.kill()


def p300():
    #--------------------------- HUB ------------------------------
    # start the utopia-hub process
    from mindaffectBCI.decoder import startUtopiaHub
    hub = Process(target=startUtopiaHub.run, daemon=True)
    hub.start()

    #---------------------------ACQUISATION ------------------------------
    # start a fake-data stream
    # with 4-channels running at 200Hz
    from mindaffectBCI.examples.acquisation import utopia_fakedata
    acq_args=dict(host='localhost', nch=4, fs=200)
    acquisation = Process(target=utopia_fakedata.run, kwargs=acq_args, daemon=True)
    acquisation.start()


    #---------------------------DECODER ------------------------------
    # start the decoder process - with settings for p300 data.

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

    from mindaffectBCI.decoder import decoder
    decoder = Process(target=decoder.mainloop, kwargs=dict(ui=ui, clsfr=clsfr, calplots=True), daemon=True)
    decoder.start()

    #--------------------------- PRESENTATION ------------------------------
    # run the presentation, and the row-column 5x5 stimuls file = p300
    from mindaffectBCI.examples.presentation import selectionMatrix
    # with the standard 4x4 letter matrix as the symbol file
    symbol_file = 'symbols.txt'
    # with the row-column stimulus sequence for a 5x5 matrix
    stimulus_file = 'rc5x5.png'
    # and with 4 frames / bit to slow the stimulus down to 60/4 = 15hz
    framesperbit = 4
    selectionMatrix.run(symbols=symbol_file, stimfile=stimulus_file, framesperbit=framesperbit)

    #--------------------------- SHUTDOWN ------------------------------
    # shutdown the background processes
    decoder.kill()
    acquisation.kill()
    hub.kill()


# N.B. we need this guard for multiprocessing on Windows!
if __name__ == '__main__':
    p300()

    #noisetag()
