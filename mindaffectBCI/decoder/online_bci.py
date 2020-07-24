import mindaffectBCI.decoder
from multiprocessing import Process
from time import sleep

# N.B. we need this guard for multiprocessing on Windows!
if __name__=='__main__':
    # start the utopia-hub process
    from mindaffectBCI.decoder import startUtopiaHub
    hub = Process(target=startUtopiaHub.run, daemon=True)
    hub.start()

    # start the ganglion acquisation process
    from mindaffectBCI.examples.acquisation import utopia_brainflow
    kwargs =dict(board_id=1, serial_port='com3') # connect to the ganglion
    acquisation  = Process(target=utopia_brainflow.run, kwargs=kwargs, daemon=True)
    acquisation.start()

    # wait for driver to startup -- N.B. NEEDED!!
    sleep(1)

    # start the decoder process
    from mindaffectBCI.decoder import decoder
    decoder = Process(target=decoder.mainloop, kwargs=dict(calplots=True), daemon=True)
    decoder.start()

    # run the stimulus, with default parameters
    from mindaffectBCI.examples.presentation import selectionMatrix
    selectionMatrix.run(symbols=[['Hello','Good bye'],['Yes','No']])

    # shutdown the background processes
    decoder.kill()
    hub.kill()