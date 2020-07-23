#!/usr/bin/env python3
import numpy as np
from UtopiaDataInterface import UtopiaDataInterface, stim2eventfilt, butterfilt_and_downsample

def sigViewer(ui: UtopiaDataInterface, timeout_ms:float=np.inf, timerange:int=5, nstimulus_lines:int=1, ndata_lines:int=-1, datastep:int=20, stimstep:int=1):
    ''' simple sig-viewer using the ring-buffer for testing '''
    import matplotlib.pyplot as plt
    ui.update()

    # initialize the plot window
    fig = plt.figure(1)
    fig.clear()    
    
    # initialize the plot window
    plt.clf()
    plt.axes((.05,.25,.9,.7))
    plt.title("EEG")
    idx=slice(-int(ui.fs*timerange),None)
    data = ui.data_ringbuffer[idx, :]
    # vertical gap between EEG lines
    data_linestep = np.arange(data[:, :ndata_lines].shape[-1]) * datastep
    # compute the x-point,  i.e. in time. Use these as ring-buffer may not be full yet
    xdata = np.arange(-data.shape[0], 0) / ui.fs
    data_lines = plt.plot(xdata, data[:, :ndata_lines] + data_linestep[np.newaxis,:])
    plt.ylim([-datastep, data[:,:ndata_lines].shape[1] * datastep])
    plt.grid(True)

    plt.axes((.05,.05,.9,.1))
    plt.title("Stimulus")
    stimulus = ui.stimulus_ringbuffer[idx, :]
    # vertical gap between stimulus lines
    stimulus_linestep = np.arange(data[:, :nstimulus_lines].shape[-1]) * stimstep
    stimulus_lines = plt.plot(xdata, stimulus[:, :nstimulus_lines] + stimulus_linestep[np.newaxis,:])
    plt.ylim([-stimstep, stimulus[:, :nstimulus_lines].shape[1] * stimstep])
    plt.grid(True)
    #plt.autoscale(axis='x')

    # start the render loop
    fig.show()
    t0 = ui.getTimeStamp()
    while ui.getTimeStamp() < t0+timeout_ms:

        # exit if figure is closed
        if not plt.fignum_exists(1):
            exit()
        
        # Update the records, at least 100ms wait..
        ui.update(timeout_ms=100, mintime_ms=95)

        # Update the EEG stream
        idx = slice(-int(ui.fs*timerange),None) # final 5s data
        data = ui.data_ringbuffer[idx, :]
        xdata = ( data[idx,-1] - data[-1,-1] ) / 1000 # time-position in seconds
        for li, ln in enumerate(data_lines):
            ln.set_xdata(xdata)
            ln.set_ydata(data[idx, li] + data_linestep[li])

        # Update the Stimulus Stream
        # TODO[]: search backwards instead of assuming the stimulus rate...
        stimulus = ui.stimulus_ringbuffer[idx,:]
        # BODGE: pad with 2 extra 0 events, if haven't had stimulus event for a while..
        if stimulus[-1,-1] < data[-1,-1]-100:
            stimulus = np.append(stimulus,np.zeros((2,stimulus.shape[1])),0)
            stimulus[-2,-1] = stimulus[-3,-1]+1
            stimulus[-1,-1] = data[-1,-1]
        xdata = ( stimulus[idx,-1] - stimulus[-1,-1] ) / 1000
        for li, ln in enumerate(stimulus_lines):
            ln.set_xdata(xdata)
            ln.set_ydata(stimulus[idx, li] + stimulus_linestep[li])

        # re-draw and sleep for the next one
        fig.canvas.draw()
        fig.canvas.flush_events()

if __name__=='__main__':
    import sys
    print("Args: {}".format(sys.argv))
    if len(sys.argv) > 1:
        hostname=sys.argv[1]
    else:
        hostname=None

    #data_preprocessor = butterfilt_and_downsample(order=6, stopband=((0,5),(25,-1)), fs_out=60)
    data_preprocessor = butterfilt_and_downsample(order=6, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=60)
    ui=UtopiaDataInterface(data_preprocessor=data_preprocessor)
    ui.connect(hostname)

    try:
        matplotlib.use('Qt4Cairo')
    except:
        pass

    sigViewer(ui)
