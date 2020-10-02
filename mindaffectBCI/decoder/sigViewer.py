#!/usr/bin/env python3
import numpy as np
from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, stim2eventfilt, butterfilt_and_downsample

def sigViewer(ui: UtopiaDataInterface=None, hostname=None, timeout_ms:float=np.inf, center=True, timerange:int=5, nstimulus_lines:int=1, ndata_lines:int=-1, datastep:int=20, stimstep:int=1):
    ''' simple sig-viewer using the ring-buffer for testing '''
    import matplotlib.pyplot as plt

    if ui is None:
        data_preprocessor = butterfilt_and_downsample(order=6, stopband=((0,3),(25,-1)), fs_out=60)#999)
        #data_preprocessor = butterfilt_and_downsample(order=6, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=60)
        ui=UtopiaDataInterface(data_preprocessor=data_preprocessor, send_signalquality=False)#, sample2timestamp='none')
        ui.connect(hostname)

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
    stimulus_linestep = np.arange(nstimulus_lines) * stimstep
    stimulus_lines = plt.plot((xdata[0],xdata[-1]),np.zeros((2,nstimulus_lines))+stimulus_linestep)
    plt.ylim([-stimstep, nstimulus_lines* stimstep])
    plt.grid(True)
    #plt.autoscale(axis='x')

    if False:
        f2=plt.figure(2)
        timing_line = plt.plot(1000*(xdata-np.arange(-xdata.shape[0], 0) / ui.fs))[0]
        plt.xlabel('sample')
        plt.ylabel('error w.r.t. sampling @ {}hz (ts - ts*fs) (ms)'.format(ui.fs))
        plt.ylim(-200,200)
        plt.grid()
        f2.show()

    # start the render loop
    fig.show()
    t0 = ui.getTimeStamp()
    while ui.getTimeStamp() < t0+timeout_ms:

        # exit if figure is closed
        if not plt.fignum_exists(1):
            quit()
        
        # Update the records, at least 100ms wait..
        ui.update(timeout_ms=100, mintime_ms=95)

        # Update the EEG stream
        idx = slice(-int(ui.fs*timerange),None) # final 5s data
        data = ui.data_ringbuffer[idx, :]
        if center:
            data = data - np.mean(data[-int(data.shape[0]*.25):],axis=0)
        xdata = ( data[idx,-1] - data[-1,-1] ) / 1000 # time-position in seconds, relative to last
        for li, ln in enumerate(data_lines):
            ln.set_xdata(xdata)
            ln.set_ydata(data[idx, li] + data_linestep[li])

        if False:
            plt.figure(2)
            timing_line.set_xdata(xdata)
            timing_line.set_ydata(1000*(xdata-np.arange(-xdata.shape[0], 0) / ui.fs))
            f2.canvas.draw()
            f2.canvas.flush_events()

        # Update the Stimulus Stream
        # TODO[]: search backwards instead of assuming the stimulus rate...
        stimulus = ui.stimulus_ringbuffer[idx,:]
        # BODGE: pad with 2 extra 0 events, if haven't had stimulus event for a while..
        if stimulus.shape[0]<2 or stimulus[-1,-1] < data[-1,-1]-100:
            stimulus = np.append(stimulus,np.zeros((2,stimulus.shape[1])),0)
            stimulus[-2,-1] = stimulus[-3,-1]+1 if stimulus.shape[0]>2 else data[-1,-1]-1000*timerange
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

    try:
        matplotlib.use('Qt4Cairo')
    except:
        pass

    sigViewer()
