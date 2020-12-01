#!/usr/bin/env python3
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

import numpy as np
from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, stim2eventfilt, butterfilt_and_downsample
   
def run(ui: UtopiaDataInterface=None, host=None, timeout_ms:float=np.inf, 
        center=True, timerange:int=5, nstimulus_lines:int=1, ndata_lines:int=-1, 
        datastep:int=20, stimstep:int=1, stopband=((0,3),(25,-1)), out_fs=60, ch_names=None):
    ''' simple sig-viewer using the ring-buffer for testing '''
    import matplotlib.pyplot as plt

    if ui is None:
        data_preprocessor = butterfilt_and_downsample(order=6, stopband=stopband, fs_out=out_fs)#999)
        #data_preprocessor = butterfilt_and_downsample(order=6, stopband='butter_stopband((0, 5), (25, -1))_fs200.pk', fs_out=60)
        ui=UtopiaDataInterface(data_preprocessor=data_preprocessor, send_signalquality=False)#, sample2timestamp='none')
        ui.connect(host)

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
            # only the non-timestamp channels
            data[:,:-1] = data[:,:-1] - np.mean(data[-int(data.shape[0]*.25):,:-1],axis=0)
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
        # BODGE: pad with 2 extra 0 events, if last stimulus before last data
        if stimulus.shape[0]<2 or stimulus[-1,-1] < data[-1,-1]:
            pad = np.zeros((2,stimulus.shape[1]),dtype=stimulus.dtype)
            pad[0,-1] = stimulus[-1,-1]+1 if stimulus.shape[0]>0 else data[-1,-1]-1000*timerange
            pad[1,-1] = data[-1,-1]
            stimulus = np.append(stimulus,pad,axis=0)
        xdata = ( stimulus[idx,-1] - stimulus[-1,-1] ) / 1000
        for li, ln in enumerate(stimulus_lines):
            ln.set_xdata(xdata)
            ln.set_ydata(stimulus[idx, li] + stimulus_linestep[li])

        # re-draw and sleep for the next one
        fig.canvas.draw()
        fig.canvas.flush_events()

def sigViewer(*args, **kwargs):
    run(*args, **kwargs)

def parse_args():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--out_fs',type=int, help='output sample rate', default=100)
    parser.add_argument('--stopband',type=json.loads, help='set of notch filters to apply to the data before analysis', default=((45,65),(5.5,25,'bandpass')))
    parser.add_argument('--ch_names', type=str, help='list of channel names, or capfile', default=None)
    args = parser.parse_args()

    if args.ch_names:
        args.ch_names = args.ch_names.split(',')

    return args


if __name__=='__main__':
    args = parse_args()
    run(**vars(args))
