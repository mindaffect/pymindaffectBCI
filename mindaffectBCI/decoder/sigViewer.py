#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jadref@gmail.com>
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
from math import log10
from mindaffectBCI.decoder.UtopiaDataInterface import UtopiaDataInterface, butterfilt_and_downsample
from numpy.lib.arraypad import _set_wrap_both
from mindaffectBCI.utopiaclient import SignalQuality, Subscribe, ModeChange
   
def run(ui: UtopiaDataInterface=None, host=None, timeout_ms:float=np.inf, 
        center:bool=True, timerange:float=5, nstimulus_lines:int=1, ndata_lines:int=-1, 
        datastep:int=20, stimstep:int=1, filterband=None, out_fs=None, ch_names=None, savefile:str=None, **kwargs):
    """simple raw real-time signal and stimulus state viewer

    Args:
        ui (UtopiaDataInterface, optional): Interface to the data and stimulus info and ring buffer. Defaults to None.
        host ([type], optional): host IP address for the utopia hub. Defaults to None.
        timeout_ms (float, optional): wait timeout for data from the hub. Defaults to np.inf.
        center (bool, optional): center the data time-series before plotting. Defaults to True.
        timerange (float, optional): the amount of time to show in the viewer, in seconds. Defaults to 5.
        nstimulus_lines (int, optional): number of object IDs to show in the stimulus plot. Defaults to 1.
        ndata_lines (int, optional): number of channel lines to show in the data plot, -1 for all. Defaults to -1.
        datastep (int, optional): [description]. Defaults to 20.
        stimstep (int, optional): [description]. Defaults to 1.
        filterband ([type], optional): filter specification to apply to the data before viewing.  Format as for `butterfilt_and_downsample`. Defaults to None.
        out_fs (int, optional): sample rate after filtering. Defaults to None.
        ch_names ([type], optional): list of channel names for display. Defaults to None.
        savefile (str, optional): file to load the data from for savefile debugging. Defaults to None.
    """    
    import matplotlib.pyplot as plt

    if ui is None:
        data_preprocessor = butterfilt_and_downsample(order=6, filterband=filterband, fs_out=out_fs)#999)
        #data_preprocessor = butterfilt_and_downsample(order=6, filterband='butter_filterband((0, 5), (25, -1))_fs200.pk', fs_out=60)
        if savefile is not None:
            from mindaffectBCI.decoder.FileProxyHub import FileProxyHub
            U = FileProxyHub(savefile,use_server_ts=True)
        else: 
            U = None
        ui=UtopiaDataInterface(U=U, data_preprocessor=data_preprocessor, send_signalquality=True)#, sample2timestamp='none')
        ui.connect(host)
        ui.sendMessage(Subscribe(None, "DEMSNQ"))
    ui.update()

    # initialize the plot window
    fig = plt.figure(1)
    fig.clear()    
    
    # initialize the plot window
    plt.clf()
    dataAx=plt.axes((.05,.25,.9,.7))
    plt.title("EEG \n MODE: idle")
    idx=slice(-int(ui.fs*timerange),None)
    data = ui.data_ringbuffer[idx, :]
    # vertical gap between EEG lines
    data_linestep = np.arange(data[:, :ndata_lines].shape[-1]) * datastep
    # compute the x-point,  i.e. in time. Use these as ring-buffer may not be full yet
    xdata = np.arange(-data.shape[0], 0) / ui.fs
    data_lines = plt.plot(xdata, data[:, :ndata_lines] + data_linestep[np.newaxis,:])
    ylabels = [0]+list(range(data[:, :ndata_lines].shape[1]))
    plt.ylim([-datastep, data[:,:ndata_lines].shape[1] * datastep])
    plt.grid(True)

    stimAx=plt.axes((.05,.05,.9,.1))
    plt.title("Stimulus")
    stimulus = ui.stimulus_ringbuffer[idx, :]
    # vertical gap between stimulus lines
    stimulus_linestep = np.arange(nstimulus_lines) * stimstep
    stimulus_lines = plt.plot((xdata[0],xdata[-1]),np.zeros((2,nstimulus_lines))+stimulus_linestep)
    plt.ylim([-stimstep, nstimulus_lines* stimstep])
    plt.grid(True)
    #plt.autoscale(axis='x')

    # start the render loop
    fig.show()
    t0 = ui.getTimeStamp()
    ui.sendMessage(Subscribe(None, "DEMSNQ"))

    while ui.getTimeStamp() < t0+timeout_ms:

        # exit if figure is closed
        if not plt.fignum_exists(1):
            quit()
        
        # Update the records, at least 100ms wait..
        newmsgs, _, _ =ui.update(timeout_ms=100, mintime_ms=95)

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
        
        # check new messages 
        for msg in newmsgs:
            #update signal quality
            if msg.msgID == SignalQuality.msgID:
                for i, qual in enumerate(msg.signalQuality[:data.shape[-1]]):
                    ylabels[i+1] = f"N2S:{qual:.1f}\nCh:{i+1}"
                    qual = log10(qual)
                    qual = max(0, min(1, qual))
                    qualcolor = (int(qual), int(1-qual), 0)
                    dataAx.get_yticklabels()[i+1].set_color(qualcolor)
                dataAx.set_yticklabels(ylabels)
            # update mode    
            if msg.msgID == ModeChange.msgID:
                dataAx.set_title("EEG \n MODE: {}".format(msg.newmode))

        # Update the Stimulus Stream
        # TODO[]: search backwards instead of assuming the stimulus rate...
        stimulus = ui.stimulus_ringbuffer[idx,:]
        # BODGE: pad with 2 extra 0 events, if last stimulus before last data
        if stimulus.shape[0]<2 or stimulus[-1,-1] < data[-1,-1]:
            pad = np.zeros((2,stimulus.shape[1]),dtype=stimulus.dtype)
            pad[0,-1] = stimulus[-1,-1]+1 if stimulus.shape[0]>0 else data[-1,-1]-1000*timerange
            pad[1,-1] = data[-1,-1]
            stimulus = np.append(stimulus,pad,axis=0)
        maxstim = np.max(stimulus[:,:-1])
        if maxstim > max(stimAx.get_ylim()):
            minstim = np.min(stimulus[:,:-1])
            stimAx.set_ylim((minstim,maxstim))
        xdata = ( stimulus[idx,-1] - stimulus[-1,-1] ) / 1000
        for li, ln in enumerate(stimulus_lines):
            ln.set_xdata(xdata)
            ln.set_ydata(stimulus[idx, li] + stimulus_linestep[li])

        # re-draw and sleep for the next one
        fig.canvas.draw()
        fig.canvas.flush_events()

def parse_args():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--out_fs',type=int, help='output sample rate', default=None)
    parser.add_argument('--filterband',type=json.loads, help='set of notch filters to apply to the data before analysis', default=((45,65),(5.5,25,'bandpass')))
    parser.add_argument('--ch_names', type=str, help='list of channel names, or capfile', default=None)
    parser.add_argument('--savefile',type=str, help='save file to play back throgh the viewer.  Use "askloadsavefile" to get a dialog to choose the save file.',default=None)
    args = parser.parse_args()

    if args.ch_names:
        args.ch_names = args.ch_names.split(',')

    return args


if __name__=='__main__':
    args = parse_args()
    if 1: # quick launch for audio testing "dashboard"
        #setattr(args,'filterband', None)
        #setattr(args,'host', '192.168.253.100') # static IP of experimetn machine on Vonets R1
        setattr(args, 'nstimulus_lines', 4)
    run(**vars(args))
