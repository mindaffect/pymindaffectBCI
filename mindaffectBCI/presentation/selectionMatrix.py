#!/usr/bin/env python3

# Copyright (c) 2019 MindAffect B.V.
#  Author: Jason Farquhar <jadref@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# get the general noisetagging framework
import pyglet
from tkinter.messagebox import askyesno
from mindaffectBCI.noisetag import Noisetag, sumstats
from mindaffectBCI.decoder.utils import search_directories_for_file, import_and_make_class
from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
from mindaffectBCI.presentation.screens.ExptManagerScreen import ExptManagerScreen
from mindaffectBCI.config_file import load_config, set_args_from_dict, askloadconfigfile


# graphic library
configmsg = None
# global user state for communication between screens etc. 
# N.B. to use in child classes you *must* import this variable directly with : 
# import mindaffectBCI.presentation.selectionMatrix as selectionMatrix
# You can then access this variable and it's state with:
#  selectionMatrix.user_state['my_state'] = 'hello'
user_state = dict()

#------------------------------------------------------------------------
# Initialization: display, utopia-connection
def init_noisetag_and_window(stimseq=None,host:str=None,fullscreen:bool=False,width=1024,height=768):
    """setup the connection to the utopia-hub, and initialize the pyglet display stack

    Args:
        stimseq (_type_, optional): _description_. Defaults to None.
        host (str, optional): _description_. Defaults to None.
        fullscreen (bool, optional): _description_. Defaults to False.
    """
    nt=Noisetag(stimSeq=stimseq, clientid='Presentation:selectionMatrix')
    if host is not None and not host in ('','-'):
        nt.connect(host, queryifhostnotfound=False)

    def flip_callback(window):
        olft=window.lastfliptime
        window.lastfliptime=nt.getTimeStamp() if nt.isConnected else (int(time.perf_counter()*1000) % (1<<31))
        if hasattr(window,'flipstats') and window.flipstats is not None:
            window.flipstats.addpoint(window.lastfliptime-olft)

    # init the graphics system
    window = initPyglet(fullscreen=fullscreen, on_flip_callback=flip_callback, width=width, height=height)
    window.lastfliptime=-1
    window.flipstats = sumstats()
    return nt, window


def run(symbols=None, 
        stimfile:str=None, stimseq:str=None, state2color:dict=None,
        fullscreen:bool=None, windowed:bool=None, width=None, height=None, fullscreen_stimulus:bool=True, host:str=None,       
        calibration_symbols=None, calibration_stimseq:str=None, calibration_screen_args:dict=dict(), #bgFraction=.1,
        drawrate:int=-1, **kwargs):
    """ run the selection Matrix with default settings

    Args:
        ncal (int, optional): number of calibration trials. Defaults to 10.
        npred (int, optional): number of prediction trials at a time. Defaults to 10.
        simple_calibration (bool, optional): flag if we show only a single target during calibration, Defaults to False.
        stimseq ([type], optional): the stimulus file to use for the codes. Defaults to None.
        framesperbit (int, optional): number of video frames per stimulus codebit. Defaults to 1.
        fullscreen (bool, optional): flag if should runn full-screen. Defaults to False.
        fullscreen_stimulus (bool, optional): flag if should run the stimulus (i.e. flicker) in fullscreen mode. Defaults to True.
        simple_calibration (bool, optional): flag if we only show the *target* during calibration.  Defaults to False
        calibration_trialduration (float, optional): flicker duration for the calibration trials. Defaults to 4.2.
        prediction_trialduration (float, optional): flicker duration for the prediction trials.  Defaults to 10.
        calibration_args (dict, optional): additional keyword arguments to pass to `noisetag.startCalibration`. Defaults to None.
        prediction_args (dict, optional): additional keyword arguments to pass to `noisetag.startPrediction`. Defaults to None.
        drawrate (int, optional): desired screen refresh rate.  If -1 run as fast as possible. Defaults to -1.
    """
    # configuration message for logging what presentation is used
    global configmsg
    configmsg = "{}".format(dict(component=__file__, args=locals()))

    global nt, window
    # N.B. init the noise-tag first, so asks for the IP
    if stimfile is None:
        stimfile = 'mgold_61_6521_psk_60hz.txt'
    if stimseq is None:
        stimseq = stimfile
    if calibration_stimseq is None:
        calibration_stimseq = stimseq
    if fullscreen is None and windowed is not None:
        fullscreen = not windowed
    if windowed == True or fullscreen == True:
        fullscreen_stimulus = False

    nt, window = init_noisetag_and_window(stimseq,host,fullscreen,width=width,height=height)

    # the logical arrangement of the display matrix
    if symbols is None:
        symbols=[['a', 'b', 'c', 'd', 'e'],
                 ['f', 'g', 'h', 'i', 'j'],
                 ['k', 'l', 'm', 'n', 'o'],
                 ['p', 'q', 'r', 's', 't'],
                 ['u', 'v', 'w', 'x', '<-']]

    # different calibration symbols if wanted
    if calibration_symbols is None:
        calibration_symbols = symbols
    if state2color is not None: # put in the cal-screen args
        calibration_screen_args['stat2color']=state2color
    # make the screen manager object which manages the app state
    exptscreen = ExptManagerScreen(window, nt, symbols=symbols,
                        calibration_screen_args=calibration_screen_args,
                        fullscreen_stimulus=fullscreen_stimulus, 
                        calibration_symbols=calibration_symbols, 
                        stimseq=stimseq, calibration_stimseq=calibration_stimseq,
                        **kwargs)

    run_screen(window=window, screen=exptscreen, drawrate=drawrate)

def parse_args():
    """parse the command line arguments -- if running from the command line

    Returns:
        _type_: _description_
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncal',type=int, help='number calibration trials', default=argparse.SUPPRESS)
    parser.add_argument('--npred',type=int, help='number prediction trials', default=argparse.SUPPRESS)
    parser.add_argument('--host',type=str, help='address (IP) of the utopia-hub', default=None)
    parser.add_argument('--stimseq',type=str, help='stimulus file to use', default=argparse.SUPPRESS)
    parser.add_argument('--framesperbit',type=int, help='number of video frames per stimulus bit. Default 1', default=argparse.SUPPRESS)
    parser.add_argument('--fullscreen_stimulus',action='store_true',help='run with stimuli in fullscreen mode')
    parser.add_argument('--windowed',action='store_true',help='run in fullscreen mode')
    parser.add_argument('--selectionThreshold',type=float,help='target error threshold for selection to occur. Default .1',default=argparse.SUPPRESS)
    parser.add_argument('--simple_calibration',action='store_true',help='flag to only show a single target during calibration',default=argparse.SUPPRESS)
    parser.add_argument('--symbols',type=str,help='file name for the symbols grid to display',default=argparse.SUPPRESS)
    parser.add_argument('--calibration_symbols',type=str,help='file name for the symbols grid to use for calibration',default=argparse.SUPPRESS)
    parser.add_argument('--extra_symbols',type=str,help='comma separated list of extra symbol files to show',default=argparse.SUPPRESS)
    parser.add_argument('--config_file', type=str, help='JSON file with default configuration for the on-line BCI', default=None)#'debug')#'online_bci.json')
    #parser.add_argument('--artificial_deficit_id',type=int, help='objID for introducing stimulus deficit', default=argparse.SUPPRESS)
    #parser.add_option('-m','--matrix',action='store',dest='symbols',help='file with the set of symbols to display',default=argparse.SUPPRESS)
    args = parser.parse_args()
    if hasattr(args,'extra_symbols') and args.extra_symbols is not None:
        args.extra_symbols = args.extra_symbols.split(',')

    if args.config_file is None:
        config_file = askloadconfigfile()
        setattr(args,'config_file',config_file)

    if args.config_file is not None:
        config = load_config(args.config_file)
        # get the presentation_args
        if 'presentation_args' in config:
            config = config['presentation_args']
        # set them
        args = set_args_from_dict(args,config)

    return args

if __name__ == "__main__":
    args = parse_args()
    # setattr(args,'symbols',[['yes','no','<-']]) #"eog.txt")#
    # #setattr(args,'extra_symbols',['3x3.txt','robot_control.txt'])
    # setattr(args,'stimfile','level8_gold_01.txt')
    # setattr(args,'calibration_stimseq','rc5x5.txt')
    # #setattr(args,'extra_symbols',['prva.txt'])
    # setattr(args,"symbols",[["+|visualacuity/grating1.jpg|visualacuity/grating1_neg.jpg|visualacuity/grating2.jpg|visualacuity/grating2_neg.jpg|visualacuity/grating3.jpg|visualacuity/grating3_neg.jpg|visualacuity/grating4.jpg|visualacuity/grating4_neg.jpg|visualacuity/grating7.jpg|visualacuity/grating7_neg.jpg|visualacuity/grating10.jpg|visualacuity/grating10_neg.jpg"]])
    # setattr(args,"stimfile","6blk_rand_pr.txt")
    # setattr(args,'extra_screens',["mindaffectBCI.examples.presentation.ImageFlashScreen.ImageFlashScreen",
    #                       "mindaffectBCI.examples.presentation.ImageFlashScreen.ImageFlashScreen",
    #                       "mindaffectBCI.examples.presentation.ImageFlashScreen.ImageFlashScreen"])
    # setattr(args,"extra_labels",["rand pr", "sweep pr", "rand"])
    # setattr(args,"extra_stimseqs",["6blk_rand_pr.txt","6blk_sweep_pr.txt","6blk_rand.txt"])

    # setattr(args,"symbols","keyboard.txt")
    # setattr(args,    "calibration_symbols", "3x3.txt")
    # setattr(args,    "extra_symbols", ["emojis.txt","robot_control.txt"])
    # setattr(args,    "stimfile","mgold_65_6532_psk_60hz.png")



    # setattr(args,'fullscreen',False)
    # setattr(args,'calibration_args',{"startframe":"random"})
    # setattr(args,'optosensor',-1)
    # setattr(args,'framesperbit',30)
    run(**vars(args))

