#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V.
#  Author: Jason Farquhar <jason@mindaffect.nl>
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
from mindaffectBCI.utopiaclient import PredictedTargetDist
import mindaffectBCI.presentation.selectionMatrix as selectionMatrix
from mindaffectBCI.examples.presentation.sound_flash_psypy import SoundFlash
#from mindaffectBCI.examples.presentation.sound_flash import SoundFlash
from mindaffectBCI.examples.presentation.image_flash import Checkerboard
from mindaffectBCI.decoder.utils import intstr2intkey

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


class SoundOrCheckerboardStackScreen(selectionMatrix.SelectionGridScreen):
    """variant of SelectionGridScreen which plays a sound dependent on the object id and it's current state
    """

    def __init__(self, window, noisetag, symbols, state2vol: dict = None, state2col: dict = None,
                 object_vol_correction: dict = None, play_audio: bool = True, play_visual: bool = True, 
                 spatialize:bool = False, use_audio_timestamp: bool = False, visual_flash: bool = True, 
                 play_delay: float = .05, adaptive_stimulus:bool=False, **kwargs):
        """[summary]

        Args:
            window ([type]): [description]
            noisetag ([type]): [description]
            symbols ([type]): specification of the objects type and parameters, e.g. audio-file-name for audio, nx,ny for checkerboards
            state2vol (dict, optional): audio-objects, mapping from states to volume. Defaults to None.
            state2col (dict, optional): visual-objects, mapping from states to colors. Defaults to None.
            object_vol_correction (dict, optional): for each audio object a correction factor to apply when computing the final volume.  So played vol = object_vol_correction[objId]*state2vol[state]
            play_audio (bool, optional): [description]. Defaults to True.
            play_visual (bool, optional): [description]. Defaults to True.
            spatialize (bool, optional): [description]. Defaults to False.
            use_audio_timestamp (bool, optional): [description]. Defaults to False.
            visual_flash (bool, optional): [description]. Defaults to True.
            play_delay (float, optional): [description]. Defaults to .05.
        """
        self.visual_flash, self.spatialize, self.use_audio_timestamp, self.play_audio, self.play_visual, self.play_delay, self.adaptive_stimulus = (
            visual_flash, spatialize, use_audio_timestamp, play_audio, play_visual, play_delay, adaptive_stimulus)
        self.state2vol = intstr2intkey(state2vol) if state2vol is not None else dict()
        self.state2col = intstr2intkey(state2col) if state2col is not None else dict()
        self.object_vol_correction = intstr2intkey(
            object_vol_correction) if object_vol_correction is not None else dict()
        self.stimtime = None
        super().__init__(window, noisetag, symbols=symbols, **kwargs)

    def init_symbols(self, symbols, x, y, w, h, bgFraction: float = .1, font_size: int = None):
        """stack the symbols on top of each other in the main screen"""
        x = x + bgFraction*w/2
        y = y + bgFraction*h/2
        w = w - bgFraction*w
        h = h - bgFraction*h
        # now create the display objects
        idx = -1
        for i in range(len(symbols)):  # rows
            for j in range(len(symbols[i])):  # cols
                if symbols[i][j] is None or symbols[i][j] == "":
                    continue
                idx = idx+1
                symb = symbols[i][j]
                self.objects[idx], self.labels[idx] = self.init_target(symb,
                                                                       x, y,
                                                                       w, h, i, j,
                                                                       font_size=font_size)

    def init_target(self, symb, x, y, w, h, i, j, font_size: int = None):
        if isinstance(symb, str):
            symb = symb.split("|")
        lab = symb[0]
        symb = symb[1:]

        # make the right type of stimulus object -- sound if string that starts with audio
        if isinstance(symb[0], str) and symb[0].endswith('.wav'):
            # make the stimulus object (sound)
            pos = 0
            if isinstance(self.spatialize, float):
                pos = self.spatialize
            elif self.spatialize == True:
                pos = (i/(len(self.symbols)-1)) * 2 - 1
            obj = SoundFlash(symb[0], pos, media_directories=self.media_directories)

        # otherwise assume it's a checkerboard spec
        else:
            # extract the desired checkerboard size
            nx, ny = (int(symb[0]), 1) if len(symb) == 1 else [int(s) for s in symb[:2]]
            obj = Checkerboard(x, y, w, h, nx=nx, ny=ny, batch=self.batch, group=self.background)

        # make the label
        label = self.init_label(lab, x, y, w, h, font_size)
        return obj, label

    def get_stimtime(self):
        if self.use_audio_timestamp and self.stimtime is not None:
            # override to use the last audio play time
            #print("audio time-stamp")
            stimtime = self.stimtime
        else:
            # fall back on the normal stim-time method
            stimtime = super().get_stimtime()
        return stimtime

    # mapping from bci-stimulus-states to display color
    state2color = {0: (5, 5, 5, 0),     # off=invisible
                   1: (160, 160, 160),  # on=white
                   2: (96, 96, 96),     # invert
                   254: (0, 255, 0),      # cue=green
                   255: (0, 0, 255),      # feedback=blue
                   None: (100, 0, 0)}     # red(ish)=task

    state2vol = {0: 0,
                 1: 0.0001,
                 2: 0.0005,
                 3: 0.001,
                 4: 0.01,
                 5: 0.1,
                 6: 0.2,
                 7: 0.3,
                 254: 1,  # cue
                 255: 1,  # feedback
                 None: 0},  # default..

    def update_object_state(self, idx: int, state):
        # play the sound -- **BUT only on CHANGE!**
        if self.objects[idx] and (self.prev_stimulus_state is None or not self.prev_stimulus_state[idx] == state):

            if self.play_audio and hasattr(self.objects[idx], 'volume'):  # it's a sound, so set it's vol
                vol = self.state2vol.get(state, 1)  # default to play at full vol!
                # apply the object vol correction, default to no correction
                vol = vol * self.object_vol_correction.get(idx, 1)
                vol = min(1.0, vol)  # bounds check vol to 0-1
                if vol > 0: # only play if non-zero vol
                    self.objects[idx].volume = vol
                    self.objects[idx].play(delay=self.play_delay)
                    print('idx={}, state={}, vol={}'.format(idx,state,vol))
                # record the start time on the noise-tag timestamp clock
                self.stimtime = self.noisetag.getTimeStamp()

            elif self.play_visual:  # assume it's a color change object
                if isinstance(state, int):  # integer state, map to color lookup table
                    color = self.state2color.get(state, self.state2color[None])
                elif isinstance(state, float):  # float state, map to intensity
                    color = tuple(int(c*state) for c in self.state2color[1])
                self.objects[idx].color = color

                self.stimtime = None

    def doPredictionDistribution(self, ptd: PredictedTargetDist):
        """adapt the stimlus distribution based on the predicted target dist

        Args:
            ptd (PredictedTargetDist): [description]
        """
        if self.adaptive_stimulus:
            try:
                self.noisetag.get_stimSeq().update_from_predicted_target_dist(ptd)
            except:
                pass



if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    selectionMatrix.run(**vars(args))
