#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V.
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

import os
import pyglet
from mindaffectBCI.decoder.utils import search_directories_for_file, intstr2intkey
from mindaffectBCI.presentation.screens.SelectionGridScreen import SelectionGridScreen

# set psychopy audio library + latency mode and import the sound module
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
prefs.hardware['audioLatencyMode'] = 3  # 3 Aggressive exclusive mode
import psychtoolbox as ptb
from psychopy import sound as snd # needs to be imported after setting the prefs


# pre-build the player for this sound
def mkPlayer(sound, pos=None, sampleRate=None):
    """make a spatialized sound player

    Args:
        sound ([type]): [description]
        pos ([type], optional):position, where -1  = left, 0=center, 1=right. Defaults to None.
        sampleRate ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    # always spatialize
    sound, sampleRate = spatializeSound(sound, pos)
    plyr = snd.Sound(sound, preBuffer=-1, sampleRate=sampleRate, stereo=True)
    return plyr


def spatializeSound(sound, pos):
    """spatialize (in stereo) the given sound

    Args:
        sound ([type]): [description]
        pos ([type]): position, where -1  = left, 0=center, 1=right

    Returns:
        [type]: [description]
    """
    import numpy as np
    # load sound if it is  a file location
    if isinstance(sound, str):
        sound = pyglet.media.load(sound, streaming=False)

    sample_rate = sound.audio_format.sample_rate
    sample_size = sound.audio_format.sample_size
    snd_bytes = sound.get_queue_source().get_audio_data(1000000).get_string_data()
    snd_array = np.frombuffer(snd_bytes, dtype=np.int8 if sample_size == 1 else np.int16)
    snd_array = snd_array / snd_array.max()

    # make it stereo and spatialize
    sound = np.zeros((len(snd_array), 2), dtype=snd_array.dtype)  # nsamp,2
    sound[:, 0] = snd_array * min(1, max(0, (-pos+1)/2))  # left channel
    sound[:, 1] = snd_array * min(1, max(0, (pos+1)/2))  # right channel
    return sound, sample_rate


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
class SoundFlash:
    def __init__(self, sound, pos=0, nplayers=5, media_directories: list = None):
        """sound-flash object, which can play multiple overlapping sounds

        Args:
            sound ([type]): sound file to play
            pos (int, optional): [description]. Defaults to 0.
            nplayers (int, optional): number players to make, i.e. max number overlapping stimuli.  Defaults to 5.
        """
        self.nplayers, self.media_directories = (nplayers, media_directories)
        if self.media_directories is None:
            self.media_directories = []
        elif not hasattr(self.media_directories, '__iter__'):
            self.media_directories = [self.media_directories]
        self.sound, self.players = self.init_players(sound, pos)
        self._volume = 0

    def init_players(self, sound, pos=0):
        """setup player objects so we can play sounds fast later.  Basically load from disk and allocate resources on the sound card.

            Note: we make self.nplayers playing objects for each sound so we can play overlapping sounds

        Args:
            sound (_type_): sound to play, or sound-file to load
            pos (fload, optional): stereo position of the sound, -1=left, 1=right. Defaults to 0.
        """
        try:
            sound = search_directories_for_file(sound,
                                                os.path.dirname(__file__),
                                                os.path.join(os.path.dirname(__file__), 'audio'),
                                                os.path.join(os.path.dirname(__file__), '..', 'audio'),
                                                *self.media_directories)
            print("Loading: {}".format(sound))
        except:
            sound = None
            import traceback
            traceback.print_exc()
            pass
        players = [mkPlayer(sound, pos) for i in range(self.nplayers)]
        return sound, players

    def get_idle_player(self):
        """get the next idle (i.e. not currently playing) player from the players pool

        Returns:
            pyglet.media.Player(): idle sound player object
        """
        # grab a player from the pool of media players
        plyr = [p for p in self.players if not p.status == 1]
        return plyr[0] if len(plyr) > 0 else None

    @property
    def volume(self): return self._volume

    @volume.setter
    def volume(self, vol):
        self._volume = vol

    def play(self, delay=None, loops=1):
        """play the sound at the currently set volume

        Args:
            delay (_type_, optional): (IGNORED) time to wait before playing -- (IGNORED). Defaults to None.
            loops (int, optional): (IGNORED) number of times to play the sound -- (IGNORED). Defaults to 1.
        """
        if self._volume == 0:
            return
        plyr = self.get_idle_player()
        plyr.volume = self._volume
        if delay is not None:
            play_time = ptb.GetSecs() + delay
            plyr.play(when=play_time, loops=loops)
        else:
            plyr.play(loops=loops)

    @property
    def status(self):
        # is any player currently active?
        return max([p.status for p in self.players])

    def pause(self):
        # stop all the players
        for p in self.players:
            if p.status == 1:
                p.pause()

    def stop(self):
        # stop all the players
        for p in self.players:
            if p.status==1:
                p.stop()

    def __str__(self):  # pretty print
        return str(self.sound)