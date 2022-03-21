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

    def __str__(self):  # pretty print
        return str(self.sound)


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
class SoundFlashScreen(SelectionGridScreen):
    """variant of SelectionGridScreen which plays a sound dependent on the object id and it's current state
    """

    def __init__(
            self, window, noisetag, symbols, visual_flash: bool = True, state2vol: dict = None, spatialize: bool = False,
            use_audio_timestamp: bool = False, pulsed_background_noise: bool = False, background_noise: str = None,
            background_noise_vol: float = .1, play_delay: float = .05, object_vol_correction: dict = None,
            adaptive_stimulus: bool = True, **kwargs):
        print("use_audio_timestamps={}".format(use_audio_timestamp))
        self.visual_flash, self.spatialize, self.use_audio_timestamp, self.background_noise, self.background_noise_vol, self.pulsed_background_noise, self.play_delay = (
            visual_flash, spatialize, use_audio_timestamp, background_noise, background_noise_vol, pulsed_background_noise, play_delay)
        self.adaptive_stimulus = adaptive_stimulus
        if state2vol is not None:
            self.state2vol = intstr2intkey(state2vol)
        # BODGE: ensure is a 0 level
        if not 0 in self.state2vol:
            print("WARNING: state2vol should include a volume for level 0.")
            self.state2vol[0] = 0
        self.object_vol_correction = intstr2intkey(
            object_vol_correction) if object_vol_correction is not None else dict()

        self.stimtime = None
        super().__init__(window, noisetag, symbols=symbols, **kwargs)

        # create background noise object, with single player as don't need to overlap playback
        self.background_noise = SoundFlash(background_noise, nplayers=1) if background_noise else None

    def init_target(self, symb, x, y, w, h, i, j, font_size: int = None):
        """initialize a sound 'object' to play the indicated sound when instructed

        Args:
            symb (_type_): the symbol name.  This is treated as a string with the sounds to play for different stimulus 'levels' encoded as a '|' separated list of file names, e.g. 'yes.wav|no.wav' would mean play yes.wav for state 0, and play no.wav for state 1.
            x (_type_): x-position on screen
            y (_type_): x-position on screen
            w (_type_): x-position on screen
            h (_type_): x-position on screen
            i (_type_): i-position in the stimulus grid -- used to set the left/right spatialization
            j (_type_): j-position in the stimulus grid.  (ignored)
            font_size (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # make the stimulus object (sound)
        if isinstance(symb, str):
            symb = symb.split("|")
        pos = 0
        if isinstance(self.spatialize, float):
            pos = self.spatialize
        elif self.spatialize == True and len(self.symbols)-1 != 0:
            pos = i*2/(len(self.symbols)-1)-1

        players = SoundFlash(symb[1], pos, media_directories=self.media_directories)

        # make the label
        label = self.init_label(symb[0], x, y, w, h, font_size)
        return players, label

    def get_stimtime(self):
        """get the time the stimulus was played -- using the recorded audio time-stamp if available, or the default time-stamp if not.

        Returns:
            _type_: _description_
        """
        if self.use_audio_timestamp and self.stimtime is not None:
            # override to use the last audio play time
            #print("audio time-stamp")
            stimtime = self.stimtime
        else:
            # fall back on the normal stim-time method
            stimtime = super().get_stimtime()
        return stimtime

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
                 None: 1},  # default..

    def update_object_state(self, idx: int, state):
        """update the indicated object with the indicated state, i.e. play the right sound

        Args:
            idx (int): index into the symbols set for the object (sound) to update it's state info
            state (int|float): the updated state for this object

        Note: for *sound* objects, the actual playing state of the object is *only* changed when it differs from the previous state of this object.
        This means that sounds only start playing when the state is changed.
        """
        # play the sound -- **BUT only on CHANGE!**
        if self.objects[idx] and (self.prev_stimulus_state is None or not self.prev_stimulus_state[idx] == state):
            vol = self.state2vol.get(state, 1)  # N.B. un-recognised levels played at full volume
            # apply the object vol correction, default to no correction
            vol = vol * self.object_vol_correction.get(int(idx), 1)
            vol = min(1.0, vol)  # bounds check vol to 0-1
            if vol > 0:
                self.objects[idx].volume = vol
                self.objects[idx].play(delay=self.play_delay)
            # record the start time on the noise-tag timestamp clock
            self.stimtime = self.noisetag.getTimeStamp() + self.play_delay*1000
            #print('state2vol={} sounds[idx]={}, state={}'.format(self.state2vol, self.objects[idx],state))

            # play the background noise at the same time if pulsed
            if self.pulsed_background_noise and self.background_noise and self.background_noise_vol>0:
                self.background_noise.volume = self.background_noise_vol
                self.background_noise.play(delay=self.play_delay, loops=1)

        if self.visual_flash:
            # flash the label with the given color
            col = self.state2color.get(state, self.state2color[None])
            self.labels[idx].color = col if len(col) == 4 else col + (255,)

        # play the background noise
        if not self.pulsed_background_noise and self.background_noise:
            self.background_noise.volume = self.background_noise_vol
            if not self.background_noise.status == 1:
                self.background_noise.play(delay=self.play_delay, loops=1)

    def is_done(self):
        if self.background_noise and self.isDone:
            self.background_noise.pause()
        return super().is_done()

        # print the player times
        #print("times: {}".format([(p[0],[1],p[2].time) for p in self.get_active_players()]))

    def doPredictionDistribution(self, ptd):
        """adapt the stimlus distribution based on the predicted target dist

        Args:
            ptd (PredictedTargetDist): [description]
        """
        if self.adaptive_stimulus:
            try:
                self.noisetag.get_stimSeq().update_from_predicted_target_dist(ptd)
            except:
                pass




# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
class VideoSoundFlashScreen(SoundFlashScreen):
    """screen which plays short sounds on command whilst playing a *silent* video on the screen

    Args:
        SoundFlashScreen (_type_): _description_
    """
    def __init__(
            self, window, noisetag, symbols, visual_flash: bool = True, state2vol: dict = None, spatialize: bool = False,
            use_audio_timestamp: bool = False, pulsed_background_noise: bool = False, background_noise: str = None,
            background_noise_vol: float = .1, video: str = "sagan.mp4", vid_state: int = None, play_delay: float = .05, 
            **kwargs):
        super().__init__(window, noisetag, symbols, visual_flash, state2vol, spatialize, use_audio_timestamp,
                         pulsed_background_noise, background_noise, background_noise_vol, play_delay, **kwargs)
        self.video, self.vid_state = (video, vid_state)
        self.init_video()

    def init_video(self):
        if self.video is None:
            from mindaffectBCI.decoder.utils import askloadsavefile
            self.video = askloadsavefile(self.media_directories[0], filetypes=(
                ('mp4', '*.mp4'), ('all', '*.*')), title='Choose video file to play')
        if isinstance(self.video, str):
            self.video = self.video.split("|")
        # init player
        self.vidplyr = pyglet.media.Player()
        for v in self.video:
            try:
                vid_path = search_directories_for_file(v,
                                                       os.path.dirname(__file__),
                                                       os.path.join(os.path.dirname(__file__), 'video'),
                                                       *self.media_directories)

                self.vidplyr.queue(pyglet.media.load(vid_path))
                print("Loading: {}".format(vid_path))
            except:
                import traceback
                traceback.print_exc()
                pass

        self.vid_trggr = False
        if self.vidplyr.source:
            self.draw_vid = False
            self.vidplyr.volume = 0

            # attach the video playback to the window
            @self.window.event
            def on_draw():
                # TODO: Auto Video scaling
                if self.draw_vid:
                    # self.vidplyr.texture.blit(x=(self.window.width-960)//2, y=(self.window.height-540)//2,
                    #                         width=960, height=540)
                    self.vidplyr.texture.blit(x=int(self.window.width*.1), y=int(self.window.height)*.3,
                                              width=int(self.window.width*.8), height=int(self.window.height*.5))

    def set_vid_state(self, state: int):
        self.vid_state = state

    def set_vid_trggr(self):
        # Play video manually on call instead of on stimstate (to start playing during baseline period)
        try:
            self.vid_trggr = True
        except:
            print("No video loaded")

    def update_object_state(self, idx: int, state):
        super().update_object_state(idx, state)
        # if we have a video that we can play
        if self.vidplyr and self.vidplyr.source and not self.vidplyr.playing:
            if state == self.vid_state or self.vid_trggr:
                # TODO[]: upate the video location with the current window config
                self.draw_vid = True
                self.vidplyr.play()

    def is_done(self):
        if self.vidplyr and self.isDone:
            self.draw_vid = False
            self.vid_trggr = False
            self.vidplyr.pause()
        return super().is_done()
