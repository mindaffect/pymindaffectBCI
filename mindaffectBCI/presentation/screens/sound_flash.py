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
import pyglet
from mindaffectBCI.presentation.screens.SelectionGridScreen import SelectionGridScreen
from mindaffectBCI.decoder.utils import search_directories_for_file, intstr2intkey
import os

# pre-build the player for this sound
def mkPlayer(sound, pos=0):
    """make a pyglet player object for an audio source, and set to play once and stop

    Args:
        sound ([type]): [description]
        pos (int, optional): [description]. Defaults to 0. -1 is hard-left, +1 is hard-right

    Returns:
        [type]: [description]
    """    
    plyr = pyglet.media.Player()
    plyr.queue(sound)
    plyr.pause()
    plyr.position = (pos*5,0,0)
    # manual play once stuff, i.e. loop but pause when reach end of stream
    plyr.loop = True
    plyr.set_handler('on_eos',lambda: plyr.pause())
    # Note: below is supposed to work, but does not!!
    #plyr.eos_action = 'pause'
    return plyr


class RawAudioSource(pyglet.media.codecs.StaticMemorySource):
    # simple wrapper class to allow direct creation of audio objects in memory
    def __init__(self, data, channels=None, sample_rate=None, sample_size=None, audio_format=None):
        if audio_format is None:
            audio_format = pyglet.media.codecs.AudioFormat(channels,sample_size,sample_rate)
        super().__init__(data,audio_format)
        self._data = data
    def get_queue_source(self): 
        return pyglet.media.codecs.StaticMemorySource(self._data,self.audio_format)

class SoundFlash:
    def __init__(self,sound,pos=0,nplayers=7, media_directories:list=None):
        """sound-flash object, which can 'flash' (play) by setting volume propety

        Args:
            sound ([type]): [description]
            pos (int, optional): [description]. Defaults to 0.
        """
        self.nplayers, self.media_directories = (nplayers, media_directories)
        if self.media_directories is None: self.media_directories = []
        elif not hasattr(self.media_directories,'__iter__'): self.media_directories=[self.media_directories]
        self.sound, self.players = self.init_players(sound,pos)
        self._volume = 0

    def init_players(self, sound:str, pos:float=0):
        """setup player objects so we can play sounds fast later.  Basically load from disk and allocate resources on the sound card.

            Note: we make self.nplayers playing objects for each sound so we can play overlapping sounds

        Args:
            sound (_type_): sound to play, or sound-file to load
            pos (fload, optional): stereo position of the sound, -1=left, 1=right. Defaults to 0.
        """
        try:
            sndfile = search_directories_for_file(sound, 
                                                os.path.dirname(__file__),
                                                os.path.join(os.path.dirname(__file__),'audio'),
                                                os.path.join(os.path.dirname(__file__),'..','audio'),
                                                *self.media_directories)
            print("Loading: {}".format(sndfile))
            sound = pyglet.media.load(sndfile, streaming=False)
        except:
            sound = None
            import traceback
            traceback.print_exc()
            pass
        players = [ mkPlayer(sound,pos) for i in range(self.nplayers) ]
        return sound, players

    @property
    def status(self):
        return max([p.playing for p in self.players])

    @property
    def volume(self): return self._volume

    def get_idle_player(self):
        """get the next idle (i.e. not currently playing) player from the players pool

        Returns:
            pyglet.media.Player(): idle sound player object
        """
        plyr = [ p for p in self.players if not p.playing ]
        return plyr[0] if len(plyr)>0 else None

    @volume.setter
    def volume(self,vol):
        self._volume = vol
        #if vol == 0 : return # don't play if not audiable
        #plyr = self.get_player()
        #plyr.volume = vol
        #plyr.play()

    def play(self, delay=None, loops=1):
        """play the sound at the currently set volume

        Args:
            delay (_type_, optional): (IGNORED) time to wait before playing -- (IGNORED). Defaults to None.
            loops (int, optional): (IGNORED) number of times to play the sound -- (IGNORED). Defaults to 1.
        """        
        if self._volume == 0 : return
        plyr = self.get_idle_player()
        plyr.volume = self._volume
        plyr.play()

    def pause(self):
        # stop all the players
        for p in self.players:
            if p.playing:
                p.pause()

    def stop(self):
        # stop all the players
        for p in self.players:
            if p.playing:
                p.pause()
                p.seek(0)

    def __str__(self): # pretty print
        return str(self.sound)

    def init_players_stack(self, sounds, spatialize:bool=False):
        """setup a stack of players for a list of sounds

        Args:
            sounds (_type_): list of sounds (filenames or sound objects) to play
            spatialize (bool, optional): flag if we should spatialize the sounds to give them different stereo positions.  If true then the sound positions run from left-to-right with the sound list. Defaults to False.

        Returns:
            tuple(sounds,players): sounds - list of loaded sounds,   players - list of created player objects
        """        
        if isinstance(sounds,str):
            sounds = sounds.split("|")
        players = [None]*len(sounds)
        if len(sounds)>1 :
            for state,sndfile in enumerate(sounds):
                if state==0: continue # state==0 is always no-stimulus
                pos = (state-(len(sounds)-1)/2)*4 if spatialize else 0
                sounds[state], players[state] = self.init_players(sndfile,pos)
        return sounds, players

