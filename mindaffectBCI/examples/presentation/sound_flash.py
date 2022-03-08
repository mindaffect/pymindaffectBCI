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
import pyglet
import mindaffectBCI.examples.presentation.selectionMatrix as selectionMatrix
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

    def init_players(self, sound, pos=0):
        try:
            sndfile = search_directories_for_file(sound, 
                                                os.path.dirname(__file__),
                                                os.path.join(os.path.dirname(__file__),'audio'),
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
    def volume(self): return self._volume

    def get_idle_player(self):
        # grab a player from the pool of media players
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
        if self._volume == 0 : return
        plyr = self.get_idle_player()
        plyr.volume = self._volume
        plyr.play()

    def __str__(self): # pretty print
        return str(self.sound)

    def init_players_stack(self, sounds, spatialize:bool=False):
        if isinstance(sounds,str):
            sounds = sounds.split("|")
        players = [None]*len(sounds)
        if len(sounds)>1 :
            for state,sndfile in enumerate(sounds):
                if state==0: continue # state==0 is always no-stimulus
                pos = (state-(len(sounds)-1)/2)*4 if spatialize else 0
                sounds[state], players[state] = self.init_players(sndfile,pos)
        return sounds, players


class SoundFlashScreen(selectionMatrix.SelectionGridScreen):
    """variant of SelectionGridScreen which plays a sound dependent on the object id and it's current state
    """    
    def __init__(self, window, noisetag, symbols, state2vol:dict=None, spatialize:bool=False, use_audio_timestamp:bool=False, visual_flash:bool=True, **kwargs):
        print("use_audio_timestamps={}".format(use_audio_timestamp))
        self.visual_flash, self.spatialize, self.use_audio_timestamp = (visual_flash, spatialize, use_audio_timestamp)
        if state2vol is not None:
            self.state2vol = intstr2intkey(state2vol)
        # BODGE: ensure is a 0 level
        if not 0 in self.state2vol:
            print("WARNING: state2vol should include a volume for level 0.") 
            self.state2vol[0]=0
        self.stimtime=None
        super().__init__(window, noisetag, symbols=symbols, **kwargs)

    def init_target(self, symb, x, y, w, h, i, j, font_size:int=None):
        # make the stimulus object (sound)
        if isinstance(symb,str): symb=symb.split("|")
        pos = (i-(len(self.symbols)-1)/2)*4 if self.spatialize else 0
        players = SoundFlash(symb[1],pos,media_directories=self.media_directories)
        
        # make the label
        label= self.init_label(symb[0],x,y,w,h,font_size)
        return players, label

    def get_stimtime(self):
        if self.use_audio_timestamp and self.stimtime is not None: 
            # override to use the last audio play time
            #print("audio time-stamp")
            stimtime = self.stimtime
        else:
            # fall back on the normal stim-time method
            stimtime = super().get_stimtime()
        return stimtime

    state2vol = {0:0,
                 1:0.0001,
                 2:0.0005,
                 3:0.001,
                 4:0.01,
                 5:0.1,
                 6:0.2,
                 7:0.3,
                 254:1, # cue
                 255:1, # feedback
                 None:1}, # default..
    def update_object_state(self, idx:int, state):
        # play the sound -- **BUT only on CHANGE!**
        if self.objects[idx] and (self.prev_stimulus_state is None or not self.prev_stimulus_state[idx]==state):
            vol = self.state2vol.get(state, 1)  # default to play at full vol!
            # apply the object vol correction, default to no correction
            vol = vol * self.object_vol_correction.get(idx, 1)
            vol = min(1.0, vol)  # bounds check vol to 0-1
            if vol > 0 :
                self.objects[idx].volume = vol
                self.objects[idx].play(delay=self.play_delay)
                print('idx={}, state={}, vol={}'.format(idx,state,vol))
            # record the start time on the noise-tag timestamp clock
            self.stimtime = self.noisetag.getTimeStamp()
            # #print('state2vol={} sounds[idx]={}, state={}'.format(self.state2vol, self.objects[idx],state))

        if self.visual_flash:
            # flash the label with the given color
            col = self.state2color.get(state,self.state2color[None])
            self.labels[idx].color = col if len(col)==4 else col + (255,)



# get the noise generator
from mindaffectBCI.examples.presentation.audio.pink_noise import Extract_AE, pink_noise
import numpy as np
class NoiseLevelSoundFlashScreen(SoundFlashScreen):

    def __init__(self, window, symbols, noisetag, state2noise=[.3,.4,.5,.6,.7,.8,.9,1.0], 
                f_ref:float=125, f_min:float=100, f_max:float=4000, frame_size:int=1000, **kwargs):
        self.state2noise, self.f_ref, self.f_min, self.f_max, self.frame_size = (state2noise, f_ref, f_min, f_max, frame_size)
        super().__init__(window, noisetag, symbols, **kwargs)

    def init_symbols(self, symbols, x=0, y=0, w=None, h=None, bgFraction:float=0, font_size:int=None, visual_flash:bool=False):
        # normal initialization
        super().init_symbols(symbols,x,y,w,h,bgFraction,font_size)

        # now make new sounds and add them instead
        # extract the raw sound streams, lengths, and envelop for each stream
        sample_rate = None
        sample_size = None
        snds=[]
        envs=[]
        lens=[]
        for sounds in self.sounds:
            # get the raw bytes
            sound = sounds[1]
            # check format consistency
            if sample_rate is None:
                sample_rate = sound.audio_format.sample_rate
            elif not sample_rate == sound.audio_format.sample_rate:
                raise ValueError("All sounds must have same sample rate")
            if sample_size is None:
                sample_size = sound.audio_format.sample_size
            elif not sample_size == sound.audio_format.sample_size:
                raise ValueError("All sounds must have same sample size")

            # BODGE: horrible bodge to get the raw bytes out of the static source
            snd_bytes = sound.get_queue_source().get_audio_data(1000000).get_string_data()
            snd_array = np.frombuffer(snd_bytes,dtype= np.int8 if sample_size==1 else np.int16)
            env = Extract_AE(snd_array, self.frame_size)

            # store the info
            snds.append(snd_array)
            envs.append(env)
            lens.append(len(snd_array))

        # compute the average envelop over all sounds -- allowing for variable length
        ave_env = np.zeros((max(lens)//self.frame_size + 1,),dtype=np.int16)
        for env in envs:
            ave_env[:len(env)] = ave_env[:len(env)] + env
        ave_env = ave_env / len(envs)

        # now add the noise-level corrupted versions
        for idx,snd in enumerate(snds):
            self.sounds[idx] = [None]*(len(self.state2noise)+1) 
            self.players[idx] = [None]*(len(self.state2noise)+1)
            for ni, level in enumerate(self.state2noise):
                state = ni + 1  # state is one bigger than noise level, so level 0 is always no sound
                # make amplitude modulated noise -- N.B. float32 -1 -> +1 range
                noise = pink_noise(self.f_ref, self.f_min, self.f_max, max(lens), sample_rate)
                # ensure -1 1 scale
                noise = noise / np.max(np.abs(noise))
                # apply amplitude modulation, which also scales to 16bit int size
                for i,a in enumerate(ave_env):
                    ii=slice(i*self.frame_size,(i+1)*self.frame_size)
                    noise[ii]=noise[ii]*a

                # convert to int16
                noise = noise.astype(np.int16)
                if sample_size == 8:
                    snd = snd.astype(np.int16) * ( 1<<8 )

                # mix the sound and the noise with level as mixing ratio
                l = min(len(snd),len(noise)) # guard diff lengths
                snd_and_noise = snd[:l] * (1-level) + noise[:l] * level
                snd_and_noise = snd_and_noise.astype(np.int16)

                # make a sound object with this data
                sound = RawAudioSource(snd_and_noise.tobytes(), 1, sample_rate, 16)

                # add a stack of players for this sound data.
                self.sounds[idx][state]= sound
                print("{}/{} -> {}".format(idx,state,level))
                pos = (idx-(len(self.sounds)-1)/2)*4 if self.spatialize else 0
                self.players[idx][state] = [ mkPlayer(sound,pos) for i in range(3) ]
                

# # building sound file test.
# import numpy as np
# samprate = 44100
# freq = 440 # A
# dur = 2
# sampsize=16
# data = np.sin( np.arange(samprate*dur)*2*np.pi * freq / samprate)
# # convert to int
# data = (data*(2<<sampsize-1)).astype(np.int16 if sampsize==16 else np.int8)

# src = RawAudioSource(data.tobytes(), 1, samprate, sampsize)
# plyr = pyglet.media.Player()
# plyr.queue(src)
# plyr.play()


if __name__ == "__main__":
    args = selectionMatrix.parse_args()
    setattr(args,'calibration_screen','mindaffectBCI.examples.presentation.sound_flash.SoundFlashScreen')
    # setattr(args,'stimfile','oddball.txt')
    setattr(args,'symbols',[[" |chirp\\800-1200-gauss.wav"]])
    #setattr(args,'calibration_screen','mindaffectBCI.examples.presentation.sound_flash.NoiseLevelSoundFlashScreen')
    #setattr(args,'symbols',[["3|digits\\MAE_3A.wav","4|digits\\MAE_4A.wav","5|digits\\MAE_5A.wav","6|digits\\MAE_6A.wav"]])

    #setattr(args,'symbols',[["500|chirp\\400-600-gauss.wav","1000|chirp\\800-1200-gauss.wav","2000|chirp\\1600-2400-gauss.wav","4000|chirp\\3000-5000-gauss.wav"]])

    setattr(args,'stimfile','level8_gold_4obj_interleaved.txt')
    setattr(args,'framesperbit',30)
    setattr(args,'calibration_trialduration',10)
    setattr(args,'cueduration',1.5)
    setattr(args,"fullscreen",False)
    setattr(args,'intertrialduration',2)
    setattr(args,'calibration_args',{"permute":False, "startframe":"random"})
    setattr(args,'calibration_screen_args',{'visual_flash':True, "spatialize":True, "state2vol":{"1":0.00001,"2":0.0001,"3":0.001,"4":0.01,"5":0.1,"6":0.2,"7":0.5,"8":1.0},"use_audio_timestamp":True})
    selectionMatrix.run(**vars(args))
