from mindaffectBCI.presentation.screens.SelectionGridScreen import SelectionGridScreen
from mindaffectBCI.decoder.utils import intstr2intkey
import random
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class SoundFlashScreen(SelectionGridScreen):
    """variant of SelectionGridScreen which plays a sound dependent on the object id and it's current state
    """
    def __init__(self, window, noisetag, symbols, visual_flash: bool = True, state2vol: dict = None, spatialize: bool = False,
            use_audio_timestamp: bool = False, 
            pulsed_background_noise: bool = False, background_noise: str = None,
            background_noise_vol: float = .1,  background_noise_start:str = 'random',
            play_delay: float = .05, object_vol_correction: dict = None,
            adaptive_stimulus: bool = True, use_psychopy:bool=True, **kwargs):

        print("use_audio_timestamps={}".format(use_audio_timestamp))
        self.visual_flash, self.spatialize, self.use_audio_timestamp, self.background_noise, self.background_noise_vol, self.pulsed_background_noise, self.background_noise_start, self.play_delay, self.adaptive_stimulus = (
            visual_flash, spatialize, use_audio_timestamp, background_noise, background_noise_vol, pulsed_background_noise, background_noise_start, play_delay, adaptive_stimulus)
        if state2vol is not None:
            self.state2vol = intstr2intkey(state2vol)
        # BODGE: ensure is a 0 level
        if not 0 in self.state2vol:
            print("WARNING: state2vol should include a volume for level 0.") 
            self.state2vol[0]=0
        self.object_vol_correction = intstr2intkey(
            object_vol_correction) if object_vol_correction is not None else dict()
        self.stimtime=None

        # setup the media backend to use
        if use_psychopy:
            from mindaffectBCI.presentation.screens.sound_flash_psypy import SoundFlash
        else:
            from mindaffectBCI.presentation.screens.sound_flash import SoundFlash
        self.SoundFlash = SoundFlash

        super().__init__(window, noisetag, symbols=symbols, **kwargs)
        self.init_background_noise(background_noise)


    def init_background_noise(self,background_noise):
        # create background noise object, with single player as don't need to overlap playback
        if background_noise is not None:
            if not hasattr(background_noise, '__iter__'): background_noise=[background_noise]
            self.background_noise = [self.SoundFlash(n, nplayers=1) for n in background_noise]
            self.background_noise_idx = 0
        else:
            self.background_noise = None

    def play_background(self,idx:int,vol:float,**kwargs):
        try:
            self.background_noise_idx=idx
            self.background_noise[idx].volume = vol
            self.background_noise[idx].play(**kwargs)
        except:
            self.background_noise_idx=None
    
    def pause_background(self,idx=None,**kwargs):
        if idx is None: idx=self.background_noise_idx
        try:
            self.background_noise[idx].pause(**kwargs)
        except:
            pass

    def get_background_noise(self):
        try:
            return self.background_noise[self.background_noise_idx]
        except:
            return self.background_noise[0]

    def init_target(self, symb, x, y, w, h, i, j, font_size:int=None):
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
        if isinstance(symb,str): symb=symb.split("|")
        pos = (i-(len(self.symbols)-1)/2)*4 if self.spatialize else 0
        players = self.SoundFlash(symb[1],pos,media_directories=self.media_directories)
        
        # make the label
        label= self.init_label(symb[0],x,y,w,h,font_size)
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
                 None:1} # default..
    def update_object_state(self, idx:int, state):
        """update the indicated object with the indicated state, i.e. play the right sound

        Args:
            idx (int): index into the symbols set for the object (sound) to update it's state info
            state (int|float): the updated state for this object

        Note: for *sound* objects, the actual playing state of the object is *only* changed when it differs from the previous state of this object.
        This means that sounds only start playing when the state is changed.
        """
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
            col = tuple(self.state2color.get(state,self.state2color[None]))
            self.labels[idx].color = col if len(col)==4 else col + (255,)

        # play the background noise
        if not self.pulsed_background_noise and self.background_noise:
            if not self.get_background_noise().status == 1:
                # pick random background noise segment to play
                idx = random.randint(0,len(self.background_noise))
                print("Using background: {}".format(idx))
                self.play_background(idx,self.background_noise_vol,delay=self.play_delay, loops=1)


    def is_done(self):
        if self.background_noise and self.isDone:
            self.get_background_noise().stop()
        return super().is_done()

        # print the player times
        #print("times: {}".format([(p[0],[1],p[2].time) for p in self.get_active_players()]))

    def doPredictionDistribution(self, ptd):
        """adapt the stimlus distribution based on the predicted target dist

        Args:
            ptd (PredictedTargetDist): [description]
        """
        if self.adaptive_stimulus and hasattr(self.noisetag.get_stimSeq(),'update_from_predicted_target_dist'):
            self.noisetag.get_stimSeq().update_from_predicted_target_dist(ptd)


# get the noise generator
from mindaffectBCI.presentation.audio.pink_noise import Extract_AE, pink_noise
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


if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    # make a noisetag object without the connection to the hub for testing
    nt = Noisetag(stimSeq='mgold_65_6532.txt', utopiaController=None)
    window = initPyglet(width=640, height=480)
    screen = SoundFlashScreen(window, nt, symbols=[["1|digits\\MAE_1A.wav", "2|digits\\MAE_2A.wav", "3|digits\\MAE_3A.wav", "4|digits\\MAE_4A.wav", "5|digits\\MAE_5A.wav"]])
    # wait for a connection to the BCI
    nt.startFlicker(framesperbit=30)
    # run the screen with the flicker
    run_screen(window, screen)