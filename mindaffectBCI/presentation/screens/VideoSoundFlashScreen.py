import os
import pyglet
from mindaffectBCI.presentation.screens.sound_flash import SoundFlashScreen
from mindaffectBCI.decoder.utils import search_directories_for_file

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