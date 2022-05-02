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
import time
from mindaffectBCI.presentation.screens.basic_screens import Screen
from mindaffectBCI.presentation.screens.sound_flash import SoundFlash

class HelloWorldScreen(Screen):
    def __init__(self,window,noisetag=None,text="HelloWorld\n\nPress <space> to quit",sound="music_fragments/BR1.wav",duration_ms:float=50000,waitKey:bool=True):
        """simple screen to demonstrate how to make experiment screens

        Args:
            window ([type]): the window to draw in
            noisetag ([type], optional): the noisetag to play. (Needed for compatiability with calibration/prediction screen usage). Defaults to None.
            text (str, optional): Text to show. Defaults to "HelloWorld\n\nPress <space> to quit".
            sound (str, optional): Sound file to play. Defaults to "music_fragments/BR1.wav".
            duration_ms (float, optional): maximum time to show this screen in milliseconds. Defaults to 50000.
            waitKey (bool, optional): flag if we stop early if a key is pressed. Defaults to True.
        """
        self.window, self.noisetag, self.text, self.sound, self.duration_ms, self.waitKey = (window, noisetag, text, sound, duration_ms, waitKey)
        # finish the screen setup
        self.reset()
        
    def reset(self):
        """reset the screen state, to start a fresh playback
        """        
        # setup the pyglet batch and group for fast GL drawing
        # see <https://pyglet.readthedocs.io/en/latest/modules/graphics/index.html>
        self.batch = pyglet.graphics.Batch()
        self.group = pyglet.graphics.OrderedGroup(0)
        # record start time for screen run time computations
        self.t0 = self.getTimeStamp()
        # init the text display
        self.init_text()
        # init the sound display
        self.init_sound()
        # set as not yet finished
        self.isDone = False

    def init_text(self):
        """initialize the text display, i.e. make a label with the text and add to the display batch+group
        """        
        self.text_obj=pyglet.text.Label(self.text, font_size=40, 
                                x=self.window.width//2, y=self.window.height//2,
                                width=self.window.width//2, height=self.window.height//2,
                                color=(255, 255, 255, 255),
                                multiline=True,
                                anchor_x='center', anchor_y='center',
                                batch=self.batch, group=self.group)

    def init_sound(self):
        """initialize the sound display, i.e. just load the sound and create the player object
        """
        self.sound_obj = SoundFlash(self.sound)
        self.next_play_time = self.getTimeStamp() + 1000

    def getTimeStamp(self):
        return (int(time.perf_counter()*1000) % (1<<31))

    def elapsed_ms(self):
        """helper function to get the current running time of the screen in milliseconds

        Returns:
            float: elapsed time in milliseconds
        """        
        return self.getTimeStamp()-self.t0 if self.t0 else -1

    def is_done(self):
        """test if this screen is finished

        Returns:
            bool: screen completion state, True=finished, False=still-running
        """        
        # exceeded the desired time?
        if self.elapsed_ms() > self.duration_ms:
            self.isDone = True
        # quit on key-press and key has been pressed
        if self.waitKey and self.window.last_key_press:
            self.isDone = True
            # N.B. be sure to mark the key as being consumed!
            self.window.last_key_press = None
        return self.isDone
    

    def draw(self,dt):
        """update the display & do stimulus work.

        This function is called every video-frame (i.e. every 1/60th second) to allow display/audio updates

        Args:
            dt (float): elapsed time since last call to draw
        """

        # if passed the next scheduled audio play time, then play the stimuli again
        if self.elapsed_ms() > self.next_play_time :
            # set the vol.  N.B. defaults to volume=0
            self.sound_obj.volume=1
            self.sound_obj.play()
            # schedule the next play time to be in 3s
            self.next_play_time = self.elapsed_ms() + 3000 # record play time so don't play too fast

            # For BCI/logging purposes, tell the BCI what we are doing so it gets saved in the log-file
            self.noisetag.sendStimulusEvent(1, None, objIDs=100) # tell the BCI we played a stimulus at level 1, for object with ID=100

        # clear the window
        self.window.clear()
        # update the text with the time-remaining
        self.text_obj.text = "{}\n\n{}s / {}s".format(self.text, self.elapsed_ms(), self.duration_ms)

        # draw all the bits in one-go
        self.batch.draw()


    #########################################
    # BODGE: empty functions to make work as a calibration screen in selectionMatrix
    def set_grid(self, **kwargs): pass
    def setliveFeedback(self, livefeedback:bool): pass
    def setshowNewTarget(self, shownewtarget:bool): pass
    def set_sentence(self, sentence:str): pass

if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    window = initPyglet(width=640, height=480)
    screen = HelloWorldScreen(window)
    run_screen(window, screen)
