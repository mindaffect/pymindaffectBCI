import pyglet
import mindaffectBCI.presentation.selectionMatrix as selectionMatrix
from mindaffectBCI.presentation.selectionMatrix import getTimeStamp
from mindaffectBCI.presentation.screens.basic_screens import Screen, ScreenList, InstructionScreen, WaitScreen
from mindaffectBCI.presentation.screens.image_flash import Checkerboard
from mindaffectBCI.decoder.utils import intstr2intkey
import json
import random

## N.B. the thresholds log load is moved to staircase_log_readers.py


class HWStaircaseScreen(Screen):
    def __init__(self, window, noisetag=None, text: str = "", label:str=None,
                    duration_ms: float = 9999999, waitKey: bool = False,
                    symbols=[[ ]], starting_level:int=9,
                    start_delay_ms:float=2000, query_play_interval_ms=500, play_query_interval_ms=500, feedback_duration_ms=1500,
                    usertext='', spatialize=False, two_thresholds = [],
                    use_psychopy:bool=True,
                    **kwargs):
        """simple screen to demonstrate how to make experiment screens
        Args:
            window ([type]): the window to draw in
            noisetag ([type], optional): the noisetag to play. (Needed for compatiability with calibration/prediction screen usage). Defaults to None.
            text (str, optional): Text to show. Defaults to "HelloWorld\n\nPress <space> to continue to quit".
            sound (str, optional): Sound file to play. Defaults to "music_fragments/BR1.wav".
            duration_ms (float, optional): maximum time to show this screen in milliseconds. Defaults to 50000.
            waitKey (bool, optional): flag if we stop early if a key is pressed. Defaults to True.
            dir_symbols(dictionary): directories of digits/symbols used
            query_play_interval_ms (int, optional): gap between query response and play of the next sound in ms. Defulats to 1000
            play_query_interval_ms (int, optional): gap between starting sound playback and next query in milliseconds.  Defaults to 1000
        """
        self.window, self.noisetag, self.symbols, self.text, self.label, self.duration_ms, self.waitKey, self.symbols, self.usertext, self.query_play_interval_ms, self.play_query_interval_ms, self.feedback_duration_ms, self.start_delay_ms, self.starting_level, self.spatialize, self.two_thresholds, self.use_psychopy = (
            window, noisetag, symbols, text, label, duration_ms, waitKey, symbols, usertext, query_play_interval_ms, play_query_interval_ms, feedback_duration_ms, start_delay_ms, starting_level, spatialize, two_thresholds, use_psychopy)
        if self.label is None: self.label = self.__class__.__name__
        self.reset()

    def reset(self):
        """reset the screen state, to start a fresh playback
        """
        # setup the pyglet batch and group for fast GL drawing
        # see <https://pyglet.readthedocs.io/en/latest/modules/graphics/index.html>

        self.batch = pyglet.graphics.Batch()
        self.group = pyglet.graphics.OrderedGroup(0)

        # record start time for screen run time computations
        self.t0 = getTimeStamp()
        # init the text display
        self.init_text()
        # init the stimuli display
        self.init_stimulus(self.symbols)
        # init the staircase function
        self.init_staircase()
        # set as not yet finished
        self.isDone = False
        
        self.correct = False
        #self.chirp_idx = 3
        self.window.last_text = None
        self.window.last_key_press = None
        # setup the playback logic
        self.next_state_time = self.elapsed_ms() + self.start_delay_ms
        self.window.last_text = None # ensurethe keyboard input is cleared!
        self.state = 'trial_start' # the state of the display, one-of 'trial_start', init_play_stimulus, play_stimulus, query_response, feedback, end
        self.next_state_time = -1  # earlist time at which this state's method will be called


    def init_text(self):
        """initialize the text display, i.e. make a label with the text and add to the display batch+group
        """
        self.text_obj = pyglet.text.Label(x=self.window.width//2,
                                            y=self.window.height//2,
                                            anchor_x='center',
                                            anchor_y='center',
                                            font_size=30,
                                            color=(255, 255, 255, 255),
                                            multiline=True,
                                            width=int(self.window.width*.8),
                                            batch=self.batch, group=self.group)


        self.inputLabel = pyglet.text.Label(self.text, font_size=25,
                                          x=self.window.width//2, y=self.window.height//2,
                                          width=self.window.width//2, height=self.window.height//2,
                                          color=(255, 255, 255, 255),
                                          multiline=True,
                                          anchor_x='center', anchor_y='center',
                                          batch=self.batch, group=self.group)
        #self.set_text('Enter the last digit: ')


    def init_stimulus(self,symbols):
        """initialize the stimuli display for this staircase
        """
        if self.use_psychopy :
            from mindaffectBCI.presentation.screens.sound_flash_psypy import SoundFlash
        else:
            from mindaffectBCI.presentation.screens.sound_flash import SoundFlash

        self.stim_objects= []
        for i,row in enumerate(self.symbols):
            self.stim_objects.append([])
            for j,stim in enumerate(row):
                self.stim_objects[i].append(None)
                # assume it's a set of sounds to play
                sndfile = stim 
                if '|' in sndfile: sndfile = sndfile.split('|')[1]
                #spatialize
                pos = 0
                if isinstance(self.spatialize,float):
                    pos = self.spatialize
                elif self.spatialize == True:
                    pos =(i-(len(self.symbols)-1)/2)*2/(len(self.symbols)-1)

                self.stim_objects[i][j] = SoundFlash(sndfile,pos)


        # set the starting level, as *last* stim-object
        self.starting_level = len(self.stim_objects)-1


    def init_staircase(self):
        self.lvl_idx = self.starting_level
        self.threshold = None
        self.behavioural_log = []
        self.n_trial = 0
        self.n_correct = []
        self.n_wrong = 0
        self.lvls_presented = []
        self.n_thresholds = 0

    def elapsed_ms(self):
        """helper function to get the current running time of the screen in milliseconds

        Returns:
            float: elapsed time in milliseconds
        """
        return getTimeStamp()-self.t0 if self.t0 else -1

    def is_done(self):
        """test if this screen is finished

        Returns:
            bool: screen completion state, True=finished, False=still-running
        """
        # exceeded the desired time?
        if self.elapsed_ms() > self.duration_ms:
            self.isDone = True
        if self.n_trial >= 80:  # just for now, change later
            self.isDone = True
        if self.waitKey and self.window.last_text is not None:
            self.isDone = True
        if self.window.last_text is not None:
            self.user_input = str(self.window.last_text)
            self.user_input = self.user_input.strip()
            if self.user_input == 'w':
                self.isDone = True
        return self.isDone

    def set_input_text(self, text):  # inspiration from QueryDialogScreen
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        elif text is None:
            text = ""
        self.inputLabel.begin_update()
        self.inputLabel.text = text
        self.inputLabel.end_update()


    def trial_start(self):
        # just play the stimulus
        if self.n_trial < 1:
            self.text_obj.text = "Please pay attention!"
            self.next_state_time = self.elapsed_ms() + 2500
            self.state = 'init_play_stimulus'
        else:
            self.next_state_time = self.elapsed_ms() + self.query_play_interval_ms
            self.state = 'init_play_stimulus'

    def init_play_stimulus(self):

        # self.text_obj.text = 'Listen to the chirps'
        # self.set_input_text("")

        # play the 0,0 stim object with the given volume
        self.stim_objects[0][0].volume = self.lvl_idx
        self.stim_objects[0][0].play()
        self.correct_response = 1 # record the correct response the user should give

        # move to playing the stimulus
        self.end_play_time = self.elapsed_ms() + 500 # simulate play takes 2s
        self.state = 'play_stimulus'

    def play_stimulus(self):
        # audio auto-plays so we don't have to do anything here -- except check for when we should stop and move to query
        if self.elapsed_ms() < self.end_play_time: # simulate we play for 2s
            return
        else:
            # finished playback move to query
            self.next_state_time = self.elapsed_ms() + self.play_query_interval_ms
            self.query_timeout = self.next_state_time + 2500
            self.state = 'query_response'

    def query_response(self):
        # ask for user input
        self.set_input_text('Press 1 if you heard the chirp and 2 if you didnt.')
        #self.text_obj.text = f'Input: {self.window.last_text}'

        #later backspace = self.window.last_key_press

        if self.window.last_text is not None:  # and self.elapsed_ms()> self.query_time: #change if we want the user to see its' digit
            try:
                self.user_input = int(self.window.last_text)
            except:
                print("Incorrect user input")
                self.user_input = None
            self.window.last_text = None  # so the user inputs don't accumulate
            self.window.last_key_press = None
            self.n_trial += 1
            self.correct = self.correct_response == self.user_input

            # log this individual response
            logmsg_user_response = {"type": "AMchirps:user_response",  "label":self.label, "stimulus":{"level_idx":self.lvl_idx}, "user_response_correct":self.correct}
            self.noisetag.log(json.dumps(logmsg_user_response))

            # get the next level to try or threshold if found
            self.lvl_idx, self.threshold = self.HW_get_next_level()
            print("{}: lvl={}".format(self.correct, self.lvl_idx))

            # next state depends on if we've found the threshold
            if self.threshold is not None:
                self.n_thresholds += 1

                if self.n_thresholds == 1:

                    #reset all the lists
                    self.lvls_presented = []
                    self.n_correct = []
                    self.n_wrong = 0
                    self.threshold = None
                    self.lvl_idx = self.starting_level
                    self.next_state_time = self.elapsed_ms() + 500
                    self.state = 'trial_start'
                else:
                    self.two_thresholds.append([self.threshold, self.lvl_idx])
                    self.average_level = (self.two_thresholds[0][0] + self.two_thresholds[1][0]) / 2
                    self.average_idx = (self.two_thresholds[0][1] + self.two_thresholds[1][1]) / 2  
                    
                    # log this result to the save-file
                    logmsg_summary = {"type": "detection_threshold",  "label":self.label, "stimulus":{"two_thresholds": self.two_thresholds,
                            "level_idx":self.average_idx, "level":self.average_level}}
                    self.noisetag.log(json.dumps(logmsg_summary))

                    self.state = 'show_feedback'
            else:
                self.state = 'trial_start'
                

    def threshold_calc(self, l_i, presentations_list, correct_list):
        n_pres = presentations_list.count(l_i)
        n_corr = correct_list.count(l_i)
        print("lvl={:2d} {:d}/{:d} = {:5.3f}".format(l_i,n_corr,n_pres,n_corr/n_pres))
        if 2 <= n_pres <=3 and (n_corr / n_pres) > 0.5:
            return True
        else:
            return False


    def HW_get_next_level(self):

        # record info on what stimuli have been presented
        if self.n_wrong >= 1: #start counting the lvls presented only after the initial descent
            self.lvls_presented.append(self.lvl_idx)

        if self.correct:  
            print('correct')
            self.n_correct.append(self.lvl_idx) #start counting the correct responses to a lvl
            if self.n_wrong >= 1 and self.threshold_calc(self.lvl_idx, self.lvls_presented, self.n_correct):   #check if it's time to calc threshold
                self.threshold = self.lvl[self.lvl_idx]
                print(f'threshold found:{self.lvl_idx}={self.threshold}')
                print(self.lvls_presented, self.n_correct)
                # log this result to the save-file
                logmsg_summary = {"type": "detection_threshold", "label":self.label, "stimulus":{"level_idx":self.lvl_idx, "level":self.threshold}}
                self.noisetag.log(json.dumps(logmsg_summary))

            else: # the user was correct = move 10db down
                self.lvl_idx = self.lvl_idx - 2

        else:       # the user was incorrect = move 5db up
            print('incorrect')
            self.n_wrong += 1
            self.lvl_idx += 1

        return self.lvl_idx, self.threshold


    def show_feedback(self):
        # set to quit on key-press
        self.waitKey = True
        # set to stop after 2s from now
        self.duration_ms = self.elapsed_ms() + self.feedback_duration_ms
        # log the threshold info to the screen
        self.text_obj.text = f"Thresholds found {self.two_thresholds}"
        #save the global variable
        selectionMatrix.user_state['staircase_threshold_idx'] = self.average_idx
        selectionMatrix.user_state['staircase_threshold_level'] = self.average_level
        print("setting user threshold level:{}".format(self.average_level))
        self.window.last_key_press = None

        self.state = 'end'




    def draw(self, dt):
        """update the display & do stimulus work.

        This function is called every video-frame (i.e. every 1/60th second) to allow display/audio updates

        Args:
            dt (float): elapsed time since last call to draw

        """
         # just to test
        # if passed the next scheduled audio play time, then play the stimuli again
        if self.elapsed_ms() > self.next_state_time:
            if self.state == 'trial_start':
                self.trial_start()
            elif self.state == 'init_play_stimulus':
                self.init_play_stimulus()
            elif self.state == 'play_stimulus':
                self.play_stimulus()
            elif self.state == 'query_response':
                self.query_response()
            elif self.state == 'show_feedback':
                self.show_feedback()
            elif self.state == 'end':
                pass

        # clear the window
        self.window.clear()

        # draw all the bits in one-go
        self.batch.draw()

    #########################################
    # BODGE: empty functions to make work as a calibration screen in selectionMatrix
    def set_grid(self, **kwargs): pass
    def setliveFeedback(self, livefeedback: bool): pass
    def setshowNewTarget(self, shownewtarget: bool): pass
    def set_sentence(self, sentence: str): pass
    def setliveSelections(self, x, **kwargs):pass


#HWStaircaseScreenAM = HWStaircaseScreen


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class HWStaircaseScreenVA(HWStaircaseScreen):
    # example replacing the stimuli with visual checkerboard stimuli
    def __init__(self, window, noisetag, bgFraction:float=.4, **kwargs):
        self.bgFraction = bgFraction # init specific to this version, N.B. before general init
        super().__init__(window,noisetag,**kwargs) # general init

    def init_stimulus(self,symbols=None):
        # make a stimulus object for each symbol
        # assume these are given in order of *decreasing* behavioural difficulty
        x,y,w,h = (int(self.window.width*self.bgFraction/2), int(self.window.height*self.bgFraction/2),
                   int(self.window.width*(1-self.bgFraction)), int(self.window.height*(1-self.bgFraction)))
        self.stim_objects = []
        for i,row in enumerate(symbols):
            for j,symb in enumerate(row):
                nx=symb[1]
                cb = Checkerboard(x,y,w,h,nx=nx,ny=1,batch=self.batch,group=self.group)
                cb.visible=False
                self.stim_objects.append(cb)
        # set the starting level, as *last* stim-object
        self.starting_level = len(self.stim_objects)-1
    
    def init_play_stimulus(self):
        #self.text_obj.text = 'Observe the visual pattern'
        self.set_input_text("")

        # bounds check the level
        self.lvl_idx = max(0, min(len(self.stim_objects)-1, self.lvl_idx))
        print("Play: {}".format(self.lvl_idx))
        self.n_reverse = 0 # count the number of pattern reversals that have happened
        # turn off all but the target stim
        for s in self.stim_objects:
            s.visible=False
        self.stim_objects[self.lvl_idx].visible = True
        self.cur_stim = self.stim_objects[self.lvl_idx]
        self.state = 'play_stimulus'

    def play_stimulus(self):
        if self.n_reverse < 5:
            if self.n_reverse % 2 == 0:
                self.stim_objects[self.lvl_idx].color = (160,160,160)
            else:
                self.stim_objects[self.lvl_idx].color = (96,96,96)

            # next reversal in 500ms
            self.n_reverse = self.n_reverse + 1
            self.next_state_time = self.elapsed_ms() + 100

        else:
            self.stim_objects[self.lvl_idx].visible = False # turn off the square
            self.correct_response = 1 # record what the right user response should be

            # switch to query in a bit of time
            self.next_state_time = self.elapsed_ms() + self.play_query_interval_ms
            self.state = 'query_response'

    def query_response(self):
        # ask for user input
        self.set_input_text('Press 1 if you saw the pattern reversal and 2 if you didnt.')
        self.text_obj.text = f'Input: {self.window.last_text}'

        #later backspace = self.window.last_key_press

        if self.window.last_text is not None:  # and self.elapsed_ms()> self.query_time: #change if we want the user to see its' digit
            try:
                self.user_input = int(self.window.last_text)
            except:
                print("Incorrect user input")
                self.user_input = None
            self.window.last_text = None  # so the user inputs don't accumulate
            self.window.last_key_press = None
            self.n_trial += 1
            self.correct = self.correct_response == self.user_input

            # log this individual response
            logmsg_user_response = {"type": "stimulus:user_response",  "label":self.label, "stimulus":{"level_idx":self.lvl_idx}, "user_response_correct":self.correct}
            self.noisetag.log(json.dumps(logmsg_user_response))

            # get the next level to try or threshold if found
            self.lvl_idx, self.threshold = self.HW_get_next_level()
            print("{}: lvl={}".format(self.correct, self.lvl_idx))

            # next state depends on if we've found the threshold
            if self.threshold is not None:
                self.n_thresholds += 1

                if self.n_thresholds == 1:
                    self.two_thresholds.append([self.threshold, self.lvl_idx])
                    print(f"the two thresholds are: {self.two_thresholds}")
                    #reset all the lists
                    self.lvls_presented = []
                    self.n_correct = []
                    self.n_wrong = 0
                    self.threshold = None
                    self.lvl_idx = self.starting_level
                    self.next_state_time = self.elapsed_ms() + 500
                    self.state = 'trial_start'
                else:
                    self.two_thresholds.append([self.threshold, self.lvl_idx])
                    self.average_level = (self.two_thresholds[0][0] + self.two_thresholds[1][0]) / 2
                    self.average_idx = (self.two_thresholds[0][1] + self.two_thresholds[1][1]) / 2  
                    
                    # log this result to the save-file
                    logmsg_summary = {"type": "detection_threshold", "stimulus":{"two_thresholds": self.two_thresholds,
                            "level_idx":self.average_idx, "level":self.average_level}}
                    self.noisetag.log(json.dumps(logmsg_summary))

                    self.state = 'show_feedback'
            else:
                self.state = 'trial_start'


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
from mindaffectBCI.decoder.utils import intstr2intkey
class HWStaircaseScreenAM(HWStaircaseScreen):
    def __init__(self, window, noisetag=None, starting_level:int=7,
                    lvl=[0, 0.00005623, 0.0001, 0.0001778, 0.0003162, 0.0005623, 0.001, 0.001778, 0.01778],
                    chirp_idx=None, object_vol_correction:dict=None, two_thresholds:list=[], label=None,
                    **kwargs):
        if starting_level is None: starting_level = len(lvl)-1
        # specific init for this version
        self.chirp_idx, self.lvl, self.two_thresholds, self.label = chirp_idx, lvl, two_thresholds, label
        self.object_vol_correction = intstr2intkey(object_vol_correction) if object_vol_correction is not None else dict()
        self.lvl = [x * self.object_vol_correction.get(self.chirp_idx,1) for x in self.lvl]
        # general init for all staircases
        super().__init__(window, noisetag, starting_level=starting_level, **kwargs)

    def flatten_symbols(self, symbols):
        return [symb for row in symbols for symb in row if symb is not None]

    def init_stimulus(self,symbols):
        if self.use_psychopy :
            from mindaffectBCI.presentation.screens.sound_flash_psypy import SoundFlash
        else:
            from mindaffectBCI.presentation.screens.sound_flash import SoundFlash
        self.sndfile = self.flatten_symbols(self.symbols)[self.chirp_idx]
        if '|' in self.sndfile: self.sndfile = self.sndfile.split('|')[1]
        print(f"snd file is: {self.sndfile}")
        print(f"symbols are: {self.symbols}")
        print(f"chirp idx is: {self.chirp_idx}")

        # spatialze if wanted
        print(self.spatialize)
        pos = 0
        if isinstance(self.spatialize,int):
            pos = self.spatialize
        self.sound = SoundFlash(self.sndfile, pos)


    def init_play_stimulus(self):


        self.lvl_idx = max(min(len(self.lvl)-1, self.lvl_idx),0)
        self.sound.volume = self.lvl[self.lvl_idx]
        print(' Play: {} @ {}'.format(self.lvl_idx,self.lvl[self.lvl_idx]))
        
        self.sound.play()
        #self.correct_response = 1 # record the correct response the user should give

        # move to playing the stimulus
        self.end_play_time = self.elapsed_ms() + 500 # simulate play takes 1s
        self.state = 'play_stimulus'

    def HW_get_next_level(self):
        # record info on what stimuli have been presented

        if len(self.two_thresholds) == 2:
            del self.two_thresholds[:]
        if self.n_wrong >= 1 and self.n_trial > 3: #start counting the lvls presented only after the initial descent and after the 3rd trial
            self.lvls_presented.append(self.lvl_idx)

        if self.correct:  
            print('correct')
            self.n_correct.append(self.lvl_idx) #start counting the correct responses to a lvl

            if self.n_wrong >= 1 and self.n_trial > 3 and self.threshold_calc(self.lvl_idx, self.lvls_presented, self.n_correct):    #check if it's time to calc threshold
                self.threshold = self.lvl[self.lvl_idx]
                print(f'threshold found:{self.lvl_idx}={self.threshold}')
                print(f'levels presented are:{self.lvls_presented}')
                print(f"length of lvls presented is : {len(self.lvls_presented)}")
                print(f'number of trials: {self.n_trial}')

            else: # the user was correct = move 10db down
                self.lvl_idx = self.lvl_idx - 2

        else:       # the user was incorrect = move 5db up
            print('incorrect')
            self.n_wrong += 1
            self.lvl_idx += 1

        return self.lvl_idx, self.threshold

    
    def query_response(self):
        # ask for user input
        self.set_input_text('Press <space> if you hear the chirp. Wait if you hear nothing.')
        self.text_obj.text = ""
        #print(f"the elapsed time is: {self.elapsed_ms()}")
        #print(f"the query timeout time is: {self.query_timeout}")


        if self.elapsed_ms() > self.query_timeout:
            self.window.last_key_press = 'did not hear it' #whichever just to note a 'wrong' one

        if self.window.last_key_press is not None:
            print(f"last key pressed is (32 indicates <space>): {self.window.last_key_press}")

            #self.correct = self.window.last_key_press = pyglet.window.key.BACKSPACE 
            self.correct = self.window.last_key_press == 32 #32 represents <space>
            self.window.last_key_press = None
            self.window.last_text = None
            self.n_trial += 1

            # log this individual response
            logmsg_user_response = {"type": "AMchirps:user_response", "label":self.label, "stimulus":{"level_idx":self.lvl_idx, "level": self.lvl[self.lvl_idx]}, "user_response_correct":self.correct}
            self.noisetag.log(json.dumps(logmsg_user_response))

            # get the next level to try or threshold if found
            self.lvl_idx, self.threshold = self.HW_get_next_level()
            print("{}: lvl={}".format(self.correct, self.lvl_idx))

            # next state depends on if we've found the threshold
            if self.threshold is not None:
                self.n_thresholds += 1

                if self.n_thresholds == 1:
                    self.two_thresholds.append([self.threshold, self.lvl_idx])
                    print(f"the two thresholds are: {self.two_thresholds}")

                    #reset all the lists
                    self.lvls_presented = []
                    self.n_correct = []
                    self.n_wrong = 0
                    self.threshold = None
                    self.lvl_idx = self.starting_level
                    self.next_state_time = self.elapsed_ms() + 500
                    self.state = 'trial_start'

                else:
                    self.two_thresholds.append([self.threshold, self.lvl_idx])
                    self.average_level = (self.two_thresholds[0][0] + self.two_thresholds[1][0]) / 2
                    self.average_idx = (self.two_thresholds[0][1] + self.two_thresholds[1][1]) / 2  
                    
                    # log this result to the save-file
                    logmsg_summary = {"type": "detection_threshold", "stimulus":{"two_thresholds": self.two_thresholds,
                            "level_idx":self.average_idx, "level":self.average_level, "tone":self.chirp_idx, "label":self.label}}
                    self.noisetag.log(json.dumps(logmsg_summary))

                    self.state = 'show_feedback'
            else:
                self.state = 'trial_start'


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class HWStaircaseScreenLE(HWStaircaseScreen):
    def __init__(self, window, noisetag=None, starting_level=None,
                    lvl=[0,0.05,0.0629,0.079,0.099,0.125,0.158,0.199,0.250,0.315,0.397, 0.499, 0.629, 0.792, 0.997],#separated by 2dB
                    lvl_idx=0, background_noise = "noise\\babble_norm_cut.wav",
                    symbols=[[]],#[["digits\\MAE_0A.wav", "digits\\MAE_1A.wav","digits\\MAE_2A.wav","digits\\MAE_3A.wav","digits\\MAE_4A.wav","digits\\MAE_5A.wav"]],
                    **kwargs):
        if starting_level is None: starting_level = len(lvl)-1
        # specific init for this version
        self.lvl, self.lvl_idx, self.background_noise, self.symbols = (lvl, lvl_idx, background_noise, symbols)
        # general init for all staircases
        super().__init__(window, noisetag, starting_level=starting_level, symbols=symbols, **kwargs)
    
    def reset(self):
        #reset the screen state, to start a fresh playback
    
        # setup the pyglet batch and group for fast GL drawing
        # see <https://pyglet.readthedocs.io/en/latest/modules/graphics/index.html>

        self.batch = pyglet.graphics.Batch()
        self.group = pyglet.graphics.OrderedGroup(0)

        # record start time for screen run time computations
        self.t0 = getTimeStamp()
        # init the text display
        self.init_text()
        # init the stimuli display
        self.init_noise(self.background_noise) 
        self.init_digits(self.symbols)
        # init the staircase function
        self.init_staircase()
        # set as not yet finished
        self.isDone = False
        
        self.correct = False
        self.window.last_text = None
        # setup the playback logic
        self.next_state_time = self.elapsed_ms() + self.start_delay_ms
        self.window.last_text = None # ensurethe keyboard input is cleared!
        self.state = 'trial_start' # the state of the display, one-of 'trial_start', init_play_stimulus, play_stimulus, query_response, feedback, end
        self.next_state_time = -1  # earliest time at which this state's method will be called


    def init_noise(self,background_noise):
        if self.use_psychopy :
            from mindaffectBCI.presentation.screens.sound_flash_psypy import SoundFlash
        else:
            from mindaffectBCI.presentation.screens.sound_flash import SoundFlash
        #init the background noise
        pos = 0
        self.sound_noise = SoundFlash(self.background_noise)

    def init_digits(self,symbols):
        if self.use_psychopy :
            from mindaffectBCI.presentation.screens.sound_flash_psypy import SoundFlash
        else:
            from mindaffectBCI.presentation.screens.sound_flash import SoundFlash

        #init the digits
        self.stim_objects= []
        for i,row in enumerate(self.symbols):
            self.stim_objects.append([])
            for j,stim in enumerate(row):
                self.stim_objects[i].append(None)
                # assume it's a set of sounds to play
                sndfile = stim 
                if '|' in sndfile: sndfile = sndfile.split('|')[1]
                pos = 0
                if isinstance(self.spatialize,float):
                    pos = self.spatialize
                elif self.spatialize == True and len(self.symbols)-1 > 0:
                    pos = i*2/(len(self.symbols)-1)-1
                
                self.stim_objects[i][j] = SoundFlash(sndfile,pos)
                print(f"stim objects are: {self.stim_objects}")
    
    def play_digit(self,list_digits,volume_lvls,volume_idx,play_delay,digit_idx=None):
        if digit_idx == None:
            self.digit_to_play = random.choice(list_digits)
        else:
            self.digit_to_play = list_digits[digit_idx]
        self.digit_to_play.volume = volume_lvls[volume_idx]
        self.digit_to_play.play(delay=play_delay)


    def init_play_stimulus(self):

        self.text_obj.text = 'Listen to the digits. Remember the last one.'
        self.set_input_text("")
        self.lvl_idx = max(0, min(len(self.stim_objects[0])-1, self.lvl_idx))
        print(f"lvl idx is:{self.lvl_idx}")

        #play the noise
        self.sound_noise.volume = 0.1
        self.sound_noise.play()

        #play the digits
        print(f"stim objects are : {self.stim_objects[0]}")
        self.play_digit(self.stim_objects[0],self.lvl,self.lvl_idx,play_delay=1)
        self.play_digit(self.stim_objects[0],self.lvl,self.lvl_idx,play_delay=2)

        self.last_digit = random.choice([i for i in range((len(self.symbols[0])-1))])
        self.play_digit(self.stim_objects[0],self.lvl,self.lvl_idx,play_delay=3,digit_idx=self.last_digit)
        print(f"last digit: {self.last_digit}")

        self.correct_response = self.last_digit # record the correct response the user should give 

        # move to playing the stimulus
        self.end_play_time = self.elapsed_ms() + 3000 # simulate play takes 3s
        self.state = 'play_stimulus'

    def query_response(self): #change so the user inputs the actual heard number
        # ask for user input
        self.set_input_text('Enter the last digit')
        self.text_obj.text = f'Input: {self.window.last_text}'
        self.sound_noise.pause()

        if self.window.last_text is not None:  # and self.elapsed_ms()> self.query_time: #change if we want the user to see its' digit
            try:
                self.user_input = int(self.window.last_text)
            except:
                print("Incorrect user input")
                self.user_input = None
            self.window.last_text = None  # so the user inputs don't accumulate
            self.window.last_key_press = None
            self.n_trial += 1
            self.correct = self.correct_response == self.user_input

            # log this individual response
            logmsg_user_response = {"type": "LEdigit:user_response", "stimulus":{"level_idx":self.lvl_idx}, "user_response_correct":self.correct}
            self.noisetag.log(json.dumps(logmsg_user_response))

            # get the next level to try or threshold if found
            self.lvl_idx, self.threshold = self.HW_get_next_level()
            print("{}: lvl={}".format(self.correct, self.lvl_idx))

            # next state depends on if we've found the threshold
            if self.threshold is not None:
                self.n_thresholds += 1

                if self.n_thresholds == 1:
                    #reset all the lists
                    self.lvls_presented = []
                    self.n_correct = []
                    self.n_wrong = 0
                    self.threshold = None
                    self.lvl_idx = self.starting_level
                    self.next_state_time = self.elapsed_ms() + 500
                    self.state = 'trial_start'
                else:
                    self.state = 'show_feedback'
            else:
                self.state = 'trial_start'

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#class for running the thresholds all together in one screen
from enum import IntEnum
from mindaffectBCI.noisetag import Noisetag
class AllThresholdsScreen(ScreenList):
    def __init__(self, window, noisetag=Noisetag, starting_level:int=7,
                    lvl=[0, 0.00005623, 0.0001, 0.0001778, 0.0003162, 0.0005623, 0.001, 0.001778, 0.01778],
                    symbols=[["|chirp\\400-600-gauss.wav","|chirp\\800-1200-gauss.wav","|chirp\\1600-2400-gauss.wav","|chirp\\3000-5000-gauss.wav",
                                "|chirp\\400-600-gauss.wav","|chirp\\800-1200-gauss.wav","|chirp\\1600-2400-gauss.wav","|chirp\\3000-5000-gauss.wav"]],
                    test_screen_args = [
                        {"chirp_idx":0, "label":"L500", "spatialize":-1},
                        {"chirp_idx":1, "label":"L1000", "spatialize":-1},
                        {"chirp_idx":2, "label":"L2000", "spatialize":-1},
                        {"chirp_idx":3, "label":"L4000", "spatialize":-1},
                        {"chirp_idx":4, "label":"R500", "spatialize":1},
                        {"chirp_idx":5, "label":"R1000", "spatialize":1},
                        {"chirp_idx":6, "label":"R2000", "spatialize":1},
                        {"chirp_idx":7, "label":"R4000", "spatialize":1},
                    ],
                    label='Behavioral Thresholds', instructions:str='',
                    use_psychopy:bool=True,
                    **kwargs):
        # general init for all staircases
        super().__init__(window, noisetag, symbols, label=label, **kwargs)
        self.lvl, self.starting_level, self.instructions, self.use_psychopy = lvl, starting_level, instructions, use_psychopy
        if self.starting_level is None : self.starting_level = len(self.lvl)-1

        # make the start, test and end screens
        self.sub_screens = []
        scr = WaitScreen(window, duration=200, waitKey=False)
        self.sub_screens.append(scr)

        # make the set of testing screens to run through in turn
        for tsargs in test_screen_args:
            scr = HWStaircaseScreenAM(symbols=symbols, window=window, noisetag=noisetag, lvl=lvl, starting_level=starting_level,
                                      use_psychopy=use_psychopy, **tsargs) 
            self.sub_screens.append(scr)
        
        scr = WaitScreen(window, duration=200, waitKey=False)
        self.sub_screens.append(scr)

        self.current_screen_idx = 0


    #########################################
    # BODGE: empty functions to make work as a calibration screen in selectionMatrix
    def set_grid(self, **kwargs): pass
    def setliveFeedback(self, livefeedback: bool): pass
    def setshowNewTarget(self, shownewtarget: bool): pass
    def set_sentence(self, sentence: str): pass
    def setliveSelections(self, x, **kwargs):pass




def run():
    #Setup and run the given screen for debugging.

    # initialize the display
    selectionMatrix.init_noisetag_and_window()
    # connect to the BCI hub
    selectionMatrix.nt.connect()

    # make the screen object
    ss = HWStaircaseScreenAM(selectionMatrix.window, selectionMatrix.nt)

    #ss = HWStaircaseScreenVA(selectionMatrix.window, selectionMatrix.nt),
     #               symbols=[[("",-1), ("",-2), ("",-3), ("",-5), ("",-10), ("",-40)]])


    # run the given screen
    selectionMatrix.run_screen(ss)


if __name__ == "__main__":
    run()
