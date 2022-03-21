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

import json
import random
from mindaffectBCI import stimseq
from enum import IntEnum
from mindaffectBCI.presentation.screens.basic_screens import ScreenSequence, WaitScreen, InstructionScreen
from mindaffectBCI.presentation.screens.QueryDialogScreen import QueryDialogScreen
from math import nan
import pyglet
import mindaffectBCI.presentation.selectionMatrix as selectionMatrix
from mindaffectBCI.presentation.selectionMatrix import window
from mindaffectBCI.noisetag import Noisetag
from mindaffectBCI.decoder.utils import search_directories_for_file, intstr2intkey
from mindaffectBCI.presentation.screens.sound_flash_psypy import SoundFlash, SoundFlashScreen, VideoSoundFlashScreen
import os

# set psychopy audio library + latency mode and import the sound module
import psychtoolbox as ptb
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
prefs.hardware['audioLatencyMode'] = 3  # 3 Aggressive exclusive mode
from psychopy import sound as snd # needs to be imported after setting the prefs

def print_evt(*args, **kwargs):
    print("args={}  kwargs={}".format(args, kwargs))


class AudioBehaviouralTestScreen(ScreenSequence):
    class SubScreens(IntEnum):
        StartScreen = 1
        EyesClosedScreen = 5
        EyesOpenScreen = 6
        StimulusInstructScreen = 8
        TrialStartScreen = 10
        TrialTypeScreen = 15
        StimulusScreen = 20
        BehaviouralScreen = 30
        BehaviouralResultsScreen = 40
        EffortScreen = 50
        EffortStatisticsScreen = 60
        BaselineScreen = 70
        SrtScreen = 80
        SrtResultScreen = 75
        SrtTrialStartScreen = 85
        SrtInstructScreen = 90
        EndScreen = 100

    start_instruct = "Audio Behavioural testing to begin.\n For video trials watch the video and ignore the sounds \n For audio trials look at the fixation cross adn ignore the video, you will be asked to repeat the last digit!\nPress <space> to continue"
    end_instruct = "That's it!\n Thanks for participating"

    eyes_open_instruct = "Please look at the fixation point for 5 seconds and relax\n\n\n\n\n"
    eyes_closed_instruct = "Please close you eyes for 5 seconds and relax\n\n\n\n\n"

    srt_start_instruct = "Speech reception threshold estimation to begin.\nLook at the red fixation point\nYou will hear spoken digits in background noise.\nYou will be asked to repeat the last digit, so listen carefully!\nPress <space> to continue"
    srt_instruct = "Speech reception threshold estimation, please enter (your best guess) of the last digit you heard\nOr 'd' for don't know.\nDigit:   "

    behavioural_instruct = "Please enter (your best guess) of the last digit you heard\nOr 'd' for don't know.\nDigit:  "

    def behavioural_callback(self, userinput: str):
        """ callback when user behavioural feedback is given, check input if valid - raise error if not, send log message if isvalid"""
        userinput = userinput.strip()
        if userinput == 'd':
            userinput = -1
        userinput = int(userinput)
        if userinput < -1 or userinput >= self.stimulus_screen.nsymb:
            raise ValueError
        self.behavioural_id = userinput
        if self.start_srt:
            self.noisetag.log('{{ SRTsignal_level:{}, true_id:{}, behavioural_id:{} }}'.format(
                self.signal_level, self.target_id, self.behavioural_id))
        else:
            self.noisetag.log('{{ signal_level:{}, true_id:{}, behavioural_id:{} }}'.format(
                self.signal_level, self.target_id, self.behavioural_id))

    effort_instruct = "Please enter your listening effort for the last trial.\n" + \
                      "Using a 5 point scale.\n" + \
                      "Where 1=No-effort, 5=Most-Intense-Effort\n\n" + \
                      "Effort:  "

    def effort_callback(self, userinput: str):
        """ callback to validate and log user effort scores """
        userinput = int(userinput)
        if userinput <= 0 or userinput > 5:
            raise ValueError
        self.effort_level = userinput
        self.noisetag.log('{{ signal_level:{}, effort_level:{} }}'.format(self.signal_level, self.effort_level))

    def __init__(self, window: pyglet.window, noisetag: Noisetag, symbols, nTrials: int = 10, nSrtTrials: int = 12,
                 nSrtIt: int = 2, srtTrialLen: int = 3, srtWinSize: int = 11, trial_len_range: int = 10, maxlevels=8,
                 nbaseline_epochs: int = 3, visual_flash: bool = True, soa: int = 30, jitter: int = 5, framesperbit: int = 1,
                 sendEvents: bool = True, active_passive: bool = True,
                 state2vol:
                 dict = {"0": 0, "1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "5": 0.5, "6": 0.6, "7": 0.8, "8": 1.0},
                 background_noise: str = None, **kwargs):

        self.trli = 0
        self.nTrials, self.nSrtTrials, self.nSrtIt, self.trial_len_range, self.srt_trial_len, self.srtWinSize, self.maxlevels, self.soa, self.jitter, self.nbaseline_epochs, self.framesperbit, self.sendEvents, self.active_passive = (
            nTrials, nSrtTrials, nSrtIt, trial_len_range, srtTrialLen, srtWinSize, maxlevels, soa, jitter, nbaseline_epochs, framesperbit, sendEvents, active_passive)
        # ensure is range
        if not hasattr(self.trial_len_range, '__iter__'):
            self.trial_len_range = [self.trial_len_range, self.trial_len_range]

        self.wait_screen = WaitScreen(window, duration=500, waitKey=False)
        self.eyes_screen = InstructionScreen(window, 'EyesTest', duration=5000, waitKey=False, fixation=True)
        self.state2vol = intstr2intkey(state2vol)
        self.state2vol_copy = intstr2intkey(state2vol)
        self.stimulus_screen = VideoSoundFlashScreen(
            window, noisetag, symbols, visual_flash=visual_flash, state2vol=state2vol, waitKey=False,
            background_noise=background_noise, **kwargs)
        self.behavioural_screen = QueryDialogScreen(
            window, self.behavioural_instruct, input_callback=self.behavioural_callback)
        self.behavioural_results_screen = InstructionScreen(window, 'Results', duration=1000, waitKey=False)
        self.trialtype_screen = InstructionScreen(window, 'trial Type', duration=1500, waitKey=False)
        self.effort_screen = QueryDialogScreen(window, self.effort_instruct, input_callback=self.effort_callback)
        super().__init__(window, noisetag, symbols, **kwargs)

    def reset(self):
        super().reset()
        self.trli = 0
        self.ncorrect = 0
        self.reset_srt()
        self.nSrtIt = 2
        self.stimulus_screen.state2vol = self.state2vol_copy
        self.state2vol = self.state2vol_copy
        self.reset_behavioural_statistics()
        # make a nicely permuted trial sequence, concatenate permute over levels
        self.trial_seq = stimseq.mkRandPerm(width=self.maxlevels, nEvents=self.nTrials+1)
        self.next_screen = self.SubScreens.StartScreen
        self.transitionNextPhase()

    def reset_behavioural_statistics(self):
        self.behavioural_statistics = {i: {"correct": [], "effort": []} for i in range(1, self.maxlevels+1)}

    def reset_srt(self):
        self.trli_srt = 0
        self.srt = [{"state": 0, "volume": self.state2vol[0]}]
        self.stimulus_screen.state2vol = self.state2vol

    def summarize_behavioural_statistics(self):
        summary = '\nlevel N  %correct  effort (+/-)'
        for level, stats in self.behavioural_statistics.items():
            correct, effort = (stats['correct'], stats['effort'])
            per_correct = sum(correct)/max(1, len(correct))
            effort_mu = sum(effort)/max(1, len(effort))
            effort_range = (max(effort)-min(effort)) / 2 if len(effort) > 0 else nan
            summary += "\n{:3.0f} {:2d} {:8d}  {:6.1f} ({:3.1f})".format(level,
                                                                         len(correct), int(per_correct*100), effort_mu, effort_range)
        return summary

    def set_subj_tailored_state2vol(self):
        # custom state2vol with n=maxlevel levels where level 1 is at SRT50
        # TODO not use a subset of the levels but create new exact levels
        if self.active_passive and self.nSrtIt == 0:  # quick hack to get passive/active trials
            self.stimulus_screen.state2vol = {0: 0, 1: self.srt[-1]["volume"], 2: self.srt[-1]["volume"]}
            self.noisetag.log('{{tailored_state2vol:{}}}'.format(self.stimulus_screen.state2vol))
            print("Tailored state2vol={}".format(self.stimulus_screen.state2vol))
        elif self.active_passive and self.nSrtIt > 0:
            db = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
            self.state2vol = {i+1: self.srt[-1]["volume"]*(10**(exp/20)) for i, exp in enumerate(db)}
            self.state2vol[0] = 0
            self.noisetag.log('{{tailored_state2vol:{}}}'.format(self.state2vol))
            print("Tailored state2vol={}".format(self.state2vol))
        else:  # pick a subset of the levels with as the most difficult level the levels closest to srt
            self.stimulus_screen.state2vol = {
                k: self.state2vol[round(self.srt[-1]['state'] + int(k) - 1)] for k in self.state2vol.keys()
                if int(k) <= self.maxlevels}

    def transitionNextPhase(self):
        """function to manage the transition between sub-screens.  Override to implement
           your desired screen transition logic.
        """
        print("stage transition")

        # move to the next stage
        if self.next_screen is not None:
            self.current_screen = self.next_screen
            self.next_screen = None
        else:  # ask the current screen what to do
            self.current_screen = None
            self.next_screen = None

        if self.current_screen == self.SubScreens.StartScreen:  # main menu
            print("start screen")
            self.wait_screen.reset()
            self.wait_screen.duration = 1000
            self.screen = self.wait_screen
            self.next_screen = self.SubScreens.SrtInstructScreen

        elif self.current_screen == self.SubScreens.EyesOpenScreen:  # eyes open
            print("eyes open screen")
            self.eyes_screen.reset()
            self.eyes_screen.set_text(self.eyes_open_instruct)
            self.noisetag.log("{{ 'phase_start':'eyes_closed' }}")
            self.screen = self.eyes_screen
            self.next_screen = self.SubScreens.EyesClosedScreen

        elif self.current_screen == self.SubScreens.EyesClosedScreen:  # eyes closed
            # TODO[]: add beep at the end of the duration
            print("eyes closed screen")
            self.eyes_screen.reset()
            self.eyes_screen.set_text(self.eyes_closed_instruct)
            self.noisetag.log("{{ 'phase_start':'eyes_open' }}")
            self.screen = self.eyes_screen

            self.next_screen = self.SubScreens.SrtInstructScreen

        elif self.current_screen == self.SubScreens.BaselineScreen:  # make and play stimulus
            self.signal_level = 0
            self.trial_len = self.trial_len_range[1]
            print("signallevel={} triallen={}".format(self.signal_level, self.trial_len))
            self.stimsequence = stimseq.mkRandPerm(
                width=self.stimulus_screen.nsymb, nEvents=self.trial_len+1, level=self.signal_level)
            self.stimsequence = stimseq.upsample_with_jitter(self.stimsequence, soa=self.soa, jitter=self.jitter)

            # set the noisetag player to use this sequence
            self.stimulus_screen.reset()
            self.stimulus_screen.set_sentence("Baseline. \nlook at the red cross and relax.\nIgnore the sounds!!!")
            # set the noisetag to play this sequence once through
            self.stimulus_screen.noisetag.set_stimSeq(self.stimsequence)
            self.stimulus_screen.noisetag.startSingleTrial(
                numframes=len(self.stimsequence.stimSeq) * self.framesperbit, tgtidx=-1, framesperbit=self.framesperbit,
                sendEvents=self.sendEvents)
            # set the screen and go
            self.screen = self.stimulus_screen
            self.next_screen = self.SubScreens.SrtInstructScreen

        elif self.current_screen == self.SubScreens.SrtInstructScreen:
            print("srt instruct screen")
            self.instruct_screen.reset()
            if self.nSrtTrials == 0:
                self.start_srt = 0
                self.instruct_screen.set_text("SRT estimation disabled, continuing to main epxeriment")
                self.next_screen = self.SubScreens.StimulusInstructScreen
            else:
                self.start_srt = 1
                self.nSrtIt -= 1
                self.srt_phase1 = 2
                self.stimulus_screen.set_vid_state(None)  # disable video playback for srt trials
                self.instruct_screen.set_text(self.srt_start_instruct)
                self.next_screen = self.SubScreens.SrtTrialStartScreen
            self.screen = self.instruct_screen

        elif self.current_screen == self.SubScreens.SrtTrialStartScreen:
            self.wait_screen.reset()
            self.wait_screen.duration = 1500
            self.screen = self.wait_screen
            # after wait do next srt estimation trial or start main epxeriment
            self.trli_srt = self.trli_srt + 1
            if self.trli_srt <= self.nSrtTrials:
                self.next_screen = self.SubScreens.SrtScreen
            else:
                # log found SRT
                avg_vol = sum([rsp['volume'] for rsp in self.srt][-self.srtWinSize:])/self.srtWinSize
                avg_state = sum([rsp['state'] for rsp in self.srt][-self.srtWinSize:])/self.srtWinSize
                self.srt.append({"state": avg_state, "volume": avg_vol})
                self.noisetag.log("{{'SRT':{} }}".format(self.srt))
                # Update state2vol
                self.set_subj_tailored_state2vol()
                self.reset_srt()
                self.next_screen = self.SubScreens.SrtInstructScreen
                if self.nSrtIt == 0:  # SRT estimation done
                    self.start_srt = 0
                    self.stimulus_screen.set_vid_state(2)  # enable video on state 2
                    self.next_screen = self.SubScreens.StimulusInstructScreen

        elif self.current_screen == self.SubScreens.SrtScreen:
            self.signal_level = self.srt[-1]['state']
            print("SRT test lvl={}".format(self.signal_level))
            # make a random digit order sequence
            self.stimsequence = stimseq.mkRandPerm(
                width=self.stimulus_screen.nsymb, nEvents=self.srt_trial_len, level=self.signal_level)
            # get the active objects at the end of the sequence
            self.target_id = [i for i, v in enumerate(self.stimsequence.stimSeq[-1]) if v > 0]
            if self.nbaseline_epochs and self.nbaseline_epochs > 0:
                # make n_baseline_epochs list of len(stimSeq[0]) of 0's for baseline epochs
                baseline = [[0 for _ in self.stimsequence.stimSeq[0]] for _ in range(self.nbaseline_epochs)]
                self.stimsequence.stimSeq = baseline + self.stimsequence.stimSeq  # add baseline as a prefix                       ss
            self.stimsequence = stimseq.upsample_with_jitter(
                self.stimsequence, soa=self.soa, jitter=self.jitter)  # do we want jitter here?

            # set the noisetag player to use this sequence
            self.stimulus_screen.reset()
            self.stimulus_screen.set_sentence("Listen to the digits.\n {}/{}".format(self.trli_srt, self.nSrtTrials))
            # set the noisetag to play this sequence once through
            self.stimulus_screen.noisetag.set_stimSeq(self.stimsequence)
            self.stimulus_screen.noisetag.startSingleTrial(numframes=len(
                self.stimsequence.stimSeq)*self.framesperbit, tgtidx=-1, framesperbit=self.framesperbit, sendEvents=False)
            # set the screen and go
            self.screen = self.stimulus_screen
            self.next_screen = self.SubScreens.BehaviouralScreen

        elif self.current_screen == self.SubScreens.StimulusInstructScreen:  # main menu
            print("stimulus screen")
            self.instruct_screen.reset()
            self.instruct_screen.set_text(self.start_instruct)
            self.screen = self.instruct_screen
            self.next_screen = self.SubScreens.TrialTypeScreen

        elif self.current_screen == self.SubScreens.TrialStartScreen:
            self.wait_screen.reset()
            self.noisetag.sendStimulusEvent(
                [81] * len(self.stimulus_screen.objects),
                objIDs=list(range(1, len(self.stimulus_screen.objects) + 1)))  # MNE compatible baseline logging
            self.wait_screen.duration = 1500
            self.screen = self.wait_screen

            # after wait either do next stimulus or quit
            self.trli = self.trli + 1
            if self.trli < len(self.trial_seq.stimSeq):
                self.next_screen = self.SubScreens.StimulusScreen
            else:
                self.next_screen = self.SubScreens.EndScreen

        elif self.current_screen == self.SubScreens.StimulusScreen:  # make and play stimulus
            # # pick the noise level -- last id with non-zero value in the trail_seq
            self.signal_level = [i+1 for i, v in enumerate(self.trial_seq.stimSeq[self.trli]) if v > 0][0]
            self.trial_len = random.randint(self.trial_len_range[0], self.trial_len_range[1])
            print("signallevel={} triallen={}".format(self.signal_level, self.trial_len))
            # make a random digit order sequence
            self.stimsequence = stimseq.mkRandPerm(
                width=self.stimulus_screen.nsymb, nEvents=self.trial_len, level=self.signal_level)
            # get the active objects at the end of the sequence
            self.target_id = [i for i, v in enumerate(self.stimsequence.stimSeq[-1]) if v > 0]
            # BODGE[]: add nbaseline all zero events at the start for the baseline period
            if self.nbaseline_epochs and self.nbaseline_epochs > 0:
                # make n_baseline_epochs list of len(stimSeq[0]) of 0's for baseline epochs
                baseline = [[0 for _ in self.stimsequence.stimSeq[0]] for _ in range(self.nbaseline_epochs)]
                self.stimsequence.stimSeq = baseline + self.stimsequence.stimSeq  # add baseline as a prefix
            self.stimsequence = stimseq.upsample_with_jitter(self.stimsequence, soa=self.soa, jitter=self.jitter)

            self.stimulus_screen.reset()
            # set the noisetag to play this sequence once through
            self.stimulus_screen.noisetag.set_stimSeq(self.stimsequence)
            self.stimulus_screen.noisetag.startSingleTrial(
                numframes=len(self.stimsequence.stimSeq) * self.framesperbit, tgtidx=-1, framesperbit=self.framesperbit,
                sendEvents=self.sendEvents)

            # set desired instruct when running passive_active experiment
            self.stimulus_screen.set_sentence("Listen to the digits.\n {}/{}".format(self.trli, self.nTrials))
            self.next_screen = self.SubScreens.BehaviouralScreen
            if self.active_passive and self.signal_level == 2:
                self.stimulus_screen.set_sentence("Watch the video.\n {}/{}".format(self.trli, self.nTrials))
                self.stimulus_screen.set_vid_trggr()
                self.next_screen = self.SubScreens.TrialTypeScreen
            # set the screen and go
            self.screen = self.stimulus_screen

        elif self.current_screen == self.SubScreens.TrialTypeScreen:
            if self.active_passive:
                try:
                    trial_type = "video" if [
                        i+1 for i, v in enumerate(self.trial_seq.stimSeq[self.trli+1]) if v > 0][0] == 2 else "audio"
                    print("trial type: {}".format(trial_type))
                    self.trialtype_screen.reset()
                    self.trialtype_screen.set_text("Upcoming trial type: {}".format(trial_type))
                    self.screen = self.trialtype_screen
                except IndexError:
                    pass
            self.next_screen = self.SubScreens.TrialStartScreen

        elif self.current_screen == self.SubScreens.BehaviouralScreen:  # check behavioural performance
            # TODO[]: also record the response time
            self.behavioural_screen.reset()
            self.screen = self.behavioural_screen
            if self.start_srt:
                self.next_screen = self.SubScreens.SrtResultScreen
            else:
                self.next_screen = self.SubScreens.EffortScreen

        elif self.current_screen == self.SubScreens.SrtResultScreen:
            # check if behav response is correct
            self.behavioural_iscorrect = self.behavioural_id in self.target_id
            print("BEHAV:={}".format(self.behavioural_iscorrect))
            # when first correct answer is given, we decrease the stepsize
            if self.behavioural_iscorrect:
                self.srt_phase1 = 1
            # update
            if self.behavioural_iscorrect:
                self.srt.append({"state": self.signal_level - 1, "volume": self.state2vol[self.signal_level - 1]})
            else:
                self.srt.append({"state": self.signal_level + 1*self.srt_phase1,
                                "volume": self.state2vol[self.signal_level + 1*self.srt_phase1]})

            self.next_screen = self.SubScreens.SrtTrialStartScreen
            if self.trli_srt == self.nSrtTrials:  # % self.srtWinSize == 0 or self.trli_srt == self.nSrtTrials: #debuggig
                # show current srt
                self.instruct_screen.reset()
                self.instruct_screen.set_text("SRT50 Lvl={:.2f}".format(
                    sum([rsp['volume'] for rsp in self.srt][-self.srtWinSize:])/self.srtWinSize))
                self.screen = self.instruct_screen

            else:
                # jump directly to the next phase
                self.transitionNextPhase()

        elif self.current_screen == self.SubScreens.BehaviouralResultsScreen:
            # give feedback on the behavioural response

            self.behavioural_iscorrect = self.behavioural_id in self.target_id

            self.behavioural_results_screen.reset()

            if self.behavioural_iscorrect:
                self.ncorrect = self.ncorrect + 1
                self.behavioural_results_screen.set_text("Correct!\n\n {}/{} correct".format(self.ncorrect, self.trli))
            else:
                self.behavioural_results_screen.set_text(
                    "Sorry, Wrong! it was {}\n\n {}/{} correct".format(self.target_id[0],
                                                                       self.ncorrect, self.trli))

            self.screen = self.behavioural_results_screen
            self.next_screen = self.SubScreens.EffortStatisticsScreen

        elif self.current_screen == self.SubScreens.EffortScreen:  # effort recording

            self.effort_screen.reset()
            self.screen = self.effort_screen
            # loop to next stimulus
            self.next_screen = self.SubScreens.BehaviouralResultsScreen

        elif self.current_screen == self.SubScreens.EffortStatisticsScreen:

            # update the statistics info, for this signal level
            self.behavioural_statistics[self.signal_level]['correct'].append(self.behavioural_iscorrect)
            self.behavioural_statistics[self.signal_level]['effort'].append(self.effort_level)

            # loop to next stimulus
            self.next_screen = self.SubScreens.TrialTypeScreen
            if self.trli % 10 == 0:
                behavioural_summary = self.summarize_behavioural_statistics()

                # show the summary
                self.instruct_screen.reset()
                self.instruct_screen.set_text("Behavioural Results\n\n{}".format(behavioural_summary))
                self.screen = self.instruct_screen
            else:
                # jump directly to the next phase
                self.transitionNextPhase()

        elif self.current_screen == self.SubScreens.EndScreen:  # effort recording

            behavioural_summary = self.summarize_behavioural_statistics()
            self.noisetag.log(json.dumps(self.behavioural_statistics))
            self.noisetag.log(behavioural_summary)

            # show the summary
            self.instruct_screen.reset()
            self.instruct_screen.set_text("Behavioural Results\n\n{}".format(behavioural_summary))

            self.screen = self.instruct_screen
            self.next_screen = None

        else:  # end
            print('quit')
            self.screen = None

    #########################################
    # BODGE: empty functions to make work as a calibration screen
    def set_grid(self, **kwargs):
        pass

    def setliveFeedback(self, livefeedback: bool):
        pass

    def setshowNewTarget(self, shownewtarget: bool):
        pass

    def set_sentence(self, sentence: str):
        pass


def run(symbols=None,
        stimfile: str = None, stimseq: str = None, state2color: dict = None,
        fullscreen: bool = None, windowed: bool = None, fullscreen_stimulus: bool = True, host: str = None,
        **kwargs):
    """ run this application

    Args:

    """
    # configuration message for logging what presentation is used
    global configmsg
    configmsg = "{}".format(dict(component=__file__, args=locals()))

    selectionMatrix.init_noisetag_and_window(stimseq, host, fullscreen)
    selectionMatrix.nt.connect()

    # make the screen manager object which manages the app state
    ss = AudioBehaviouralTestScreen(selectionMatrix.window, selectionMatrix.nt, symbols=symbols, nTrials=10, **kwargs)
    selectionMatrix.run_screen(ss)


if __name__ == "__main__":
    args = selectionMatrix.parse_args()

    # setattr(args,'calibration_screen','mindaffectBCI.examples.presentation.sound_flash.SoundFlashScreen')
    setattr(args, 'calibration_screen', 'mindaffectBCI.examples.presentation.sound_flash_psypy.AudioBehaviouralTestScreen')
    # setattr(args,'symbols',[["+|beep\\tone60_3_500hz.wav|beep\\tone60_3_1000hz.wav|beep\\tone60_3_2000hz.wav"]])
    # setattr(args,'stimfile','oddball.txt')
    #setattr(args,'symbols',[[" |chirp\\800-1200-gauss.wav"]])
    # setattr(args,'calibration_screen','mindaffectBCI.examples.presentation.sound_flash.NoiseLevelSoundFlashScreen')
    setattr(args, 'symbols', [["0|digits\\MAE_0A.wav", "1|digits\\MAE_1A.wav",
            "2|digits\\MAE_2A.wav", "3|digits\\MAE_3A.wav"]])

    # setattr(args,'symbols',[["500|chirp\\400-600-gauss.wav","1000|chirp\\800-1200-gauss.wav","2000|chirp\\1600-2400-gauss.wav","4000|chirp\\3000-5000-gauss.wav"]])

    setattr(args, 'stimfile', 'level8_gold_4obj_interleaved.txt')
    setattr(args, 'framesperbit', 30)
    setattr(args, 'calibration_trialduration', 10)
    setattr(args, 'cueduration', 1.5)
    setattr(args, "fullscreen", False)
    setattr(args, 'intertrialduration', 2)
    setattr(args, 'calibration_args', {"permute": False, "startframe": "random"})
    #setattr(args,'calibration_screen_args',{'visual_flash':False, "spatialize":True, "state2vol":{"1":0.00001,"2":0.0001,"3":0.001,"4":0.01,"5":0.1,"6":0.2,"7":0.5,"8":1.0},"use_audio_timestamp":True,"background_noise":"noise\\babble.wav"})
    setattr(
        args, 'calibration_screen_args',
        {'visual_flash': False, "spatialize": True, "nTrials": 5,
         "state2vol": {"1": 0.1, "2": 0.15, "3": 0.2, "4": 0.25, "5": 0.3, "6": 0.5, "7": 0.7, "8": 1.0},
         "use_audio_timestamp": True, "background_noise": "noise\\babble.wav"})
    selectionMatrix.run(**vars(args))
    # run(**vars(args))
