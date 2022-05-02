import pyglet
from mindaffectBCI.presentation.screens.basic_screens import Screen

class MinimalStimulusScreen(Screen):
    '''Screen which shows a blank screen for duration or until key-pressed'''
    def __init__(self, window, noisetag=None):
        super().__init__(window)
        self.noisetag = noisetag
        # make a batch for rendering
        self.batch = pyglet.graphics.Batch()

        # make the text object, in the middle of the screen, and add to the batch
        self.text =pyglet.text.Label("Hello World", font_size=12, 
                                        x=self.window.width//2, y=self.window.height//2,
                                        color=(255, 255, 255, 255),
                                        anchor_x='center', anchor_y='center',
                                        batch=self.batch)
        self.reset()

    def reset(self):
        self.nframe = 0

    def is_done(self):
        ''' we are finished after 200 frames ~= 4s '''
        return not self.noisetag.isRunning()

    def draw(self, t):
        ''' draw the batch '''
        self.nframe = self.nframe + 1

        # get the stimulus state from the noisetag object
        if self.noisetag:
            # log the previour stimulus state info
            self.noisetag.sendStimulusState()
            # get the next stimulus state to display
            try:
                stimulusState, target_idx, objIDs, sendEvent = self.noisetag.getNextStimulusState()
            except StopIteration:
                pass

            # update the display to reflect the current stimulus state
            col = (255,255,255,255)
            if stimulusState[0]==0 : # state 0 -> grey
                col = (5,5,5,255)
            elif stimulusState[0]==1: # state 1 -> red
                col = (255,0,0,255)
            elif stimulusState[0]==2: # state 2 -> green
                col = (0,255,0,255)
            # update the label color
            self.text.color = col

        self.batch.draw()



if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    window = initPyglet(width=640, height=480)
    nt = Noisetag(stimSeq='level2_gold.txt')
    screen = MinimalStimulusScreen(window,noisetag=nt)
    nt.startFlicker()
    run_screen(window, screen)