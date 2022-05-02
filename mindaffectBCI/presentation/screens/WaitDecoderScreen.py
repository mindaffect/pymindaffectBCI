from mindaffectBCI.presentation.screens.InstructionScreen import InstructionScreen

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class WaitDecoderScreen(InstructionScreen):
    '''Modified instruction screen with waits for the decoder to be running -- as indicated by a signal-quality message'''

    waiting_text = "Waiting for decoder to startup\n\nPlease wait"
    results_text = "Running decoder detected"
    def __init__(self, window, noisetag, duration=20000, waitKey=False):
        self.noisetag = noisetag
        self.qual = None
        super().__init__(window, text=self.waiting_text, duration=duration, waitKey=waitKey)

    def reset(self):
        self.noisetag.clearLastSignalQuality()
        self.qual = None
        super().reset()

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        if not self.isRunning:
            self.reset()
        # check for new qualictions
        qual = self.noisetag.getLastSignalQuality()
        # update text if got qualicted performance
        if qual is not None and self.qual is None:
            self.qual = qual
            self.waitKey = True
            self.set_text(self.results_text)
            self.duration = self.elapsed_ms() + 1000
        super().draw(t)