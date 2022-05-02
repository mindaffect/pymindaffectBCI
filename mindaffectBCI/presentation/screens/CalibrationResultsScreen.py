from mindaffectBCI.presentation.screens.InstructionScreen import InstructionScreen

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class CalibrationResultsScreen(InstructionScreen):
    '''Modified instruction screen with waits for and presents calibration results'''

    waiting_text = "Waiting for performance results from decoder\n\nPlease wait"
    results_text = "Calibration Performance: %3.0f%% Correct\n\n<space> to continue"
    def __init__(self, window, noisetag, duration=20000, waitKey=False):
        self.noisetag = noisetag
        self.pred = None
        super().__init__(window, text=self.waiting_text, duration=duration, waitKey=waitKey)

    def reset(self):
        self.noisetag.clearLastPrediction()
        self.pred = None
        super().reset()

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        if not self.isRunning:
            self.reset()
        # check for new predictions
        pred = self.noisetag.getLastPrediction()
        # update text if got predicted performance
        if pred is not None and (self.pred is None or pred.timestamp > self.pred.timestamp) :
            self.pred = pred
            print("Prediction:{}".format(self.pred))
            self.waitKey = True
            self.set_text(self.results_text%((1.0-self.pred.Perr)*100.0))
            self.duration = self.elapsed_ms() + 2000
        super().draw(t)