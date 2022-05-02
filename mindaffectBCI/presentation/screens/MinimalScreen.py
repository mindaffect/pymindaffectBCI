import pyglet
from mindaffectBCI.presentation.screens.basic_screens import Screen

class MinimalScreen(Screen):
    '''Screen which shows a blank screen for duration or until key-pressed'''
    def __init__(self, window):
        super().__init__(window)
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
        self.isRunning = False
        self.isDone = False
        self.nframe = 0

    def is_done(self):
        ''' we are finished after 200 frames ~= 4s '''
        return self.nframe > 60*4

    def draw(self, t):
        ''' draw the batch '''
        self.nframe = self.nframe + 1
        self.batch.draw()


if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    window = initPyglet(width=640, height=480)
    screen = MinimalScreen(window)
    run_screen(window, screen)