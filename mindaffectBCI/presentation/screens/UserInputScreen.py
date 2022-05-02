from mindaffectBCI.presentation.screens.basic_screens import WaitScreen
import pyglet

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class UserInputScreen(WaitScreen):
    '''Modified screen to enter text'''

    def __init__(self, window, callback=None, title_text=None, text=None, valid_text=None, duration=150000, logo="Mindaffect_Logo.png", **kwargs):
        super().__init__(window, duration=duration, waitKey=False, logo=logo, **kwargs)
        self.valid_text = valid_text
        self.usertext = ''
        self.callback = callback

        # initialize the instructions screen
        self.titleLabel = pyglet.text.Label(x=int(self.window.width*.1),
                                            y=self.window.height,
                                            anchor_x='left',
                                            anchor_y='top',
                                            font_size=24,
                                            color=(255, 255, 255, 255),
                                            multiline=True,
                                            width=int(self.window.width*.8))
        self.set_title(title_text)

        self.inputLabel = pyglet.text.Label(x=self.window.width//2,
                                            y=self.window.height//2,
                                            anchor_x='center',
                                            anchor_y='center',
                                            font_size=24,
                                            color=(255, 255, 255, 255),
                                            multiline=True,
                                            width=int(self.window.width*.8))
        self.set_text(text)

    def set_text(self, text):
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        elif text is None: 
            text=""
        self.inputLabel.begin_update()
        self.inputLabel.text=text
        self.inputLabel.end_update()

    def set_title(self, text):
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        elif text is None: 
            text=""
        self.titleLabel.begin_update()
        self.titleLabel.text=text
        self.titleLabel.end_update()

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        if not self.isRunning:
            WaitScreen.draw(self,t)
            return
        WaitScreen.draw(self,t)

        # query the user for host/port
        # accumulate user inputs
        if self.window.last_key_press:
            if self.window.last_key_press == pyglet.window.key.BACKSPACE:
                # remove last character
                self.usertext = self.usertext[:-1]
            self.window.last_key_press = None
            if self.window.last_text:
                print(self.window.last_text + ":" + str(ord(self.window.last_text)))
            if self.window.last_text == '\n' or self.window.last_text == '\r':
                self.isDone = True
                if self.callback is not None:
                    self.callback(self.usertext)
            elif self.window.last_text:
                if self.valid_text is None or self.window.last_text in self.valid_text:
                    # add to the host string
                    self.usertext += self.window.last_text
            self.window.last_text = None
            self.set_text(self.usertext)

        # draw the screen bits..
        self.titleLabel.draw()
        self.inputLabel.draw()
