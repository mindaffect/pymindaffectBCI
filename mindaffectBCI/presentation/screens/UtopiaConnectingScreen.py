import pyglet
from mindaffectBCI.presentation.screens.basic_screens import InstructionScreen

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class ConnectingScreen(InstructionScreen):
    '''Modified instruction screen with waits for the noisetag to connect to the decoder'''

    prefix_text = "Welcome to the mindaffectBCI\n\n"
    searching_text = "Searching for the mindaffect decoder\n\nPlease wait"
    trying_text = "Trying to connect to: %s\n Please wait"
    connected_text = "Success!\nconnected to: %s"
    query_text = "Couldnt auto-discover mindaffect decoder\n\nPlease enter decoder address: %s"
    drawconnect_timeout_ms = 50
    autoconnect_timeout_ms = 5000

    def __init__(self, window, noisetag, duration=150000):
        super().__init__(window, self.prefix_text + self.searching_text, duration, False)
        self.noisetag = noisetag
        self.host = None
        self.port = -1
        self.usertext = ''
        self.stage = 0

    def draw(self, t):
        '''check for results from decoder.  show if found..'''
        if not self.isRunning:
            super().draw(t)
            return

        if not self.noisetag.isConnected():
            if self.stage == 0: # try-connection
                print('Not connected yet!!')
                self.noisetag.connect(self.host, self.port,
                                      queryifhostnotfound=False,
                                      timeout_ms=self.drawconnect_timeout_ms)
                if self.noisetag.isConnected():
                    self.set_text(self.prefix_text + self.connected_text%(self.noisetag.gethostport()))
                    self.t0 = self.getTimeStamp()
                    self.duration = 1000
                    # # P=Target Prediction F=Target Dist S=selection N=new-target M=mode-change E=stimulus-event Q=signal-quality
                    # # self.noisetag.subscribe("MSPQFT")
                elif self.elapsed_ms() > self.autoconnect_timeout_ms:
                    # waited too long, giveup and ask user
                    self.stage = 1
                    # ensure old key-presses are gone
                    self.window.last_text = None
                    self.window.last_key_press = None

            elif self.stage == 1:  # query hostname
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
                        # set as new host to try
                        self.host = self.usertext
                        self.usertext = ''
                        self.set_text(self.prefix_text + self.trying_text%(self.host))
                        self.stage = 0 # back to try-connection stage
                    elif self.window.last_text:
                        # add to the host string
                        self.usertext += self.window.last_text
                    self.window.last_text = None
                if self.stage == 1: # in same stage
                    # update display with user input
                    self.set_text(self.prefix_text + self.query_text%(self.usertext))
        super().draw(t)

