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
from mindaffectBCI.presentation.screens.basic_screens import InstructionScreen


#-----------------------------------------------------------------
class QueryResponseScreen(InstructionScreen):
    '''Modified instruction screen queries the user for textual input and finishes on key press'''

    def __init__(self, window, text:str='This is a test dialog screen.\nEnter some text:', duration=50000, waitKey:bool=False, waitMouse:bool=False, input_callback=None):
        """simple screen for asking for input from the user

        Args:
            window (pyglet.window): pyglet window
            text (str): the text to show
            duration (int, optional): max length of time to show the window. Defaults to 50000.
            waitKey (bool, optional): finish on key-press. IGNORED!. Defaults to False.
            waitMouse (bool, optional): finish on mouse-press.  IGNORED.  Defaults to False
            input_callback ([type], optional): function to validate user input as string, raise ValueError if invalid input. Defaults to None.
        """
        super().__init__(window, text=text, duration=duration, waitKey=False, waitMouse=False)
        self.query = text
        self.usertext = ''
        self.input_callback = input_callback

    def reset(self):
        super().reset()
        # clear text on reset
        self.usertext = ''
        self.set_text(self.query)

    def draw(self, t):
        '''grab key presses, process, validate, finish is good.'''
        if self.window.last_key_press:
            self.window.last_key_press = None
        if self.window.last_text:
            self.usertext += str(self.window.last_text)
            if self.input_callback is None:
                # No validation so we are done.
                self.isDone = True
            else:
                # validate first
                try:
                    self.input_callback(self.usertext)
                    self.isDone = True
                except ValueError:
                    # todo []: flash to indicate invalid input
                    self.usertext = ''
            self.window.last_text = None
            # update display with user input
            self.set_text(self.query + self.usertext)
        super().draw(t)



if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    window = initPyglet(width=640, height=480)
    screen = QueryResponseScreen(window)
    run_screen(window, screen)