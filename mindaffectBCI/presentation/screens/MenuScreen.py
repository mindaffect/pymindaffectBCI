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
from mindaffectBCI.presentation.screens.basic_screens import InstructionScreen

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class MenuScreen(InstructionScreen):
    '''Screen which shows a textual instruction for duration or until key-pressed'''
    def __init__(self, window, text:str, title:str=None, valid_keys=None):
        """Screen which shows a textual instruction for duration or until key-pressed

        Args:
            window (pyglet window): the window to draw into
            text (str|list-of-str): the menu-text
            valid_keys (list-of-str): set of valid key-presses to select a menu option
        """        
        super().__init__(window, text=text, title=title, duration=99999999, waitKey=True)
        self.valid_keys = valid_keys
        self.key_press = None
        #print("Menu")

    def reset(self):
        super().reset()
        # clear the key/mouse state
        self.window.last_mouse_release = None
        self.window.last_text=None
        self.window.last_key_press=None

    def handle_mouse_events(self):
        """handle mouse events, by mapping mouse presses to equivalent key presses
        """
        if self.window.last_mouse_release is None: return
        mx,my,mb,mm = self.window.last_mouse_release
        self.window.last_mouse_release = None
        # check which (if any) menu line this click was in
        # get the bounding box of the text
        x,y,w,h = self.instructLabel.x, self.instructLabel.y, self.instructLabel.content_width, self.instructLabel.content_height
        if self.instructLabel.anchor_x == 'center': # move to the bbox start
            x = x - self.instructLabel.width/2
        if self.instructLabel.anchor_y == 'center':
            y = y - self.instructLabel.content_height/2
        # get the line the mouse was released in
        if x < mx and mx < x+w and y < my and mx < y+h:
            # assume the height is equally spaced in the menu lines
            lines = self.instructLabel.text.split('\n')
            line_height = h / len(lines)
            li = int((my-y)//line_height)
            try:
                # Note: lines render top to bottom, but Y is bottom to top!
                print('mouse press in line {}  = {}'.format(li, lines[::-1][li]))
                # convert to button press on this line
                self.window.last_key_press = lines[::-1][li][0]
            except IndexError:
                pass

    def draw(self,t):
        super().draw(t)
        self.handle_mouse_events()

    def is_done(self):
        # check termination conditions
        if not self.isRunning:
            self.isDone = False
            return self.isDone

        # valid key is pressed
        if self.window.last_key_press:
            self.key_press = self.window.last_key_press
            if self.valid_keys is not None or self.key_press in self.valid_keys:
                self.isDone = True
            self.window.last_key_press = None
            self.window.last_text = None

        # time-out
        if self.elapsed_ms() > self.duration:
            self.isDone = True
        return self.isDone



if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    window = initPyglet(width=640, height=480)
    screen = MenuScreen(window,['0) First entry','1) Second Entry'])
    run_screen(window, screen)

