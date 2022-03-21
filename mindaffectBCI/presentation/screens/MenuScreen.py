from mindaffectBCI.presentation.screens.basic_screens import InstructionScreen

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class MenuScreen(InstructionScreen):
    '''Screen which shows a textual instruction for duration or until key-pressed'''
    def __init__(self, window, text, valid_keys):
        super().__init__(window, text, 99999999, True)
        self.valid_keys = valid_keys
        self.key_press = None
        #print("Menu")

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
            # Note: lines render top to bottom, but Y is bottom to top!
            print('mouse press in line {}  = {}'.format(li, lines[::-1][li]))
            # convert to button press on this line
            self.window.last_key_press = lines[::-1][li][0]

    def draw(self,t):
        self.handle_mouse_events()
        super().draw(t)

    def is_done(self):
        # check termination conditions
        if not self.isRunning:
            self.isDone = False
            return self.isDone

        # valid key is pressed
        if self.window.last_key_press:
            self.key_press = self.window.last_key_press
            if self.key_press in self.valid_keys:
                self.isDone = True
            self.window.last_key_press = None
            self.window.last_text = None

        # time-out
        if self.elapsed_ms() > self.duration:
            self.isDone = True
        return self.isDone

