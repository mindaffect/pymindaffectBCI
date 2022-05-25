#!/usr/bin/env python3
#  Copyright [c] 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jadref@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files [the "Software"], to deal
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
from mindaffectBCI.presentation.screens.basic_screens import ScreenGraph
from mindaffectBCI.presentation.screens.MenuScreen import MenuScreen

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class SubscreenMenuScreen(ScreenGraph):
    """Screen which shows a textual instruction for duration or until key-pressed"""
    def __init__(self, window, label:str=None,  title:str="Main Menu",
                subscreens:dict={"ins1":["InstructionScreen",{"text":"This is a default start screen....\nPress <space> to continue", "waitKey":True, "duration":1000}],
                            "ins2":["InstructionScreen",{"text":"And this is a second default screen to show transitions", "waitKey":True, "duration":1000}]},
                subscreen_transitions:dict=dict(),
                noisetag=None,
                start_screen:str=None
    ):
        """Screen which makes a menu sub-screens to select from, and allows a user to select which sub-screen to run with keypress or mouse

        Args:
            window (pyglet.window): pyglet window to run in 
            label (str, optional): Human readable name for this screen, also used in menu entries. Defaults to None.
            subscreens (dict, optional): Dictionary of named sub-screens.  Key is the screen name, Value is either a created screen, or a 2-tuple with the fully-qualified screen class name and the arguments to pass to the screen. Defaults to {"ins1":["InstructionScreen",{"text":"This is a default start screen....\nPress <space> to continue"}], "ins2":["InstructionScreen",{"text":"And this is a second default screen to show transitions"}]}.
            subscreen_transitions (dict, optional): Dictionary of transitions between screens.  Key the current-screen name, value is the screen to move to, or function to call to get the screen to transition to. Defaults to {"ins1":"ins2", "ins2":"end"}.
            noisetag (): noisetag object (if needed).  Defaults to None.
            start_screen (str, optional): Name of the screen to start with.  If None then the main menu screen. Defaults to "menu".
        """
        # init the normal subscreen graph
        super().__init__(window,label=label,subscreens=subscreens,subscreen_transitions=subscreen_transitions,start_screen=start_screen,default_screen="menu", subscreen_args={"noisetag":noisetag})
        self.noisetag = noisetag
        # reset the subscreen state
        self.subscreen_transitions = subscreen_transitions
        self.start_screen = start_screen

        # make the menu-screen
        menu_screen = self.init_menu_subscreen(self.subscreens, title=title)
        # add the menu to the sub-screens list
        self.subscreens["menu"]=menu_screen
        # with a transition callback here to get the next screen to call
        self.subscreen_transitions["menu"]=self.call_menu_subscreen
        # re-set the start screen to the new menu screen if wanted
        if start_screen is None or start_screen == "menu":
            self.start_screen = "menu"
            self.current_screen = "menu"
            self.screen = self.subscreens[self.current_screen]

    def init_menu_subscreen(self, subscreens, title:str=None):
        menu_text, menu_keys = [], []
        for i,(k,v) in enumerate(subscreens.items()):
            menu_text.append( "{:d}) {}  ({})".format(i,v.label,k) )
            menu_keys.append( "{:d}".format(i) )
        menu = MenuScreen(self.window,text=menu_text, title=title, valid_keys=menu_keys)
        return menu

    def call_menu_subscreen(self, menu_screen):
        subscreen = "menu"
        # get the menu line first
        menu_text = menu_screen.text
        if isinstance(menu_text,str): menu_text = menu_text.split("\n")
        key_press = menu_screen.key_press if isinstance(menu_screen.key_press,str) else chr(menu_screen.key_press)
        selected_menu_entry = [ l for l in menu_text if l.startswith(key_press) ]
        if len(selected_menu_entry)==0:
            return "menu"
        if len(selected_menu_entry)>1:  print("Warning multple lines matched? taking 1st")
        selected_menu_str = selected_menu_entry[0]
        # find the sub-screen this menu entry refers to
        # by matching either the screen label or the sub-screen key
        subscreen = [ k for k,s in self.subscreens.items() if "({})".format(k) in selected_menu_str ]
        if len(subscreen) == 0: # fall back on matching the class label
            subscreen = [ k for k,s in self.subscreens.items() if s.label in selected_menu_str ]
        # return the first matching entry
        subscreen = subscreen[0] if len(subscreen)>0 else "menu"
        # return the key for this sub-screen
        return subscreen


if __name__=="__main__":
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    window = initPyglet(width=640, height=480)
    # subscreens = {"ins1":["InstructionScreen",{"text":"This is a default start screen....\nPress <space> to continue", "waitKey":True, "duration":1000}],
    #                         "ins2":["InstructionScreen",{"text":"And this is a second default screen to show transitions", "waitKey":True, "duration":1000}]}
    # subscreen_transitions = {}
    # screen = SubscreenMenuScreen[window, subscreens=subscreens, subscreen_transitions=subscreen_transitions]
    # run_screen[window, screen]

    nt = Noisetag(stimSeq="mgold_65_6532.txt", utopiaController=None)
    #nt.startFlicker(framesperbit=6, numframes=60*60)

    # make a galery app
    subscreens = {
        "ins1":["InstructionScreen",{"label":"Instruction Screen", "text":"This is a default start screen....\nPress <space> to continue", "waitKey":True, "duration":1000}],
        "usr":["UserInputScreen",{"label":"User Input Screen", "text":"type some stuff here:\n"}],
        "sel":["SelectionGridScreen",{"symbols":[["1","2"],["3","4"]]}],
        "check":["CheckerboardGridScreen",{"symbols":[["1","2"],["3","4"]]}],
        "fovea":["FovealGridScreen",{"symbols":[["1","2"],["3","4"]]}],
        "wheel":["SelectionWheelScreen",{"symbols":[["1","2","3","4"],["5","6","7","8"]]}],
        "checkwheel":["SelectionWheelScreen",{"symbols":[["1","2","3","4"],["5","6","7","8"]], "ischeckerboard":4}],
        "sound":["SoundFlashScreen",{"symbols":[["1|digits/MAE_1A.wav","2|digits/MAE_2A.wav","3|digits/MAE_3A.wav","4|digits/MAE_4A.wav"]], "use_psychopy":False, "state2vol":{"1":.5,"2":1}}],
        "text":["TextFlashScreen",{"symbols":[["1|o1.e1|o1.e2|o1.e3|o1.e4","2|o2.e1|o2.e2|o2.e3|o2.e4"]]}]
    }
    subscreen_transitions = {}
    screen = SubscreenMenuScreen(window, subscreens=subscreens, subscreen_transitions=subscreen_transitions, noisetag=nt)
    run_screen(window, screen)

