from mindaffectBCI.noisetag import Noisetag

from kivy.uix.floatlayout import FloatLayout
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.graphics import Color


state2color={0:(.2,.2,.2,1), # off=grey
             1:(1,1,1,1),    # on=white
             2:(0,1,0,1),    # cue=green
             3:(0,0,1,1)}    # feedback=blue

# Description of the app layout
kv="""
<Label>
    font_size : 30
<Button>
    background_normal: ""
    background_color: (.5,.5,.5,1)

<minimalview>
    GridLayout:
        orientation: 'vertical'
        padding: 20
        rows:2
        cols:2

        Button:
            id: but_1
            text: '1'
            on_press: 

        Button:
            id: but_2
            text: '2'
            on_press: 

        Button:
            id: but_3
            text: '3'
            on_press: 

        Button:
            id: but_4
            text: '4'
            on_press: 

"""

class MinimalView(FloatLayout):
    '''Create a controller that receives a custom widget from the kv lang file.

    Add an action to be called from the kv lang file.
    '''

    nframe=0
    def update(self, dt):
        global nt, lastfliptime
        '''draw the display with colors from noisetag'''
        # send info on the *previous* stimulus state, with the recorded vsync time (if available)
        nt.sendStimulusState(timestamp=lastfliptime)
        # update and get the new stimulus state to display
        try : 
            nt.updateStimulusState()
            stimulus_state,target_idx,objIDs,sendEvents=nt.getStimulusState()
            target_state = stimulus_state[target_idx] if target_idx>=0 else -1
        except StopIteration :
            exit() # terminate app when noisetag is done

        # draw the display with the instructed colors
        if stimulus_state :
            self.ids.but_1.background_color = state2color[stimulus_state[0]]
            self.ids.but_2.background_color = state2color[stimulus_state[1]]
            self.ids.but_3.background_color = state2color[stimulus_state[2]]
            self.ids.but_4.background_color = state2color[stimulus_state[3]]
    
        # some textual logging of what's happening
        if target_state is not None and target_state>=0:
            print("*" if target_state>0 else '.',end='',flush=True)

    # define a trival selection handler
    def selectionHandler(self, objID:int):
        print("Selected: %d"%(objID))
        if objID==1:
            self.ids.but_1.trigger_action(.1)
        elif objID==2:
            self.ids.but_2.trigger_action(.1)
        elif objID==3:
            self.ids.but_3.trigger_action(.1)
        elif objID==4:
            self.ids.but_4.trigger_action(.1)

            
lastfliptime=0
def timedflip(self):
    '''pseudo method type which records the timestamp for window flips'''
    global lastfliptime, nt
    lastfliptime= nt.getTimeStamp()

class MinimalViewApp(App):
    def __init__(self,nt:Noisetag):
        super().__init__()
        self.nt = nt

    def build(self):
        self.view = MinimalView()#info='Welcome to mindaffectBCI')
        # record accurate flip times.
        Window.bind(on_flip=timedflip)
        # attach to catch BCI selections
        self.nt.addSelectionHandler(self.view.selectionHandler)
        
        # run the game @ 60hz
        Clock.schedule_interval(self.view.update,1/60.0)
        return self.view    


def run():
    global lastfliptime, nt
    # Initialize the noise-tagging connection
    nt = Noisetag()
    nt.connect(timeout_ms=5000)

    # run the mainloop
    # start the UI thread
    Builder.load_string(kv)
    app = MinimalViewApp(nt)

    # tell the noisetag framework to run a full : calibrate->prediction sequence
    nt.setnumActiveObjIDs(4)
    nt.startExpt(nCal=2,nPred=10,duration=4)

    # run the main-loop
    app.run()


if __name__ == '__main__':
    run()