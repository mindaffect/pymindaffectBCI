#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
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

from mindaffectBCI.noisetag import Noisetag, sumstats
nt = Noisetag()
nt.connect()

import pyglet
# make a default window, with fixed size for simplicty
window=pyglet.window.Window(width=640,height=480)
# define a simple 2-squares drawing function
def draw_squares(col1,col2):
    # draw square 1: @100,190 , width=100, height=100
    x=100; y=190; w=100; h=100;
    pyglet.graphics.draw(4,pyglet.gl.GL_QUADS,
                         ('v2f',(x,y,x+w,y,x+w,y+h,x,y+h)),
             ('c3f',(col1)*4))
    # draw square 2: @440,100
    x=640-100-100
    pyglet.graphics.draw(4,pyglet.gl.GL_QUADS,
                         ('v2f',(x,y,x+w,y,x+w,y+h,x,y+h)),
             ('c3f',(col2)*4))    

# dictionary mapping from stimulus-state to colors
state2color={0:(.2,.2,.2), # off=grey
             1:(1,1,1),    # on=white
             2:(0,1,0),    # cue=green
             3:(0,0,1)}    # feedback=blue
def draw(dt):
    '''draw the display with colors from noisetag'''
    # send info on the *previous* stimulus state.
    # N.B. we do it here as draw is called as soon as the vsync happens
    nt.sendStimulusState(timestamp=window.lastfliptime)
    # update and get the new stimulus state to display
    try : 
        nt.updateStimulusState()
        stimulus_state,target_state,objIDs,sendEvents=nt.getStimulusState()
    except StopIteration :
        pyglet.app.exit() # terminate app when noisetag is done
        return
    # draw the display with the instructed colors
    if target_state is not None and target_state>=0:
        print("*" if target_state>0 else '.',end='',flush=True)
        pass

    if stimulus_state : 
        draw_squares(state2color[stimulus_state[0]],
                     state2color[stimulus_state[1]])

# used to record statistics about the flip timing -- for debugging
logTime=0
ss=sumstats()

# override window's flip method to record the exact *time* the
# flip happended
import types
def timedflip(self):
    '''pseudo method type which records the timestamp for window flips'''
    type(self).flip(self) # call the 'real' flip method...
    oft=self.lastfliptime
    self.lastfliptime=nt.getTimeStamp()
    ss.addpoint(self.lastfliptime-oft)
    global logTime
    if self.lastfliptime > logTime :
        print("\nFlipTimes:"+str(ss))
        print("Hist:\n"+ss.hist())
        logTime=self.lastfliptime+5000
        
window.flip = types.MethodType(timedflip,window)
# ensure the field is already there.
window.lastfliptime=nt.getTimeStamp()

# define a trival selection handler
def selectionHandler(objID):
    print("Selected: %d"%(objID))    
nt.addSelectionHandler(selectionHandler)

# tell the noisetag framework to run a full : calibrate->prediction sequence
nt.setnumActiveObjIDs(2)
nt.startExpt(nCal=4,nPred=10,duration=4)
# run the pyglet main loop
pyglet.clock.schedule(draw)
pyglet.app.run()
