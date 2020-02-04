from mindaffectBCI.noisetag import Noisetag
nt = Noisetag()

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
    nt.sendStimulusState()
    # update and get the new stimulus state to display
    try : 
        nt.updateStimulusState()
        stimulus_state,target_state,objIDs,sendEvents=nt.getStimulusState()
    except StopIteration :
        pyglet.app.exit() # terminate app when noisetag is done
        return
    # draw the display with the instructed colors
    if stimulus_state : 
        draw_squares(state2color[stimulus_state[0]],
                     state2color[stimulus_state[1]])

# define a trival selection handler
def selectionHandler(objID):
    print("Selected: %d"%(objID))    
nt.addSelectionHandler(selectionHandler)

# tell the noisetag framework to run a full : calibrate->prediction sequence
nt.startExpt([1,2],nCal=10,nPred=10)
# run the pyglet main loop
pyglet.clock.schedule(draw)
pyglet.app.run()
