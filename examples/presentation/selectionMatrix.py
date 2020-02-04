# get the general noisetagging framework
from mindaffectBCI.noisetag import Noisetag
from mindaffectBCI.utopiaclient import TimeStampClock

# graphic library
import pyglet
isi=1/60
drawrate=0 # rate at which draw is called

class Screen:
    '''Screen abstract-class which draws stuff on the screen until finished'''
    def __init__(self,window):
        self.window=window
    def draw(self,t):
        '''draw the display, N.B. NOT including flip!'''
        pass
    def is_done(self):
        '''test if this screen wants to quit'''
        return False

class ScreenStack(Screen):
    '''Special type of screen which contains a stack of screens to show,
    which are processed in stack order, i.e. LIFO'''
    def __init__(self,window):
        self.window=window
        self.screenStack=[]
    def draw(self,t):
        '''re-direct to the draw method of the top of the screen stack'''
        cur_screen=self.get()
        cur_screen.draw(t)
        # remove completed screens from the stack
        if cur_screen.is_done(): 
            self.pop()        
    def is_done(self):
        # only done when the screen stack is empty
        return not self.screenStack
    def push(self,screen):
        '''add to top of stack = run's first'''
        #print("push(%d)"%(len(self.screenStack))+str(screen))
        self.screenStack.append(screen)
    def pushback(self,screen):
        '''add to the bottom of the screen stack = runs's last'''
        #print("pushback(%d)"%(len(self.screenStack))+str(screen))
        self.screenStack.insert(0,screen)
    def pop(self):
        #print("pop(%d)"%(len(self.screenStack))+str(self.get()))
        return self.screenStack.pop()
    def get(self):
        return self.screenStack[-1] if self.screenStack else None
            
class InstructionScreen(Screen):
    '''Screen which shows a textual instruction for duration or until key-pressed'''
    def __init__(self,window,text,duration=5000,waitKey=True):
        super().__init__(window)
        self.tsc=None # timer for the duration
        self.duration=duration
        self.waitKey=waitKey
        self.isRunning=False
        self.clearScreen=True
        # initialize the instructions screen
        self.instructLabel=pyglet.text.Label(x=window.width//2,y=window.height//2,anchor_x='center',anchor_y='center',font_size=24,color=(255,255,255,255),multiline=True,width=int(window.width*.8))
        self.set_text(text)
        print("Instruct (%dms): %s"%(duration,text))

    def set_text(self,text):
        '''set/update the text to show in the instruction screen'''
        if type(text) is list:
            text = "\n".join(text)
        self.instructLabel.begin_update()
        self.instructLabel.text=text
        self.instructLabel.end_update()

    def is_done(self):
        # check termination conditions
        isDone=False
        if not self.isRunning :
            return False
        if self.waitKey :
            global last_key_press
            if last_key_press :
                self.key_press=last_key_press
                isDone=True
                last_key_press=None
        if self.tsc.getTime() > self.duration :
            isDone=True
        return isDone
        
    def draw(self,t):
        '''Show a block of text to the user for a given duration on a blank screen'''
        if not self.isRunning :
            self.tsc=TimeStampClock()
            self.isRunning=True # mark that we're running
        if self.clearScreen:
            self.window.clear()
        self.instructLabel.draw()

class SelectionGridScreen(Screen):
    '''Screen which shows a grid of symbols which will be flickered with the noisecode
    and which can be selected from by the mindaffect decoder Brain Computer Interface'''

    LOGLEVEL=1
    
    def __init__(self,window,symbols,noisetag,objIDs=None,
                 bgFraction=.3,clearScreen=True,sendEvents=True,liveFeedback=True):
        '''Intialize the stimulus display with the grid of strings in the 
        shape given by symbols.
        Store the grid object in the fakepresentation.objects list so can 
        use directly with the fakepresentation BCI presentation wrapper.'''
        self.window=window
        # create set of sprites and add to render batch
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)
        self.clearScreen= True
        self.isRunning=False
        self.isDone=False
        self.sendEvents=sendEvents
        self.liveFeedback=liveFeedback
        self.set_grid(symbols,objIDs,bgFraction)
        # N.B. noisetag does the whole stimulus sequence
        self.set_noisetag(noisetag)

    def set_noisetag(self,noisetag):
        self.noisetag=noisetag

    def set_grid(self,symbols,objIDs=None,bgFraction=.3):
        '''set/update the grid of symbols to be selected from'''
        winw,winh=window.get_size()
        # get size of the matrix
        gridwidth  = len(symbols)
        gridheight = len(symbols[0])
        ngrid      = gridwidth * gridheight        
    
        # add a background sprite with the right color
        self.objects=[None]*ngrid
        self.labels=[None]*ngrid
        if objIDs is None :
            objIDs = list(range(1,ngrid+1))
        self.objIDs=objIDs

        # now create the display objects
        w=winw/gridwidth # cell-width
        bgoffsetx = w*bgFraction
        h=winh/gridheight # cell-height
        bgoffsety = h*bgFraction
        for i in range(gridheight): # rows
            y = i/gridheight*winh # top-edge cell
            for j in range(gridwidth): # cols
                idx = i*gridwidth+j 
                x = j/gridwidth*winw # left-edge cell
                # create a 1x1 white image for this grid cell
                img = pyglet.image.SolidColorImagePattern(color=(255,255,255,255)).create_image(1,1)
                # convert to a sprite (for fast re-draw) and store in objects list
                # and add to the drawing batch (as background)
                self.objects[idx]=pyglet.sprite.Sprite(img,x=x+bgoffsetx,y=y+bgoffsety,
                                                       batch=self.batch,group=self.background)
                # re-scale (on GPU) to the size of this grid cell
                self.objects[idx].update(scale_x=int(w-bgoffsetx*2),scale_y=int(h-bgoffsety*2))
                # add the foreground label for this cell, and add to drawing batch
                self.labels[idx]=pyglet.text.Label(symbols[i][j],font_size=32,x=x+w/2,y=y+h/2,
                                                   color=(255,255,255,255),
                                                   anchor_x='center',anchor_y='center',
                                                   batch=self.batch,group=self.foreground)

        # add opto-sensor block - as last object
        img = pyglet.image.SolidColorImagePattern(color=(255,255,255,255)).create_image(1,1)
        self.opto_sprite=pyglet.sprite.Sprite(img,x=0,y=winh-h/2,
                                              batch=self.batch,group=self.background)
        self.opto_sprite.update(scale_x=int(w/2),scale_y=int(h/2))

    def is_done(self):
        return self.isDone

    # mapping from bci-stimulus-states to display color
    state2color={0:(60,60,60),   # off=grey
                 1:(255,255,255),# on=white
	             2:(0,255,0),    # cue=green
	             3:(0,0,255)}    # feedback=blue
    def draw(self,t):
        """draw the letter-grid with given stimulus state for each object.
        Note: To maximise timing accuracy we send the info on the grid-stimulus state
        at the start of the *next* frame, as this happens as soon as possible after
        the screen 'flip'. """
        if not self.isRunning :
            self.isRunning=True
        self.framestart=self.noisetag.getTimeStamp()
        if self.sendEvents:
            self.noisetag.sendStimulusState()

        # get the current stimulus state to show
        try :
            self.noisetag.updateStimulusState()
            stimulus_state,target_state,objIDs,sendEvents=self.noisetag.getStimulusState()
            # get prediction info if any
            if self.liveFeedback :
                predMessage=self.noisetag.getLastPrediction()
        except StopIteration :
            self.isDone=True
            return
        
        if stimulus_state is None :
            return
        
        # draw the white background onto the surface
        if self.clearScreen:
            window.clear()
        # update the state
        # TODO[]: iterate over objectIDs and match with those from the
        #         stimulus state!
        for idx in range(min(len(self.objects),len(stimulus_state))): 
            # set background color based on the stimulus state (if set)
            try :
                self.objects[idx].color=self.state2color[stimulus_state[idx]]
            except KeyError :
                pass
            
                
        # show live-feedback (if wanted)
        if self.liveFeedback and predMessage and predMessage.Yest in objIDs :
            predidx=objIDs.index(predMessage.Yest) # convert from objID -> objects index
            # BODGE: manually mix in the feedback color as blue tint.
            fbcol=self.objects[predidx].color
            fbcol=(fbcol[0]*.6, fbcol[1]*.6, fbcol[2]*.6+255*(1-predMessage.Perr))
            self.objects[predidx].color=fbcol

        # disp opto-sensor if targetState is set    
        if self.opto_sprite is not None and target_state is not None and target_state in (0,1):
            self.opto_sprite.visible=True
            self.opto_sprite.color = (0,0,0) if target_state==0 else (255,255,255)
        else:
            self.opto_sprite.visible=False

        # do the draw                
        self.batch.draw()
        self.frameend=self.noisetag.getTimeStamp()
    
        # frame flip time logging info
        if self.LOGLEVEL>0 and self.noisetag.isConnected() :
            opto = target_state if target_state is not None else 0
            logstr="FrameIdx:%d FlipTime:%d FlipLB:%d FlipUB:%d Opto:%d"%(nframe,self.framestart,self.framestart,self.frameend,opto)
            self.noisetag.log(logstr)
    
#------------------------------------------------------------------------
# Initialization : display, utopia-connection
def initPyglet(fullscreen=False):
    '''intialize the pyglet window, keyhandler'''
    global window
    # set up the window
    if fullscreen : 
        # N.B. accurate video timing only guaranteed with fullscreen
        config = pyglet.gl.Config(double_buffer=True)
        window = pyglet.window.Window(vsync=True,config=config)
    else :
        config = pyglet.gl.Config(double_buffer=True)
        window = pyglet.window.Window(width=1024,height=768,vsync=True,config=config)

    # setup a key press handler, just store key-press in global variable
    window.push_handlers(on_key_press)#,on_close)

def draw(dt):
    '''main window draw function, which redirects to the screen stack'''
    global ss, nframe
    nframe=nframe+1
    ss.draw(dt)
    # check for termination
    if ss.is_done() :
        pyglet.app.exit()
    print('.',end='',flush=True)

last_key_press=None
def on_key_press(symbols,modifiers):
    '''main key-press handler, which stores the last key in a global variable'''
    global last_key_press
    last_key_press=symbols

if __name__ == "__main__":
    # init the graphics system
    initPyglet()
    nframe=0
    # the logical arrangement of the display matrix
    symbols=[['a','b','c','d','e'],
             ['f','g','h','i','j'],
             ['k','l','m','n','o'],
             ['p','q','r','s','t'],
             ['u','v','w','x','y']]
    nsymb  = sum([ len(x) for x in symbols])
    # N.B. objIDs must be unique and >0 and <255
    objIDs = list(range(1,nsymb+1))

    # make the calibration screen
    calnoisetag=Noisetag()
    calnoisetag.startCalibration(objIDs,nTrials=1,numframes=4/isi)
    calibrationScreen = SelectionGridScreen(window,symbols,calnoisetag)
    # make the prediction screen - different noise-tag mode.
    prednoisetag=Noisetag()
    prednoisetag.startPrediction(objIDs,nTrials=10,numframes=10/isi)
    predictionScreen = SelectionGridScreen(window,symbols,prednoisetag)
    print(prednoisetag)
    
    
    # make a screen stack to show the sequence of screens:
    #   instruct, calibration, instruct, prediction, instruct
    ss = ScreenStack(window)
    # N.B. use pushback so order is order of execution
    ss.pushback(InstructionScreen(window,"Welcome Instruction\n\nkey to continue",duration=50000))
    ss.pushback(calibrationScreen)
    ss.pushback(InstructionScreen(window,"Prediction Message\n\nkey to continue",duration=50000))
    ss.pushback(predictionScreen)
    ss.pushback(InstructionScreen(window,"Closing Message\nThankyou\nkey to continue"))
    
    # set per-frame callback to the draw function    
    if drawrate>0 :
        # slow down for debugging
        pyglet.clock.schedule_interval(draw,drawrate)
    else :
        # call the draw method as fast as possible, i.e. at video frame rate!
        pyglet.clock.schedule(draw)
    # mainloop
    pyglet.app.run()
