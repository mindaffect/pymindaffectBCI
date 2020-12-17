import mindaffectBCI.examples.presentation.selectionMatrix as selectionMatrix
from mindaffectBCI.examples.presentation.snakegame import SnakeGame

class SnakeGameScreen(selectionMatrix.WaitScreen):
    def __init__(self, window, symbols, noisetag, grid_width:int=20, grid_height:int=20, duration:float=None, waitKey:bool=False, logo:str="Mindaffect_Logo.png", framespermove:int=60*40, target_only:bool=False, clearScreen:bool=True, sendEvents:bool=True, **kwargs):
        super().__init__(window, duration, waitKey, logo)
        self.window=window
        self.noisetag = noisetag
        self.liveSelections = None
        self.show_newtarget = None
        self.clearScreen = clearScreen
        self.framespermove = framespermove
        self.target_only = target_only
        self.sendEvents = sendEvents
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.reset()

    def set_grid(self,**kwargs):
        self.objIDs = [1,2,3,4] # [L,R,U,D]
        self.noisetag.setActiveObjIDs(self.objIDs) # register these objIDs

    def set_sentence(self, text):
        pass

    def setliveSelections(self, value):
        if self.liveSelections is None :
            self.noisetag.addSelectionHandler(self.doSelection)
        self.liveSelections = value

    def doSelection(self,objID):
        if objID in self.objIDs:
                print("doSelection: {}".format(objID))
                symbIdx = self.objIDs.index(objID)
                # move the snake in the desired direction
                self.snakegame.turn(symbIdx)
                #self.doGameTick()

    def is_done(self):
        if self.snakegame.death : # extra end-of-game check
            self.isDone = True
        return selectionMatrix.WaitScreen.is_done(self)

    def setshowNewTarget(self, value):
        if self.show_newtarget is None:
            self.noisetag.addNewTargetHandler(self.doNewTarget)
        self.show_newtarget=value

    def doNewTarget(self):
        self.doGameTick()

    def doGameTick(self):
        self.snakegame.run_rules()
        self.snakegame.move()

    def reset(self):
        selectionMatrix.WaitScreen.reset(self)
        self.nframe = 0
        self.snakegame = SnakeGame(self.window, self.grid_width, self.grid_height)

    def draw(self, t):
        """draw the letter-grid with given stimulus state for each object.
        Note: To maximise timing accuracy we send the info on the grid-stimulus state
        at the start of the *next* frame, as this happens as soon as possible after
        the screen 'flip'. """
        if not self.isRunning:
            self.nframe = 0
            self.isRunning=True
        self.framestart=self.noisetag.getTimeStamp()
        winflip = self.window.lastfliptime
        self.nframe = self.nframe+1
        if self.sendEvents:
            self.noisetag.sendStimulusState(timestamp=winflip)

        # get the current stimulus state to show
        try:
            self.noisetag.updateStimulusState()
            stimulus_state, target_idx, objIDs, sendEvents=self.noisetag.getStimulusState()
            target_state = stimulus_state[target_idx] if target_idx>=0 else -1
            if target_idx >= 0 : self.last_target_idx = target_idx
        except StopIteration:
            self.isDone=True
            return

        # turn all off if no stim-state
        if stimulus_state is None:
            stimulus_state = [0]*len(self.objIDs)

        # insert the flicker state into the game grid...
        # draw the white background onto the surface
        if self.clearScreen:
            self.window.clear()
        # update the state
        for idx in range(min(len(self.objIDs), len(stimulus_state))):
            # set background color based on the stimulus state (if set)
            try:
                ssi = stimulus_state[idx]
                if self.target_only and not target_idx == idx :
                    ssi = 0
                self.update_object_state(idx,ssi)
            except KeyError:
                pass

        # update the game state
        self.nframe = self.nframe + 1
        if self.nframe % self.framespermove == 0 :
            self.doGameTick()

        # call the game draw functions
        selectionMatrix.WaitScreen.draw(self,t)
        self.snakegame.draw()
        self.snakegame.draw_score()


    state2color={0:(5, 5, 5),       # off=grey
                 1:(255, 255, 255), # on=white
                 2:(0, 255, 0),     # cue=green
                 3:(0, 0, 255)}     # feedback=blue
    def update_object_state(self,idx,state):
        # get the snake head coord
        x,y = self.snakegame.body[-1]
        if idx == 0: # L  # match the command order [LRUD]
            x = x-1
        elif idx == 1: # R
            x = x+1
        elif idx == 2: # U
            y = y+1
        elif idx == 3: # D
            y = y-1
        # set the cell state, occupied and given color
        if 0<=x and x<self.snakegame.grid_width and 0<=y and y<self.snakegame.grid_height:
            if self.snakegame.cells[x][y][0]==0 or not self.snakegame.cells[x][y][1]==self.snakegame.snake_clr: # only if not occupied
                self.snakegame.cells[x][y] = (1, self.state2color[state])


if __name__=="__main__":
    args = selectionMatrix.parse_args()
    setattr(args,'calibration_symbols','3x3.txt')
    setattr(args,'symbols',None)
    setattr(args,'predictionScreen','mindaffectBCI.examples.presentation.snake_game.SnakeGameScreen')
    setattr(args,'npred',1000)
    selectionMatrix.run(**vars(args))
