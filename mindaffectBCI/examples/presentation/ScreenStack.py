class ScreenStack(Screen):
    '''Special type of screen which contains a stack of screens to show,
    which are processed in stack order, i.e. LIFO'''
    def __init__(self,window):
        self.window=window
        self.screenStack=[]
    def draw(self,t):
        '''re-direct to the draw method of the top of the screen stack'''
        cur_screen=self.get()
        if cur_screen : 
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
