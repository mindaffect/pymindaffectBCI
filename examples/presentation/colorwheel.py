import pyglet
import sys
import time
import math

#import os
#scriptpath=os.path.dirname(os.path.realpath(__file__))
#import sys
#sys.path.append(os.path.join(scriptpath,'../../'))

from mindaffectBCI.noisetag import Noisetag

class Quad:
    '''object to hold a single graphics quad'''
    def __init__(self,objID,vertices,color,theta):
        self.objID=objID
        self.vertices=vertices
        self.color=color
        self.theta=theta
    def draw(self,maskcolor=None,alpha=.5):
        '''draw this quad, optionally with given color'''
        ''' given color overrides'''
        if maskcolor :
            # use alpha to blend color and maskcolor
            color = tuple(maskcolor[i]*alpha + self.color[i]*(1-alpha) for i in range(len(self.color)))
        else : 
            color=self.color
        # draw colored segement
        pyglet.graphics.draw(4,pyglet.gl.GL_QUADS,
                            ('v2f',self.vertices),
                            ('c3f',(color)*4))

class DrawWindow(pyglet.window.Window):

    def __init__(self,*args,**kwargs):
        super(DrawWindow,self).__init__(*args,**kwargs)
        self.frame=0
        self.segments=[]
        self.objIDs=[]
        self.noisetag=None
        
    def init_pie(self,nsegements=12,iradius=1/6,oradius=1/3,objIDs=None):
        self.objIDs = objIDs
        if self.objIDs is None :
            self.objIDs = list(range(1,nsegements+1))
        if nsegements is None :
            nsegements = len(self.objIDs)
        winw,winh=self.get_size()
        w=winw/3 
        h=winh/3
        cent   =(winw/2,winh/2)
        iradius=min([winw,winh])*iradius
        oradius=min([winw,winh])*oradius
        segwidth= math.pi*2 / nsegements
        # center of the segement
        thetas=[ i*segwidth for i in range(nsegements) ]

        self.segements=[]
        for i,theta in enumerate(thetas):
            # inner-top
            it = (cent[0]+iradius*math.sin(theta),
                  cent[1]+iradius*math.cos(theta))
            # outer-top
            ot = (cent[0]+oradius*math.sin(theta),
                  cent[1]+oradius*math.cos(theta))
            # outer-bot
            ob = (cent[0]+oradius*math.sin(theta+segwidth),
                  cent[1]+oradius*math.cos(theta+segwidth))
            # inner-bot
            ib = (cent[0]+iradius*math.sin(theta+segwidth),
                  cent[1]+iradius*math.cos(theta+segwidth))

            # color
            colsegwidth=2.0*math.pi/3.0
            if   0*colsegwidth <=theta and theta<= 1*colsegwidth : 
                alpha = (theta-0*colsegwidth)/(colsegwidth)
                bg    = ((1-alpha), alpha,       0) # r-g
            elif 1*colsegwidth <=theta and theta<= 2*colsegwidth :
                alpha = (theta-1*colsegwidth)/(colsegwidth)
                bg    = (0,        (1-alpha),   alpha) # g-b
            elif 2*colsegwidth <=theta and theta<= 3*colsegwidth : 
                alpha = (theta-2*colsegwidth)/(colsegwidth)
                bg    = (alpha,       0,     (1-alpha)) # b-r
            
            #print('theta %g  alpha %g = (%g,%g,%g)'%(theta,alpha,*bg))
            # each segment is a single graphics quad
            self.segements.append(Quad(self.objIDs[i],it+ot+ob+ib,bg,theta))

    def setNoisetag(self,noisetag):
        self.noisetag=noisetag
            
    def draw_pie(self):

        stimulus_state=None
        if self.noisetag :
            self.noisetag.sendStimulusState()
            self.noisetag.updateStimulusState()
            stimulus_state,target_state,objIDs,sendEvents=self.noisetag.getStimulusState()

        self.clear()
        alpha=.5
        # modify the blend strength if in cue/feedback mode
        if stimulus_state :
            if 3 in stimulus_state :
                alpha=.9
            elif 2 in stimulus_state : 
                alpha=.8
        for i,seg in enumerate(self.segements):
            # apply the noise code -> override with white when on..
            bs=stimulus_state[i] if stimulus_state else None
            # color depends on the requested stimulus state
            bg=None

            if bs==0 :    # off
                bg=(0,0,0)
            elif bs==1 :  # flash
                bg=(1,1,1)
            elif bs==2 :  # cue
                bg=(0,1,0)
            elif bs==3 :  # feedback
                bg=seg.color
            
            seg.draw(maskcolor=bg,alpha=alpha)            
            
    def on_draw(self):
        self.frame=self.frame+1
        self.draw_pie()

    def update(self,dt):
        self.draw_pie()


def selectionHandler(objID):
    '''function to map selection to action'''    
    print('Sel: %d'%(objID))
    # if is obj we are responsible for
    if objID in window.objIDs :
        # get the matching segment
        segidx = window.objIDs.index(objID)
        seg    = window.segements[segidx]
        # and it's color
        color  = seg.color
        # set the light to this color...
        if hue_bridge : 
            lights=hue_bridge.lights()
            for k,v in lights.items():
                if v['state']['reachable'] :
                    hue_bridge.lights[k]('state',bri=255,hue=int(65535*seg.theta/(2*math.pi)))


if __name__=='__main__':
    #config = pyglet.gl.Config(double_buffer=True)
    window = DrawWindow()#vsync=True,config=config)
    window.init_pie(nsegements=10)
    noisetag=Noisetag()
    noisetag.startExpt(window.objIDs,nCal=1,nPred=100,
                        cueduration=4,duration=10,feedbackduration=4)
    window.setNoisetag(noisetag)

    # Initialize the connection to the hue light
    from qhue import Bridge
    bridgeip="192.168.253.100"
    username="AlGbD1GTQHxwDVK0j3tKkpdwBUv8Cijtvbkokzxk"
    hue_bridge = Bridge(bridgeip, username)
    # register selection handler
    noisetag.addSelectionHandler(selectionHandler)

    # run the mainloop
    pyglet.clock.schedule(window.update)
    pyglet.app.run()
