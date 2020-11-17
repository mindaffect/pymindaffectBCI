#!/usr/bin/env python3
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


import pyglet
import sys
import time
import math

from mindaffectBCI.noisetag import Noisetag, sumstats

class Quad:
    '''object to hold a single graphics quad'''
    def __init__(self, objID:int, vertices, color, theta:float):
        """object to hold a GL-QUADS based BCI selectable object

        Args:
            objID (int): the objID for this object
            vertices (list-of-tuples-of-float): list of 4 (x,y) vertices for this quad
            color (list-of-float): rgb float base color for this quad
            theta (float): angle of this segment on the colorwheel
        """        
        self.objID=objID
        self.vertices=vertices
        self.color=color
        self.theta=theta

    def draw(self, maskcolor=None, alpha=.5):
        """ draw this quad -- optionally with given masking color

        Args:
            maskcolor (color, optional): (3,float) mask color blended with the quads base color. Defaults to None.
            alpha (float, optional): strength of the blending. Defaults to .5.
        """        
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

    def __init__(self,noisetag,*args,**kwargs):
        super(DrawWindow,self).__init__(*args,**kwargs)
        self.frame=0
        self.segments=[]
        self.objIDs=[]
        self.noisetag=None
        self.lastfliptime=0
        self.flipstats = sumstats()
        self.fliplogtime=self.lastfliptime
        self.noisetag=noisetag
        
    def init_pie(self,nsegements:int=12,iradius:float=1/6,oradius:float=1/3,objIDs=None):
        """initialize the colorwheel given number of segements and inner/outer radii

        Args:
            nsegements (int, optional): number of colorwheel segements to use. Defaults to 12.
            iradius (float, optional): wheel inner radius, as fraction of window size. Defaults to 1/6.
            oradius (float, optional): wheel outer radius, as fraction of window size. Defaults to 1/3.
            objIDs (list-of-int, optional): set of object IDs to use for these segments. Defaults to None.
        """        
        self.objIDs = objIDs
        if self.objIDs is None :
            self.objIDs = list(range(1,nsegements+1))
        if nsegements is None :
            nsegements = len(self.objIDs)
        winw,winh=self.get_size()
        cent   =(winw/2,winh/2)
        iradius=min([winw,winh])*iradius
        oradius=min([winw,winh])*oradius
        segwidth= math.pi*2 / nsegements
        # center of the segement
        thetas=[ i*segwidth for i in range(nsegements) ]

        # TODO [] : make a pyglet BATCH...
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
        """set the noisetag object we use for BCI communication

        Args:
            noisetag (Noisetag): the noisetag object we use for BCI communication
        """
        self.noisetag=noisetag

    # mapping between bci-stimulus-states and on-screen colors
    state2color={0:(.2,.2,.2), # off=grey
                 1:(1,1,1),    # on=white
                 2:(1,1,1)}    # cue=flash

    def draw_pie(self):
        """draw the complete colorwheel with the flicker state as obtained from the noisetag object
        """        
        self.frame=self.frame+1
        # get the bci-stimulus state
        stimulus_state=None
        target_state = -1
        if self.noisetag :
            self.noisetag.sendStimulusState(timestamp=self.lastfliptime)
            self.noisetag.updateStimulusState()
            stimulus_state,target_idx,objIDs,sendEvents=self.noisetag.getStimulusState()
            target_state = stimulus_state[target_idx] if target_idx>=0 else -1
        # do nothing if no bci-stimulus
        if stimulus_state is None :
            stimulus_state = [0]*len(self.segements)

        self.clear()        
        # modify the blend strength if in cue/feedback mode
        alpha=.6
        if 1 in stimulus_state :
            alpha=.7 # medium background in flicker mode
        elif 3 in stimulus_state :
            alpha=.9 # dark background if feedback mode
        elif 2 in stimulus_state : 
            alpha=.6 # medium background in cue-mode
        # draw the segements with correct bci-stim mask blending
        for i,seg in enumerate(self.segements):
            # color depends on the requested stimulus state
            try:
                bg=self.state2color[stimulus_state[i]]
            except KeyError :
                bg=None # normal color
            seg.draw(maskcolor=bg,alpha=alpha)            

        if target_state is not None and target_state>0:
            print("*" if target_state>0 else '.',end='',flush=True)


    def update(self,dt):
        self.draw_pie()
        
    def flip(self):
        """ override window's flip method to send stimulus state as close in
        time as possible to when the screen re-freshed
        """        
        super().flip()
        olft=self.lastfliptime
        self.lastfliptime=self.noisetag.getTimeStamp()
        self.flipstats.addpoint(self.lastfliptime-olft)
        if self.lastfliptime > self.fliplogtime :
            self.fliplogtime=self.lastfliptime+10000    
            print("\nFlipTimes:"+str(self.flipstats))
            print("Hist:\n"+self.flipstats.hist())
            
def selectionHandler(objID):
    """function called back when a BCI selection is made which updates the light color

    Args:
        objID (int): the ID number of the selected object
    """    
    global window
    print('Sel: %d'%(objID))
    # if is obj we are responsible for
    if objID in window.objIDs :
        # get the matching segment
        segidx = window.objIDs.index(objID)
        seg    = window.segements[segidx]
        # set the light to this color, by matching the hue.
        if hue_bridge : 
            try : 
                lights=hue_bridge.lights()
                for k,v in lights.items():
                    if v['state']['reachable'] :
                        hue_bridge.lights[k]('state',bri=128,hue=int(65535*seg.theta/(2*math.pi)))
            except :
                print("couldnt talk to the hue!") 

window=None
hue_bridge=None
def run(hue_bridgeip=None, hue_username=None, **kwargs):
    """run the Phillips Hue lights control demo

    Args:
        hue_bridgeip (str, optional): the address of the hue bridge control device. Defaults to "192.168.253.100".
        hue_username (str, optional): the username to use when communicating with the hue-bridge device. Note: for this script to work you **must** have pre-generated an authorized username for this particular philips HUE. Defaults to "AlGbD1GTQHxwDVK0j3tKkpdwBUv8Cijtvbkokzxk".
    """    
    global window, hue_bridge
    # N.B. init the noisetag first so asks for decoder IP
    noisetag=Noisetag()
    # auto connect to the decoder
    noisetag.connect()

    config = pyglet.gl.Config(double_buffer=True)
    window = DrawWindow(noisetag,width=1024,height=768,vsync=True,config=config)
    window.init_pie(nsegements=10)

    # set the current noise-tagging state
    noisetag.setActiveObjIDs(window.objIDs)
    noisetag.startExpt(nCal=10,nPred=20,
                       cueduration=2,calduration=4,predduration=20,
                       feedbackduration=4)

    # Initialize the connection to the hue light
    try :
        from qhue import Bridge
        hue_bridge = Bridge(hue_bridgeip, hue_username)
        # register selection handler
        noisetag.addSelectionHandler(selectionHandler)
    except ImportError :
        import warnings
        warnings.warn("You need to install the qhue module to connect to your hue!")
    
    # run the mainloop
    pyglet.clock.schedule(window.update)
    pyglet.app.run()

if __name__=='__main__':
    run(hue_bridgeip="192.168.253.100",hue_username="AlGbD1GTQHxwDVK0j3tKkpdwBUv8Cijtvbkokzxk")