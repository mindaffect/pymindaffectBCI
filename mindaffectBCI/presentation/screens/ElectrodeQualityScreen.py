#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jadref@gmail.com>
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
from mindaffectBCI.presentation.screens.basic_screens import Screen
from mindaffectBCI.utopiaclient import DataPacket
import pyglet
from math import log10
from collections import deque
import time

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
class ElectrodeQualityScreen(Screen):
    '''Screen which shows the electrode signal quality information'''

    instruct = "Electrode Quality\n\nAdjust headset until all electrodes are green\n(or noise to signal ratio < 5)"
    def __init__(self, window, noisetag, nch=4, duration=3600*1000, waitKey:bool=True, waitMouse:bool=True, label:str=None):
        super().__init__(window, label=label)

        self.noisetag = noisetag
        self.t0 = None  # timer for the duration
        self.duration = duration
        self.waitKey, self.waitMouse = waitKey, waitMouse
        self.clearScreen = True
        self.isRunning = False
        self.update_nch(nch)
        self.dataringbuffer = deque()  # deque so efficient sliding data window
        self.datawindow_ms = 4000  # 5seconds data plotted
        self.datascale_uv = 20  # scale of gap between ch plots
        print("Electrode Quality (%dms)"%(duration))

    def getTimeStamp(self):
        return (int(time.perf_counter()*1000) % (1<<31))

    def update_nch(self, nch):
        self.batch      = pyglet.graphics.Batch()
        self.background_group = pyglet.graphics.OrderedGroup(0)
        self.foreground_group = pyglet.graphics.OrderedGroup(1)
        winw, winh = self.window.get_size()
        if nch > 12: # 2-col mode
            col2ch = nch//2
            r = min((winh*.8)/(col2ch+1),winw*.2/2)
        else:
            col2ch = nch+1
            r = min((winh*.8)/(nch+1), winw*.2)
        # make a sprite to draw the electrode qualities
        img = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(2, 2)
        # anchor in the center to make drawing easier
        img.anchor_x = 1
        img.anchor_y = 1
        self.labels  = [None]*nch
        self.sprites = [None]*nch
        self.quallabels=[None]*nch
        self.linebboxs = [None]*nch # bounding box for the channel line
        for i in range(nch):
            # get the rectangle this channel should remain inside
            if nch<col2ch:
                chrect = [.05*winw,(i+1)*r,.9*winw,r] # xywh
            else:
                if i < col2ch: # left-col
                    chrect = [.025*winw,(i+1)*r,.45*winw,r] # xywh
                else: # right col
                    chrect = [.525*winw,(i-col2ch+1)*r,.45*winw,r] # xywh

            x,y,w,h = chrect
            # channel label name
            self.labels[i] = pyglet.text.Label("{:2d}".format(i), font_size=26,
                                            x=x+r*.5, y=y,
                                            color=self.get_linecol(i),
                                            anchor_x='right',
                                            anchor_y='center',
                                            batch=self.batch,
                                            group=self.foreground_group)

            # signal quality box
            # convert to a sprite and make the right size, N.B. anchor'd to the image center!
            self.sprites[i] = pyglet.sprite.Sprite(img, x=x+r*.8, y=y,
                                                batch=self.batch,
                                                group=self.background_group)
            # make the desired size
            self.sprites[i].update(scale_x=r*.4/img.width, scale_y=r*.8/img.height)
            # and a text label object
            self.quallabels[i] = pyglet.text.Label("{:2d}".format(i), font_size=15,
                                            x=x+r*.8, y=y,
                                            color=(255,255,255,255),
                                            anchor_x='center',
                                            anchor_y='center',
                                            batch=self.batch,
                                            group=self.foreground_group)
            # bounding box for the datalines
            self.linebboxs[i] = (x+r, y, w-.5*r, h)
        # title for the screen
        self.title=pyglet.text.Label(self.instruct, font_size=32,
                                     x=winw*.1, y=winh, color=(255, 255, 255, 255),
                                     anchor_y="top",
                                     width=int(winw*.9),
                                     multiline=True,
                                     batch=self.batch,
                                     group=self.foreground_group)

    def reset(self):
        self.isRunning = False

    def is_done(self):
        # check termination conditions
        isDone=False
        if not self.isRunning:
            return False
        if self.waitKey:
            if self.window.last_key_press:
                self.key_press = self.window.last_key_press
                isDone = True
                self.window.last_key_press = None
        if self.waitMouse:
            if self.window.last_mouse_release:
                self.mouse_release = self.window.last_mouse_release
                isDone = True
                self.window.last_mouse_release = None
        if self.getTimeStamp() > self.t0+self.duration:
            isDone=True
        if isDone:
            self.noisetag.removeSubscription("D")
            self.noisetag.modeChange("idle")
        return isDone

    def get_linecol(self,i):
        col = [0,0,0,255]; col[i%3]=255
        return col

    def draw(self, t):

        '''Show a set of colored circles based on the lastSigQuality'''
        if not self.isRunning:
            self.isRunning = True # mark that we're running
            self.t0 = self.getTimeStamp()
            self.noisetag.addSubscription("D") # subscribe to "DataPacket" messages
            self.noisetag.modeChange("ElectrodeQuality")
            self.dataringbuffer.clear()
        if self.clearScreen:
            self.window.clear()
        # get the sig qualities
        electrodeQualities = self.noisetag.getLastSignalQuality()
        if not electrodeQualities: # default to 4 off qualities
            electrodeQualities = [.5]*len(self.sprites)

        if len(electrodeQualities) != len(self.sprites):
            self.update_nch(len(electrodeQualities))

        issig2noise = True #any([s>1.5 for s in electrodeQualities])
        # update the colors
        #print("Qual:", end='')
        for i, qual in enumerate(electrodeQualities):
            self.quallabels[i].text = "%2.0f"%(qual)
            #print(self.label[i].text + " ", end='')
            if issig2noise:
                qual = log10(qual)/1 # n2s=50->1 n2s=10->.5 n2s=1->0
            qual = max(0, min(1, qual))
            qualcolor = (int(255*qual), int(255*(1-qual)), 0) #red=bad, green=good
            self.sprites[i].color=qualcolor
        #print("")
        # draw the updated batch
        self.batch.draw()

        # get the raw signals
        msgs=self.noisetag.getNewMessages()
        for m in msgs:
            if m.msgID == DataPacket.msgID:
                print('D', end='', flush=True)
                self.dataringbuffer.extend(m.samples)
                if self.getTimeStamp() > self.t0+self.datawindow_ms: # slide buffer
                    # remove same number of samples we've just added
                    for i in range(len(m.samples)):
                        self.dataringbuffer.popleft()


        if self.dataringbuffer:
            if len(self.dataringbuffer[0]) != len(self.sprites):
                self.update_nch(len(self.dataringbuffer[0]))

            # transpose and flatten the data
            # and estimate it's summary statistics
            from statistics import median

            # CAR
            dataringbuffer =[]
            for t in self.dataringbuffer:
                mu = median(t)
                dataringbuffer.append([c-mu for c in t])
            
            # other pre-processing
            data = []
            mu = [] # mean
            mad = [] # mean-absolute-difference
            nch=len(self.linebboxs)
            for ci in range(nch):
                d = [ t[ci] for t in dataringbuffer ]
                # mean last samples
                tmp = d[-int(len(d)*.2):]
                mui = sum(tmp)/len(tmp)
                # center (in time)
                d = [ t-mui for t in d ]
                # scale estimate
                madi = sum([abs(t-mui) for t in tmp])/len(tmp)
                data.append(d)
                mu.append(mui)
                mad.append(madi)

            
            datascale_uv = max(5,median(mad)*4)

            for ci in range(nch):
                d = data[ci]
                # map to screen coordinates
                bbox=self.linebboxs[ci]

                # downsample if needed to avoid visual aliasing
                #if len(d) > (bbox[2]-bbox[1])*2:
                #    subsampratio = int(len(d)//(bbox[2]-bbox[1]))
                #    d = [d[i] for i in range(0,len(d),subsampratio)]

                # map to screen coordinates
                xscale = bbox[2]/len(d)
                yscale = bbox[3]/datascale_uv #datascale_uv # 10 uV between lines
                y = [ bbox[1] + s*yscale for s in d ]
                x = [ bbox[0] + i*xscale for i in range(len(d)) ]
                # interleave x, y to make gl happy
                coords = tuple( c for xy in zip(x, y) for c in xy )
                # draw this line
                col = self.get_linecol(ci)
                pyglet.graphics.glColor4d(*col)
                pyglet.gl.glLineWidth(1)
                pyglet.graphics.draw(len(d), pyglet.gl.GL_LINE_STRIP, ('v2f', coords))

                # axes scale
                x = bbox[0]+bbox[2]+20 # at *right* side of the line box
                y = bbox[1]
                pyglet.graphics.glColor4d(255,255,255,255)
                pyglet.gl.glLineWidth(10)
                pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                                        ('v2f', (x,y-10/2*yscale, x,y+10/2*yscale)))



if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    # make a noisetag object without the connection to the hub for testing
    nt = Noisetag(stimSeq='mgold_65_6532.txt', utopiaController=None)
    window = initPyglet(width=640, height=480)
    screen = ElectrodeQualityScreen(window, nt, symbols=[['1','2'],['3','4']])
    # wait for a connection to the BCI
    nt.autoconnect()
    # run the screen with the flicker
    run_screen(window, screen)