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
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from mindaffectBCI.utopiaclient import UtopiaClient,Subscribe,DataPacket,SignalQuality
from collections import deque

# rate at which to update the display
updatet_ms=50
# amout of data to display
datawindow_ms=5000
# gap in data units between channel lines
linespace_uv=20

def makeLines(ax, nch):
  line = [None]*nch
  bbox = [None]*nch
  h = ax.get_ylim()[1]/nch
  x = ax.get_xlim()[0]
  w = ax.get_xlim()[1]-x
  for li in range(nch):
    y = li*h
    bbox[li] = (x, y, w, h)
    line[li], = ax.plot(0, 0)
  return (line, bbox)

# init the display
fig=plt.figure(1)
fig.clear()
ax=fig.add_subplot(111) # single fullfig axes

ax.set_xlim(-datawindow_ms, 0) # xlim set to match data duration
ax.set_xlabel('time (ms)')
ax.set_ylabel('uV')

# make lines and bounding boxes for the lines
line=[]
bbox=[]
fig.show()

# connect to the mindaffectDecoder
U=UtopiaClient()
U.autoconnect(timeout_ms=5000, queryifhostnotfound=True)
# subscribe to the raw-data messages
U.sendMessage(Subscribe(None, "D"))

# render loop
qual=[]
dataringbuffer = deque()
t0 = U.getTimeStamp()
while True:
  # get enough new data for update window
  t=U.getTimeStamp()
  while U.getTimeStamp() < t+updatet_ms: 
    msgs = U.getNewMessages(0)
    for m in msgs:
      if m.msgID == DataPacket.msgID:
        print('D', end='', flush=True)
        if U.getTimeStamp() > t0+datawindow_ms: # slide buffer
          dataringbuffer.popleft()
        dataringbuffer.append(m.samples)
      elif m.msgID == SignalQuality.msgID:
        qual = m.qual
  # draw the lines
  if dataringbuffer:
    nch=len(dataringbuffer[0][0])
    if not len(line) == nch: # update the line set if number channel changed
      ax.set_ylim(-1*linespace_uv,nch*linespace_uv) # ylim are set to match uV
      line, bbox = makeLines(ax,nch)
    for ci in range(nch):
      # flatten into list samples for each channel
      d = [t[ci] for m in dataringbuffer for t in m]
      # map to screen coordinates
      yscale = bbox[ci][3]/linespace_uv # 10 uV between lines
      mu = sum(d)/len(d) # center
      y = [bbox[ci][1] + (s-mu)*yscale for s in d]
      x = [bbox[ci][0] + bbox[ci][2]*i/len(y) for i in range(len(y))]
      line[ci].set_data(x,y)
      # add qual info
      
  # force plot to re-draw 
  fig.canvas.draw()
  fig.canvas.flush_events()
