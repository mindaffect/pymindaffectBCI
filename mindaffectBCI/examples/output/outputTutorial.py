# Copyright (c) 2019 MindAffect B.V. 
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

from utopiaclient import *

def doAction(msgs):
    for msg in msgs:
        # skip non-selection messages
        if not msg.msgID==Selection.msgID :
            continue
        try:
            # get the function to execute for selections we are responsible for
            selectionAction = objectID2Actions[msg.objID]
            # run the action function
            selectionAction(msg.objID)
        except KeyError:
            # This is fine, just means it's a selection we are not responsible for
            Pass

# the set of actions to perform
def forward(objID):
    print('move forward')
def backward(objID):
    print('move backward')
def left(objID):
    print('move left')
def right(objID):
    print('move right')

# map from objectIDs to the function to execute
objectID2Actions = { 64:forward, 65:backward, 66:left, 67:right }

client = UtopiaClient()
client.autoconnect()
client.sendMessage(Subscribe(client.getTimeStamp(),"S"))
while True:
    newmsgs=client.getNewMessages()
    doAction(newmsgs)
