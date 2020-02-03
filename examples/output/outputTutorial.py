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
