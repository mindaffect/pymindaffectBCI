from mindaffectBCI import utopiaclient
from pylsl import StreamInlet, resolve_stream
from mindaffectBCI.utopiaclient import UtopiaClient

def run(client:UtopiaClient=None, host:str=None, inlet:StreamInlet=None, stream_type:str="Markers"):
    if client is None:
        client = UtopiaClient()
        # resolve and connect to the utopia hub
        client.autoconnect(host, queryifhostnotfound=True)
        client.disableHeartbeats() # no heartbeats as we use the lsl time-stamps
        client.sendMessage(utopiaclient.Subscribe(None,"")) # don't subscribe to anything

    # first resolve a marker stream on the lab network
    if inlet is None:
        print("looking for a marker stream...")
        streams = resolve_stream('type', stream_type)

        # create a new inlet to read from the stream
        inlet = StreamInlet(streams[0])

    # loop-forever forwarding markers to utopia
    markerdict=dict() # mapping from marker names to integer states
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        print("got %s at time %s" % (sample[0], timestamp))
        # transform into a stimulus event message and 
        client.sendMessage(lslsample2stimulusevent(timestamp,sample,markerdict))

def lslsample2stimulusevent(sample,timestamp,markerdict):
    """map from lsl marker samples to mindaffect stimulus events

    Args:
        sample ([type]): lsl marker sample
        timestamp ([type]): lsl marker time-stamp
        markerdict ([type]): mapping from string marker names to byte event codes

    Returns:
        [type]: [description]
    """
    objIDs=[]
    stimState=[]
    for si,ss in enumerate(sample):
        if isinstance(ss,int):
            pass
        elif isinstance(ss,str):
            if not ss in markerdict:
                markerdict[ss]=len(markerdict)+1
            ss = markerdict[ss]
        else:
            print("Warning don't know how to handle this type of marker stream")
        objIDs.append(si)
        stimState.append(ss)
    return utopiaclient.StimulusEvent(timestamp,objIDs,stimState)


if __name__=="__main__":
    run()