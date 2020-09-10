from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_message
from time import sleep

class FileProxyHub:
    ''' Proxy UtopiaClient which gets messages from a saved log file '''
    def __init__(self, filename:str=None, speedup:float=None, use_server_ts:bool=True):
        self.filename = filename
        if self.filename is None or self.filename == '-':
            # default to last log file if not given
            import glob
            import os
            files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
            self.filename = max(files, key=os.path.getctime)
        self.speedup = speedup
        self.isConnected = True
        self.lasttimestamp = None
        self.use_server_ts = use_server_ts
        self.file = open(self.filename,'r')
    
    def getTimeStamp(self):
        return self.lasttimestamp
    
    def autoconnect(self, *args,**kwargs):
        pass

    def sendMessage(self, msg):
        pass

    def getNewMessages(self, timeout_ms):
        msgs = []
        for line in self.file:
            msg = read_mindaffectBCI_message(line)
            if msg is None:
                continue
            # re-write timestamp to server time stamp
            if self.use_server_ts:
                msg.timestamp = msg.sts
            # initialize the time-stamp tracking
            if self.lasttimestamp is None: 
                self.lasttimestamp = msg.timestamp
            # add to outgoing message queue
            msgs.append(msg)
            # check if should stop
            if self.lasttimestamp+timeout_ms < msg.sts:
                break
        else:
            # mark as disconneted at EOF
            self.isConnected = False
        if self.speedup :
            sleep(timeout_ms/1000./self.speedup)
        # update the time-stamp cursor
        self.lasttimestamp = self.lasttimestamp + timeout_ms
        return msgs

if __name__=="__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv)>1 else None

    U = FileProxyHub(filename)
    t=0
    while U.isConnected:
        msgs = U.getNewMessages(100)
        print("{}) {}msgs\n{}\n".format(t,len(msgs),msgs))