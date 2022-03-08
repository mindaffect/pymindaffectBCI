from mindaffectBCI.examples.acquisition.cyton import OpenBCICyton
import asyncore # needed for the openBCI server
import numpy as np
from time import time, sleep
import socket

# add utopiaclient to path and import
import sys, os
from mindaffectBCI import utopiaclient

LOGINTERVAL_S = 2
SCALE_FACTOR_EEG = 1000000 * 4.5 / 24 / (2**23 - 1)
SCALE_FACTOR_AUX = .002 / 2**4

lastdata = None # last valid data seen
lastsamp = None # last sample id processed
lastcount = None # number of times we have seen this sample id -- for multi-sample packets
t0 = None # start time
nextLogTime = None # time print next log message
nSamp = 0 # number samples processed so far
ftBlockSample = 200/20 # minimum number samples in each block sent, so stream @20hz
nBlock = 0 # number blocks sent to ft-buffer
ftBlock = [] # storage for the current block samples to go to ft-buffer
LOGFILE = None # "ganglion.log"
fSample = None
samp0   = None

def getTime_ms():
    if client:
        return client.getTimeStamp()
    return None
def getTime():
    return getTime_ms()/1000

def incsampCount(sample, count):
    ''' wrap arround sample incrementer, with repeate tracker '''
    sample = sample+1 if sample < 255 else 0
    return (sample, count)

nerr = 0 # number of samples with sample underestimate
estndropped = 0 # estimated magnitude of the missing samples
epsilon_err = 1 # size of missing sample deadzone
thresh_nerr_s = 6 # number seconds consistently to few samples to padd
thresh_track_s = [6, 300] # time-constants for tracking slow drift in offset
def trackDroppedSamples(nsamp, ts_ms):
    '''function to detect dropped samples so can insert padding'''
    global nerr, estndropped, samp0
    if nsamp < fSample*10: # warm-up the offset
        samp0est = nsamp - ts_ms*fSample/1000
        samp0 = max(samp0, samp0est) if not samp0 is None else samp0est
        estndropped = fSample*10
        return 0

    # update the tracking info
    samp_est = int(ts_ms/1000*fSample + samp0)
    err = samp_est - nsamp
    # slow tracking of sample offset
    if err < 0: # proportional step for neg err => quadratic-lowerbound
        samp0 = samp0 - err/(thresh_track_s[0]*fSample)
    else: # fixed step for positive err => linear upper-bound
        samp0 = samp0 - min(1, err)/(thresh_track_s[1]*fSample)

    if err < epsilon_err:
        # inside deadzone
        nerr = 0
        return 0

    # outside the deadzone, i.e. err>=epsilon_err
    # track the total time with consistently fewer samples than expected
    nerr = nerr + 1
    
    if nerr > thresh_nerr_s*fSample/2: # record the size of the offset
        estndropped = min(estndropped, err)

    # if long enough with missing samples -> mark to insert the missing
    if nerr > thresh_nerr_s*fSample:
        nerr = 0
        ndropped = estndropped
        estndropped = fSample*10 # start with inf
        return ndropped # return the delta to the error
    return 0

def processandpadsample(sample):
    global nSamp, ftBlock, lastdata, lastsamp, lastcount
    data = [d*SCALE_FACTOR_EEG for d in sample.channels_data] # [ nEEG ]
    sample.sample_number = sample.id

    # get the sample number we are looking for
    if lastsamp:
      nextsamp, nextcount = incsampCount(lastsamp, lastcount)
    else:
      nextsamp = sample.sample_number
      nextcount = 0 # always start with 1st sample in packet

    if len(data) == 0:
        print("%d) Packet with no channel data, skipped!"%(nSamp))
        return
    
    # check for missing samples, and pad if necessary   
    if not sample.sample_number == nextsamp:
        print("%d) skipped a sample! (%d, %d)->%d, padding: "%(nSamp, lastsamp, lastcount, sample.sample_number), end = '')
        if not lastdata is None:
            while not nextsamp == sample.sample_number:
                ftBlockAppend(lastdata, nextsamp, nextcount, 's')
                nextsamp, nextcount = incsampCount(nextsamp, nextcount)
                print("s", end="")
            print()

    # second check based on tracking the sample arrival time
    ndroppedsamp = trackDroppedSamples(nSamp, getTime_ms())
    if ndroppedsamp:
        print("%d) %d dropped samples detected, padding: "%(nSamp, ndroppedsamp))
        if not lastdata is None:
            for _ in range(ndroppedsamp):
                ftBlockAppend(lastdata, lastsamp, lastcount, 'd')
                print("d", end="")
            print()
        print("%d %d: (%d,%d)->%d end-dropped"%(getTime_ms(), nSamp, lastsamp, lastcount, sample.sample_number))
      
    # finally put the actual new data
    # put this data sample and updated tracking info
    ftBlockAppend(data, sample.sample_number, None)


def ftBlockAppend(data, sample_number, sample_count=None, why=0):
    '''function to add a new data sample to the block data to send to ftbuffer,
        and log info about where the data came from'''
    global nSamp, ftBlock, lastdata, lastsamp, lastcount
    ftBlock.append(data)
    nSamp = nSamp + 1
    # store info on the last data we have processed
    lastdata = data
    if sample_count is None:
        lastcount = 1 if sample_number == lastsamp else 0
    else:
        lastcount = sample_count
    lastsamp = sample_number
    if LOGFILE:
        why = ord(why) if isinstance(why, str) else why
        print("%d %d %d %d %d"%(getTime_ms(), nSamp, why, lastsamp, lastcount), file=LOGFILE)
    return nSamp
    
def utopia_putsamples(samples):
    '''process multiple samples at a time'''
    global nSamp, nBlock, ftBlock
    if hasattr(samples, '__iter__'): # set samples
        for sample in samples:
            processandpadsample(sample)
    else: # single sample
        processandpadsample(samples)

    if len(ftBlock) > ftBlockSample:
        client.getNewMessages(0) # flush the incomming message queue
        client.sendMessage(utopiaclient.DataPacket(getTime_ms(), ftBlock))
        nBlock = nBlock+1
        ftBlock = []
    printLog(nSamp, nBlock)

def printLog(nSamp, nBlock):
    ''' textual logging of the data arrivals etc.'''
    global t0, nextLogTime
    if getTime() > nextLogTime:
        elapsed = getTime()-t0
        print("%d %d %f %f (samp,blk,s,hz)"%(nSamp, nBlock, elapsed, nSamp/elapsed), flush=True)
        nextLogTime = getTime()+LOGINTERVAL_S

board = None
client = None
def initConnections(host=None, obciport=None):
    global board, client, fSample, nChans
    print("Opening OpenBCI Cyton on {}".format(obciport)) 
    # N.B. latency is min time between packets in us -> 1 packet = 5 samples -> 50hz
    #      high_speed => binary format
    #      sample_rate => requested sample rate
    board = OpenBCICyton(port=obciport)
    
    # Get the board info    
    nChans = 16 if board.daisy else 8
    fSample = 250
    print("Opened openBCI on %s with %d ch @ %g hz"%(board.port, nChans, fSample))

    client = utopiaclient.UtopiaClient()
    client.disableHeartbeats() # disabled for data sources
    client.autoconnect(host)
    # don't subscribe to anything
    client.sendMessage(utopiaclient.Subscribe(getTime_ms(), ""));
    print("Putting header.")    
    client.sendMessage(utopiaclient.DataHeader(getTime_ms(), nChans, fSample, ""))
    return (board, client)

def parse_arguments():
    import sys
    argv = sys.argv
    print("Arguments:",*argv)
    # N.B. argv0 is the script name
    host=argv[1] if len(argv)>1 else 'localhost'
    obciport=argv[2] if len(argv)>2 else 'com4'
    return dict(host=host, obciport=obciport)

def run(host='localhost', obciport='com4', **kwargs):
    initConnections(host=host, obciport=obciport)
    # record the stream start time
    global t0, nextLogTime
    t0 = getTime()
    nextLogTime = t0
    
    global LOGFILE
    if LOGFILE:
        LOGFILE = open(LOGFILE, "w")
    # start the forwarder
    print("Starting the data forwarder")
    board.start_stream(utopia_putsamples)

if __name__ == "__main__":
    args=parse_arguments()
    run(**args)
