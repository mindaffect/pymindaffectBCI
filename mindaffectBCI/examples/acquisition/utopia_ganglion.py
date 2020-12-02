from mindaffectBCI.examples.acquisition.ganglion import OpenBCIGanglion
import asyncore # needed for the openBCI server
import numpy as np
from time import time,sleep
import socket

# add utopiaclient to path and import
import sys, os
from mindaffectBCI import utopiaclient

SCALE_FACTOR_EEG = 1000000.0/(1.2*8388607.0*1.5*51.0) #uV/count
SCALE_FACTOR_AUX = 0.032
LOGINTERVAL_S=2

lastdata=None # last valid data seen
lastsamp=None # last sample id processed
lastcount=None # number of times we have seen this sample id -- for multi-sample packets
t0=None # start time
nextLogTime=None # time print next log message
nSamp=0 # number samples processed so far
ftBlockSample=200/20 # minimum number samples in each block sent, so stream @20hz
nBlock=0 # number blocks sent to ft-buffer
ftBlock=[] # storage for the current block samples to go to ft-buffer
logFile=None # "ganglion.log"
fSample=None
samp0  =None

def getTime_ms():
    if client:
        return client.getTimeStamp()
    return None
def getTime():
    return getTime_ms()/1000

def incsampCount(lastsample,count):
    ''' wrap arround sample incrementer, with repeate tracker '''
    if lastsample==0 :
        sample=101
        count=0
    elif count==0 :
        sample=lastsample
        count=count+1
    elif count==1 :
        if lastsample==200 or lastsample==100: # wrap around
            sample=0
            count=1 # BODGE: 0 doesn't repeat
        else : # increment sample counter
            sample=lastsample+1
            count=0
    return (sample,count)

nerr=0 # number of samples with sample underestimate
estndropped=0 # estimated magnitude of the missing samples
epsilon_err=1 # size of missing sample deadzone
thresh_nerr_s=6 # number seconds consistently to few samples to padd
thresh_track_s=[6,300] # time-constants for tracking slow drift in offset
def trackDroppedSamples(nsamp,ts_ms):
    '''function to detect dropped samples so can insert padding'''
    global nerr, estndropped, samp0
    if nsamp<fSample*10 : # warm-up the offset
        samp0est = nsamp - ts_ms*fSample/1000
        samp0 = max(samp0,samp0est) if not samp0 is None else samp0est
        estndropped=fSample*10
        return 0

    # update the tracking info
    samp_est = int(ts_ms/1000*fSample + samp0)
    err = samp_est - nsamp
    # slow tracking of sample offset
    if err < 0 : # proportional step for neg err => quadratic-lowerbound
        samp0 = samp0 - err/(thresh_track_s[0]*fSample)
    else : # fixed step for positive err => linear upper-bound
        samp0 = samp0 - min(1,err)/(thresh_track_s[1]*fSample)

    if err < epsilon_err :
        # inside deadzone
        nerr = 0
        return 0

    # outside the deadzone, i.e. err>=epsilon_err
    # track the total time with consistently fewer samples than expected
    nerr = nerr + 1
    
    if nerr>thresh_nerr_s*fSample/2 : # record the size of the offset
        estndropped = min(estndropped,err)

    # if long enough with missing samples -> mark to insert the missing
    if nerr>thresh_nerr_s*fSample :
        nerr    = 0
        ndropped= estndropped
        estndropped = fSample*10 # start with inf
        return ndropped # return the delta to the error



def processandpadsample(sample):
    global nSamp, ftBlock, lastdata, lastsamp, lastcount
    data=[d*SCALE_FACTOR_EEG for d in sample.channels_data] # [ nEEG ]
    sample.sample_number=sample.id

    # get the sample number we are looking for
    if lastsamp :
      nextsamp,nextcount=incsampCount(lastsamp,lastcount)
    else:
      nextsamp =sample.sample_number
      nextcount=0 # always start with 1st sample in packet

    if len(data)==0 : 
        print("%d) Packet with no channel data, skipped!"%(nSamp))
        return
    
    # check for missing samples, and pad if necessary   
    if not sample.sample_number==nextsamp :
        print("%d) skipped a sample! (%d,%d)->%d, padding: "%(nSamp,lastsamp,lastcount,sample.sample_number),end='')
        if not lastdata is None :
            padsamp=0
            while not nextsamp == sample.sample_number:
                ftBlockAppend(lastdata,nextsamp,nextcount,'s')
                nextsamp,nextcount = incsampCount(nextsamp,nextcount)
                print("s",end="")
            print()

    # second check based on tracking the sample arrival time
    ndroppedsamp = trackDroppedSamples(nSamp,getTime_ms())
    if ndroppedsamp :
        print("%d) %d dropped samples detected, padding: "%(nSamp,ndroppedsamp))
        if not lastdata is None :
            for i in range(ndroppedsamp) : 
                ftBlockAppend(lastdata,lastsamp,lastcount,'d')
                print("d",end="")
            print()
        print("%d %d: (%d,%d)->%d end-dropped"%(getTime_ms(),nSamp,lastsamp,lastcount,sample.sample_number))
      
    # finally put the actual new data
    # put this data sample and updated tracking info
    ftBlockAppend(data,sample.sample_number,None)


def ftBlockAppend(data,sample_number,sample_count=None,why=0):
    '''function to add a new data sample to the block data to send to ftbuffer,
        and log info about where the data came from'''
    global nSamp, ftBlock, lastdata, lastsamp, lastcount
    ftBlock.append(data)
    nSamp = nSamp + 1
    # store info on the last data we have processed
    lastdata = data
    if sample_count is None:
        lastcount = 1 if sample_number==lastsamp else 0
    else:
        lastcount = sample_count
    lastsamp  = sample_number
    if logFile :
        why = ord(why) if type(why) is str else why
        print("%d %d %d %d %d"%(getTime_ms(),nSamp,why,lastsamp,lastcount),file=logFile)
    return nSamp
    
def utopia_putsamples(samples):
    '''process multiple samples at a time'''
    global nSamp, nBlock, ftBlock
    for sample in samples:
        processandpadsample(sample)

    if len(ftBlock)>ftBlockSample:
        client.getNewMessages(0) # flush the incomming message queue
        client.sendMessage(utopiaclient.DataPacket(getTime_ms(),ftBlock))
        nBlock = nBlock+1
        ftBlock= []
    printLog(nSamp,nBlock)    

def printLog(nSamp,nBlock):
    ''' textual logging of the data arrivals etc.'''
    global t0, nextLogTime
    if getTime() > nextLogTime:
        elapsed = getTime()-t0
        print("%d %d %f %f (samp,blk,s,hz)"%(nSamp,nBlock,elapsed,nSamp/elapsed),flush=True)
        nextLogTime=getTime()+LOGINTERVAL_S

def init_bluetooth_settings(min_interval=6,max_interval=9,latency=0):
    import os
    pypath = os.path.dirname(os.path.abspath(__file__))
    shname = 'initGanglionBluetoothSettings.sh'
    resp = os.system("sudo bash {} {} {} {}".format(os.path.join(pypath,shname),min_interval,max_interval,latency))
    if resp != 0:
        print("Error setting the BLE parameters!  Perhaps you need to set the `initGanglionBluetoothSettings.sh` to run as root or as executable?")
    # try:
    #     with open("/sys/kernel/debug/bluetooth/hci0/conn_min_interval",'w') as f:
    #         print("%d"%(min_interval),file=f)
    #     with open("/sys/kernel/debug/bluetooth/hci0/conn_max_interval",'w') as f:
    #         print("%d"%(max_interval),file=f)
    #     with open("/sys/kernel/debug/bluetooth/hci0/latency",'w') as f:
    #         print("%d"%(latency),file=f)
    # except:
    #     print("Error setting the BLE parameters!  perhaps you need to run as root?")
    #     pass

board=None
client=None
def initConnections(host=None,obcimac=None):
    global board
    global client
    global fSample
    if sys.platform == "linux":
        init_bluetooth_settings()

    print("Opening OpenBCI Ganglion")
    board = OpenBCIGanglion(mac=obcimac,sample_rate=200)
    nChans = 4
    fSample= board.sample_rate
    print("Opened openBCI %s on %s with %d ch @ %g hz"%(board.board_type,obcimac,nChans,fSample))
    print("Opening ftBuffer connection") 

    client=utopiaclient.UtopiaClient()
    client.disableHeartbeats() # disabled for data sources
    client.autoconnect(host)
    # don't subscribe to anything
    client.sendMessage(utopiaclient.Subscribe(getTime_ms(),""));
    print("Putting header.")    
    client.sendMessage(utopiaclient.DataHeader(getTime_ms(),nChans, fSample, ""))
    return (board,client)

def readConfigFile(fname):
    '''Read the ftbuffer_ganglion cofiguration from a file'''
    config={}
    with open(fname,'r') as f:
        # parse into a dict
        for line in f:
            if line.startswith("#") : continue
            if "=" in line : 
                key,val=line.split("=",1)
                config[key.strip()]=val.strip()
    return config

def run(host=None, mac_address=''):    
    print("ftbuffer hostname:  %s"%(host))
    print("Ganglion MACadress: %s"%(mac_address))
    initConnections(host=host,obcimac=mac_address)
    # record the stream start time
    global t0, nextLogTime
    t0=getTime()
    nextLogTime=t0
    
    global logFile
    if logFile : 
        logFile = open(logFile,"w")
    
    # start the forwarder
    print("Starting the data forwarder")
    board.start_stream(utopia_putsamples)


def parse_arguments():
    import sys
    import argparse
    parser = argparse.ArgumentParser ()
    parser.add_argument ('--mac_address', type = str, help  = 'mac address', required = True)
    parser.add_argument ('--host', type = str, help  = 'hub host address', default="localhost")
    args = parser.parse_args ()
    return args

if __name__=="__main__":
    args = parse_arguments()
    run(**vars(args))
