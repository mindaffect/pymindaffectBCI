import os
import numpy as np

def getPosInfo(chnames=None,capFile='1010',overridechnms=0,prefixMatch=False,verb=0,capDir=None):
    """
     add electrode position info to a dimInfo structure
    
     cnames,pos2d,pos3d,iseeg=getPosInfo(di,capFile,overridechnms,prefixMatch,verb,capDir)
    
    Inputs:
      cnames -- channel names to get pos-info for
      capFile -- file name of a file which contains the pos-info for this cap
      overridechnms -- flag that we should ignore the channel names in di
      prefixMatch  -- [bool] match channel names if only the start matches?
      verb         -- [int] verbosity level  (0)
      capDir       -- 'str' directory to search for capFile
    Outputs:
      cnames -- [str] the names of the channels
      xy -- (nCh,2):float the 2d position of the channels
      xyz -- (nCh,2):float the 3d position of the channels
      iseeg - (nCh,):bool flag if this is an eeg channel, i.e. if it's name matched
    """
    cfnames,latlong,xy,xyz,cfile=readCapInf(capFile,capDir)
    
    if overridechnms  or (isinstance(cfnames,np.ndarray) and np.issubdtype(cfnames.dtype,np.number)) :
        if chnames is None:
            chnames = cfnames
        else:
            if len(chnames)<len(cfnames):
                chnames = cfnames[1:len(chnames)]
            else:
                chnames[1:len(cfnames)]=cfnames
                
    # Add the channel position info, and iseeg status
    # pre-allocate for the output
    pos2d=np.zeros((len(chnames),2),dtype=np.float)
    pos3d=np.zeros((len(chnames),3),dtype=np.float)
    iseeg=np.zeros((len(chnames)),  dtype=np.bool)
    cfmatched=np.zeros((len(cfnames),1),dtype=np.bool) # file channels matched
    for i,chnamei in enumerate(chnames):
        ti=None
        if isinstance(chnamei,str):
            for j,cfnamej in enumerate(cfnames):
                if verb>1 : print("Trying %s == %s?"%(chnamei,cfnamej),end='')
                if not cfmatched[j] and chnamei.lower()==cfnamej.lower():
                    if verb>1 : print("matched")
                    ti=j
                    cfmatched[j]=True
                    break
            if prefixMatch and ti is None:
                for j in range(0,numel(cfnames)):
                    if not cfmatched[j] and lower(cfnamej).startswith(lower(chnamei)):
                        ti=j
                        cfmatched[j]=True
                        break
        else:
            if (isinstance(chnamei,np.ndarray) and i < len(cfnames)):
                ti=chnamei
                cfmatched[chnamei]=True
            else:
                ti=None
                print('WARNING:: Channel names are difficult')

        # got the match, so update the info
        if (not ti is None and ti >= 0 and ti < len(cfnames)):
            if verb > 0:
                print('%3d) Matched : %s\t ->\t %s'%(i,chnames[i],cfnames[ti]))
            chnames[i] =cfnames[ti]
            pos2d[i,:]=xy[:,ti]
            pos3d[i,:]=xyz[:,ti]
            iseeg[i]  =True
            if any(np.isnan(pos2d[i,:])) or any(np.isnan(pos3d[i,:])) or any(np.isinf(pos2d[i,:])) or any(np.isinf(pos3d[i,:])):
                iseeg[i] = False
        else:
            pos2d[i,:] = (-1,1)
            pos3d[i,:] = (-1,1,0)
            iseeg[i] = False
    
    return (chnames,pos2d,pos3d,iseeg)

def testCase():
    import getPosInfo
    cn,xy,xyz,iseeg=getPosInfo.getPosInfo(['CPz','AFz'],'1010')

def readCapInf(cap='1010.txt',capRoots=None, verb=0):
    """     read a cap file
 
    Cname,latlong,xy,xyz,capfile=readCapInf(cap,capDir)
    
     Inputs:
    cap     -- file name of the cap-file
    capRoot -- directory(s) to look for caps in (['.',mfiledir,mfiledir'/positions','../../resources/caps/'])
    """

    if capRoots is None:
        filedir = os.path.dirname(os.path.abspath(__file__))
        capRoots = ('.',filedir,os.path.join(filedir,'positions'),os.path.join(filedir,'..','..','resources','caps'));    
    elif isinstance(capRoots,basestring):
        capRoots=(capRoots)

    capDir,capFnExt=os.path.split(cap)
    capFn,sep,capExt=capFnExt.partition('.') # get extn if given
    # search given directories for the capfile
    for cr in range(0,len(capRoots)):
        capRoot=capRoots[cr]
        if len(capExt)>0 : 
            capFile=os.path.join(capRoot,capDir,capFn+'.'+capExt)
            if verb > 0: print('Trying: ' + capFile)
            if os.path.exists(capFile):
                if verb > 0: print('success')
                break
        else:
            capFile=os.path.join(capRoot,capDir,capFn+'.txt')
            if verb > 0: print('Trying: ' + capFile)
            if os.path.exists(capFile):
                capExt='txt'
                if verb > 0: print('success')
                break
            capFile=os.path.join(capRoot,capDir,capFn+'.lay')
            if verb > 0: print('Trying: ' + capFile)
            if os.path.exists(capFile):
                capExt='lay'
                if verb > 0: print('success')
                break

    if not os.path.exists(capFile):
        raise ValueError('Couldnt find the capFile: '+cap)
    
    if 'xyz' in capExt:
        Cname,x,y,z= np.loadtxt(capFile,
                                dtype={'names': ('Cname', 'x', 'y', 'z'),
                                       'formats': ('|S12', 'float', 'float', 'float')},
                                unpack=True)
        #Cname,x,y,z=textread(capFile,'b'%s %f %f %f'',nargout=4)
        xyz=np.stack((x,y,z),1).T # [3 x nCh]
        xy=xyz2xy(xyz)
        latlong=xy2latlong(xy)
    else:
        if 'xy' in capExt:
            Cname,x,y,z= np.loadtxt(capFile,
                                    dtype={'names': ('Cname', 'x', 'y'),
                                           'formats': ('|S12', 'float', 'float')},
                                    unpack=True)
            #Cname,x,y=textread(capFile,'b'%s %f %f'',nargout=3)
            xy=np.stack((x,y),1).T #[2 x nCh]
            latlong=xy2latlong(xy)
            xyz=latlong2xyz(latlong)
        else:
            if 'lay' in capExt:
                tmp,x,y,w,h,Cname=np.loadtxt(capFile,
                                             dtype={'names': ('rownum', 'x', 'y', 'w','h','Cname'),
                                                    'formats': ('float', 'float', 'float', 'float', 'float', '|S12')},
                                             unpack=True)
                #ans,x,y,w,h,Cname=textread(capFile,'b'%d %f %f %f %f %s'',nargout=6)
                xy=np.stack((x,y),1).T # [2 x nCh]
                xy= xy - np.mean(xy,1)
                xy= xy / np.sqrt(np.mean(xy*xy,1))
                latlong=xy2latlong(xy)
                xyz=latlong2xyz(latlong)
            else:
                Cname,lat,lng=np.loadtxt(capFile,
                                          dtype={'names': ('Cname','lat','lng'),
                                                 'formats': ('|S12', 'float', 'float')},
                                             unpack=True)
                #Cname,lat,lng=textread(capFile,'b'%s %f %f'',nargout=3)
                latlong=np.stack((lat,lng),1).T # [2 x nCh]
                if np.max(np.abs(latlong[:])) > 2*np.pi:
                    latlong= latlong *np.pi / 180.0
                xyz=latlong2xyz(latlong)
                xy=latlong2xy(latlong)

    Cname = np.char.decode(Cname,'utf8')
                
    return Cname,latlong,xy,xyz,capFile
    
def xyz2xy(xyz):
    "convert 3d xyz coords to unrolled 2d xy positions"
    eegch=np.logical_not(np.any(np.logical_or(isnan(xyz),isinf(xyz)),0))
    cz=mean(xyz[3,eegch])
    r=np.abs(max(abs(xyz[3,eegch] - cz))*1.1)
    if r < eps:
        r=1
    h=xyz[2,eegch] - cz
    rr=np.sqrt(np.dot(2,(r*r - np.dot(r,h))) / (r*r-h*h))
    xy=np.zeros((2,xyz.shape[1]))
    xy[0,eegch]=xyz[0,eegch]*rr
    xy[1,eegch]=xyz[1,eegch]*rr
    xy[:,logical_not(eegch)]=np.NaN
    return xy
    
def xy2latlong(xy):
    "convert xy to lat-long, taking care to prevent division by 0"
    eegch=np.logical_not(np.any(np.logical_or(np.isnan(xy),np.isinf(xy)),0))
    latlong=np.zeros(np.shape(xy))
    latlong[0,eegch]= np.sqrt(np.sum(xy[:,eegch] ** 2,1))
    latlong[1,eegch]= np.atan2(xy[1,eegch],xy[0,eegch])
    latlong[:,np.logical_not(eegch)]=np.NaN
    return latlong
    
def latlong2xyz(latlong):
    "convert lat-long into 3-d x,y,z coords on the sphere"
    eegch=np.logical_not(np.any(np.logical_or(np.isnan(latlong),np.isinf(latlong)),0))
    xyz=np.zeros((3,latlong.shape[1]))
    xyz[0,eegch]=np.sin(latlong[0,eegch]) * np.cos(latlong[1,eegch])
    xyz[1,eegch]=np.sin(latlong[0,eegch]) * np.sin(latlong[1,eegch])
    xyz[2,eegch]=np.cos(latlong[0,eegch])
    xyz[:,np.logical_not(eegch)]=np.NaN
    return xyz
    
def latlong2xy(latlong):
    "convert lat-long to 2-d unrolled x,y coordinates"
    eegch=np.logical_not(np.any(np.logical_or(np.isnan(latlong),np.isinf(latlong)),0))
    xy=np.zeros((2,latlong.shape[1]))
    xy[0,eegch]=np.sin(latlong[0,eegch]) * np.cos(latlong[1,eegch])
    xy[1,eegch]=np.sin(latlong[0,eegch]) * np.sin(latlong[1,eegch])
    return xy
    
