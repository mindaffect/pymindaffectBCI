import numpy as np
import matplotlib.pyplot as plt
from mindaffectBCI.decoder.plot_utils.packBoxes import packBoxes

def posplot(Xs=None,Ys=None,idx=None,XYs=None,interplotgap=.003,plotsposition=[0.05,0.05,.93,.90],scaling='any',sizes='any',postype='position',minSize=.05,**kwargs):
    """  
    Function to generate sub-plots at given 2-d positions
    
    [hs,(Xs,Ys),(rX,rY)]=posPlot(XYs[,options])
    Args:
     Xs OR XYs -- X (horz) positions of the plots (N,) or XYs (N,2)
     Ys -- Y (vert) positions of the plots (N,)
     idx -- subplot to make current   (1)
     scaling -- do we preserve the relative scaling of x y axes?
                'none': don't preserver, 'square' : do preserve
     sizes   -- do we constrain the plots to be the same size?
                'none': no size constraint, 'equal': all plots have same x/y size
             -- Everything else is treated as an option for axes
     plotsposition-- [4 x 1] vector of the figure box to put the plots: [x,y,w,h]
                     ([0 0 1 1])
     minSize -- minium plot size. Defaults to .05

    Returns:
     hs -- if output is requested this is the set of sub-plot handles, 
           (N,) if idx is None else scalar handle of the idx'th plot if specified
    """
    
    if len(plotsposition) == 1:
        plotsposition[0:3]=plotsposition
    elif not (len(plotsposition) == 0 or len(plotsposition)==4):
        raise ValueError('Figure boundary gap should be 1 or 4 element vector')
    
    if not type(interplotgap) is np.ndarray :
        tmp          = interplotgap
        interplotgap = np.zeros((4,1))
        interplotgap[0:3]=tmp
    elif len(interplotgap) == 1:
        tmp          = interplotgap
        interplotgap = np.zeros((4,1))        
        interplotgap[0:3]=tmp
    elif not (len(interplotgap) == 0 or len(interplotgap)==4):
        raise ValueError('Interplot gap should be 1 or 4 element vector')

    # extract the separate Xs, Ys from XYs
    if Xs is None:
        Xs = [ x[0] for x in XYs]
        Ys = [ x[1] for x in XYs]
    if Ys is None:
        Ys = [x[1] for x in Xs]
        Xs = [x[0] for x in Xs]
        
    if len(Ys) != len(Xs):
        raise ValueError('Xs and Ys *must* have same number of elements')
    
    if not idx is None and not idx in range(len(Xs)):
        raise ValueError('idx greater than the number of sub-plots')
    
    N=len(Xs)
    # Compute the radius between the points
    rX,rY=packBoxes(Xs,Ys)
    if sizes == 'equal':
        rX[:]=np.min(rX)
        rY[:]=np.min(rY)
    
    rX=np.tile(rX[:,np.newaxis],(1,2)) # allow different left/right radii [nPlot x 2]
    rY=np.tile(rY[:,np.newaxis],(1,2))
    
    # Next compute scaling for the input to the unit 0-1 square, centered on .5
    minX=np.min(Xs - rX[:,0])
    maxX=np.max(Xs + rX[:,1])
    
    minY=np.min(Ys - rY[:,0])
    maxY=np.max(Ys + rY[:,1])
    
    W=maxX - minX
    W=W / plotsposition[2]
    if W <= 0:
        W=1
    
    H=maxY - minY
    H=H / plotsposition[3]
    if H <= 0:
        H=1
    
    if not scaling is None and scaling.lower()=='square':
        W=max(W,H)
        H=max(W,H)
    
    Xs=(Xs - (maxX + minX) / 2) / W
    rX=rX / W
    Xs=Xs + plotsposition[0] + .5*plotsposition[2]
    Ys=(Ys - (maxY + minY) / 2) / H
    rY=rY / H
    Ys=Ys + plotsposition[1] + .5*plotsposition[3]
    # subtract the inter-plot gap if necessary.
    rX[:,0]=rX[:,0] - interplotgap[0]
    rX[:,1]=rX[:,1] - interplotgap[1]
    rY[:,0]=rY[:,0] - interplotgap[2]
    rY[:,1]=rY[:,1] - interplotgap[3]
    
    # Check if this is a reasonable layout
    if minSize > 0:
        print('Not enough room between points to make plot')
        rX[np.logical_or(rX <= minSize, np.isnan(rX))]=minSize
        rY[np.logical_or(rY <= minSize, np.isnan(rY))]=minSize#(min(len(emptySize),1))
    
    # generate all subplots if handles wanted
    if idx is None: 
        hs=[]
        for i in range(0,N):
            pos=[Xs[i] - rX[i,0], Ys[i] - rY[i,0], np.sum(rX[i,:]), np.sum(rY[i,:])]
            h=plt.axes(pos,*args)
            hs.append(h)
    else: # only make the idx'th plot
        pos=(Xs[idx] - rX[idx,0],Ys[idx] - rY[idx,0],
                        np.sum(rX[idx,:]),np.sum(rY[idx,:]))
        hs=plt.axes(pos,**kwargs)
    
    return hs

    
def testCase():
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.plot_utils.posplot import posplot
    plt.clf();posplot([[0,0]])
    hs=posplot(Xs=[1,2,3],Ys=[2,2,2])
    posplot(([1,2,3],[2,2,2]))
    plt.show(block=True)
    plt.clf()
    h=posplot(Xs=(1,2,3),Ys=(1,2,3))
    plt.show(block=True)
    plt.clf()
    h=posplot(Xs=np.random.uniform(size=(10)),Ys=np.random.uniform(size=(10)))
    plt.show(block=True)
    plt.clf()
    h=posplot(Xs=np.random.uniform(size=(10)),Ys=np.random.uniform(size=(10)),sizes='any')
    plt.show(block=True)
    plt.clf()
    h=posplot(Xs=(1,2,3),Ys=(1,1.5,2),sizes='any')
    plt.show(block=True)
    plt.clf()
    h=posplot(Xs=(0.2,0.6,0.7),Ys=(0.5,0.4,0.45),sizes='any')
    plt.show(block=True)
    plt.clf()
    h=posplot(Xs=np.sin(np.linspace(0,2*np.pi,10,endpoint=False)),Ys=np.cos(np.linspace(0,2*np.pi,10,endpoint=False)),sizes='any')
    plt.show(block=True)
    
if __name__=='__main__':
    testCase()