import numpy as np


def packBoxes(Xs, Ys):
    """Give a set of X,Y positions pack non-overlapping rectangular boxes
     rX,rY=packBoxes(Xs,Ys)
     Inputs:
     Xs - [N x 1] x positions
     Ys - [N x 1] y positions
     Outputs:
     rX - [N x 1] x radius
     rY - [N x 1] y radius
    """
    try:
        Xs = np.array(Xs, dtype=np.float32)
        Ys = np.array(Ys, dtype=np.float32)
    except:
        raise TypeError('Only for lists/arrays')

    Xs = Xs.ravel()
    Ys = Ys.ravel()

    N = len(Xs)
    # Now, Find the all plots pairwise distance matrix, w.r.t. this scaling
    Dx = np.abs(Xs[np.newaxis, :] - Xs[:, np.newaxis])
    Dy = np.abs(Ys[np.newaxis, :] - Ys[:, np.newaxis])
    rX = np.zeros(N)
    rY = np.zeros(N)
    for i in range(N):
        Dx[i,i]=np.inf; Dy[i,i]=np.inf
        rX[i] = np.min(Dx[Dx[:, i] >= Dy[:,i], i]) / 2
        rY[i] = np.min(Dy[Dy[:, i] >= Dx[:,i], i]) / 2

    # Unconstrained boundaries are limited by the max/min of the constrained ones
    # or .5 if nothing else...
    if np.any(np.isinf(rX)):
        if np.all(np.isinf(rX)):
            rX[:] = 0.5
        else:
            unconsX = np.isinf(rX)
            rX[unconsX] = 0
            minX = np.min(Xs - rX)
            maxX = np.max(Xs + rX)
            rX[unconsX] = np.min(np.stack((maxX - Xs[unconsX], Xs[unconsX] - minX), 1), 1)

    if np.any(np.isinf(rY)):
        if np.all(np.isinf(rY)):
            rY[:] = .5
        else:
            unconsY = np.isinf(rY)
            rY[unconsY] = 0
            minY = np.min(Ys - rY)
            maxY = np.max(Ys + rY)
            rY[unconsY] = np.min(np.stack((maxY - Ys[unconsY], Ys[unconsY] - minY), 1), 1)

    return rX, rY


def testCase():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    Xs = np.random.rand(10)*10  # linspace(0,10,10)
    Ys = np.random.rand(10)*10  # linspace(0,10,10)
    rX, rY = packBoxes(Xs, Ys)

    a = plt.axes()
    a.set(xlim=[0, 10], ylim=[0, 10])
    for i in range(len(Xs)):
        a.add_patch(Rectangle((Xs[i]-rX[i], Ys[i]-rY[i]), 2*rX[i], 2*rY[i]))
    plt.show(block=True)


if __name__ == '__main__':
    testCase()
