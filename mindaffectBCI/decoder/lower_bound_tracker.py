#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
# This file is part of pymindaffectBCI <https://github.com/mindaffect/pymindaffectBCI>.
#
# pymindaffectBCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pymindaffectBCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pymindaffectBCI.  If not, see <http://www.gnu.org/licenses/>

import numpy as np
from mindaffectBCI.decoder.utils import RingBuffer

class lower_bound_tracker():
    """ sliding window lower bound trend tracker, which fits a line to the lower-bound of the inputs x,y
    """   

    def __init__(self, window_size=200, C=(.1,None), outlier_thresh=(1,None), step_size=10, step_threshold=2, a0=1, b0=0, warmup_size=10):
        """sliding window lower bound trend tracker, which fits a line to the lower-bound of the inputs x,y

        Args:
            window_size (int, optional): number of points in the sliding window. Defaults to 200.
            C (tuple, optional): relative cost for over-vs-under estimates. Defaults to (.1,None).
            outlier_thresh (tuple, optional): threshold for detection of outlying over (resp. under) estimated inputs. Defaults to (1,None).
            step_size (int, optional): number of points in the step-detection sliding window. Defaults to 10.
            step_threshold (int, optional): threshold (in std-dev) for step-detection. Defaults to 2.
            a0 (int, optional): initial slope. Defaults to 1.
            b0 (int, optional): initial offset. Defaults to 0.
            warmup_size (int, optional): number of points before which we trust or fit. Defaults to 10.
        """        
        self.window_size = window_size
        self.step_size = int(step_size*window_size) if step_size<1 else step_size
        self.a0 = a0 if a0 is not None else 1
        self.b0 = b0
        self.warmup_size = int(warmup_size*window_size) if warmup_size<1 else warmup_size
        self.step_threshold = step_threshold
        self.outlier_thresh = outlier_thresh
        if not hasattr(self.outlier_thresh,'__iter__'):
            self.outlier_thresh = (self.outlier_thresh, self.outlier_thresh)
        self.C = C
        if not hasattr(self.C,'__iter__'):
            self.C = None

    def reset(self, keep_model=False):
        """reset the data for model fitting

        Args:
            keep_model (bool, optional): keep the model as well as data. Defaults to False.
        """        
        self.buffer.clear()
        if not keep_model:
            self.a = self.a0
            self.b = self.b0

    def fit(self,X,Y):
        self.buffer = RingBuffer(self.window_size, (2,), dtype=X.dtype if isinstance(X,np.ndarray) else float)
        self.append(X,Y)
        self.a = self.a0
        self.b = self.buffer[-1,1] - self.a * self.buffer[-1,0]

    def append(self,X,Y):
        if isinstance(X, np.ndarray) and X.ndim > 1:
            self.buffer.extend(np.hstack((X,Y)))
        else:
            self.buffer.append(np.array((X,Y)))

    def transform(self,X,Y):
        if not hasattr(self,'buffer'):
            self.fit(X,Y)
            return Y
        
        # add into ring-buffer
        self.append(X,Y)

        # update the fit
        self.update()

        # return Yest
        return self.getY(X)

    def update(self):
        """update the model based on the data in the sliding window
        """        
        if self.buffer.shape[0]>0 and self.buffer.shape[0] < self.warmup_size :
            if self.buffer.shape[0]>2:
                self.a = np.median(np.diff(self.buffer[:,1])/np.diff(self.buffer[:,0]))
            self.b = self.buffer[-1,1] - self.a * self.buffer[-1,0]
            return

        # get the data
        x = self.buffer[:,0].astype(float)
        y = self.buffer[:,1].astype(float)

        # center the data to improve the numerical robustness
        mu_x  = np.median(x)
        x = x - mu_x
        mu_y = np.median(y)
        y = y - mu_y

        # IRWLS fit to the data, with outlier suppression and asymetric huber loss
        # add constant feature  for the intercept
        x = np.append(x[:,np.newaxis],np.ones((x.shape[0],1),dtype=x.dtype),1)
        # weighted LS  solve
        sqrt_w = np.ones((x.shape[0],),dtype=x.dtype) # square root of weight so can use linalg.lstsq
        # start from previous solution
        a0 = self.a
        b0 = self.b + self.a*mu_x - mu_y # shift bias to compensate for shift in x/y
        a = a0
        b = b0
        # TODO[]: convergence criteria, to stop early
        for i in range(3):
            y_est = x[:,0]*a + b
            err = y - y_est # server > true, clip positive errors
            mu_err = np.mean(err)
            #print('{} mu_err={}\n'.format(i,mu_err))
            scale = np.mean(np.abs(err-mu_err))

            # 1 by default
            sqrt_w[:] = 1.0

            if self.C is not None:
                # sign based cost function

                # postive errors
                if self.C[0] is not None:
                    pos_l1 = err > self.C[0]
                    # wght = C/abs(err)
                    sqrt_w[pos_l1] = np.sqrt(self.C[0]/err[pos_l1])

                if self.C[1] is not None:
                    neg_l1 = err < -self.C[1]
                    sqrt_w[neg_l1] = np.sqrt(self.C[1]/-err[neg_l1])

            if self.outlier_thresh is not None:

                if self.outlier_thresh[0] is not None:
                    # clip over-estimates
                    pos_outlier = err-mu_err > self.outlier_thresh[0]*scale
                    sqrt_w[pos_outlier] = 0
                    #print("{} overestimates".format(np.sum(clipIdx)))
                    #y_fit[pos_outlier] = y_est[pos_outlier] + self.outlier_thresh[0]*scale

                if self.outlier_thresh[1] is not None:
                    # clip under-estimates
                    neg_outlier = err-mu_err < -self.outlier_thresh[1]*scale
                    sqrt_w[neg_outlier] = 0
                    #print("{} underestimates".format(np.sum(clipIdx)))
                    #y_fit[neg_outlier] = y_est[neg_outlier] - self.outlier_thresh[1]*scale

            # solve weighted least squares
            try:
                ab, _, _, _ = np.linalg.lstsq(x*sqrt_w[:,np.newaxis], y*sqrt_w, rcond=-1)
            except:
                # LS didn't converge.... abort this update
                a = a0
                b = b0
                break
            # extract parameters
            a = ab[0]
            b = ab[1]
                    
        self.a = a
        self.b = b - self.a*mu_x + mu_y 

        # check for steps in the last part of the data - if enough data
        if err.shape[0]> 2*self.step_size:
            mu_err = np.median(err[:-self.step_size])
            std_err = np.std(err[:-self.step_size])
            mu = np.median(err[-self.step_size:])
            if mu > mu_err + self.step_threshold * std_err or mu < mu_err - self.step_threshold * std_err:
                #print('step detected!')
                self.reset(keep_model=True)

    def getX(self,y):
        return ( y  - self.b ) / self.a 

    def getY(self,x):
        return self.a * x + self.b

    @staticmethod
    def testcase(savefile='-'):
        if savefile is None:
            np.random.seed(0) # make reproducable
            X = np.arange(1000,dtype=np.float64) + np.random.randn(1)*1e6
            a = np.random.uniform(0,1)*10+10 
            b = 1000*np.random.randn(1) +.3
            Ytrue= a*X+b
            # add some steps
            for i in range(5):
                stepi = np.random.randint(Ytrue.shape[0])
                Ytrue[stepi:] += np.random.randint(3)*a
            N = np.random.standard_normal(Ytrue.shape)
            #Y    = Ytrue+ *10
            Y = Ytrue + np.exp((N-1)*3)
            print("{}) a={} b={}".format('true',a,b))

        else: # load from file
            import glob
            import os
            files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../logs/mindaffectBCI*.txt')) # * means all if need specific format then *.csv
            savefile = max(files, key=os.path.getctime)
            #savefile = "C:\\Users\\Developer\\Downloads\\mark\\mindaffectBCI_brainflow_200911_1339.txt" 
            #savefile = "C:/Users/Developer/Downloads/khash/mindaffectBCI_brainflow_ipad_200908_1938.txt"
            from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_messages
            from mindaffectBCI.utopiaclient import DataPacket
            print("Loading: {}".format(savefile))
            msgs = read_mindaffectBCI_messages(savefile,regress=None) # load without time-stamp fixing.
            dp = [ m for m in msgs if isinstance(m,DataPacket)]
            nsc = np.array([ (m.samples.shape[0],m.sts,m.timestamp) for m in dp])
            X = np.cumsum(nsc[:,0])
            Y = nsc[:,1]
            Ytrue = nsc[:,2]

        #ltt = lower_bound_tracker(window_size=100, outlier_thresh=(.5,None), step_size=10, step_threshold=2)#, C=(.1,None))
        # differential loss.  Linear above, quadratic below
        #ltt = lower_bound_tracker(window_size=200, C=(.1,None), step_size=10, step_threshold=2)#, C=(.1,None))
        # both, lower-bound + clipping
        ltt = lower_bound_tracker(window_size=100, outlier_thresh=(1,None), C=(.1,None), step_size=10, step_threshold=2)
        ltt.fit(X[0],Y[0]) # check scalar inputs
        step = 1
        idxs = list(range(1,X.shape[0],step))
        ab = np.zeros((len(idxs),2))
        dts = np.zeros((Y.shape[0],))
        dts[0] = ltt.getY(X[0])
        for i,j in enumerate(idxs):
            dts[j:j+step] = ltt.transform(X[j:j+step],Y[j:j+step])
            ab[i,:] = (ltt.a,ltt.b)
            yest = ltt.getY(X[j])
            err =  yest - Y[j]
            if abs(err)> 1000:
                print("{}) argh! yest={} ytrue={} err={}".format(i,yest,Ytrue[j],err))
            if i < 100:
                print("{:4d}) a={:5f} b={:5f}\ty_est-y={:2.5f}".format(j,ab[i,0],ab[i,1],
                        Y[j]-yest))

        import matplotlib.pyplot as plt
        ab,res,_,_ = np.linalg.lstsq(np.append(X[:,np.newaxis],np.ones((X.shape[0],1)),1),Y,rcond=-1)
        ots = X*ab[0]+ab[1]        
        idx=range(X.shape[0])
        plt.plot((Y[idx]-Y[0])- (X[idx]-X[0])*ab[0],'.-',label='server ts')
        plt.plot((dts[idx]-Y[0]) - (X[idx]-X[0])*ab[0],'.-',label='regressed ts (samp vs server)')
        plt.plot((ots[idx]-Y[0]) - (X[idx]-X[0])*ab[0],'.-',label='regressed ts (samp vs server) offline')

        err = (Y-Y[0]) - (X-X[0])*ab[0]
        cent = np.median(err); scale=np.median(np.abs(err-cent))
        plt.ylim((cent-scale*5,cent+scale*5))
        plt.grid()
        plt.legend()
        plt.show()

if __name__ == "__main__":
    lower_bound_tracker.testcase('-')
