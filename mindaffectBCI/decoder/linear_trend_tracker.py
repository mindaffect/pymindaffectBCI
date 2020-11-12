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

class linear_trend_tracker():
    """ linear trend tracker with adaptive forgetting factor
    """   

    def __init__(self,halflife=70,int_err_halflife=5, K_int_err=10, a0=1, b0=0):
        self.alpha=np.exp(np.log(.5)/halflife) if halflife else .99
        self.warmup_weight= (1-self.alpha**20)/(1-self.alpha); # >20 points for warmup
        self.N=0
        if int_err_halflife is None and halflife:
            int_err_halflife = max(halflife/100,1) 
        self.int_err_alpha = np.exp(np.log(.5)/int_err_halflife) if int_err_halflife else .999
        self.K_int_err = K_int_err
        self.a0 = a0
        self.b0 = b0

    def reset(self, keep_err=False, keep_model=False):
        self.N = 0
        self.X0 = None
        self.Y0 = None
        self.sX = 0
        self.sY = 0
        self.sXX = 0
        self.sYX = 0
        self.sYY = 0
        if not keep_model:
            self.a=self.a0
            self.b=self.b0
        if not keep_err:
            self.int_err = 0
            self.abs_err=0
            self.int_err_N = 0

    def fit(self,X,Y,keep_err=False):
        self.reset(keep_err)
        if hasattr(X,'__iter__'):
            self.N = len(X)
            self.X0=X[0,...] 
            self.Y0=Y[0,...] 
            if len(X)>1 :
                self.transform(X[1:,...],Y[1:,...])
        else:
            self.N = 1
            self.X0 = X
            self.Y0 = Y
        self.a = self.a0
        self.b = self.Y0 - self.a * self.X0

    def transform(self,X,Y):
        if self.N==0:
            self.fit(X,Y)
            return Y

        #if np.all(X==self.Xlast) or np.all(Y==self.Ylast):
        #    return self.getY(X)
        # get our prediction for this point and use to track our prediction error
        Yest = self.getY(X)
        err  = np.mean(np.abs(Y-Yest))
  
        ## center x/y
        # N.B. be sure to do all the analysis in floating point to avoid overflow
        cX  = np.array(X - self.X0, dtype=float)
        cY  = np.array(Y - self.Y0, dtype=float)
        N = len(X) if hasattr(X,'__iter__') else 1
        # update the 1st and 2nd order summary statistics 
        wght    = self.alpha**N
        # adaptive learning rate as function of integerated error
        if self.N > self.warmup_weight:
            ptwght = max(1, abs(self.int_err/self.int_err_N)*self.K_int_err) #1/max(1,err) if self.N > self.warmup_weight else 1
        else:
            ptwght = 1
        # adaptive learning rate as a function of the direction of the error...
        #if err < 0:
        #    ptwght = ptwght*.1
        self.N  = wght*self.N   + ptwght*N
        self.sY = wght*self.sY  + ptwght*np.sum(cY,axis=0)
        self.sX = wght*self.sX  + ptwght*np.sum(cX,axis=0)
        self.sYY= wght*self.sYY + ptwght*np.sum(cY*cY,axis=0)
        self.sYX= wght*self.sYX + ptwght*np.sum(cY*cX,axis=0)
        self.sXX= wght*self.sXX + ptwght*np.sum(cX*cX,axis=0)

        # update the slope when warmed up
        if self.N > self.warmup_weight:
            Yvar  = self.sYY - self.sY * self.sY / self.N
            Xvar  = self.sXX - self.sX * self.sX / self.N
            YXvar = self.sYX - self.sY * self.sX / self.N
            self.a = (YXvar / Xvar + Yvar/YXvar )/2        
        # update the bias given the estimated slope b = mu_y - a * mu_x
        # being sure to include the shift to the origin!
        self.b = (self.Y0 + self.sY / self.N) - self.a * (self.X0 + self.sX / self.N)

        # check for steps in the inputs
        # get our prediction for this point and use to track our prediction error
        Yest = self.getY(X)
        err  = np.mean(Y-Yest)

        # track the prediction error, with long and short half-life
        self.abs_err = self.abs_err*wght + abs(err)
        # track with step window
        int_err_wght = self.int_err_alpha**N
        self.int_err_N = self.int_err_N * int_err_wght + N
        self.int_err = self.int_err*int_err_wght + err

        # only return the estimate if we've warmed up the tracker
        if self.N < self.warmup_weight:
            return Y

        # detect change in statistics by significant difference in error statistics
        # between long and short halflife
        #if (self.step_err / self.step_N) > (self.err / self.N) * self.step_threshold:
        #    print("step-detected")
        #    #self.fit(X,Y,keep_err=True)
        #    #Yest = Y

        return Yest

    def getX(self,y):
        return ( y  - self.b ) / self.a 

    def getY(self,x):
        return self.a * x + self.b

    @staticmethod
    def testcase():
        X = np.arange(1000) + np.random.randn(1)*1e6
        a = 1000/50 
        b = 1000*np.random.randn(1)
        Ytrue= a*X+b
        Y    = Ytrue+ np.random.standard_normal(Ytrue.shape)*10

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

        ltt = linear_trend_tracker(1000)
        ltt.fit(X[0],Y[0]) # check scalar inputs
        step = 1
        idxs = list(range(1,X.shape[0],step))
        ab = np.zeros((len(idxs),2))
        print("{}) a={} b={}".format('true',a,b))
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
        plt.plot(X[idx],Y[idx]- X[idx]*ab[0]-Y[0],label='server ts')
        plt.plot(X[idx],dts[idx] - X[idx]*ab[0]-Y[0],label='regressed ts (samp vs server)')
        plt.plot(X[idx],ots[idx] - X[idx]*ab[0]-Y[0],label='regressed ts (samp vs server) offline')

        err = Y - X*ab[0] - Y[0]
        cent = np.median(err); scale=np.median(np.abs(err-cent))
        plt.ylim((cent-scale*5,cent+scale*5))

        plt.legend()
        plt.show()

if __name__ == "__main__":
    linear_trend_tracker.testcase('-')
