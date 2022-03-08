import numpy as np
from mindaffectBCI.decoder.preprocess_transforms import *

class MoveAxis:
    def __init__(self,src=None,dst=None): self.src,self.dst=(src,dst)
    def fit(self,X,y=None): return self
    def transform(self,X,y=None):   return np.moveaxis(X,self.src,self.dst)
    fit_transform = transform

class ShapePrinter:
    def fit(self,X,y=None): print("X={} Y={}".format(X.shape,y.shape if y is not None else None)); return self
    def transform(self,X,y=None): return X
    def fit_transform(self,X,y=None): self.fit(X,y); return X

class CovMatrixizer:
    def __init__(self,axis=-1,accum_axis=-2): self.axis,self.accum_axis=(axis,accum_axis)
    def fit(self,X,y=None): return self
    def transform(self,X,y=None):
        if self.axis==-1 and self.accum_axis==-2:
            XX = np.einsum('...td,...te->...ed',X,X)
        elif self.axis==-2 and self.accum_axis==-1:
            XX = np.einsum('...dt,...et->...de',X,X)
        return XX
    fit_transform = transform

class AveragePower:
    def __init__(self,axis=-1,center=True, log=False): self.axis,self.center, self.log=(axis,center,log)
    def fit(self,X,y=None): return self
    def transform(self,X,y=None):
        if self.center: X=X-np.mean(X,axis=self.axis,keepdims=True)
        X = np.mean(X**2,axis=self.axis)
        if self.log : X = np.log(X)
        return X
    fit_transform = transform

class Mean:
    def __init__(self,axis=-1): self.axis, =(axis,)
    def fit(self,X,y=None): return self
    def transform(self,X,y=None):
        X = np.mean(X,axis=self.axis)
        return X
    fit_transform = transform

class AverageAmplitude:
    def __init__(self,axis=-1,center=True, log=False): self.axis, self.center, self.log=(axis,center,log)
    def fit(self,X,y=None): return self
    def transform(self,X,y=None):
        if self.center: X=X-np.mean(X,axis=self.axis,keepdims=True)
        X = np.mean(np.abs(X),axis=self.axis)
        if self.log : X = np.log(X)
        return X
    fit_transform = transform

class UnitAmp:
    def __init__(self, axis=1, reg:float=1e-8): self.axis, self.reg= (axis,reg)
    def fit(self,X,y=None):
        axis = self.axis if hasattr(self.axis,'__iter__') else (self.axis,)
        acc_axis = tuple(d for d in range(X.ndim) if not d in axis)
        self.amp_ = np.mean(np.abs(X), axis=acc_axis, keepdims=True) + self.reg
        return self
    def transform(self,X,y=None):  return X / self.amp_
    def fit_transform(self,X,y=None): return self.fit(X,y).transform(X)

class UnitPower:
    def __init__(self, axis=1, reg:float=1e-16): self.axis, self.reg= (axis,reg)
    def fit(self,X,y=None):
        axis = self.axis if hasattr(self.axis,'__iter__') else (self.axis,)
        acc_axis = tuple(d for d in range(X.ndim) if not d in axis)
        self.rms_ = np.sqrt(np.mean(X**2, axis=acc_axis, keepdims=True) + self.reg)
        return self
    def transform(self,X,y=None):  return X / self.rms_
    def fit_transform(self,X,y=None): return self.fit(X,y).transform(X)


class Power2Amp:
    def __init__(self, axis=1, reg:float=1e-8): self.axis, self.reg= (axis,reg)
    def fit(self,X,y=None):
        axis = self.axis if hasattr(self.axis,'__iter__') else (self.axis,)
        acc_axis = tuple(d for d in range(X.ndim) if not d in axis)
        self.amp_ = np.mean(np.abs(X), axis=acc_axis, keepdims=True) + self.reg
        return self
    def transform(self,X,y=None):  return X / np.sqrt(self.amp_)
    def fit_transform(self,X,y=None): return self.fit(X,y).transform(X)
