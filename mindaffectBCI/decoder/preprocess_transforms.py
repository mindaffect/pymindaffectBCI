import mindaffectBCI.decoder.preprocess as pp
from mindaffectBCI.decoder.updateSummaryStatistics import updateCxx, plot_trial
from mindaffectBCI.decoder.multipleCCA import robust_whitener
from mindaffectBCI.decoder.preprocess import ola_welch
from scipy.signal import welch
from mindaffectBCI.decoder.preprocess import get_window_step, pool_axis
from mindaffectBCI.decoder.spherical_interpolation import make_spherical_spline_interpolation_matrix
from mindaffectBCI.decoder.utils import idOutliers
from mindaffectBCI.decoder.stim2event import stim2event, plot_stim_encoding
import matplotlib.pyplot as plt
from mindaffectBCI.decoder.model_fitting import *
from mindaffectBCI.decoder.utils import InfoArray, import_and_make_class
import numpy as np
from mindaffectBCI.decoder.utils import index_along_axis, window_axis
from mindaffectBCI.decoder.utils import sosfilt, butter_sosfilt, sosfilt_zi_warmup, butter_sosfiltfilt

try:
    from sklearn.base import ClassifierMixin, BaseEstimator
    from sklearn.base import TransformerMixin
except:
    # fake the class if sklearn is not available, e.g. Android/iOS
    class TransformerMixin:
        """base class for SKLEARN transformers, which only modify X and don't change number of examples.
        """

        def transform(self, X, y=None):
            pass

    class BaseEstimator:
        pass


class ModifierMixin:
    """base class for agumented transformers, which can modify both X and Y, 
        retain compatiability with sklearn signature by passing XY as a single 
        argument and returning the modified XY as a single output.
    """
    # def __init__(self):
    #    pass
    # def fit(self,XY):
    #    pass
    # def transform(self,XY,**kwargs):
    #     """wrapper method to retain sklearn compatiability

    #     Args:
    #         XY ([type]): tuple containing the X and Y as a single argument

    #     Returns:
    #         [type]: [description]
    #     """
    #     X, Y = XY
    #     X, Y = self.modify(X,Y,**kwargs)
    #     return (X,Y)

    def fit(self, X, y=None, **kwargs):
        return self

    # TODO[]: think of a better name than modify, e.g. dataset_transform
    def modify(self, X, Y, **kwargs):
        pass

    def fit_modify(self, X, Y, **kwargs):
        return self.fit(X, Y, **kwargs).modify(X, Y)


def make_test_signal(dur=3, fs=100, blksize=1):
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.utils import testSignal
    X_TSd, Y_TSy, st, A, B = testSignal(tau=10, noise2signal=1, nTrl=20, nSamp=300)
    fs = fs
    X_TSd = InfoArray(X_TSd,info={"fs":fs})
    return X_TSd, Y_TSy, fs


def plot_test_signal(X_TSd, Y_TSy, fs, title):
    plot_trial(X_TSd[:1, ...], Y_TSy[:1, ...], fs)
    plt.suptitle(title)
    plt.show(block=False)


def test_transform(mod, dur=3, fs=100, blksize=10):
    """[summary]

    Args:
        dur (int, optional): [description]. Defaults to 3.
        fs (int, optional): [description]. Defaults to 100.
        blksize (int, optional): [description]. Defaults to 10.
    """
    X_TSd, Y_TSy, fs = make_test_signal(dur=dur, fs=fs, blksize=blksize)
    # X = np.cumsum(X,-2) # 1/f spectrum
    plt.figure(1)
    plt.clf()
    plot_test_signal(X_TSd, Y_TSy, fs, 'raw')

    mod.fit(X_TSd, Y_TSy)
    wX = []
    wY = []
    for i in range(X_TSd.shape[0]):
        if hasattr(mod,'modify'):
            Xi, Yi = mod.modify(X_TSd[i:i+1, ...], Y_TSy[i:i+1, ...])
        else:
            Xi = mod.transform(X_TSd[i:i+1, ...], Y_TSy[i:i+1, ...])
            Yi = Y_TSy[i:i+1, ...]
        wX.append(Xi)
        wY.append(Yi)
    X_TSdd = np.concatenate(wX, 0)
    Y_TSyy = np.concatenate(wY, 0)

    # X = np.cumsum(X,-2) # 1/f spectrum
    plt.figure(2)
    plt.clf()
    plot_test_signal(X_TSdd, Y_TSyy, fs, str(mod))
    plt.show()

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class ShapePrinter(BaseEstimator):
    def fit(self,X, y=None):
        print("X={} Y={}".format(X.shape,y.shape if y else None))


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class Log(BaseEstimator):
    """log transform postive-parts of X """

    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        """log transform X
        """
        return np.log(np.maximum(X, 1e-3))

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class Power(BaseEstimator):
    """raise X to a power """

    def __init__(self, power: float = 2):
        self.power = power

    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        return X**self.power


class Square(BaseEstimator):
    """square X to a power """

    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        return X*X


class Abs(BaseEstimator):
    """square X to a power """

    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        return np.abs(X)


class ScalarFunctionApplier(BaseEstimator):
    """ apply elementwise operation to X """

    def __init__(self, operation=np.add, op=None, **kwargs):
        self.op, self.kwargs = (operation if op is None else op, kwargs)

    def fit(self, X, y=None):
        if isinstance(self.op, str):
            # convert to string to callable function
            fn = locals().get(self.op, None)
            if fn is None:
                fn = globals().get(self.op, None)
            if fn is None:
                fn = getattr(locals().get("np"), self.op, None) if 'np' in locals() else None
            if fn is None:
                fn = getattr(globals().get("np"), self.op, None) if 'np' in globals() else None
            self.op = fn
        return self

    def transform(self, X, y=None):
        return self.op(X, **self.kwargs)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class FeatureDimCompressor(BaseEstimator):
    """compress multiple feature dims into a single one"""

    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        if X.ndim > 2:
            X = X.reshape(X.shape[:2]+(-1,))
        return X


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class Passthrough(BaseEstimator):
    """a do nothing transformer """

    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        """pass the data trough un-changed
        """
        return X

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


class TrialPlotter(BaseEstimator):
    """plot the incomming trials """

    def __init__(
            self, fs: float = None, plot_trial_number: int = 0, plot_x: bool = True, plot_y: bool = True,
        plot_args: dict = dict(),
            plot_output_idx: int = None, fig: int = None, block: bool = False, suptitle: str = None, savefile:str=None):
        """Plot one or more trials of the input data

        Args:
            fs (float, optional): data sample rate. Defaults to None.
            plot_trial_number (int, optional): if not None then plot only this trial number. Defaults to True.
            plot_args (dict, optional): additional key-value args to pass to plot_trial
            fig (int, optional): figure to put the plot in. N.B. clf'd before plotting
            block (bool, optional): blocking or non-blocking show after plotting
            suptitle (str, optional): override the super-title on the plot with this string
        """
        self.fs, self.plot_trial_number, self.plot_args, self.fig, self.block, self.suptitle, self.plot_output_idx, self.plot_x, self.plot_y, self.savefile = (
            fs, plot_trial_number, plot_args, fig, block, suptitle, plot_output_idx, plot_x, plot_y, savefile)

    def fit(self, X, y):
        self.ntrial_ = 0

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        # get sample rate, priority order is: fit-argument, X.info, __init__-argument
        return self.transform(X, y, **kwargs)

    def transform(self, X, y=None, fs=None, ch_names=None, outputs=None, evtlabs=None):
        """plot X,y
        """
        import matplotlib.pyplot as plt
        if self.plot_trial_number is not None:
            if hasattr(self.plot_trial_number, '__iter__'):
                idx = [i for i in range(X.shape[0]) if self.ntrial_+i in self.plot_trial_number]
            else:
                idx = [i for i in range(X.shape[0]) if self.ntrial_+i == self.plot_trial_number]
        else:
            idx = slice(None)

        # only plot if should plot all, or this is the wanted trial to plot
        if idx:
            # extract meta-info to make a pretty plot:
            if outputs is None and hasattr(y, 'info'):
                outputs = y.info.get('outputs', None)
            if evtlabs is None and hasattr(y, 'info'):
                evtlabs = y.info.get('evtlabs', None)
            if ch_names is None and hasattr(X, 'info'):
                ch_names = X.info.get('ch_names', None)
            if fs is None and hasattr(X, 'info'):
                fs = X.info.get('fs', None)

            if self.fig:
                plt.figure(self.fig.number if hasattr(self.fig, 'number') else self.fig)
                plt.clf()
            else:
                self.fig = plt.figure()
                plt.clf()

            if y is not None:
                Y_TSye = y[idx, ...]
                if self.plot_output_idx is not None:
                    Y_TSye = Y_TSye[:, :, [self.plot_output_idx], ...]
            else:
                Y_TSye = y

            X_plt = X[idx, ...] if self.plot_x else None
            Y_plt = Y_TSye if self.plot_y else None
            plot_trial(X_plt, Y_plt, fs=fs, ch_names=ch_names, outputs=outputs, evtlabs=evtlabs, **self.plot_args)

            if self.savefile is not None:
                plt.savefig(self.savefile)
            if self.suptitle:
                plt.suptitle(self.suptitle)
            plt.show(block=self.block)
        # update the trial counter
        self.ntrial_ = self.ntrial_ + X.shape[0]
        return X


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

class TargetEncodingPlotter(BaseEstimator):
    """plot the incomming trials """

    def __init__(
            self, fs: float = None, plot_trial_number: int = 0,
        plot_args: dict = dict(),
            plot_output_idx: int = None, fig: int = None, block: bool = False, suptitle: str = None, savefile: str = None):
        """[summary]

        Args:
            fs (float, optional): data sample rate. Defaults to None.
            plot_trial_number (int, optional): if not None then plot only this trial number. Defaults to True.
            plot_args (dict, optional): additional key-value args to pass to plot_trial
            fig (int, optional): figure to put the plot in. N.B. clf'd before plotting
            block (bool, optional): blocking or non-blocking show after plotting
            suptitle (str, optional): override the super-title on the plot with this string
            savefile (str, optional): if given then save copy of the plot to this file
        """
        self.fs, self.plot_trial_number, self.plot_args, self.fig, self.block, self.suptitle, self.plot_output_idx, self.savefile = (
            fs, plot_trial_number, plot_args, fig, block, suptitle, plot_output_idx, savefile)

    def fit(self, X, y):
        self.ntrial_ = 0

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        # get sample rate, priority order is: fit-argument, X.info, __init__-argument
        return self.transform(X, y, **kwargs)

    def transform(self, X, y=None, fs=None, ch_names=None, outputs=None, evtlabs=None):
        """plot X,y
        """
        import matplotlib.pyplot as plt
        if self.plot_trial_number is not None:
            if hasattr(self.plot_trial_number, '__iter__'):
                idx = [i for i in range(X.shape[0]) if self.ntrial_+i in self.plot_trial_number]
            else:
                idx = [i for i in range(X.shape[0]) if self.ntrial_+i == self.plot_trial_number]
        else:
            idx = slice(None)

        # only plot if should plot all, or this is the wanted trial to plot
        if idx:
            # extract meta-info to make a pretty plot:
            if outputs is None and hasattr(y, 'info'):
                outputs = y.info.get('outputs', None)
            if evtlabs is None and hasattr(y, 'info'):
                evtlabs = y.info.get('evtlabs', None)
            if fs is None and hasattr(X, 'info'):
                fs = X.info.get('fs', None)

            if self.fig:
                plt.figure(self.fig.number if hasattr(self.fig, 'number') else self.fig)
                plt.clf()
            else:
                self.fig = plt.figure()

            if y is not None:
                Y_TSye = y[idx, ...]
                if self.plot_output_idx is not None:
                    Y_TSye = Y_TSye[:, :, [self.plot_output_idx], ...]
            else:
                Y_TSye = y

            plot_stim_encoding(Y_TSye,None,fs=fs,outputs=outputs,evtlabs=evtlabs,suptitle=self.suptitle)

            if self.savefile is not None:
                plt.savefig(self.savefile)

            plt.show(block=self.block)
        # update the trial counter
        self.ntrial_ = self.ntrial_ + X.shape[0]
        return X


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def plot_class_average(
        X_TSd, Y_TSye=None, y0_is_true: bool = True, fs=None, outputs=None, ch_names=None, evtlabs=None, tau: int = None,
        tau_ms: float = 500, offset: int = None, offset_ms: float = None, fig=None, suptitle=None, **kwargs):
    """plot class average responses, i.e. Event Related Potentials

    Args:
        X_TSd (InfoArray|ndarray): The data.  Shape (T,S,d)=(#Trials, #samples per trial, #sensors)
        Y_TSye (InfoArray|ndarray, optional): The stimulus properties. Shape (T,S,y,e)=(#Trails, #Samples per trial, #outputs, #stimulus feature types). Defaults to None.
        y0_is_true (bool, optional): flag we should treat the first output, y, at the true label sequence and ignore the rest. Defaults to True.
        fs (_type_, optional): _description_. Defaults to None.
        outputs (_type_, optional): Labels for the y outputs (in Y_TSye). Defaults to None.
        ch_names (_type_, optional): Labels for the d channels (in X_TSd). Defaults to None.
        evtlabs (_type_, optional): Labels for the e event types (in Y_TSye). Defaults to None.
        tau (int, optional): Length of the computed ERP in samples. Defaults to None.
        tau_ms (float, optional): Length of the computed ERP in milliseconds. Defaults to 500.
        offset (int, optional): Offset for computing the class averge w.r.t. the event in Y in samples. Defaults to None.
        offset_ms (float, optional): Offset for computing the class average w.r.t. the event in Y in milliseconds. Defaults to None.
        fig (_type_, optional): Figure to draw into. Defaults to None.
        suptitle (str, optional): title for the drawing figure. Defaults to None.

    Returns:
        _type_: _description_
    """
    # get sample rate, priority order is: fit-argument, X.info, __init__-argument
    if hasattr(X_TSd, 'info'):
        fs = X_TSd.info['fs']
    if tau is None:
        tau = max(1, int(tau_ms * fs / 1000))
    if offset is None:
        if not offset_ms is None:
            offset = int(offset_ms * fs / 1000)
        else:
            offset = 0

    # extract the meta-info
    if outputs is None and hasattr(Y_TSye, 'info'):
        outputs = Y_TSye.info.get('outputs', None)
    if ch_names is None and hasattr(X_TSd, 'info'):
        ch_names = X_TSd.info.get('ch_names', None)
    if evtlabs is None and hasattr(Y_TSye, 'info'):
        evtlabs = Y_TSye.info.get('evtlabs', None)

    times = np.arange(offset, tau + offset) / fs

    # only for the 0th output
    if y0_is_true and not np.all(Y_TSye[:, :, 0, ...] == 0):
        Y_TSye = Y_TSye[:, :, 0:1, ...]
        if outputs is not None:
            outputs = outputs[0]
    if Y_TSye.ndim < 4:  # add a feature dim
        Y_TSye = Y_TSye[..., np.newaxis]
    elif Y_TSye.ndim > 4:  # compress multiple feature dims
        Y_TSye = Y_TSye.reshape(Y_TSye.shape[:3]+(-1,))

    Cxy_yetd = updateCxy(None, X_TSd, Y_TSye, tau=tau, offset=offset)
    N = np.sum(Y_TSye, axis=(0, 1, 2))
    Cxy_yetd = Cxy_yetd / N[:, np.newaxis, np.newaxis]  # normalize

    import matplotlib.pyplot as plt
    if fig is not None:
        plt.figure(fig.number if hasattr(fig, 'number') else fig)
        plt.clf()
    else:
        fig = plt.figure()
        plt.clf()

    # add back in feature dims
    Cxy_yetd = Cxy_yetd.reshape(Cxy_yetd.shape[:3]+X_TSd.shape[2:])

    plot_erp(Cxy_yetd, fs=fs, ch_names=ch_names, times=times,
             evtlabs=evtlabs, outputs=outputs, suptitle=suptitle, **kwargs)
    return fig


def plot_f_classif(X_TSd, Y_TSye=None, y0_is_true: bool = True, fs=None, outputs=None, ch_names=None, evtlabs=None,
                   tau=None, tau_ms=None, offset=None, offset_ms=None, fig=None, suptitle=None, **kwargs):
    """plot the per-feature classificability score

    Args:
        X_TSd (InfoArray|ndarray): The data.  Shape (T,S,d)=(#Trials, #samples per trial, #sensors)
        Y_TSye (InfoArray|ndarray, optional): The stimulus properties. Shape (T,S,y,e)=(#Trails, #Samples per trial, #outputs, #stimulus feature types). Defaults to None.
        y0_is_true (bool, optional): flag we should treat the first output, y, at the true label sequence and ignore the rest. Defaults to True.
        fs (float, optional): sample rate of the data. Defaults to None.
        outputs (_type_, optional): Labels for the y outputs (in Y_TSye). Defaults to None.
        ch_names (_type_, optional): Labels for the d channels (in X_TSd). Defaults to None.
        evtlabs (_type_, optional): Labels for the e event types (in Y_TSye). Defaults to None.
        tau (int, optional): Length of the computed ERP in samples. Defaults to None.
        tau_ms (float, optional): Length of the computed ERP in milliseconds. Defaults to 500.
        offset (int, optional): Offset for computing the class averge w.r.t. the event in Y in samples. Defaults to None.
        offset_ms (float, optional): Offset for computing the class average w.r.t. the event in Y in milliseconds. Defaults to None.
        fig (_type_, optional): Figure to draw into. Defaults to None.
        suptitle (str, optional): title for the drawing figure. Defaults to None.

    Returns:
        _type_: _description_
    """
    from sklearn.feature_selection import f_classif
    # get sample rate, priority order is: fit-argument, X.info, __init__-argument
    if hasattr(X_TSd, 'info'):
        fs = X_TSd.info['fs']

    # TÔDO[]: support tau+offset + full sliding window scoring
    assert tau is None and offset is None and offset_ms is None
    if tau is None:
        tau = X_TSd.shape[1]  # max(1,int(tau_ms * fs / 1000))
    if offset is None:
        if not offset_ms is None:
            offset = int(offset_ms * fs / 1000)
        else:
            offset = 0

    # extract the meta-info
    if outputs is None and hasattr(Y_TSye, 'info'):
        outputs = Y_TSye.info.get('outputs', None)
    if ch_names is None and hasattr(X_TSd, 'info'):
        ch_names = X_TSd.info.get('ch_names', None)
    if evtlabs is None and hasattr(Y_TSye, 'info'):
        evtlabs = Y_TSye.info.get('evtlabs', None)

    times = np.arange(offset, tau + offset) / fs

    # only for the 0th output
    if y0_is_true:
        Y_TSye = Y_TSye[:, :, 0:1, ...]
        if outputs is not None:
            outputs = outputs[0]
    if Y_TSye.ndim < 4:  # add a feature dim
        Y_TSye = Y_TSye[..., np.newaxis]
    elif Y_TSye.ndim > 4:  # compress multiple feature dims
        Y_TSye = Y_TSye.reshape(Y_TSye.shape[:3]+(-1,))

    # get label number for each trial
    # TODO[]: slice correctly to get per-window dataset
    lab_T = np.argmax(np.max(Y_TSye[:, :, 0, :], 1), -1)+1  # label is max event index
    # compute score
    score, p_values = f_classif(X_TSd.reshape((X_TSd.shape[0], -1)), lab_T)
    score_td = score.reshape(X_TSd.shape[1:])
    p_values_td = p_values.reshape(X_TSd.shape[1:])

    import matplotlib.pyplot as plt
    if fig is not None:
        plt.figure(fig.number if hasattr(fig, 'number') else fig)
        plt.clf()
    else:
        fig = plt.figure()

    plot_erp(p_values_td, fs=fs, ch_names=ch_names, times=times, suptitle=suptitle, **kwargs)
    return fig


class EventRelatedPotentialPlotter(BaseEstimator):
    """plot the incomming trials """

    def __init__(
            self, fs: float = None, tau: int = None, tau_ms: float = 500, offset: int = None, offset_ms: float = 0,
            plot_trial_number: int = 0, plot_args: dict = dict(),
            fig: int = None, block: bool = None, suptitle: str = None, cls_diff: bool = False, savefile:str=None):
        """plot a class average of the incomming data

        Args:
            tau (int, optional): lenght of the response window in samples. Defaults to None.
            tau_ms (float, optional): length of the response window in milliseconds. Defaults to None.
            offset (int, optional): offset from event trigger time in samples. Defaults to None.
            offset_ms (float, optional): offset from the event trigger time in milliseconds. Defaults to 0.
            fs (float, optional): _description_. Defaults to None.
            plot_args (dict, optional): additional key-value args to pass to plot_trial
            fig (int, optional): figure to put the plot in. N.B. clf'd before plotting
            block (bool, optional): if true show and block, false show and don't block, None don't show. Defaults to None.
            suptitle (str, optional): override the super-title on the plot with this string
            savefile (str, optional): filename to save a .png copy of the plot
        """
        self.fs, self.tau, self.tau_ms, self.offset, self.offset_ms, self.plot_trial_number, self.plot_args, self.fig, self.block, self.suptitle, self.cls_diff, self.savefile = (
            fs, tau, tau_ms, offset, offset_ms, plot_trial_number, plot_args, fig, block, suptitle, cls_diff, savefile)

    def fit(self, X, y=None, y0_is_true: bool = True, fs=None, outputs=None, ch_names=None, evtlabs=None):
        # get sample rate, priority order is: fit-argument, X.info, __init__-argument
        if fs is None:
            fs = self.fs

        self.ntrial_ = self.ntrial_+1 if hasattr(self, 'ntrial_') else 0
        if self.plot_trial_number is not None and not self.ntrial_ == self.plot_trial_number:
            return self

        self.fig = plot_class_average(X, y, y0_is_true, fs=fs, outputs=outputs, ch_names=ch_names, evtlabs=evtlabs,
                                      tau=self.tau, tau_ms=self.tau_ms, offset=self.offset, offset_ms=self.offset_ms, suptitle=self.suptitle)

        if self.savefile is not None:
            plt.savefig(self.savefile)
        if self.block is not None:
            plt.show(block=self.block)
        return self

    def transform(self, X, y):
        return X

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return X


ClassAveragePlotter = EventRelatedPotentialPlotter


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class SummaryStatisticsPlotter(BaseEstimator):
    """plot the incomming trials """

    def __init__(
            self, fs: float = None, tau: int = None, tau_ms: float = 500, offset: int = None, offset_ms: float = 0,
            plot_args: dict = dict(),
            fig: int = None, block: bool = None, suptitle: str = None):
        """ Plot the summary statistics of the incomming data, where summary statistics are: Cxx=per-channel covariance,  Cyx=ERP=class-average response,  Cyy=per-time-point stimulus cross-auto-covariance

        Args:
            tau (int, optional): lenght of the response window in samples. Defaults to None.
            tau_ms (float, optional): length of the response window in milliseconds. Defaults to None.
            offset (int, optional): offset from event trigger time in samples. Defaults to None.
            offset_ms (float, optional): offset from the event trigger time in milliseconds. Defaults to 0.
            fs (float, optional): _description_. Defaults to None.
            plot_trial_number (int, optional): if not None then plot only this trial number. Defaults to True.
            plot_args (dict, optional): additional key-value args to pass to plot_trial
            fig (int, optional): figure to put the plot in. N.B. clf'd before plotting
            block (bool, optional): if true show and block, false show and don't block, None don't show. Defaults to None.
            suptitle (str, optional): override the super-title on the plot with this string
        """
        self.fs, self.tau, self.tau_ms, self.offset, self.offset_ms, self.plot_args, self.fig, self.block, self.suptitle = (
            fs, tau, tau_ms, offset, offset_ms, plot_args, fig, block, suptitle)

    def fit(self, X, y=None, y0_is_true: bool = True, fs=None, outputs=None, ch_names=None, evtlabs=None):
        # get sample rate, priority order is: fit-argument, X.info, __init__-argument
        if fs is not None:
            self.fs_ = fs
        elif hasattr(X, 'info'):
            self.fs_ = X.info['fs']
        else:
            self.fs_ = self.fs
        self.tau_ = self.tau
        if self.tau_ is None:
            self.tau_ = max(1, int(self.tau_ms * self.fs_ / 1000))
        self.offset_ = self.offset
        if self.offset_ is None:
            if not self.offset_ms is None:
                self.offset_ = int(self.offset_ms * self.fs_ / 1000)
            else:
                self.offset_ = 0

        import matplotlib.pyplot as plt
        if self.fig:
            plt.figure(self.fig.number if hasattr(self.fig, 'number') else self.fig)
            plt.clf()
        else:
            self.fig = plt.figure()

        # extract the meta-info
        if outputs is None and hasattr(y, 'info'):
            outputs = y.info.get('outputs', None)
        if ch_names is None and hasattr(X, 'info'):
            ch_names = X.info.get('ch_names', None)
        if evtlabs is None and hasattr(y, 'info'):
            evtlabs = y.info.get('evtlabs', None)

        times = np.arange(self.offset_, self.tau_ + self.offset_) / self.fs_

        # only for the 0th output
        if y0_is_true:
            y = y[:, :, 0:1, ...]
            if outputs is not None:
                outputs = outputs[0]
        if y.ndim < 4:  # add a feature dim
            y = y[..., np.newaxis]
        elif y.ndim >= 4:  # compress multiple feature dims
            y = y.reshape(y.shape[:3]+(-1,))

        Cxx, Cxy, Cyy = updateSummaryStatistics(X.reshape(X.shape[:2]+(-1,)), y, tau=self.tau_, offset=self.offset_)
        plot_summary_statistics(Cxx, Cxy, Cyy, evtlabs=evtlabs, times=times, ch_names=ch_names)
        if self.block is not None:
            plt.show(block=self.block)
        return self

    def transform(self, X, y):
        return X


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class TwoDTransformerWrapper(BaseEstimator):
    """wrap a normal sklearn transformer which expects 2-dimensional input (trial,features) to work with multi-dimensional X
    """

    def __init__(self, transformer, **kwargs):
        """wrap a normal sklearn transformer which expects 2-dimensional input (trial,features) to work with multi-dimensional X

        Args:
            transformer (BaseTransformer): conventional sklearn transformer which expects 2d inputs we want to wrap

        Examples:
            TwoDTransformerWrapper(transformer=StandardScaler()).fit_transform(X,y)
        """
        if isinstance(transformer, str):
            try:
                transformer = globals()[transformer](**kwargs)
            except:
                transformer = import_and_make_class(transformer, **kwargs)
        self.transformer = transformer

    def make2d(self, X, y=None):
        X2d = X
        if X2d.ndim > 2:
            X2d = X2d.reshape((X2d.shape[0], -1))
        y2d = y
        if y2d is not None and y2d.ndim > 2:
            y2d = y2d.reshape((y2d.shape[0], -1))
        return X2d, y2d

    def unmake2d(self, X2d, X):
        if X.ndim > 2:
            X2d = X2d.reshape((X2d.shape[0],)+X.shape[1:])
        if hasattr(X, 'info'):  # re-attache meta-info
            X2d = InfoArray(X2d, info=X.info)
        return X2d

    def fit_transform(self, X, y=None, **fit_params):
        X2d, y2d = self.make2d(X, y)
        X = self.transformer.fit_transform(X2d, y2d, **fit_params)
        X = self.unmake2d(X2d, X)
        return X

    def fit(self, X, y=None, **fit_params):
        X2d, y2d = self.make2d(X, y)
        X2d = self.transformer.fit(X2d, y2d, **fit_params)
        return self

    def transform(self, X, y=None):
        X2d, _ = self.make2d(X)
        X2d = self.transformer.transform(X2d, y=Y)
        X = self.unmake2d(X2d, X)
        return X


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class TimeShifter(BaseEstimator, ModifierMixin):
    """time shift X relative to Y or vice-versa
    """

    def __init__(
            self, axis=1, shift_who: str = 'y', timeshift: int = None, timeshift_ms: float = None, fs: float = None,
            padding: str = 0, verb=0):
        """time shift X (eeg) relative to Y (stimulus) to simply move the analysis window spec for the impulse response function

        Args:
            axis (int, optional): the axis of X which corrosponds to 'time'. Defaults to 1.
            shift_who (str, optional): shift X or Y?. Defaults to 'y'.
            timeshift (int, optional): shift size in samples. Defaults to None.
            timeshift_ms (float, optional): shift size in milliseconds. Defaults to None.
            fs (float, optional): sampling rate of the data, got from X's meta-info if not given. Defaults to None.
            padding (str, optional): Value/method to use to padd X/Y for the missing values after the shift. Defaults to 0.
            verb (int, optional): verbosity level for debugging. Defaults to 0.
        """            
        self.axis, self.shift_who, self.timeshift, self.timeshift_ms, self.fs, self.padding, self.verb = \
            (axis, shift_who, timeshift, timeshift_ms, fs, padding, verb)

    def fit(self, X, y=None, fs: float = None):
        # setup the window, and step
        if self.timeshift is not None:  # compute shift in samples
            self.timeshift_ = self.timeshift  # store the result
        else:
            if fs is None: # extract sample rate
                fs = X.info['fs'] if hasattr(X, 'info') else self.fs
            self.timeshift_ = int(self.timeshift_ms * fs / 1000)
        assert self.axis == 1

        if self.verb > 0:
            print("TimeShifter: {} samples".format(self.timeshift_))

        return self

    def timeshift_inplace(self, M, axis, shift, padding):
        if not padding == 0:
            raise ValueError('Unsupported padding type! {}'.format(padding))

        # N.B. shift-in-place
        if shift > 0:  # 0->shift, so padd before
            M[index_along_axis(slice(shift, None), axis)] = M[index_along_axis(slice(None, -shift), axis)]
            M[index_along_axis(slice(0, shift), axis)] = padding
        elif shift < 0:  # -shift->0, so padd after
            M[index_along_axis(slice(None, shift), axis)] = M[index_along_axis(slice(-shift, None), axis)]
            M[index_along_axis(slice(shift, None), axis)] = padding

        return M

    def modify(self, X, y=None):
        """ convert from features to feature covariances in temporal blocks
        """
        if not hasattr(self, 'timeshift_'):
            raise ValueError("Must fit before transform!")

        # ensure 3-d input
        X_TSd = X
        Y_TSy = y
        if X_TSd.ndim < 3:
            X_TSd = X_TSd.reshape((-1,)*(3-X_TSd.ndim) + X_TSd.shape)
            if Y_TSy is not None:
                Y_TSy = Y_TSy.reshape((-1,)*(3-X_TSd.ndim) + Y_TSy.shape)

        if self.shift_who.lower() == 'X':
            X_TSd = self.timeshift_inplace(X_TSd, self.axis, self.timeshift_, self.padding)
        else:
            Y_TSy = self.timeshift_inplace(Y_TSy, self.axis, self.timeshift_, self.padding)

        return X_TSd, Y_TSy

    def testcase(self):
        test_transform(TimeShifter(timeshift=2))
        test_transform(TimeShifter(timeshift=-2))



# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


class TargetEncoder(BaseEstimator, ModifierMixin):
    """(re)encode the target information 
    """

    def __init__(self, evtlabs=('re', 'fe'), axis=1, squeeze_feature_dim: bool = True, verb=0, **kwargs):
        """(re)encode the stimulus sequence information

        Args:
            evtlabs (tuple|callable, optional): the evtlabs list to pass to `stim2event`.   
               OR
                  If evtlabs is callable with call signature signature `y_new, recode_state, output_labels = evtlabs(y,axis=axis,**kwargs)`. Defaults to ('re', 'fe').
            axis (int, optional): the 'time' axis of Y for the re-coding. Defaults to 1.
            squeeze_feature_dim (bool, optional): if true and re-coding of the targets results in a new feature dim with one entry then remove this extra dimension so the stim-seq size remains constant. Defaults to True.
            verb (int, optional): verbosity level for progress printing. Defaults to 0.
        """        
        self.evtlabs, self.axis, self.verb, self.squeeze_feature_dim, self.kwargs = \
            (evtlabs, axis, verb, squeeze_feature_dim, kwargs)

    def fit_modify(self, X, y=None, prevY=None):
        yinfo = y.info if hasattr(y, 'info') else None  # preserve the metainfo
        if self.verb > 0:
            print("Y_in={}".format(y.shape))
        if self.evtlabs is not None:
            if callable(self.evtlabs): # call the re-coder directly
                y, self.s2estate_, self.evtlabs_ = self.evtlabs(y,axis=self.axis,**self.kwargs)
            else: # use stim2event
                y, self.s2estate_, self.evtlabs_ = stim2event(
                    y, evtypes=self.evtlabs, axis=self.axis, oM=prevY, **self.kwargs)  # (tr, samp, Y, e)
        else:
            self.e2state_, self.evtlabs_ = (self.evtlabs, None)
        if self.verb > 0:
            print("Y_out={}".format(y.shape))

        # attach meta-info
        y = self.add_evtlabs_metainfo(y, yinfo, self.evtlabs_)

        return X, y

    def add_evtlabs_metainfo(self, y, yinfo:dict, evtlabs):
        """add meta-info about the event labels to propogate to other later functions

        Args:
            y (_type_): the stimulus sequence
            yinfo (dict): the current meta-information about y, specifically including the `evtlabs` field with the human-readable names for the different stimulus events
            evtlabs (tuple-of-str): the new stimulus event human readable names  

        Returns:
            InfoArray: Y with the updated meta-info
        """
        if not hasattr(y, 'info'):
            y = InfoArray(y, yinfo)
        if y.info is None:
            y.info = {'evtlabs': evtlabs}
        else:
            # combine the 2 sets of evtlabs
            el = y.info.get('evtlabs', None)
            # TODO[]: make it a list of lists? instead of flattening it early?
            el2 = evtlabs
            if el is not None:
                if isinstance(el,str): el=[el]
                if isinstance(evtlabs,str): evtlabs=[evtlabs]
                el2 = ["{}_{}".format(e1, e2) for e1 in el for e2 in evtlabs]
                #print('new-lab: {}'.format(el2))
            y.info.update({'evtlabs': el2})
        return y

    def modify(self, X, y=None, prevY=None):
        """ update the label mapping
        """
        if not hasattr(self, 'evtlabs_'):
            return self.fit_modify(X, y)

        # ensure 3-d input
        X_TSd = X
        Y_TSy = y
        if X_TSd.ndim < 3:
            if Y_TSy is not None:
                Y_TSy = Y_TSy.reshape((-1,)*(3-X_TSd.ndim) + Y_TSy.shape)

        yinfo = Y_TSy.info if hasattr(Y_TSy, 'info') else None  # preserve the metainfo
        if self.s2estate_ is not None:
            if callable(self.evtlabs):
                Y_TSy, _, _ = self.evtlabs(Y_TSy, axis=self.axis, **self.kwargs)
            else:
                Y_TSy, _, _ = stim2event(Y_TSy, evtypes=self.s2estate_, axis=self.axis, oM=prevY)
        else:
            if Y_TSy.ndim == 3:  # add event dim
                Y_TSy = Y_TSy[:, :, :, np.newaxis]  # (tr,samp,Y,e)

        if self.squeeze_feature_dim and Y_TSy.ndim > 4 and Y_TSy.shape[-1] == 1:
            Y_TSy = Y_TSy[..., 0]

        # update/attach the meta-info
        Y_TSy = self.add_evtlabs_metainfo(Y_TSy, yinfo, self.evtlabs_)

        return X_TSd, Y_TSy

    def testcase(self):
        test_transform(TargetEncoder(evtlabs=('re', 'fe')))


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class TemporalDecorrelator(BaseEstimator, TransformerMixin):
    """Incremental streaming tranformer to decorrelate temporally channels in an input stream
    """

    def __init__(self, order=10, reg=1e-4, eta=1e-5, axis=-2):
        """Incremental streaming tranformer to decorrelate temporally channels in an input stream

        Args:
            order (int, optional): order of the AR model fit for decorrelation. Defaults to 10.
            reg (_type_, optional): regularization factor. Defaults to 1e-4.
            eta (_type_, optional): minimum value threshold. Defaults to 1e-5.
            axis (int, optional): axis of X which corrosponds to time. Defaults to -2.
        """        
        self.reg, self.eta, self.axis, self.order = (reg, eta, axis, order)

    def fit(self, X, y):
        self.fit_transform(X.copy(), y)
        return self

    def fit_transform(self, X, y=None):
        """[summary]

        Args:
            X ([type]): [description]
        """
        self.W_ = np.zeros((self.order, X.shape[-1]), dtype=X.dtype)
        self.W_[-1, :] = 1
        X = self.transform(X)
        return X

    def transform(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate
            nsamp(int): number of samples to interpolate

        Returns:
            np.ndarray: the decorrelated data
        """
        if not hasattr(self, 'W_'):
            self.fit(X)

        X, self.W_ = pp.temporally_decorrelate(X, W=self.W_, reg=self.reg, eta=self.eta, axis=self.axis)

        return X

    def testcase(self):
        mod = TemporalDecorrelator()
        test_transform(mod)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class StandardScaler(BaseEstimator):
    """ standardize X to have zero-mean and unit-standard-deviation in the feature dimensions
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, reg: float = 1e-4, axis: int = (0, 1)):
        """standardize X to have zero-mean and unit-standard-deviation in the feature dimensions

        Args:
            with_mean (bool, optional): flag if we zero-mean the data. Defaults to True.
            with_std (bool, optional): flag if we unit standard deviation the data. Defaults to True.
            reg (float, optional): regularization strength for computing the standard deviation and avoiding div-by-zero. Defaults to 1e-4.
            axis (int, optional): axes of X over which we compute the statistics.  Defaults to 0,1 which normally means compute a per-channel mean/std over all trials and samples. Defaults to (0, 1).
        """        
        self.reg, self.axis, self.with_mean, self.with_std = \
            (reg, axis, with_mean, with_std)

    def fit(self, X, y=None):
        try:
            N = np.prod(X.shape[a] for a in self.axis)
        except:
            N = X.shape[self.axis]
        mean = np.sum(X, self.axis, keepdims=True) / N
        self.mean_ = mean if self.with_mean else None
        if self.with_std:
            cX = X - mean if self.with_mean else X
            self.std_ = np.sqrt(np.sum(cX*cX+self.reg, axis=self.axis, keepdims=True) / N)
        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'mean_'):
            self.fit(X)

        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.std_
        return X


class TargetStandardScaler(BaseEstimator):
    def __init__(self, with_mean: bool = True, with_std: bool = True, reg: float = 1e-4, axis: int = 1):
        """apply standard scaling (zero-mean, unit-stddev) to the target values

        Args:
            with_mean (bool, optional): [description]. Defaults to True.
            with_std (bool, optional): [description]. Defaults to True.
            reg (float, optional): [description]. Defaults to 1e-4.
            axis (int, optional): [description]. Defaults to 1.
        """
        self.reg, self.axis, self.with_mean, self.with_std = \
            (reg, axis, with_mean, with_std)

    def fit(self, X, y):
        self.standardscalar_ = StandardScaler(self.with_mean, self.with_std, self.reg, self.axis)
        self.standardscalar_.fit(y)

    def modify(self, X, Y):
        Y = self.standardscalar_.transform(Y)
        return (X, Y)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class ChannelPowerStandardizer(BaseEstimator, TransformerMixin):
    """Channel power normalization in an input stream"""

    def __init__(self, reg=1e-4, axis=-2):
        """Channel power normalization in an input stream

        Args:
            reg (float, optional): regularization constant for compute the per-channel standard deviation. Defaults to 1e-4.
            axis (int, optional): the 'time' axis of the data. Defaults to -2.
        """        
        self.reg, self.axis = reg, axis

    def fit(self, X, y=None):
        self.fit_transform(X.copy(), y)
        return self

    def fit_transform(self, X, y=None):
        """[summary]

        Args:
            X ([type]): [description]
        """
        self._sigma2 = np.mean(X*X, axis=tuple(range(X.ndim-1)), keepdims=True)
        X = self.transform(X, y)
        return X

    def transform(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate

        Returns:
            np.ndarray: the decorrelated data
        """
        if not hasattr(self, '_sigma2'):
            self.fit(X)

        X = X / np.sqrt((self._sigma2 + self.reg*np.median(self._sigma2))/2)

        return X

    def testcase(self, dur=3, fs=100, blksize=10):
        """[summary]

        Args:
            dur (int, optional): [description]. Defaults to 3.
            fs (int, optional): [description]. Defaults to 100.
            blksize (int, optional): [description]. Defaults to 10.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mindaffectBCI.decoder.utils import testSignal
        X_TSd, Y_TSy, st, A, B = testSignal(tau=10, noise2signal=1, nTrl=20, nSamp=300)
        fs = 100

        X_TSd[:, :, 1] = X_TSd[:, :, 1]*10  # much more power in 1 channel
        X_TSd[1, :, :] = X_TSd[1, :, :]*2  # and in 1 trial

        # X = np.cumsum(X,-2) # 1/f spectrum
        print("X={}".format(X_TSd.shape))
        plt.figure(1)
        plot_trial(X_TSd[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('Raw')
        plt.show(block=False)

        trans = ChannelPowerStandardizer()
        wX = []
        for i in range(X_TSd.shape[0]):
            wX.append(trans.transform(X_TSd[i:i+1, ...]))
        wX = np.concatenate(wX, 0)

        plt.figure(2)
        plot_trial(wX[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('Power Standardized')
        plt.show()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class AdaptiveChannelPowerStandardizer(BaseEstimator, TransformerMixin):
    """Incremental streaming tranformer to channel power normalization in an input stream
    """

    def __init__(self, reg=1e-4, axis=-2):
        """Incremental streaming tranformer to channel power normalization in an input stream

        N.B. incremental means it processes each trial in sequential order and using a moving average to compute the normalization statistics

        Args:
            reg (float, optional): regularization constant for compute the per-channel standard deviation. Defaults to 1e-4.
            axis (int, optional): the 'time' axis of the data. Defaults to -2.
        """        
        self.reg, self.axis = reg, axis

    def fit(self, X, y=None):
        self.fit_transform(X.copy(), y)
        return self

    def fit_transform(self, X, y=None):
        """[summary]

        Args:
            X ([type]): [description]
        """
        # ensure 3-d input
        X_TSd = X.reshape((-1,)*(3-X.ndim) + X.shape)
        self.sigma2_ = np.zeros((X.shape[-1],), dtype=X.dtype)
        self.sigma2_ = X[0, 0, :]*X[0, 0, :]  # warmup with 1st sample power
        X = self.transform(X)
        return X

    def transform(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate

        Returns:
            np.ndarray: the decorrelated data
        """
        if not hasattr(self, 'sigma2_'):
            self.fit(X)

        X, self.W_ = pp.standardize_channel_power(X, sigma2=self.sigma2_, reg=self.reg, axis=self.axis)

        return X

    def testcase(self, dur=3, fs=100, blksize=10):
        """[summary]

        Args:
            dur (int, optional): [description]. Defaults to 3.
            fs (int, optional): [description]. Defaults to 100.
            blksize (int, optional): [description]. Defaults to 10.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mindaffectBCI.decoder.utils import testSignal
        X_TSd, Y_TSy, st, A, B = testSignal(tau=10, noise2signal=1, nTrl=20, nSamp=300)
        fs = 100

        X_TSd[:, :, 1] = X_TSd[:, :, 1]*10  # much more power in 1 channel
        X_TSd[1, :, :] = X_TSd[1, :, :]*2  # and in 1 trial

        # X = np.cumsum(X,-2) # 1/f spectrum
        print("X={}".format(X_TSd.shape))
        plt.figure(1)
        plot_trial(X_TSd[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('Raw')
        plt.show(block=False)

        trans = ChannelPowerStandardizer()
        wX = []
        for i in range(X_TSd.shape[0]):
            wX.append(trans.transform(X_TSd[i:i+1, ...]))
        wX = np.concatenate(wX, 0)

        plt.figure(2)
        plot_trial(wX[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('Power Standardized')
        plt.show()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


class BadChannelRemover(BaseEstimator):
    """bad channel remover
    """

    def __init__(
            self, thresh: float = 3, mode: str = 'remove', minthresh: float = 4, disconnected_power: float = 1e-3,
            max_power: float = 1e8, center: bool = True, verb: int = 0):
        """Remove channels with statistically outlying power, i.e. more than thresh standard-deviations more/less power than an average channel

        Args:
            thresh (float, optional): outlier threshold in standard deviations. Defaults to 3.
            mode (str, optional): mode to do the removal.  One of: 'remove'-channel is removed, 'zero'-channel is set to zero value. Defaults to 'remove'.
            minthresh (float, optional): ????. Defaults to 4.
            disconnected_power (float, optional): channels with less than this amount of power are identified as disconneted and marked at 'bad'. Defaults to 1e-3.
            max_power (float, optional): channels with more than max_power are automatically marked as 'bad' even if not outliers. Defaults to 1e8.
            center (bool, optional): flag if we should center the data-per-trial before computing the power statistics. Defaults to True.
            verb (int, optional): verbosity level. Defaults to 0.
        """        
        self.thresh, self.mode, self.minthresh, self.disconnected_power, self.max_power, self.center, self.verb = \
            (thresh, mode, minthresh, disconnected_power, max_power, center, verb)

    def fit(self, X, y=None):
        """[summary]

        Args:
            X ([type]): (nTrial, nSample, d) time series data
        """
        # ensure 3-d input
        if X.ndim < 3:
            X = X.reshape((1,)*(3-X.ndim) + X.shape)
        isbad, pow = idOutliers(X, thresh=self.thresh, axis=(0, 1), minthresh=self.minthresh, center=self.center)
        mu_power = np.median(pow[pow > 0])
        if self.disconnected_power is not None:
            isbad[pow < mu_power*self.disconnected_power] = True
        if self.max_power is not None:
            isbad[pow > mu_power*self.max_power] = True
        if self.verb > 0:
            print("Ch-power={}".format(["{:5.3f}".format(p) for p in pow.ravel()]))
            if hasattr(X, 'info') and 'ch_names' in X.info and X.info['ch_names'] is not None:
                print("{} bad-ch: {}".format(np.sum(isbad),
                      [c for c, b in zip(X.info['ch_names'], isbad.ravel()) if b]))
            else:
                print("{} bad-ch: {}".format(np.sum(isbad), np.flatnonzero(isbad[0, 0, ...])))
        self._isbad = isbad[0, 0, ...]
        return self

    def transform(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate

        Returns:
            np.ndarray: the decorrelated data
        """
        if not hasattr(self, '_isbad'):
            self.fit(X)

        if not np.any(self._isbad):
            return X

        if self.mode == 'remove':
            X = X[..., self._isbad == False]
        else:
            X[..., self._isbad] = 0

        # update the meta-info
        if hasattr(X, 'info') and 'ch_names' in X.info and X.info['ch_names'] is not None:
            if self.mode == 'remove':
                X.info['ch_names'] = [c for c, b in zip(X.info['ch_names'], self._isbad) if not b]
            else:
                X.info['ch_names'] = [c+'.bad' if self._isbad[i] else c for i, c in enumerate(X.info['ch_names'])]

        return X

    def testcase(self, dur=3, fs=100, blksize=10):
        """[summary]

        Args:
            dur (int, optional): [description]. Defaults to 3.
            fs (int, optional): [description]. Defaults to 100.
            blksize (int, optional): [description]. Defaults to 10.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mindaffectBCI.decoder.utils import testSignal
        X_TSd, Y_TSy, st, A, B = testSignal(tau=10, noise2signal=1, nTrl=20, nSamp=300)
        fs = 100

        X_TSd[:, :, 1] = X_TSd[:, :, 1]*10  # much more power in 1 channel
        X_TSd[1, :, :] = X_TSd[1, :, :]*2  # and in 1 trial

        # X = np.cumsum(X,-2) # 1/f spectrum
        print("X={}".format(X_TSd.shape))
        plt.figure(1)
        plot_trial(X_TSd[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('Raw')
        plt.show(block=False)

        trans = BadChannelRemover()
        wX = []
        for i in range(X_TSd.shape[0]):
            wX.append(trans.transform(X_TSd[i:i+1, ...]))
        wX = np.concatenate(wX, 0)

        plt.figure(2)
        plot_trial(wX[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('BadChannelRemover')
        plt.show()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


class BadChannelInterpolator(BaseEstimator):
    """bad channel interpolator
    """

    def __init__(self, ch_names: list = None, bad_ch: list = None, verb: int = 0):
        """bad-channel interpolator -- based on spherical spline interpolation

        Args:
            ch_names (list, optional): list of channel names in 10-10 format. Defaults to None.
            bad_ch (list, optional): list of 'bad' channel names. If not given then bad-channels to be interploated are identified by channel names ending in '.bad'. Defaults to None.
            verb (int, optional): verbosity level for debugging. Defaults to 0.
        """        
        self.ch_names, self.bad_ch, self.verb = (ch_names, bad_ch, verb)

    def fit(self, X, y=None, ch_names: list = None, bad_ch: list = None, ch_pos: list = None):
        """[summary]

        Args:
            X ([type]): (nTrial, nSample, d) time series data
        """
        if ch_names is None:
            ch_names = self.ch_names
        if bad_ch is None:
            bad_ch = self.bad_ch
        if ch_names is None:
            # get from the meta-info
            if hasattr(X, 'info'):
                ch_names = X.info.get('ch_names', None)
        if bad_ch is None:
            # get from .bad postfix
            bad_ch = [c.endswith('.bad') for c in ch_names]
        elif isinstance(bad_ch[0], str):
            bad_ch = [c.lower() for c in bad_ch]
            bad_ch = [c.lower() in bad_ch for c in ch_names]
        # index for the bad channels
        self.isbad_ = np.zeros(X.shape[-1], dtype=bool)
        self.isbad_[bad_ch] = True
        isgood = np.logical_not(self.isbad_)
        # get 3d electrode position info
        if ch_pos is None:
            from mindaffectBCI.decoder.readCapInf import getPosInfo
            cnames, xy, xyz, iseeg = getPosInfo(ch_names)
            ch_pos = xyz
        self.ch_pos_ = ch_pos
        # compute the interpolation matrix
        A_all2good = make_spherical_spline_interpolation_matrix(ch_pos[isgood, :], ch_pos)
        # up-sample to make applicable directly to X
        self.Agd2all_dd_ = np.zeros((X.shape[-1], X.shape[-1]), dtype=X.dtype)
        self.Agd2all_dd_[isgood, :] = A_all2good.T
        return self

    def transform(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate

        Returns:
            np.ndarray: the decorrelated data
        """
        if not hasattr(self, 'Agd2all_dd_'):
            self.fit(X)

        X = np.einsum('...d,de->...e', X, self.Agd2all_dd_)

        # update the meta-info
        if hasattr(X, 'info') and 'ch_names' in X.info and X.info['ch_names'] is not None:
            X.info['ch_names'] = [c+'.int' if self.isbad_[i] else c for i, c in enumerate(X.info['ch_names'])]

        return X

    def testcase(self, dur=3, fs=100, blksize=10):
        """[summary]

        Args:
            dur (int, optional): [description]. Defaults to 3.
            fs (int, optional): [description]. Defaults to 100.
            blksize (int, optional): [description]. Defaults to 10.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mindaffectBCI.decoder.utils import testSignal
        X_TSd, Y_TSy, st, A, B = testSignal(d=6, tau=10, noise2signal=1, nTrl=20, nSamp=300)
        fs = 100

        X_TSd[:, :, 1] = X_TSd[:, :, 1]*10  # much more power in 1 channel
        X_TSd[1, :, :] = X_TSd[1, :, :]*2  # and in 1 trial

        # X = np.cumsum(X,-2) # 1/f spectrum
        print("X={}".format(X_TSd.shape))
        plt.figure(1)
        plot_trial(X_TSd[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('Raw')
        plt.show(block=False)

        trans = BadChannelInterpolator(bad_ch=['Cz', 'Pz'], ch_names=('Fz', 'Cz', 'Pz', 'C3', 'C4', 'CPz'))
        wX = []
        for i in range(X_TSd.shape[0]):
            wX.append(trans.transform(X_TSd[i:i+1, ...]))
        wX = np.concatenate(wX, 0)

        plt.figure(2)
        plot_trial(wX[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('BadChannelInterpolator')
        plt.show()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# TODO: allow tol as a number of components, other ways selecting the eigen-basis
# TODO: allow specify subset of X in trials/samples to compute the matrices over
def estimate_noise_subspace_decorrelation_filter(X_TSd, ch_idx, tol=5e-4, sample_idx=None):
    """decorrelate all channels in the input array w.r.t. a given subset of channels (in a given time range)

    Args:
        X_TSd ([type]): the data containing all channels
        ch_idx ([type]): indices into the **final** dimension of X which contains the artifact channels
        sample_idx (tuple): temporal subset to use to compute the projections
        tol ([type], optional): tolerance for the deflation, only signals stronger than this are removed. Defaults to 5e-3.

    Returns:
        [type]: X_TSd with the ch_idx channels decorrelated away
    """
    # artifact channel deflator
    # get data as 2d matrix
    if sample_idx is None:
        X_sd = X_TSd
    else:  # given a temporal noise subset, so use it
        if not isinstance(sample_idx, tuple):
            sample_idx = (sample_idx,)
        X_sd = X_TSd[(Ellipsis,)+sample_idx+(slice(None),)]
    X_sd = X_sd.reshape((-1, X_sd.shape[-1]))

    if ch_idx is not None and len(ch_idx) > 1:  # Artifact channel decorrelation
        # extract the artifact sub-set of channels
        A_sd = X_sd[..., ch_idx]
        A_sd = A_sd.reshape((-1, A_sd.shape[-1]))  # make 2d
        # compute the within and accross covariance matrices
        AA_aa = A_sd.T @ A_sd
        AX_ad = A_sd.T @ X_sd

        # solve ls problem for estimating each X as linear combination of A
        lambda_a, U_aa = np.linalg.eig(AA_aa)
        keep = lambda_a > max(lambda_a)*tol if tol > 0 else slice(-tol)
        #print('Noise subspace: {} {}/{}'.format(lambda_a,sum(keep),len(keep)))
        pinvAA_aa = U_aa[:, keep] @ np.diag(1/(lambda_a[keep])) @ U_aa[:, keep].T
        U_ad = pinvAA_aa @ AX_ad

        # make spatial filter maxtrix to deflate w.r.t. this best estimate
        # X - w*X = (I-w)*X
        sf_dd = np.eye(X_TSd.shape[-1])
        sf_dd[ch_idx, :] = sf_dd[ch_idx, :] - U_ad

    else:
        # Signal Sub-space Projection on all channels
        XX_dd = X_sd.T @ X_sd
        lambda_a, U_da = np.linalg.eig(XX_dd)
        keep = lambda_a > max(lambda_a)*tol if tol > 0 else slice(-tol)

        # X - U*U.T*X = (I-U*U.T)*X
        sf_dd = np.eye(X_TSd.shape[-1]) - U_da[:, keep] @ U_da[:, keep].T

    return sf_dd

# TODO[]: allow specify the noise time range in 'natural' units, i.e. time w.r.t. trigger etc.


class NoiseSubspaceDecorrelator(BaseEstimator):
    """Decorrelate the data w.r.t a noise sub-space -- either a channel subset or a time subset
    """

    def __init__(
            self, tol: float = 5e-4, ch_idx: tuple = None, noise_idx: tuple = 'idOutliers', ch_names: tuple = None,
            filterband: list = None, verb: int = 0):
        """ Decorrelate the data w.r.t a noise sub-space -- either a channel subset or a time subset

        Args:
            tol (float, optional): tolerance for defining the noise sub-space.  if >0 is fraction of max-noise eigenvalue, if <0 is number components to keep. Defaults to 5e-3.
            ch_idx (tuple, optional): index for the set of noise channels. Defaults to None.
            noise_idx (tuple, optional): index for the noise time points. Defaults to None.
            ch_names (tuple, optional): list of the noise channel names.  Only used if ch_idx is None. Defaults to None.
            verb (int, optional): verbosity level. Defaults to 0.
        """
        self.tol, self.ch_idx, self.ch_names, self.verb, self.noise_idx, self.filterband = \
            tol, ch_idx, ch_names, verb, noise_idx, filterband

    def fit(self, X, y=None, ch_names: tuple = None, fs=None):
        """[summary]

        Args:
            X ([type]): (nTrial, nSample, d) time series data
        """
        if fs is None:
            fs = X.info.get('fs', 250) if hasattr(X, 'info') else 250
        # use the chanel names meta-info to identify the channels to deflate w.r.t.
        self.ch_idx_ = self.ch_idx
        if self.ch_idx_ is None and self.ch_names is not None:
            self.ch_names = [c.lower() for c in self.ch_names]  # ensure lower case
            if ch_names is None and hasattr(X, 'info'):
                ch_names = X.info.get('ch_names', None)
            if ch_names is not None:
                ch_names = [c.lower() for c in ch_names]  # lower-case
                self.ch_idx_ = [i for i, n in enumerate(ch_names) if n in self.ch_names]

        if self.ch_idx_ is not None and len(self.ch_idx_) > 1:
            if self.filterband is not None:
                # pre-filter the artifact channels with the given filter
                X[..., self.ch_idx_], _, _ = butter_sosfiltfilt(
                    X[..., self.ch_idx_], self.filterband, fs=fs, order=4, axis=-2)

            noise_idx = self.noise_idx
            if isinstance(noise_idx, str) and noise_idx.startswith('idOutliers'):
                try:
                    thresh = float(noise_idx[len('idOutliers'):])
                except:
                    thresh = 2  # only to limit to ranges where artifact is present, so can be low
                # identify windows which contain excessive power
                winsz, step = int(fs), int(fs/2)
                Art_Twtd = window_axis(X[..., self.ch_idx_], winsz=winsz, axis=-2, step=step)
                # ID outlying windows
                nidx_Tw, pow = idOutliers(Art_Twtd, axis=(-1, -2))
                if self.verb > 0:
                    print("{}/{} artifacts (NSD)".format(np.sum(nidx_Tw), nidx_Tw.size))
                # convert to logical sample idx
                nidx_Tw = nidx_Tw.squeeze((-1, -2))
                noise_idx = np.zeros(X.shape[:2], dtype=bool)
                tmp_idx = np.arange(0, nidx_Tw.shape[1], dtype=int)*step
                for tau in range(winsz):
                    noise_idx[:, tau+tmp_idx] = nidx_Tw

            # fit the filter
            self.W_dd_ = estimate_noise_subspace_decorrelation_filter(X, self.ch_idx_, self.tol, noise_idx)
        else:
            self.W_dd_ = None

        return self

    def transform(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate

        Returns:
            np.ndarray: the decorrelated data
        """
        if not hasattr(self, 'W_dd_'):
            self.fit(X)

        # apply the deflation, N.B. use @ to preserve meta-info
        if self.W_dd_ is not None:
            X = X @ self.W_dd_  # np.einsum('...d,de->...e',X,self.W_dd_)

        return X

    def testcase(self, dur=3, fs=100, blksize=10):
        """[summary]

        Args:
            dur (int, optional): [description]. Defaults to 3.
            fs (int, optional): [description]. Defaults to 100.
            blksize (int, optional): [description]. Defaults to 10.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mindaffectBCI.decoder.utils import testSignal
        X_TSd, Y_TSy, st, A, B = testSignal(d=4, nY=3, tau=10, noise2signal=1, nTrl=20, nSamp=300)
        fs = 100

        X_TSd[:, :, 1] = X_TSd[:, :, 1]*4  # much more power in 1 channel
        X_TSd[:, :100, 2] = X_TSd[:, :100, 2]*5  # extra noise source at start of each trial

        # X = np.cumsum(X,-2) # 1/f spectrum
        print("X={}".format(X_TSd.shape))
        plt.figure(1)
        plot_trial(X_TSd[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('Raw')
        plt.show(block=False)

        # Noise channels with 1st 100 samples in each trial as noise sub-space
        trans = NoiseSubspaceDecorrelator(ch_idx=(1, 2), noise_idx=slice(100), tol=-1).fit(X_TSd)
        wX = []
        for i in range(X_TSd.shape[0]):
            wX.append(trans.transform(X_TSd[i:i+1, ...]))
        wX = np.concatenate(wX, 0)

        plt.figure(2)
        plot_trial(wX[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('NoiseSubspaceDecorrelator')
        plt.show(block=False)

        # Noise channels with 1st 100 samples in each trial as noise sub-space
        trans = NoiseSubspaceDecorrelator(ch_idx=None, noise_idx=slice(100), tol=-1).fit(X_TSd)
        wX = []
        for i in range(X_TSd.shape[0]):
            wX.append(trans.transform(X_TSd[i:i+1, ...]))
        wX = np.concatenate(wX, 0)

        plt.figure(3)
        plot_trial(wX[:1, ...], Y_TSy[:1, ...], fs)
        plt.suptitle('SignalSubspaceProjection')
        plt.show()


# class name alias for other users who use the old (less informative) name
SignalSpaceProjector = NoiseSubspaceDecorrelator


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class Slicer(BaseEstimator, ModifierMixin):
    """slice continous data into trials
    """

    def __init__(self, slice: tuple = None):
        """slice continous data into trials

        Args:
            slice (tuple, optional): tuple with the slicing indices, either as slice objects, or as strings in numpy slice format, e.g. ":" or "1:4". Defaults to None.
        """        
        self.slice = slice

    def fit(self, X, y=None):
        """fit the slicer

        Args:
            X ([type]): [description]
            y ([type], optional): [description]. Defaults to None.
            slice (tuple, optional): tuple with the slicing indices. Defaults to None.
        """
        tmp = list(self.slice)
        for i, s in enumerate(tmp):
            #if s=='...':   tmp[i]=Ellipsis
            if isinstance(s, str):
                if s == ':':
                    tmp[i] = slice(None)
                elif ":" in s:
                    tmp[i] = slice(*[int(i) for i in s.split(":")])
        self.slice_ = tuple(tmp)
        return self

    def modify(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate

        Returns:
            np.ndarray: the decorrelated data
        """
        X = X[self.slice_]
        if y is not None:
            # bodge, only slice Y for trials and samples
            y = y[self.slice_[:2]+(Ellipsis,)]

        return X, y

    def testcase(self):
        mod = Slicer(slice=(Ellipsis, slice(1, 3)))
        test_transform(mod)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class ChannelSlicer(BaseEstimator, TransformerMixin):
    """select a sub-set of channels
    """

    def __init__(self, channels: tuple = None, exclude_channels: list = None, ch_names: list = None, axis: int = -1):
        """slice a subset of channels from the data

        Args:
            channels (tuple, optional): the list of channels to keep, as list of channel indices, or list of channel-names, or slice object. Defaults to None.
            exclude_channels (list, optional): list of channels to remove, as list-of-int for channel indices, or list-of-str for channel names. Defaults to None.
            ch_names (list-of-str, optional): human readable channel names, or channel name matching. Defaults to None.
            axis (int, optional): axis of X which contains the channels. Defaults to -1.
        """        
        self.channels, self.exclude_channels, self.ch_names, self.axis = channels, exclude_channels, ch_names, axis

    def fit(self, X, y=None, fs=None, ch_names=None):
        """fit the channel picker

        Args:
            X ([type]): [description]
            y ([type], optional): [description]. Defaults to None.
            slice (tuple, optional): tuple with the slicing indices. Defaults to None.
        """
        self.ch_idx_ = None
        if self.channels is None and self.exclude_channels is None:  # fast path passthrough case
            return self

        # use the chanel names meta-info to identify the channels to keep
        ch_idx = [True]*X.shape[-1]  # assume all channels are OK
        if ch_names is None and hasattr(X, 'info'):
            ch_names = X.info.get('ch_names', None)
        if ch_names is None or len(ch_names) == 0:  # default channel names are just numbers
            ch_names = ["{:d}".format(i) for i in range(X.shape[-1])]
        if self.channels is not None:
            if isinstance(self.channels[0], str):
                channels = [c.lower() for c in self.channels]
                ch_idx = [c.lower() in channels for c in ch_names]  # boolean index
                #ch_idx = np.array(self.ch_idx_,dtype=bool)
            elif isinstance(self.channels[0], int) or isinstance(self.channels[0], bool):
                ch_idx = self.channels

        # remove excluded channels
        if isinstance(self.exclude_channels[0], str):
            ex_ch = [c.lower() for c in self.exclude_channels]
            ex_idx = [c.lower() in ex_ch for c in ch_names]
            # update so only included and not excluded are possible
            ch_idx = [c and not x for c, x in zip(ch_idx, ex_idx)]
        # ensure is logical index with the right shape
        self.ch_idx_ = np.zeros(X.shape[-1], dtype=bool)
        self.ch_idx_[ch_idx] = True
        return self

    def transform(self, X, y=None):
        if self.ch_idx_ is None:  # fast path passthrough case
            return X

        if not (self.axis == X.ndim-1 or self.axis == -1):
            raise ValueError("axis not equal last dim not currently supported")

        X = X[..., self.ch_idx_]
        # update the meta info
        if hasattr(X, 'info') and 'ch_names' in X.info and X.info['ch_names'] is not None:
            X.info['ch_names'] = [c for c, i in zip(X.info['ch_names'], self.ch_idx_) if i]

        return X

    def testcase(self):
        mod = ChannelSlicer(channels=('Fp1', 'Fp2'))
        test_transform(mod)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def chunk_data(X_TSd, Y_TSy, tau: int = 100, offset: int = 0, chunk_size: int = 1, per_chunk_label: bool = False):
    # slice X to get block of data per sample (with optional step-size)
    X_TStd = window_axis(X_TSd, winsz=tau, step=chunk_size, axis=1)
    # TODO[]: include label offset....
    assert offset is None or offset == 0, 'Offset not yet supported'
    Y_TSty = window_axis(Y_TSy, winsz=tau, step=chunk_size, axis=1)

    # convert the labels to be per-chunk, i.e. only non-zero at 1st sample in chunk
    if per_chunk_label:
        lab = pool_axis(Y_TSty, pool_type=per_chunk_label, axis=2, keepdims=False)
        Y_TSty = np.zeros_like(Y_TSty)
        Y_TSty[:, :, 0, ...] = lab  # lable on 1st sample of the chunk

    X_TSd = np.reshape(X_TStd, (X_TStd.shape[0]*X_TStd.shape[1],)+X_TStd.shape[2:])
    Y_TSy = np.reshape(Y_TSty, (Y_TSty.shape[0]*Y_TSty.shape[1],)+Y_TSty.shape[2:])
    return X_TSd, Y_TSy


class Chunker(BaseEstimator, ModifierMixin):
    """slice continous data chunks
    """

    def __init__(self, tau: int = None, tau_ms: float = None, offset: int = None, offset_ms: float = 0, fs: float = None,
                 chunk_size: int = None, chunk_size_ms: float = None, per_chunk_label: bool = False):
        """slice continuous data into chunks/trials

        Args:
            tau (int, optional): _description_. Defaults to None.
            tau_ms (float, optional): _description_. Defaults to None.
            offset (int, optional): _description_. Defaults to None.
            offset_ms (float, optional): _description_. Defaults to 0.
            fs (float, optional): _description_. Defaults to None.
            chunk_size (int, optional): _description_. Defaults to None.
            chunk_size_ms (float, optional): _description_. Defaults to None.
            per_chunk_label (bool, optional): _description_. Defaults to False.
        """        
        self.tau, self.tau_ms, self.offset, self.offset_ms, self.fs, self.chunk_size, self.chunk_size_ms, self.per_chunk_label =\
            (tau, tau_ms, offset, offset_ms, fs, chunk_size, chunk_size_ms, per_chunk_label)

    def fit(self, X, y=None, fs=None):
        if fs is None:
            if hasattr(X, 'info'):
                fs = X.info['fs']
            else:
                fs = self.fs
        self.tau_ = self.tau if self.tau is not None else int(self.tau_ms * fs / 1000)
        self.offset_ = self.offset if self.offset is not None else int(
            self.offset_ms * fs / 1000) if self.offset_ms is not None else 0
        self.chunk_size_ = self.chunk_size if self.chunk_size is not None else int(
            self.chunk_size_ms * fs / 1000) if self.chunk_size_ms is not None else 1
        return self

    def modify(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate

        Returns:
            np.ndarray: the decorrelated data
        """
        # propogate the meta-info
        info = X.info if hasattr(X, 'info') else None

        X, y = chunk_data(X, y, tau=self.tau_, offset=self.offset_,
                          chunk_size=self.chunk_size_, per_chunk_label=self.per_chunk_label)

        if info:
            X = InfoArray(X, info)  # replace meta-info
        return X, y

    def testcase(self):
        mod = Chunker(tau=10, chunk_size=10, fs=100)
        test_transform(mod)


class TrialLabelSelector(BaseEstimator, ModifierMixin):
    """ Select a sub-set of trials which match a label specification
    """

    def __init__(self, label_matcher='non_zero'):
        """Select a sub-set of trials which match a label specification

        Args:
            label_matcher (str|callable|list, optional): criteria for selecting trials.  If str then must be 'non_zero' or '>0' to only select trials with non-zero stimulus info. If callable then with signature 'idx = label_matcher(y)' to return a boolean array with true for trials to keep, if list-of-int or list-of-bool then directly index and keep these trials. Defaults to 'non_zero'.
        """        
        self.label_matcher = label_matcher

    def modify(self, X, y=None):
        """select which trials to keep based on their labels
        """
        keep_T = None
        if self.label_matcher == 'non_zero' or self.label_matcher == '>0':
            keep_T = np.any(y != 0, axis=tuple(range(1, y.ndim)))
        elif callable(self.label_matcher):  # function to do the matching
            keep_T = self.label_matcher(y)
        elif hasattr(self.label_matcher, '__iter__'):
            # list of class labels to keep
            keep_T = False
            for v in self.label_matcher:
                keep_T = np.logical_or(keep_T, np.any(y == v, axis=tuple(range(1, y.ndim))))
        else:
            raise ValueError('Unknown label_matcher')

        if np.any(keep_T):
            #print("Selected {} / {}  trials as {}".format(sum(keep_T),X.shape[0],str(self.label_matcher)))
            X = X[keep_T, ...]
            y = y[keep_T, ...]

        return X, y

    def testcase(self):
        mod = TrialLabelSelector(label_matcher='non_zero')
        test_transform(mod)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class FFTfilter(BaseEstimator, TransformerMixin):
    """filterband transform the inputs
    """

    def __init__(self, axis:int=1, filterband: list = None, isfilterbank:bool= False, fs: float = None, blksz:int=None, blksz_ms:float=None, window=None, center:bool=True, 
                 fftlen: int = None, overlap: float = .5, squeeze_feature_dim: bool = True, prefix_feature_dim: bool = False,
                 ola_filter: bool = False, verb=1):
        """apply a (set-of) spectral filters to the data using the FFT-filter technique

        TODO[]: Imnplement downsampling...

        Args:
            axis (int, optional): axis of X which corrosponds to time. Defaults to 1.
            filterband (list, optional): list-of-filter specifications.  Filter specification is in the format (low,high,type) where low and high are frequencies in Hz, and type is one-of 'bandpass', 'bandstop' or 'hilbert' for envelop transform. Defaults to None.
            isfilterbank (bool, optional): If True, then run as filterbank where each band in filterband generates it's own output feature.  If false, then operate as a single filter where the filterbands are combined to generate a single output feature, where the final filter is the OR'd combination of these filters.  Defaults to False.
            fs (float, optional): sampling rate of the data.  Got from X's meta-info if not given. Defaults to None.
            blksz (int, optional): block-size in samples for applying the FFT. Defaults to None.
            blksz_ms (float, optional): block-size in milliseconds for applying the FFT. Defaults to None.
            window (str, optional): temporal window to apply to data blocks before FFT transform.  One of 'hamming','hanning','cos'. Defaults to None.
            fftlen (int, optional): length of the analysis window in the fft-space.  Defaults to blksz/2. Defaults to None.
            overlap (float, optional): fractional overlap of the FFT analysis windows. Defaults to .5.
            squeeze_feature_dim (bool, optional): if true and only a single band is given then remove the resulting singleton feature dimension. Defaults to False.
            prefix_feature_dim (bool, optional): put the feature dim before the time dimension?. Defaults to False.
            ola_filter (bool, optional): if trun then use the overlap-add fitler, otherwise use the fft a whole-trial-at-a-time filter approach. Defaults to True.
            verb (int, optional): verbosity level. Defaults to 1.
        """
        self.axis, self.filterband, self.isfilterbank, self.fs, self.blksz, self.blksz_ms, self.window, self.center, self.fftlen, self.overlap, self.squeeze_feature_dim, self.prefix_feature_dim, self.verb, self.ola_filter = (
            axis, filterband, isfilterbank, fs, blksz, blksz_ms, window, center, fftlen, overlap, squeeze_feature_dim, prefix_feature_dim, verb, ola_filter)

    def fit(self, X, y=None, fs: float = None, blksz: int = None, fftlen: int = None):
        """[summary]

        Args:
            X ([type]): [description]
        """
        # setup the window
        if fs is not None:
            self.fs_ = fs
        elif hasattr(X, 'info'):
            self.fs_ = X.info['fs']
        else:
            self.fs_ = self.fs  # meta-data ndarray

        if self.ola_filter:
            self.blksz_ = blksz if blksz is not None else int(
                self.blksz_ms * self.fs_ / 1000) if self.blksz_ms is not None else self.fs_/2 if self.fs_ is not None else 100  # default to 1/2 sec or 100 samples
            self.blksz_ = int(self.blksz_)
            self.window_, self.step_ = pp.get_window_step(self.blksz_, self.window, self.overlap)
            # get to right shape
            self.window_ = self.window_
            self.fftlen_ = fftlen if fftlen is not None else self.blksz_*2
        else:
            self.fftlen_ = X.shape[self.axis]
            self.blksz_ = self.fftlen_
            self.window_, self.step_ = pp.get_window_step(self.blksz_, self.window)

        #print("FFTfilter: fs={} filterbank={}".format(self.fs, filterbank))
        filter_bf, bandtype, self.freqs_ = pp.fftfilter_masks(
            self.fftlen_, self.filterband, fs=self.fs_, dtype=X.dtype)
        if self.isfilterbank:
            self.filter_bf_ = filter_bf
            self.bandtype_  = bandtype
        else:
            # combine filter specs to make a single pass filter
            self.filter_bf_ = np.any(filter_bf, axis=0, keepdims=True)
            self.bandtype_ = bandtype if isinstance(bandtype,str) or len(bandtype)==1 else ['hilbert'] if any(b=='hilbert' for b in bandtype) else ['bandpass']

        #print("freq={}\nmask={}\nbandtype={}\n".format(self.freqs_,self.filter_bf_, self.bandtype_))
        return self

    def transform(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate
            nsamp(int): number of samples to interpolate

        Returns:
            np.ndarray: the decorrelated data
        """
        if not hasattr(self, 'filter_bf_'):
            raise ValueError("Must fit before transform!")

        # ensure 3-d input
        X_TSd = X
        if X_TSd.ndim < 3:
            X_TSd = X_TSd.reshape((-1,)*(3-X_TSd.ndim) + X_TSd.shape)

        # TODO[]: make truely incremental... buffer and release when fully processed!
        if self.ola_filter:
            X_TSfd = pp.inner_ola_fftfilterbank(
                X_TSd, self.filter_bf_, self.bandtype_, self.window_, self.step_, axis=self.axis, center=self.center,
                prefix_band_dim=self.prefix_feature_dim)
        else:
            X_TSfd = pp.inner_fftfilterbank(
                X_TSd, self.filter_bf_, self.bandtype_, window=self.window_, axis=self.axis, center=self.center,
                prefix_band_dim=self.prefix_feature_dim)

        if hasattr(X, 'info') and not hasattr(X_TSfd, 'info'):
            # manually re-attach the meta-info
            X_TSfd = InfoArray(X_TSfd, info=X.info)

        if self.squeeze_feature_dim and X_TSfd.shape[self.axis+1] == 1:
            X_TSfd = X_TSfd.squeeze(self.axis+1)

        return X_TSfd

    def testcase(self):
        mod = FFTfilter(blksz_ms=500, filterband=(1, 5, 15, 20, 'bandpass'))
        test_transform(mod)

        mod = FFTfilter(blksz_ms=500, filterband=(1, 5, 15, 20, 'hilbert'))
        test_transform(mod)

        mod = FFTfilter(blksz_ms=500, filterband=((1, 5, 15, 20, 'bandpass'), (25,25,'bandpass')))
        test_transform(mod)




class FFTfilterbank(FFTfilter):
    def __init__(self, axis:int=1, filterbank: list = None, fs: float = None, blksz:int=None, blksz_ms:float=None, window=None,
                 fftlen: int = None, overlap: float = .5, squeeze_feature_dim: bool = False, prefix_feature_dim: bool = False,
                 ola_filter: bool = False, verb=1):
        """apply a (set-of) spectral filters to the data using the FFT-filter technique

        Args:
            axis (int, optional): axis of X which corrosponds to time. Defaults to 1.
            filterbank (list, optional): list-of-filter specifications.  Filter specification is in the format (low,high,type) where low and high are frequencies in Hz, and type is one-of 'bandpass', 'bandstop' or 'hilbert' for envelop transform. Defaults to None.
            fs (float, optional): sampling rate of the data.  Got from X's meta-info if not given. Defaults to None.
            blksz (int, optional): block-size in samples for applying the FFT. Defaults to None.
            blksz_ms (float, optional): block-size in milliseconds for applying the FFT. Defaults to None.
            window (str, optional): temporal window to apply to data blocks before FFT transform.  One of 'hamming','hanning','cos'. Defaults to None.
            fftlen (int, optional): length of the analysis window in the fft-space.  Defaults to blksz/2. Defaults to None.
            overlap (float, optional): fractional overlap of the FFT analysis windows. Defaults to .5.
            squeeze_feature_dim (bool, optional): if true and only a single band is given then remove the resulting singleton feature dimension. Defaults to False.
            prefix_feature_dim (bool, optional): put the feature dim before the time dimension?. Defaults to False.
            ola_filter (bool, optional): if trun then use the overlap-add fitler, otherwise use the fft a whole-trial-at-a-time filter approach. Defaults to True.
            verb (int, optional): verbosity level. Defaults to 1.
        """
        self.filterbank = filterbank
        super().__init__(axis=axis,filterband=filterbank,isfilterbank=True,fs=fs,blksz=blksz,blksz_ms=blksz_ms,window=window,fftlen=fftlen,overlap=overlap,squeeze_feature_dim=squeeze_feature_dim,prefix_feature_dim=prefix_feature_dim,ola_filter=ola_filter,verb=verb)

    def fit(self,X,y,**kwargs):
        # ensure to override the band specification (in case updated in set-config)
        self.filterband = self.filterbank
        super().fit(X,y,**kwargs)

    def testcase(self):
        mod = FFTfilterbank(blksz_ms=500, filterbank=((1, 5, 40, 50, 'bandpass'), (1, 5, 40, 50, 'hilbert'),))
        test_transform(mod)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def block_cov_mx(X_TSd, Y_TSy, axis, window, step, resample_y, mean: bool = False):
    axis = axis if axis >= 0 else X_TSd.ndim+axis  # ensure positive axis spec
    window = window.reshape((-1,)+(1,)*(X_TSd.ndim-axis-1))
    blkidxs = np.arange(0, X_TSd.shape[axis], step)
    Y_Tby = np.zeros(Y_TSy.shape[:axis]+(len(blkidxs),)+Y_TSy.shape[axis+1:]
                     ) if Y_TSy is not None else None  # downsampled Y
    X_Tbdd = np.zeros(
        X_TSd.shape[: axis] + (len(blkidxs),) + X_TSd.shape[axis + 1:] + (X_TSd.shape[-1],),
        dtype=X_TSd.dtype)  # block cov downsampled X
    for bi, i in enumerate(blkidxs):  # overlap segments
        # get this segment
        blksamp = len(window) if i+len(window) < X_TSd.shape[axis] else X_TSd.shape[axis]-i
        idx = slice(i, i+blksamp)

        # extract, apply window and cov
        if blksamp == len(window):
            Xi = X_TSd[:, idx, ...] * window
        else:  # not a full window of data available -- reverse pad?
            Xi = X_TSd[:, idx, ...] * window[:blksamp, ...]
        XXi = np.einsum("TS...d,TS...e->T...ed", Xi, Xi)  # only outer-prod for the *last* dim
        if mean:
            XXi = XXi / np.sum(np.abs(window[:blksamp]))
        X_Tbdd[:, bi, ...] = XXi

        # downsample Y
        if Y_TSy is not None:
            Y_Tby[:, bi, ...] = pool_axis(Y_TSy[:, idx, ...], resample_y, axis)
    return X_Tbdd, Y_Tby


class BlockCovarianceMatrixizer(BaseEstimator, ModifierMixin):
    """transform features to cross feature covariance matrices
    """

    def __init__(
            self, axis=1, fs: float = None, blksz: int = None, blksz_ms: float = None, resample_y='max', window: str = 1,
            overlap: float = 0, squeeze_feature_dim: bool = False, mean: bool = True, verb=1):
        self.axis, self.fs, self.blksz, self.blksz_ms, self.window, self.overlap, self.resample_y, self.squeeze_feature_dim, self.mean, self.verb = (
            axis, fs, blksz, blksz_ms, window, overlap, resample_y, squeeze_feature_dim, mean, verb)
        assert self.axis == 1

    def fit(self, X, y=None, fs: float = None, blksz: int = None, blksz_ms: float = None):
        # setup the window, and step
        self.fs_ = fs if fs is not None else X.info['fs'] if hasattr(X, 'info') else self.fs
        self.blksz_ = int(self.blksz) if self.blksz is not None else int(
            self.blksz_ms * self.fs_ / 1000) if self.blksz_ms is not None else X.shape[self.axis]
        self.blksz_ = min(X.shape[self.axis], max(1, self.blksz_))

        self.window_, self.step_ = get_window_step(self.blksz_, window=self.window, overlap=self.overlap)

        return self

    def modify(self, X, y=None):
        """ convert from features to feature covariances in temporal blocks
        """
        if not hasattr(self, 'window_'):
            raise ValueError("Must fit before transform!")

        # ensure 3-d input
        X_TSd = X
        Y_TSy = y
        if X_TSd.ndim < 3:
            X_TSd = X_TSd.reshape((1,)*(3-X_TSd.ndim) + X_TSd.shape)
            if Y_TSy is not None:
                Y_TSy = Y_TSy.reshape((1,)*(3-X_TSd.ndim) + Y_TSy.shape)

        X_TBdd, Y_TBy = block_cov_mx(X_TSd, Y_TSy, self.axis, self.window_, self.step_, self.resample_y, self.mean)

        # update the meta-info
        if hasattr(X, 'info') and not X.info is None:
            if not hasattr(X_TBdd, 'info'):
                X_TBdd = InfoArray(X_TBdd, info=X.info)
            X_TBdd.info['fs'] = X_TBdd.info['fs'] / self.step_
        if y is not None and hasattr(y, 'info') and not y.info is None:
            if not hasattr(Y_TBy, 'info'):
                Y_TBy = InfoArray(Y_TBy, info=y.info)
            Y_TBy.info['fs'] = X_TBdd.info['fs'] / self.step_

        return X_TBdd, Y_TBy

    def testcase(self):
        mod = BlockCovarianceMatrixizer(blksz=blksize, overlap=.5)
        test_transform(mod)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


class DiagonalExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, axis1=-2, axis2=-1):
        """extract the diagonal entries of the input data

        Args:
            axis1 (int, optional): _description_. Defaults to -2.
            axis2 (int, optional): _description_. Defaults to -1.
        """        
        self.axis1, self.axis2 = (axis1, axis2)

    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
        X = np.diagonal(X, axis1=self.axis1, axis2=self.axis2)
        return X


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


class WelchPSD(BaseEstimator, ModifierMixin):
    def __init__(self, axis=-2, detrend: bool = 'constant', blksz=None, blksz_ms=None, fs: float = None, resample_y: str = None):
        """compute the power spectral density of the input data along the given axis using welch's method

        Args:
            axis (int, optional): _description_. Defaults to -2.
            detrend (bool, optional): type of detrending to do before computing the power. Defaults to 'constant'.
            blksz (_type_, optional): blocks size for the welch blocks in samples. Defaults to None.
            blksz_ms (_type_, optional): block size for the welch blocks in milliseconds. Defaults to None.
            fs (float, optional): sample rate of the data.  Got from X's meta-info if not given. Defaults to None.
            resample_y (str, optional): Technique to use to re-sample the stimulus info Y to match that of the re-sampled X. Defaults to None.
        """        
        self.axis, self.detrend, self.blksz, self.blksz_ms, self.fs, self.resample_y = (
            axis, detrend, blksz, blksz_ms, fs, resample_y)

    def fit(self, X, y=None, fs=None):
        self.fs_ = fs if fs is not None else X.info['fs'] if hasattr(X, 'info') else self.fs
        self.blksz_ = int(self.blksz) if self.blksz is not None else int(
            self.blksz_ms * self.fs_ / 1000) if self.blksz_ms is not None else self.fs_//4
        self.blksz_ = max(1, self.blksz_)
        # print(self.fs_,self.blksz_ms,self.blksz_)
        return self

    def modify(self, X, y=None):
        axis = self.axis if self.axis > 0 else X.ndim+self.axis  # postive ax spec
        X = self.transform(X, y=None)
        if axis == 1:  # we have colapsed the sample dim!
            X = X[:, np.newaxis, ...]  # re-insert a singlenton sample dim
            # re-sample the time dim of y
            y = pool_axis(y, self.resample_y, axis=axis)
        return X, y

    def transform(self, X, y=None):
        info = X.info if hasattr(X, 'info') else None
        freqs, X = welch(X, axis=self.axis, fs=self.fs_, nperseg=self.blksz_,
                         return_onesided=True, detrend=self.detrend)
        #ola_welch(X,axis=self.axis, window_t=self.blksz_)
        # print(freqs)
        # update the meta-info
        if info:
            X = InfoArray(X, info)
            X.info['freqs'] = freqs
        return X


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


class CommonSpatialPatterner(BaseEstimator):
    """transform uses common spatial pattern algorithm to map to a virtual channel space
    """

    def __init__(
            self, nfilt_per_class: int = 3, mean: bool = True, compress_feature_dim=False, 
            y0_is_true: bool=True, tau_ms: float = None, tau:int=None, fs:float=None,
            spoc: bool = False, reg: float = 1e-6, rcond: float = 1e-6, verb=0):
        """use the common spatial pattern algorithm to map from sensors to virtual channels

        WARNING: Note this function assumes *one-class-per-trial* structure. 

        Args:
            nfilt_per_class (int, optional): _description_. Defaults to 3.
            mean (bool, optional): _description_. Defaults to True.
            compress_feature_dim (bool, optional): _description_. Defaults to False.
            y0_is_true (bool, optional): if True then the first output is assumed to contain the unique event labels.   Defaults to True.
            tau (int, optional): the length in samples of the stimulus response. Defaults to None.
            tau_ms (int, optional): the lenght in milliseconds of the stimulus response. Defaults to None
            fs (float, optional): the sampling rate of the data, used for coverting milliseconds to samples.  Defaults to None.
            spoc (bool, optional): _description_. Defaults to False.
            reg (float, optional): _description_. Defaults to 1e-6.
            rcond (float, optional): _description_. Defaults to 1e-6.
            verb (int, optional): _description_. Defaults to 1.
        """
        self.nfilt_per_class, self.mean, self.compress_feature_dim, self.y0_is_true, self.tau, self.tau_ms, self.fs, self.spoc, self.reg, self.rcond, self.verb = \
            (nfilt_per_class, mean, compress_feature_dim, y0_is_true, tau, tau_ms, fs, spoc, reg, rcond, verb)

    def fit_csp(self, Cxx_ydd, Cxx_dd, reg, rcond):
        """fit common spatial patterns to the data summarized by it's covariance matrices

        Args:
            Cxx_ydd ([type]): [description]
            Cxx_dd ([type]): [description]
            reg ([type]): [description]
            rcond ([type]): [description]

        Returns:
            [type]: [description]
        """
        # solve by 2-step, global-whiten, and per-class eigen-decomposition approach
        # TODO: fast-path the binary case!
        isqrtCxx, sqrtCxx = robust_whitener(Cxx_dd, reg, rcond)
        # if self.reg:
        #     Cxx_dd = Cxx_dd * (1-self.reg) + np.eye(Cxx_dd.shape[0])*self.reg*np.mean(Cxx_dd.diagonal())
        W_ekd = np.zeros((Cxx_ydd.shape[0], self.nfilt_per_class, Cxx_ydd.shape[-1]), dtype=Cxx_ydd.dtype)
        A_ekd = np.zeros((Cxx_ydd.shape[0], self.nfilt_per_class, Cxx_ydd.shape[-1]), dtype=Cxx_ydd.dtype)
        for ei in range(Cxx_ydd.shape[0]):
            isqrtCxxCxx_yisqrtCxx = isqrtCxx.T @ Cxx_ydd[ei, ...] @ isqrtCxx
            l_k, U_dk = np.linalg.eigh(isqrtCxxCxx_yisqrtCxx)

            # take the largest entries (signed)
            sidx = np.argsort(np.abs(l_k))[::-1]  # sorted order, biggest amplitude
            if self.verb>0:
                print("cls={}) lambda={}".format(ei,l_k[sidx]))
            U_dk = U_dk[:, sidx[:self.nfilt_per_class]]  # * np.sign(l_k[sidx[:self.nfilt_per_class]])[np.newaxis,:]
            W = isqrtCxx.T @ U_dk  # include the whitener to make filter
            A = sqrtCxx.T @ U_dk
            # TODO[]: compute and save the spatial pattern as well?
            W_ekd[ei, :W.shape[0], ...] = W.T
            A_ekd[ei, :A.shape[0], ...] = A.T
        return W_ekd, A_ekd

    def get_class_covariances(self, X_TSd, Y_TSye, spoc: bool = False, y0_is_true: bool = True, tau:int = None):
        """compute the class specific and global spatial covariances from given inputs

        Args:
            X_TSd ([type]): [description]
            Y_TSye ([type]): [description]
            y0_is_true (bool, optional): [description]. Defaults to True.

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """        
        # ensure inputs have the right shape
        if X_TSd.ndim < 3:
            X_TSd = X_TSd.reshape((1,)*(3-X_TSd.ndim) + X_TSd.shape)
        elif X_TSd.ndim > 3:  # compress extra feature dims
            X_TSd = X_TSd.reshape(X_TSd.shape[:2]+(-1,))
        if Y_TSye.ndim < 4:
            Y_TSye = Y_TSye.reshape((1,)*(4-Y_TSye.ndim) + Y_TSye.shape)
        elif Y_TSye.ndim > 4:
            Y_TSye = Y_TSye.reshape(Y_TSye.shape[:3]+(-1,))
        if not y0_is_true:
            raise NotImplementedError('multi-Y isnt supported yet')

        # extract the labelling for the first output.  This should be unique
        Y_TSe = Y_TSye[:, :, 0, :]
        # get per-event-type covariance matrices
        Cxx_ydd = np.zeros((Y_TSe.shape[-1], X_TSd.shape[-1], X_TSd.shape[-1]), dtype=X_TSd.dtype)
        Cxx_dd = np.zeros((X_TSd.shape[-1], X_TSd.shape[-1]), dtype=X_TSd.dtype)  # all data cov
        for ti in range(X_TSd.shape[0]):
            # update the total covariance matrix
            Cxx_dd = Cxx_dd + X_TSd[ti, ...] .T @ X_TSd[ti, ...]
            # Compute the per-event type sub-covariance matrices
            for ei in range(Y_TSe.shape[-1]):
                if spoc:  # SPOC -> Y is regression target for this event-type
                    samp_wght = Y_TSe[ti, :, ei]
                    tmp = X_TSd[ti, :, :] * samp_wght[:, np.newaxis]
                else:  # CSP -> Y is class indicator, or trial_start indicator
                    if tau is not None: # trial-start indicator
                        trl_idx = np.argmax(Y_TSe[ti, :, ei] > 0)
                        trl_idx = slice(trl_idx,trl_idx+tau) 
                    else:
                        trl_idx = Y_TSe[ti, :, ei] > 0
                    tmp = X_TSd[ti, trl_idx, :]
                    #print("tmp={}".format(tmp.shape))
                Cxxty = tmp.T @ tmp
                Cxx_ydd[ei, ...] = Cxx_ydd[ei, ...] + Cxxty
        return Cxx_ydd, Cxx_dd

    def fit(self, X_TSd, Y_TSye=None, fs=None):
        if self.tau_ms is not None:
            if fs is None: # get sample rate from meta-info
                fs = X_TSd.info['fs'] if hasattr(X_TSd,'info') else self.fs
            self.tau = int(self.tau_ms * fs / 1000)

        # get the summary statistics
        Cxx_ydd, Cxx_dd = self.get_class_covariances(X_TSd, Y_TSye, self.spoc, self.y0_is_true, self.tau)
        # compute the per-class CSP directions
        self.W_ekd_, self.A_ekd_ = self.fit_csp(Cxx_ydd, Cxx_dd, self.reg, self.rcond)
        return self

    def transform(self, X_TSd, y=None):
        """ map to virtual channels
        """
        if not hasattr(self, 'W_ekd_'):
            raise ValueError("Must fit before transform!")
        # ensure 3-d input
        if X_TSd.ndim < 3:
            X_TSd = X_TSd.reshape((1,)*(3-X_TSd.ndim) + X_TSd.shape)

        # apply the mapping
        X_TSek = np.einsum("TSd,ekd->TSek",X_TSd, self.W_ekd_)

        # X_TSek = np.zeros(X_TSd.shape[:2] + self.W_ekd_.shape[:2], dtype=X_TSd.dtype)
        # for ei in range(self.W_ekd_.shape[0]):
        #     for ki in range(self.W_ekd_.shape[1]):
        #         X_TSek[..., ei, ki] = X_TSd @ self.W_ekd_[ei, ki, :]

        if self.compress_feature_dim:  # make into single big d
            X_TSek = X_TSek.reshape(X_TSek.shape[:2]+(-1,))

        # transfer the meta-info (if any)
        if hasattr(X_TSd, 'info'):
            if not hasattr(X_TSek, 'info'):
                # update the channel names
                info = X_TSd.info.copy()
                if self.compress_feature_dim:
                    info['ch_names'] = [ "csp{:d}.{:d}".format(cls,comp) for cls in range(self.W_ekd_.shape[0]) for comp in range(self.W_ekd_.shape[1]) ]
                else:
                    info['ch_names'] = [ "csp{:d}".format(comp) for comp in range(self.W_ekd_.shape[1]) ]
                X_TSek = InfoArray(X_TSek, info=info)

        return X_TSek

    def plot_model(self, evtlabs=None, plot_pattern: bool = True, ch_names=None, **kwargs):
        sp = (self.A_ekd_, 'spatial-pattern') if plot_pattern and hasattr(self,
                                                                          'A_ekd_') else (self.W_ekd_, 'spatial-filter')
        plot_factoredmodel(sp[0].reshape((-1, sp[0].shape[-1])), None, evtlabs=evtlabs,
                           spatial_filter_type=sp[1], ncol=1, ch_names=ch_names, **kwargs)

    def testcase(self, dur=3, fs=100, blksize=10):
        """[summary]

        Args:
            dur (int, optional): [description]. Defaults to 3.
            fs (int, optional): [description]. Defaults to 100.
            blksize (int, optional): [description]. Defaults to 10.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mindaffectBCI.decoder.utils import testSignal
        from mindaffectBCI.decoder.stim2event import stim2event
        # make a continuously AM varying test problem
        X_TSd, Y_TSye, st, A, B = testSignal(d=3, tau=1, nY=1, nE=1, noise2signal=.2, nTrl=100,
                                            nSamp=300, isi=0, classification=True, induced=-1)
        fs = 100

        # convert to binary event coding
        Y_TSye, _, evtlabs = stim2event(Y_TSye[...,0],'hotone')

        # X = np.cumsum(X,-2) # 1/f spectrum
        print("X={}".format(X_TSd.shape))
        plt.figure(1)
        plot_trial(X_TSd[:1, ...], Y_TSye[:1, ...], fs)
        plt.suptitle('Raw')
        plt.show(block=False)

        mod = CommonSpatialPatterner(nfilt_per_class=1)
        mod.fit(X_TSd, Y_TSye)
        plt.figure()
        mod.plot_model()

        mod = CommonSpatialPatterner(nfilt_per_class=1, spoc=True)
        mod.fit(X_TSd, Y_TSye)
        plt.figure()
        mod.plot_model()

        X_TSke = mod.transform(X_TSd)

        # compare raw vs summed filterbank
        plt.figure()
        plot_trial(X_TSke[:1, ...], Y_TSye[:1, ...], fs)
        plt.suptitle('CSP')
        plt.show()


class FilterBankCommonSpatialPatterner(CommonSpatialPatterner):
    def __init__(self, nfilt_per_class: int = 3, filterbank=None, mean: bool = True, compress_feature_dim=False,
                 spoc: bool = False, reg: float = 1e-6, rcond: float = 1e-6, verb=1):
        super().__init__(nfilt_per_class, mean, compress_feature_dim, spoc, reg, rcond, verb)
        self.filterbank = filterbank
        raise NotImplementedError("Sorry not made yet!")

    def get_filterbank_class_covariances(
            self, X_TSd, Y_TSye, filterbank=None, spoc: bool = False, y0_is_true: bool = True):
        pass

    def fit_fbcsp(self, Cxx_yfdd, Cxx_fdd):
        pass

    def fit(self, X, y=None, fs=None):
        if fs is None and hasattr(X, 'info'):  # meta-data ndarray
            fs = X.info['fs']


# #--------------------------------------------------------------------------
# #--------------------------------------------------------------------------
# #--------------------------------------------------------------------------
# #--------------------------------------------------------------------------
# class TriUExtractor(BaseEstimator, TransformerMixin):
#     def __init__(self,axis1=-2,axis2=-1):
#         self.axis1, self.axis2 = (axis1,axis2)
#     def transform(self,X,y=None):
#         X = np.diagonal(X,axis1=self.axis1,axis2=self.axis2)
#         return X


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


class AdaptiveSpatialWhitener(BaseEstimator, TransformerMixin):
    """Incremental streaming tranformer to channel power normalization in an input stream
    """

    def __init__(self, halflife: int = None, halflife_s: float = 10):
        """Incremental streaming tranformer to channel power normalization in an input stream

        Args:
            halflife (int, optional): half-life in samples for the exp-moving-average estimation of the data-covariance for the whitener. Defaults to None.
            halflife_s (float, optional): half-life in seconds for the exp-moving-average estimation window. Defaults to 10.
        """
        self.halflife, self.halflife_s = (halflife, halflife_s)

    def fit(self, X, y=None, fs=None):
        self.fit_transform(X.copy(), y, fs=fs)
        return self

    def fit_transform(self, X, y=None, fs=None):
        """[summary]

        Args:
            X ([type]): [description]
        """
        if fs is None and hasattr(X, 'info'):  # meta-data ndarray
            fs = X.info['fs']
        if self.halflife is None:
            self.halflife_ = self.halflife_s * fs
        else:
            self.halflife_ = self.halflife

        self.Cxx_ = np.zeros((X.shape[-1], X.shape[-1]), dtype=X.dtype)
        self.N_ = 0
        self.W_ = None
        self.iW_ = None
        X = self.transform(X)  # N.B. use a copy so don't change it twice!
        return X

    def transform(self, X, y=None):
        """add per-sample timestamp information to the data matrix

        Args:
            X (float): the data to decorrelate

        Returns:
            np.ndarray: the decorrelated data
        """
        # ensure 3-d input
        X_TSd = X.reshape((1,)*(3-X.ndim) + X.shape)

        if not hasattr(self, 'Cxx_'):
            self.fit(X_TSd)

        # process trial at a time if given multiple trials
        for ti in range(X_TSd.shape[0]):
            Xi = X_TSd[ti, ...].copy()
            Cxx = updateCxx(None, Xi, unitnorm=False)

            wght = 1 / (2.0**(X_TSd.shape[1]/self.halflife_))  # weight exp in terms of halflifes
            # print('wght={}'.format(wght))
            oCxx_ = self.Cxx_
            self.Cxx_ = self.Cxx_*wght + Cxx
            self.N_ = self.N_ * wght + X.shape[1]
            #print( "{:d}) {:4.3f} * {:4.3f} + {:4.3f} = {:4.3f}".format(ti,np.trace(oCxx_),wght,np.trace(Cxx),np.trace(self.Cxx_) ))
            # TODO[]: allow set of dims for the whitener
            self.W_, self.iW_ = robust_whitener((self.Cxx_/self.N_).astype(X_TSd.dtype), symetric=True)
            # TODO[]: allow arbitary axis for the whitening transform
            X_TSd[ti, ...] = Xi @ self.W_

        return X_TSd

    def testcase(halflife_s=50000):
        mod = AdaptiveSpatialWhitener(halflife_s=halflife_s)
        test_transform(mod)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class SpatialWhitener(BaseEstimator, TransformerMixin):
    """Spatially whiten or spherize over channels
    """

    def __init__(self, axis: int = -1, symetric: bool = True, reg: float = None, rcond: float = None):
        """spatially whiten or spherize the input data X

        Args:
            axis (int, optional): the axis of X to whiten. Defaults to -1.
            symetric (bool, optional): if true then compute a symetric whitener. Defaults to True.
            reg (float, optional): regularization strenght for computing the inverse-square-root covariance matrix. Defaults to None.
            rcond (float, optional): recopial-condition-number for thresholding components as zero-valued and discarded in computing the inverse. Defaults to None.
        """
        self.axis, self.symetric, self.reg, self.rcond = (axis, symetric, reg, rcond)

    def fit(self, X, y=None):
        """fit the spatial whitener to the data in X

        Args:
            X ([type]): [description]
        """
        assert self.axis == -1 or self.axis == X.ndim-2

        # ensure 3-d input
        Cxx = updateCxx(None, X)
        # TODO[]: allow set of dims for the whitener
        self.W_de_, self.iW_de_ = robust_whitener(Cxx, symetric=self.symetric, reg=self.reg, rcond=self.rcond)
        return self

    def transform(self, X, y=None):
        """apply the spatial whitener to X
        Args:
            X (float): the data to decorrelate
        Returns:
            np.ndarray: the decorrelated data
        """
        # ensure 3-d input
        X_TSd = X.reshape((1,)*(3-X.ndim) + X.shape)

        if not hasattr(self, 'W_de_'):
            self.fit(X_TSd)

        for ti in range(X_TSd.shape[0]):
            X_TSd[ti, ...] = X_TSd[ti, ...] @ self.W_de_

        return X_TSd

    def testcase(self):
        mod = SpatialWhitener()
        test_transform(mod)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class Resampler(BaseEstimator, ModifierMixin):
    """Incremental streaming transformer for downsampling data transformations

    Args:
        ModifierMixin ([type]): sklearn compatible transformer
    """

    def __init__(
            self, axis: int = 1, fs: float = 250, fs_out: float = 60, mode: str = 'sample', resample_y: str = 'max', nsamp: int = 0,
            verb: int = 0):
        """Incremental streaming transformer for downsampling data transformations

        Args:
            axis (int, optional): the 'time' axis to resample. Defaults to 1.
            fs (float, optional): input sample rate of the data. Defaults to 250.
            fs_out (float, optional): desired output sample rate. Defaults to 60.
            mode (str, optional): mode for the re-sampler. Defaults to 'sample'.
            resample_y (str, optional): method used to re-sample the stimulus info, Y, to match the resampled X. Defaults to 'max'.
            nsamp (int, optional): starting sample counter for incremental calls. Defaults to 0.
            verb (int, optional): _description_. Defaults to 0.

        Raises:
            ValueError: _description_
        """
        self.fs, self.fs_out, self.axis, self.nsamp, self.verb, self.mode, self.resample_y = \
            (fs, fs_out, axis, nsamp, verb, mode, resample_y)
        if not self.axis == 1:
            raise ValueError("axis != 1 is not yet supported!")

    def fit(self, X, y=None, fs: float = None):
        """[summary]

        Args:
            X ([type]): [description]
            fs (float, optional): [description]. Defaults to None.
            zi ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if fs is not None:  # parameter overrides stored fs
            self.fs_ = fs
        elif hasattr(X, 'info'):  # in case we have rich X
            self.fs_ = X.info['fs']
        else:
            self.fs_ = self.fs

        # preprocess -> downsample
        self.nsamp = 0
        self.resamprate_ = int(round(self.fs_*2.0/min(self.fs_, self.fs_out)))/2.0 if self.fs_out is not None else 1
        self.fs_out_ = self.fs/self.resamprate_
        if self.verb > 1:
            print("resample: {}->{}hz rsrate={}".format(self.fs_, self.fs_out_, self.resamprate_))

        return self

    def modify(self, X, Y):
        """[summary]

        Args:
            X_TSd ([type]): [description]
            Y_TSy ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if X.ndim < 3:  # add leading dims to match shape
            X = X.reshape((1,)*(3-X.ndim) + X.shape)
            Y = Y.reshape((1,)*(3-X.ndim) + Y.shape)  # same extra dims for Y
        if Y.ndim < 3:
            Y = Y.reshape((1,)*(3-Y.ndim) + Y.shape)

        # propogate the filter coefficients between calls
        if not hasattr(self, 'resamprate_'):
            self.fit(X[0:1, :])

        nsamp = self.nsamp
        self.nsamp = self.nsamp + X.shape[self.axis]  # track *raw* sample counter

        # preprocess -> downsample @60hz
        if self.resamprate_ > 1:
            # number samples through this cycle due to remainder of last block
            resamp_start = nsamp % self.resamprate_
            # convert to number samples needed to complete this cycle
            # this is then the sample to take for the next cycle
            if resamp_start > 0:
                resamp_start = self.resamprate_ - resamp_start

            # allow non-integer resample rates
            idx = np.arange(resamp_start, X.shape[self.axis], self.resamprate_, dtype=X.dtype)

            if self.resamprate_ % 1 > 0 and idx.size > 0:  # non-integer re-sample, interpolate
                idx_l = np.floor(idx).astype(int)  # sample above
                idx_u = np.ceil(idx).astype(int)  # sample below
                # BODGE: guard for packet ending at sample boundary.
                idx_u[-1] = idx_u[-1] if idx_u[-1] < X.shape[self.axis] else X.shape[self.axis]-1
                w_u = (idx - idx_l).astype(X.dtype)  # linear weight of the upper sample
                w_u = w_u.reshape((-1,)+(1,)*(X.ndim-2))  # add trailing singlentons
                X = X[:, idx_u, ...] * w_u + X[:, idx_l, ...] * (1-w_u)  # linear interpolation

            else:
                idx = idx.astype(int)
                X = X[:, idx, ...]  # decimate X (trl, samp, d)

            if Y is not None:
                # preserve y's meta-info
                yinfo = None if not hasattr(Y, 'info') else Y.info

                rY = np.zeros((Y.shape[0], len(idx))+Y.shape[2:], dtype=Y.dtype)
                for i in range(len(idx)):
                    sampIdx = slice(int(idx[i]), int(idx[i+1])
                                    ) if i < len(idx)-1 else slice(int(idx[i]), Y.shape[self.axis])
                    rY[:, i, ...] = pool_axis(Y[:, sampIdx, ...], self.resample_y,
                                              self.axis)  # decimate Y (trl, samp, y)
                Y = rY

                if yinfo:
                    Y = InfoArray(Y, yinfo)

            if hasattr(X, 'info') and not X.info is None:  # update the meta-info
                X.info['fs'] = X.info['fs'] / self.resamprate_
            if Y is not None and hasattr(Y, 'info') and not Y.info is None and 'fs' in Y.info: 
                Y.info['fs'] = Y.info['fs'] / self.resamprate_

        return (X, Y)

    def testcase(self):
        fs_out = 10
        mod = Resampler(fs=fs, fs_out=fs_out)
        test_transform(mod)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class ButterFilterAndResampler(BaseEstimator, ModifierMixin):
    """Incremental streaming transformer for downsampling data transformations
    """

    def __init__(
            self, filterband=((0, 5),
                              (5, -1)),
            order: int = 6, axis: int = 1, fs: float = None, fs_out: float = None, ftype: str = 'butter', filter_y: bool = False,
            resample_y: str = 'max', verb: int = 0):
        """Incremental streaming transformer for downsampling data transformations

        Args:
            filterband (tuple, optional): specification of the filter to apply, as for butterfilt_and_downsample. Defaults to ((0,5),(5,-1)).
            order (int, optional): filter order. Defaults to 6.
            axis (int, optional): axis to apply the filter along, i.e. time-axis. Defaults to 1.
            fs (float, optional): sampling rate of the input data. Defaults to None.
            fs_out (float, optional): output sampling rate. Defaults to None.
            ftype (str, optional): type of filter to compute and apply. Defaults to 'butter'.
            filter_y (bool, optional): do we filter Y as well as X?. Defaults to False.
            resample_y (str, optional): how to downsample the target values. Defaults to 'max'.
            verb (int, optional): verbosity level. Defaults to 0.

        Raises:
            ValueError: [description]
        """
        self.filterband, self.order, self.axis, self.fs, self.fs_out, self.ftype, self.filter_y, self.resample_y, self.verb = (
            filterband, order, axis, fs, fs_out, ftype, filter_y, resample_y, verb)
        #if self.filterband is None and filterband is not None: self.filterband = filterband
        if not self.axis == 1:
            raise ValueError("axis != 1 is not yet supported!")
        if self.filterband is None and self.fs_out is not None:
            self.filterband = [self.fs_out*.48, -1]  # high pass at just under nyquist before sub-sampling
        self.nsamp = 0

    def fit(self, X, y=None, fs: float = None, zi=None):
        """[summary]

        Args:
            X ([type]): [description]
            fs (float, optional): sampling rate of X. Defaults to None.

        Returns:
            self: fitted modifer object
        """
        if fs is not None:  # parameter overrides stored fs
            self.fs_ = fs
        elif hasattr(X, 'info'):  # in case we have rich X
            self.fs_ = X.info['fs']
        else:
            self.fs_ = self.fs

        # estimate them from the given information
        if X.ndim < 3:  # add leading dims to match shape
            X = X.reshape((1,)*(3-X.ndim) + X.shape)
        # init with 1st trials info -- as we run incrementally over trials.
        _, self.sos_, self.zi_ = butter_sosfilt(
            X[0, ...], self.filterband, self.fs_, order=self.order, axis=self.axis-1, zi=zi, ftype=self.ftype)

        if self.filter_y:
            if y.ndim < 3:
                y = y.reshape((1,)*(3-y.ndim) + y.shape)
            _, self.sos_y_, self.zi_y_ = butter_sosfilt(
                y[0, ...], self.filterband, self.fs_, order=self.order, axis=self.axis-1, ftype=self.ftype)
        else:
            self.sos_y_, self.zi_y_ = (None, None)

        # preprocess -> downsample
        self.nsamp = 0
        self.resamprate_ = max(1, int(round(self.fs_*2.0/self.fs_out))/2.0) if self.fs_out is not None else 1
        self.fs_out_ = self.fs_/self.resamprate_
        if self.verb > 0:
            print("resample: {}->{}hz rsrate={}".format(self.fs_, self.fs_out_, self.resamprate_))

        return self

    def modify(self, X, Y):
        """[summary]

        Args:
            X_TSd ([type]): [description]
            Y_TSy ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if X.ndim < 3:  # add leading dims to match shape
            Y = Y.reshape((1,)*(3-X.ndim) + Y.shape)  # same extra dims for Y
            X = X.reshape((1,)*(3-X.ndim) + X.shape)
        if Y.ndim < 3:  # add leading dims to match shape
            Y = Y.reshape((1,)*(3-Y.ndim) + Y.shape)  # same extra dims for Y

        # propogate the filter coefficients between calls
        if not hasattr(self, 'sos_'):
            self.fit(X[0:1, :])

        if self.sos_ is not None:
            if X.shape[0] == 1:  # incremental calling
                X[0, ...], self.zi_ = sosfilt(self.sos_, X[0, ...], axis=self.axis-1, zi=self.zi_)
            else:
                for ti in range(X.shape[0]):  # process trials incrementally in order
                    zi = sosfilt_zi_warmup(None, X[ti, ...], self.axis-1, self.sos_)  # re-warmup each trial
                    X[ti, ...], zi = sosfilt(self.sos_, X[ti, ...], axis=self.axis-1, zi=zi)
                # save final filter state for later incremental calls
                self.zi_ = zi

        # also filter Y if wanted
        if self.sos_y_ is not None:
            if Y.shape[0] == 1:  # incremental calling
                Y[0, ...], self.zi_y_ = sosfilt(self.sos_y_, Y[0, ...], axis=self.axis-1, zi=self.zi_y_)
            else:
                for ti in range(Y.shape[0]):  # process trials incrementally in order
                    zi = sosfilt_zi_warmup(None, Y[ti, ...], self.axis-1, self.sos_y_)  # re-warmup each trial
                    Y[ti, ...], zi = sosfilt(self.sos_y_, Y[ti, ...], axis=self.axis-1, zi=zi)
                # save final filter state for later incremental calls
                self.zi_y_ = zi

        nsamp = self.nsamp
        self.nsamp = self.nsamp + X.shape[self.axis]  # track *raw* sample counter

        # preprocess -> downsample @60hz
        if self.resamprate_ > 1:
            # number samples through this cycle due to remainder of last block
            resamp_start = nsamp % self.resamprate_
            # convert to number samples needed to complete this cycle
            # this is then the sample to take for the next cycle
            if resamp_start > 0:
                resamp_start = self.resamprate_ - resamp_start

            # allow non-integer resample rates
            idx = np.arange(resamp_start, X.shape[self.axis], self.resamprate_, dtype=X.dtype)

            if self.resamprate_ % 1 > 0 and idx.size > 0:  # non-integer re-sample, interpolate
                idx_l = np.floor(idx).astype(int)  # sample above
                idx_u = np.ceil(idx).astype(int)  # sample below
                # BODGE: guard for packet ending at sample boundary.
                idx_u[-1] = idx_u[-1] if idx_u[-1] < X.shape[self.axis] else X.shape[self.axis]-1
                w_u = (idx - idx_l).astype(X.dtype)  # linear weight of the upper sample
                w_u = w_u.reshape((-1,)+(1,)*(X.ndim-2))  # add trailing singlentons
                X = X[:, idx_u, ...] * w_u + X[:, idx_l, ...] * (1-w_u)  # linear interpolation

            else:
                idx = idx.astype(int)
                X = X[:, idx, ...]  # decimate X (trl, samp, d)

            if Y is not None:
                # preserve y's meta-info
                yinfo = None if not hasattr(Y, 'info') else Y.info

                rY = np.zeros((Y.shape[0], len(idx))+Y.shape[2:], dtype=Y.dtype)
                for i in range(len(idx)):
                    sampIdx = slice(int(idx[i]), int(idx[i+1])
                                    ) if i < len(idx)-1 else slice(int(idx[i]), Y.shape[self.axis])
                    rY[:, i, ...] = pool_axis(Y[:, sampIdx, ...], self.resample_y,
                                              self.axis)  # decimate Y (trl, samp, y)
                Y = rY

                if yinfo:
                    Y = InfoArray(Y, yinfo)

            if hasattr(X, 'info') and not X.info is None:  # update the meta-info
                X.info['fs'] = X.info['fs'] / self.resamprate_
            if hasattr(Y, 'info') and not Y.info is None and 'fs' in Y.info:  # update the meta-info
                Y.info['fs'] = Y.info['fs'] / self.resamprate_

        return (X, Y)

    def testcase(self):
        fs_out = 100
        bands = ((0, 10, 'bandpass'))
        mod = ButterFilterAndResampler(filterband=bands, fs=fs, fs_out=fs_out)
        test_transform(mod)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class VirtualTargetAdder(BaseEstimator, ModifierMixin):
    """Add virtual target sequences to the target matrix, so can assess performance at sequence level

    Args:
        ModifierMixin ([type]): sklearn compatible transformer
    """

    def __init__(self, nvirt_out=-20):
        """Add virtual target sequences to the target matrix, so can assess performance at sequence level

        Args:
            nvirt_out (int, optional): number of virtual outputs to add to the data. If <0 then this is the total number of outputs to have. Defaults to -20.
        """        
        self.nvirt_out = nvirt_out

    def fit(self, X, y=None):
        return self

    def modify(self, X, Y):
        """modify Y to add some additional virtual targets

        Args:
            X_TSd ([type]): [description]
            Y_TSy or Y_TSye ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if self.nvirt_out is not None:
            if Y.ndim < 3:  # add trial dim if not there
                Y = Y.reshape((1,)*(3-Y.ndim) + Y.shape)

            # generate virtual outputs for testing -- so get valid results even if only 1 sequence
            Y_perm_source = Y[:, :, 1:, ...] if Y.shape[2] > 1 else Y
            virt_Y = block_permute(Y_perm_source, self.nvirt_out, axis=2, perm_axis=1)
            Y2 = np.concatenate((Y, virt_Y), axis=2)
            if hasattr(Y, 'info') and not hasattr(Y2, 'info'):
                # manually re-attach the meta-info
                Y2 = InfoArray(Y2, info=Y.info)

        return (X, Y2)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class MetaInfoAdder(BaseEstimator, TransformerMixin):
    """add arbitary meta info to X
    """

    def __init__(self, info:dict=None, force: bool = False):
        """add arbitary meta info to X

        Args:
            info (dict, optional): dictionary of meta-information to add. Defaults to None.
            force (bool, optional): if true then force override of any existing meta-info of X. Defaults to False.
        """        
        self.info, self.force = info, force

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """modify Y to add some additional virtual targets

        Args:
            X_TSd ([type]): [description]
            Y_TSy ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        X2 = X
        if self.info is not None:
            if not hasattr(X, 'info'):
                X2 = InfoArray(X, info=self.info)
            elif self.force:
                # TODO[]: update meta-info and info separately?
                if X.info is None:
                    X.info = self.info
                else:
                    X.info.update(self.info)
        return X2


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class PreprocessPipeline(BaseEstimator,ModifierMixin):
    """meta-estimator to implement a pipeline which can contain ModifierMixin's 

    Args:
        BaseEstimator ([type]): [description]

        TODO[]: add utility functions to access and modify stages, e.g. by name, number, etc.
    """

    def __init__(self, stages: list, verb: int = 0):
        """meta-estimator to make a pipeline of transform stages

        Args:
            stages (list-of-Transforms): list of preprocess_pipeline tranformation objects
            verb (int, optional): verbosity level. Defaults to 0.
        """
        self.stages = stages
        self.verb = verb

    def get_params(self, deep=True):
        params = super().get_params()
        if not deep:
            return params
        for n, s in self.stages:
            p = s.get_params(deep=deep)
            params.update({n+"__"+k: v for (k, v) in p.items()})
        return params

    def set_params(self, **kwargs):
        for i in range(len(self.stages)):
            stagename, stage = self.stages[i]
            # get all parameters for this stage
            stagename = stagename.lower()

            # TODO[X]: allow to disable stage by setting stagname:None?
            keys = tuple(kwargs.keys())
            for k in keys:
                if k.lower() == stagename.lower():  # dict of parameters
                    val = kwargs.pop(k)
                    if val is None:  # don't change anything
                        pass
                    elif val.lower() == 'passthrough' or val.lower() == 'skip':  # replace this stage with None
                        self.stages[i] = (val, None)

                    elif isinstance(val, BaseEstimator):  # replace this stage with val
                        # override
                        self.stages[i] = (self.stages[i][0], val)
                    else:
                        stage.set_params(**val)

                elif k.lower().startswith(stagename.lower()+"__"):  # specific parameter
                    val = kwargs.pop(k)
                    paramname = k[len(stagename.lower()+"__"):]
                    stage.set_params(**{paramname: val})

        super().set_params(**kwargs)
        return self

    def fit(self, X, Y, **kwargs):
        self.fit_modify(X.copy(), Y.copy(), **kwargs)
        return self

    def fit_modify(self, X, Y, until_stage: int = None, verb: int = 0, **kwargs):
        for i in range(len(self.stages)):
            # early termination if wanted
            if until_stage is not None and \
                    i == (until_stage if until_stage >= 0 else len(self.stages)+until_stage):
                break

            name, stage = self.stages[i]
            # get params for this stage -- key has name as prefix
            if name in kwargs.keys():  # given dict with args directly
                stage_args = kwargs[name]
            elif name.lower() in kwargs.keys():
                stage_args = kwargs[name.lower()]
            else:  # given key with args
                prefix = name.lower()+"__"
                # extract matching args into stage specific dict
                stage_args = {k[len(prefix):]: v for (k, v) in kwargs.items() if k.startswith(prefix)}

            # replace pipeline entry with a clone of the stage
            #stage = stage.clone()
            # one-call versions
            if hasattr(stage, 'fit_modify'):
                X, Y = stage.fit_modify(X, Y, **stage_args)
            elif hasattr(stage, 'fit_transform'):
                X = stage.fit_transform(X, Y, **stage_args)
            else:
                # two-step versions
                if hasattr(stage, 'fit'):
                    stage.fit(X, Y, **stage_args)
                #self.stages[i] = stage

                if hasattr(stage, 'predict'):
                    X = stage.predict(X, y=Y)
                elif hasattr(stage, 'modify'):  # change X and Y
                    X, Y = stage.modify(X, Y)
                elif hasattr(stage, 'transform'):  # just change X
                    X = stage.transform(X, y=Y)
                else:
                    raise ValueError("Pipeline stage with no modify/predict/transform method?")
            if verb > 0:
                print("{:d}) {}  X={} Y={}".format(i, name, X.shape, Y.shape))
        return (X, Y)

    # TODO[]: think of a better name than modify, e.g. dataset_transform
    def modify(self, X, Y, until_stage: int = None, verb: int = 0, isprediction:bool=False):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            until_stage (int, optional): Apply pipeline until this stage.  If None then all stages. Defaults to None.
            isprediction (bool, optional): If True and stage has predict method then use this method in preference to modify or transform. Defaults to False.
            verb (int, optional): Verbosity level, higher means more debug messages. Defaults to 0.

        Returns:
            _type_: _description_
        """        
        for i, (name, stage) in enumerate(self.stages):
            # early termination if wanted
            if until_stage is not None and \
                    i == (until_stage if until_stage >= 0 else len(self.stages)+until_stage):
                break

            if self.verb > 1:
                print("{}) {}  X={} Y={} -> ".format(i, name, X.shape, Y.shape))

            if isprediction and hasattr(stage, 'predict'):  # generate final predicted Y
                X = stage.predict(X, Y)
            elif hasattr(stage, 'modify'):  # change X and Y
                X, Y = stage.modify(X, Y)
            elif hasattr(stage, 'transform'):  # only change X
                X = stage.transform(X, y=Y)

            if verb > 0:
                print("{:d}) {}  X={} Y={}".format(i, name, X.shape, Y.shape))
        return (X, Y)

    def predict(self, X, Y):
        X, Y = self.modify(X.copy(), Y.copy(), isprediction=True)
        return X

    def score(self, X, Y, isprediction:bool=True):
        """apply the pipeline up to the last stage and then use the final stages score method

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            isprediction (bool, optional): If True and stage has predict method then use this method in preference to modify or transform. Defaults to False.

        Returns:
            _type_: _description_
        """
        X, Y = self.modify(X.copy(),Y.copy(),until_stage=-1)
        return self.stages[-1][1].score(X, Y)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def make_preprocess_pipeline(pipeline: list):
    """make a preprocessing pipeline.

    This is similar to a sklearn Pipeline, except accepts Modifier, which can modify both X and Y of the inputs.

    Args:
        pipeline (list): list of pipeline stages, in the format ('label', Transformer|Modifier) or ('ClassName', dict-of-args) or ("ClassName:label", dict-of-args)
            where label is a unique label for this pipeline stage, and Transformer|Modifier is a class instance, 
            'ClassName' is a string containing a Transformer|Modifier class to be constructed, dict-of-args is a dictionary of arguments to pass to the class constructor

    Returns:
        PreprocessPipeline: a constructed pre-processing pipeline
    """
    stages = []
    for pi, stage in enumerate(pipeline):
        args = dict()
        if isinstance(stage, str):
            stagename = stage
        else:
            stagename = stage[0]
            if len(stage) > 1:
                args = stage[1]
                if len(stage) > 2:
                    print("Warning: extra args list ignored!")
        try:
            stagename, stagelabel = stagename.split(":")
        except ValueError:
            stagelabel = stagename
        # stagename is final class name if it's a string
        if isinstance(args,dict) and '.' in stagelabel:
            stagelabel = stagelabel.split('.')[-1]

        # ensure stagelabel is unique
        n_before = sum(lab.startswith(stagelabel.lower()) for lab,_ in stages)
        if n_before>0 :
            stagelabel = "{:s}_{:d}".format(stagelabel,n_before)

        if isinstance(args, dict):
            transformer = None
            if stagename in globals():
                transformer = globals()[stagename](**args)
            if transformer is None:
                try:
                    transformer = import_and_make_class(
                        "mindaffectBCI.decoder.preprocess_transforms." + stagename, **args)
                except:
                    pass
                    # print("Error creating class {}".format(stagename))
                    # import traceback
                    # traceback.print_exc()
            if transformer is None:
                try:
                    # last ditch -- try from full string
                    transformer = import_and_make_class(stagename, **args)
                except:
                    print("Error creating class {}".format(stagename))
                    import traceback
                    traceback.print_exc()
            if transformer is None or isinstance(transformer, str):
                raise ValueError("Can't find transformer class: {}".format(stagename))

        else:  # pre-initialized class
            transformer = args
        # TODO[]: Better error message if constructor fails!
        if not (hasattr(transformer, 'modify') or hasattr(transformer, 'transform')):
            raise ValueError("Transformer {} without `transform` or `modify` method!".format(transformer))
        stages.append((stagelabel.lower(), transformer))
    return PreprocessPipeline(stages)


def testcase_pipeline():
    """testcase for pipeline objects
    """    
    import numpy as np
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.utils import testSignal, InfoArray
    import pickle
    X_TSd, Y_TSye, st, A, B = testSignal(d=2, tau=10, noise2signal=1, nTrl=20, nSamp=300, nE=1)
    Y_TSy = Y_TSye[..., 0]
    fs = 100

    X_TSd = InfoArray(X_TSd, info=dict(fs=fs))
    Y_TSy = InfoArray(Y_TSy, info=dict(fs=fs))

    p = pickle.dumps(X_TSd)
    q = pickle.loads(p)
    print(q)

    print("X={}".format(X_TSd.shape))
    plt.figure()
    plot_trial(X_TSd[:1, ...], Y_TSy[:1, ...], fs)
    plt.suptitle('Raw')
    plt.show(block=False)
    # make a preprocessing pipeline
    filterband = ((0, 0, 40, 50, 'bandpass'))  # ,(1,5,40,50,'hilbert'),)
    fs_out = 50

    pipeline = [('MetaInfoAdder', dict(info=dict(fs=fs))),
                ["TrialPlotter", {"suptitle": "0: +Input"}],
                # ('Resampler',dict(fs_out=50)),
                # 'AdaptiveSpatialWhitener',
                ["ScalarFunctionApplier", {"op": "square"}],
                ('FFTfilter:filt1', dict(filterband=filterband, squeeze_feature_dim=True)),
                # "Log",
                ['AdaptiveSpatialWhitener', {"halflife_s": 1}],
                ["TrialPlotter", {"suptitle": "1: +AdaptiveWhiten"}],

                ["TargetEncoder", {"evtlabs": "hoton"}],
                ["TrialPlotter", {"suptitle": "2: +hoton"}],

                ["TargetEncoder", {"evtlabs": "re"}],
                ["TrialPlotter", {"suptitle": "2: +re"}],

                ["VirtualTargetAdder", {"nvirt_out": -50}],

                ["TimeShifter", {"timeshift_ms": -50}],
                ["TrialPlotter", {"suptitle": "3: +TimeShift"}],
                "EventRelatedPotentialPlotter",
                # 'SpatialWhitener',
                # ("TwoDTransformerWrapper",dict(transformer='sklearn.preprocessing.StandardScaler')),
                # ('FFTfilter:filt2',dict(filterband=(0,0,8,9,'bandpass'),squeeze_feature_dim=True)),
                # ('Resampler',dict(fs_out=fs_out)),
                # ('ButterFilterAndResampler:filt2',dict(filterband=(8,-1),fs_out=20)),
                ["BlockCovarianceMatrixizer", {"blksz_ms": 25}],
                ["TrialPlotter", {"suptitle": "4: +CovMx"}],

                "DiagonalExtractor",
                ["TrialPlotter", {"suptitle": "5: +DiagExtractor"}],
                "EventRelatedPotentialPlotter",

                # 'VirtualTargetAdder',
                ('MultiCCA2', dict(evtlabs=None, tau_ms=800, offset_ms=0))
                ]
    ppp = make_preprocess_pipeline(pipeline)
    ppp.fit(X_TSd.copy(), Y_TSy.copy())

    ppX_TSd, ppY_TSy = ppp.modify(X_TSd, Y_TSy)
    if hasattr(ppX_TSd, 'info'):
        fs_out = ppX_TSd.info['fs']

    print("ppX={}".format(ppX_TSd.shape))
    plt.figure()
    plot_trial(ppX_TSd[:1, ...], ppY_TSy[:1, ...], fs_out)
    plt.suptitle('1-step pre-processed:\n {}'.format(pipeline))

    pX = []
    pY = []
    for i in range(X_TSd.shape[0]):
        Xi, Yi = ppp.modify(X_TSd[(i,), ...], Y_TSy[(i,), ...])
        pX.append(Xi)
        pY.append(Yi)
    pX_TSd = np.concatenate(pX, 0)
    pY_TSy = np.concatenate(pY, 0)

    print("pX={}".format(pX_TSd.shape))
    plt.figure()
    plot_trial(pX_TSd[:1, ...], pY_TSy[:1, ...], fs_out)
    plt.suptitle('pre-processed:\n {}'.format(pipeline))

    plt.show()

    from sklearn.model_selection import GridSearchCV
    tuned_parameters = dict(multicca__rank=(1, 2, 5),
                            multicca__tau=[int(dur*fs_out) for dur in [.2, .3, .5, .7]],
                            #multicca__evtlabs=(('re', 'fe'),('re','ntre')),
                            # log=[None,"passthrough"]
                            )
    cv_ppp = GridSearchCV(ppp, tuned_parameters, n_jobs=-1)
    cv_ppp.fit(X_TSd, Y_TSy)
    print("CVOPT:\n\n{} = {}\n".format(cv_ppp.best_estimator_, cv_ppp.best_score_))
    means = cv_ppp.cv_results_['mean_test_score']
    stds = cv_ppp.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, cv_ppp.cv_results_['params']):
        print("{:5.3f} (+/-{:5.3f}) for {}".format(mean, std * 2, params))
    print()


if __name__ == '__main__':

    FFTfilter().testcase()

    FFTfilterband().testcase()

    CommonSpatialPatterner().testcase()

    TimeShifter().testcase()

    BadChannelInterpolator().testcase()

    Chunker().testcase()

    ChannelPowerStandardizer().testcase()

    NoiseSubspaceDecorrelator().testcase()

    Slicer().testcase()

    BadChannelRemover().testcase()


    quit()

    testcase_pipeline()

    AdaptiveSpatialWhitener().testcase()

    FFTfilter().testcase()
    Resampler().testcase()
    ChannelPowerStandardizer().testcase()
    SpatialWhitener().testcase()


    TargetEncoder().testcase()

    BlockCovarianceMatrixizer().testcase()

    ButterFilterAndResampler().testcase()

    TemporalDecorrelator().testcase()
    # Log().testcase()
