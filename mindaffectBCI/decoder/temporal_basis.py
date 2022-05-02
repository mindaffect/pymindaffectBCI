import numpy as np


def get_rbf_basis(shape, width=5, step=None, windowed: bool = False):
    """get a radial-basis-function (rbf) basis

    Args:
        shape (list): shape of the nd-array to build the basis for, e.g. [4,3,5] for a 3-dimensional nd-array with 3 elm in the first dim, 4 in the second, and 5 in the third
        width (list|float, optional): width of the basis in each of the dimensions of the shape. Defaults to 5.
        step (list|int, optional): gap between basis centers. Defaults to 1.
        windowed (bool, optional): flag if we use a 'windowed' basis which forces the value to zero at the edges of the array. Defaults to False.

    Returns:
        ndarray: basis_bf mapping, with shape (b,f) = (number of basis, number of imput features = prod(shape))
    """
    if not hasattr(shape, '__iter__'):
        shape = [shape]  # ensure iterable
    if not hasattr(width, '__iter__'):
        width = [width]  # ensure iterable
    if len(width) < len(shape):  # pad shape
        width = width + [width[-1]]*(len(shape)-len(width))
    if step is None: step=1
    if not hasattr(step,'__iter__'): # pad step
        step=[step]*len(width)
    step = [max(1, int(s)) for s in step]  # ensure int
    # get centers of each rbf
    if windowed:
        # remove edge centers to allow trend to 0 at edges
        cents_bd = np.meshgrid(*(np.arange(s, max(l-s, s+1), s) for l, s in zip(shape, step)), indexing='ij')
        cents_bd = np.stack(cents_bd, -1).reshape((-1, len(shape)))
    else:
        cents_bd = np.meshgrid(*(np.arange(0, l, s) for l, s in zip(shape, step)), indexing='ij')
        cents_bd = np.stack(cents_bd, -1).reshape((-1, len(shape)))
    # get coords of each point in the space
    pts_fd = np.meshgrid(*(np.arange(l) for l in shape), indexing='ij')  # shape,d
    pts_fd = np.stack(pts_fd, -1).reshape((-1, len(shape)))
    # compute the basis mapping, i.e. weight of each center at each point in the space
    basis_bf = np.exp(-.5*np.sum((pts_fd[np.newaxis, :, :] - cents_bd[:, np.newaxis, :])
                      ** 2/np.maximum(np.array(width)**2, 1e-8), -1))
    # normalize
    basis_bf = basis_bf / np.sqrt(np.sum(basis_bf*basis_bf, -1, keepdims=True))
    return basis_bf


def get_fourier_basis(shape, maxcycles=None, mincycles=1):
    """get a fourier (sin,cos) basis

    Args:
        shape (int): the length of the basis in samples 
        maxcycles (_type_, optional): min number cycles. Defaults to shape//2.
        mincycles (int, optional): max number of cycles. Defaults to 1.

    Returns:
        ndarray: basis_bf mapping, with shape (b,f) = (number of basis = 2*(maxcycles-mincycles), number of imput features = prod(shape))
    """
    if maxcycles is None:
        maxcycles = shape//2
    if mincycles is None:
        mincycles = 1
    basis_bf = []
    for n_cycles in range(mincycles, maxcycles):
        basis_bf.append(np.sin(np.linspace(0, 2*np.pi, shape)*n_cycles))
        basis_bf.append(np.cos(np.linspace(0, 2*np.pi, shape)*n_cycles))
    basis_bf = np.stack(basis_bf, 0)
    # unit-norm each basis
    basis_bf = basis_bf / np.sqrt(np.sum(basis_bf*basis_bf, -1, keepdims=True))
    return basis_bf


def get_edge_suppression_window(shape, edge_fraction):
    """get a basis which forces the values at the edges of the array towards zero.

    Args:
        shape (_type_): length of the basis in samples
        edge_fraction (_type_): fraction of the shape for the start/finish ramp to 0

    Returns:
        _type_: _description_
    """
    temporal_basis = np.ones(shape, dtype=np.float32)
    edge_fraction = int(edge_fraction*shape)
    if edge_fraction > 0:
        # make a edge_frac edge suppression - trapzoid window
        # TODO[]: raise cosine / smoothed version
        temporal_basis[:edge_fraction] = np.linspace(0, 1, edge_fraction+1)[1:]
        temporal_basis[-edge_fraction:] = np.linspace(1, 0, edge_fraction+1)[:-1]
    return temporal_basis


def get_step_basis(shape):
    """ get a basis which consists of a series of step-functions

    Args:
        shape (_type_): length of the basis in samples

    Returns:
        _type_: _description_
    """
    basis_bf = np.zeros((shape, shape), dtype=np.float32)
    for i, b in enumerate(basis_bf):
        b[i:] = 1
    # unit-norm each basis
    basis_bf = basis_bf / np.sqrt(np.sum(basis_bf*basis_bf, -1, keepdims=True))
    return basis_bf


def get_hinge_basis(shape):
    """get a basis which consists of a series of hinge-functions

    Args:
        shape (_type_): length of the basis in samples

    Returns:
        _type_: _description_
    """
    basis_bf = np.zeros((shape, shape), dtype=np.float32)
    for i, b in enumerate(basis_bf):
        b[i:] = np.linspace(0, 1, len(b)-i+1)[1:]
    # unit-norm each basis
    basis_bf = basis_bf / np.sqrt(np.sum(basis_bf*basis_bf, -1, keepdims=True))
    return basis_bf


def get_temporal_basis(tau: int, basis_type: str, dtype=np.float32) -> np.ndarray:
    """get a temporal basis from a basis definition string

    As temporal basis is an array mapping from a parameter basis, such as the amplitude of a particular sinusoidal frequency, to a temporal basis, which is the raw signal amplitude at each time point.
    As such a temporal basis represents a particular paramterization of a time-series.  Useful such time series are the fourier-basis (which have limited spectral content), or the radial-basis-function basis which have a defined smoothness.
    This function will compute a (b,f) matrix which maps from the temporal basis in terms of samples to the parameter basis in terms of parameter amplitudes.

    Args:
        tau (int): length of the basis to generate
        basis_type (str): string basis name, or given basis.  mindaffectBCI.decoder.temporal_basis for a list of supported basis types.  Currently this includes: rbf, fourier, step, hinge, ndrbf
        dtype ([type], optional): number type for the computed basis. Defaults to np.float32.

    Raises:
        ValueError: [description]

    Returns:
        np.ndarray: basis_bt matrix mapping from time, t, to basis parameter space, b 
    """
    temporal_basis_bt = None

    # treat 2-d as stack of 1d basis
    if hasattr(tau, '__iter__') and len(tau) == 2 and \
        not (isinstance(basis_type, str) and
             (basis_type[0] in '23456789' and basis_type[1] == 'd')):  # not ?d basis
        raw_bt = get_temporal_basis(tau[-1], basis_type, dtype)
        if raw_bt is None:
            return raw_bt
        basis_bt = np.zeros((tau[0], raw_bt.shape[0], tau[0], tau[1]), dtype=raw_bt.dtype)
        for i in range(tau[0]):
            basis_bt[i, :, i, :] = raw_bt
        basis_bt = basis_bt.reshape((basis_bt.shape[0]*basis_bt.shape[1], basis_bt.shape[2]*basis_bt.shape[3]))
        return basis_bt

    if isinstance(basis_type, np.ndarray):
        temporal_basis_bt = basis_type.astype(dtype)

    elif isinstance(basis_type, str):
        if basis_type.endswith('.json'):  # load as matrix from file
            import json
            import os
            # search for the file location
            fname = basis_type  # here
            if not os.path.exists(fname):  # source route
                fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), basis_type)
            if not os.path.exists(fname):  # source route resources directory
                fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', basis_type)
            with open(fname, 'r') as f:
                temporal_basis_bt = json.load(f)
            temporal_basis_bt = np.array(temporal_basis_bt)

        elif basis_type.startswith('edge_suppress'):
            try:
                edge_frac = float(basis_type[len('edge_suppress'):])
            except:
                edge_frac = .1
            temporal_basis_bt = get_edge_suppression_window(tau, edge_frac)

        elif basis_type.startswith('rbf') or basis_type.startswith('wrbf'):
            try:
                if basis_type.startswith('rbf'):
                    widths = basis_type[len('rbf'):]
                else:
                    widths = basis_type[len('wrbf'):]
                widths = [float(e) for e in widths.replace(',', '_').split('_')]
            except:
                widths = [5]
            width, step = widths[:2] if len(widths) > 1 else (widths[0], None)
            temporal_basis_bt = get_rbf_basis(tau, width, step=step, windowed=('wrbf' in basis_type))

        elif basis_type.startswith('2drbf') or basis_type.startswith('2dwrbf'):
            try:
                if basis_type.startswith('2drbf'):
                    widths = basis_type[len('2drbf'):]
                else:
                    widths = basis_type[len('2dwrbf'):]
                widths = [float(e) for e in widths.replace(',', '_').split('_')]
            except:
                widths = [5]
            if len(widths) > 2:
                if len(widths) == 4 and widths[0] == tau[0]:
                    width, step = widths[2:], None
                else:
                    width, step = widths[:2], widths[2:]
            else:
                width, step = widths, None
            temporal_basis_bt = get_rbf_basis(tau, width, step=step, windowed=('wrbf' in basis_type))

        elif basis_type[1:].startswith('drbf') or basis_type[1:].startswith('dwrbf'):
            try:
                widths = basis_type[len('?dwrbf'):] if 'wrbf' in basis_type else basis_type[len('?drbf'):]
                ndim = int(basis_type[0])
                widths = [float(e) for e in widths.replace(',', '_').split('_')]
            except:
                ndim = len(tau)
                widths = [1]*ndim
            shape, width, step = widths[:ndim], widths[ndim:2*ndim], widths[2*ndim:]
            # pad missing entries
            width = width + width[-1:]*(ndim-len(width)) if len(width) > 0 else None
            step = step + step[-1:]*(ndim-len(step)) if len(step) > 0 else None
            # infer negative entry
            if any([l == -1 for l in shape]):
                i = shape.index(-1)
                shape[i] = np.prod(tau) // np.prod([l for l in shape if l >= 0])

            temporal_basis_bt = get_rbf_basis(shape, width, step=step, windowed=('wrbf' in basis_type))

        elif basis_type.startswith('fourier') or basis_type.startswith('f'):
            try:
                if basis_type.startswith('fourier'):
                    cycless = basis_type[len('fourier'):]
                else:
                    cycless = basis_type[len('f'):]
                cycless = [int(e) for e in cycless.replace(',', '_').split('_')]
                mincycles, maxcycles = cycless[:2] if len(cycless) > 1 else (1, cycless[0])
            except:
                mincycles, maxcycles = None, None
            temporal_basis_bt = get_fourier_basis(tau, maxcycles, mincycles)

        elif basis_type.startswith('winfourier') or basis_type.startswith('wf'):
            try:
                if basis_type.startswith('wf'):
                    cycless = basis_type[len('wf'):]
                else:
                    cycless = basis_type[len('winfourier'):]
                cycless = [int(e) for e in cycless.replace(',', '_').split('_')]
                mincycles, maxcycles = cycless[:2] if len(cycless) > 1 else (None, cycless[0])
            except:
                mincycles, maxcycles = None, None
            temporal_basis_bt = get_fourier_basis(tau, maxcycles, mincycles)
            window = get_edge_suppression_window(tau, .1)
            temporal_basis_bt = temporal_basis_bt * window
            # unit-norm each basis
            temporal_basis_bt = temporal_basis_bt / np.sqrt(
                np.sum(temporal_basis_bt*temporal_basis_bt, -1, keepdims=True))

        elif basis_type.startswith('step'):
            temporal_basis_bt = get_step_basis(tau)

        elif basis_type.startswith('hinge'):
            temporal_basis_bt = get_hinge_basis(tau)

        elif basis_type.lower() == 'none':
            temporal_basis_bt = None

        else:
            raise ValueError("Unknown temporal basis {}".format(basis_type))

    if temporal_basis_bt is not None and not hasattr(tau, '__iter__') and temporal_basis_bt.shape[-1] > tau:
        print("Warning: basis has longer response than asked for tau, truncated")
        temporal_basis_bt = temporal_basis_bt[..., :tau]

    return temporal_basis_bt


def apply_temporal_basis_cov(temporal_basis, Cxx_dd, Cyx_yetd, Cyy_yetet):
    """apply the given temporal basis function to the data-summary statistics (covariances) to transform from a raw-sample basis to the given parameter basis.

    Args:
        temporal_basis (str|ndarray): the temporal basis mapping to use. 
        Cxx_dd (_type_): Spatial covariance of the data with d-channels/electrodes.  This has shape (d,d) = (nElectrodes, nElectrodes)
        Cyx_yetd (_type_): Spatio-temporal cross-covariance between the data (X_Td) and the stimuli (Y_Tye).   This has shape (y,e,t,d) = (nOuputs, num-event-type, response-length tau, n-electrodes)
        Cyy_yetet (ndarray): Temporal auto-covariance of the stimulus.  This has shape (y,e,t,e,t) = (number outputs, number stimulus event types, response length tau, number stimulus event types, response length tau)

    Returns:
        Cxx_dd (_type_): Spatial covariance of the data with d-channels/electrodes.  This has shape (d,d) = (nElectrodes, nElectrodes)
        Cyx_yebd (_type_): Spatio-temporal cross-covariance between the data (X_Td) and the stimuli (Y_Tye).   This has shape (y,e,b,d) = (nOuputs, num-event-type, transformed parameter basis, n-electrodes)
        Cyy_yebeb (ndarray): Temporal auto-covariance of the stimulus.  This has shape (y,e,t,e,t) = (number outputs, number stimulus event types, transformed parameter basis, number stimulus event types, transformed parameter basis)
        temporal_basis_bf (str|ndarray): matrix mapping from time, t, to basis parameter space, b.
    """
    if isinstance(temporal_basis, str) and temporal_basis[:2] in ('2d', '3d', '4d', '5d', '6d', 'nd'):
        # mapping over events and time
        tb_b_et = get_temporal_basis(Cyx_yetd.shape[1:-1], temporal_basis, Cxx_dd.dtype)
        tb_bet = tb_b_et.reshape((tb_b_et.shape[0],)+Cyx_yetd.shape[1:-1])
        if tb_bet is not None:
            Cyx_ybd = np.einsum('yetd,bet->ybd', Cyx_yetd, tb_bet)
            Cyy_ybb = np.einsum('bet,yetfu,cfu->ybc', tb_bet, Cyy_yetet, tb_bet, optimize=True)
            # put singlenton event-type dims back in to make ndim confirm to expectations
            Cyx_yetd = Cyx_ybd.reshape((Cyx_ybd.shape[0], 1)+Cyx_ybd.shape[1:])
            Cyy_yetet = Cyy_ybb.reshape((Cyy_ybb.shape[0], 1, Cyy_ybb.shape[1], 1, Cyy_ybb.shape[2]))
        return Cxx_dd, Cyx_yetd, Cyy_yetet, tb_bet
    else:
        tb_bt = get_temporal_basis(Cyx_yetd.shape[-2], temporal_basis, Cxx_dd.dtype)
        if tb_bt is not None:
            if tb_bt.ndim == 1:
                Cyx_yetd = Cyx_yetd * tb_bt[:, np.newaxis]
                Cyy_yetet = tb_bt[:, np.newaxis, np.newaxis] * Cyy_yetet * tb_bt
            elif tb_bt.ndim == 2:
                Cyx_yetd = np.einsum("yetd,bt->yebd", Cyx_yetd, tb_bt)
                Cyy_yetet = np.einsum("bt,yetfu,cu->yebfc", tb_bt, Cyy_yetet, tb_bt)
            elif tb_bt.ndim == 3:
                Cyx_yetd = np.einsum("yetd,bet->yebd", Cyx_yetd, tb_bt[:, :Cyx_yetd.shape[1], :])
                Cyy_yetet = np.einsum(
                    "bet,yetfu,ceu->yebfc", tb_bt[:, : Cyx_yetd.shape[1],
                                                  :],
                    Cyy_yetet, tb_bt[:, : Cyx_yetd.shape[1],
                                     :])
        return Cxx_dd, Cyx_yetd, Cyy_yetet, tb_bt


apply_temporal_basis = apply_temporal_basis_cov


def apply_temporal_basis_X_TStd(temporal_basis, X_TStd):
    """apply the given temporal basis function directly to the input raw channel data

    Args:
        temporal_basis (str|ndarray): the temporal basis mapping to use. 
        X_TStd (ndarray): the raw input data. shape is (T,S,t,d) = (number trials, number samples per trial, number samples in response tau, number electrodes)

    Returns:
        X_TStd (ndarray): the raw input data. shape is (T,S,t,d) = (number trials, number samples per trial, number parameters in the basis, number electrodes)
    """
    tb_bt = get_temporal_basis(X_TStd.shape[-2], temporal_basis, X_TStd.dtype)
    if tb_bt is not None:
        if tb_bt.ndim == 1:
            X_TStd = X_TStd * tb_bt[:, np.newaxis]
        elif tb_bt.ndim == 2:
            X_TStd = np.einsum("TStd,bt->TSbd", X_TStd, tb_bt)
        elif tb_bt.ndim > 2:
            raise ValueError("Per-event type temporal basis not supproted with raw features")
    return X_TStd


def apply_temporal_basis_tdtd(temporal_basis, C_tdtd):
    tb_bt = get_temporal_basis(C_tdtd.shape[-2], temporal_basis, C_tdtd.dtype)
    if tb_bt is not None:
        if tb_bt.ndim == 1:
            C_tdtd = tb_bt[:, np.newaxis, np.newaxis, np.newaxis] * C_tdtd * tb_bt[:, np.newaxis]
        elif tb_bt.ndim == 2:
            C_tdtd = np.einsum("bt,tdue,cu->bdce", tb_bt, C_tdtd, tb_bt)
        elif tb_bt.ndim > 2:
            raise ValueError("Per-event type temporal basis not supproted with raw features")
    return C_tdtd


def apply_temporal_basis_tde(temporal_basis, C_tde):
    tb_bt = get_temporal_basis(C_tde.shape[0], temporal_basis, C_tde.dtype)
    if tb_bt is not None:
        if tb_bt.ndim == 1:
            C_tde = C_tde * tb_bt[:, np.newaxis, np.newaxis]
        elif tb_bt.ndim == 2:
            C_tde = np.einsum("tde,bt->bde", C_tde, tb_bt)
        elif tb_bt.ndim > 2:
            raise ValueError("Per-event type temporal basis not supproted with raw features")
    return C_tde


def invert_temporal_basis_mapping(tb_bt, R_Mket):
    """apply the temporal basis mapping to the temporal impulse response function defined in the basis-parameter space to generate the equivalent activation defined in terms of raw sample amplitudes
    N.B. map a weighting defined in basis parameter space back to sample space

    Args:
        tb_bt (ndarray): basis_bt matrix mapping from time, t, to basis parameter space, b 
        R_Mkeb (ndarray): Stimulus impulse response estimate, with shape (M,k,e,t)= (number of models, number of neural sources, number of stimulus event types, temporal parameter basis weights)

    Returns:
        R_Mket (ndarray): Stimulus impulse response estimate, with shape (M,k,e,t)= (number of models, number of neural sources, number of stimulus event types, response length tau)
    """
    if tb_bt is not None:  # map from feature space back to tau for direct application
        if tb_bt.ndim == 1:  # window function
            R_Mket = R_Mket * tb_bt
        elif tb_bt.ndim == 2:  # basis transformation
            R_Mket = np.einsum("...b,bt->...t", R_Mket, tb_bt)
        elif tb_bt.ndim == 3:
            R_Mket = np.einsum("...zb,bet->...et", R_Mket, tb_bt)
        elif tb_bt.ndim == 4:  # combined event/temporal mapping
            R_Mket = np.einsum("...fu,fuet->...et", R_Mket, tb_bt)
    return R_Mket


def invert_temporal_basis_mapping_spatiotemporal(tb_bt, W_etd):
    if tb_bt is not None:  # map from feature space back to tau for direct application
        if tb_bt.ndim == 1:  # window function
            W_etd = W_etd * tb_bt[:, np.newaxis]
        elif tb_bt.ndim == 2:  # basis transformation
            W_etd = np.einsum("...bd,bt->...td", W_etd, tb_bt)
        elif tb_bt.ndim == 3:
            raise ValueError("Per-event type temporal basis not supproted with spatial-temporal weights")
    return W_etd
