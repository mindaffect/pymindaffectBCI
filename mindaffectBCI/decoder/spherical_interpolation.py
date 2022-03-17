from os import remove
import numpy as np
from numpy.polynomial.legendre import legval
from scipy.special import sph_harm


def calc_G(cosang, stiffness: int = 4, n_legendre_terms: int = 50):
    """Calculate spherical spline G function between points on a sphere.
    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffness of the spline.
    n_legendre_terms : int
        number of Legendre terms to evaluate.
    Returns
    -------
    G : np.ndrarray of float, shape(n_channels, n_channels)
        The G matrix.
    """
    factors = [
        (2 * n + 1) / (n**(stiffness) * (n + 1)**(stiffness) * 4 * np.pi)
        for n in range(1, n_legendre_terms + 1)
    ]
    return legval(cosang, [0] + factors)


def make_spherical_spline_interpolation_matrix(from_nd,
                                               to_md,
                                               alpha=1e-5,
                                               stiffness: int = 4,
                                               remove_bias: bool = False):
    """Compute interpolation matrix based on spherical splines.
    Implementation based on [1]
    Parameters
    ----------
    from_nd : np.ndarray of float, shape(N, 3)
        The positions to interpoloate from. (source)
    to_Md : np.ndarray of float, shape(n_bad_sensors, 3)
        The positions to interpoloate to. (dest)
    alpha : float
        Regularization parameter. Defaults to 1e-5.
    remove_bias: bool
        if true then discard the constant spline coefficient, i.e. the average response
    Returns
    -------
    A_nd : np.ndarray of float, shape(N,M)
        matrix mapping from -> to
    References
    ----------
    [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
        Spherical splines for scalp potential and current density mapping.
        Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
    """
    s_nd = from_nd.copy()
    d_md = to_md.copy()

    # enusure these are unit-vectors
    s_nd = s_nd / np.sqrt(np.sum(s_nd * s_nd, -1, keepdims=True))
    d_md = d_md / np.sqrt(np.sum(d_md * d_md, -1, keepdims=True))

    # cosine angles between source positions
    cosss_nn = s_nd @ s_nd.T
    cosds_mn = d_md @ s_nd.T

    # spherical spline matrices within and between positions
    Gss_nn = calc_G(cosss_nn)
    Gds_mn = calc_G(cosds_mn)

    # add regularization parameter to the diagonal of Gaa
    if alpha is not None:
        Gss_nn.flat[::Gss_nn.shape[0] + 1] += alpha

    # add constant bias term to make the full matrix
    nrm = np.mean(np.diagonal(Gss_nn))
    Gss_n1n1 = np.vstack([
        np.hstack([Gss_nn, np.ones((Gss_nn.shape[0], 1)) * nrm]),
        np.hstack([np.ones((1, Gss_nn.shape[0])) * nrm, [[0]]])
    ])
    Gds_mn1 = np.hstack([Gds_mn, np.ones((Gds_mn.shape[0], 1)) * nrm])

    # Compute (least-squares) mapping from input to spherical-spline basis with knot at each point in a, by pinv
    iGss_n1n1 = np.linalg.pinv(Gss_n1n1)

    # Compute (least-squares) mapping from input to output by: in->spline->out
    if remove_bias:
        Ads_mn = Gds_mn1[:, :-1] @ iGss_n1n1[:-1, :-1]
    else:
        Ads_mn = Gds_mn1 @ iGss_n1n1[:, :-1]
    return Ads_mn


def make_surface_lapacian_matrix(from_nd,
                                 to_md=None,
                                 alpha=1e-5,
                                 stiffness: int = 4):
    """Compute surface laplacian matrix based on spherical splines interpolation and differientiation
    Implementation based on [1]
    Parameters
    ----------
    from_nd : np.ndarray of float, shape(N, 3)
        The positions to interpoloate from.
    to_Md : np.ndarray of float, shape(n_bad_sensors, 3)
        The positions to interpoloate.
    alpha : float
        Regularization parameter. Defaults to 1e-5.
    Returns
    -------
    A_nd : np.ndarray of float, shape(N,M)
        matrix mapping from -> to
    References
    ----------
    [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
        Spherical splines for scalp potential and current density mapping.
        Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
    """
    s_nd = from_nd.copy()
    d_md = to_md.copy() if to_md is not None else from_nd.copy()

    # enusure these are unit-vectors
    s_nd = s_nd / np.sqrt(np.sum(s_nd * s_nd, -1, keepdims=True))
    d_md = d_md / np.sqrt(np.sum(d_md * d_md, -1, keepdims=True))

    # cosine angles between source positions
    cosss_nn = s_nd @ s_nd.T
    cosds_mn = d_md @ s_nd.T

    # spherical spline matrices within and between positions
    Gss_nn = calc_G(cosss_nn, stiffness=4)
    Hds_mn = calc_G(cosds_mn, stiffness=3)

    # add regularization parameter to the diagonal of Gaa
    if alpha is not None:
        Gss_nn.flat[::Gss_nn.shape[0] + 1] += alpha

    # add constant bias term to make the full matrix
    nrm = np.mean(np.diagonal(Gss_nn))
    Gss_n1n1 = np.vstack([
        np.hstack([Gss_nn, np.ones((Gss_nn.shape[0], 1)) * nrm]),
        np.hstack([np.ones((1, Gss_nn.shape[0])) * nrm, [[0]]])
    ])
    Hds_mn1 = np.hstack([Hds_mn, np.ones((Hds_mn.shape[0], 1)) * nrm])

    # Compute (least-squares) mapping from input to spherical-spline basis with knot at each point in a, by pinv
    iGss_n1n1 = np.linalg.pinv(Gss_n1n1)

    # Compute (least-squares) mapping from input to output by: in->spline->out
    Ads_mn = Hds_mn1 @ iGss_n1n1[:, :-1]
    return Ads_mn


def mk_forward_model(sensor=10, src=3):
    sensor_n3 = np.random.uniform(
        -1, 1, size=(sensor, 3)) if np.isscalar(sensor) else sensor
    sensor_n3 = sensor_n3 / np.sqrt(
        np.sum(sensor_n3 * sensor_n3, -1, keepdims=True))
    src_m3 = np.random.uniform(-.5, .3, size=(src,
                                              3)) if np.isscalar(src) else src
    dist2_nm = np.sum(
        (src_m3[:, np.newaxis, :] - sensor_n3[np.newaxis, :, :])**2, -1)
    A_nm = 1 / dist2_nm  # patter is 1/r^2
    return A_nm


def test_spline_self():
    a_n3 = np.random.standard_normal((10, 3))
    a_n3 = a_n3 / np.sqrt(np.sum(a_n3 * a_n3, -1, keepdims=True))
    A_nn = make_spherical_spline_interpolation_matrix(a_n3,
                                                      a_n3,
                                                      remove_bias=False)
    # check if this is, approx, the identity transform
    err = A_nn - np.eye(A_nn.shape[0])
    print("Err={}".format(np.mean(np.abs(err))))
    assert np.mean(np.abs(err)) < 1e-1


def test_spline_interpoltate():
    import matplotlib.pyplot as plt
    a_n3 = np.random.standard_normal((10, 3))
    a_n3 = a_n3 / np.sqrt(np.sum(a_n3 * a_n3, -1, keepdims=True))
    # make test forward model for these sensors
    S_Sm = np.random.standard_normal((1000, 3))
    Fwd_nm = mk_forward_model(sensor=a_n3, src=3)
    X_Sn = S_Sm @ Fwd_nm

    # drop pt 3
    bad_idx = 3
    keep = np.ones(a_n3.shape[0], dtype=bool)
    keep[bad_idx] = False
    b_m3 = a_n3[keep]
    # make interpolation matrix to reconstruct the bad channel
    A_nm = make_spherical_spline_interpolation_matrix(b_m3,
                                                      a_n3,
                                                      remove_bias=False)
    # interpolate
    Xbad_Sm = X_Sn[:, keep]
    Xinterp_Sn = Xbad_Sm @ A_nm.T

    # compare the true and interpolated result
    plt.subplot(311)
    plt.plot(X_Sn)
    plt.subplot(312)
    plt.plot(Xinterp_Sn)
    plt.subplot(313)
    plt.plot(X_Sn[:, bad_idx], label='X')
    plt.plot(Xinterp_Sn[:, bad_idx], label='interp')
    plt.plot(X_Sn[:, bad_idx] - Xinterp_Sn[:, bad_idx], label='diff')
    plt.legend()
    plt.show()

    assert np.sum(np.abs(Xinterp_Sn[:, bad_idx] - X_Sn[:, bad_idx])) / np.sum(
        np.abs(X_Sn[:, bad_idx])) < .4


def test_slap():
    import matplotlib.pyplot as plt
    a_n3 = np.random.standard_normal((10, 3))
    a_n3 = a_n3 / np.sqrt(np.sum(a_n3 * a_n3, -1, keepdims=True))
    # make test forward model for these sensors
    S_Sm = np.random.standard_normal((1000, 3))
    Fwd_nm = mk_forward_model(sensor=a_n3, src=3)
    X_Sn = S_Sm @ Fwd_nm

    # make interpolation matrix to reconstruct the bad channel
    A_nn = make_surface_lapacian_matrix(a_n3)
    # interpolate
    Xinterp_Sn = X_Sn @ A_nn.T

    # compare the true and interpolated result
    plt.subplot(311)
    plt.plot(X_Sn)
    plt.subplot(312)
    plt.plot(Xinterp_Sn)
    plt.show()


if __name__ == '__main__':
    test_spline_self()
    test_spline_interpoltate()
    test_slap()