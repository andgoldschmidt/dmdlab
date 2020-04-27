import numpy as np
from scipy.linalg import logm, expm
from numpy.linalg import svd


def delay_embed(X, shift):
    """Delay-embed the matrix X with measurements from future times.

    Args:
        X (:obj:`ndarray` of float): Data matrix with columns storing states at sequential time measurements.
        shift (int): Number of future times copies to augment to the current time state.

    Returns:
        (:obj:`ndarray` of float): The function maps (d, t) to (shift+1, d, t-shft) which is then stacked
            into ((shift+1)*d, t-shift).
    """
    if X.ndim != 2:
        raise ValueError('In delay_embed, invalid X matrix shape of ' + str(X.shape))
    _, T = X.shape
    return np.vstack([X[:, i:(T - shift) + i] for i in range(shift + 1)])


def dag(X):
    # Conjugate transpose (dagger) shorthand
    return X.conj().T


def dst_from_cts(cA, cB, dt):
    """
    Convert constant continuous state space matrices to discrete
    matrices with time step dt using:

    exp(dt*[[cA, cB],  = [[dA, dB],
            [0,  0 ]])    [0,  1 ]]

    Require cA in R^(na x na) and cB in R^(na x nb). The zero and
    identity components make the matrix square.

    Args:
        cA (:obj:`ndarray` of float): Continuous A.
        cB (:obj:`ndarray` of float): Continuous B.
        dt (float): Time step.

    Returns:
        (tuple): tuple containing:
            (:obj:`ndarray` of float): discrete A.
            (:obj:`ndarray` of float): discrete B.
    """
    na, _ = cA.shape
    _, nb = cB.shape
    cM = np.block([[cA, cB],
                   [np.zeros([nb, na]), np.zeros([nb, nb])]])
    dM = expm(cM * dt)
    return dM[:na, :na], dM[:na, na:]


def cts_from_dst(dA, dB, dt):
    """
    Convert discrete state space matrices with time step dt to
    continuous matrices by inverting

    exp(dt*[[cA, cB],  = [[dA, dB],
            [0,  0 ]])    [0,  1 ]]

    Require dA in R^(na x na) and dB in R^(na x nb). The zero and
    identity components make the matrix square.

    Args:
        dA (:obj:`ndarray` of float): discrete A.
        dB (:obj:`ndarray` of float): discrete B.
        dt (float): Time step.

    Returns:
        (tuple): tuple containing:
            (:obj:`ndarray` of float): Continuous A.
            (:obj:`ndarray` of float): Continuous B.
    """
    na, _ = dA.shape
    _, nb = dB.shape
    dM = np.block([[dA, dB],
                   [np.zeros([nb, na]), np.identity(nb)]])
    cM = logm(dM)/dt
    return cM[:na, :na], cM[:na, na:]


def _threshold_svd(X, threshold, threshold_type):
    """
    Args:
        X:
            Matrix for SVD
        threshold: Pos. real, int, or None
            Truncation value for SVD results
        threshold_type: 'percent', 'count'
            Type of truncation, ignored if threshold=None'

    Returns:
        (tuple): Tuple containing U,S,Vt of a truncated SVD
    """
    U, S, Vt = svd(X, full_matrices=False)
    if threshold_type == 'percent':
        r = np.sum(S / np.max(S) > threshold)
    elif threshold_type == 'count':
        r = threshold
    else:
        raise ValueError('Invalid threshold_type.')
    return U[:, :r], S[:r], Vt[:r, :]