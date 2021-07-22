"""
This module provides utility functions for working with `rigidity`.

Functions
---------
    projectionMatrix
    project
    projectAway
    orth
    infRotations

"""
__all__ = ['projectionMatrix', 'project', 'projectAway', 'orth',
           'infRotations'
           ]

import numpy as np
import scipy.linalg


def projectionMatrix(X):
    """
    Generate a projection matrix for `X`.

    Returns an array representing the projection matrix from R^n to the
    subspace spanned by the columns of `X`.

    Parameters
    ----------
    X : numpy.ndarray
        The matrix for which the column span's projection matrix is
        calculated.

    Returns
    -------
    numpy.ndarray
        The representation of the projection matrix.
    """
    orthX = orth(X)
    return orthX@orthX.T


def project(X, V):
    """
    Project the vectors of `V` onto the column span of `X`.

    Returns an array representing an orthonormal basis of span(Z), where Z is
    the projection of the columns in V onto the column span of X.

    Parameters
    ----------
    X : numpy.ndarray
        The matrix defining the column span to project onto.
    V : numpy.ndarray
        The matrix with columns to project.

    Returns
    -------
    numpy.ndarray
        The representation of the basis of the projected vectors.
    """
    P = projectionMatrix(X)
    return orth(P@V)


def projectAway(X, V):
    """
    Project the vectors of `V` away from the column span of `X`.

    Returns an array representing an orthonormal basis of span(Z), where Z is
    the projection of the columns in V onto the orthogonal complement of the
    column span of `X`.

    Parameters
    ----------
    X : numpy.ndarray
        The matrix defining the column span to project away from.
    V : numpy.ndarray
        The matrix with columns to project.

    Returns
    -------
    numpy.ndarray
        The representation of the basis of the projected vectors.
    """
    dim = X.shape[0]
    Pa = np.eye(dim) - projectionMatrix(X)
    return orth(Pa@V)


def orth(A, rcond=None, absTol=1e-6):
    """
    Construct an orthonormal basis for the column span of A.

    Parameters
    ----------
    A : numpy.ndarray
        Input array
    rcond : float, optional
        Relative condition number. Singular values `s` in the singular value
        decomposition smaller than `rcond * max(s)` are considered zero.
        Default: floating point eps * max(M,N).
    absTol : float, optional
        Absolute tolerance. Singular values `s` in the singular value
        decomposition smaller than `absTol` are considered zero.
        Default: 1e-6.

    See also
    --------
    scipy.linalg.orth

    Notes
    -----
    This is a variation on the scipy.linalg implementation of orth. As well as
    having a minimum tolerance dependent on the contnt of the SVD, we add the
    option to specify an absolute tolerance below which singular values are
    considered zero.
    """
    u, s, vh = scipy.linalg.svd(A, full_matrices=False)
    M, N = u.shape[0], vh.shape[1]

    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)

    relTol = np.amax(s) * rcond
    tol = max(relTol, absTol)

    num = np.sum(s > tol, dtype=int)
    Q = u[:, :num]

    if Q.shape[1] == 0:
        Q = None
    return Q


def infRotations(d):
    """
    Create a generator for the `d`-dimensional infinitesimal rotation matrices.

    The infinitesimal rotations matrices are the square matrices P_ij where
    P_ij[i,j] = 1, P_ij[j,i] = -1 and 0 otherwise, for all 1 <= i < j <= `d`.

    Parameters
    ----------
    d : int
        The dimension of the rotation matrices

    Yields
    -------
    rotation : numpy.ndarray
        Infinitesimal rotation matrix
    """
    assert d > 1, "d must be greater than 1"

    for i in range(0, d):
        for j in range(i+1, d):
            R = np.zeros((d, d))
            R[i, j] = 1
            R[j, i] = -1
            yield R
