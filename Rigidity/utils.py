"""
This module provides utility functions for working with rigidity.

Functions
---------
    projectionMatrix
    project
    projectAway
    infRotations

"""
__all__ = ['projectionMatrix', 'project', 'projectAway', 'infRotations']

import numpy as np
import scipy.linalg


def projectionMatrix(X):
    """
    Generate a projection matrix for X.

    Returns an array representing the projection matrix from R^n to the
    subspace spanned by the columns of X.

    Parameters
    ----------
    X : numpy.ndarray
        The matrix for which the column span's projection matrix is
        calculated.

    Returns
    -------
    Y : numpy.ndarray
        The representation of the projection matrix.
    """
    orthX = scipy.linalg.orth(X)
    return orthX@orthX.T


def project(X, V):
    """
    Project the vectors of V onto the column span of X.

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
    Y : numpy.ndarray
        The representation of the basis of the projected vectors.
    """
    P = projectionMatrix(X)
    return scipy.linalg.orth(P@V)


def projectAway(X, V):
    """
    Project the vectors of V away from the column span of X.

    Returns an array representing an orthonormal basis of span(Z), where Z is
    the projection of the columns in V onto the orthogonal complement of the
    column span of X.

    Parameters
    ----------
    X : numpy.ndarray
        The matrix defining the column span to project away from.
    V : numpy.ndarray
        The matrix with columns to project.

    Returns
    -------
    Y : numpy.ndarray
        The representation of the basis of the projected vectors.
    """
    dim = X.shape[0]
    Pa = np.eye(dim) - projectionMatrix(X)
    return scipy.linalg.orth(Pa@V)


def infRotations(d):
    """
    Create a generator for the d-dimensional infinitesimal rotation matrices.

    The infinitesimal rotations matrices are the square matrices P_ij where
    P_ij[i,j] = 1, P_ij[j,i] = -1 and 0 otherwise, for all 1 <= i < j <= d.

    Parameters
    ----------
    d : int
        The dimension of the rotation matrices

    Returns
    -------
    rotations : generator
        The initialised generator object
    """
    assert d > 1, "d must be greater than 1"

    for i in range(0, d):
        for j in range(i+1, d):
            R = np.zeros((d, d))
            R[i, j] = 1
            R[j, i] = -1
            yield R
