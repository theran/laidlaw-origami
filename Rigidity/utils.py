import numpy as np
import scipy.linalg


def proj_colsp_matrix(X):
    XX = scipy.linalg.orth(X)
    return XX@XX.T


def proj_colsp(X, V):
    P = proj_colsp_matrix(X)
    return scipy.linalg.orth(P@V)


def proj_colsp_away_matrix(X):
    m = X.shape[0]
    return np.eye(m) - proj_colsp_matrix(X)


def proj_colsp_away(X, V):
    Pa = proj_colsp_away_matrix(X)
    return scipy.linalg.orth(Pa@V)
