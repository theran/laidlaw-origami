"""
Create tools to work with rigidity theory.

Classes:

    AffineTansform
    Framework

Functions:

"""


import numpy as np


class AffineTransform:
    """A class to represent affine transformations.

    ...

    Attributes
    ----------
    linear : numpy.matrix
        dxd matrix for linear component of the transformation
    translation : numpy.matrix
        dx1 translational component of the transformation
    dimension : int
        dimension of the space in which the transformation acts
    """

    def __init__(self, squareMatrix, translationVector):
        """Initialise an affine transformation of the form Ax+B."""
        matShape = np.shape(squareMatrix)
        d = matShape[0]
        assert len(matShape) == 2, "squareMatrix must be two dimensional"
        assert matShape[1] == d, "squareMatrix must be a square matrix"
        assert np.shape(translationVector) == (d, 1),\
            "translationVector must have shape ("+d+", 1)"

        self.dimension = d
        self.linear = squareMatrix
        self.translation = translationVector

    def __call__(self, v, pow=1):
        """Apply the affine transformation to a vector.

        Params:
        v : numpy.matrix
            The vector in which the transformation is applied. Shape (3,1).
        pow : int
            The number of times in which the transformation is applied.
            Negative power gives inverses.

        Returns:
        ans : numpy.matrix
            The result of applying the transformation to v.
        """
        d = self.dimension
        assert isinstance(pow, int), "pow must be an int"
        assert np.shape(v) == (d, 1), "v must have shape ("+d+", 1)"

        ans = v

        if pow >= 1:
            for i in range(pow):
                ans = self.rotation*ans + self.translation
        elif pow <= -1:
            assert self.isInvertible(), "Transformation is not invertible"
            for i in range(abs(pow)):
                ans = (self.rotation**(-1))*(ans - self.translation)
        return ans

    def isInvertible(self):
        """Check if transformation is invertible."""
        return np.linalg.matrix_rank(self.linear) == self.dimension

    def inverse(self):
        """Return the inverse affine transformation if such a transformation exists.

        Returns:
        inverse : AffineTransform
            The inverse of the current affine transformation
        """
        assert self.isInvertible(), "Transformation is not invertible"

        invLinear = self.linear**-1
        invTranslation = -1*self.translation
        return AffineTransform(invLinear, invTranslation)


class Framework:
    """A class to represent infinite, screw periodic frameworks.

    ...

    Attributes
    ----------
    graph : tuple
        graph[0] is a list of vertices in the base graph.
        graph[1] is a dictionary of edge markings. Each key
        is a vertex, and its corresponding value is a list
        of 3-tuple markings that give the end vertex of the
        edge, and the group element in Z^2 of the edge.
    baseConfig : list of numpy.matrix
        List of 3x1 matrices containing the coordinates of
        the 'base' or 'identity' vertices.
    nMax : tuple of int
        The largest group element of Z^2 to display when
        drawing a portion of the infinite framework.
    """
    def __init__(self, G, P, n, T1, T2):
        self.graph = G
        self.baseConfig = P
        self.nMax = n
        self.T1 = T1
        self.T2 = T2

    def vertexLocation(self, r_id, n):
        n1, n2 = n

        r = self.T1(r_id, pow=n1)
        r = self.T2(r, pow=n2)

        return r

    def repeatedConfig(self):
        n1, n2 = self.nMax
        nv = len(self.baseConfig)
        repeatedConfig = np.zeros((nv, n1 + 1, n2 + 1, 3))

        for vertex, coord in enumerate(self.baseConfig):
            for i in range(n1+1):
                for j in range(n2+1):
                    repeatedConfig[vertex, i, j, :] = self.vertexLocation(
                        coord, (i, j))[:]
        return repeatedConfig

    def edgeVector(self, edge, n):
        i, j = edge
        p_i = self.baseConfig[i]
        p_j = self.baseConfig[j]
        Tp_j = self.vertexLocation(p_j, n)

        return Tp_j - p_i

    def rigidityMatrix(self):
        vertices, edges = self.graph
        edgesFlat = [edge for adjList in edges.values() for edge in adjList]
        nEdges = len(edgesFlat)
        nVerts = len(vertices)
        count = 0

        R = np.zeros(shape=(nEdges, 3*nVerts))
        for i in vertices:
            nbs = edges[i]
            for edge in nbs:
                j = edge[0]
                marking = edge[1:]

                edgeVector = self.edgeVector((i, j), marking)
                current = R[count, 3*i: 3*(i+1)]
                R[count, 3*i: 3*(i+1)] = current + edgeVector.transpose()

                rEdgeVector = self.edgeVector((j, i), [-n for n in marking])
                current = R[count, 3*j: 3*(j+1)]
                R[count, 3*j: 3*(j+1)] = current + rEdgeVector.transpose()

                count += 1
        return R
