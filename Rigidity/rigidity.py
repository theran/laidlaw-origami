"""
Create tools to work with rigidity theory.

Classes
-------
AffineTansform
Framework
ScrewFramework
"""


import numpy as np
from scipy import linalg
from sympy import symbols, solve
from sympy import Matrix as symMatrix
import matplotlib.pyplot as plt
import utils


class AffineTransform:
    """
    A class to represent callable affine transformations.

    Parameters
    ----------
    squareMatrix : numpy.ndarray
        (d,d) array for linear component of the transformation.
    translationVector : numpy.ndarray
        (d,1) translational component of the transformation.

    Attributes
    ----------
    linear : numpy.ndarray
        (d,d) array for linear component of the transformation.
    translation : numpy.ndarray
        (d,) translational component of the transformation.
    dimension : int
        Dimension of the space in which the transformation acts.

    Methods
    -------
    isInvertible
        Checks if the transformation is invertible
    inverse
        Returns the inverse of the transformation.
    """

    def __init__(self, squareMatrix, translationVector):
        """Initialise an affine transformation of the form Ax+B."""
        matShape = np.shape(squareMatrix)
        d = matShape[0]

        assert len(matShape) == 2, "squareMatrix must be two dimensional"
        assert matShape[1] == d, "squareMatrix must be a square matrix"
        assert len(translationVector) == d,\
            "translationVector must have shape ("+d+",)"

        self.dimension = d
        self.linear = squareMatrix
        self.translation = translationVector
        self.__invertible = None
        self.__inverse = None

    def __call__(self, v, pow=1):
        """
        Apply the affine transformation to a vector.

        Parameters
        ----------
        v : numpy.ndarray
            The vector in which the transformation is applied. Shape (3,1).
        pow : int, optional
            The number of times in which the transformation is applied.
            Negative power gives inverses, if possible.

        Returns
        -------
        numpy.ndarray
            The result of applying the transformation to v.
        """
        d = self.dimension

        assert isinstance(pow, int), "pow must be an int"
        assert np.shape(v)[0] == d,\
            "v must have have shape (" + str(d) + ", ...)"

        ans = v

        if pow >= 1:
            for i in range(pow):
                ans = self.linear@ans + self.translation
        elif pow <= -1:
            assert self.isInvertible(), "Transformation is not invertible"
            ans = self.inverse()(v, abs(pow))
        return ans

    def isInvertible(self):
        """Check if transformation is invertible."""
        if self.__invertible is None:
            if np.linalg.matrix_rank(self.linear) == self.dimension:
                self.__invertible = True
            else:
                self.__invertible = False
        return self.__invertible

    def inverse(self):
        """
        Return the inverse affine transformation.

        Returns
        -------
        inverse : AffineTransform
            The inverse of the current affine transformation, if such a
            transformation exists.
        """
        assert self.isInvertible(), "Transformation is not invertible"

        if self.__inverse is None:
            invLinear = np.linalg.inv(self.linear)
            invTranslation = -1*self.translation
            self.__inverse = AffineTransform(invLinear, invTranslation)

        return self.__inverse


class Framework:
    """
    A class to represent classical rigidity frameworks.

    Parameters
    ----------
    G : tuple
        graph[0] must be a list of vertices in the graph, of the form
        [0, 1, ..., n].
        graph[1] must be a dictionary of edges. Each key must be  a vertex,
        and its corresponding value a list of edges that vertex is adjacent
        to. In this implementation of a graph, we do not allow loops or
        multiple edges.
    P : numpy.ndarray
        (d,n) array with columns as points of the configuration of the
        framework.

    Attributes
    ----------
    graph : tuple
        graph[0] is a list of vertices in the graph.
        graph[1] is a dictionary of edges.
    config : numpy.ndarray
        (d,n) array with columns as points of the configuration of the
        framework.
    dimension : int
        The dimension of the Euclidean space in which to embed the framework.

    Methods
    -------
    rigidityMatrix
        Returns the numeric rigidity matrix of the framework.
    symbolicRigidityMatrix
        Returns the symbolic rigidity matrix of the framework, making use of
        sympy.Matrix.
    infinitesimalFlex
        Finds the non-trivial infinitesimal flexes of the framework if the
        framework is flexible.
    draw
        Draws the framework using matplotlib.pyplot.
    drawFlex
        Draws a flex on the framework using matplotlib.pyplot.
    nonTrivialFlex
        Calculates an orthonormal basis for the space of non-trivial flexes.
    """

    def __init__(self, G, P):
        """Initialise a framework."""
        verts = G[0]
        edges = G[1]

        # Validate input
        assert type(verts) == list, "Vertices of G must be a list"

        nv = len(verts)
        assert verts == [i for i in range(nv)],\
            "Vertices of G must have the form [0, 1, ..., nv]"

        assert type(edges) == dict, "Edges of G must be a dict"
        for vert, edge in edges.items():
            assert vert in verts, "Edges must comprise of vertices"
            for end in edge:
                assert end in verts, "Edges must comprise of vertices"
            assert list(set(edge)) == edge, "No multiple edges allowed"
            assert vert not in edge, "No loops allowed"

        assert type(P) == np.ndarray, "P must be an array"
        assert P.shape[1] == nv, ("There must be the same number of vertices"
                                  " as points in the configuration")

        if nv == 0:
            d = 0
        else:
            d = len(P)

        self.graph = G
        self.config = P
        self.dimension = d

    def rigidityMatrix(self):
        """
        Generate the numerical rigidity matrix.

        Returns
        -------
        numpy.ndarray
            The rigidity matrix of (G,p).
        """
        verts, edges = self.graph
        edgesFlat = [edge for adjList in edges.values() for edge in adjList]
        nEdges = len(edgesFlat) // 2
        nVerts = len(verts)
        row = 0
        d = self.dimension

        R = np.zeros(shape=(nEdges, d*nVerts))
        for i in verts:
            nbs = edges[i]
            for j in nbs:
                if i < j:
                    edgeVector = self.config[:, i] - self.config[:, j]
                    R[row, d*i: d*(i+1)] = edgeVector
                    R[row, d*j: d*(j+1)] = -1 * edgeVector

                    row += 1
        return R

    def symbolicRigidityMatrix(self, var='p'):
        """
        Create the symbolic rigidity matrix.

        Parameters
        ----------
        var : str, optional
            The variable to use in the rigidity matrix.

        Returns
        -------
        sympy.MutableDenseMatrix
            The rigidity matrix of (G,p).
        """
        verts, edges = self.graph
        edgesFlat = [edge for adjList in edges.values() for edge in adjList]
        nEdges = len(edgesFlat) // 2
        nVerts = len(verts)
        d = self.dimension
        row = 0

        R = symMatrix.zeros(nEdges, nVerts*d)
        for i in verts:
            nbs = edges[i]
            coords = var + '_' + str(i) + r'\,:' + str(d)
            pi = np.array(symbols(coords))
            for j in nbs:
                if i < j:
                    coords = var + '_' + str(j) + r'\,:' + str(d)
                    pj = np.array(symbols(coords))

                    edgeSymb = (pi - pj).reshape(1, d)
                    R[row, i*d:(i+1)*d] = edgeSymb
                    R[row, j*d:(j+1)*d] = -1 * edgeSymb

                    row += 1
        return R

    def infinitesimalFlex(self):
        """
        Calculate a non-trivial infinitesimal flex if such a flex exists.

        Returns
        -------
        flex : list
            A non-trivial infinitesimal flex if one exists, else all zeros.
        """
        verts, edges = self.graph
        nVerts = len(verts)
        d = self.dimension
        P = symMatrix(self.config)

        bars = [P[:, i]-P[:, j] for i in verts for j in edges[i] if i < j]

        # Create a list of symbols that will represent our flex
        flex = []
        for i in range(nVerts):
            col = symMatrix([symbols('x' + str(i) + str(k)) for k in range(d)])
            flex.append(col)

        # Create a list of constraints that fix the first point in the
        # configuration, then constrains the second flex to be in the line
        # between p0 and p1, the third flex to lie in the plan# defined by p0,
        # p1 and p2 and so on. This is possible if n > dim and the
        # configuration is in general position. We do this by finding the
        # nullspace of the matrix of vectors that define the hyperplane, which
        # gives a normal vector. To ensure the flex remains in the hyperplane,
        # we enforce that the dot product between the normal and the flex is
        # zero.
        nonTrivConstraints = []
        for entry in flex[0]:
            nonTrivConstraints.append(entry)

        p0 = P[:, 0]
        for i in range(1, d):
            vects = [p0 - pi for pi in P[:, 1:i+1].T]
            fixedHypPlane = symMatrix([v.transpose() for v in vects])
            normal = fixedHypPlane.nullspace()[0]

            constraint = flex[i].dot(normal)
            nonTrivConstraints.append(constraint)

        # Solve the system and apply the constraints to flex
        nonTrivSoln = solve(nonTrivConstraints)
        flex = [delta.subs(nonTrivSoln) for delta in flex]

        # Create the constraints necessary for infinitesimal rigidity
        flexEqns = [flex[i]-flex[j] for i in verts for j in edges[i] if i < j]
        rigidConstraints = [bar.dot(flexEqns[i]) for i, bar in enumerate(bars)]

        # Solve the system
        soln = solve(rigidConstraints, dict=True)
        flex = [delta.subs(soln[0]) for delta in flex]

        return flex

    def draw(self, labels=False, fmt='-k', EKwargs={}, VKwargs={}):
        """
        Draw the framework with given configuration.

        Parameters
        ----------
        labels : bool, optional
            Whether to draw vertices with or without labels.
            Default : False
        fmt : str, optional
            Format string for drawing edges of the form
            '[marker][line][colour]'.
        EKwargs, VKwargs : dict, optional
            Key word arguments used in the drawing of edges and vertices.

        Returns
        -------
        matplotlib.figure.Figure
            Figure on which the configuration is drawn.

        See also
        --------
        matplotlib.pyplot.plot

        matplotlib.pyplot.scatter
        """
        d = self.dimension
        assert d == 2 or d == 3,\
            "Unfortunately, we cannot draw in dimensions higher than 3"

        fig = plt.figure()

        if d == 2:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection='3d')

        verts, edges = self.graph
        P = self.config

        for i in verts:
            for j in edges[i]:
                if i < j:
                    ri = P[:, i]
                    rj = P[:, j]
                    edge = np.column_stack([ri, rj])
                    ax.plot(*edge, fmt, **EKwargs)

        ax.scatter(*P, **VKwargs)

        if labels:
            Q = P + 0.02
            name = 0
            for vertex in Q.T:
                ax.text(*vertex, str(name))
                name += 1
        return fig

    def drawFlex(self, flex, fig=None):
        """
        Draw a flex on the framework.

        Parametters
        -----------
        flex : numpy.ndarray
            (d*n, ) array representing the flex.
        fig : matplotlib.figure.Figure, optional
            The figure to add the flex drawing to.

        Returns
        -------
        matplotlib.figure.Figure
            The figure on which the flex is added to.
        """
        if fig is None:
            fig = self.draw()
        ax = fig.get_axes()[0]

        d = self.dimension
        nv = len(self.graph[0])
        P = self.config

        vectors = np.reshape(flex, (d, nv), 'F')

        ax.quiver(*P, *vectors)
        return fig

    def nonTrivialFlex(self):
        """
        Calculate a basis for the non-trivial flexes of the framework.

        Returns
        -------
        M : numpy.ndarray
            An orthonormal basis for the non-trivial flexes
        """
        P = self.config
        R = self.rigidityMatrix()
        n = len(P.T)
        d = self.dimension

        translations = []
        for i in range(d):
            translation = np.zeros(d*n)
            translation[i::d] = 1
            translations.append(translation)
        trivials = np.column_stack(translations)

        rotations = utils.infRotations(d)
        for rotMat in rotations:
            infRot = (rotMat@P).ravel('F')
            trivials = np.column_stack([trivials, infRot])

        kernell = linalg.null_space(R)

        return utils.projectAway(trivials, kernell)


class ScrewFramework:
    """
    A class to represent infinite, screw periodic frameworks.

    Parameters
    ----------
    G : list
        G[0] is a list of vertices in the base graph.
        G[1] is a dictionary of edge markings. Each key is a vertex, and
        its corresponding value is a list of 3-tuple markings that give the end
        vertex of the edge, and the group element in Z^2 of the edge.
    P : numpy.ndarray
        (3,n) array containing with columns as the coordinates of the 'base' or
        'identity' vertices.
    T1, T2 : AffineTransform
        Affine transformations that define the periodicity of the graph.

    Attributes
    ----------
    graph : tuple
        graph[0] is a list of vertices in the base graph.
        graph[1] is a dictionary of edge markings. Each key is a vertex, and
        its corresponding value is a list of 3-tuple markings that give the end
        vertex of the edge, and the group element in Z^2 of the edge.
    baseConfig : numpy.ndarray
        (3,n) array containing with columns as the coordinates of the 'base' or
        'identity' vertices.
    T1, T2 : AffineTransform
        The two transformations that define the periodicity of the framework.
    """

    def __init__(self, G, P, T1, T2):
        """Initialise screw periodic framework."""
        self.graph = G
        self.baseConfig = P
        self.T1 = T1
        self.T2 = T2

    def vertexLocation(self, idVertex, n):
        """
        Calculate the location of a vertex given a base vertex and index.

        Parameters
        ----------
        idVertex : int
            Vertex in the 'base' vertex set of the graph.
        n: tuple of int
            Length 2 tuple representing the element of Z^2 that indexes a
            vertex in the repeating framework.

        Returns
        -------
        np.ndarray
            The (d,) array representing the coordinates the indexed vertex.
        """
        n1, n2 = n
        idVector = self.baseConfig[:, idVertex]

        r = self.T1(idVector, pow=n1)
        r = self.T2(r, pow=n2)
        return r

    def repeatedConfig(self, nMax):
        """
        Generate the coordinates of the repreating.

        This creates a four dimensional array of points to represent the
        coordinates of a finite portion of the repeating framework from the
        base copy to the (n1_max, n2_max) copy.

        Parameters
        ----------
        nMax : tuple of int
            Length 2 tuple representing maximum index of the repeating
            framework to display.

        Returns
        -------
        np.ndarray
            Four-dimensional array representing a finite section of the
            infinite framework. The first axis represents the vertex, the
            second and third axis represent the element of Z^2 that indexes the
            framework, and the fourth axis represents the x, y and z
            coordinates.
        """
        n1, n2 = nMax
        vertices = self.graph[0]
        nv = len(vertices)
        repeatedConfig = np.zeros((nv, n1 + 1, n2 + 1, 3))

        for vertex in vertices:
            for i in range(n1+1):
                for j in range(n2+1):
                    repeatedConfig[vertex, i, j] = self.vertexLocation(
                        vertex, (i, j)).flatten()
        return repeatedConfig

    def edgeVector(self, edge, n):
        """
        Calculate the vector between two vertices in the framework.

        Parameters
        ----------
        edge : tuple of int
            Length 2 tuple of integers that represent vertices.
        n : tuple of int
            Length 2 tuple representing the marking of the edge.
        """
        i, j = edge
        p_i = self.baseConfig[:, i]
        Tp_j = self.vertexLocation(j, n)

        return Tp_j - p_i

    def rigidityMatrix(self):
        """
        Generate the the rigidity matrix.

        Returns
        -------
        numpy.ndarray
            The rigidity matrix of (G,p).
        """
        vertices, edges = self.graph
        edgesFlat = [edge for adjList in edges.values() for edge in adjList]
        nEdges = len(edgesFlat)
        nVerts = len(vertices)
        row = 0

        R = np.zeros(shape=(nEdges, 3*nVerts))
        for i in vertices:
            nbs = edges[i]
            for edge in nbs:
                j = edge[0]
                marking = edge[1:]

                edgeVector = self.edgeVector((i, j), marking)
                current = R[row, 3*i: 3*(i+1)]
                R[row, 3*i: 3*(i+1)] = current + edgeVector

                rEdgeVector = self.edgeVector((j, i), [-n for n in marking])
                current = R[row, 3*j: 3*(j+1)]
                R[row, 3*j: 3*(j+1)] = current + rEdgeVector

                row += 1
        return R

    def draw(self, nMax, fmt='-b', VKwargs={}, EKwargs={}):
        """
        Draw a portion of the periodic framework.

        Draws the section of the periodic framework from the (0, 0) index to
        the `nMax` index.

        Parameters
        ----------
        nMax : tuple of int
            The maximum index of the framework to draw.
        fmt : str, optional
            Format string for drawing edges of the form
            '[marker][line][colour]'.
            Default : '-b' (A blue line)
        EKwargs, VKwargs : dict, optional
            Key word arguments used in the drawing of edges and vertices.

        Returns
        -------
        matplotlib.figure.Figure
            Figure on which the portion of the configuration is drawn.
        """
        verts, edges = self.graph
        repeatedConfig = self.repeatedConfig(nMax)
        n1, n2 = nMax
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for v in verts:
            repeats = repeatedConfig[v, ...]
            xs = repeats[..., 0]
            ys = repeats[..., 1]
            zs = repeats[..., 2]

            ax.scatter(xs, ys, zs, **VKwargs)

        for v in verts:
            nbs = edges[v]

            for edge in nbs:
                endNode = edge[0]
                mark1, mark2 = edge[1:]

                for i in range(n1+1-mark1):
                    for j in range(n2+1-mark2):
                        start = repeatedConfig[v, i, j]
                        end = repeatedConfig[endNode, i + mark1, j + mark2]
                        E = np.column_stack([start, end])
                        ax.plot(*E, fmt, **EKwargs)
        return fig
