"""
Create tools to work with rigidity theory.

Classes:

    AffineTansform
    Framework

Functions:

"""


import numpy as np
from sympy import symbols, solve
from sympy import Matrix as symMatrix
import matplotlib.pyplot as plt


class AffineTransform:
    """A class to represent callable affine transformations.

    ...

    Attributes
    ----------
    linear : numpy.ndarray
        dxd array for linear component of the transformation
    translation : numpy.ndarray
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
        self.__invertible = None
        self.__inverse = None

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
        """Return the inverse affine transformation if such a transformation exists.

        Returns:
        inverse : AffineTransform
            The inverse of the current affine transformation
        """
        assert self.isInvertible(), "Transformation is not invertible"

        if self.__inverse is None:
            invLinear = np.linalg.inv(self.linear)
            invTranslation = -1*self.translation
            self.__inverse = AffineTransform(invLinear, invTranslation)

        return self.__inverse


class Framework:
    """A class to represent classical rigidity frameworks.

    ...

    Attributes
    ----------
    graph : tuple
        graph[0] is a list of vertices in the graph.
        graph[1] is a dictionary of edges.
    config : list of numpy.ndarray
        List of dx1 arrays containing the coordinates of
        the of the vertices in the framework
    dimension : int
        The dimension of the Euclidean space in which to
        embed the framework.

    Methods
    -------
    rigidityMatrix
        Returns the numeric rigidity matrix of the framework.
    symbolicRigidityMatrix
        Returns the symbolic rigidity matrix of the framework,
        making use of sympy.Matrix.
    infinitesimalFlex
        Finds the non-trivial infinitesimal flexes of the
        framework if the framework is flexible.
    """

    def __init__(self, G, P):
        """Initialise a framework.

        Params
        ------
        G : tuple
            graph[0] must be a list of vertices in the graph,
            of the form [0, 1, ..., n]
            graph[1] must be a dictionary of edges. Each key
            must be  a vertex, and its corresponding value a
            list of edges that vertex is adjacent to. In this
            implementation of a graph, we do not allow loops
            or multiple edges.
        P : list of numpy.ndarray
            dx1 vectors for the configuration of the framework.
        """
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

        assert type(P) == list, "P must be a list"
        assert len(P) == nv, ("There must be the same number of vertices"
                              " as points in the configuration")

        if nv == 0:
            d = 0
        else:
            for vector in P:
                assert type(vector) == np.ndarray,\
                    "P must be a list of numpy.ndarray"

            vShape = np.shape(P[0])
            assert len(vShape) == 2, "Elements of p must be 2-dimensional"
            assert vShape[1] == 1, "Elements of p must have shape (d,1)"
            for vector in P:
                assert np.shape(vector) == vShape, \
                    "All elements of P must have the same shape"

            d = vShape[0]

        self.graph = G
        self.config = P
        self.dimension = d

    def rigidityMatrix(self):
        """Create the numerical rigidity matrix.

        Returns
        -------
        R : numpy.ndarray
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
                    edgeVector = self.config[i] - self.config[j]
                    R[row, d*i: d*(i+1)] = edgeVector.transpose()
                    R[row, d*j: d*(j+1)] = -1 * edgeVector.transpose()

                    row += 1
        return R

    def symbolicRigidityMatrix(self):
        """Create the symbolic rigidity matrix.

        Returns
        -------
        R : MutableDenseMatrix
            The rigidity matrix of (G,p).
        """
        verts, edges = self.graph
        edgesFlat = [edge for adjList in edges.values() for edge in adjList]
        nEdges = len(edgesFlat) // 2
        nVerts = len(verts)
        row = 0

        R = symMatrix.zeros(nEdges, nVerts)
        for i in verts:
            nbs = edges[i]
            for j in nbs:
                if i < j:
                    edgeSymb = symbols('p' + str(i)) - symbols('p' + str(j))
                    R[row, i:i+1] = [edgeSymb]
                    R[row, j:j+1] = [-1 * edgeSymb]

                    row += 1
        return R

    def infinitesimalFlex(self):
        """Calculate a non-trivial infinitesimal flex if such a flex exists.

        Returns
        -------
        flex : list
            A non-trivial infinitesimal flex if one exists, else all zeros.
        """
        verts, edges = self.graph
        nVerts = len(verts)
        d = self.dimension
        P = self.config
        P = [symMatrix(vector) for vector in P]

        bars = [P[i]-P[j] for i in verts for j in edges[i] if i < j]

        # Create a list of symbols that will represent our flex
        flex = []
        for i in range(nVerts):
            col = symMatrix([symbols('x' + str(i) + str(k)) for k in range(d)])
            flex.append(col)

        # Create a list of constraints that fix the first point in the
        # configuration, then constrains the second flex to be in the
        # line between p0 and p1, the third flex to lie in the plan
        # defined by p0, p1 and p2 and so on. This is possible if n > dim
        # and the configuration is in general position. We do this by finding
        # the nullspace of the matrix of vectors that define the hyperplane,
        # which gives a normal vector. To ensure the flex remains in the
        # hyperplane, we enforce that the dot product between the normal and
        # the flex is zero.
        nonTrivConstraints = []
        for entry in flex[0]:
            nonTrivConstraints.append(entry)

        p0 = P[0]
        for i in range(1, d):
            vects = [p0 - pi for pi in P[1:i+1]]
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

    def draw(self):
        """Draw the framework with given configuration."""
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
        PMat = np.column_stack(P)

        for i in verts:
            for j in edges[i]:
                if i < j:
                    ri = P[i]
                    rj = P[j]
                    ax.plot(*[[ri[k, 0], rj[k, 0]] for k in range(d)], '-k')

        ax.scatter(*[PMat[i, :] for i in range(d)])


class ScrewFramework:
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
    """

    def __init__(self, G, P, T1, T2):
        self.graph = G
        self.baseConfig = P
        self.T1 = T1
        self.T2 = T2

    def vertexLocation(self, idVector, n):
        n1, n2 = n

        r = self.T1(idVector, pow=n1)
        r = self.T2(r, pow=n2)

        return r

    def repeatedConfig(self, nMax):
        n1, n2 = nMax
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
        row = 0

        R = np.zeros(shape=(nEdges, 3*nVerts))
        for i in vertices:
            nbs = edges[i]
            for edge in nbs:
                j = edge[0]
                marking = edge[1:]

                edgeVector = self.edgeVector((i, j), marking)
                current = R[row, 3*i: 3*(i+1)]
                R[row, 3*i: 3*(i+1)] = current + edgeVector.transpose()

                rEdgeVector = self.edgeVector((j, i), [-n for n in marking])
                current = R[row, 3*j: 3*(j+1)]
                R[row, 3*j: 3*(j+1)] = current + rEdgeVector.transpose()

                row += 1
        return R

    def draw(self, nMax):
        verts, edges = self.graph
        repeatedConfig = self.repeatedConfig(nMax)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for v in verts:
            repeats = repeatedConfig[v,:,:,:]
            xs = repeats[:,:,0]
            ys = repeats[:,:,1]
            zs = repeats[:,:,2]

            ax.scatter(xs,ys,zs)    

        for v in verts:
            nbs = edges[v]
            for endNode, edge in enumerate(edges):
                if edge != None:
                    for i in range(n[0]+1):
                        for j in range(n[1]+1):
                            if i + edge[0] <= n[0] and j + edge[1] <= n[1]:
                                start = conf[vertex,i,j]
                                end = conf[endNode, i + edge[0], j + edge[1]]
                                ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], '-b', alpha = 0.5)                    
                    
        plt.show()