import numpy as np


"""
Create tools to work with rigidity theory.

Classes:

    AffineTansform

Functions:

"""


class AffineTransform:
    """A class to represent affine transformations.

    ...

    Attributes
    ----------
    rotation : numpy.matrix
        3x3 rotational component of the transformation
    translation : numpy,matrix
        3x1 translational component of the transformation
    """

    def __init__(self, rotationMatrix, translationVector):
        """Initialise an affine transformation of the form Ax+B."""
        self.rotation = rotationMatrix
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
        assert isinstance(pow, int), "pow must be an int"
        ans = v

        if pow >= 1:
            for i in range(pow):
                ans = self.rotation*ans + self.translation
        elif pow <= -1:
            for i in range(abs(pow)):
                ans = (self.rotation**(-1))*(ans - self.translation)
        return ans


class Framework:
    def __init__(self, G, P, n, T1, T2):
        self.graph = G
        self.baseConfig = P
        self.n = n
        self.T1 = T1
        self.T2 = T2
        self.repeatedConfig = self.__generateRepeatedConfig()

    def vertexLocation(self, r_id, n):
        n1, n2 = n

        r = self.T1(r_id, pow=n1)
        r = self.T2(r, pow=n2)

        return r

    def __generateRepeatedConfig(self):
        n1, n2 = self.n
        nv = len(self.baseConfig)
        repeatedConfig = np.zeros((nv, n1 + 1, n2 + 1, 3))

        for vertex, coord in enumerate(self.baseConfig):
            for i in range(n1+1):
                for j in range(n2+1):
                    repeatedConfig[vertex, i, j, :] = self.vertexLocation(
                        coord, (i, j))[:]
        return repeatedConfig
