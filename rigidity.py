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
