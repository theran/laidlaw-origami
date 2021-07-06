class AffineTransform:
    def __init__(self, rotationMatrix, translationVector):
        self.rotation = rotationMatrix
        self.translation = translationVector

    def __call__(self, v, pow=1):
        assert isinstance(pow, int), "pow must be an int"
        ans = v

        if pow >= 1:
            for i in range(pow):
                ans = self.rotation*ans + self.rotation
        elif pow <= -1:
            for i in range(abs(pow)):
                ans = (self.rotation**(-1))*(ans - self.translation)
        return ans
