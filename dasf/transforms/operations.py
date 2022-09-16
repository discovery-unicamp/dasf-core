#!/usr/bin/env python3

from dasf.pipeline.pipeline import Operator


class Reshape(Operator):
    """Reshape data with a new shape.

    Parameters
    ----------
    shape : tuple
        The new shape of the data.

    """
    def __init__(self, shape: tuple):
        super().__init__(name="Reshape Data")

        self.shape = shape

    def run(self, X):
        print(X.shape)
        return X.reshape(self.shape)


class SliceArray:
    def __init__(self, output_size):
        self.x = list(output_size)

    def __transform(self, X):
        if len(self.x) == 1:
            return X[0:self.x[0]]
        elif len(self.x) == 2:
            return X[0:self.x[0], 0:self.x[1]]
        elif len(self.x) == 3:
            return X[0:self.x[0], 0:self.x[0], 0:self.x[0]]
        else:
            raise Exception("The dimmension is not known")

    def transform(self, X):
        return self.__transform(X)


class SliceArrayOperator(Operator):
    def __init__(self, x=None, y=None, z=None):
        super().__init__(name="Slice Array")

        self.x = x
        self.y = y
        self.z = z

    def run(self, X):
        if X.ndim == 1:
            if self.x is None:
                self.x = X.shape[0]

            return X[0:self.x]
        elif X.ndim == 2:
            if self.x is None:
                self.x = X.shape[0]

            if self.y is None:
                self.y = X.shape[1]

            return X[0:self.x, 0:self.y]
        elif X.ndim == 3:
            if self.x is None:
                self.x = X.shape[0]

            if self.y is None:
                self.y = X.shape[1]

            if self.z is None:
                self.z = X.shape[2]

            return X[0:self.x, 0:self.y, 0:self.z]
        else:
            raise Exception("The dimmension is not known")


class SliceArrayByPercent(Operator):
    def __init__(self, x=100.0, y=100.0, z=100.0):
        super().__init__(name="Slice Array")

        self.x = float(x/100.0)
        self.y = float(y/100.0)
        self.z = float(z/100.0)

    def run(self, X):
        if self.x > 1 or self.y > 1 or self.z > 1:
            raise Exception("Percentages cannot be higher than 100% (1.0)")

        if X.ndim == 1:
            return X[0:int(self.x*X.shape[0])]
        elif X.ndim == 2:
            return X[0:int(self.x*X.shape[0]),
                     0:int(self.y*X.shape[1])]
        elif X.ndim == 3:
            return X[0:int(self.x*X.shape[0]),
                     0:int(self.y*X.shape[1]),
                     0:int(self.z*X.shape[2])]
        else:
            raise Exception("The dimmension is not known")
