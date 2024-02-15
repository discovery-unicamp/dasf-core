#!/usr/bin/env python3

import numpy as np

try:
    import cupy as cp
except ImportError: # pragma: no cover
    pass

from dasf.transforms.base import Transform, ReductionTransform


class Reshape:
    """Reshape data with a new shape.

    Parameters
    ----------
    shape : tuple
        The new shape of the data.

    """
    def __init__(self, shape: tuple):
        self.shape = shape

    def run(self, X):
        print(X.shape)
        return X.reshape(self.shape)


class SliceArray(Transform):
    def __init__(self, output_size):
        self.x = list(output_size)

    def transform(self, X):
        if len(self.x) == 1:
            return X[0:self.x[0]]
        elif len(self.x) == 2:
            return X[0:self.x[0], 0:self.x[1]]
        elif len(self.x) == 3:
            return X[0:self.x[0], 0:self.x[1], 0:self.x[2]]
        else:
            raise Exception("The dimmension is not known")


class SliceArrayByPercent(Transform):
    def __init__(self, x=100.0, y=100.0, z=100.0):
        self.x = float(x / 100.0)
        self.y = float(y / 100.0)
        self.z = float(z / 100.0)

    def transform(self, X):
        if self.x > 1 or self.y > 1 or self.z > 1:
            raise Exception("Percentages cannot be higher than 100% (1.0)")

        if X.ndim == 1:
            return X[0:int(self.x * X.shape[0])]
        elif X.ndim == 2:
            return X[0:int(self.x * X.shape[0]), 0:int(self.y * X.shape[1])]
        elif X.ndim == 3:
            return X[
                0:int(self.x * X.shape[0]),
                0:int(self.y * X.shape[1]),
                0:int(self.z * X.shape[2]),
            ]
        else:
            raise Exception("The dimmension is not known")


class SliceArrayByPercentile(Transform):
    def __init__(self, percentile):
        self.p = percentile

    def __internal_chunk_array_positive(self, block, axis=None, keepdims=False, xp=np):
        block[block < 0] = 0
        block[block != 0]
        return xp.array([xp.percentile(block.flatten(), self.p)])

    def __internal_aggregate_array_positive(self, block, axis=None, keepdims=False, xp=np):
        return xp.array([xp.max(block)])

    def __internal_chunk_array_negative(self, block, axis=None, keepdims=False, xp=np):
        block *= -1
        block[block < 0] = 0
        block[block != 0]
        return xp.array([-xp.percentile(block.flatten(), self.p)])

    def __internal_aggregate_array_negative(self, block, axis=None, keepdims=False, xp=np):
        return xp.array([xp.min(block)])

    def _lazy_transform_cpu(self, X):
        positive = ReductionTransform(func_chunk=self.__internal_chunk_array_positive,
                                      func_aggregate=self.__internal_aggregate_array_positive,
                                      output_size=[0])

        negative = ReductionTransform(func_chunk=self.__internal_chunk_array_negative,
                                      func_aggregate=self.__internal_aggregate_array_negative,
                                      output_size=[0])

        p = positive._lazy_transform_cpu(X, axis=[0])
        n = negative._lazy_transform_cpu(X, axis=[0])

        # Unfortunately, we need to compute first.
        pos_cutoff = p.compute()[0]
        neg_cutoff = n.compute()[0]

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _lazy_transform_gpu(self, X):
        positive = ReductionTransform(func_chunk=self.__internal_aggregate_array_positive,
                                      func_aggregate=self.__internal_aggregate_array_positive,
                                      output_size=[0])

        negative = ReductionTransform(func_chunk=self.__internal_aggregate_array_negative,
                                      func_aggregate=self.__internal_aggregate_array_negative,
                                      output_size=[0])

        p = positive._lazy_transform_gpu(X)
        n = negative._lazy_transform_gpu(X)

        # Unfortunately, we need to compute first.
        pos_cutoff = p.compute()[0]
        neg_cutoff = n.compute()[0]

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _transform_cpu(self, X):
        pos_cufoff = self.__internal_chunk_array_positive(X, xp=np)
        neg_cutoff = self.__internal_chunk_array_negative(X, xp=np)

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X

    def _transform_gpu(self, X):
        pos_cufoff = self.__internal_chunk_array_positive(X, xp=cp)
        neg_cutoff = self.__internal_chunk_array_negative(X, xp=cp)

        X[X > pos_cutoff] = pos_cutoff
        X[X < neg_cutoff] = neg_cutoff

        return X
