#!/usr/bin/env python3

import numpy as np
import dask.array as da

try:
    import cupy as cp
except ImportError:
    pass

from dasf.transforms.base import Transform
from dasf.transforms.base import TargeteredTransform


class Histogram(TargeteredTransform, Transform):
    def __init__(self, bins=None, range=None, normed=False, weights=None,
                 density=None, *args, **kwargs):
        TargeteredTransform.__init__(self, *args, **kwargs)

        self._bins = bins
        self._range = range
        self._normed = normed
        self._weights = weights
        self._density = density

    def __lazy_transform_generic(self, X):
        return da.histogram(
            X,
            bins=self._bins,
            range=self._range,
            normed=self._normed,
            weights=self._weights,
            density=self._density,
        )

    def __transform_generic(self, X, xp):
        return xp.histogram(
            X,
            bins=self._bins,
            range=self._range,
            normed=self._normed,
            weights=self._weights,
            density=self._density,
        )

    def _lazy_transform_cpu(self, X):
        return self.__lazy_transform_generic(X)

    def _lazy_transform_gpu(self, X, **kwargs):
        return self.__lazy_transform_generic(X)

    def _transform_cpu(self, X, **kwargs):
        return self.__transform_generic(X, np)

    def _transform_gpu(self, X, **kwargs):
        return self.__transform_generic(X, cp)
