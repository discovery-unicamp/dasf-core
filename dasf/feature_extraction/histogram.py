#!/usr/bin/env python3

import dask.array as da
import numpy as np

try:
    import cupy as cp
except ImportError: # pragma: no cover
    pass

from dasf.transforms.base import TargeteredTransform, Transform


class Histogram(TargeteredTransform, Transform):
    """Operator to extract the histogram of a data.

    Parameters
    ----------
    bins : Optional[int]
        Number of bins (the default is None).
    range : tuple
        2-element tuple with the lower and upper range of the bins. If not
        provided, range is simply (X.min(), X.max()) (the default is None).
    normed : bool
        If the historgram must be normalized (the default is False).
    weights : type
        An array of weights, of the same shape as X. Each value in a only
        contributes its associated weight towards the bin count
        (the default is None).
    density : type
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at the bin, normalized such that the integral over the range is 1
        (the default is None).

    Attributes
    ----------
    bins
    range
    normed
    weights
    density

    """
    def __init__(self,
                 bins: int = None,
                 range: tuple = None,
                 normed: bool = False,
                 weights=None,
                 density=None,
                 *args,
                 **kwargs):
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
