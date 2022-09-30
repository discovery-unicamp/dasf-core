#!/usr/bin/env python3

import math

import numpy as np
import pandas as pd

from dasf.utils.types import is_array
from dasf.utils.types import is_dask_array
from dasf.transforms.base import Transform

try:
    import cupy as cp
    import cudf
except ImportError:
    pass


class Normalize(Transform):
    def transform(self, X):
        return (X - X.mean()) / (X.std(ddof=0))


class ArraysToDataFrame(Transform):
    def __transform_generic(self, X, y):
        assert len(X) == len(y), "Data and labels should have the same length."

        dfs = None
        for i, x in enumerate(X):
            if is_array(x):
                # Dask has some facilities to convert to DataFrame
                if is_dask_array(x):
                    new_chunk = math.prod(x.chunksize)
                    flat = x.flatten().rechunk(new_chunk)

                    if dfs is None:
                        dfs = flat.to_dask_dataframe(columns=[y[i]])
                    else:
                        dfs = dfs.join(flat.to_dask_dataframe(columns=[y[i]]))
                else:
                    flat = x.flatten()

                    if dfs is None:
                        dfs = list()
                    dfs.append(flat)
            else:
                raise Exception("This is not an array. This is a '%s'."
                                % str(type(x)))

        return dfs

    def _lazy_transform_cpu(self, X=None, **kwargs):
        X = list(kwargs.values())
        y = list(kwargs.keys())

        return self.__transform_generic(X, y)

    def _lazy_transform_gpu(self, X=None, **kwargs):
        X = list(kwargs.values())
        y = list(kwargs.keys())

        return self.__transform_generic(X, y)

    def _transform_gpu(self, X=None, **kwargs):
        X = list(kwargs.values())
        y = list(kwargs.keys())

        dfs = self.__transform_generic(X, y)

        if is_array(dfs) and not is_dask_array(dfs):
            datas = cp.stack(dfs, axis=-1)
            datas = cudf.DataFrame(datas, columns=y)
        else:
            datas = dfs

        return datas

    def _transform_cpu(self, X=None, **kwargs):
        X = list(kwargs.values())
        y = list(kwargs.keys())

        dfs = self.__transform_generic(X, y)

        if is_array(dfs) and not is_dask_array(dfs):
            datas = np.stack(dfs, axis=-1)
            datas = pd.DataFrame(datas, columns=y)
        else:
            datas = dfs

        return datas
