#!/usr/bin/env python3

import math
import zarr

import numpy as np
import pandas as pd

from dasf.utils.types import is_array
from dasf.utils.types import is_dask_array
from dasf.transforms.base import Transform

from dasf.datasets import DatasetArray
from dasf.datasets import DatasetZarr

try:
    import cupy as cp
    import cudf
except ImportError:
    pass


class Normalize(Transform):
    def transform(self, X):
        return (X - X.mean()) / (X.std(ddof=0))


class ArrayToZarr(Transform):
    def __init__(self, chunks=None, save=True, filename=None):
        self.chunks = chunks
        # TODO: implement the possibility of not saving
        self.save = True
        self.filename = filename

    @staticmethod
    def __convert_filename(url):
        if url.endswith(".npy"):
            return url.replace(".npy", ".zarr")
        return url

    def __lazy_transform_generic(self, X, **kwargs):
        name = None
        chunks = None
        url = None
        if isinstance(X, DatasetArray):
            name = X._name
            chunks = X._data.chunks

            if self.filename:
                url = self.filename
            else:
                url = X._root_file
            url = self.__convert_filename(url)

            X._data.to_zarr(url)
        elif is_dask_array(X):
            chunks = X.chunks

            if not self.filename:
                raise Exception("Dask Array requires a valid path to convert to Zarr.")

            url = self.filename
            X.to_zarr(url)
        else:
            raise Exception("It is not an Array type.")

        return DatasetZarr(name=name, download=False, root=url, chunks=chunks)

    def __transform_generic(self, X, **kwargs):
        name = None
        chunks = None
        url = None
        if isinstance(X, DatasetArray):
            name = X._name
            chunks = X._chunks

            if self.filename:
                url = self.filename
            else:
                url = X._root_file

            if not chunks:
                raise Exception("Chunks needs to be passed for non lazy arrays.")

            url = self.__convert_filename(url)

            z = zarr.open(url, mode='w', shape=X._data.shape,
                          chunks=chunks, dtype='i4')

            z = X._data
        elif is_dask_array(X):
            chunks = X.chunks

            if not self.filename:
                raise Exception("Dask Array requires a valid path to convert to Zarr.")

            url = self.filename

            z = zarr.open(url, mode='w', shape=X.shape,
                          chunks=chunks, dtype='i4')

            z = X
        else:
            raise Exception("It is not an Array type.")

        return DatasetZarr(name=name, download=False, root=url, chunks=chunks)

    def _lazy_transform_gpu(self, X, **kwargs):
        return self.__lazy_transform_generic(X, **kwargs)

    def _lazy_transform_cpu(self, X, **kwargs):
        return self.__lazy_transform_generic(X, **kwargs)

    def _transform_gpu(self, X, **kwargs):
        return self.__transform_generic(X, **kwargs)

    def _transform_cpu(self, X, **kwargs):
        return self.__transform_generic(X, **kwargs)

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
