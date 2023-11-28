#!/usr/bin/env python3

import h5py
import math
import zarr

import numpy as np
import pandas as pd
import dask
import dask.dataframe as ddf

from dasf.utils.types import is_array
from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_gpu_array
from dasf.transforms.base import Transform

try:
    import cupy as cp
    import cudf
except ImportError: # pragma: no cover
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
    def _convert_filename(url):
        if url.endswith(".npy"):
            return url.replace(".npy", ".zarr")
        return url + ".zarr"

    def _lazy_transform_generic_all(self, data):
        if self.filename:
            url = self.filename
        elif hasattr(data, '_root_file'):
            url = data._root_file
        else:
            raise Exception("Array requires a valid path to convert to Zarr.")

        if data is None:
            raise Exception("Dataset needs to be loaded first.")

        url = self._convert_filename(url)

        # XXX: Workaround to avoid error with CuPy and Zarr library
        if is_dask_gpu_array(data):
            data = data.map_blocks(lambda x: x.get())

        data.to_zarr(url)

        return url

    def _transform_generic_all(self, data, chunks, **kwargs):
        if data is None:
            raise Exception("Dataset needs to be loaded first.")

        if not chunks:
            raise Exception("Chunks needs to be passed for non lazy arrays.")

        if self.filename:
            url = self.filename
        else:
            raise Exception("Array requires a valid path to convert to Zarr.")

        url = self._convert_filename(url)

        z = zarr.open(store=url, mode='w', shape=data.shape,
                      chunks=chunks, dtype='i4')

        z = data

        return url

    def _lazy_transform_generic(self, X, **kwargs):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray
        from dasf.datasets.base import DatasetZarr

        name = None

        if isinstance(X, DatasetArray):
            name = X._name
            chunks = X._chunks

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._lazy_transform_generic_all(X._data)
        elif is_dask_array(X):
            chunks = X.chunks

            url = self._lazy_transform_generic_all(X)
        else:
            raise Exception("It is not an Array type.")

        return DatasetZarr(name=name, download=False, root=url, chunks=chunks)

    def _transform_generic(self, X, **kwargs):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray
        from dasf.datasets.base import DatasetZarr

        name = None
        url = None

        if hasattr(X, '_chunks') and \
           (X._chunks is not None and X._chunks != 'auto'):
            chunks = X._chunks
        else:
            chunks = self.chunks

        if chunks is None:
            raise Exception("Chunks needs to be specified.")

        if isinstance(X, DatasetArray):
            name = X._name

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._transform_generic_all(X._data, chunks)
        elif is_array(X):
            url = self._transform_generic_all(X, chunks)
        else:
            raise Exception("It is not an Array type.")

        return DatasetZarr(name=name, download=False, root=url, chunks=chunks)

    def _lazy_transform_gpu(self, X, **kwargs):
        return self._lazy_transform_generic(X, **kwargs)

    def _lazy_transform_cpu(self, X, **kwargs):
        return self._lazy_transform_generic(X, **kwargs)

    def _transform_gpu(self, X, **kwargs):
        return self._transform_generic(X, **kwargs)

    def _transform_cpu(self, X, **kwargs):
        return self._transform_generic(X, **kwargs)


class ArrayToHDF5(Transform):
    def __init__(self, dataset_path, chunks=None, save=True, filename=None):
        # Avoid circular dependency
        from dasf.datasets.base import DatasetArray
        from dasf.datasets.base import DatasetHDF5

        self.dataset_path = dataset_path
        self.chunks = chunks
        # TODO: implement the possibility of not saving
        self.save = True
        self.filename = filename

    @staticmethod
    def _convert_filename(url):
        if url.endswith(".npy"):
            return url.replace(".npy", ".hdf5")
        return url + ".hdf5"

    def _lazy_transform_generic_all(self, data):
        if self.filename:
            url = self.filename
        elif hasattr(data, '_root_file'):
            url = data._root_file
        else:
            raise Exception("Array requires a valid path to convert to HDF5.")

        if data is None:
            raise Exception("Dataset needs to be loaded first.")

        url = self._convert_filename(url)

        data.to_hdf5(url, self.dataset_path)

        return url

    def _transform_generic_all(self, data):
        if data is None:
            raise Exception("Dataset needs to be loaded first.")

        if self.filename:
            url = self.filename
        else:
            raise Exception("Array requires a valid path to convert to Zarr.")

        url = self._convert_filename(url)

        h5f = h5py.File(url, 'w')
        h5f.create_dataset(self.dataset_path, data=data)
        h5f.close()

        return url

    def _lazy_transform_generic(self, X, **kwargs):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray
        from dasf.datasets.base import DatasetHDF5

        name = None
        chunks = None

        if isinstance(X, DatasetArray):
            name = X._name
            chunks = X._chunks

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._lazy_transform_generic_all(X._data)
        elif is_dask_array(X):
            chunks = X.chunks

            url = self._lazy_transform_generic_all(X)
        else:
            raise Exception("It is not an Array type.")

        return DatasetHDF5(name=name, download=False, root=url, chunks=chunks,
                           dataset_path=self.dataset_path)

    def _transform_generic(self, X, **kwargs):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray
        from dasf.datasets.base import DatasetHDF5

        name = None
        url = None

        if hasattr(X, '_chunks') and \
           (X._chunks is not None and X._chunks != 'auto'):
            chunks = X._chunks
        else:
            chunks = self.chunks

        if isinstance(X, DatasetArray):
            name = X._name

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._transform_generic_all(X._data)
        elif is_array(X):
            url = self._transform_generic_all(X)
        else:
            raise Exception("It is not an Array type.")

        return DatasetHDF5(name=name, download=False, root=url, chunks=chunks,
                           dataset_path=self.dataset_path)

    def _lazy_transform_gpu(self, X, **kwargs):
        return self._lazy_transform_generic(X, **kwargs)

    def _lazy_transform_cpu(self, X, **kwargs):
        return self._lazy_transform_generic(X, **kwargs)

    def _transform_gpu(self, X, **kwargs):
        return self._transform_generic(X, **kwargs)

    def _transform_cpu(self, X, **kwargs):
        return self._transform_generic(X, **kwargs)


class ZarrToArray(Transform):
    def __init__(self, chunks=None, save=True, filename=None):
        self.chunks = chunks
        self.save = save
        self.filename = filename

    @staticmethod
    def _convert_filename(url):
        if url.endswith(".zarr"):
            return url.replace(".zarr", ".npy")
        return url + ".npy"

    def transform(self, X):
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetZarr

        if issubclass(X.__class__, DatasetZarr):
            if self.save:
                if self.filename:
                    url = self.filename
                elif hasattr(X, '_root_file'):
                    url = X._root_file
                else:
                    raise Exception("Array requires a valid path to convert to Array.")

                url = self._convert_filename(url)

                np.save(url, X._data)

            # This is just a place holder
            return X._data
        else:
            raise Exception("Input is not a Zarr dataset.")


class ArraysToDataFrame(Transform):
    def _build_dataframe(self, data, columns, xp, df):
        data = [d.flatten() for d in data]
        stacked_data = xp.stack(data, axis=1)
        return df.DataFrame(
            stacked_data,
            columns=columns
        )

    def _lazy_transform(self, xp, df, **kwargs):
        X = list(kwargs.values())
        y = list(kwargs.keys())
        assert len(X) == len(y), "Data and labels should have the same length."

        meta = ddf.utils.make_meta([
            (col, data.dtype)
            for col, data in zip(y, X)
        ],
            parent_meta=None if df == pd else cudf.DataFrame
        )

        lazy_dataframe_build = dask.delayed(self._build_dataframe)
        data_chunks = [x.to_delayed().ravel() for x in X]
        partial_dataframes = [
            ddf.from_delayed(lazy_dataframe_build(data=mapped_chunks, columns=y, xp=xp, df=df), meta=meta)
            for mapped_chunks in zip(*data_chunks)
        ]

        return ddf.concat(partial_dataframes)

    def _lazy_transform_cpu(self, X=None, **kwargs):
        return self._lazy_transform(np, pd, **kwargs)
    
    def _lazy_transform_gpu(self, X=None, **kwargs):
        return self._lazy_transform(cp, cudf, **kwargs)
    
    def _transform(self, xp, df, **kwargs):
        X = list(kwargs.values())
        y = list(kwargs.keys())
        assert len(X) == len(y), "Data and labels should have the same length."

        return self._build_dataframe(data=X, columns=y, xp=xp, df=df)

    def _transform_cpu(self, X=None, **kwargs):
        return self._transform(np, pd, **kwargs)

    def _transform_gpu(self, X=None, **kwargs):
        return self._transform(cp, cudf, **kwargs)
