#!/usr/bin/env python3

import h5py
import math
import zarr

import numpy as np
import pandas as pd

from dasf.utils.types import is_array
from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_gpu_array
from dasf.transforms.base import Transform

try:
    import cupy as cp
    import cudf

    import dask_cudf as dcudf
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
        # TODO: implement the possibility of not saving
        self.save = True
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
                        if is_dask_gpu_array(x):
                            dfs = dcudf.from_dask_dataframe(dfs)
                    else:
                        if is_dask_gpu_array(x):
                            dfs_aux = flat.to_dask_dataframe(columns=[y[i]])
                            dfs = dfs.join(dcudf.from_dask_dataframe(dfs_aux))
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
