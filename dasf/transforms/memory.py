#!/usr/bin/env python3

from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_dataframe
from dasf.transforms.base import Transform


class PersistDaskData(Transform):
    """Allow persisting a dask array to memory and return a copy of the object.
    It will gather the data blocks from all workers and resembles locally.
    """
    def __lazy_transform_generic(self, X):
        if is_dask_array(X) or is_dask_dataframe(X):
            new_data = X.persist()
        else:
            new_data = X

        return new_data

    def _lazy_transform_cpu(self, X):
        return self.__lazy_transform_generic(X)

    def _lazy_transform_gpu(self, X):
        return self.__lazy_transform_generic(X)

    def _transform_cpu(self, X):
        # Bypass because the data is local
        return X

    def _transform_gpu(self, X):
        # Bypass because the data is local
        return X


class ComputeDaskData(Transform):
    """Allow persisting a dask array to memory. It will gather the data blocks
    from all workers and resembles locally.
    """
    def __lazy_transform_generic(self, X):
        if is_dask_array(X) or is_dask_dataframe(X):
            new_data = X.compute()
        else:
            new_data = X

        return new_data

    def _lazy_transform_cpu(self, X):
        return self.__lazy_transform_generic(X)

    def _lazy_transform_gpu(self, X):
        return self.__lazy_transform_generic(X)

    def _transform_cpu(self, X):
        # Bypass because the data is local
        return X

    def _transform_gpu(self, X):
        # Bypass because the data is local
        return X
