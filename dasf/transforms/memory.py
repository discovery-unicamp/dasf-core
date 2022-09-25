#!/usr/bin/env python3

import dask.array as da
import dask.dataframe as ddf

from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_dataframe
from dasf.transforms import Transform


class PersistDaskData(Transform):
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


class LoadDaskData(Transform):
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
