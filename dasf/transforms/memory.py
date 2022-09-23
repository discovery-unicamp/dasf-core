#!/usr/bin/env python3

import dask.array as da
import dask.dataframe as ddf

from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_dataframe
from dasf.transforms.transforms import _Transform


class PersistDaskData(_Transform):
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


class LoadDaskData(_Transform):
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
