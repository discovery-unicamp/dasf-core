#!/usr/bin/env python3

import math

import numpy as np
import pandas as pd

from dasf.utils.types import is_array
from dasf.utils.types import is_dask_array
from dasf.utils.decorators import task_handler


try:
    import cupy as cp
    import cudf
except ImportError:
    pass


class _Fit:
    def _lazy_fit_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _lazy_fit_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    @task_handler
    def fit(self, X, y, sample_weight=None, **kwargs):
        ...


class _FitPredict:
    def _lazy_fit_predict_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _lazy_fit_predict_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_predict_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_predict_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    @task_handler
    def fit_predict(self, X, y, **kwargs):
        ...


class _FitTransform:
    def _lazy_fit_transform_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _lazy_fit_transform_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_transform_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_transform_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    @task_handler
    def fit_transform(self, X, y=None, **kwargs):
        ...


class _Predict:
    def _lazy_predict_cpu(self, X, sample_weight=None, **kwargs):
        raise NotImplementedError

    def _lazy_predict_gpu(self, X, sample_weight=None, **kwargs):
        raise NotImplementedError

    def _predict_cpu(self, X, sample_weight=None, **kwargs):
        raise NotImplementedError

    def _predict_gpu(self, X, sample_weight=None, **kwargs):
        raise NotImplementedError

    @task_handler
    def predict(self, X, sample_weight=None, **kwargs):
        ...


class _GetParams:
    def _lazy_get_params_cpu(self, deep=True, **kwargs):
        raise NotImplementedError

    def _lazy_get_params_gpu(self, deep=True, **kwargs):
        raise NotImplementedError

    def _get_params_cpu(self, deep=True, **kwargs):
        raise NotImplementedError

    def _get_params_gpu(self, deep=True, **kwargs):
        raise NotImplementedError

    @task_handler
    def get_params(self, deep=True, **kwargs):
        ...


class _SetParams:
    def _lazy_set_params_cpu(self, **params):
        raise NotImplementedError

    def _lazy_set_params_gpu(self, **params):
        raise NotImplementedError

    def _set_params_cpu(self, **params):
        raise NotImplementedError

    def _set_params_gpu(self, **params):
        raise NotImplementedError

    @task_handler
    def set_params(self, **params):
        ...


class _Transform(_Fit):
    def _lazy_transform_cpu(self, X, **kwargs):
        raise NotImplementedError

    def _lazy_transform_gpu(self, X, **kwargs):
        raise NotImplementedError

    def _transform_cpu(self, X, **kwargs):
        raise NotImplementedError

    def _transform_gpu(self, X, **kwargs):
        raise NotImplementedError

    @task_handler
    def transform_gpu(self, X, **kwargs):
        ...


class Transform(_Transform):
    pass


class ArraysToDataFrame(Transform):
    def __transform_generic(self, X, y):
        assert len(X) == len(y), "Data and labels should have the same length."

        dfs = None
        for x in X:
            i = X.index(x)

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

    def _lazy_transform_cpu(self, X, y):
        return self.__transform_generic(X, y)

    def _lazy_transform_gpu(self, X, y):
        return self.__transform_generic(X, y)

    def _transform_gpu(self, X, y):
        dfs = self.__transform_generic(X, y)

        if is_array(dfs) and not is_dask_array(dfs):
            datas = cp.stack(dfs, axis=-1)
            datas = cudf.DataFrame(datas, columns=y)
        else:
            datas = dfs

        return datas

    def _transform_cpu(self, X, y):
        dfs = self.__transform_generic(X, y)

        if is_array(dfs) and not is_dask_array(dfs):
            datas = np.stack(dfs, axis=-1)
            datas = pd.DataFrame(datas, columns=y)
        else:
            datas = dfs

        return datas
