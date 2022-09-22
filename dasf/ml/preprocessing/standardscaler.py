#!/usr/bin/env python3

from sklearn.preprocessing import StandardScaler as StandardScaler_CPU
from dask_ml.preprocessing import StandardScaler as StandardScaler_MCPU

from dasf.utils.utils import is_gpu_supported
from dasf.utils.decorators import task_handler

try:
    from cuml.preprocessing import StandardScaler as StandardScaler_GPU
except ImportError:
    pass


class StantardScaler:
    def __init__(self, copy=True, with_mean=True, with_std=True):

        self.__std_scaler_cpu = StandardScaler_CPU(
            copy=copy, with_mean=with_mean, with_std=with_std
        )

        self.__std_scaler_dask = StandardScaler_MCPU(
            copy=copy, with_mean=with_mean, with_std=with_std
        )

        if is_gpu_supported():
            self.__std_scaler_gpu = StandardScaler_GPU(
                copy=copy, with_mean=with_mean, with_std=with_std
            )

    def _lazy_fit_cpu(self, X, y=None):
        return self.__std_scaler_dask.fit(X=X, y=y)

    def _lazy_fit_gpu(self, X, y=None):
        return self.__std_scaler_dask.fit(X=X, y=y)

    def _fit_cpu(self, X, y=None):
        return self.__std_scaler_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None):
        return self.__std_scaler_gpu.fit(X=X, y=y)

    def _lazy_fit_transform_cpu(self, X, y=None):
        return self.__std_scaler_dask.fit_transform(X=X, y=y)

    def _lazy_fit_transform_gpu(self, X, y=None):
        return self.__std_scaler_dask.fit_transform(X=X, y=y)

    def _fit_transform_cpu(self, X, y=None):
        return self.__std_scaler_cpu.fit_transform(X=X, y=y)

    def _fit_transform_gpu(self, X, y=None):
        ret = self.__std_scaler_gpu.fit(X=X, y=y)
        return ret.transform(X=X)

    def _lazy_partial_fit_cpu(self, X, y=None):
        return self.__std_scaler_dask.partial_fit(X=X, y=y)

    def _lazy_partial_fit_gpu(self, X, y=None):
        return self.__std_scaler_dask.partial_fit(X=X, y=y)

    def _fit_partial_cpu(self, X, y=None):
        return self.__std_scaler_cpu.partial_fit(X=X, y=y)

    def _fit_partial_gpu(self, X, y=None):
        return self.__std_scaler_gpu.partial_fit(X=X, y=y)

    def _lazy_transform_cpu(self, X, copy=None):
        return self.__std_scaler_dask.transform(X=X, copy=copy)

    def _lazy_transform_gpu(self, X, copy=None):
        return self.__std_scaler_dask.transform(X=X, copy=copy)

    def _transform_cpu(self, X, copy=None):
        return self.__std_scaler_cpu.transform(X=X, copy=copy)

    def _transform_gpu(self, X, copy=None):
        return self.__std_scaler_gpu.transform(X=X, copy=copy)

    def _lazy_inverse_transform_cpu(self, X, copy=None):
        return self.__std_scaler_dask.inverse_transform(X=X, copy=copy)

    def _lazy_inverse_transform_gpu(self, X, copy=None):
        return self.__std_scaler_dask.inverse_transform(X=X, copy=copy)

    def _inverse_transform_cpu(self, X, copy=None):
        return self.__std_scaler_cpu.inverse_transform(X=X, copy=copy)

    def _inverse_transform_gpu(self, X, copy=None):
        return self.__std_scaler_gpu.inverse_transform(X=X, copy=copy)

    @task_handler
    def fit(self, X, y=None):
        ...

    @task_handler
    def transform(self, X, copy=None):
        ...

    @task_handler
    def fit_transform(self, X, y=None):
        ...

    @task_handler
    def partial_fit(self, X, y=None):
        ...

    @task_handler
    def inverse_transform(self, X, copy=None):
        ...
