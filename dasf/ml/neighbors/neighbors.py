#!/usr/bin/env python3

from sklearn.neighbors import NearestNeighbors as NearestNeighbors_CPU

from dasf.utils.utils import is_gpu_supported
from dasf.transforms.base import Fit
from dasf.transforms.base import GetParams
from dasf.transforms.base import SetParams

try:
    from cuml.neighbors import NearestNeighbors as NearestNeighbors_GPU
except ImportError:
    pass


class NearestNeighbors(Fit, GetParams, SetParams):
    def __init__(self, n_neighbors=5, radius=1.0, algorithm='auto',
                 leaf_size=30, metric='minkowski', p=2,
                 metric_params=None, n_jobs=None, handle=None, verbose=False,
                 output_type=None, **kwargs):

        self.__nn_cpu = NearestNeighbors_CPU(n_neighbors=n_neighbors,
                                             radius=radius,
                                             algorithm=algorithm,
                                             leaf_size=leaf_size,
                                             metric=metric, p=p,
                                             metric_params=metric_params,
                                             n_jobs=n_jobs, **kwargs)

        if is_gpu_supported():
            self.__nn_gpu = NearestNeighbors_GPU(n_neighbors=n_neighbors,
                                                 radius=radius,
                                                 algorithm=algorithm,
                                                 leaf_size=leaf_size,
                                                 metric=metric, p=p,
                                                 metric_params=metric_params,
                                                 n_jobs=n_jobs,
                                                 handle=handle,
                                                 verbose=verbose,
                                                 output_type=output_type,
                                                 **kwargs)

    def _fit_cpu(self, X, y=None, **kwargs):
        return self.__nn_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None, **kwargs):
        return self.__nn_gpu.fit(X=X, **kwargs)

    def _get_params_cpu(self, deep=True, **kwargs):
        return self.__nn_cpu.get_params(deep=deep)

    def _set_params_cpu(self, **params):
        return self.__nn_cpu.set_params(**params)
