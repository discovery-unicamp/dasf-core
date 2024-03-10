#!/usr/bin/env python3

from dask_ml.datasets import make_blobs as make_blobs_MCPU
from sklearn.datasets import make_blobs as make_blobs_CPU

try:
    import cupy as cp
    from cuml.dask.datasets import make_blobs as make_blobs_MGPU
    from cuml.datasets import make_blobs as make_blobs_GPU
except ImportError: # pragma: no cover
    pass

from dask_ml.datasets import make_classification as make_classification_MCPU
from sklearn.datasets import make_classification as make_classification_CPU

try:
    from cuml.dask.datasets import make_classification as make_classification_MGPU
    from cuml.datasets import make_classification as make_classification_GPU
except ImportError: # pragma: no cover
    pass

from dask_ml.datasets import make_regression as make_regression_MCPU
from sklearn.datasets import make_regression as make_regression_CPU

try:
    from cuml.dask.datasets import make_regression as make_regression_MGPU
    from cuml.datasets import make_regression as make_regression_GPU
except ImportError: # pragma: no cover
    pass

from dasf.utils.funcs import is_dask_gpu_supported, is_dask_supported, is_gpu_supported
from dasf.utils.types import is_cpu_array


class make_blobs:
    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        if kwargs is None:
            return instance
        else:
            return instance(**kwargs)

    def _lazy_make_blobs_cpu(self, **kwargs):
        return make_blobs_MCPU(**kwargs)

    def _lazy_make_blobs_gpu(self, **kwargs):
        return make_blobs_MGPU(**kwargs)

    def _make_blobs_cpu(self, **kwargs):
        return make_blobs_CPU(**kwargs)

    def _make_blobs_gpu(self, **kwargs):
        return make_blobs_GPU(**kwargs)

    def __call__(self, **kwargs):
        if is_dask_gpu_supported():
            if "centers" in kwargs and is_cpu_array(kwargs["centers"]):
                kwargs["centers"] = cp.asarray(kwargs["centers"])
            return self._lazy_make_blobs_gpu(**kwargs)
        elif is_dask_supported():
            return self._lazy_make_blobs_cpu(**kwargs)
        elif is_gpu_supported():
            if "centers" in kwargs and is_cpu_array(kwargs["centers"]):
                kwargs["centers"] = cp.asarray(kwargs["centers"])
            return self._make_blobs_gpu(**kwargs)
        else:
            return self._make_blobs_cpu(**kwargs)


class make_classification:
    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        if kwargs is None:
            return instance
        else:
            return instance(**kwargs)

    def _lazy_make_classification_cpu(self, **kwargs):
        return make_classification_MCPU(**kwargs)

    def _lazy_make_classification_gpu(self, **kwargs):
        return make_classification_MGPU(**kwargs)

    def _make_classification_cpu(self, **kwargs):
        return make_classification_CPU(**kwargs)

    def _make_classification_gpu(self, **kwargs):
        return make_classification_GPU(**kwargs)

    def __call__(self, **kwargs):
        if is_dask_gpu_supported():
            return self._lazy_make_classification_gpu(**kwargs)
        elif is_dask_supported():
            return self._lazy_make_classification_cpu(**kwargs)
        elif is_gpu_supported():
            return self._make_classification_gpu(**kwargs)
        else:
            return self._make_classification_cpu(**kwargs)


class make_regression:
    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        if kwargs is None:
            return instance
        else:
            return instance(**kwargs)

    def _lazy_make_regression_cpu(self, **kwargs):
        return make_regression_MCPU(**kwargs)

    def _lazy_make_regression_gpu(self, **kwargs):
        return make_regression_MGPU(**kwargs)

    def _make_regression_cpu(self, **kwargs):
        return make_regression_CPU(**kwargs)

    def _make_regression_gpu(self, **kwargs):
        return make_regression_GPU(**kwargs)

    def __call__(self, **kwargs):
        if is_dask_gpu_supported():
            return self._lazy_make_regression_gpu(**kwargs)
        elif is_dask_supported():
            return self._lazy_make_regression_cpu(**kwargs)
        elif is_gpu_supported():
            return self._make_regression_gpu(**kwargs)
        else:
            return self._make_regression_cpu(**kwargs)
