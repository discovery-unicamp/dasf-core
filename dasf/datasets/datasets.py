#!/usr/bin/env python3

from sklearn.datasets import make_blobs as make_blobs_CPU
from dask_ml.datasets import make_blobs as make_blobs_MCPU

try:
    import cupy as cp

    from cuml.datasets import make_blobs as make_blobs_GPU
    from cuml.dask.datasets import make_blobs as make_blobs_MGPU
except ImportError: # pragma: no cover
    pass

from sklearn.datasets import make_classification as make_classification_CPU
from dask_ml.datasets import make_classification as make_classification_MCPU

try:
    from cuml.datasets import make_classification as make_classification_GPU
    from cuml.dask.datasets import make_classification as make_classification_MGPU
except ImportError: # pragma: no cover
    pass

from sklearn.datasets import make_regression as make_regression_CPU
from dask_ml.datasets import make_regression as make_regression_MCPU

try:
    from cuml.datasets import make_regression as make_regression_GPU
    from cuml.dask.datasets import make_regression as make_regression_MGPU
except ImportError: # pragma: no cover
    pass

from dasf.utils.types import is_cpu_array
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.funcs import is_dask_supported
from dasf.utils.funcs import is_dask_gpu_supported


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
