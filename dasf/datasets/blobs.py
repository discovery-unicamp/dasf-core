#!/usr/bin/env python3

from sklearn.datasets import make_blobs as make_blobs_CPU
from dask_ml.datasets import make_blobs as make_blobs_MCPU

try:
    from cuml.datasets import make_blobs as make_blobs_GPU
    from cuml.dask.datasets import make_blobs as make_blobs_MGPU
except ImportError:
    pass

from dasf.utils.utils import is_gpu_supported
from dasf.utils.utils import is_dask_supported
from dasf.utils.utils import is_dask_gpu_supported


class make_blobs:
    """Singleton class used to generate isotropic Gaussian blobs for clustering.
    It automatically selects the implementation based on hardware and available
    libraries and return a container suitable for it (cupy, numpy, cupy+dask or
    numpy+dask).

    The class implements `__call__` being a callable object. 
    """

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
            return self._lazy_make_blobs_gpu(**kwargs)
        elif is_dask_supported():
            return self._lazy_make_blobs_cpu(**kwargs)
        elif is_gpu_supported():
            return self._make_blobs_gpu(**kwargs)
        else:
            return self._make_blobs_cpu(**kwargs)
