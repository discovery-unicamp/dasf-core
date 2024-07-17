#!/usr/bin/env python3

import unittest

import dask.array as da
import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

from mock import Mock, patch
from sklearn.datasets import make_blobs

from dasf.ml.cluster import SOM
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import (
    is_cpu_array,
    is_dask_cpu_array,
    is_dask_gpu_array,
    is_gpu_array,
)


class TestSOM(unittest.TestCase):
    def setUp(self):
        self.size = 1000
        self.centers = 3
        random_state = 42

        self.X, self.y, self.centroids = make_blobs(n_samples=self.size,
                                                    centers=self.centers,
                                                    n_features=2,
                                                    return_centers=True,
                                                    random_state=random_state)

        # We use a fixed seed to avoid outliers between test stanzas
        np.random.seed(1234)

        if is_gpu_supported():
            cp.random.seed(1234)

    def test_som_cpu(self):
        som = SOM(x=3, y=2, input_len=2, num_epochs=300)

        q1 = som._quantization_error_cpu(self.X)

        som._fit_cpu(self.X)

        y = som._predict_cpu(self.X)

        q2 = som._quantization_error_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(q1 > q2)

    def test_som_mcpu(self):
        som = SOM(x=3, y=2, input_len=2, num_epochs=300)

        da_X = da.from_array(self.X, meta=np.array((), dtype=np.float32))

        q1 = som._lazy_quantization_error_cpu(da_X)

        som._lazy_fit_cpu(da_X)

        y = som._lazy_predict_cpu(da_X)

        q2 = som._lazy_quantization_error_cpu(da_X)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertTrue(q1 > q2)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_som_gpu(self):
        som = SOM(x=3, y=2, input_len=2, num_epochs=300)

        cp_X = cp.asarray(self.X)

        q1 = som._quantization_error_gpu(cp_X)

        som._fit_gpu(cp_X)

        y = som._predict_gpu(cp_X)

        q2 = som._quantization_error_gpu(cp_X)

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(q1 > q2)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_som_mgpu(self):
        som = SOM(x=3, y=2, input_len=2, num_epochs=300)

        da_X = da.from_array(cp.asarray(self.X), meta=cp.array((), dtype=cp.float32))

        q1 = som._lazy_quantization_error_gpu(da_X)

        som._lazy_fit_gpu(da_X)

        y = som._lazy_predict_gpu(da_X)

        q2 = som._lazy_quantization_error_gpu(da_X)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertTrue(q1 > q2)

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_som_mcpu_local(self):
        som = SOM(x=3, y=2, input_len=2, num_epochs=300, run_local=True)

        da_X = da.from_array(self.X, meta=np.array((), dtype=np.float32))

        q1 = som.quantization_error(da_X)

        som.fit(da_X)

        y = som.predict(da_X)

        self.assertTrue(is_cpu_array(y))

        q2 = som.quantization_error(da_X)

        self.assertTrue(q1 > q2)
