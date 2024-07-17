#!/usr/bin/env python3

import unittest

import dask.array as da
import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

from dasf.transforms import ComputeDaskData, PersistDaskData
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import (
    is_cpu_array,
    is_dask_cpu_array,
    is_dask_gpu_array,
    is_gpu_array,
)


class TestMemory(unittest.TestCase):
    def test_persist_dask_data_cpu(self):
        X = np.random.random((40, 40, 40))

        persist = PersistDaskData()

        result = persist._transform_cpu(X=X)

        self.assertTrue(is_cpu_array(result))

    def test_persist_dask_data_mcpu(self):
        X = da.ones((40, 40, 40), chunks=(10, 10, 10), meta=np.array(()))

        persist = PersistDaskData()

        result = persist._lazy_transform_cpu(X=X)

        self.assertTrue(is_dask_cpu_array(result))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_persist_dask_data_gpu(self):
        X = cp.random.random((40, 40, 40))

        persist = PersistDaskData()

        result = persist._transform_gpu(X=X)

        self.assertTrue(is_gpu_array(result))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_persist_dask_data_mgpu(self):
        X = da.ones((40, 40, 40), chunks=(10, 10, 10), meta=cp.array(()))

        persist = PersistDaskData()

        result = persist._lazy_transform_gpu(X=X)

        self.assertTrue(is_dask_gpu_array(result))

    def test_compute_dask_data_cpu(self):
        X = np.random.random((40, 40, 40))

        persist = ComputeDaskData()

        result = persist._transform_cpu(X=X)

        self.assertTrue(is_cpu_array(result))

    def test_compute_dask_data_mcpu(self):
        X = da.ones((40, 40, 40), chunks=(10, 10, 10), meta=np.array(()))

        persist = ComputeDaskData()

        result = persist._lazy_transform_cpu(X=X)

        self.assertTrue(is_cpu_array(result))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_compute_dask_data_gpu(self):
        X = cp.random.random((40, 40, 40))

        persist = ComputeDaskData()

        result = persist._transform_gpu(X=X)

        self.assertTrue(is_gpu_array(result))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_compute_dask_data_mgpu(self):
        X = da.ones((40, 40, 40), chunks=(10, 10, 10), meta=cp.array(()))

        persist = ComputeDaskData()

        result = persist._lazy_transform_gpu(X=X)

        self.assertTrue(is_gpu_array(result))
