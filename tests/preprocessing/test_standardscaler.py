#!/usr/bin/env python3

import unittest
import numpy as np
import dask.array as da

try:
    import cupy as cp
except ImportError:
    pass

from dasf.ml.preprocessing import StantardScaler
from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_gpu_array
from dasf.utils.types import is_dask_cpu_array
from dasf.utils.types import is_dask_gpu_array
from dasf.utils.funcs import is_gpu_supported


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        size = 20
        self.X = np.array([np.arange(size)])
        self.X.shape = (size, 1)

        mean = np.mean(self.X)
        std = np.std(self.X)

        self.y = (self.X - mean) / std

    def test_standardscaler_cpu(self):
        ss = StantardScaler()

        y = ss._fit_transform_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_standardscaler_mcpu(self):
        ss = StantardScaler()

        y = ss._lazy_fit_transform_cpu(da.from_array(self.X))

        self.assertTrue(is_dask_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_gpu(self):
        ss = StantardScaler()

        y = ss._fit_transform_gpu(cp.asarray(self.X))

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_mgpu(self):
        ss = StantardScaler()

        y = ss._lazy_fit_transform_gpu(da.from_array(cp.asarray(self.X)))

        self.assertTrue(is_dask_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute().get(), equal_nan=True))
