#!/usr/bin/env python3

import unittest

import dask.array as da
import numpy as np
from mock import Mock, patch

try:
    import cupy as cp
except ImportError:
    pass

from dasf.ml.preprocessing import StandardScaler
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import (
    is_cpu_array,
    is_dask_cpu_array,
    is_dask_gpu_array,
    is_gpu_array,
)


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        size = 20
        self.X = np.array([np.arange(size)])
        self.X.shape = (size, 1)

        mean = np.mean(self.X)
        std = np.std(self.X)

        self.y = (self.X - mean) / std

    def test_standardscaler_cpu(self):
        ss = StandardScaler()

        y = ss._fit_transform_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_standardscaler_mcpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_transform_cpu(da.from_array(self.X))

        self.assertTrue(is_dask_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_gpu(self):
        ss = StandardScaler()

        y = ss._fit_transform_gpu(cp.asarray(self.X))

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_mgpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_transform_gpu(da.from_array(cp.asarray(self.X)))

        self.assertTrue(is_dask_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute().get(), equal_nan=True))

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_standardscaler_mcpu_local(self):
        ss = StandardScaler(run_local=True)

        y = ss._fit_transform_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_standardscaler_2_cpu(self):
        ss = StandardScaler()

        y = ss._fit_cpu(self.X)._transform_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_standardscaler_2_mcpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_cpu(da.from_array(self.X))._lazy_transform_cpu(da.from_array(self.X))

        self.assertTrue(is_dask_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_2_gpu(self):
        ss = StandardScaler()

        y = ss._fit_gpu(cp.asarray(self.X))._transform_gpu(cp.asarray(self.X))

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_2_mgpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_gpu(da.from_array(cp.asarray(self.X)))._lazy_transform_gpu(da.from_array(cp.asarray(self.X)))

        self.assertTrue(is_dask_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute().get(), equal_nan=True))

    def test_standardscaler_partial_cpu(self):
        ss = StandardScaler()

        y = ss._partial_fit_cpu(self.X)._transform_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_standardscaler_partial_mcpu(self):
        ss = StandardScaler()

        with self.assertRaises(NotImplementedError) as context:
            y = ss._lazy_partial_fit_cpu(da.from_array(self.X))._lazy_transform_cpu(da.from_array(self.X))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_partial_gpu(self):
        ss = StandardScaler()

        y = ss._partial_fit_gpu(cp.asarray(self.X))._transform_gpu(cp.asarray(self.X))

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_partial_mgpu(self):
        ss = StandardScaler()

        with self.assertRaises(NotImplementedError) as context:
            y = ss._lazy_partial_fit_gpu(da.from_array(cp.asarray(self.X)))._lazy_transform_gpu(da.from_array(cp.asarray(self.X)))

    def test_standardscaler_inverse_cpu(self):
        ss = StandardScaler()

        y = ss._fit_cpu(self.X)._transform_cpu(self.X)

        x = ss._inverse_transform_cpu(y)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.X, x, equal_nan=True))

    def test_standardscaler_inverse_mcpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_cpu(da.from_array(self.X))._lazy_transform_cpu(da.from_array(self.X))

        x = ss._lazy_inverse_transform_cpu(y)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertTrue(np.array_equal(self.X, x.compute(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_inverse_gpu(self):
        ss = StandardScaler()

        y = ss._fit_gpu(cp.asarray(self.X))._transform_gpu(cp.asarray(self.X))

        x = ss._inverse_transform_gpu(y)

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.X, x.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_inverse_mgpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_gpu(da.from_array(cp.asarray(self.X)))._lazy_transform_gpu(da.from_array(cp.asarray(self.X)))

        x = ss._lazy_inverse_transform_gpu(y)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertTrue(np.array_equal(self.X, x.compute().get(), equal_nan=True))
