#!/usr/bin/env python3

import unittest

from mock import patch, Mock

try:
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
except ImportError:
    pass

from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_gpu_array
from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_gpu_array

from dasf.datasets import make_blobs


class TestDatasets(unittest.TestCase):
    @patch('dasf.datasets.datasets.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.datasets.datasets.is_dask_supported', Mock(return_value=False))
    @patch('dasf.datasets.datasets.is_dask_gpu_supported', Mock(return_value=False))
    def test_make_blobs_cpu(self):
        n_samples = 500000
        n_bins = 3

        centers = [(-6, -6), (0, 0), (9, 1)]
        X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)

        self.assertTrue(is_cpu_array(X))
        self.assertTrue(is_cpu_array(y))

    @patch('dasf.datasets.datasets.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.datasets.datasets.is_dask_supported', Mock(return_value=True))
    @patch('dasf.datasets.datasets.is_dask_gpu_supported', Mock(return_value=False))
    def test_make_blobs_mcpu(self):
        n_samples = 500000
        n_bins = 3

        centers = [(-6, -6), (0, 0), (9, 1)]
        X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42, chunks=(5000))

        self.assertTrue(is_dask_array(X))
        self.assertTrue(is_dask_array(y))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    @patch('dasf.datasets.datasets.is_gpu_supported', Mock(return_value=True))
    @patch('dasf.datasets.datasets.is_dask_supported', Mock(return_value=False))
    @patch('dasf.datasets.datasets.is_dask_gpu_supported', Mock(return_value=False))
    def test_make_blobs_gpu(self):
        n_samples = 500000
        n_bins = 3

        centers = [(-6, -6), (0, 0), (9, 1)]
        X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)
        
        self.assertTrue(is_gpu_array(X))
        self.assertTrue(is_gpu_array(y))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    @patch('dasf.datasets.datasets.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.datasets.datasets.is_dask_supported', Mock(return_value=False))
    @patch('dasf.datasets.datasets.is_dask_gpu_supported', Mock(return_value=True))
    def test_make_blobs_mgpu(self):
        client = Client(LocalCUDACluster())

        n_samples = 500000
        n_bins = 3

        centers = [(-6, -6), (0, 0), (9, 1)]
        X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)

        print(type(X))

        self.assertTrue(is_dask_gpu_array(X))
        self.assertTrue(is_dask_gpu_array(y))
