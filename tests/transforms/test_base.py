#!/usr/bin/env python3

import os
import shutil
import tempfile
import unittest
import numpy as np
import dask.array as da

try:
    import cupy as cp
except ImportError:
    pass

from dasf.transforms.base import MappedTransform
from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_gpu_array
from dasf.utils.types import is_dask_cpu_array
from dasf.utils.types import is_dask_gpu_array
from dasf.utils.types import is_cpu_dataframe
from dasf.utils.types import is_gpu_dataframe
from dasf.utils.types import is_dask_cpu_dataframe
from dasf.utils.types import is_dask_gpu_dataframe
from dasf.utils.funcs import is_gpu_supported


class TestMappedTransform(unittest.TestCase):
    def __internal_max_function(self, block, nxp=np, block_info=None):
        # Using `nxp` parameter to avoid conflicts
        return nxp.asarray([[nxp.max(block)]])

    def __internal_max_min_function(self, block, nxp=np, block_info=None):
        # Using `nxp` parameter to avoid conflicts
        return nxp.asarray([[nxp.min(block), nxp.max(block)]])

    def test_rechunk_max_min_cpu(self):
        X = np.random.random((40, 40, 40))

        mapped = MappedTransform(function=self.__internal_max_min_function,
                                 output_chunk=(1, 2))

        X_t = mapped._transform_cpu(X, nxp=np)

        # Numpy does not use blocks.
        self.assertEqual(X_t.shape, (1, 2))

        self.assertTrue(is_cpu_array(X_t))
        
    def test_rechunk_max_min_mcpu(self):
        X = da.random.random((40, 40, 40), chunks=(10, 40, 40))

        mapped = MappedTransform(function=self.__internal_max_min_function,
                                 output_chunk=(1, 2))

        X_t = mapped._lazy_transform_cpu(X, nxp=np)

        self.assertEqual(X_t.shape, (1, 2))
        # We split data in 4 chunks of axis 0. If the new chunk size is
        # (1, 2), it means that the final size is (1 * 4, 2).
        self.assertEqual(X_t.compute().shape, (1, 2))

        self.assertTrue(is_dask_cpu_array(X_t))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_rechunk_max_min_gpu(self):
        X = cp.random.random((40, 40, 40))

        mapped = MappedTransform(function=self.__internal_max_min_function,
                                 output_chunk=(1, 2))

        X_t = mapped._transform_gpu(X, nxp=cp)

        # Numpy does not use blocks.
        self.assertEqual(X_t.shape, (1, 2))

        self.assertTrue(is_gpu_array(X_t))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_rechunk_max_min_mgpu(self):
        # We need to convert to Dask to use Cupy random.
        X_cp = cp.random.random((40, 40, 40))

        X = da.from_array(X_cp, chunks=(10, 40, 40), meta=cp.array(()))

        mapped = MappedTransform(function=self.__internal_max_min_function,
                                 output_chunk=(1, 2))

        X_t = mapped._lazy_transform_gpu(X, nxp=cp)

        self.assertEqual(X_t.shape, (1, 2))
        # We split data in 4 chunks of axis 0. If the new chunk size is
        # (1, 2), it means that the final size is (1 * 4, 2).
        self.assertEqual(X_t.compute().get().shape, (1, 2))

        self.assertTrue(is_dask_gpu_array(X_t))

