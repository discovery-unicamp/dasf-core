#!/usr/bin/env python3

import unittest
import numpy as np
import dask.array as da

try:
    import cupy as cp
except ImportError:
    pass

from dasf.utils.decorators import fetch_args_from_dask
from dasf.utils.decorators import fetch_args_from_gpu
from dasf.utils.utils import is_gpu_supported


class TestFetchData(unittest.TestCase):
    def test_fetch_args_from_dask(self):

        @fetch_args_from_dask
        def fake_sum(array):
            return array + array

        array = da.from_array(np.random.random(1000), chunks=(10,))

        ret1 = fake_sum(array)
        ret2 = fake_sum(array=array)

        self.assertTrue(isinstance(ret1, np.ndarray))
        self.assertTrue(isinstance(ret2, np.ndarray))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_fetch_args_from_gpu(self):

        @fetch_args_from_gpu
        def fake_sum(array):
            return array + array

        array = cp.random.random(1000)

        ret1 = fake_sum(array)
        ret2 = fake_sum(array=array)

        self.assertTrue(isinstance(ret1, np.ndarray))
        self.assertTrue(isinstance(ret2, np.ndarray))
