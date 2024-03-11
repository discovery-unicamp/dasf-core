#!/usr/bin/env python3

import unittest

import numpy as np
import dask.array as da
import pandas as pd
import dask.dataframe as ddf

try:
    import cudf
    import cupy as cp
    import dask_cudf as cuddf
except ImportError:
    pass

from mock import MagicMock

from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_gpu_array
from dasf.utils.types import is_dask_cpu_array
from dasf.utils.types import is_dask_gpu_array
from dasf.utils.funcs import is_gpu_supported
from dasf.transforms.operations import Reshape
from dasf.transforms.operations import SliceArray
from dasf.transforms.operations import SliceArrayByPercent


class TestReshape(unittest.TestCase):
    def test_reshape_array_cpu(self):
        data = np.random.random((10, 10, 10))

        reshape = Reshape(shape=(1000))

        y = reshape.fit(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (1000, ))

    def test_reshape_dask_array_cpu(self):
        data = da.random.random((10, 10, 10), chunks=(5, 5, 5))

        reshape = Reshape(shape=(1000))

        y = reshape.fit(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (1000, ))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reshape_array_gpu(self):
        data = cp.random.random((10, 10, 10))

        reshape = Reshape(shape=(1000))

        y = reshape.fit(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (1000, ))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reshape_dask_array_gpu(self):
        data = cp.random.random((10, 10, 10))
        data = da.from_array(data, chunks=(5, 5, 5))

        reshape = Reshape(shape=(1000))

        y = reshape.fit(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (1000, ))

    def test_reshape_array_cpu_from_array(self):
        data = np.random.random((10, 10, 10))
        copy = np.random.random((1000,))

        reshape = Reshape()

        y = reshape.fit(data, y=copy)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (1000, ))

    def test_reshape_dask_array_cpu_from_array(self):
        data = da.random.random((10, 10, 10), chunks=(5, 5, 5))
        copy = da.random.random((1000,), chunks=(5,))

        reshape = Reshape()

        y = reshape.fit(data, y=copy)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (1000, ))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reshape_array_gpu_from_array(self):
        data = cp.random.random((10, 10, 10))
        copy = cp.random.random((1000,))

        reshape = Reshape()

        y = reshape.fit(data, y=copy)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (1000, ))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reshape_dask_array_gpu_from_array(self):
        data = cp.random.random((10, 10, 10))
        data = da.from_array(data, chunks=(5, 5, 5))
        copy = cp.random.random((1000,))

        reshape = Reshape()

        y = reshape.fit(data, y=copy)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (1000, ))

    def test_reshape_array_list(self):
        data = np.random.random((2, 5))
        copy = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        reshape = Reshape()

        with self.assertRaises(Exception) as context:
            y = reshape.fit(data, y=copy)

        self.assertTrue('Missing shape input' in str(context.exception))

    def test_reshape_unknown_datatype(self):
        data = MagicMock(shape=(2, 5))

        reshape = Reshape(shape=(10))

        with self.assertRaises(Exception) as context:
            y = reshape.fit(data)

        self.assertTrue('X is not a known datatype' in str(context.exception))

    def test_reshape_dataframe_cpu(self):
        data = pd.DataFrame(np.random.random((3, 4)), columns=['A', 'B', 'C', 'D'])

        reshape = Reshape(shape=(12))

        y = reshape.fit(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (12, ))

    def test_reshape_dask_dataframe_cpu(self):
        raise unittest.SkipTest("DataFrame in Dask does not return the proper shape to reshape (BUG?)")

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reshape_dataframe_gpu(self):
        data = cudf.DataFrame(cp.random.random((3, 4)), columns=['A', 'B', 'C', 'D'])

        reshape = Reshape(shape=(12))

        y = reshape.fit(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (12, ))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reshape_dask_dataframe_gpu(self):
        raise unittest.SkipTest("DataFrame in Dask does not return the proper shape to reshape (BUG?)")


class TestSliceArray(unittest.TestCase):
    def test_slice_array_cpu_1d(self):
        data = np.random.random((40,))

        slice_t = SliceArray(output_size=(10,))

        y = slice_t.transform(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (10,))

    def test_slice_dask_array_cpu_1d(self):
        data = da.random.random((40,), chunks=(5))

        slice_t = SliceArray(output_size=(10,))

        y = slice_t.transform(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (10,))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_gpu_1d(self):
        data = cp.random.random((40,))

        slice_t = SliceArray(output_size=(10,))

        y = slice_t.transform(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (10,))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_dask_array_gpu_1d(self):
        data = cp.random.random((40,))
        data = da.from_array(data, chunks=(5))

        slice_t = SliceArray(output_size=(10,))

        y = slice_t.transform(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (10,))

    def test_slice_array_cpu_2d(self):
        data = np.random.random((40, 40))

        slice_t = SliceArray(output_size=(10, 10))

        y = slice_t.transform(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (10, 10))

    def test_slice_dask_array_cpu_2d(self):
        data = da.random.random((40, 40), chunks=(5, 5))

        slice_t = SliceArray(output_size=(10, 10))

        y = slice_t.transform(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (10, 10))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_gpu_2d(self):
        data = cp.random.random((40, 40))

        slice_t = SliceArray(output_size=(10, 10))

        y = slice_t.transform(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (10, 10))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_dask_array_gpu_2d(self):
        data = cp.random.random((40, 40))
        data = da.from_array(data, chunks=(5, 5))

        slice_t = SliceArray(output_size=(10, 10))

        y = slice_t.transform(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (10, 10))

    def test_slice_array_cpu_3d(self):
        data = np.random.random((40, 40, 40))

        slice_t = SliceArray(output_size=(10, 10, 10))

        y = slice_t.transform(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (10, 10, 10))

    def test_slice_dask_array_cpu_3d(self):
        data = da.random.random((40, 40, 40), chunks=(5, 5, 5))

        slice_t = SliceArray(output_size=(10, 10, 10))

        y = slice_t.transform(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (10, 10, 10))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_gpu_3d(self):
        data = cp.random.random((40, 40, 40))

        slice_t = SliceArray(output_size=(10, 10, 10))

        y = slice_t.transform(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (10, 10, 10))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_dask_array_gpu_3d(self):
        data = cp.random.random((40, 40, 40))
        data = da.from_array(data, chunks=(5, 5, 5))

        slice_t = SliceArray(output_size=(10, 10, 10))

        y = slice_t.transform(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (10, 10, 10))

    def test_slice_array_unknown_dim(self):
        data = np.random.random((2, 2, 2, 2))

        slice_t = SliceArray(output_size=(1, 1, 1, 1))

        with self.assertRaises(Exception) as context:
            y = slice_t.transform(data)

        self.assertTrue('The dimmension is not known' in str(context.exception))


class TestSliceArrayByPercent(unittest.TestCase):
    def test_slice_array_cpu_1d(self):
        data = np.random.random((40,))

        slice_t = SliceArrayByPercent(x=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (10,))

    def test_slice_dask_array_cpu_1d(self):
        data = da.random.random((40,), chunks=(5))

        slice_t = SliceArrayByPercent(x=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (10,))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_gpu_1d(self):
        data = cp.random.random((40,))

        slice_t = SliceArrayByPercent(x=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (10,))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_dask_array_gpu_1d(self):
        data = cp.random.random((40,))
        data = da.from_array(data, chunks=(5))

        slice_t = SliceArrayByPercent(x=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (10,))

    def test_slice_array_cpu_2d(self):
        data = np.random.random((40, 40))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (10, 10))

    def test_slice_dask_array_cpu_2d(self):
        data = da.random.random((40, 40), chunks=(5, 5))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (10, 10))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_gpu_2d(self):
        data = cp.random.random((40, 40))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (10, 10))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_dask_array_gpu_2d(self):
        data = cp.random.random((40, 40))
        data = da.from_array(data, chunks=(5, 5))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (10, 10))

    def test_slice_array_cpu_3d(self):
        data = np.random.random((40, 40, 40))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0, z=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (10, 10, 10))

    def test_slice_dask_array_cpu_3d(self):
        data = da.random.random((40, 40, 40), chunks=(5, 5, 5))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0, z=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (10, 10, 10))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_gpu_3d(self):
        data = cp.random.random((40, 40, 40))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0, z=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (10, 10, 10))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_dask_array_gpu_3d(self):
        data = cp.random.random((40, 40, 40))
        data = da.from_array(data, chunks=(5, 5, 5))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0, z=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (10, 10, 10))

    def test_slice_array_unknown_dim(self):
        data = np.random.random((2, 2, 2, 2))

        slice_t = SliceArrayByPercent(x=50.0, y=50.0, z=50.0)

        with self.assertRaises(Exception) as context:
            y = slice_t.transform(data)

        self.assertTrue('The dimmension is not known' in str(context.exception))

    def test_slice_array_exceeding_percentage(self):
        data = np.random.random((4, 4, 4))

        slice_t = SliceArrayByPercent(x=150.0, y=150.0, z=50.0)

        with self.assertRaises(Exception) as context:
            y = slice_t.transform(data)

        self.assertTrue('Percentages cannot be higher than 100% (1.0)' in str(context.exception))

    def test_slice_array_zero_percentage(self):
        data = np.random.random((4, 4, 4))

        slice_t = SliceArrayByPercent(x=50.0, y=0.0, z=50.0)

        with self.assertRaises(Exception) as context:
            y = slice_t.transform(data)

        self.assertTrue('Percentages cannot be negative or 0' in str(context.exception))

    def test_slice_array_negative_percentage(self):
        data = np.random.random((4, 4, 4))

        slice_t = SliceArrayByPercent(x=-50.0, y=50.0, z=50.0)

        with self.assertRaises(Exception) as context:
            y = slice_t.transform(data)

        self.assertTrue('Percentages cannot be negative or 0' in str(context.exception))
