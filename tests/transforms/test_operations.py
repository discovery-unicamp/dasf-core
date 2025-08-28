#!/usr/bin/env python3

import unittest

import dask.array as da
import dask.dataframe as ddf
import numpy as np
import pandas as pd

try:
    from numba import cuda
    assert len(cuda.gpus) != 0 # check if GPU are available in current env
    import cudf
    import cupy as cp
    import dask_cudf as cuddf
except:
    pass

from mock import MagicMock

from dasf.transforms.operations import (
    Reshape,
    SliceArray,
    SliceArrayByPercent,
    SliceArrayByPercentile,
    Overlap,
    Trim,
    Roll,
)
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import (
    is_cpu_array,
    is_dask_cpu_array,
    is_dask_gpu_array,
    is_gpu_array,
)


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
        data = np.random.random((100,))

        slice_t = SliceArrayByPercent(x=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (25,))

    def test_slice_dask_array_cpu_1d(self):
        data = da.random.random((100,), chunks=(10))

        slice_t = SliceArrayByPercent(x=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (25,))

        res = y.compute()

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_gpu_1d(self):
        data = cp.random.random((100,))

        slice_t = SliceArrayByPercent(x=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (25,))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_dask_array_gpu_1d(self):
        data = cp.random.random((100,))
        data = da.from_array(data, chunks=(10))

        slice_t = SliceArrayByPercent(x=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (25,))

        res = y.compute()

    def test_slice_array_cpu_2d(self):
        data = np.random.random((100, 100))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (25, 25))

    def test_slice_dask_array_cpu_2d(self):
        data = da.random.random((100, 100), chunks=(10, 10))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (25, 25))

        res = y.compute()

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_gpu_2d(self):
        data = cp.random.random((100, 100))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (25, 25))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_dask_array_gpu_2d(self):
        data = cp.random.random((100, 100))
        data = da.from_array(data, chunks=(10, 10))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (25, 25))

        res = y.compute()

    def test_slice_array_cpu_3d(self):
        data = np.random.random((100, 100, 100))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0, z=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (25, 25, 25))

    def test_slice_dask_array_cpu_3d(self):
        data = da.random.random((100, 100, 100), chunks=(10, 10, 10))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0, z=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (25, 25, 25))

        res = y.compute()

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_gpu_3d(self):
        data = cp.random.random((100, 100, 100))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0, z=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (25, 25, 25))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_dask_array_gpu_3d(self):
        data = cp.random.random((100, 100, 100))
        data = da.from_array(data, chunks=(10, 10, 10))

        slice_t = SliceArrayByPercent(x=25.0, y=25.0, z=25.0)

        y = slice_t.transform(data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (25, 25, 25))

        res = y.compute()

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


class TestSliceArrayByPercentile(unittest.TestCase):
    def test_slice_array_by_percentile_cpu(self):
        data = np.arange(40 * 40 * 40).reshape((40, 40, 40))

        slice_p = SliceArrayByPercentile(percentile=90.0)

        y = slice_p._transform_cpu(X=data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (40, 40, 40))
        self.assertGreater(len(np.where(y == 57599.1)), 0)

    def test_slice_array_by_percentile_mcpu(self):
        data = da.from_array(np.arange(40 * 40 * 40).reshape((40, 40, 40)), chunks=(5, 5, 5))

        slice_p = SliceArrayByPercentile(percentile=90.0)

        y = slice_p._lazy_transform_cpu(X=data)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (40, 40, 40))
        self.assertGreater(len(np.where(y.compute() == 63916.6)), 0)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_by_percentile_gpu(self):
        data = cp.arange(40 * 40 * 40).reshape((40, 40, 40))

        slice_p = SliceArrayByPercentile(percentile=90.0)

        y = slice_p._transform_gpu(X=data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (40, 40, 40))
        self.assertGreater(len(cp.where(y == 57599.1)), 0)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_slice_array_by_percentile_mcpu(self):
        data = da.from_array(cp.arange(40 * 40 * 40).reshape((40, 40, 40)), chunks=(5, 5, 5))

        slice_p = SliceArrayByPercentile(percentile=90.0)

        y = slice_p._lazy_transform_gpu(X=data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (40, 40, 40))
        self.assertGreater(len(cp.where(y.compute() == 63916.6)), 0)


class TestOverlap(unittest.TestCase):
    def test_overlap_cpu(self):
        data = np.arange(40 * 40 * 40).reshape((40, 40, 40))

        overlap = Overlap(pad=(2, 3, 5))

        y = overlap._transform_cpu(X=data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (44, 46, 50))

    def test_overlap_mcpu(self):
        data = da.from_array(np.arange(40 * 40 * 40).reshape((40, 40, 40)), chunks=(5, 5, 5))

        overlap = Overlap(pad=(2, 3, 5))

        y = overlap._lazy_transform_cpu(X=data)


        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (72, 88, 120))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_overlap_gpu(self):
        data = cp.arange(40 * 40 * 40).reshape((40, 40, 40))

        overlap = Overlap(pad=(2, 3, 5))

        y = overlap._transform_gpu(X=data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (44, 46, 50))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_overlap_mgpu(self):
        data = da.from_array(cp.arange(40 * 40 * 40).reshape((40, 40, 40)), chunks=(5, 5, 5))

        overlap = Overlap(pad=(2, 3, 5))

        y = overlap._lazy_transform_gpu(X=data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (72, 88, 120))


class TestTrim(unittest.TestCase):
    def test_trim_cpu(self):
        data = np.arange(40 * 40 * 40).reshape((40, 40, 40))

        trim = Trim(trim=(1, 0, 2))

        y = trim._transform_cpu(X=data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (38, 40, 36))

    def test_trim_mcpu(self):
        data = da.from_array(np.arange(40 * 40 * 40).reshape((40, 40, 40)), chunks=(5, 5, 5))

        trim = Trim(trim=(1, 0, 2))

        y = trim._lazy_transform_cpu(X=data)


        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (24, 40, 8))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_trim_gpu(self):
        data = cp.arange(40 * 40 * 40).reshape((40, 40, 40))

        trim = Trim(trim=(1, 0, 2))

        y = trim._transform_gpu(X=data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (38, 40, 36))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_trim_mgpu(self):
        data = da.from_array(cp.arange(40 * 40 * 40).reshape((40, 40, 40)), chunks=(5, 5, 5))

        trim = Trim(trim=(1, 0, 2))

        y = trim._lazy_transform_gpu(X=data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (24, 40, 8))


class TestRoll(unittest.TestCase):
    def test_roll_cpu(self):
        data = np.arange(40 * 40 * 40).reshape((40, 40, 40))

        roll = Roll(shift=(1, 2, 3))

        y = roll._transform_cpu(X=data)

        self.assertTrue(is_cpu_array(y))
        self.assertEqual(y.shape, (40, 40, 40))

    def test_roll_mcpu(self):
        data = da.from_array(np.arange(40 * 40 * 40).reshape((40, 40, 40)), chunks=(5, 5, 5))

        roll = Roll(shift=(1, 2, 3))

        y = roll._lazy_transform_cpu(X=data)


        self.assertTrue(is_dask_cpu_array(y))
        self.assertEqual(y.shape, (40, 40, 40))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_roll_gpu(self):
        data = cp.arange(40 * 40 * 40).reshape((40, 40, 40))

        roll = Roll(shift=(1, 2, 3))

        y = roll._transform_gpu(X=data)

        self.assertTrue(is_gpu_array(y))
        self.assertEqual(y.shape, (40, 40, 40))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_roll_mgpu(self):
        data = da.from_array(cp.arange(40 * 40 * 40).reshape((40, 40, 40)), chunks=(5, 5, 5))

        roll = Roll(shift=(1, 2, 3))

        y = roll._lazy_transform_gpu(X=data)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertEqual(y.shape, (40, 40, 40))
