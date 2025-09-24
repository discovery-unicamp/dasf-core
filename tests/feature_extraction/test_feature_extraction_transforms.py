#!/usr/bin/env python3

import unittest

import dask.array as da
import dask.dataframe as ddf
import numpy as np
import pandas as pd

try:
    import GPUtil
    assert len(GPUtil.getGPUs()) != 0 # check if GPU are available in current env
    import cudf
    import cupy as cp
    import dask_cudf as cuddf
except:
    pass

from dasf.feature_extraction import (
    ConcatenateToArray,
    GetSubCubeArray,
    GetSubDataframe,
    SampleDataframe,
)
from dasf.utils.funcs import is_gpu_supported


class TestConcatenateToArray(unittest.TestCase):
    def test_concatenate_to_array_cpu(self):
        data1 = np.random.random((40, 40, 40))
        data2 = np.random.random((40, 40, 40))
        data3 = np.random.random((40, 40, 40))

        T = ConcatenateToArray()

        T_1 = T._transform_cpu(data1=data1, data2=data2, data3=data3)

        self.assertEqual(T_1.shape, (40, 40, 40, 3))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_concatenate_to_array_gpu(self):
        data1 = cp.random.random((40, 40, 40))
        data2 = cp.random.random((40, 40, 40))
        data3 = cp.random.random((40, 40, 40))

        T = ConcatenateToArray()

        T_1 = T._transform_gpu(data1=data1, data2=data2, data3=data3)

        self.assertEqual(T_1.shape, (40, 40, 40, 3))

    def test_concatenate_to_array_flat_cpu(self):
        data1 = np.random.random((40, 40, 40))
        data2 = np.random.random((40, 40, 40))
        data3 = np.random.random((40, 40, 40))

        T = ConcatenateToArray(flatten=True)

        T_1 = T._transform_cpu(data1=data1, data2=data2, data3=data3)

        self.assertEqual(T_1.shape, (64000, 3))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_concatenate_to_array_flat_gpu(self):
        data1 = cp.random.random((40, 40, 40))
        data2 = cp.random.random((40, 40, 40))
        data3 = cp.random.random((40, 40, 40))

        T = ConcatenateToArray(flatten=True)

        T_1 = T._transform_gpu(data1=data1, data2=data2, data3=data3)

        self.assertEqual(T_1.shape, (64000, 3))


class TestSampleDataframe(unittest.TestCase):
    def test_sample_dataframe_cpu(self):
        data = pd.DataFrame(np.random.random((100)), columns=['A'])

        sample = SampleDataframe(75.0)

        new = sample.transform(X=data)

        self.assertTrue(70 <= len(new) <= 80)

    def test_sample_dataframe_mcpu(self):
        data = ddf.from_pandas(pd.DataFrame(np.random.random((100)), columns=['A']), npartitions=4)

        sample = SampleDataframe(75.0)

        new = sample.transform(X=data)

        # Dask usually returns values depending on the partition size
        self.assertTrue(70 <= len(new) <= 80)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_sample_dataframe_gpu(self):
        data = cudf.DataFrame(cp.random.random((100)), columns=['A'])

        sample = SampleDataframe(75.0)

        new = sample.transform(X=data)

        self.assertTrue(70 <= len(new) <= 80)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_sample_dataframe_mgpu(self):
        data = cuddf.from_cudf(cudf.DataFrame(cp.random.random((100)), columns=['A']), npartitions=4)

        sample = SampleDataframe(75.0)

        new = sample.transform(X=data)

        # Dask usually returns values depending on the partition size
        self.assertTrue(70 <= len(new) <= 80)

    def test_sample_dataframe_with_array(self):
        data = np.random.random((40, 40, 40))

        sample = SampleDataframe(75.0)

        self.assertRaises(ValueError, sample.transform, X=data)


class TestGetSubDataframe(unittest.TestCase):
    def test_get_sub_dataframe_cpu(self):
        data = pd.DataFrame(np.random.random((100)), columns=['A'])

        sample = GetSubDataframe(75.0)

        new = sample.transform(X=data)

        self.assertEqual(len(new), 75)

    def test_get_sub_dataframe_mcpu(self):
        data = ddf.from_pandas(pd.DataFrame(np.random.random((100)), columns=['A']), npartitions=4)

        sample = GetSubDataframe(75.0)

        self.assertRaises(NotImplementedError, sample.transform, X=data)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_get_sub_dataframe_gpu(self):
        data = cudf.DataFrame(cp.random.random((100)), columns=['A'])

        sample = GetSubDataframe(75.0)

        new = sample.transform(X=data)

        self.assertEqual(len(new), 75)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_get_sub_dataframe_mgpu(self):
        data = cuddf.from_cudf(cudf.DataFrame(cp.random.random((100)), columns=['A']), npartitions=4)

        sample = GetSubDataframe(75.0)

        self.assertRaises(NotImplementedError, sample.transform, X=data)


class TestGetSubCubeArray(unittest.TestCase):
    def test_get_sub_cube_array_cpu(self):
        data = np.random.random((40, 40, 40))

        subcube = GetSubCubeArray(75.0)

        new = subcube.transform(X=data)

        self.assertEqual(new.shape, (30, 30, 30))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_get_sub_cube_array_gpu(self):
        data = cp.random.random((40, 40, 40))

        subcube = GetSubCubeArray(75.0)

        new = subcube.transform(X=data)

        self.assertEqual(new.shape, (30, 30, 30))

    def test_get_sub_cube_array_mcpu(self):
        data = da.random.random((40, 40, 40))

        subcube = GetSubCubeArray(75.0)

        new = subcube.transform(X=data)

        self.assertEqual(new.shape, (30, 30, 30))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_concatenate_to_array_flat_gpu(self):
        data = cp.random.random((40, 40, 40))
        data = da.from_array(data)

        subcube = GetSubCubeArray(75.0)

        new = subcube.transform(X=data)

        self.assertEqual(new.shape, (30, 30, 30))
