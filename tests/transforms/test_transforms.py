#!/usr/bin/env python3

import os
import zarr
import shutil
import tempfile
import unittest
import numpy as np
import dask.array as da

try:
    import cupy as cp
except ImportError:
    pass

from dasf.datasets import DatasetArray
from dasf.datasets import DatasetZarr
from dasf.datasets import DatasetHDF5
from dasf.transforms import Normalize
from dasf.transforms import ArrayToZarr
from dasf.transforms import ArrayToHDF5
from dasf.transforms import ZarrToArray
from dasf.transforms import ArraysToDataFrame
from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_gpu_array
from dasf.utils.types import is_dask_cpu_array
from dasf.utils.types import is_dask_gpu_array
from dasf.utils.types import is_cpu_dataframe
from dasf.utils.types import is_gpu_dataframe
from dasf.utils.types import is_dask_cpu_dataframe
from dasf.utils.types import is_dask_gpu_dataframe
from dasf.utils.funcs import is_gpu_supported


class TestNormalize(unittest.TestCase):
    def setUp(self):
        size = 20
        self.X = np.array([np.arange(size)])
        self.X.shape = (size, 1)

        mean = np.mean(self.X)
        std = np.std(self.X)

        self.y = (self.X - mean) / std

    def test_normalize_cpu(self):
        norm = Normalize()

        y = norm.transform(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_normalize_mcpu(self):
        norm = Normalize()

        y = norm.transform(da.from_array(self.X))

        self.assertTrue(is_dask_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_normalize_gpu(self):
        norm = Normalize()

        y = norm.transform(cp.asarray(self.X))

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_normalize_mgpu(self):
        norm = Normalize()

        y = norm.transform(da.from_array(cp.asarray(self.X)))

        self.assertTrue(is_dask_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute().get(), equal_nan=True))


class TestArrayToZarr(unittest.TestCase):
    def setUp(self):
        self.array = os.path.abspath(f"{tempfile.gettempdir()}/array.npy")
        self.zarr = os.path.abspath(f"{tempfile.gettempdir()}/array.zarr")

        random = np.random.random(10000)
        np.save(self.array, random)

    @staticmethod
    def remove(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove dir and all contains
        else:
            raise ValueError("file {} is not a file or dir.".format(path))

    def test_array_to_zarr_cpu(self):
        dataset = DatasetArray(root=self.array, download=False, name="Test Array")

        dataset = dataset._load_cpu()

        T = ArrayToZarr(chunks=(100,))

        T_1 = T._transform_cpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetZarr))

    def test_array_to_zarr_mcpu(self):
        dataset = DatasetArray(root=self.array, download=False, name="Test Array")

        dataset = dataset._lazy_load_cpu()

        T = ArrayToZarr(chunks=(100,))

        T_1 = T._lazy_transform_cpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetZarr))

    def tearDown(self):
        self.remove(self.array)
        self.remove(self.zarr)


class TestArrayToHDF5(unittest.TestCase):
    def setUp(self):
        self.array = os.path.abspath(f"{tempfile.gettempdir()}/array.npy")
        self.hdf5 = os.path.abspath(f"{tempfile.gettempdir()}/array.hdf5")

        random = np.random.random(10000)
        np.save(self.array, random)

    @staticmethod
    def remove(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove dir and all contains
        else:
            raise ValueError("file {} is not a file or dir.".format(path))

    def test_array_to_hdf5_cpu(self):
        dataset = DatasetArray(root=self.array, download=False, name="Test Array")

        dataset = dataset._load_cpu()

        T = ArrayToHDF5(dataset_path="/dataset")

        T_1 = T._transform_cpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetHDF5))

    def test_array_to_hdf5_mcpu(self):
        dataset = DatasetArray(root=self.array, download=False, name="Test Array")

        dataset = dataset._lazy_load_cpu()

        T = ArrayToHDF5(dataset_path="/dataset")

        T_1 = T._lazy_transform_cpu(dataset)

        self.assertTrue(isinstance(T_1, DatasetHDF5))

    def tearDown(self):
        self.remove(self.array)
        self.remove(self.hdf5)


class TestZarrToArray(unittest.TestCase):
    def setUp(self):
        self.array = os.path.abspath(f"{tempfile.gettempdir()}/array.npy")
        self.zarr = os.path.abspath(f"{tempfile.gettempdir()}/array.zarr")

        random = z = zarr.array(np.random.random(10000), chunks=100)
        zarr.save(self.zarr, random)

    @staticmethod
    def remove(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove dir and all contains
        else:
            raise ValueError("file {} is not a file or dir.".format(path))

    def test_zarr_to_array_cpu(self):
        dataset = DatasetZarr(root=self.zarr, download=False, name="Test Zarr")

        dataset = dataset._load_cpu()

        T = ZarrToArray()

        T_1 = T.transform(dataset)

        print(type(T_1))

        self.assertTrue(is_cpu_array(T_1))

    def test_zarr_to_array_mcpu(self):
        dataset = DatasetZarr(root=self.zarr, download=False, name="Test Zarr")

        dataset = dataset._lazy_load_cpu()

        T = ZarrToArray()

        print(dataset._data)

        T_1 = T.transform(dataset)

        self.assertTrue(is_dask_cpu_array(T_1))

    def tearDown(self):
        self.remove(self.array)
        self.remove(self.zarr)


class TestArraysToDataFrame(unittest.TestCase):
    def test_arrays_to_dataframe_cpu(self):
        array_1 = np.random.random((40, 40, 40))
        array_2 = np.random.random((40, 40, 40))
        array_3 = np.random.random((40, 40, 40))

        arr_to_df = ArraysToDataFrame()

        y = arr_to_df._transform_cpu(array_1=array_1,
                                     array_2=array_2,
                                     array_3=array_3)

        self.assertTrue(is_cpu_dataframe(y))

    def test_arrays_to_dataframe_mcpu(self):
        chunks = (10, 10, 10)

        darray_1 = da.random.random((40, 40, 40), chunks=chunks)
        darray_2 = da.random.random((40, 40, 40), chunks=chunks)
        darray_3 = da.random.random((40, 40, 40), chunks=chunks)

        arr_to_df = ArraysToDataFrame()

        y = arr_to_df._lazy_transform_cpu(array_1=darray_1,
                                          array_2=darray_2,
                                          array_3=darray_3)

        self.assertTrue(is_dask_cpu_dataframe(y))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_arrays_to_dataframe_gpu(self):
        array_1 = cp.random.random((40, 40, 40))
        array_2 = cp.random.random((40, 40, 40))
        array_3 = cp.random.random((40, 40, 40))

        arr_to_df = ArraysToDataFrame()

        y = arr_to_df._transform_gpu(array_1=array_1,
                                     array_2=array_2,
                                     array_3=array_3)

        self.assertTrue(is_gpu_dataframe(y))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_arrays_to_dataframe_mgpu(self):
        # Need to generate a Cupy first because Dask random does not accept meta
        array_1 = cp.random.random((40, 40, 40))
        array_2 = cp.random.random((40, 40, 40))
        array_3 = cp.random.random((40, 40, 40))

        chunks = (10, 10, 10)

        darray_1 = da.from_array(array_1, chunks=chunks, meta=cp.array(()))
        darray_2 = da.from_array(array_2, chunks=chunks, meta=cp.array(()))
        darray_3 = da.from_array(array_3, chunks=chunks, meta=cp.array(()))

        arr_to_df = ArraysToDataFrame()

        y = arr_to_df._lazy_transform_gpu(array_1=darray_1,
                                          array_2=darray_2,
                                          array_3=darray_3)

        self.assertTrue(is_dask_gpu_dataframe(y))
