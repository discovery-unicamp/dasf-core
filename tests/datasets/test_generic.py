#!/usr/bin/env python3

import os
import unittest

import numpy as np
from parameterized import parameterized_class
from pytest import fixture

from dasf.datasets import (
    DatasetArray,
    DatasetDataFrame,
    DatasetHDF5,
    DatasetLabeled,
    DatasetParquet,
    DatasetXarray,
    DatasetZarr,
)
from dasf.transforms import ExtractData
from dasf.utils.funcs import is_gpu_supported


def parameterize_dataset_type():
    datasets = [
        {"name": "Array", "cls": "DatasetArray", "file": "Array.npy", "extra_args": {}},
        {"name": "Zarr", "cls": "DatasetZarr", "file": "Zarr.zarr", "extra_args": {}},
        {"name": "HDF5", "cls": "DatasetHDF5", "file": "HDF5.h5", "extra_args": {"dataset_path": "dataset"}},
        {"name": "Xarray", "cls": "DatasetXarray", "file": "Xarray.nc", "extra_args": {"chunks": {"x": 10, "y": 10, "z": 10}}},
        {"name": "DataFrame", "cls": "DatasetDataFrame", "file": "DataFrame.csv", "extra_args": {}},
        {"name": "Parquet", "cls": "DatasetParquet", "file": "Parquet.parquet", "extra_args": {}},
    ]
    
    return datasets
    

@parameterized_class(parameterize_dataset_type())
class TestTypes(unittest.TestCase):
    @fixture(autouse=True)
    def data_dir(self, request):
        filename = request.module.__file__
        self.test_dir, _ = os.path.splitext(filename)
        
    def test_dataset_load(self):
        raw_path = os.path.join(self.test_dir, "simple",
                                self.file)
                                
        dataset = eval(self.cls)(name=self.name, root=raw_path, download=False, **self.extra_args)
        dataset.load()

        self.assertTrue(hasattr(dataset, '_metadata'))
        self.assertTrue("size" in dataset._metadata)


class TestDatasetArray(unittest.TestCase):
    def test_shape(self):
        filename = os.getenv('PYTEST_CURRENT_TEST')
        test_dir, _ = os.path.splitext(filename)
        raw_path = os.path.join(test_dir, "simple", "Array.npy")

        dataset = DatasetArray(name="Array", root=raw_path, download=False)

        self.assertEqual(dataset.shape, (40, 40, 40))

    def test_add(self):
        filename = os.getenv('PYTEST_CURRENT_TEST')
        test_dir, _ = os.path.splitext(filename)
        raw_path = os.path.join(test_dir, "simple", "Array.npy")

        dataset1 = DatasetArray(name="Array", root=raw_path, download=False)
        dataset2 = DatasetArray(name="Array", root=raw_path, download=False)

        dataset1._load_cpu()
        dataset2._load_cpu()

        np1 = np.load(raw_path)
        np2 = np.load(raw_path)

        dataset3 = dataset1 + dataset2

        np3 = np1 + np2

        self.assertTrue(np.array_equal(dataset3, np3))

    def test_sub(self):
        filename = os.getenv('PYTEST_CURRENT_TEST')
        test_dir, _ = os.path.splitext(filename)
        raw_path = os.path.join(test_dir, "simple", "Array.npy")

        dataset1 = DatasetArray(name="Array", root=raw_path, download=False)
        dataset2 = DatasetArray(name="Array", root=raw_path, download=False)

        dataset1._load_cpu()
        dataset2._load_cpu()

        np1 = np.load(raw_path)
        np2 = np.load(raw_path)

        dataset3 = dataset1 - dataset2

        np3 = np1 - np2

        self.assertTrue(np.array_equal(dataset3, np3))

    def test_mul(self):
        filename = os.getenv('PYTEST_CURRENT_TEST')
        test_dir, _ = os.path.splitext(filename)
        raw_path = os.path.join(test_dir, "simple", "Array.npy")

        dataset1 = DatasetArray(name="Array", root=raw_path, download=False)
        dataset2 = DatasetArray(name="Array", root=raw_path, download=False)

        dataset1._load_cpu()
        dataset2._load_cpu()

        np1 = np.load(raw_path)
        np2 = np.load(raw_path)

        dataset3 = dataset1 * dataset2

        np3 = np1 * np2

        self.assertTrue(np.array_equal(dataset3, np3))

#    def test_div(self):
#        filename = os.getenv('PYTEST_CURRENT_TEST')
#        test_dir, _ = os.path.splitext(filename)
#        raw_path = os.path.join(test_dir, "simple", "Array.npy")
#
#        dataset1 = DatasetArray(name="Array", root=raw_path, download=False)
#        dataset2 = DatasetArray(name="Array", root=raw_path, download=False)
#
#        dataset1.load()
#        dataset2.load()
#
#        np1 = np.load(raw_path)
#        np2 = np.load(raw_path)
#
#        dataset3 = dataset1 / dataset2
#
#        np3 = np1 / np2
#
#        self.assertTrue(np.array_equal(dataset3, np3))
#
#    def test_avg(self):
#        filename = os.getenv('PYTEST_CURRENT_TEST')
#        test_dir, _ = os.path.splitext(filename)
#        raw_path = os.path.join(test_dir, "simple", "Array.npy")
#
#        dataset = DatasetArray(name="Array", root=raw_path, download=False)
#
#        dataset.load()
#
#        self.assertEqual(dataset.avg(), 0.0)

    def test_extract_data(self):
        filename = os.getenv('PYTEST_CURRENT_TEST')
        test_dir, _ = os.path.splitext(filename)
        raw_path = os.path.join(test_dir, "simple", "Array.npy")

        dataset = DatasetArray(name="Array", root=raw_path, download=False)
        extract = ExtractData()

        dataset._load_cpu()

        data = extract.transform(X=dataset)

        self.assertTrue(isinstance(data, np.ndarray))

    def test_extract_data_exception(self):
        filename = os.getenv('PYTEST_CURRENT_TEST')
        test_dir, _ = os.path.splitext(filename)
        raw_path = os.path.join(test_dir, "simple", "Array.npy")

        dataset = DatasetArray(name="Array", root=raw_path, download=False)
        extract = ExtractData()

        with self.assertRaises(ValueError) as context:
            data = extract.transform(X=dataset)

            self.assertTrue('Data could not be extracted.' in str(context.exception))
