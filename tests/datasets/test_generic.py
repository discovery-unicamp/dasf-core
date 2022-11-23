#!/usr/bin/env python3

import os
import unittest

from pytest import fixture
from parameterized import parameterized_class

from dasf.utils.funcs import is_gpu_supported
from dasf.datasets import DatasetArray
from dasf.datasets import DatasetZarr
from dasf.datasets import DatasetHDF5
from dasf.datasets import DatasetXarray
from dasf.datasets import DatasetLabeled
from dasf.datasets import DatasetDataFrame
from dasf.datasets import DatasetParquet


def parameterize_dataset_type():
    datasets = [
        {"name": "Array", "cls": "DatasetArray", "file": "Array.npy", "extra_args": {}},
        {"name": "Zarr", "cls": "DatasetZarr", "file": "Zarr.zarr", "extra_args": {}},
        {"name": "HDF5", "cls": "DatasetHDF5", "file": "HDF5.h5", "extra_args": {"path": "dataset"}},
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
