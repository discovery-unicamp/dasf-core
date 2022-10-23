#!/usr/bin/env python3

import unittest
import numpy as np
import dask.array as da

try:
    import cupy as cp
except ImportError:
    pass

from mock import patch, Mock

from dasf.utils.decorators import fetch_args_from_dask
from dasf.utils.decorators import fetch_args_from_gpu
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.funcs import is_dask_supported
from dasf.utils.funcs import is_dask_gpu_supported
from dasf.transforms.base import Transform


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


class TestTaskHandler(unittest.TestCase):
    def generate_simple_transform(self):
        simple_transform = Transform()

        setattr(simple_transform, '_run_local', None)
        setattr(simple_transform, '_run_gpu', None)

        simple_transform._lazy_transform_gpu = Mock(return_value=1)
        simple_transform._lazy_transform_cpu = Mock(return_value=2)
        simple_transform._transform_gpu = Mock(return_value=3)
        simple_transform._transform_cpu = Mock(return_value=4)

        return simple_transform

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=True))
    def test_task_handler_lazy_gpu(self):
        simple_transform = self.generate_simple_transform()

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 1)

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_task_handler_lazy_cpu(self):
        simple_transform = self.generate_simple_transform()

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 2)

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_task_handler_gpu(self):
        simple_transform = self.generate_simple_transform()

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 3)

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_task_handler_cpu(self):
        simple_transform = self.generate_simple_transform()

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 4)

    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    def test_task_handler_cpu(self):
        simple_transform = self.generate_simple_transform()

        simple_transform._run_gpu = True

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 1)

    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_task_handler_cpu(self):
        simple_transform = self.generate_simple_transform()

        simple_transform._run_gpu = False

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 2)

    def test_task_handler_is_local_and_gpu(self):
        simple_transform = self.generate_simple_transform()

        simple_transform._run_local = True
        simple_transform._run_gpu = True

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 3)

    def test_task_handler_is_local_and_cpu(self):
        simple_transform = self.generate_simple_transform()

        simple_transform._run_local = True
        simple_transform._run_gpu = False

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 4)
