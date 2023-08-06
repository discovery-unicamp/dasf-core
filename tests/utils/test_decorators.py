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
from dasf.utils.decorators import task_handler
from dasf.utils.funcs import is_gpu_supported
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

    def test_fetch_args_from_dask_but_numpy(self):

        @fetch_args_from_dask
        def fake_sum(array):
            return array + array

        array = np.random.random(1000)

        ret1 = fake_sum(array)
        ret2 = fake_sum(array=array)

        self.assertTrue(isinstance(ret1, np.ndarray))
        self.assertTrue(isinstance(ret2, np.ndarray))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_fetch_args_from_gpu_but_numpy(self):

        @fetch_args_from_gpu
        def fake_sum(array):
            return array + array

        array = np.random.random(1000)

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
    def test_task_handler_lazy_gpu_force(self):
        simple_transform = self.generate_simple_transform()

        simple_transform._run_gpu = True

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 1)

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_task_handler_lazy_cpu_force(self):
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

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    def test_task_handler_is_local_and_cpu(self):
        simple_transform = self.generate_simple_transform()

        simple_transform._run_local = True
        simple_transform._run_gpu = False

        X = da.random.random((10, 10, 10))

        self.assertEqual(simple_transform.transform(X), 4)

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_task_handler_only_transform(self):
        simple_transform = Mock()
        simple_transform.transform = Mock(return_value=5)
        simple_transform.transform.__name__ = 'transform'
        simple_transform._run_local = None
        simple_transform._run_gpu = None

        # Remove the possibility to assign _transform_cpu to Mock
        delattr(simple_transform, '_transform_cpu')

        X = da.random.random((10, 10, 10))

        self.assertEqual(task_handler(simple_transform.transform)(simple_transform, X), 5)

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_task_handler_not_implemented(self):
        simple_transform = Mock()
        simple_transform.transform.__name__ = 'transform'

        odd_transform = Mock()
        odd_transform._run_local = None
        odd_transform._run_gpu = None

        # Remove the possibility to assign _transform_cpu and transform to Mock
        delattr(odd_transform, 'transform')
        delattr(odd_transform, '_transform_cpu')

        X = da.random.random((10, 10, 10))

        with self.assertRaises(NotImplementedError) as context:
            task_handler(simple_transform.transform)(odd_transform, X)

        self.assertTrue('There is no implementation of' in str(context.exception))
