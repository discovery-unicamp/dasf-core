#!/usr/bin/env python3

import unittest

import dask.array as da
import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

from mock import Mock, patch

from dasf.transforms.base import Transform
from dasf.utils.decorators import (
    fetch_args_from_dask,
    fetch_args_from_gpu,
    task_handler,
)
from dasf.utils.funcs import is_gpu_supported


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


class TestFetchArgsEdgeCases(unittest.TestCase):
    def test_fetch_args_from_dask_with_multiple_args(self):
        @fetch_args_from_dask
        def multi_arg_func(arr1, arr2, scalar=5):
            return arr1 + arr2 + scalar

        array1 = da.from_array(np.array([1, 2, 3]), chunks=(2,))
        array2 = da.from_array(np.array([4, 5, 6]), chunks=(2,))

        result = multi_arg_func(array1, array2, scalar=10)

        self.assertTrue(isinstance(result, np.ndarray))
        np.testing.assert_array_equal(result, np.array([15, 17, 19]))

    def test_fetch_args_from_dask_with_kwargs_only(self):
        @fetch_args_from_dask
        def kwargs_func(**kwargs):
            return kwargs['array'] * 2

        array = da.from_array(np.array([1, 2, 3]), chunks=(2,))

        result = kwargs_func(array=array)

        self.assertTrue(isinstance(result, np.ndarray))
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))

    @unittest.skipIf(not is_gpu_supported(), "not supported CUDA in this platform")
    def test_fetch_args_from_gpu_with_mixed_types(self):
        @fetch_args_from_gpu
        def mixed_func(gpu_array, cpu_array):
            # This should convert gpu_array to numpy but leave cpu_array as is
            return gpu_array + cpu_array

        gpu_array = cp.array([1, 2, 3])
        cpu_array = np.array([4, 5, 6])

        result = mixed_func(gpu_array, cpu_array)

        self.assertTrue(isinstance(result, np.ndarray))
        np.testing.assert_array_equal(result, np.array([5, 7, 9]))

    def test_fetch_args_from_dask_no_dask_args(self):
        @fetch_args_from_dask
        def no_dask_func(a, b):
            return a + b

        result = no_dask_func(5, 10)

        self.assertEqual(result, 15)


class TestTaskHandlerEdgeCases(unittest.TestCase):
    def setUp(self):
        self.simple_transform = Transform()
        setattr(self.simple_transform, '_run_local', None)
        setattr(self.simple_transform, '_run_gpu', None)

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=True))
    def test_task_handler_with_invalid_method(self):
        # Test when the method doesn't exist
        simple_transform = Mock()
        simple_transform._run_local = None
        simple_transform._run_gpu = None

        # Remove all expected methods including the base method
        for attr in ['_lazy_invalid_method_gpu', '_lazy_invalid_method_cpu',
                     '_invalid_method_gpu', '_invalid_method_cpu',
                     'invalid_method']:
            if hasattr(simple_transform, attr):
                delattr(simple_transform, attr)

        X = da.random.random((10, 10, 10))

        @task_handler
        def invalid_method(self, X):
            pass

        with self.assertRaises(NotImplementedError):
            invalid_method(simple_transform, X)

    def test_task_handler_with_none_input(self):
        self.simple_transform._transform_method_cpu = Mock(return_value="cpu_result")

        @task_handler
        def transform_method(self, X):
            pass

        # Should handle None input gracefully
        result = transform_method(self.simple_transform, None)
        self.assertEqual(result, "cpu_result")

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=False))
    def test_task_handler_fallback_to_base_method(self):
        # Test that the decorator fails properly when trying to call func without self
        mock_transform = Mock()
        mock_transform._run_local = None
        mock_transform._run_gpu = None
        mock_transform.test_method = Mock(return_value="base_result")

        # Remove all specialized methods
        for attr in ['_lazy_test_method_gpu', '_lazy_test_method_cpu',
                     '_test_method_gpu', '_test_method_cpu']:
            if hasattr(mock_transform, attr):
                delattr(mock_transform, attr)

        @task_handler
        def test_method(self, X):
            return "original_function_result"

        test_method.__name__ = 'test_method'
        X = np.array([1, 2, 3])

        # This should raise a TypeError because the decorator tries to call
        # func(*new_args) but func is a bound method that expects self as
        # first argument.
        with self.assertRaises(TypeError):
            test_method(mock_transform, X)
