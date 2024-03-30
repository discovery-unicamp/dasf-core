#!/usr/bin/env python3

import unittest

import numpy as np
import dask.array as da

from dask.delayed import Delayed
from mock import patch, Mock
from parameterized import parameterized

from distributed.utils import TimeoutError as DistributedTimeoutError

from dasf.utils.funcs import human_readable_size
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.funcs import is_jax_supported
from dasf.utils.funcs import is_dask_local_supported
from dasf.utils.funcs import is_dask_supported
from dasf.utils.funcs import is_dask_gpu_supported
from dasf.utils.funcs import get_gpu_count
from dasf.utils.funcs import get_dask_gpu_count
from dasf.utils.funcs import get_dask_gpu_names
from dasf.utils.funcs import block_chunk_reduce
from dasf.utils.funcs import trim_chunk_location
from dasf.utils.funcs import get_backend_supported
from dasf.utils.funcs import get_worker_info
from dasf.utils.funcs import sync_future_loop
from dasf.utils.funcs import is_notebook
from dasf.utils.funcs import set_executor_default
from dasf.utils.funcs import set_executor_cpu
from dasf.utils.funcs import set_executor_gpu
from dasf.utils.funcs import set_executor_multi_cpu
from dasf.utils.funcs import set_executor_multi_gpu
from dasf.utils.funcs import is_executor_single
from dasf.utils.funcs import is_executor_cluster
from dasf.utils.funcs import is_executor_cpu
from dasf.utils.funcs import is_executor_gpu
from dasf.utils.funcs import executor_to_string
from dasf.utils.funcs import get_machine_memory_avail

from dasf.pipeline.types import TaskExecutorType


class TestArchitetures(unittest.TestCase):
    def test_human_readable_size_bytes(self):
        b = 42
        self.assertTrue(human_readable_size(b).endswith(" B"))

    def test_human_readable_size_kbytes(self):
        kb = 42 * 1000
        self.assertTrue(human_readable_size(kb).endswith(" KB"))

    def test_human_readable_size_mbytes(self):
        mb = 42 * (1000 ** 2)
        self.assertTrue(human_readable_size(mb).endswith(" MB"))

    def test_human_readable_size_gbytes(self):
        b = 42 * (1000 ** 3)
        self.assertTrue(human_readable_size(b).endswith(" GB"))

    def test_human_readable_size_tbytes(self):
        b = 42 * (1000 ** 4)
        self.assertTrue(human_readable_size(b).endswith(" TB"))

    def test_human_readable_size_pbytes(self):
        b = 42 * (1000 ** 5)
        self.assertTrue(human_readable_size(b).endswith(" TB"))

    def test_human_readable_size_decimal(self):
        b = 42 * (1000 ** 3)
        decimal = 5
        number_str = human_readable_size(b, decimal)[:-3]
        self.assertTrue(len(number_str.split(".")[1]) == decimal)

    @patch('dasf.utils.funcs.GPU_SUPPORTED', True)
    @patch('dasf.utils.funcs.get_gpu_count', Mock(return_value=1))
    def test_is_gpu_supported_true(self):
        self.assertTrue(is_gpu_supported())

    @patch('dasf.utils.funcs.GPU_SUPPORTED', False)
    def test_is_gpu_supported_false(self):
        self.assertFalse(is_gpu_supported())

    @patch('dask.config.get', return_value=Mock())
    def test_is_dask_local_supported_true(self, dask_config_get):
        self.assertTrue(is_dask_local_supported())

    @patch('dask.config.get', side_effect=Exception('Test'))
    def test_is_dask_local_supported_false(self, dask_config_get):
        self.assertFalse(is_dask_local_supported())

    @patch('dask.config.get', return_value=Mock())
    def test_is_dask_supported_local_true(self, dask_config_get):
        self.assertTrue(is_dask_supported())

    @patch('dask.distributed.Client.current', return_value=Mock())
    @patch('dasf.utils.funcs.is_executor_cluster', return_value=True)
    def test_is_dask_supported_remote_true(self, client_current, is_executor_cluster):
        self.assertTrue(is_dask_supported())
        is_executor_cluster.assert_called()

    @patch('dask.config.get', side_effect=Exception('Test'))
    @patch('dask.distributed.Client.current', side_effect=Exception('Test'))
    def test_is_dask_supported_false(self, dask_config_get, client_current):
        self.assertFalse(is_dask_supported())

    @patch('dasf.utils.funcs.is_dask_local_supported', side_effect=Exception('Test'))
    def test_is_dask_supported_exception(self, local_supported):
        self.assertFalse(is_dask_supported())

    @patch('dasf.utils.funcs.is_dask_supported', return_value=False)
    def test_is_dask_gpu_supported_false(self, dask_check):
        self.assertFalse(is_dask_gpu_supported())
        dask_check.assert_called()

    @patch('GPUtil.getGPUs', Mock(return_value=[]))
    def test_is_dask_gpu_supported_zero_gpus(self):
        self.assertFalse(is_dask_gpu_supported())

    @patch('dasf.utils.funcs.is_dask_supported', return_value=True)
    @patch('GPUtil.getGPUs', Mock(return_value=[0, 1]))
    def test_is_dask_gpu_supported_nonzero_gpus(self, dask_check):
        self.assertTrue(is_dask_gpu_supported())
        dask_check.assert_called()

    @patch('dasf.utils.funcs.is_dask_supported', return_value=True)
    @patch('dasf.utils.funcs.get_dask_gpu_count', return_value=0)
    def test_is_dask_gpu_supported_zero_gpus_count(self, dask_check, dask_gpu_count):
        self.assertFalse(is_dask_gpu_supported())

    @patch('dasf.utils.funcs.JAX_SUPPORTED', True)
    def test_is_jax_supported_true(self):
        self.assertTrue(is_jax_supported())

    @patch('dasf.utils.funcs.JAX_SUPPORTED', False)
    def test_is_jax_supported_false(self):
        self.assertFalse(is_jax_supported())

    @patch('GPUtil.getGPUs', Mock(return_value=[0, 1]))
    def test_get_gpu_count_2(self):
        self.assertTrue(get_gpu_count() == 2)

    @patch('GPUtil.getGPUs', Mock(return_value=[0, 1]))
    def test_get_dask_gpu_count_2(self):
        self.assertTrue(get_dask_gpu_count() == 2)

    @patch('GPUtil.getGPUs', Mock(return_value=[0, 1]))
    def test_get_dask_gpu_count_2_no_fetch(self):
        self.assertTrue(len(get_dask_gpu_count(fetch=False).compute()) == 2)

    @patch('GPUtil.getGPUs', Mock(return_value=[]))
    def test_get_gpu_count_0(self):
        self.assertTrue(get_gpu_count() == 0)

    @patch('GPUtil.getGPUs', Mock(return_value=[]))
    def test_get_dask_gpu_count_0(self):
        self.assertTrue(get_dask_gpu_count() == 0)

    @patch('GPUtil.getGPUs', Mock(return_value=[]))
    def test_get_dask_gpu_count_0_no_fetch(self):
        self.assertTrue(len(get_dask_gpu_count(fetch=False).compute()) == 0)

    def test_get_dask_gpu_names(self):
        gpu = Mock()
        gpu.configure_mock(name='GPU Foo Bar(tm)',
                           memoryTotal=1024.0)

        with patch('GPUtil.getGPUs', Mock(return_value=[gpu])):
            self.assertEqual(get_dask_gpu_names(fetch=True), ["GPU Foo Bar(tm), 1024.0 MB"])

    def test_get_dask_gpu_names_as_dd(self):
        gpu = Mock()
        gpu.configure_mock(name='GPU Foo Bar(tm)',
                           memoryTotal=1024.0)

        with patch('GPUtil.getGPUs', Mock(return_value=[gpu])):
            self.assertTrue(isinstance(get_dask_gpu_names(fetch=False), Delayed))


class TestBlockChunkReduce(unittest.TestCase):
    def test_different_chunks(self):
        output_size = (100, 20)
        data = da.random.random((20, 30, 40), chunks=(5, 15, 10))

        drop_axis, new_axis = block_chunk_reduce(data, output_size)

        self.assertTrue(np.array_equal(drop_axis, np.asarray([0, 1, 2])))
        self.assertTrue(np.array_equal(new_axis, np.asarray([0, 1])))

    def test_different_chunks_but_same_x(self):
        output_size = (5, 20)
        data = da.random.random((20, 30, 40), chunks=(5, 15, 10))

        drop_axis, new_axis = block_chunk_reduce(data, output_size)

        self.assertTrue(np.array_equal(drop_axis, np.asarray([1, 2])))
        self.assertTrue(np.array_equal(new_axis, np.asarray([1])))

    def test_different_chunks_but_same_y(self):
        output_size = (100, 15)
        data = da.random.random((20, 30, 40), chunks=(5, 15, 10))

        drop_axis, new_axis = block_chunk_reduce(data, output_size)

        self.assertTrue(np.array_equal(drop_axis, np.asarray([0, 2])))
        self.assertTrue(np.array_equal(new_axis, np.asarray([0])))

    def test_different_chunks_but_same_z(self):
        output_size = (100, 10)
        data = da.random.random((20, 30, 40), chunks=(5, 15, 10))

        drop_axis, new_axis = block_chunk_reduce(data, output_size)

        self.assertTrue(np.array_equal(drop_axis, np.asarray([0, 1])))
        self.assertTrue(np.array_equal(new_axis, np.asarray([0])))

    def test_different_chunks_but_greater_than_original(self):
        output_size = (1, 1, 1, 100, 20, 20)
        data = da.random.random((20, 30, 40), chunks=(5, 15, 10))

        drop_axis, new_axis = block_chunk_reduce(data, output_size)

        self.assertTrue(np.array_equal(drop_axis, np.asarray([0, 1, 2])))
        self.assertTrue(np.array_equal(new_axis, np.asarray([0, 1, 2, 3, 4, 5])))

    def test_different_chunks_but_greater_equal_than_original(self):
        output_size = (5, 15, 10, 1, 1, 1)
        data = da.random.random((20, 30, 40), chunks=(5, 15, 10))

        drop_axis, new_axis = block_chunk_reduce(data, output_size)

        self.assertTrue(np.array_equal(drop_axis, np.asarray([])))
        self.assertTrue(np.array_equal(new_axis, np.asarray([3, 4, 5])))

    def test_trim_chunk_location_1d(self):
        depth = (5,)

        block_info = [{'array-location': [(40, 60)], 'chunk-location': (2,)}]

        loc = np.asarray(trim_chunk_location(block_info, depth))

        self.assertTrue(np.array_equal(loc, np.asarray([(20, 30)])))

    def test_trim_chunk_location_2d(self):
        depth = (5, 0)

        block_info = [{'array-location': [(40, 60), (0, 40)], 'chunk-location': (2, 0)}]

        loc = np.asarray(trim_chunk_location(block_info, depth))

        self.assertTrue(np.array_equal(loc, np.asarray([(20, 30), (0, 40)])))

    def test_trim_chunk_location_3d(self):
        depth = (5, 0, 0)

        block_info = [{'array-location': [(40, 60), (0, 40), (0, 40)], 'chunk-location': (2, 0, 0)}]

        loc = np.asarray(trim_chunk_location(block_info, depth))

        self.assertTrue(np.array_equal(loc, np.asarray([(20, 30), (0, 40), (0, 40)])))

    def test_trim_chunk_location_3d_index(self):
        depth = (5, 0, 0)

        block_info = [{}, {}, {}, {}, {}, {'array-location': [(40, 60), (0, 40), (0, 40)], 'chunk-location': (2, 0, 0)}]

        loc = np.asarray(trim_chunk_location(block_info, depth, index=5))

        self.assertTrue(np.array_equal(loc, np.asarray([(20, 30), (0, 40), (0, 40)])))

    def test_trim_chunk_location_3d_no_array_location(self):
        depth = (5, 0, 0)

        block_info = [{}, {}, {}, {}, {}, {'chunk-location': (2, 0, 0)}]

        with self.assertRaises(IndexError) as context:
            loc = np.asarray(trim_chunk_location(block_info, depth, index=5))

        self.assertTrue('Key \'array-location\' was not found in block-info' in str(context.exception))

    def test_trim_chunk_location_3d_no_chunk_location(self):
        depth = (5, 0, 0)

        block_info = [{}, {}, {}, {}, {}, {'array-location': [(40, 60), (0, 40), (0, 40)]}]

        with self.assertRaises(IndexError) as context:
            loc = np.asarray(trim_chunk_location(block_info, depth, index=5))

        self.assertTrue('Key \'chunk-location\' was not found in block-info' in str(context.exception))

    def test_trim_chunk_location_3d_wrong_len(self):
        depth = (5, 0)

        block_info = [{}, {}, {}, {}, {}, {'array-location': [(40, 60), (0, 40), (0, 40)], 'chunk-location': (2, 0, 0)}]

        with self.assertRaises(ValueError) as context:
            loc = np.asarray(trim_chunk_location(block_info, depth, index=5))

        self.assertTrue("Depth 2, location 3 and/or chunks 3 do not match.")


class TestBackendSignature(unittest.TestCase):
    def func1(self, a, b , c, d, e, f):
        return a + b + c + d + e + f

    def func2(self, a, b, backend=None):
        return a + b

    def func3(self, *args, **kwargs):
        return False

    def test_backend_func1(self):
        self.assertFalse(get_backend_supported(self.func1))

    def test_backend_func2(self):
        self.assertTrue(get_backend_supported(self.func2))

    def test_backend_func3(self):
        self.assertFalse(get_backend_supported(self.func3))


class TestWorkerInfo(unittest.TestCase):
    def test_get_worker_info_empty(self):
        client = Mock()
        client.scheduler_info.return_value = {'workers': {}}

        workers = get_worker_info(client)

        self.assertEqual(len(workers), 0)

    def test_get_worker_info_regular(self):
        worker_data = {'workers': {
                           'tcp://127.0.0.1:11111': {
                               'host': '127.0.0.1',
                               'nthreads': 4,
                               },
                           'tcp://127.0.0.1:22222': {
                               'host': '127.0.0.1',
                               'nthreads': 4,
                               }
                           }
                      }
        client = Mock()
        client.scheduler_info.return_value = worker_data

        workers = get_worker_info(client)

        self.assertEqual(len(workers), 2)

        for worker in workers:
            self.assertEqual(worker['local_rank'], 0)
            self.assertEqual(worker['world_size'], 1)


class TestSyncFutureLoop(unittest.TestCase):
    @patch('dasf.utils.funcs.wait', return_value=None)
    def test_sync_future_loop_no_futures(self, wait):
        sync_future_loop(None)

        self.assertFalse(wait.called)

    @patch('dasf.utils.funcs.wait', return_value=Mock())
    def test_sync_future_loop_with_futures(self, wait):
        futures = [None, None]

        result = Mock()

        ret_1 = Mock()
        ret_2 = Mock()

        ret_1.result.return_value = None
        ret_2.result.return_value = None

        result.done = [ret_1, ret_2]
        result.not_done = []

        wait.side_effect = [DistributedTimeoutError(), result]

        sync_future_loop(futures)

        self.assertTrue(wait.called)

    @patch('dasf.utils.funcs.wait', return_value=Mock())
    def test_sync_future_loop_with_futures_and_exception(self, wait):
        futures = [None, None]

        result = Mock()

        ret_1 = Mock()
        ret_2 = Mock()

        ret_1.result.return_value = None
        ret_2.result.side_effect = Exception("Test mock")

        result.done = [ret_1, ret_2]
        result.not_done = []

        wait.side_effect = [DistributedTimeoutError(), result]

        with self.assertRaises(Exception) as context:
            sync_future_loop(futures)

        self.assertTrue('Test mock' in str(context.exception))


class TestIsNotebook(unittest.TestCase):
    @patch('dasf.utils.funcs.get_ipython', return_value=Mock())
    def test_is_notebook_zmq(self, get_ipython):
        get_ipython.return_value.__class__.__name__ = "ZMQInteractiveShell"

        self.assertTrue(is_notebook())

    @patch('dasf.utils.funcs.get_ipython', return_value=Mock())
    def test_is_notebook_terminal(self, get_ipython):
        get_ipython.return_value.__class__.__name__ = "TerminalInteractiveShell"

        self.assertFalse(is_notebook())

    @patch('dasf.utils.funcs.get_ipython', side_effect=NameError())
    def test_is_notebook_exception(self, get_ipython):
        self.assertFalse(is_notebook())


class TestGetMachineMemoryAvailable(unittest.TestCase):
    def test_get_machine_memory_avail(self):
        mem_mock = Mock()
        mem_mock.configure_mock(free=1024)
        with patch('psutil.virtual_memory', return_value=mem_mock):
            self.assertEqual(get_machine_memory_avail(), 1024)


class TestDTypes(unittest.TestCase):
    @parameterized.expand([
       (TaskExecutorType.single_cpu, set_executor_default),
       (TaskExecutorType.single_cpu, set_executor_cpu),
       (TaskExecutorType.single_gpu, set_executor_gpu),
       (TaskExecutorType.multi_cpu, set_executor_multi_cpu),
       (TaskExecutorType.multi_gpu, set_executor_multi_gpu),
    ])
    def test_set_executor(self, dtype, func):
        self.assertEqual(dtype, func())

    @parameterized.expand([
       (TaskExecutorType.single_cpu, is_executor_single, True),
       (TaskExecutorType.single_gpu, is_executor_single, True),
       (TaskExecutorType.multi_cpu, is_executor_single, False),
       (TaskExecutorType.multi_gpu, is_executor_single, False),
       (TaskExecutorType.single_cpu, is_executor_cluster, False),
       (TaskExecutorType.single_gpu, is_executor_cluster, False),
       (TaskExecutorType.multi_cpu, is_executor_cluster, True),
       (TaskExecutorType.multi_gpu, is_executor_cluster, True),
       (TaskExecutorType.single_cpu, is_executor_cpu, True),
       (TaskExecutorType.single_gpu, is_executor_cpu, False),
       (TaskExecutorType.multi_cpu, is_executor_cpu, True),
       (TaskExecutorType.multi_gpu, is_executor_cpu, False),
       (TaskExecutorType.single_cpu, is_executor_gpu, False),
       (TaskExecutorType.single_gpu, is_executor_gpu, True),
       (TaskExecutorType.multi_cpu, is_executor_gpu, False),
       (TaskExecutorType.multi_gpu, is_executor_gpu, True),
    ])
    def test_is_executor(self, dtype, func, ret):
        self.assertEqual(func(dtype), ret)

    @parameterized.expand([
       (TaskExecutorType.single_cpu, "CPU"),
       (TaskExecutorType.single_gpu, "GPU"),
       (TaskExecutorType.multi_cpu, "Multi CPU"),
       (TaskExecutorType.multi_gpu, "Multi GPU"),
    ])
    def test_executor_strings(self, dtype, ret):
        self.assertEqual(executor_to_string(dtype), ret)
