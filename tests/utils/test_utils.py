#!/usr/bin/env python3

import unittest

from mock import patch, Mock

from dasf.utils.funcs import human_readable_size
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.funcs import is_dask_local_supported
from dasf.utils.funcs import is_dask_supported
from dasf.utils.funcs import is_dask_gpu_supported
from dasf.utils.funcs import get_gpu_count
from dasf.utils.funcs import get_dask_gpu_count


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

    def test_human_readable_size_decimal(self):
        b = 42 * (1000 ** 3)
        decimal = 5
        number_str = human_readable_size(b, decimal)[:-3]
        self.assertTrue(len(number_str.split(".")[1]) == decimal)

    @patch('dasf.utils.funcs.GPU_SUPPORTED', True)
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

    @patch('GPUtil.getGPUs', Mock(return_value=[0, 1]))
    def test_get_gpu_count_2(self):
        self.assertTrue(get_gpu_count() == 2)

    @patch('GPUtil.getGPUs', Mock(return_value=[0, 1]))
    def test_get_dask_gpu_count_2(self):
        self.assertTrue(get_gpu_count() == 2)

    @patch('GPUtil.getGPUs', Mock(return_value=[]))
    def test_get_gpu_count_0(self):
        self.assertTrue(get_dask_gpu_count() == 0)

    @patch('GPUtil.getGPUs', Mock(return_value=[]))
    def test_get_dask_gpu_count_0(self):
        self.assertTrue(get_dask_gpu_count() == 0)
