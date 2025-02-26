#!/usr/bin/env python3

import os
import unittest

import numpy as np
from mock import Mock, patch

try:
    import cupy as cp
except ImportError:
    pass

from dasf.pipeline.executors import LocalExecutor
from dasf.pipeline.types import TaskExecutorType
from dasf.utils.funcs import is_gpu_supported


class TestLocalExecutor(unittest.TestCase):
    @patch('dasf.pipeline.executors.wrapper.get_gpu_count', Mock(return_value=0))
    def test_local_executor_no_gpu(self):
        local = LocalExecutor()

        self.assertEqual(local.dtype, TaskExecutorType.single_cpu)
        self.assertEqual(local.get_backend(), np)

    @patch('dasf.pipeline.executors.wrapper.is_gpu_supported', Mock(return_value=False))
    def test_local_executor_no_gpu_but_use_gpu(self):
        local = LocalExecutor(use_gpu=True)

        self.assertEqual(local.dtype, TaskExecutorType.single_cpu)
        self.assertEqual(local.get_backend(), np)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_local_executor_use_gpu(self):
        local = LocalExecutor(use_gpu=True)

        self.assertEqual(local.dtype, TaskExecutorType.single_gpu)
        self.assertEqual(local.get_backend(), cp)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_local_executor_use_gpu_backend_cupy(self):
        local = LocalExecutor(use_gpu=True, backend="cupy")

        self.assertEqual(local.dtype, TaskExecutorType.single_gpu)
        self.assertEqual(local.get_backend(), cp)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_local_executor_use_gpu_backend_cupy(self):
        local = LocalExecutor(backend="cupy")

        self.assertEqual(local.dtype, TaskExecutorType.single_gpu)
        self.assertEqual(local.backend, "cupy")
        self.assertEqual(local.get_backend(), cp)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    @patch('dasf.pipeline.executors.wrapper.rmm.reinitialize')
    def test_local_executor_with_rmm(self, rmm):
        local = LocalExecutor(gpu_allocator="rmm")

        self.assertEqual(local.dtype, TaskExecutorType.single_gpu)
        self.assertEqual(local.get_backend(), cp)

        rmm.assert_called_once_with(managed_memory=True)
