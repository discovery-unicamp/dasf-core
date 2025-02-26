#!/usr/bin/env python3

import os
import unittest

from mock import Mock, patch

from dasf.pipeline.executors import LocalExecutor
from dasf.pipeline.types import TaskExecutorType
from dasf.utils.funcs import is_gpu_supported


class TestLocalExecutor(unittest.TestCase):
    @patch('dasf.pipeline.executors.wrapper.get_gpu_count', Mock(return_value=0))
    def test_local_executor_no_gpu(self):
        local = LocalExecutor()

        self.assertTrue(local.dtype, TaskExecutorType.single_cpu)
