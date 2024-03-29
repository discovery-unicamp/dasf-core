#!/usr/bin/env python3

import unittest

from dasf.pipeline.executors.base import Executor


class TestExecutorBase(unittest.TestCase):
    def setUp(self):
        self.executor = Executor()

    def test_connected(self):
        self.assertFalse(self.executor.is_connected)

    def test_info(self):
        self.assertEqual(self.executor.info, "This executor has no info to show.")

    def test_has_dataset(self):
        self.assertFalse(self.executor.has_dataset(None))

    def test_execution(self):
        def foo_bar():
            return "Nothing"

        self.executor.pre_run(None)
        self.executor.execute(fn=foo_bar)
        self.executor.post_run(None)
        self.executor.shutdown()
