#!/usr/bin/env python3

import os
import tempfile
import unittest
import urllib.parse

import networkx as nx
from dask.distributed import Client, LocalCluster
from mock import Mock, patch

from dasf.pipeline.executors import DaskPipelineExecutor, DaskTasksPipelineExecutor
from dasf.pipeline.executors.dask import setup_dask_protocol
from dasf.utils.funcs import is_gpu_supported


class TestDaskProtocol(unittest.TestCase):
    def test_setup_dask_protocol_none(self):
        self.assertEqual(setup_dask_protocol(), "tcp://")

    def test_setup_dask_protocol_tcp(self):
        self.assertEqual(setup_dask_protocol("tcp"), "tcp://")

    def test_setup_dask_protocol_ucx(self):
        self.assertEqual(setup_dask_protocol("ucx"), "ucx://")

    def test_setup_dask_protocol_foo(self):
        with self.assertRaises(Exception) as context:
            setup_dask_protocol("foo")

        self.assertTrue('Protocol foo is not supported.' in str(context.exception))


class TestDaskExecutor(unittest.TestCase):
    def setUp(self):
        self.scheduler_file = os.path.abspath(f"{tempfile.gettempdir()}/scheduler.json")

    def test_dask_executor_remote(self):

        with LocalCluster() as cluster:
            conn = urllib.parse.urlsplit(cluster.scheduler.address)

            dask = DaskPipelineExecutor(address=conn.hostname, port=conn.port)

            # Compute everything to gracefully shutdown
            dask.shutdown(gracefully=True)
            dask.close()

            self.assertFalse(dask.is_connected)

    def test_dask_executor_local_no_args(self):
        dask = DaskPipelineExecutor()

        client = Client.current()

        self.assertEqual(hash(dask.client), hash(client))

        # Compute everything to gracefully shutdown
        client.close()
        dask.shutdown(gracefully=True)
        dask.close()

        self.assertFalse(dask.is_connected)

    def test_dask_executor_local_no_args_no_gracefully(self):
        dask = DaskPipelineExecutor()

        client = Client.current()

        self.assertEqual(hash(dask.client), hash(client))

        # Compute everything to gracefully shutdown
        client.close()
        dask.shutdown(gracefully=False)
        dask.close()

        self.assertFalse(dask.is_connected)

    def test_dask_executor_local(self):
        dask = DaskPipelineExecutor(local=True)

        client = Client.current()

        self.assertTrue(dask.is_connected)
        self.assertEqual(hash(dask.client), hash(client))

        # Compute everything to gracefully shutdown
        client.close()
        dask.shutdown(gracefully=True)
        dask.close()

        self.assertFalse(dask.is_connected)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_dask_executor_local_gpu(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):

            dask = DaskPipelineExecutor(local=True, use_gpu=True)

            self.assertTrue('Available GPUs:' in dask.info)

            client = Client.current()

            self.assertEqual(hash(dask.client), hash(client))
            self.assertGreater(dask.ngpus, 0)

            # Compute everything to gracefully shutdown
            client.close()
            dask.shutdown(gracefully=True)
            dask.close()

            self.assertFalse(dask.is_connected)
            self.assertTrue('Executor is not connected!' in dask.info)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_dask_executor_local_gpu_and_rmm(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):

            dask = DaskPipelineExecutor(local=True, use_gpu=True, gpu_allocator="rmm")

            client = Client.current()

            self.assertEqual(hash(dask.client), hash(client))

            # Compute everything to gracefully shutdown
            client.close()
            dask.shutdown(gracefully=True)
            dask.close()

            self.assertFalse(dask.is_connected)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_dask_executor_local_gpu_and_unknown_allocator(self):
        with self.assertRaises(ValueError) as context:

            dask = DaskPipelineExecutor(local=True, use_gpu=True, gpu_allocator="foo")

            client = Client.current()

            self.assertEqual(hash(dask.client), hash(client))

            # Compute everything to gracefully shutdown
            client.close()
            dask.shutdown(gracefully=True)
            dask.close()

            self.assertTrue('\'foo\' GPU Memory allocator is not known'
                            in str(context.exception))
            self.assertFalse(dask.is_connected)

    def test_dask_executor_scheduler_file(self):
        with LocalCluster() as cluster:
            client = Client(cluster)

            client.write_scheduler_file(self.scheduler_file)

            client_kwargs = {}
            client_kwargs["scheduler_file"] = self.scheduler_file

            client.close()

            dask = DaskPipelineExecutor(client_kwargs=client_kwargs)

            client = Client.current()

            self.assertEqual(hash(dask.client), hash(client))

            # Compute everything to gracefully shutdown
            dask.shutdown(gracefully=True)
            dask.close()

            self.assertFalse(dask.is_connected)

    def tearDown(self):
        if os.path.isfile(self.scheduler_file) or os.path.islink(self.scheduler_file):
            os.remove(self.scheduler_file)


class TestDaskTasksPipelineExecutor(unittest.TestCase):
    def setUp(self):
        self.scheduler_file = os.path.abspath(f"{tempfile.gettempdir()}/scheduler.json")

    def test_dask_tasks_executor_remote(self):

        with LocalCluster() as cluster:
            conn = urllib.parse.urlsplit(cluster.scheduler.address)

            dask = DaskTasksPipelineExecutor(address=conn.hostname,
                                             port=conn.port,
                                             use_gpu=False)

            self.assertTrue('Executor is connected!' in dask.info)
            self.assertTrue('Executor Type: ' in dask.info)
            self.assertTrue('Executor Backend: ' in dask.info)

            # Compute everything to gracefully shutdown
            dask.shutdown(gracefully=True)
            dask.close()

            self.assertFalse(dask.is_connected)
            self.assertTrue('Executor is not connected!' in dask.info)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_dask_tasks_executor_local_gpu(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            dask = DaskTasksPipelineExecutor(local=True, use_gpu=True)

            # Compute everything to gracefully shutdown
            dask.shutdown(gracefully=False)
            dask.close()

            self.assertFalse(dask.is_connected)

    def test_dask_tasks_executor_local_execution(self):

        dask = DaskTasksPipelineExecutor(local=True, use_gpu=False)

        def func1():
            return 2

        def func2(X):
            return X + 4

        def func3(X):
            return X - 4

        def func4(X, Y):
            return X + Y

        pipeline = Mock()
        pipeline._dag = nx.DiGraph([(hash(func1), hash(func2)),
                                    (hash(func1), hash(func3)),
                                    (hash(func2), hash(func4)),
                                    (hash(func3), hash(func4))])

        dask.pre_run(pipeline)

        X_1 = dask.execute(func1)
        X_2 = dask.execute(func2, X_1)
        X_3 = dask.execute(func3, X_1)
        X_4 = dask.execute(func4, X=X_2, Y=X_3)

        self.assertEqual(X_4.result(), 4)

        dask.post_run(pipeline)

        # Compute everything to gracefully shutdown
        dask.shutdown(gracefully=True)
        dask.close()

        self.assertFalse(dask.is_connected)

    def tearDown(self):
        if os.path.isfile(self.scheduler_file) or os.path.islink(self.scheduler_file):
            os.remove(self.scheduler_file)
