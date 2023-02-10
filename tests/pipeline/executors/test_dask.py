#!/usr/bin/env python3

import os
import tempfile
import unittest
import urllib.parse

from dask.distributed import Client, LocalCluster

from dasf.pipeline.executors import DaskPipelineExecutor


class TestDaskExecutor(unittest.TestCase):
    def setUp(self):
        self.scheduler_file = os.path.abspath(f"{tempfile.gettempdir()}/scheduler.json")

    def test_dask_executor_remote(self):
        with LocalCluster() as cluster:
            conn = urllib.parse.urlsplit(cluster.scheduler.address)

            dask = DaskPipelineExecutor(address=conn.hostname, port=conn.port)

            # Compute everything to gracefully shutdown
            dask.client.close()

    def test_dask_executor_local_no_args(self):
        dask = DaskPipelineExecutor()

        client = Client.current()

        self.assertEqual(hash(dask.client), hash(client))

        # Compute everything to gracefully shutdown
        client.close()
        dask.client.close()

    def test_dask_executor_local(self):
        dask = DaskPipelineExecutor(local=True)

        client = Client.current()

        self.assertEqual(hash(dask.client), hash(client))

        # Compute everything to gracefully shutdown
        client.close()
        dask.client.close()

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
            dask.client.close()

    def tearDown(self):
        if os.path.isfile(self.scheduler_file) or os.path.islink(self.scheduler_file):
            os.remove(self.scheduler_file)
