#!/usr/bin/env python3

import os
import unittest

from mock import patch
from parameterized import parameterized

from dasf.ml.dl.clusters.dask import DaskClusterEnvironment


class TestDaskClusterEnvironment(unittest.TestCase):
    def test_dask_cluster_from_metadata(self):
        metadata = {
            "master": "1.2.3.4",
            "world_size": 0,
            "global_rank": 1,
            "local_rank": 2,
        }

        dask = DaskClusterEnvironment(metadata=metadata)

        self.assertTrue(dask.detect())
        self.assertTrue(dask.creates_processes_externally)
        self.assertFalse(dask.creates_children())
        self.assertEqual(dask.main_address, "1.2.3.4")
        self.assertEqual(dask.main_port, 23456)
        self.assertEqual(dask.world_size(), 0)
        self.assertEqual(dask.global_rank(), 1)
        self.assertEqual(dask.local_rank(), 2)
        self.assertEqual(dask.node_rank(), 1)

    def test_dask_cluster_from_metadata_no_local_rank(self):
        metadata = {
            "master": "1.2.3.4",
        }

        dask = DaskClusterEnvironment(metadata=metadata)

        dask.set_world_size(0)
        dask.set_global_rank(1)

        self.assertTrue(dask.detect())
        self.assertTrue(dask.creates_processes_externally)
        self.assertFalse(dask.creates_children())
        self.assertEqual(dask.main_address, "1.2.3.4")
        self.assertEqual(dask.main_port, 23456)
        self.assertEqual(dask.world_size(), 0)
        self.assertEqual(dask.global_rank(), 1)
        self.assertEqual(dask.local_rank(), 0)
        self.assertEqual(dask.node_rank(), 1)

    def test_dask_cluster_from_os_env(self):
        with patch.dict(os.environ, {'MASTER': '1.2.3.4',
                                     'WORLD_SIZE': '0',
                                     'GLOBAL_RANK': '1',
                                     'LOCAL_RANK': '2'}):
            dask = DaskClusterEnvironment()

            self.assertTrue(dask.detect())
            self.assertTrue(dask.creates_processes_externally)
            self.assertFalse(dask.creates_children())
            self.assertEqual(dask.main_address, "1.2.3.4")
            self.assertEqual(dask.main_port, 23456)
            self.assertEqual(dask.world_size(), 0)
            self.assertEqual(dask.global_rank(), 1)
            self.assertEqual(dask.local_rank(), 2)
            self.assertEqual(dask.node_rank(), 1)

    @parameterized.expand([
        ({'MASTER': '1.2.3.4', 'WORLD_SIZE': '0', 'GLOBAL_RANK': '1', 'LOCAL_RANK': '2'}, True),
        ({'WORLD_SIZE': '0', 'GLOBAL_RANK': '1', 'LOCAL_RANK': '2'}, False),
        ({'MASTER': '1.2.3.4', 'GLOBAL_RANK': '1', 'LOCAL_RANK': '2'}, False),
        ({'MASTER': '1.2.3.4', 'WORLD_SIZE': '0', 'LOCAL_RANK': '2'}, False),
    ])
    def test_dask_cluster_detect(self, os_env, ret):
        with patch.dict(os.environ, os_env):
            dask = DaskClusterEnvironment()

            self.assertEqual(dask.detect(), ret)
