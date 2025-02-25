#!/usr/bin/env python3

import os
import ray
import subprocess
import tempfile
import unittest
import urllib.parse

import networkx as nx
from mock import Mock, patch

from dasf.pipeline.executors import RayPipelineExecutor
from dasf.utils.funcs import is_gpu_supported


class TestRayExecutor(unittest.TestCase):
    def test_ray_executor_from_existing_nodes(self):
        rc = subprocess.call(["ray", "start", "--head", "--port=9191"])
        self.assertEqual(rc, 0)

        ray = RayPipelineExecutor(address='127.0.0.1', port=9191)

        self.assertTrue(ray.is_connected)

        rc = subprocess.call(["ray", "stop"])
        self.assertEqual(rc, 0)

        self.assertFalse(ray.is_connected)

    def test_ray_executor_local_with_no_args(self):
        ray = RayPipelineExecutor(local=True)

        self.assertTrue(ray.is_connected)

    def test_ray_executor_local_with_args(self):
        ray = RayPipelineExecutor(local=True, ray_kwargs={'num_cpus': 2})

        self.assertTrue(ray.is_connected)

        nodes = ray.nodes()

        for node in nodes:
            if 'Resources' in node and 'CPU' in node['Resources']:
                self.assertEqual(int(node['Resources']['CPU']), 2)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_ray_executor_local_gpu(self):
        ray = RayPipelineExecutor(local=True, use_gpu=True)

        self.assertTrue(ray.is_connected)

        self.assertGreater(ray.ngpus, 0)
