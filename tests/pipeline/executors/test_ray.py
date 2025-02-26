#!/usr/bin/env python3

import os
import socket
import subprocess
import unittest

import ray as ray_default
from mock import Mock, patch

from dasf.pipeline.executors import RayPipelineExecutor
from dasf.pipeline.executors.ray import setup_ray_protocol
from dasf.utils.funcs import is_gpu_supported


class TestRayProtocol(unittest.TestCase):
    def test_setup_ray_protocol_none(self):
        self.assertEqual(setup_ray_protocol(), "")

    def test_setup_ray_protocol_tcp(self):
        self.assertEqual(setup_ray_protocol("tcp"), "")

    def test_setup_ray_protocol_ray(self):
        self.assertEqual(setup_ray_protocol("ray"), "ray://")

    def test_setup_ray_protocol_foo(self):
        with self.assertRaises(Exception) as context:
            setup_ray_protocol("foo")

        self.assertTrue('Protocol foo is not supported.' in str(context.exception))


class TestRayExecutor(unittest.TestCase):
    def test_ray_executor_from_existing_nodes(self):
        proc = subprocess.run(["ray", "start", "--head", "--port=9191"],
                              stdout=subprocess.DEVNULL)

        self.assertEqual(proc.returncode, 0)

        address = socket.gethostbyname(socket.gethostname())

        ray = RayPipelineExecutor(address=address, port=9191)

        self.assertTrue(ray.is_connected)

        # Gracefully close the idle tasks
        ray_default.shutdown()

        proc = subprocess.run(["ray", "stop"],
                              stdout=subprocess.DEVNULL)

        self.assertEqual(proc.returncode, 0)

        self.assertFalse(ray.is_connected)

    def test_ray_executor_local_with_no_args(self):
        ray = RayPipelineExecutor(local=True)

        self.assertTrue(ray.is_connected)

        del ray

    def test_ray_executor_local_with_args(self):
        ray = RayPipelineExecutor(local=True, ray_kwargs={'num_cpus': 2})

        self.assertTrue(ray.is_connected)

        nodes = ray_default.nodes()

        for node in nodes:
            if 'Resources' in node and 'CPU' in node['Resources']:
                self.assertEqual(int(node['Resources']['CPU']), 2)

        del ray

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_ray_executor_local_gpu(self):
        ray = RayPipelineExecutor(local=True, use_gpu=True)

        self.assertTrue(ray.is_connected)

        self.assertGreater(ray.ngpus, 0)

    def tearDown(self):
        ray_default.shutdown()

        rc = subprocess.call(["ray", "stop"])
