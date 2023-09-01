#!/usr/bin/env python3

import os

from typing import Union

try:
    import rmm
    import cupy as cp
except ImportError: # pragma: no cover
    pass

import networkx as nx

import dask_memusage as dmem

from pathlib import Path

from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster

from distributed.diagnostics.plugin import WorkerPlugin
from distributed.diagnostics.plugin import NannyPlugin

from dask_jobqueue import PBSCluster

from dasf.pipeline.types import TaskExecutorType
from dasf.pipeline.executors.base import Executor
from dasf.utils.funcs import is_dask_gpu_supported
from dasf.utils.funcs import get_dask_gpu_count
from dasf.utils.funcs import get_worker_info
from dasf.utils.funcs import is_gpu_supported


def setup_dask_protocol(protocol=None):
    if protocol is None or protocol == "tcp":
        return "tcp://"

    if protocol == "ucx":
        return "ucx://"

    raise ValueError(f"Protocol {protocol} is not supported.")


class DaskPipelineExecutor(Executor):
    """
    A pipeline engine based on dask data flow.

    Keyword arguments:
    address -- address of the Dask scheduler (default None).
    port -- port of the Dask scheduler (default 8786).
    local -- kicks off a new local Dask cluster (default False).
    use_gpu -- in conjunction with `local`, it kicks off a local CUDA Dask
                cluster (default False).
    profiler -- sets a Dask profiler.
    protocol -- sets the Dask protocol (default TCP)
    gpu_allocator -- sets which is the memory allocator for GPU (default cupy).
    cluster_kwargs -- extra Dask parameters like memory, processes, etc.
    client_kwargs -- extra Client parameters.
    """

    def __init__(
        self,
        address=None,
        port=8786,
        local=False,
        use_gpu=False,
        profiler=None,
        protocol=None,
        gpu_allocator="cupy",
        cluster_kwargs=None,
        client_kwargs=None,
    ):
        self.address = address
        self.port = port

        if not cluster_kwargs:
            cluster_kwargs = dict()

        if not client_kwargs:
            client_kwargs = dict()

        # If address is not set, consider local
        local = local or (address is None and "scheduler_file" not in client_kwargs)

        if address:
            address = f"{setup_dask_protocol()}{address}:{port}"

            self.client = Client(address=address)
        elif "scheduler_file" in client_kwargs:
            self.client = Client(scheduler_file=client_kwargs["scheduler_file"])
        elif local:
            if use_gpu:
                self.client = Client(
                    LocalCUDACluster(**cluster_kwargs), **client_kwargs
                )
            else:
                self.client = Client(LocalCluster(**cluster_kwargs),
                                     **client_kwargs)

        # Ask workers for GPUs
        if local and not use_gpu:
            self.dtype = TaskExecutorType.multi_cpu
        else:
            # Ask workers for GPUs
            if is_dask_gpu_supported():
                self.dtype = TaskExecutorType.multi_gpu

                if gpu_allocator == "cupy":
                    # Nothing is required yet.
                    pass
                elif gpu_allocator == "rmm" and is_gpu_supported():
                    self.client.run(cp.cuda.set_allocator, rmm.rmm_cupy_allocator)
                    rmm.reinitialize(managed_memory=True)
                    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
                else:
                    raise Exception(f"'{gpu_allocator}' GPU Memory allocator is not "
                                    "known")
            else:
                self.dtype = TaskExecutorType.multi_cpu

        # Share dtype attribute to client
        if not hasattr(self.client, "dtype"):
            setattr(self.client, "dtype", self.dtype)

        # Share which is the default backend of a cluster
        if not hasattr(self.client, "backend"):
            if self.dtype == TaskExecutorType.single_gpu or \
               self.dtype == TaskExecutorType.multi_gpu:
                setattr(self.client, "backend", "cupy")
            else:
                setattr(self.client, "backend", "numpy")

        if profiler == "memusage":
            profiler_dir = os.path.abspath(
                os.path.join(str(Path.home()),
                             "/.cache/dasf/profiler/"))
            os.makedirs(profiler_dir, exist_ok=True)

            dmem.install(
                self.client.cluster.scheduler,
                os.path.abspath(profiler_dir + "/dask-memusage"),
            )

    @property
    def ngpus(self):
        return len(get_dask_gpu_count())

    @property
    def is_connected(self):
        if "running" in self.client.status:
            return True
        return False

    def execute(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def register_plugin(self, plugin: Union[WorkerPlugin,
                                            NannyPlugin]):
        if isinstance(plugin, WorkerPlugin):
            self.client.register_worker_plugin(plugin)
        elif isinstance(plugin, NannyPlugin):
            self.client.register_worker_plugin(plugin, nanny=True)

    def register_dataset(self, **kwargs):
        self.client.publish_dataset(**kwargs)

    def has_dataset(self, key):
        return key in self.client.list_datasets()

    def get_dataset(self, key):
        return self.client.get_dataset(name=key)

    def shutdown(self, gracefully=True):
        if gracefully:
            info = get_worker_info(self.client)

            worker_names = []
            for worker in info:
                worker_names.append(worker["worker"])

            if worker_names:
                self.client.retire_workers(worker_names, close_workers=True)
        else:
            self.client.shutdown()


class DaskTasksPipelineExecutor(DaskPipelineExecutor):
    """
    A not centric execution engine based on dask.

    Keyword arguments:
    address -- address of the Dask scheduler (default None).
    port -- port of the Dask scheduler (default 8786).
    local -- kicks off a new local Dask cluster (default False).
    use_gpu -- in conjunction with `local`, it kicks off a local CUDA Dask
                cluster (default False).
    profiler -- sets a Dask profiler.
    gpu_allocator -- sets which is the memory allocator for GPU (default cupy).
    cluster_kwargs -- extra Dask parameters like memory, processes, etc.
    client_kwargs -- extra Client parameters.
    """
    def __init__(
        self,
        address=None,
        port=8786,
        local=False,
        use_gpu=True,
        profiler=None,
        protocol=None,
        gpu_allocator="cupy",
        cluster_kwargs=None,
        client_kwargs=None,
    ):

        super().__init__(
            address=address,
            port=port,
            local=local,
            use_gpu=use_gpu,
            profiler=profiler,
            protocol=protocol,
            gpu_allocator=gpu_allocator,
            cluster_kwargs=cluster_kwargs,
            client_kwargs=client_kwargs,
        )

        # Ask workers for GPUs
        if use_gpu:
            if is_dask_gpu_supported():
                self.dtype = TaskExecutorType.single_gpu
            else:
                self.dtype = TaskExecutorType.single_cpu
        else:
            self.dtype = TaskExecutorType.single_cpu

        # Share dtype attribute to client
        if not hasattr(self.client, "dtype"):
            setattr(self.client, "dtype", self.dtype)

        self._tasks_map = dict()

    def pre_run(self, pipeline):
        nodes = list(nx.topological_sort(pipeline._dag))

        # TODO: we need to consider other branches for complex pipelines
        dag_paths = nx.all_simple_paths(pipeline._dag, nodes[0], nodes[-1])
        all_paths = []
        for path in dag_paths:
            all_paths.append(path)

        workers = get_worker_info(self.client)

        worker_idx = 0
        for path in all_paths:
            for node in path:
                if node not in self._tasks_map:
                    self._tasks_map[node] = workers[worker_idx]

            # Increment workers to all new path and repeat if there
            # are more paths to assign.
            if worker_idx == len(workers):
                worker_idx = 0
            else:
                worker_idx += 1

    def post_run(self, pipeline):
        pass

    def execute(self, fn, *args, **kwargs):
        key = hash(fn)

        worker = self._tasks_map[key]["worker"]

        return self.client.submit(fn, *args, **kwargs, workers=[worker])

    def register_dataset(self, **kwargs):
        self.client.publish_dataset(**kwargs)

    def has_dataset(self, key):
        return key in self.client.list_datasets()

    def get_dataset(self, key):
        return self.client.get_dataset(name=key)

    def shutdown(self, gracefully=True):
        if gracefully:
            info = get_worker_info(self.client)

            worker_names = []
            for worker in info:
                worker_names.append(worker["worker"])

            if worker_names:
                self.client.retire_workers(worker_names, close_workers=True)
        else:
            self.client.shutdown()


class DaskPBSPipelineExecutor(Executor):
    def __init__(self, **kwargs):
        self.client = Client(PBSCluster(**kwargs))

        # Ask workers for GPUs
        if is_dask_gpu_supported():
            self.dtype = TaskExecutorType.multi_gpu
        else:
            self.dtype = TaskExecutorType.multi_cpu
