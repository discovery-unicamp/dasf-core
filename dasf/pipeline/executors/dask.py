#!/usr/bin/env python3

import os

import networkx as nx

import dask_memusage as dmem

from pathlib import Path

from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster

from dask_jobqueue import PBSCluster

from dasf.pipeline.types import TaskExecutorType
from dasf.pipeline.executors.base import Executor
from dasf.utils.utils import is_dask_gpu_supported
from dasf.utils.utils import get_dask_gpu_count
from dasf.utils.utils import get_worker_info


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
        cluster_kwargs=None,
        client_kwargs=None,
    ):
        self.address = address
        self.port = port

        # If address is not set, consider local
        local = local or address is None

        if not cluster_kwargs:
            cluster_kwargs = dict()

        if not client_kwargs:
            client_kwargs = dict()

        if address:
            self.client = Client(f"{address}:{port}")
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
            else:
                self.dtype = TaskExecutorType.multi_cpu

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
        cluster_kwargs=None,
        client_kwargs=None,
    ):

        super().__init__(
            address=address,
            port=port,
            local=local,
            use_gpu=use_gpu,
            profiler=profiler,
            cluster_kwargs=cluster_kwargs,
            client_kwargs=client_kwargs,
        )

        os.environ["DASK_TASKS"] = "True"

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


class DaskPBSPipelineExecutor(Executor):
    def __init__(self, **kwargs):
        self.client = Client(PBSCluster(**kwargs))

        # Ask workers for GPUs
        if is_dask_gpu_supported():
            self.dtype = TaskExecutorType.multi_gpu
        else:
            self.dtype = TaskExecutorType.multi_cpu
