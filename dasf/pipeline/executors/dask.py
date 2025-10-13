#!/usr/bin/env python3
""" Dask executor module. """

import os
from typing import Union

try:
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
except ImportError:  # pragma: no cover
    pass

from pathlib import Path

import dask_memusage as dmem
import networkx as nx
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
from dask_jobqueue import PBSCluster
from distributed.diagnostics.plugin import NannyPlugin, WorkerPlugin

from dasf.pipeline.executors.base import Executor
from dasf.pipeline.types import TaskExecutorType
from dasf.utils.funcs import (
    executor_to_string,
    get_dask_gpu_count,
    get_dask_gpu_names,
    get_worker_info,
    is_dask_gpu_supported,
    is_gpu_supported,
    set_executor_gpu,
    set_executor_multi_cpu,
    set_executor_multi_gpu,
)


def setup_dask_protocol(protocol=None):
    """
    Setup the Dask protocol.

    Parameters
    ----------
    protocol : str, optional
        The protocol to use. (default is None)

    Returns
    -------
    str
        The protocol to use.

    Raises
    ------
    ValueError
        If the protocol is not supported.
    """
    if protocol is None or protocol == "tcp":
        return "tcp://"

    if protocol == "ucx":
        return "ucx://"

    raise ValueError(f"Protocol {protocol} is not supported.")


class DaskPipelineExecutor(Executor):
    """
    A pipeline engine based on dask data flow.

    Parameters
    ----------
    address : str, optional
        Address of the Dask scheduler (default is None).
    port : int, optional
        Port of the Dask scheduler (default is 8786).
    local : bool, optional
        Kicks off a new local Dask cluster (default is False).
    use_gpu : bool, optional
        In conjunction with `local`, it kicks off a local CUDA Dask
        cluster (default is False).
    profiler : str, optional
        Sets a Dask profiler.
    protocol : str, optional
        Sets the Dask protocol (default is TCP)
    gpu_allocator : str, optional
        Sets which is the memory allocator for GPU (default is cupy).
    cluster_kwargs : dict, optional
        Extra Dask parameters like memory, processes, etc.
    client_kwargs : dict, optional
        Extra Client parameters.
    """

    def __init__(  # noqa: C901
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
        """
        Constructor of the DaskPipelineExecutor.

        Initializes a Dask-based pipeline executor that can connect to
        existing clusters or create local clusters with CPU or GPU support.

        Parameters
        ----------
        address : str, optional
            Address of the Dask scheduler (default is None).
        port : int, optional
            Port of the Dask scheduler (default is 8786).
        local : bool, optional
            Kicks off a new local Dask cluster (default is False).
        use_gpu : bool, optional
            In conjunction with `local`, it kicks off a local CUDA Dask
            cluster (default is False).
        profiler : str, optional
            Sets a Dask profiler.
        protocol : str, optional
            Sets the Dask protocol (default is TCP).
        gpu_allocator : str, optional
            Sets which is the memory allocator for GPU (default is cupy).
        cluster_kwargs : dict, optional
            Extra Dask parameters like memory, processes, etc.
        client_kwargs : dict, optional
            Extra Client parameters.
        """
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
                # This avoids initializing workers on GPU:0 when available
                self.client = Client(LocalCluster(**cluster_kwargs),
                                     **client_kwargs)

        # Ask workers for GPUs
        if local and not use_gpu:
            self.dtype = set_executor_multi_cpu()
        else:
            # Ask workers for GPUs
            if is_dask_gpu_supported():
                self.dtype = set_executor_multi_gpu()

                if gpu_allocator == "cupy":
                    # Nothing is required yet.
                    pass
                elif gpu_allocator == "rmm" and is_gpu_supported():
                    self.client.run(cp.cuda.set_allocator, rmm_cupy_allocator)
                    rmm.reinitialize(managed_memory=True)
                    cp.cuda.set_allocator(rmm_cupy_allocator)
                else:
                    raise ValueError(f"'{gpu_allocator}' GPU Memory allocator is not "
                                     "known")
            else:
                self.dtype = set_executor_multi_cpu()

        # Share dtype attribute to client
        if not hasattr(self.client, "dtype"):
            setattr(self.client, "dtype", self.dtype)

        # Share which is the default backend of a cluster
        if not hasattr(self.client, "backend"):
            if self.dtype == set_executor_gpu() or \
               self.dtype == set_executor_multi_gpu():
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
    def ngpus(self) -> int:
        """Returns the number of GPUs available in the cluster."""
        return get_dask_gpu_count()

    @property
    def is_connected(self) -> bool:
        """Returns true if the executor is connected to a backend."""
        if "running" in self.client.status:
            return True
        return False

    @property
    def info(self) -> str:
        """Returns a string with the executor information."""
        info = ""
        if self.is_connected:
            info += "Executor is connected!\n"
        else:
            info += "Executor is not connected!\n"

        info += f"Executor Type: {executor_to_string(self.dtype)}\n"
        info += f"Executor Backend: {self.client.backend}\n"

        if self.is_connected and self.ngpus > 0:
            info += f"With {self.ngpus} GPUs\n"

            info += "Available GPUs:\n"
            for gpu_name in list(set(get_dask_gpu_names())):
                info += f"- {gpu_name}\n"
        return info

    def execute(self, fn, *args, **kwargs):
        """Executes a function in the executor.

        Parameters
        ----------
        fn : function
            The function to be executed.
        args : list
            The arguments of the function.
        kwargs : dict
            The keyword arguments of the function.
        """
        return fn(*args, **kwargs)

    def register_plugin(self, plugin: Union[WorkerPlugin,
                                            NannyPlugin]):
        """Registers a plugin in the executor.

        Parameters
        ----------
        plugin : Union[WorkerPlugin, NannyPlugin]
            The plugin to be registered.
        """
        if isinstance(plugin, WorkerPlugin):
            self.client.register_worker_plugin(plugin)
        elif isinstance(plugin, NannyPlugin):
            self.client.register_worker_plugin(plugin, nanny=True)

    def register_dataset(self, **kwargs):
        """Registers a dataset in the executor."""
        self.client.publish_dataset(**kwargs)

    def has_dataset(self, key):
        """Returns true if a dataset is registered in the executor."""
        return key in self.client.list_datasets()

    def get_dataset(self, key):
        """Gets a dataset from the executor."""
        return self.client.get_dataset(name=key)

    def shutdown(self, gracefully=True):
        """Shutdowns the executor.

        Parameters
        ----------
        gracefully : bool, optional
            If true, it will try to shutdown the executor gracefully.
            (default is True)
        """
        if gracefully:
            info = get_worker_info(self.client)

            worker_names = []
            for worker in info:
                worker_names.append(worker["worker"])

            if worker_names:
                self.client.retire_workers(worker_names, close_workers=True)
        else:
            self.client.shutdown()

    def close(self):
        """Closes the executor."""
        self.client.close()


class DaskTasksPipelineExecutor(DaskPipelineExecutor):
    """
    A not centric execution engine based on dask.

    Parameters
    ----------
    address : str, optional
        Address of the Dask scheduler (default is None).
    port : int, optional
        Port of the Dask scheduler (default is 8786).
    local : bool, optional
        Kicks off a new local Dask cluster (default is False).
    use_gpu : bool, optional
        In conjunction with `local`, it kicks off a local CUDA Dask
        cluster (default is False).
    profiler : str, optional
        Sets a Dask profiler.
    gpu_allocator : str, optional
        Sets which is the memory allocator for GPU (default is cupy).
    cluster_kwargs : dict, optional
        Extra Dask parameters like memory, processes, etc.
    client_kwargs : dict, optional
        Extra Client parameters.
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
        """
        Constructor of the DaskTasksPipelineExecutor.

        Initializes a task-specific Dask executor that distributes individual
        tasks across workers rather than using centralized data management.

        Parameters
        ----------
        address : str, optional
            Address of the Dask scheduler (default is None).
        port : int, optional
            Port of the Dask scheduler (default is 8786).
        local : bool, optional
            Kicks off a new local Dask cluster (default is False).
        use_gpu : bool, optional
            In conjunction with `local`, it kicks off a local CUDA Dask
            cluster (default is True).
        profiler : str, optional
            Sets a Dask profiler.
        protocol : str, optional
            Sets the Dask protocol.
        gpu_allocator : str, optional
            Sets which is the memory allocator for GPU (default is cupy).
        cluster_kwargs : dict, optional
            Extra Dask parameters like memory, processes, etc.
        client_kwargs : dict, optional
            Extra Client parameters.
        """
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
        setattr(self.client, "dtype", self.dtype)

        self._tasks_map = dict()

    def pre_run(self, pipeline):
        """Executes before the pipeline starts.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to be executed.
        """
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
        """Executes after the pipeline finishes.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline that was executed.
        """
        pass

    def execute(self, fn, *args, **kwargs):
        """Executes a function in the executor.

        Parameters
        ----------
        fn : function
            The function to be executed.
        args : list
            The arguments of the function.
        kwargs : dict
            The keyword arguments of the function.
        """
        key = hash(fn)

        worker = self._tasks_map[key]["worker"]

        return self.client.submit(fn, *args, **kwargs, workers=[worker])

    def register_dataset(self, **kwargs):
        """Registers a dataset in the executor."""
        self.client.publish_dataset(**kwargs)

    def has_dataset(self, key):
        """Returns true if a dataset is registered in the executor."""
        return key in self.client.list_datasets()

    def get_dataset(self, key):
        """Gets a dataset from the executor."""
        return self.client.get_dataset(name=key)

    def shutdown(self, gracefully=True):
        """Shutdowns the executor.

        Parameters
        ----------
        gracefully : bool, optional
            If true, it will try to shutdown the executor gracefully.
            (default is True)
        """
        if gracefully:
            info = get_worker_info(self.client)

            worker_names = []
            for worker in info:
                worker_names.append(worker["worker"])

            if worker_names:
                self.client.retire_workers(worker_names, close_workers=True)
        else:
            self.client.shutdown()

    def close(self):
        """Closes the executor."""
        self.client.close()


class DaskPBSPipelineExecutor(Executor):
    """A PBS-based execution engine based on dask.

    Parameters
    ----------
    kwargs : dict
        The arguments to be passed to the PBSCluster.
    """
    def __init__(self, **kwargs):
        """
        Constructor of the DaskPBSPipelineExecutor.

        Initializes a PBS (Portable Batch System) based Dask executor
        for running pipelines on HPC clusters.

        Parameters
        ----------
        **kwargs
            Arguments to be passed to the PBSCluster constructor.
            See dask_jobqueue.PBSCluster documentation for available options.
        """
        self.client = Client(PBSCluster(**kwargs))

        # Ask workers for GPUs
        if is_dask_gpu_supported():
            self.dtype = TaskExecutorType.multi_gpu
        else:
            self.dtype = TaskExecutorType.multi_cpu
