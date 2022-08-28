#!/usr/bin/env python3

import os
import GPUtil

import dask.delayed as dd
import dask_memusage as dmem

from pathlib import Path

from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster

from dask_jobqueue import PBSCluster

from prefect.executors.local import LocalExecutor
from prefect.executors.dask import DaskExecutor
from prefect.executors.dask import LocalDaskExecutor

from dasf.pipeline.types import TaskExecutorType
from dasf.utils.utils import is_dask_gpu_supported


class DaskPipelineExecutor(LocalExecutor):
    """
    A pipeline engine based on dask.

    Keyword arguments:
    address -- address of the Dask scheduler (default None).
    port -- port of the Dask scheduler (default 8786).
    local -- kicks off a new local Dask cluster (default False).
    use_cuda -- in conjunction with `local`, it kicks off a local CUDA Dask
                cluster (default False).
    profiler -- sets a Dask profiler.
    cluster_kwargs -- extra Dask parameters like memory, processes, etc.
    client_kwargs -- extra Client parameters.
    """
    def __init__(self,
                 address=None,
                 port=8786,
                 local=False,
                 use_cuda=False,
                 profiler=None,
                 cluster_kwargs=None,
                 client_kwargs=None):

        self.address = address
        self.port = port

        # If address is not set, consider local
        local = local or address is None

        if address:
            self.client = Client(f'{address}:{port}')
        elif local:
            if use_cuda:
                self.client = Client(LocalCUDACluster(**cluster_kwargs),
                                     **client_kwargs)
            else:
                self.client = Client(LocalCluster(**cluster_kwargs),
                                     **client_kwargs)

        # Ask workers for GPUs
        if local and not use_cuda:
            self.dtype = TaskExecutorType.multi_cpu
        else:
            # Ask workers for GPUs
            if is_dask_gpu_supported():
                self.dtype = TaskExecutorType.multi_gpu
            else:
                self.dtype = TaskExecutorType.multi_cpu

        if profiler == "memusage":
            profiler_dir = os.path.abspath(str(Path.home()) +
                                           "/.cache/dasf/profiler/")
            os.makedirs(profiler_dir, exist_ok=True)

            dmem.install(self.client.cluster.scheduler,
                         os.path.abspath(profiler_dir + "/dask-memusage"))

    @property
    def ngpus(self):
        return len(self.client.ncores())


class DaskPrefectPipelineExecutor(DaskExecutor):
    """
    A not centric execution engine based on dask.

    address -- address of a currently running dask scheduler (default None).
    cluster_class -- the cluster class to use when creating a temporary Dask
                     cluster (default None).
    cluster_kwargs -- addtional kwargs to pass to the cluster_class when
                      creating a temporary dask cluster (default None).
    adapt_kwargs -- additional kwargs to pass to `cluster.adapt` when creating
                    a temporary dask cluster (default None).
    client_kwargs -- additional kwargs to use when creating a Dask Client
                     (default None).
    debug -- When running with a local cluster, setting `debug=True` will
             increase dask's logging level, providing potentially useful
             debug info (default False).
    performance_report_path -- An optional path for the dask performance
                               report (default None).
    """
    def __init__(self,
                 address=None,
                 cluster_class=None,
                 cluster_kwargs=None,
                 adapt_kwargs=None,
                 client_kwargs=None,
                 debug=False,
                 performance_report_path=None):

        super().__init__(address=address, cluster_class=cluster_class,
                         cluster_kwargs=cluster_kwargs,
                         adapt_kwargs=adapt_kwargs,
                         client_kwargs=client_kwargs, debug=debug,
                         performance_report_path=performance_report_path)

        # Ask workers for GPUs
        if is_dask_gpu_supported():
            self.dtype = TaskExecutorType.multi_gpu
        else:
            self.dtype = TaskExecutorType.multi_cpu


class LocalDaskPrefectPipelineExecutor(LocalDaskExecutor):
    """
    A not centric execution engine based on dask (threads only).

    scheduler -- The local dask scheduler to use; common options are
                 "threads", "processes", and "synchronous" (default "threads").
    **kwargs -- Additional keyword arguments to pass to dask config.
    """
    def __init__(self,
                 scheduler="threads",
                 **kwargs):
        super().__init__(scheduler=scheduler, **kwargs)

    @property
    def dtype(self):
        # TODO: Need to define a way to check Dask multi GPU support.
        return TaskExecutorType.multi_cpu


class DaskPBSPipelineExecutor(LocalExecutor):
    def __init__(self, **kwargs):
        self.client = Client(PBSCluster(**kwargs))

        # Ask workers for GPUs
        if is_dask_gpu_supported():
            self.dtype = TaskExecutorType.multi_gpu
        else:
            self.dtype = TaskExecutorType.multi_cpu
