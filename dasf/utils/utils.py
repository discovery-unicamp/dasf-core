#!/usr/bin/env python3

import os
import dask
import wget
import psutil
import pandas
import GPUtil
import numpy as np

from pathlib import Path

import dask.delayed as dd
from dask.distributed import Client

from distributed.client import wait, FIRST_COMPLETED
from distributed.utils import TimeoutError as DistributedTimeoutError

from dasf.pipeline.types import TaskExecutorType

GPU_SUPPORTED = True
try:
    import cupy as cp # noqa
except ImportError:
    GPU_SUPPORTED = False


def human_readable_size(size, decimal=3):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal}f} {unit}"


def get_full_qualname(obj):
    klass = obj.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__


def get_worker_info(client):
    """
    returns a list of workers (sorted), and the DNS name for the master host
    The master is the 0th worker's host
    """
    workers = client.scheduler_info()["workers"]
    worker_keys = sorted(workers.keys())
    workers_by_host = {}
    for key in worker_keys:
        worker = workers[key]
        host = worker["host"]
        workers_by_host.setdefault(host, []).append(key)
    host = workers[worker_keys[0]]["host"]
    all_workers = []
    global_rank = 0
    world_size = len(workers_by_host)
    hosts = sorted(workers_by_host.keys())
    for host in hosts:
        local_rank = 0
        for worker in workers_by_host[host]:
            all_workers.append(
                dict(
                    master=hosts[0],
                    worker=worker,
                    nthreads=workers[worker]["nthreads"],
                    local_rank=0,
                    global_rank=global_rank,
                    host=host,
                    world_size=world_size,
                )
            )
            local_rank += 1
            global_rank += 1
    return all_workers


def sync_future_loop(futures):
    while True:
        if not futures:
            break

        try:
            result = wait(futures, 0.1, FIRST_COMPLETED)
        except DistributedTimeoutError:
            continue

        for fut in result.done:
            try:
                fut.result(timeout=7200)
            except Exception as e:  # pylint: disable=broad-except
                print(str(e))
                raise
        futures = result.not_done


def download_file(url, filename=None, directory=None):
    if directory is not None:
        os.makedirs(os.path.dirname(directory), exist_ok=True)

    if filename and directory:
        output = os.path.abspath(os.path.join(directory, filename))

        if not os.path.exists(output):
            wget.download(url, out=output)
    elif filename:
        output = os.path.abspath(os.path.join(os.getcwd(), filename))

        if not os.path.exists(output):
            wget.download(url, out=output)
    elif directory:
        output = os.path.abspath(os.path.join(directory, wget.download(url)))
    else:
        output = os.path.abspath(os.path.join(os.getcwd(), wget.download(url)))

    return output


def download_file_from_gdrive(file_id, filename=None, directory=None):
    URL = ("https://drive.google.com/uc?export=download&confirm=9iBg&id=%s"
           % file_id)

    return download_file(URL, filename=filename, directory=directory)


def get_machine_memory_avail():
    return psutil.virtual_memory().free


def set_executor_default():
    return TaskExecutorType.single_cpu


def set_executor_gpu():
    return TaskExecutorType.single_gpu


def is_executor_single(dtype):
    return (dtype == TaskExecutorType.single_cpu or
            dtype == TaskExecutorType.single_gpu)


def is_executor_cluster(dtype):
    return (dtype == TaskExecutorType.multi_cpu or
            dtype == TaskExecutorType.multi_gpu)


def is_executor_cpu(dtype):
    return (dtype == TaskExecutorType.single_cpu or
            dtype == TaskExecutorType.multi_cpu)


def is_executor_gpu(dtype):
    return (dtype == TaskExecutorType.single_gpu or
            dtype == TaskExecutorType.multi_gpu)


def is_gpu_supported():
    return GPU_SUPPORTED


def is_dask_local_supported():
    try:
        scheduler = dask.config.get(key="scheduler")
        return scheduler is not None
    except Exception:
        return False


def get_dask_running_client():
    return Client.current()


def is_dask_supported():
    try:
        if is_dask_local_supported():
            return True
        else:
            cur = get_dask_running_client()
            return cur is not None
    except Exception:
        return False


def is_dask_gpu_supported():
    if is_dask_supported():
        if len(get_dask_gpu_count()) > 0:
            return True

    return False


def get_gpu_count():
    return GPUtil.getGPUs()


def get_dask_gpu_count(fetch=True):
    y = dd(GPUtil.getGPUs)()
    if fetch:
        return y.compute()
    return y


def block_chunk_reduce(dask_data, output_chunk):
    drop_axis = np.array([])
    new_axis = None

    if output_chunk is None or \
       not isinstance(output_chunk, tuple):
        return drop_axis.tolist(), new_axis

    data_chunk_range = len(dask_data.chunksize)
    output_chunk_range = len(output_chunk)

    data_indexes = np.arange(data_chunk_range)
    output_indexes = np.arange(output_chunk_range)

    if data_chunk_range > output_chunk_range:
        inter = np.intersect1d(data_indexes, output_indexes)

        drop_axis = np.delete(data_indexes, inter)
    elif data_chunk_range < output_chunk_range:
        inter = np.intersect1d(data_indexes, output_indexes)

        new_axis = np.delete(output_chunk_range, inter).tolist()

    return drop_axis.tolist(), new_axis


def return_local_and_gpu(executor, local, gpu):
    if local is not None and gpu is None:
        if local is True:
            return TaskExecutorType(executor.dtype.value & 2)
        elif local is False:
            return TaskExecutorType(executor.dtype.value | 1)
    elif local is None and gpu is not None:
        if gpu is True:
            return TaskExecutorType((executor.dtype >> 1) + 2)
        elif gpu is False:
            return TaskExecutorType(executor.dtype & 1)
    elif local is not None and gpu is not None:
        if local is True and gpu is False:
            return TaskExecutorType.single_cpu
        elif local is False and gpu is False:
            return TaskExecutorType.multi_cpu
        elif local is True and gpu is True:
            return TaskExecutorType.single_gpu
        elif local is False and gpu is True:
            return TaskExecutorType.multi_gpu

    return executor.dtype


def get_pipeline_result_by_name(result, name):
    results = result.result

    for key in results:
        if key.name == name:
            return result.result[key]._result.value
    return None


def get_dask_mem_usage(profiler):
    profiler_dir = os.path.abspath(str(Path.home()) + "/.cache/dasf/profiler/")

    if profiler == "memusage":
        os.makedirs(profiler_dir, exist_ok=True)

        mem = pandas.read_csv(os.path.abspath(profiler_dir + "/dask-memusage"))

        column = mem["max_memory_mb"]
        max_index = column.idxmax()

        return mem["max_memory_mb"][max_index]
    else:
        return 0.0
