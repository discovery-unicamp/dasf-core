""" Generic and regular functions. """
#!/usr/bin/env python3

import os
import time
import threading

from pathlib import Path

import wget
import pandas
import psutil
import GPUtil
import numpy as np

import dask
import dask.delayed as dd
from dask.distributed import Client

from distributed.client import wait, FIRST_COMPLETED
from distributed.utils import TimeoutError as DistributedTimeoutError

from dasf.pipeline.types import TaskExecutorType

from IPython import display as disp
from ipywidgets import HBox, FloatProgress, Label

try:
    import cupy as cp
    GPU_SUPPORTED = isinstance(cp.__version__, str)
except ImportError:
    GPU_SUPPORTED = False


def human_readable_size(size, decimal=3) -> str:
    """
    converts data size into the proper measurement
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal}f} {unit}"


def get_full_qualname(obj) -> str:
    """
    Return fully qualified name of objects.
    """
    klass = obj.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__
    return module + "." + klass.__qualname__


def get_worker_info(client) -> list:
    """
    Returns a list of workers (sorted), and the DNS name for the master host
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
    """
    Synchronize all futures submitted to workers.
    """
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
            except Exception as exc:  # pylint: disable=broad-except
                print(str(exc))
                raise
        futures = result.not_done


class NotebookProgressBar(threading.Thread):
    MIN_CUR = -2
    MIN_TOTAL = -1

    def __init__(self):
        threading.Thread.__init__(self)

        # pylint: disable=disallowed-name
        self.bar = None
        self.percentage = None
        self.data = None

        self.__lock = threading.Lock()
        self.__current = self.MIN_CUR
        self.__total = self.MIN_TOTAL
        self.__error = False

    def show(self):
        self.bar = FloatProgress(value=0, min=0, max=100)
        self.percentage = Label(value='0 %')
        self.data = Label(value='')
        box = HBox((self.percentage, self.bar, self.data))
        disp.display(box)

    def set_current(self, current, total):
        with self.__lock:
            self.__current = current
            self.__total = total

    def set_error(self, error):
        self.__error = error

    def run(self):
        while (not self.__error and self.__current < self.__total):
            time.sleep(1)

            if self.__current != self.MIN_CUR and self.__total != self.MIN_TOTAL:
                progress = (self.__current / self.__total) * 100
                self.bar.value = progress
                self.percentage.value = f"{int(self.bar.value)} %%"
                self.data.value = f"{int(self.__current)} / {int(self.__total)}"

        if not self.__error:
            self.bar.style.bar_color = '#03c04a'
        else:
            self.bar.style.bar_color = '#ff0000'


def download_file(url, filename=None, directory=None):
    """
    Download a generic file and save it.
    """
    if directory is not None:
        os.makedirs(os.path.dirname(directory), exist_ok=True)

    progressbar = None

    if is_notebook():
        progressbar = NotebookProgressBar()

        def update_notebook_bar(current, total):
            progressbar.set_current(current, total)

    try:
        if filename and directory:
            output = os.path.abspath(os.path.join(directory, filename))

            if not os.path.exists(output):
                if is_notebook():
                    # Activate the notebook progress bar
                    progressbar.show()
                    progressbar.start()

                    wget.download(url, out=output, bar=update_notebook_bar)
                else:
                    wget.download(url, out=output)
        elif filename:
            output = os.path.abspath(os.path.join(os.getcwd(), filename))

            if not os.path.exists(output):
                if is_notebook():
                    # Activate the notebook progress bar
                    progressbar.show()
                    progressbar.start()

                    wget.download(url, out=output, bar=update_notebook_bar)
                else:
                    wget.download(url, out=output)
        elif directory:
            if is_notebook():
                # Activate the notebook progress bar
                progressbar.show()
                progressbar.start()

                output = \
                    os.path.abspath(os.path.join(directory,
                                                 wget.download(url,
                                                               bar=update_notebook_bar)))
            else:
                output = os.path.abspath(os.path.join(directory, wget.download(url)))
        else:
            if is_notebook():
                # Activate the notebook progress bar
                progressbar.show()
                progressbar.start()

                output = \
                    os.path.abspath(os.path.join(os.getcwd(),
                                                 wget.download(url,
                                                               bar=update_notebook_bar)))
            else:
                output = os.path.abspath(os.path.join(os.getcwd(), wget.download(url)))
    except Exception as exc:
        if progressbar:
            progressbar.set_error(True)

    return output


def download_file_from_gdrive(file_id, filename=None, directory=None):
    """
    Download a file from Google Drive using gdrive file id.
    """
    url = f"https://drive.google.com/uc?export=download&confirm=9iBg&id={file_id}"

    return download_file(url, filename=filename, directory=directory)


def get_machine_memory_avail():
    """
    Return free memory available from a single machine.
    """
    return psutil.virtual_memory().free


def set_executor_default():
    """
    Return executor as a CPU (default) instance.
    """
    return TaskExecutorType.single_cpu


def set_executor_gpu():
    """
    Return executor as a GPU instance.
    """
    return TaskExecutorType.single_gpu


def is_executor_single(dtype) -> bool:
    """
    Return if the executor is a single machine instance.
    """
    return dtype in (TaskExecutorType.single_cpu, TaskExecutorType.single_gpu)


def is_executor_cluster(dtype) -> bool:
    """
    Return if the executor is a cluster instance.
    """
    return dtype in (TaskExecutorType.multi_cpu, TaskExecutorType.multi_gpu)


def is_executor_cpu(dtype) -> bool:
    """
    Return if the executor is a CPU instance.
    """
    return dtype in (TaskExecutorType.single_cpu, TaskExecutorType.multi_cpu)


def is_executor_gpu(dtype) -> bool:
    """
    Return if the executor is a GPU instance.
    """
    return dtype in (TaskExecutorType.single_gpu, TaskExecutorType.multi_gpu)


def is_gpu_supported() -> bool:
    """
    Return if GPU is supported.
    """
    return GPU_SUPPORTED


def is_dask_local_supported() -> bool:
    """
    Return if Dask is supported locally by the executor.
    """
    try:
        scheduler = dask.config.get(key="scheduler")
        return scheduler is not None
    except Exception:
        return False


def get_dask_running_client():
    """
    Get Dask runner stanza.
    """
    return Client.current()


def is_dask_supported() -> bool:
    """
    Return if Dask is supported by the executor.
    """
    try:
        if is_dask_local_supported():
            return True

        cur = get_dask_running_client()
        if hasattr(cur, 'dtype'):
            return is_executor_cluster(cur.dtype)
        return cur is not None
    except Exception:
        return False


def is_dask_gpu_supported() -> bool:
    """
    Return if any node supports GPU.
    """
    if is_dask_supported():
        if len(get_dask_gpu_count()) > 0:
            return True

    return False


def get_gpu_count() -> int:
    """
    Get single node GPU count.
    """
    return GPUtil.getGPUs()


def get_dask_gpu_count(fetch=True) -> int:
    """
    Get how many GPUs are available in each worker.
    """
    # pylint: disable=not-callable
    ret = dd(GPUtil.getGPUs)()
    if fetch:
        return ret.compute()
    return ret


def block_chunk_reduce(dask_data, output_chunk):
    """
    Reduce the chunk according the new output size.
    """
    drop_axis = np.array([])
    new_axis = None

    if output_chunk is None or not isinstance(output_chunk, tuple):
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
    """
    Return executor type based on passed preferences.
    """
    # pylint: disable=too-many-return-statements
    if local is not None and gpu is None:
        if local is True:
            return TaskExecutorType(executor.dtype.value & 2)
        if local is False:
            return TaskExecutorType(executor.dtype.value | 1)
    elif local is None and gpu is not None:
        if gpu is True:
            return TaskExecutorType((executor.dtype >> 1) + 2)
        if gpu is False:
            return TaskExecutorType(executor.dtype & 1)
    elif local is not None and gpu is not None:
        if local is True and gpu is False:
            return TaskExecutorType.single_cpu
        if local is False and gpu is False:
            return TaskExecutorType.multi_cpu
        if local is True and gpu is True:
            return TaskExecutorType.single_gpu
        if local is False and gpu is True:
            return TaskExecutorType.multi_gpu

    return executor.dtype


def get_dask_mem_usage(profiler):
    """
    Get Dask memory usage profile.
    """
    profiler_dir = os.path.abspath(str(Path.home()) + "/.cache/dasf/profiler/")

    if profiler == "memusage":
        os.makedirs(profiler_dir, exist_ok=True)

        mem = pandas.read_csv(os.path.abspath(profiler_dir + "/dask-memusage"))

        column = mem["max_memory_mb"]
        max_index = column.idxmax()

        return mem["max_memory_mb"][max_index]
    return 0.0


def is_notebook() -> bool:
    """
    Return if the code is being executed in a IPyNotebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
    except NameError:
        pass

    return False
