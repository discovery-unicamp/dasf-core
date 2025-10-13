"""A module for profiling plugins."""
import os
import socket
import time

import nvtx
import pynvml
from dask.distributed.compatibility import PeriodicCallback
from dask.distributed.system_monitor import SystemMonitor
from distributed.diagnostics.plugin import WorkerPlugin

from dasf.profile.profiler import EventProfiler


class WorkerTaskPlugin(WorkerPlugin):
    """
    A Dask worker plugin to trace task execution.

    Parameters
    ----------
    name : str
        The name of the plugin.
    """
    def __init__(
        self,
        name: str = "TracePlugin",
    ):
        """
        Initialize the plugin.

        Parameters
        ----------
        name : str, optional
            The name of the plugin, by default "TracePlugin"
        """
        self.name = name

    def setup(self, worker):
        """
        Set up the plugin.

        Parameters
        ----------
        worker : dask.distributed.Worker
            The Dask worker.
        """
        self.worker = worker
        self.hostname = socket.gethostname()
        self.worker_id = f"worker-{self.hostname}-{self.worker.name}"
        self.database = EventProfiler(
            database_file=f"{self.name}-{self.hostname}.msgpack",
        )

    def transition(self, key, start, finish, *args, **kwargs):
        """
        Trace the task transition.

        Parameters
        ----------
        key : str
            The key of the task.
        start : str
            The start state of the task.
        finish : str
            The finish state of the task.
        """
        now = time.monotonic()

        if finish == "memory":
            # Get the last compute event
            startstops = next(
                (
                    x
                    for x in reversed(self.worker.state.tasks[key].startstops)
                    if x["action"] == "compute"
                ),
                None,
            )
            if startstops is not None:
                # Add information about the task execution
                shape = tuple()
                dtype = "unknown"
                if hasattr(self.worker.data[key], "shape"):
                    if isinstance(getattr(self.worker.data[key], "shape"), tuple):
                        shape = getattr(self.worker.data[key], "shape")

                if hasattr(self.worker.data[key], "dtype"):
                    dtype = str(getattr(self.worker.data[key], "dtype"))

                task = self.worker.state.tasks[key]
                nbytes = task.nbytes or 0

                self.database.record_complete_event(
                    name="Compute",
                    timestamp=now,
                    # TODO check startstop returning None
                    duration=startstops["stop"] - startstops["start"],
                    process_id=self.hostname,
                    thread_id=self.worker_id,
                    args={
                        "key": key,
                        "name": "-".join(key.split(",")[0][2:-1].split("-")[:-1]),
                        "state": finish,
                        "size": nbytes,
                        "shape": shape,
                        "dtype": dtype,
                        "type": str(type(self.worker.data[key])),
                        "dependencies": [dep.key for dep in task.dependencies],
                        "dependents": [dep.key for dep in task.dependents],
                    },
                )

        if finish == "memory" or finish == "erred":
            # Additionally add the total in-memory tasks
            self.database.record_instant_event(
                name="Managed Memory",
                timestamp=now,
                process_id=self.hostname,
                thread_id=self.worker_id,
                args={
                    "key": key,
                    "state": finish,
                    "size": self.worker.state.nbytes,
                    "tasks": len(self.worker.data),
                }
            )


class ResourceMonitor:
    """
    A resource monitor for Dask workers.

    Parameters
    ----------
    time : int
        The time interval to update the monitor in ms.
    autostart : bool
        Whether to start the monitor automatically.
    name : str
        The name of the monitor.
    """
    def __init__(self,
                 time: int = 100,
                 autostart: bool = True,
                 name: str = "ResourceMonitor",
                 **monitor_kwargs
                 ):
        """
        Initialize the monitor.

        Parameters
        ----------
        time : int, optional
            The time in ms to wait between updates, by default 100
        autostart : bool, optional
            Start the monitor automatically, by default True
        name : str, optional
            The name of the monitor, by default "ResourceMonitor"
        """
        self.time = time
        self.name = name
        self.hostname = socket.gethostname()
        self.database = EventProfiler(
            database_file=f"{self.name}-{self.hostname}.msgpack",
        )
        self.monitor = SystemMonitor(**monitor_kwargs)
        self.callback = PeriodicCallback(self.update, callback_time=self.time)
        if autostart:
            self.start()

    def __del__(self):
        """Delete the monitor."""
        self.stop()

    def update(self):
        """Update the monitor."""
        res = self.monitor.update()
        self.database.record_instant_event(
            name="Resource Usage",
            timestamp=time.monotonic(),
            process_id=self.hostname,
            thread_id=None,
            args=res
        )
        return res

    def start(self):
        """Start the monitor."""
        self.callback.start()

    def stop(self):
        """Stop the monitor."""
        self.database.commit()
        self.callback.stop()


class GPUAnnotationPlugin(WorkerPlugin):
    """
    A Dask worker plugin to annotate GPU tasks.

    Parameters
    ----------
    name : str
        The name of the plugin.
    """
    def __init__(
        self,
        name: str = "GPUAnnotationPlugin",
    ):
        """
        Initialize the plugin.

        Parameters
        ----------
        name : str, optional
            The name of the plugin, by default "GPUAnnotationPlugin"
        """
        self.name = name
        self.gpu_num = None
        self.marks = {}

    def setup(self, worker):
        """
        Set up the plugin.

        Parameters
        ----------
        worker : dask.distributed.Worker
            The Dask worker.
        """
        self.worker = worker
        self.gpu_num = int(os.environ['CUDA_VISIBLE_DEVICES'].split(",")[0])
        print("Setting up GPU annotation plugin for worker "
              f"{self.worker.name}. GPU: {self.gpu_num}")

    def transition(self, key, start, finish, *args, **kwargs):
        """
        Trace the task transition.

        Parameters
        ----------
        key : str
            The key of the task.
        start : str
            The start state of the task.
        finish : str
            The finish state of the task.
        """
        if finish == "executing":
            _ = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_num)
            mark = nvtx.start_range(message=key, domain="compute")
            self.marks[key] = mark
        if start == "executing":
            _ = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_num)
            nvtx.end_range(self.marks[key])
            del self.marks[key]
