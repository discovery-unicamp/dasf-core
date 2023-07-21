import time
import socket
from typing import Any
from distributed.diagnostics.plugin import WorkerPlugin
from dasf.profile.profiler import EventProfiler

from dask.distributed.system_monitor import SystemMonitor
from dask.distributed.compatibility import PeriodicCallback

from pynvml import *
import nvtx


class WorkerTaskPlugin(WorkerPlugin):
    def __init__(
        self,
        name: str = "TracePlugin",
    ):
        self.name = name

    def setup(self, worker):
        self.worker = worker
        self.hostname = socket.gethostname()
        self.worker_id = f"worker-{self.hostname}-{self.worker.name}"
        self.database = EventProfiler(
            database_file=f"{self.name}-{self.hostname}.msgpack",
        )

    def transition(self, key, start, finish, *args, **kwargs):        
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
    def __init__(self, time = 100, autostart: bool = True, name: str = "ResourceMonitor", **monitor_kwargs):
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
        self.stop()
        
    def update(self):
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
        self.callback.start()
        
    def stop(self):
        self.database.commit()
        self.callback.stop()
        
        
class GPUAnnotationPlugin(WorkerPlugin):
    def __init__(
        self,
        name: str = "GPUAnnotationPlugin",
    ):
        self.name = name
        self.marks = {}
        
    def setup(self, worker):
        pynvml.nvmlInit()
        self.worker = worker
    
    def transition(self, key, start, finish, *args, **kwargs):
        if finish == "executing":
            mark = nvtx.start_range(message=key, color="blue")
            self.marks[key] = mark
        if start == "executing":
            nvtx.end_range(self.marks[key])
            del self.marks[key]