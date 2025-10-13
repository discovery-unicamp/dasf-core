"""A module for profile utilities."""
import atexit
import time
from typing import List

from dasf.pipeline import Pipeline
from dasf.profile.plugins import GPUAnnotationPlugin, ResourceMonitor, WorkerTaskPlugin
from dasf.profile.profiler import EventDatabase


class MultiEventDatabase:
    """A class to handle multiple event databases."""

    def __init__(self, databases: List[EventDatabase]):
        """Initialize the multi-event database.

        Parameters
        ----------
        databases : List[EventDatabase]
            A list of event databases to handle.
        """
        self._databases = databases

    def __iter__(self):
        """Iterate over all traces in the databases."""
        for database in self._databases:
            yield from database.get_traces()

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return f"MultiEventDatabase with {len(self._databases)} databases"

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return str(self)


def register_default_profiler(pipeline: Pipeline,
                              name: str = None,
                              enable_nvtx: bool = False,
                              add_time_suffix: bool = True):
    """Register the default profiler plugins to a pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to register the plugins to.
    name : str, optional
        The name of the profiler, by default None
    enable_nvtx : bool, optional
        Enable NVTX annotations, by default False
    add_time_suffix : bool, optional
        Add a time suffix to the profiler name, by default True
    """
    if name is None:
        name = "default"

    if add_time_suffix:
        name += f"-{int(time.time())}"

    worker_plugin = WorkerTaskPlugin(name=f"{name}-TracePlugin")
    pipeline.register_plugin(worker_plugin)
    print(f"Registered worker plugin: {name}-TracePlugin")

    resource_plugin = ResourceMonitor(name=f"{name}-ResourceMonitor")
    print(f"Registered resource plugin: {name}-ResourceMonitor")

    def close():
        """Stop the resource monitor."""
        resource_plugin.stop()

    if enable_nvtx:
        ptx_annotator = GPUAnnotationPlugin()
        pipeline.register_plugin(ptx_annotator)
        print("Registered GPU annotation plugin (NVTX)")

    atexit.register(close)
