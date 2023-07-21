import time
import atexit
from typing import List
from dasf.profile.profiler import EventDatabase
from dasf.pipeline import Pipeline
from dasf.profile.plugins import WorkerTaskPlugin, ResourceMonitor

from typing import List

class MultiEventDatabase:
    def __init__(self, databases: List[EventDatabase]):
        self._databases = databases

    def __iter__(self):
        for database in self._databases:
            yield from database.get_traces()
            
            
def register_default_profiler(pipeline: Pipeline, name: str = None):
    if name is None:
        name = "default"
        
    name += f"-{int(time.time())}"
        
    worker_plugin = WorkerTaskPlugin(name=f"{name}-TracePlugin")
    resource_plugin = ResourceMonitor(name=f"{name}-ResourceMonitor")
    
    def close():
        resource_plugin.stop()

    pipeline.register_plugin(worker_plugin)
    atexit.register(close)