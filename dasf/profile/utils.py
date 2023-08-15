import time
import atexit
from typing import List
from dasf.profile.profiler import EventDatabase
from dasf.pipeline import Pipeline
from dasf.profile.plugins import WorkerTaskPlugin, ResourceMonitor, GPUAnnotationPlugin

from typing import List

class MultiEventDatabase:
    def __init__(self, databases: List[EventDatabase]):
        self._databases = databases

    def __iter__(self):
        for database in self._databases:
            yield from database.get_traces()
            
            
def register_default_profiler(pipeline: Pipeline, name: str = None, enable_nvtx: bool = False):
    if name is None:
        name = "default"
        
    name += f"-{int(time.time())}"
        
    worker_plugin = WorkerTaskPlugin(name=f"{name}-TracePlugin")
    pipeline.register_plugin(worker_plugin)
    
    resource_plugin = ResourceMonitor(name=f"{name}-ResourceMonitor")   
    
    def close():
        resource_plugin.stop()

    if enable_nvtx:
        ptx_annotator = GPUAnnotationPlugin()
        pipeline.register_plugin(ptx_annotator)
    
    atexit.register(close)