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
            
    def __str__(self) -> str:
        return f"MultiEventDatabase with {len(self._databases)} databases"
    
    def __repr__(self) -> str:
        return str(self)
            
            
def register_default_profiler(pipeline: Pipeline, name: str = None, enable_nvtx: bool = False, add_time_suffix: bool = True):
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
        resource_plugin.stop()

    if enable_nvtx:
        ptx_annotator = GPUAnnotationPlugin()
        pipeline.register_plugin(ptx_annotator)
        print(f"Registered GPU annotation plugin (NVTX)")
    
    atexit.register(close)