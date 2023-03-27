#!/usr/bin/env python3

try:
    import ray
    
    from ray.util.dask import ray_dask_get
    from ray.util.dask import enable_dask_on_ray
    from ray.util.dask import disable_dask_on_ray

    USE_RAY=True
except ImportError:
    USE_RAY=False

from dasf.pipeline.executors.base import Executor
from dasf.utils.funcs import get_dask_gpu_count

class RayPipelineExecutor(Executor):
    """
    A pipeline engine based on ray data flow.

    Keyword arguments:
    address -- address of the Dask scheduler (default None).
    port -- port of the Ray head (default 8786).
    local -- kicks off a new local Ray cluster (default False).
    use_gpu -- in conjunction with `local`, it kicks off a local CUDA Ray
                cluster (default False).
    """

    def __init__(
        self,
        address=None,
        port=6379,
        local=False,
        use_gpu=False,
        ray_kwargs=None,
    ):
    
        self.address = address
        self.port = port

        if not ray_kwargs:
            ray_kwargs = dict()

        enable_dask_on_ray()

        if address:
            address_str = f"ray://{address}:{str(port)}"

            ray.init(address=address, **ray_kwargs)
        elif local:
            ray.init(**ray_kwargs)

    @property
    def ngpus(self):
        return len(get_dask_gpu_count())

    @property
    def is_connected(self):
        return ray.is_initialized()            

    def execute(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def __del__(self):
        disable_dask_on_ray()

        ray.shutdown()
