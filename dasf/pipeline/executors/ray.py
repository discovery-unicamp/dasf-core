#!/usr/bin/env python3

""" Ray executor module. """

try:
    import ray
    from ray.util.dask import disable_dask_on_ray, enable_dask_on_ray

    USE_RAY = True
except ImportError: # pragma: no cover
    USE_RAY = False

from dasf.pipeline.executors.base import Executor
from dasf.utils.funcs import get_dask_gpu_count


class RayPipelineExecutor(Executor):
    """A pipeline executor based on ray data flow.

    Parameters
    ----------
    address : str
        Address of the Dask scheduler, default=None.
    port : int
        Port of the Ray head, default=8786.
    local : bool
        Kicks off a new local Ray cluster, default=False.
    use_gpu : bool
        In conjunction with `local`, it kicks off a local CUDA Ray
        cluster, default=False.
    """

    def __init__(
        self,
        address=None,
        port=6379,
        local=False,
        use_gpu=False,
        ray_kwargs=None,
    ):
        """ Constructor of the object RayPipelineExecutor. """
        if not USE_RAY:
            raise Exception("Ray executor is not support. "
                            "Check if you have it installed first.")

        self.address = address
        self.port = port

        if not ray_kwargs:
            ray_kwargs = dict()

        enable_dask_on_ray()

        if address:
            address_str = f"ray://{address}:{str(port)}"

            ray.init(address=address_str, **ray_kwargs)
        elif local:
            ray.init(**ray_kwargs)

    @property
    def ngpus(self):
        """Return the number of GPUs in total.

        Returns
        -------
        ngpus : Number of GPUs in total
        """
        return len(get_dask_gpu_count())

    @property
    def is_connected(self):
        """Return wether the executor is connected or not.

        Returns
        -------
        bool : if the executor is connected.
        """
        return ray.is_initialized()

    def execute(self, fn, *args, **kwargs):
        """Return wether the executor is connected or not.

        Parameters
        ----------
        fn : Callable
           Function to call when executor is performing a task.

        Returns
        -------
        ret : the same return of function `fn`.
        """
        return fn(*args, **kwargs)

    def __del__(self):
        """Destructor of object.

        It also shutdowns Dask on Ray properly with `disable_dask_on_ray`.
        """
        disable_dask_on_ray()

        ray.shutdown()
