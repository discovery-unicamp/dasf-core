#!/usr/bin/env python3
""" Local executor module. """

import numpy as np  # noqa: F401

try:
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
except ImportError:  # pragma: no cover
    pass

from dasf.pipeline.types import TaskExecutorType
from dasf.utils.funcs import get_backend_supported, get_gpu_count, is_gpu_supported


class LocalExecutor:
    """This class implements a local executor that can run on a single CPU or
    GPU.

    Parameters
    ----------
    use_gpu : bool, optional
        If true, it will try to use the GPU. If false, it will use the CPU.
        If None, it will try to use the GPU if it is available.
        (default is None)
    backend : str, optional
        The backend to use for the computation. (default is "numpy")
    gpu_allocator : str, optional
        The GPU allocator to use. (default is "cupy")

    """
    def __init__(self,
                 use_gpu=None,
                 backend="numpy",
                 gpu_allocator="cupy"):
        """ Constructor of the object LocalExecutor. """

        self.backend = backend

        if use_gpu is None:
            if self.ngpus > 0:
                self.dtype = TaskExecutorType.single_gpu
            else:
                self.dtype = TaskExecutorType.single_cpu
        elif use_gpu and is_gpu_supported():
            self.dtype = TaskExecutorType.single_gpu
        else:
            self.dtype = TaskExecutorType.single_cpu

        if gpu_allocator == "rmm" and self.dtype == TaskExecutorType.single_gpu:
            rmm.reinitialize(managed_memory=True)
            cp.cuda.set_allocator(rmm_cupy_allocator)

    @property
    def ngpus(self) -> int:
        """Returns the number of GPUs available."""
        return get_gpu_count()

    @property
    def is_connected(self) -> bool:
        """Returns true if the executor is connected to a backend."""
        return True

    def pre_run(self, pipeline):
        """Executes before the pipeline starts.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to be executed.
        """
        pass

    def post_run(self, pipeline):
        """Executes after the pipeline finishes.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline that was executed.
        """
        pass

    def get_backend(self):
        """Returns the backend to use for the computation."""
        if self.dtype == TaskExecutorType.single_gpu:
            return eval("cp")

        return eval("np")

    def execute(self, fn, *args, **kwargs):
        """Executes a function in the executor.

        Parameters
        ----------
        fn : function
            The function to be executed.
        args : list
            The arguments of the function.
        kwargs : dict
            The keyword arguments of the function.
        """
        if get_backend_supported(fn):
            kwargs['backend'] = self.get_backend()

        return fn(*args, **kwargs)
