#!/usr/bin/env python3

import numpy as np

try:
    import cupy as cp
    import rmm
except ImportError: # pragma: no cover
    pass

try:
    from jax import jit
except ImportError: # pragma: no cover
    pass

from dasf.pipeline.types import TaskExecutorType
from dasf.utils.funcs import (
    get_backend_supported,
    get_gpu_count,
    is_gpu_supported,
    is_jax_supported,
)


class LocalExecutor:
    def __init__(self,
                 use_gpu=None,
                 backend="numpy",
                 gpu_allocator="cupy"):

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
            cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

    @property
    def ngpus(self) -> int:
        return get_gpu_count()

    @property
    def is_connected(self) -> bool:
        return True

    def pre_run(self, pipeline):
        pass

    def post_run(self, pipeline):
        pass

    def get_backend(self):
        if self.backend == "numpy" and \
           self.dtype == TaskExecutorType.single_gpu:
            return eval("cupy")

        return eval("cupy")

    def execute(self, fn, *args, **kwargs):
        if get_backend_supported(fn):
            kwargs['backend'] = self.get_backend()

        return fn(*args, **kwargs)
