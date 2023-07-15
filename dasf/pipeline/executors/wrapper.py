#!/usr/bin/env python3

try:
    import rmm
    import cupy as cp
except ImportError:
    pass

from dasf.utils.funcs import get_gpu_count
from dasf.pipeline.types import TaskExecutorType


class LocalExecutor:
    def __init__(self,
                 use_gpu=None,
                 gpu_allocator="cupy"):

        if use_gpu is None:
            if self.ngpus > 0:
                self.dtype = TaskExecutorType.single_gpu
            else:
                self.dtype = TaskExecutorType.single_cpu
        elif use_gpu:
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

    def execute(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)
