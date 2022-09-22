#!/usr/bin/env python3

from dasf.utils.utils import is_gpu_supported
from dasf.utils.utils import is_dask_supported
from dasf.utils.utils import is_dask_gpu_supported


def task_handler(func):
    def wrapper(*args):
        cls = args[0]
        new_args = args[1:]
        func_name = func.__name__
        func_type = ""
        arch = "cpu"

        if is_dask_gpu_supported():
            arch = "gpu"
            func_type = "_lazy"
        elif is_dask_supported():
            arch = "cpu"
            func_type = "_lazy"
        elif is_gpu_supported():
            arch = "gpu"

        wrapper_func_attr = f"{func_type}_{func_name}_{arch}"

        if hasattr(cls, wrapper_func_attr):
            return getattr(cls, wrapper_func_attr)(*new_args)
        else:
            return func(*new_args)

    return wrapper
