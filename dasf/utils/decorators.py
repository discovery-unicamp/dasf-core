#!/usr/bin/env python3

from functools import wraps

from dasf.utils.types import is_dask_array
from dasf.utils.types import is_gpu_array
from dasf.utils.utils import is_gpu_supported
from dasf.utils.utils import is_dask_supported
from dasf.utils.utils import is_dask_gpu_supported


def task_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
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

        if (not hasattr(cls, wrapper_func_attr) and
            hasattr(cls, func_name)):
            return func(*new_args, **kwargs)
        elif (not hasattr(cls, wrapper_func_attr) and
              not hasattr(cls, func_name)):
            raise NotImplementedError(
                f"There is no implementation of {wrapper_func_attr} nor "
                 "{func_name}"
            )
        else:
            return getattr(cls, wrapper_func_attr)(*new_args, **kwargs)

    return wrapper


def fetch_args_from_dask(func):
    def wrapper(*args, **kwargs):
        new_kwargs = dict()
        new_args = []

        for k, v in kwargs.items():
            if is_dask_array(v):
                new_kwargs[k] = v.compute()
            else:
                new_kwargs[k] = v

        for v in args:
            if is_dask_array(v):
                new_args.append(v.compute())
            else:
                new_args.append(v)

        return func(*new_args, **new_kwargs)

    return wrapper


def fetch_args_from_gpu(func):
    def wrapper(*args, **kwargs):
        new_kwargs = dict()
        new_args = []

        for k, v in kwargs.items():
            if is_gpu_array(v):
                new_kwargs[k] = v.get()
            else:
                new_kwargs[k] = v

        for v in args:
            if is_gpu_array(v):
                new_args.append(v.get())
            else:
                new_args.append(v)

        return func(*new_args, **new_kwargs)

    return wrapper
