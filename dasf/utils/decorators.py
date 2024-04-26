""" Implementations of important library decorators. """
#!/usr/bin/env python3

from functools import wraps

from dasf.utils.funcs import (
    get_dask_running_client,
    is_dask_gpu_supported,
    is_dask_supported,
    is_gpu_supported,
)
from dasf.utils.types import is_dask_array, is_gpu_array


def is_forced_local(cls):
    """
    Returns if object is forced to run in a CPU.
    """
    # pylint: disable=protected-access
    if hasattr(cls, "_run_local") and cls._run_local is not None:
        # pylint: disable=protected-access
        return cls._run_local
    return None


def is_forced_gpu(cls):
    """
    Returns if object is forced to run in a GPU.
    """
    # pylint: disable=protected-access
    if hasattr(cls, "_run_gpu") and cls._run_gpu is not None:
        # pylint: disable=protected-access
        return cls._run_gpu
    return None


def fetch_from_dask(*args, **kwargs) -> tuple:
    """
    Fetches to CPU all parameters in a Dask data type.
    """
    new_kwargs = {}
    new_args = []

    for key, value in kwargs.items():
        if is_dask_array(value):
            new_kwargs[key] = value.compute()
        else:
            new_kwargs[key] = value

    for value in args:
        if is_dask_array(value):
            new_args.append(value.compute())
        else:
            new_args.append(value)

    return new_args, new_kwargs


def fetch_from_gpu(*args, **kwargs) -> tuple:
    """
    Fetches to CPU all parameters in a GPU data type.
    """
    new_kwargs = {}
    new_args = []

    for key, value in kwargs.items():
        if is_gpu_array(value):
            new_kwargs[key] = value.get()
        else:
            new_kwargs[key] = value

    for value in args:
        if is_gpu_array(value):
            new_args.append(value.get())
        else:
            new_args.append(value)

    return new_args, new_kwargs


def fetch_args_from_dask(func):
    """
    Fetches to CPU all function parameters in a Dask data type.
    """
    def wrapper(*args, **kwargs):
        """
        Wrapper to fetch parameters from Dask data.
        """
        new_args, new_kwargs = fetch_from_dask(*args, **kwargs)

        return func(*new_args, **new_kwargs)

    return wrapper


def fetch_args_from_gpu(func):
    """
    Fetches to CPU all function parameters in a GPU data type.
    """
    def wrapper(*args, **kwargs):
        """
        Wrapper to fetch parameters from GPU.
        """
        new_args, new_kwargs = fetch_from_gpu(*args, **kwargs)

        return func(*new_args, **new_kwargs)

    return wrapper


def task_handler(func):
    """
    Returns all mapped functions corresponding to the executor in place.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper of the function to map the proper object function.
        """
        cls = args[0]
        new_args = args[1:]
        func_name = func.__name__
        func_type = ""
        arch = "cpu"
        client = get_dask_running_client()
        if client is not None: # Runs task according to current client configuration, i.e, Pipeline Executor
            func_type = "_lazy"
            arch = "gpu" if getattr(client, "backend", None) == "cupy" else "cpu"
        else:
            if not is_forced_local(cls) and (is_dask_gpu_supported() or is_dask_supported()):
                func_type = "_lazy"
            if is_dask_gpu_supported() or is_gpu_supported():
                arch = "gpu"

        if is_forced_local(cls):
            func_type = ""
            new_args, kwargs = fetch_from_dask(*new_args, **kwargs)
        
        if is_forced_gpu(cls):
                arch = "gpu"

        if arch == "cpu":
            new_args, kwargs = fetch_from_gpu(*new_args, **kwargs)

        wrapper_func_attr = f"{func_type}_{func_name}_{arch}"

        if (not hasattr(cls, wrapper_func_attr) and
           hasattr(cls, func_name)):
            return func(*new_args, **kwargs)
        if (not hasattr(cls, wrapper_func_attr) and
              not hasattr(cls, func_name)):
            raise NotImplementedError(
                f"There is no implementation of {wrapper_func_attr} nor "
                f"{func_name}"
            )
        return getattr(cls, wrapper_func_attr)(*new_args, **kwargs)

    return wrapper
