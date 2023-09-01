""" Implementations of important library decorators. """
#!/usr/bin/env python3

from functools import wraps

from dasf.utils.types import is_dask_array
from dasf.utils.types import is_gpu_array
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.funcs import is_dask_supported
from dasf.utils.funcs import is_dask_gpu_supported


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
        new_args, new_kwargs = fetch_from_dask(*args, **kwargs)

        return func(*new_args, **new_kwargs)

    return wrapper


def fetch_args_from_gpu(func):
    """
    Fetches to CPU all function parameters in a GPU data type.
    """
    def wrapper(*args, **kwargs):
        new_args, new_kwargs = fetch_from_gpu(*args, **kwargs)

        return func(*new_args, **new_kwargs)

    return wrapper


def task_handler(func):
    """
    Returns all mapped functions corresponding to the executor in place.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        cls = args[0]
        new_args = args[1:]
        func_name = func.__name__
        func_type = ""
        arch = "cpu"

        if is_forced_local(cls):
            new_args, kwargs = fetch_from_dask(*new_args, **kwargs)

        if not is_forced_local(cls) and (is_dask_gpu_supported() or is_dask_supported()):
            func_type = "_lazy"

        if is_forced_gpu(cls) is not None:
            if is_forced_gpu(cls):
                arch = "gpu"
            else:
                new_args, kwargs = fetch_from_gpu(*new_args, **kwargs)
                arch = "cpu"
        elif is_dask_gpu_supported() or is_gpu_supported():
            arch = "gpu"

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
