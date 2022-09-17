#!/usr/bin/env python3

from dasf.utils.utils import is_gpu_supported
from dasf.utils.utils import is_dask_supported
from dasf.utils.utils import is_dask_gpu_supported


def generate_generic(cls, func_name):
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

    # Do not overwrite existing func_name.
    if hasattr(cls, wrapper_func_attr):
        setattr(cls, func_name, getattr(cls, wrapper_func_attr))

    return cls


def generate_load(cls):
    return generate_generic(cls, "load")


def generate_fit(cls):
    return generate_generic(cls, "fit")


def generate_fit_predict(cls):
    return generate_generic(cls, "fit_predict")


def generate_fit_transform(cls):
    return generate_generic(cls, "fit_transform")


def generate_partial_fit(cls):
    return generate_generic(cls, "partial_fit")


def generate_predict(cls):
    return generate_generic(cls, "predict")


def generate_transform(cls):
    return generate_generic(cls, "transform")


def generate_inverse_transform(cls):
    return generate_generic(cls, "transform_inverse")


def generate_get_params(cls):
    return generate_generic(cls, "get_params")


def generate_set_params(cls):
    return generate_generic(cls, "set_params")
