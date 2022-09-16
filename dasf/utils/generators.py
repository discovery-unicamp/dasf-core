#!/usr/bin/env python3

from dasf.utils.utils import is_gpu_supported
from dasf.utils.utils import is_dask_supported
from dasf.utils.utils import is_dask_gpu_supported


# TODO: this will not generate generic function....
def generate_generic(cls, func_name: str):
    """Given a method of a class, generate specialized methods depending on
    the supported architecture and functionalities.

    Parameters
    ----------
    cls : type
        The class to inspect.
    func_name : str
        The method of the class to

    Returns
    -------
    Any
        The class with new methods.

    """

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

    wrapper_func_attr = f'{func_type}_{func_name}_{arch}'

    # Do not overwrite existing func_name.
    if hasattr(cls, wrapper_func_attr):
        setattr(cls,
                func_name,
                getattr(cls, wrapper_func_attr))

    return cls


def generate_load(cls):
    """Generate specialized methods for any `load` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "load")


def generate_fit(cls):
    """Generate specialized methods for any `fit` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "fit")


def generate_fit_predict(cls):
    """Generate specialized methods for any `fit_predict` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "fit_predict")


def generate_fit_transform(cls):
    """Generate specialized methods for any `fit_transform` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "fit_transform")


def generate_partial_fit(cls):
    """Generate specialized methods for any `partial_fit` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "partial_fit")


def generate_predict(cls):
    """Generate specialized methods for any `predict` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "predict")


def generate_transform(cls):
    """Generate specialized methods for any `transform` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "transform")


def generate_inverse_transform(cls):
    """Generate specialized methods for any `transform_inverse` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "transform_inverse")


def generate_get_params(cls):
    """Generate specialized methods for any `get_params` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "get_params")


def generate_set_params(cls):
    """Generate specialized methods for any `set_params` method from a given class.

    Parameters
    ----------
    cls : Type
        The class to inspect.

    Returns
    -------
    Any
        The class with new methods.

    """
    return generate_generic(cls, "set_params")
