""" Data types handlers. """
#!/usr/bin/env python3

from typing import Union, get_args

import dask.array as da
import dask.dataframe as ddf
import numpy as np
import pandas as pd
import xarray as xr
import zarr

try:
    import cudf
    import cupy as cp
    import dask_cudf as dcudf
except ImportError: # pragma: no cover
    pass

from dasf.utils.funcs import is_gpu_supported

ArrayCPU = Union[list, np.ndarray, zarr.core.Array]
DataFrameCPU = Union[pd.DataFrame]

DataCPU = Union[ArrayCPU, DataFrameCPU]

DaskArray = Union[da.core.Array]
DaskDataFrameCPU = Union[ddf.core.DataFrame]

XDataArray = Union[xr.DataArray]

Array = Union[ArrayCPU, DaskArray, XDataArray]
DaskDataFrame = Union[DaskDataFrameCPU]
DataFrame = Union[DataFrameCPU, DaskDataFrameCPU]
DataDask = Union[DaskArray, DaskDataFrameCPU]
try:
    ArrayGPU = Union[cp.ndarray]
    DataFrameGPU = Union[cudf.DataFrame]

    DataGPU = Union[ArrayGPU, DataFrameGPU]

    DaskDataFrameGPU = Union[dcudf.core.DataFrame]

    Array = Union[Array, ArrayGPU]
    DaskDataFrame = Union[DaskDataFrame, DaskDataFrameGPU]
    DataFrame = Union[DataFrame, DaskDataFrame, DataFrameGPU]
    DataDask = Union[DataDask, DaskDataFrame]
except NameError: # pragma: no cover
    pass


def is_array(data) -> bool:
    """
    Returns if data is a generic array.
    """
    return isinstance(data, get_args(Array))


def is_dataframe(data) -> bool:
    """
    Returns if data is a generic dataframe.
    """
    return isinstance(data, get_args(DataFrame))


def is_cpu_array(data) -> bool:
    """
    Returns if data is a CPU arrau like Numpy.
    """
    return isinstance(data, get_args(ArrayCPU))


def is_cpu_dataframe(data) -> bool:
    """
    Returns if data is a CPU dataframe like Pandas.
    """
    return isinstance(data, DataFrameCPU)


def is_cpu_datatype(data) -> bool:
    """
    Returns if data is a CPU data type.
    """
    return isinstance(data, get_args(DataCPU))


def is_gpu_array(data) -> bool:
    """
    Returns if data is a GPU array like Cupy.
    """
    return is_gpu_supported() and isinstance(data, ArrayGPU)


def is_gpu_dataframe(data) -> bool:
    """
    Returns if data is a GPU dataframe like Cudf.
    """
    return is_gpu_supported() and isinstance(data, DataFrameGPU)


def is_gpu_datatype(data) -> bool:
    """
    Returns if data is a GPU data type.
    """
    return is_gpu_supported() and isinstance(data, get_args(DataGPU))


def is_dask_cpu_array(data) -> bool:
    """
    Returns if data is a Dask array with CPU internal array.
    """
    if isinstance(data, DaskArray):
        # pylint: disable=protected-access
        if isinstance(data._meta, get_args(ArrayCPU)):
            return True
    return False


def is_dask_cpu_dataframe(data) -> bool:
    """
    Returns if data is a Dask dataframe with CPU internal dataframe.
    """
    try:
        if is_gpu_supported() and isinstance(data, get_args(DaskDataFrame)):
            # pylint: disable=protected-access
            if isinstance(data._meta, DataFrameCPU):
                return True
        elif isinstance(data, DaskDataFrame):
            # pylint: disable=protected-access
            if isinstance(data._meta, DataFrameCPU):
                return True
    # We need a Exception here due to Numpy bug.
    except TypeError: # pragma: no cover
        pass
    return False


def is_dask_gpu_array(data) -> bool:
    """
    Returns if data is a Dask array with GPU internal array.
    """
    if is_gpu_supported() and isinstance(data, DaskArray):
        # pylint: disable=protected-access
        if isinstance(data._meta, ArrayGPU):
            return True
    return False


def is_dask_gpu_dataframe(data) -> bool:
    """
    Returns if data is a Dask dataframe with GPU internal dataframe.
    """
    if is_gpu_supported() and isinstance(data, get_args(DaskDataFrame)):
        # pylint: disable=protected-access
        if isinstance(data._meta, DataFrameGPU):
            return True
    return False


def is_dask_array(data) -> bool:
    """
    Returns if data is a Dask array.
    """
    return isinstance(data, DaskArray)


def is_dask_dataframe(data) -> bool:
    """
    Returns if data is a Dask dataframe.
    """
    if is_gpu_supported():
        return isinstance(data, get_args(DaskDataFrame))
    return isinstance(data, DaskDataFrame)


def is_dask(data) -> bool:
    """
    Returns if data is a Dask data type.
    """
    return isinstance(data, get_args(DataDask))


def is_xarray_array(data) -> bool:
    """
    Returns if data is a Xarray.
    """
    return isinstance(data, XDataArray)
