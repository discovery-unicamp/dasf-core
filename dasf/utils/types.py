#!/usr/bin/env python3
""" Data types handlers. """

from typing import Union, get_args

import dask.array as da
import dask.dataframe as ddf
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.base import is_dask_collection
from dask.utils import is_arraylike, is_cupy_type, is_dataframe_like, is_series_like

try:
    import GPUtil
    if len(GPUtil.getGPUs()) == 0:  # check if GPU are available in current env
        raise ImportError("There is no GPU available here")
    import cudf
    import cupy as cp
    import dask_cudf as dcudf
except:  # pragma: no cover
    pass

ArrayCPU = Union[list, np.ndarray, zarr.Array]
DataFrameCPU = Union[pd.DataFrame]

DataCPU = Union[ArrayCPU, DataFrameCPU]

DaskArray = Union[da.core.Array]
DaskDataFrameCPU = Union[ddf.DataFrame]

XDataArray = Union[xr.DataArray]

Array = Union[ArrayCPU, DaskArray, XDataArray]
DaskDataFrame = Union[DaskDataFrameCPU]
DataFrame = Union[DataFrameCPU, DaskDataFrameCPU]
DataDask = Union[DaskArray, DaskDataFrameCPU]
try:
    ArrayGPU = Union[cp.ndarray]
    DataFrameGPU = Union[cudf.DataFrame]

    DataGPU = Union[ArrayGPU, DataFrameGPU]

    DaskDataFrameGPU = Union[dcudf.core.DataFrame,
                             dcudf._expr.collection.DataFrame]

    Array = Union[Array, ArrayGPU]
    DaskDataFrame = Union[DaskDataFrame, DaskDataFrameGPU]
    DataFrame = Union[DataFrame, DaskDataFrame, DataFrameGPU]
    DataDask = Union[DataDask, DaskDataFrame]
except NameError:  # pragma: no cover
    pass


def is_array(data) -> bool:
    """
    Returns if data is a generic array.
    """
    return isinstance(data, list) or is_arraylike(data)


def is_dataframe(data) -> bool:
    """
    Returns if data is a generic dataframe.
    """
    return is_dataframe_like(data)


def is_series(data) -> bool:
    """
    Returns if data is a generic series.
    """
    return is_series_like(data)


def is_cpu_array(data) -> bool:
    """
    Returns if data is a CPU arrau like Numpy.
    """
    return isinstance(data, get_args(ArrayCPU))


def is_cpu_dataframe(data) -> bool:
    """
    Returns if data is a CPU dataframe like Pandas.
    """
    return (isinstance(data, DataFrameCPU) and
            not is_dask_collection(data) and
            is_dataframe(data))


def is_cpu_datatype(data) -> bool:
    """
    Returns if data is a CPU data type.
    """
    return isinstance(data, get_args(DataCPU))


def is_gpu_array(data) -> bool:
    """
    Returns if data is a GPU array like Cupy.
    """
    return is_cupy_type(data)


def is_gpu_dataframe(data) -> bool:
    """
    Returns if data is a GPU dataframe like Cudf.
    """
    try:
        return (isinstance(data, DataFrameGPU) and
                not is_dask_collection(data) and
                is_dataframe(data))
    except NameError:
        return False


def is_gpu_datatype(data) -> bool:
    """
    Returns if data is a GPU data type.
    """
    try:
        return isinstance(data, get_args(DataGPU))
    except NameError:
        return False


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
    return (hasattr(data, '_meta') and
            is_cpu_dataframe(data._meta) and
            is_dask_collection(data) and
            is_dataframe(data))

def is_dask_gpu_array(data) -> bool:
    """
    Returns if data is a Dask array with GPU internal array.
    """
    if isinstance(data, DaskArray):
        # pylint: disable=protected-access
        if is_cupy_type(data._meta):
            return True
    return False


def is_dask_gpu_dataframe(data) -> bool:
    """
    Returns if data is a Dask dataframe with GPU internal dataframe.
    """
    return (hasattr(data, '_meta') and
            is_gpu_dataframe(data._meta) and
            is_dask_collection(data) and
            is_dataframe(data))


def is_dask_array(data) -> bool:
    """
    Returns if data is a Dask array.
    """
    return isinstance(data, DaskArray)


def is_dask_dataframe(data) -> bool:
    """
    Returns if data is a Dask dataframe.
    """
    return (is_dask_collection(data) and
            is_dataframe(data))


def is_dask(data) -> bool:
    """
    Returns if data is a Dask data type.
    """
    return is_dask_collection(data)


def is_xarray_array(data) -> bool:
    """
    Returns if data is a Xarray.
    """
    return isinstance(data, XDataArray)
