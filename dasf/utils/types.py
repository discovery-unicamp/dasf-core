#!/usr/bin/env python3

import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf
import xarray as xr

try:
    import cupy as cp
    import cudf
    import dask_cudf as dcudf
except ImportError:
    pass

from typing import Union, get_args

from dasf.utils.funcs import is_gpu_supported


ArrayCPU = Union[list, np.ndarray]
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
except NameError:
    pass


def is_array(data):
    return isinstance(data, get_args(Array))


def is_dataframe(data):
    return isinstance(data, get_args(DataFrame))


def is_cpu_array(data):
    return isinstance(data, get_args(ArrayCPU))


def is_cpu_dataframe(data):
    return isinstance(data, DataFrameCPU)


def is_cpu_datatype(data):
    return isinstance(data, get_args(DataCPU))


def is_gpu_array(data):
    return is_gpu_supported() and isinstance(data, ArrayGPU)


def is_gpu_dataframe(data):
    return is_gpu_supported() and isinstance(data, DataFrameGPU)


def is_gpu_datatype(data):
    return is_gpu_supported() and isinstance(data, get_args(DataPU))


def is_dask_cpu_array(data):
    if isinstance(data, DaskArray):
        if isinstance(data._meta, get_args(ArrayCPU)):
            return True
    return False


def is_dask_cpu_dataframe(data):
    try:
        if is_gpu_supported() and isinstance(data, get_args(DaskDataFrame)):
            if isinstance(data._meta, DataFrameCPU):
                return True
        elif isinstance(data, DaskDataFrame):
            if isinstance(data._meta, DataFrameCPU):
                return True
    # We need a Exception here due to Numpy bug.
    except TypeError:
        pass
    return False


def is_dask_gpu_array(data):
    if is_gpu_supported() and isinstance(data, DaskArray):
        if isinstance(data._meta, ArrayGPU):
            return True
    return False


def is_dask_gpu_dataframe(data):
    if is_gpu_supported() and isinstance(data, get_args(DaskDataFrame)):
        if isinstance(data._meta, DataFrameGPU):
            return True
    return False


def is_dask_array(data):
    return isinstance(data, DaskArray)


def is_dask_dataframe(data):
    if is_gpu_supported():
        return isinstance(data, get_args(DaskDataFrame))
    return isinstance(data, DaskDataFrame)


def is_dask(data):
    return isinstance(data, get_args(DataDask))


def is_xarray_array(data):
    return isinstance(data, XDataArray)
