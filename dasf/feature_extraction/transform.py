#!/usr/bin/env python3

import math

import numpy as np
import pandas as pd

import dask.array as da
import dask.dataframe as ddf

from dasf.utils import utils
from dasf.pipeline import Operator
from dasf.utils.utils import is_gpu_supported
from dasf.pipeline.types import TaskExecutorType
from dasf.utils.types import ALL_ARRAY_TYPES
from dasf.utils.types import ALL_DATAFRAME_TYPES

try:
    import cupy as cp
    import cudf
except ImportError:
    pass


class Normalize(Operator):
    """Normalize the data using standard scaler normalizator.

    """
    def __init__(self):
        super().__init__(name="Normalize Data")

        self.set_inputs(data=ALL_ARRAY_TYPES)

        self.set_output(ALL_ARRAY_TYPES)

    def run(self, X):
        ret = (X - X.mean())/(X.std(ddof=0))

        return ret


class ConcatenateToArray(Operator):
    """Concatenate data from different Arrays into a single array.

    Parameters
    ----------
    flatten : bool
        If the arrays must be flatten prior concatenating. If `False`, the
        arrays must share the shape of last dimansions in order to be
        concatenated (the default is False).

    """
    def __init__(self, flatten: bool = False):
        super().__init__(name="Concatenate and Flatten Data to Array")

        self.flatten = flatten

        self.set_inputs(data=ALL_ARRAY_TYPES)

        self.set_output(ALL_ARRAY_TYPES)

        self.dtype = TaskExecutorType.single_cpu

    def run(self, **kwargs):
        """Concatenate arrays passed as keyworkded aguments. The key represent
        Array names and the values are the arrays.

        Parameters
        ----------
        **kwargs : type
            Dictionary with datasets.

        Returns
        -------
        Any
            A concatenated Array.

        """
        datas = None
        for key in kwargs:
            if datas is None:
                if self.flatten:
                    flat = kwargs[key].flatten()
                    datas = self.xp.asarray([flat])
                else:
                    data = self.xp.asarray(kwargs[key])
                    datas = self.xp.expand_dim(data, axis=len(data.shape))
            else:
                if self.flatten:
                    flat = kwargs[key].flatten()
                    datas = self.xp.append(datas,
                                           self.xp.asarray([flat]),
                                           axis=0)
                else:
                    data = self.xp.asarray(kwargs[key])
                    datas = self.xp.append(datas, data, axis=len(data.shape))

        if self.flatten:
            data = self.xp.transpose(datas)
        else:
            data = datas

        return data.rechunk({1: data.shape[1]})


class SampleDataframe(Operator):
    """Return a subset with random samples of the original dataset.

    Parameters
    ----------
    percent : float
        Percentage of the samples to get from the dataset.

    """
    def __init__(self, percent: float):
        super().__init__(name="Sample DataFrame using " + str(percent) + "%")

        self.set_inputs(data=ALL_ARRAY_TYPES)

        self.set_output(ALL_ARRAY_TYPES)

        self.__percent = float(percent/100.0)

    def run(self, X):
        """Returns a subset with random samples from the dataset `X`.

        Parameters
        ----------
        X : Any
            The dataset.

        Returns
        -------
        Any
            The sampled subset.

        """
        return X.sample(n=int(len(X) * self.__percent))


class GetSubeCubeArray(Operator):
    """Get a subcube with x% of samples from the original one.

    Parameters
    ----------
    percent : float
        Percentage of the samples to get from the cube.

    """
    def __init__(self, percent: float):
        super().__init__(name="Get Smaller 3D Data")

        self.__percent = float(percent/100.0)

        assert self.__percent > 0 and self.__percent <= 1.0, \
            "Percent must be in [0,1] range."

        self.set_inputs(data=ALL_ARRAY_TYPES)

        self.set_output(ALL_ARRAY_TYPES)

    def run(self, data):
        """Returns a subcube from the original one.

        Parameters
        ----------
        data : Any
            The cube.

        Returns
        -------
        Any
            A subcube from the original.

        """
        i_num, x_num, t_num = data.shape

        i_start_idx = int((i_num - (i_num * self.__percent)) / 2)
        i_end_idx = int(i_start_idx + (self.__percent * i_num))

        x_start_idx = int((x_num - (x_num * self.__percent)) / 2)
        x_end_idx = int(x_start_idx + (self.__percent * x_num))

        t_start_idx = int((t_num - (t_num * self.__percent)) / 2)
        t_end_idx = int(t_start_idx + (self.__percent * t_num))

        return data[i_start_idx:i_end_idx,
                    x_start_idx:x_end_idx,
                    t_start_idx:t_end_idx]


class SliceDataframe(Operator):
    """Get a slice of a cube. An inline slice is a section over the x-axis.

    Parameters
    ----------
    iline_index : int
        The index of the inline to get.

    """
    def __init__(self, iline_index: int):
        super().__init__(name="Slice Dataframe in " + str(iline_index))

        self.iline_index = iline_index

        self.set_inputs(dataframe=ALL_DATAFRAME_TYPES,
                        data=ALL_ARRAY_TYPES)
        self.set_output(ALL_ARRAY_TYPES)

    def run(self, X, y):
        cube_shape = y.shape

#        slice_idx = (self.iline_index * cube_shape[1] * cube_shape[2],
#                    (self.iline_index + 1) * cube_shape[1] * cube_shape[2])

#        slice_array = X[slice_idx[0] : slice_idx[1]]
#        slice_array = slice_array.reshape(cube_shape[1], cube_shape[2])

#        return slice_array.T
        if isinstance(X, da.core.Array):
            slice_array = X
        elif isinstance(X, ddf.core.DataFrame):
            slice_array = X.values

        return slice_array.reshape(cube_shape)


class GetSubDataframe(Operator):
    """Get the first x% samples from the dataset.

    Parameters
    ----------
    percent : float
        Percentage of the samples to get from the dataframe.

    """
    def __init__(self, percent: float):
        super().__init__(name="Get Smaller Dataframe")

        self.__percent = float(percent/100.0)

        self.set_inputs(data=ALL_ARRAY_TYPES)
        self.set_output(ALL_ARRAY_TYPES)

    def run(self, data):
        """Return x% samples of the dataset.

        Parameters
        ----------
        data : Any
            A dataframe type data.

        Returns
        -------
        Any
            A dataframe with only the x% samples.

        """
        new_size = int(len(data) * self.__percent)

        return data.iloc[0:new_size]
