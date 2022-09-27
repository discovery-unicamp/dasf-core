#!/usr/bin/env python3

import numpy as np
import dask.array as da
import dask.dataframe as ddf

try:
    import cupy as cp
except ImportError:
    pass

from dasf.utils.types import is_array
from dasf.utils.types import is_dataframe
from dasf.pipeline.types import TaskExecutorType
from dasf.transforms.base import Transform, Fit


class Normalize(Transform):
    def transform(self, X):
        return (X - X.mean()) / (X.std(ddof=0))


class ConcatenateToArray(Transform):
    def __init__(self, flatten=False):
        self.flatten = flatten

    def __transform_generic(self, xp, **kwargs):
        datas = None
        for key in kwargs:
            if datas is None:
                if self.flatten:
                    flat = kwargs[key].flatten()
                    datas = xp.asarray([flat])
                else:
                    data = xp.asarray(kwargs[key])
                    datas = xp.expand_dim(data, axis=len(data.shape))
            else:
                if self.flatten:
                    flat = kwargs[key].flatten()
                    datas = xp.append(datas, xp.asarray([flat]),
                                      axis=0)
                else:
                    data = xp.asarray(kwargs[key])
                    datas = xp.append(datas, data, axis=len(data.shape))

        if self.flatten:
            data = xp.transpose(datas)
        else:
            data = datas

        return data
#        return data.rechunk({1: data.shape[1]})

    def _transform_cpu(self, **kwargs):
        return self.__transform_generic(np, **kwargs)

    def _transform_gpu(self, **kwargs):
        return self.__transform_generic(cp, **kwargs)


class SampleDataframe:
    def __init__(self, percent):
        self.__percent = float(percent / 100.0)

    def run(self, X):
        return X.sample(n=int(len(X) * self.__percent))


class GetSubeCubeArray:
    def __init__(self, percent):
        self.__percent = float(percent / 100.0)

        assert (
            self.__percent > 0 and self.__percent <= 1.0
        ), "Percent must be in [0,1] range."

    def transform(self, X):
        i_num, x_num, t_num = X.shape

        i_start_idx = int((i_num - (i_num * self.__percent)) / 2)
        i_end_idx = int(i_start_idx + (self.__percent * i_num))

        x_start_idx = int((x_num - (x_num * self.__percent)) / 2)
        x_end_idx = int(x_start_idx + (self.__percent * x_num))

        t_start_idx = int((t_num - (t_num * self.__percent)) / 2)
        t_end_idx = int(t_start_idx + (self.__percent * t_num))

        return X[i_start_idx:i_end_idx,
                 x_start_idx:x_end_idx,
                 t_start_idx:t_end_idx]


class SliceDataframe(Fit):
    def __init__(self, iline_index):
        self.iline_index = iline_index

    def fit(self, X, y):
        cube_shape = y.shape

        if is_array(X):
            slice_array = X
        elif is_dataframe(X):
            slice_array = X.values
        else:
            raise ValueError("X is not a known datatype.")

        return slice_array.reshape(cube_shape)


class GetSubDataframe:
    def __init__(self, percent):
        self.__percent = float(percent / 100.0)

    def transform(self, X):
        new_size = int(len(X) * self.__percent)

        return X.iloc[0:new_size]
