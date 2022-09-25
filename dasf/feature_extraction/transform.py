#!/usr/bin/env python3

import dask.array as da
import dask.dataframe as ddf

from dasf.pipeline.types import TaskExecutorType
from dasf.transforms.base import Transform


class Normalize(Transform):
    def transform(self, X):
        return (X - X.mean()) / (X.std(ddof=0))


class ConcatenateToArray(Transform):
    def __init__(self, flatten=False):
        self.flatten = flatten

    def transform(self, **kwargs):
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
                    datas = self.xp.append(datas, self.xp.asarray([flat]),
                                           axis=0)
                else:
                    data = self.xp.asarray(kwargs[key])
                    datas = self.xp.append(datas, data, axis=len(data.shape))

        if self.flatten:
            data = self.xp.transpose(datas)
        else:
            data = datas

        return data.rechunk({1: data.shape[1]})


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


class SliceDataframe:
    def __init__(self, iline_index):
        self.iline_index = iline_index

    def run(self, X, y):
        cube_shape = y.shape

#       slice_idx = (self.iline_index * cube_shape[1] * cube_shape[2],
#                   (self.iline_index + 1) * cube_shape[1] * cube_shape[2])

#       slice_array = X[slice_idx[0] : slice_idx[1]]
#       slice_array = slice_array.reshape(cube_shape[1], cube_shape[2])

#       return slice_array.T
        if isinstance(X, da.core.Array):
            slice_array = X
        elif isinstance(X, ddf.core.DataFrame):
            slice_array = X.values

        return slice_array.reshape(cube_shape)


class GetSubDataframe:
    def __init__(self, percent):
        self.__percent = float(percent / 100.0)

    def transform(self, X):
        new_size = int(len(X) * self.__percent)

        return X.iloc[0:new_size]
