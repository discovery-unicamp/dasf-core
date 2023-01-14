#!/usr/bin/env python3

import unittest
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

from dasf.utils.types import is_array
from dasf.utils.types import is_dataframe
from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_cpu_dataframe
from dasf.utils.types import is_cpu_datatype
from dasf.utils.types import is_gpu_array
from dasf.utils.types import is_gpu_dataframe
from dasf.utils.types import is_gpu_datatype
from dasf.utils.types import is_dask_cpu_array
from dasf.utils.types import is_dask_cpu_dataframe
from dasf.utils.types import is_dask_gpu_array
from dasf.utils.types import is_dask_gpu_dataframe
from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_dataframe
from dasf.utils.types import is_dask
from dasf.utils.types import is_xarray_array
from dasf.utils.funcs import is_gpu_supported



class TestTypes(unittest.TestCase):
    def setUp(self):
        base = np.random.randint(10, size=100)

        self.data_types = {
            "Numpy Array": [base, False],
            "List": [base.tolist(), False],
            "Dask Numpy Array": [da.from_array(base), False],
            "Pandas DataFrame": [pd.DataFrame(base.tolist(),
                                              columns=["test"]),
                                 False],
            "Dask DataFrame": [ddf.from_pandas(pd.DataFrame(base.tolist(),
                                                            columns=["test"]),
                                               npartitions=20),
                               False],
            "Xarray DataArray": [xr.DataArray(base), False],
        }

        if is_gpu_supported():
            cupy_base = cp.asarray(base)

            self.data_types.update({
                "Cupy Array": [cupy_base, False],
                "Dask Cupy Array": [da.from_array(cupy_base), False],
                "CuDF": [cudf.DataFrame(base.tolist(),
                                        columns=["test"]),
                         False],
                "Dask CuDF": [dcudf.from_cudf(cudf.DataFrame(base.tolist(),
                                                             columns=["test"]),
                                              npartitions=20),
                              False],
            })

    def __set_data(self, keys):
        for key in self.data_types:
            self.data_types[key][1] = False
            if key in keys:
                self.data_types[key][1] = True

    def __assert_all_data(self, func):
        for key in self.data_types:
            self.assertEqual(func(self.data_types[key][0]),
                             self.data_types[key][1],
                             "Type '%s' is %s while is expects %s" %
                             (str(type(self.data_types[key][0])),
                              func(self.data_types[key][0]),
                              self.data_types[key][1]))

    def test_is_array(self):
        keys = [
            "Numpy Array",
            "List",
            "Dask Numpy Array",
            "Cupy Array",
            "Dask Cupy Array",
            "Xarray DataArray"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_array)

    def test_is_dataframe(self):
        keys = [
            "Pandas DataFrame",
            "Dask DataFrame",
            "CuDF",
            "Dask CuDF"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_dataframe)

    def test_is_cpu_array(self):
        keys = [
            "Numpy Array",
            "List"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_cpu_array)

    def test_is_cpu_dataframe(self):
        keys = [
            "Pandas DataFrame"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_cpu_dataframe)

    def test_is_cpu_datatype(self):
        keys = [
            "Numpy Array",
            "List",
            "Pandas DataFrame"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_cpu_datatype)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_is_gpu_array(self):
        keys = [
            "Cupy Array"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_gpu_array)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_is_gpu_dataframe(self):
        keys = [
            "CuDF"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_gpu_dataframe)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_is_gpu_datatype(self):
        keys = [
            "Cupy Array",
            "CuDF"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_gpu_datatype)

    def test_is_dask_cpu_array(self):
        keys = [
            "Dask Numpy Array"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_dask_cpu_array)

    def test_is_dask_cpu_dataframe(self):
        keys = [
            "Dask DataFrame"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_dask_cpu_dataframe)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_is_dask_gpu_array(self):
        keys = [
            "Dask Cupy Array"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_dask_gpu_array)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_is_dask_gpu_dataframe(self):
        keys = [
            "Dask CuDF"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_dask_gpu_dataframe)

    def test_is_dask_array(self):
        keys = [
            "Dask Numpy Array",
            "Dask Cupy Array"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_dask_array)

    def test_is_dask_dataframe(self):
        keys = [
            "Dask DataFrame",
            "Dask CuDF"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_dask_dataframe)

    def test_is_dask(self):
        keys = [
            "Dask Numpy Array",
            "Dask DataFrame",
            "Dask Cupy Array",
            "Dask CuDF"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_dask)

    def test_is_xarray_array(self):
        keys = [
            "Xarray DataArray"
        ]

        self.__set_data(keys)

        self.__assert_all_data(is_xarray_array)
