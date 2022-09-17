#!/usr/bin/env python3

import dask.array as da
import dask.dataframe as ddf

from dasf.pipeline import Operator


class PersistDaskData(Operator):
    def __init__(self):
        super().__init__(name="Persist Dask data")

    def run(self, X):
        if isinstance(X, da.core.Array) or isinstance(X, ddf.core.DataFrame):
            new_data = X.persist()
        else:
            new_data = X

        return new_data


class LoadDaskData(Operator):
    def __init__(self):
        super().__init__(name="Load Dask data locally")

    def run(self, data):
        if isinstance(data, da.core.Array) or isinstance(data, ddf.core.DataFrame):
            new_data = data.compute()
        else:
            new_data = data

        return new_data
