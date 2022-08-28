#!/usr/bin/env python3

import dask.array as da
import dask.dataframe as ddf

from IPython.core.display import HTML as iHTML
from IPython.core.display import display as idisplay

from dasf.pipeline import Operator


class Debug(Operator):
    def __init__(self, name=None, **kwargs):
        self.__name = name

        if name is None:
            self.__name = "Debug"

        super().__init__(name=self.__name, **kwargs)

    def run(self, X):
        if hasattr(X, "shape"):
            print("Datashape is:", X.shape)

        if isinstance(X, da.core.Array) or isinstance(X, ddf.core.DataFrame):
            idisplay(iHTML(X._repr_html_()))
        else:
            print("Datatype is:", type(X))
            print("Data content is:", X)

        return X


class VisualizeDaskData(Operator):
    def __init__(self, filename=None, **kwargs):
        super().__init__(name="Visualize Dask Data", **kwargs)

        self.filename = filename

    def run(self, X):
        if not isinstance(X, da.core.Array) and \
           not isinstance(X, ddf.core.DataFrame):
            self.logger.warning("This is not a Dask element.")
            return X

        if self.filename is not None:
            X.visualize(self.filename)
        else:
            X.visualize()

        return X
