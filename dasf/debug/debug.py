#!/usr/bin/env python3

import dask.array as da
import dask.dataframe as ddf

from IPython.core.display import HTML as iHTML
from IPython.core.display import display as idisplay

from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_dataframe


class Debug:
    def display(self, X):
        if hasattr(X, "shape"):
            print("Datashape is:", X.shape)

        if is_dask_array(X) or is_dask_dataframe(X):
            idisplay(iHTML(X._repr_html_()))
        else:
            print("Datatype is:", type(X))
            print("Data content is:", X)

        return X


class VisualizeDaskData:
    def __init__(self, filename=None):
        self.filename = filename

    def display(self, X):
        if not is_dask_array(X) and not is_dask_dataframe(X):
            self.logger.warning("This is not a Dask element.")
            return X

        if self.filename is not None:
            X.visualize(self.filename)
        else:
            X.visualize()

        return X
