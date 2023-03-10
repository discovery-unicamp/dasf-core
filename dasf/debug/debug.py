#!/usr/bin/env python3

from IPython.core.display import HTML as iHTML
from IPython.core.display import display as idisplay

from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_dataframe
from dasf.utils.funcs import is_notebook


class Debug:
    """Print information about an operator (shape, datatype, etc.), and return
    the self object reference.

    Parameters
    ----------
    name : str
        Name of the operator.
    **kwargs : type
        Additional keyworkded arguments to `Operator`.

    """
    def display(self, X):
        if hasattr(X, "shape"):
            print("Datashape is:", X.shape)

        if (is_dask_array(X) or is_dask_dataframe(X)) and is_notebook():
            idisplay(iHTML(X._repr_html_()))
        else:
            print("Datatype is:", type(X))
            print("Data content is:", X)

        return X


class VisualizeDaskData:
    """Visualize DASK data from an operator.

    Parameters
    ----------
    filename : str
        A path to save the DASK visualization (the default is None).
    **kwargs : type
        Additional keyworkded arguments to `Operator`.

    """
    def __init__(self, filename: str = None):
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
