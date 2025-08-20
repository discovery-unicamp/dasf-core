#!/usr/bin/env python3

""" Debug DASF module. """

from IPython.core.display import HTML as iHTML
from IPython.core.display import display_html as idisplay

from dasf.utils.funcs import is_notebook
from dasf.utils.types import is_dask_array, is_dask_dataframe


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
        """Display useful information of the input data.

        Parameters
        ----------
        X : Any
            Any data that can be represented as a dataset.

        Returns
        -------
        data : Any
            Same input data without any transformation.
        """
        print(is_dask_array(X), is_dask_dataframe(X), is_notebook())
        if (is_dask_array(X) or is_dask_dataframe(X)) and is_notebook():
            idisplay(iHTML(X._repr_html_()), raw=True)
        else:
            if hasattr(X, "shape"):
                print("Datashape is:", X.shape)

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
        """ Generic constructor of the VisualizeDaskData object. """
        self.filename = filename

    def display(self, X):
        """Display Dask task graph using visualize method.

        Parameters
        ----------
        X : Any
            Any data that can be represented as a dataset.

        Returns
        -------
        data : Any
            Same input data without any transformation.
        """
        if not is_dask_array(X) and not is_dask_dataframe(X):
            print("WARNING: This is not a Dask element.")
            return X

        if self.filename is not None:
            X.visualize(self.filename)
        else:
            X.visualize()

        return X
