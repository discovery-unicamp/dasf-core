#!/usr/bin/env python3

""" Memory Management module. """

from dasf.transforms.base import Transform
from dasf.utils.types import is_dask_array, is_dask_dataframe


class PersistDaskData(Transform):
    """Allow persisting a dask array to memory and return a copy of the object.
    It will gather the data blocks from all workers and resembles locally.
    """
    def __lazy_transform_generic(self, X):
        """
        Generic lazy transform to persist Dask data in memory.

        Parameters
        ----------
        X : dask.array.Array or dask.dataframe.DataFrame
            Input Dask data structure to persist.

        Returns
        -------
        dask.array.Array or dask.dataframe.DataFrame
            Persisted data structure.
        """
        if is_dask_array(X) or is_dask_dataframe(X):
            new_data = X.persist()
        else:
            new_data = X

        return new_data

    def _lazy_transform_cpu(self, X):
        """
        CPU lazy transform to persist Dask data.

        Parameters
        ----------
        X : dask.array.Array or dask.dataframe.DataFrame
            Input Dask data structure.

        Returns
        -------
        dask.array.Array or dask.dataframe.DataFrame
            Persisted data structure.
        """
        return self.__lazy_transform_generic(X)

    def _lazy_transform_gpu(self, X):
        """
        GPU lazy transform to persist Dask data.

        Parameters
        ----------
        X : dask.array.Array or dask.dataframe.DataFrame
            Input Dask data structure.

        Returns
        -------
        dask.array.Array or dask.dataframe.DataFrame
            Persisted data structure.
        """
        return self.__lazy_transform_generic(X)

    def _transform_cpu(self, X):
        """
        CPU transform (bypass for local data).

        Parameters
        ----------
        X : Any
            Input data.

        Returns
        -------
        Any
            Input data unchanged (bypass).
        """
        # Bypass because the data is local
        return X

    def _transform_gpu(self, X):
        """
        GPU transform (bypass for local data).

        Parameters
        ----------
        X : Any
            Input data.

        Returns
        -------
        Any
            Input data unchanged (bypass).
        """
        # Bypass because the data is local
        return X


class ComputeDaskData(Transform):
    """Allow persisting a dask array to memory. It will gather the data blocks
    from all workers and resembles locally.
    """
    def __lazy_transform_generic(self, X):
        """
        Generic lazy transform to compute Dask data and bring to memory.

        Parameters
        ----------
        X : dask.array.Array or dask.dataframe.DataFrame
            Input Dask data structure to compute.

        Returns
        -------
        array-like or DataFrame
            Computed data structure in local memory.
        """
        if is_dask_array(X) or is_dask_dataframe(X):
            new_data = X.compute()
        else:
            new_data = X

        return new_data

    def _lazy_transform_cpu(self, X):
        """
        CPU lazy transform to compute Dask data.

        Parameters
        ----------
        X : dask.array.Array or dask.dataframe.DataFrame
            Input Dask data structure.

        Returns
        -------
        array-like or DataFrame
            Computed data structure.
        """
        return self.__lazy_transform_generic(X)

    def _lazy_transform_gpu(self, X):
        """
        GPU lazy transform to compute Dask data.

        Parameters
        ----------
        X : dask.array.Array or dask.dataframe.DataFrame
            Input Dask data structure.

        Returns
        -------
        array-like or DataFrame
            Computed data structure.
        """
        return self.__lazy_transform_generic(X)

    def _transform_cpu(self, X):
        """
        CPU transform (bypass for local data).

        Parameters
        ----------
        X : Any
            Input data.

        Returns
        -------
        Any
            Input data unchanged (bypass).
        """
        # Bypass because the data is local
        return X

    def _transform_gpu(self, X):
        """
        GPU transform (bypass for local data).

        Parameters
        ----------
        X : Any
            Input data.

        Returns
        -------
        Any
            Input data unchanged (bypass).
        """
        # Bypass because the data is local
        return X
