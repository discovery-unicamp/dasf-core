#!/usr/bin/env python3

""" All the essential data transforms module. """


import dask
import dask.dataframe as ddf
import h5py
import numpy as np
import pandas as pd
import zarr

from dasf.transforms.base import Transform
from dasf.utils.types import is_array, is_dask_array, is_dask_gpu_array

try:
    import GPUtil
    if len(GPUtil.getGPUs()) == 0:  # check if GPU are available in current env
        raise ImportError("There is no GPU available here")
    import cudf
    import cupy as cp
except ImportError:  # pragma: no cover
    pass


class ExtractData(Transform):
    """
    Extract data from Dataset object

    """
    def transform(self, X):
        """
        Extract data from datasets that contains internal data.

        Parameters
        ----------
        X : Dataset-like
            A dataset object that could be anything that contains an internal
            structure representing the raw data.

        Returns
        -------
        data : Any
            Any representation of the internal Dataset data.

        """
        if hasattr(X, "_data") and X._data is not None:
            return X._data
        raise ValueError("Data could not be extracted. "
                         "Dataset needs to be previously loaded.")


class Normalize(Transform):
    """
    Normalize data object

    """
    def transform(self, X):
        """
        Normalize the input data based on mean() and std().

        Parameters
        ----------
        X : Any
            Any data that could be normalized based on mean and standard
            deviation.

        Returns
        -------
        data : Any
            Normalized data

        """
        return (X - X.mean()) / (X.std(ddof=0))


class ArrayToZarr(Transform):
    """
    Transform array data to Zarr format.

    This class converts array-like objects (DatasetArray, Dask arrays, or
    regular arrays) into Zarr format for efficient storage and access.

    Parameters
    ----------
    chunks : tuple or None
        Chunk size for the Zarr array. If None, uses existing chunks.
    save : bool
        Whether to save the converted array to disk (default: True).
    filename : str or None
        Output filename for the Zarr array. If None, uses original filename.
    """
    def __init__(self, chunks=None, save=True, filename=None):
        """
        Initialize ArrayToZarr transform.

        Parameters
        ----------
        chunks : tuple or None
            Chunk size for the Zarr array.
        save : bool
            Whether to save the converted array to disk.
        filename : str or None
            Output filename for the Zarr array.
        """
        self.chunks = chunks
        # TODO: implement the possibility of not saving
        self.save = True
        self.filename = filename

    @staticmethod
    def _convert_filename(url):
        """
        Convert filename to Zarr format by replacing .npy with .zarr.

        Parameters
        ----------
        url : str
            Original filename.

        Returns
        -------
        str
            Filename with .zarr extension.
        """
        if url.endswith(".npy"):
            return url.replace(".npy", ".zarr")
        return url + ".zarr"

    def _lazy_transform_generic_all(self, data):
        """
        Convert lazy data to Zarr format.

        Parameters
        ----------
        data : dask.array.Array
            Lazy array data to convert.

        Returns
        -------
        str
            Path to the created Zarr file.
        """
        if self.filename:
            url = self.filename
        elif hasattr(data, '_root_file'):
            url = data._root_file
        else:
            raise Exception("Array requires a valid path to convert to Zarr.")

        if data is None:
            raise Exception("Dataset needs to be loaded first.")

        url = self._convert_filename(url)

        # XXX: Workaround to avoid error with CuPy and Zarr library
        if is_dask_gpu_array(data):
            data = data.map_blocks(lambda x: x.get())

        data.to_zarr(url)

        return url

    def _transform_generic_all(self, data, chunks, **kwargs):
        """
        Convert regular array data to Zarr format.

        Parameters
        ----------
        data : array-like
            Array data to convert.
        chunks : tuple
            Chunk size for the Zarr array.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        str
            Path to the created Zarr file.
        """
        if data is None:
            raise Exception("Dataset needs to be loaded first.")

        if not chunks:
            raise Exception("Chunks needs to be passed for non lazy arrays.")

        if self.filename:
            url = self.filename
        else:
            raise Exception("Array requires a valid path to convert to Zarr.")

        url = self._convert_filename(url)

        z = zarr.open(store=url, mode='w', shape=data.shape,
                      chunks=chunks, dtype='i4')

        z = data  # noqa: F841

        return url

    def _lazy_transform_generic(self, X, **kwargs):
        """
        Generic lazy transform for converting arrays to Zarr.

        Parameters
        ----------
        X : DatasetArray or dask.array.Array
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetZarr
            Zarr dataset object.
        """
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray, DatasetZarr

        name = None

        if isinstance(X, DatasetArray):
            name = X._name
            chunks = X._chunks

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._lazy_transform_generic_all(X._data)
        elif is_dask_array(X):
            chunks = X.chunks

            url = self._lazy_transform_generic_all(X)
        else:
            raise Exception("It is not an Array type.")

        return DatasetZarr(name=name, download=False, root=url, chunks=chunks)

    def _transform_generic(self, X, **kwargs):
        """
        Generic transform for converting arrays to Zarr.

        Parameters
        ----------
        X : DatasetArray or array-like
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetZarr
            Zarr dataset object.
        """
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray, DatasetZarr

        name = None
        url = None

        if hasattr(X, '_chunks') and \
           (X._chunks is not None and X._chunks != 'auto'):
            chunks = X._chunks
        else:
            chunks = self.chunks

        if chunks is None:
            raise Exception("Chunks needs to be specified.")

        if isinstance(X, DatasetArray):
            name = X._name

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._transform_generic_all(X._data, chunks)
        elif is_array(X):
            url = self._transform_generic_all(X, chunks)
        else:
            raise Exception("It is not an Array type.")

        return DatasetZarr(name=name, download=False, root=url, chunks=chunks)

    def _lazy_transform_gpu(self, X, **kwargs):
        """
        GPU lazy transform for converting arrays to Zarr.

        Parameters
        ----------
        X : DatasetArray or dask.array.Array
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetZarr
            Zarr dataset object.
        """
        return self._lazy_transform_generic(X, **kwargs)

    def _lazy_transform_cpu(self, X, **kwargs):
        """
        CPU lazy transform for converting arrays to Zarr.

        Parameters
        ----------
        X : DatasetArray or dask.array.Array
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetZarr
            Zarr dataset object.
        """
        return self._lazy_transform_generic(X, **kwargs)

    def _transform_gpu(self, X, **kwargs):
        """
        GPU transform for converting arrays to Zarr.

        Parameters
        ----------
        X : DatasetArray or array-like
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetZarr
            Zarr dataset object.
        """
        return self._transform_generic(X, **kwargs)

    def _transform_cpu(self, X, **kwargs):
        """
        CPU transform for converting arrays to Zarr.

        Parameters
        ----------
        X : DatasetArray or array-like
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetZarr
            Zarr dataset object.
        """
        return self._transform_generic(X, **kwargs)


class ArrayToHDF5(Transform):
    """
    Transform array data to HDF5 format.

    This class converts array-like objects (DatasetArray, Dask arrays, or
    regular arrays) into HDF5 format for efficient storage and access.

    Parameters
    ----------
    dataset_path : str
        The path within the HDF5 file where the dataset will be stored.
    chunks : tuple or None
        Chunk size for the HDF5 array. If None, uses existing chunks.
    save : bool
        Whether to save the converted array to disk (default: True).
    filename : str or None
        Output filename for the HDF5 array. If None, uses original filename.
    """
    def __init__(self, dataset_path, chunks=None, save=True, filename=None):
        """
        Initialize ArrayToHDF5 transform.

        Parameters
        ----------
        dataset_path : str
            The path within the HDF5 file where the dataset will be stored.
        chunks : tuple or None
            Chunk size for the HDF5 array.
        save : bool
            Whether to save the converted array to disk.
        filename : str or None
            Output filename for the HDF5 array.
        """

        self.dataset_path = dataset_path
        self.chunks = chunks
        # TODO: implement the possibility of not saving
        self.save = True
        self.filename = filename

    @staticmethod
    def _convert_filename(url):
        """
        Convert filename to HDF5 format by replacing .npy with .hdf5.

        Parameters
        ----------
        url : str
            Original filename.

        Returns
        -------
        str
            Filename with .hdf5 extension.
        """
        if url.endswith(".npy"):
            return url.replace(".npy", ".hdf5")
        return url + ".hdf5"

    def _lazy_transform_generic_all(self, data):
        """
        Convert lazy data to HDF5 format.

        Parameters
        ----------
        data : dask.array.Array
            Lazy array data to convert.

        Returns
        -------
        str
            Path to the created HDF5 file.
        """
        if self.filename:
            url = self.filename
        elif hasattr(data, '_root_file'):
            url = data._root_file
        else:
            raise Exception("Array requires a valid path to convert to HDF5.")

        if data is None:
            raise Exception("Dataset needs to be loaded first.")

        url = self._convert_filename(url)

        data.to_hdf5(url, self.dataset_path)

        return url

    def _transform_generic_all(self, data):
        """
        Convert regular array data to HDF5 format.

        Parameters
        ----------
        data : array-like
            Array data to convert.

        Returns
        -------
        str
            Path to the created HDF5 file.
        """
        if data is None:
            raise Exception("Dataset needs to be loaded first.")

        if self.filename:
            url = self.filename
        else:
            raise Exception("Array requires a valid path to convert to Zarr.")

        url = self._convert_filename(url)

        h5f = h5py.File(url, 'w')
        h5f.create_dataset(self.dataset_path, data=data)
        h5f.close()

        return url

    def _lazy_transform_generic(self, X, **kwargs):
        """
        Generic lazy transform for converting arrays to HDF5.

        Parameters
        ----------
        X : DatasetArray or dask.array.Array
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetHDF5
            HDF5 dataset object.
        """
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray, DatasetHDF5

        name = None
        chunks = None

        if isinstance(X, DatasetArray):
            name = X._name
            chunks = X._chunks

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._lazy_transform_generic_all(X._data)
        elif is_dask_array(X):
            chunks = X.chunks

            url = self._lazy_transform_generic_all(X)
        else:
            raise Exception("It is not an Array type.")

        return DatasetHDF5(name=name, download=False, root=url, chunks=chunks,
                           dataset_path=self.dataset_path)

    def _transform_generic(self, X, **kwargs):
        """
        Generic transform for converting arrays to HDF5.

        Parameters
        ----------
        X : DatasetArray or array-like
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetHDF5
            HDF5 dataset object.
        """
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetArray, DatasetHDF5

        name = None
        url = None

        if hasattr(X, '_chunks') and \
           (X._chunks is not None and X._chunks != 'auto'):
            chunks = X._chunks
        else:
            chunks = self.chunks

        if isinstance(X, DatasetArray):
            name = X._name

            if not self.filename and hasattr(X, '_root_file'):
                self.filename = X._root_file

            url = self._transform_generic_all(X._data)
        elif is_array(X):
            url = self._transform_generic_all(X)
        else:
            raise Exception("It is not an Array type.")

        return DatasetHDF5(name=name, download=False, root=url, chunks=chunks,
                           dataset_path=self.dataset_path)

    def _lazy_transform_gpu(self, X, **kwargs):
        """
        GPU lazy transform for converting arrays to HDF5.

        Parameters
        ----------
        X : DatasetArray or dask.array.Array
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetHDF5
            HDF5 dataset object.
        """
        return self._lazy_transform_generic(X, **kwargs)

    def _lazy_transform_cpu(self, X, **kwargs):
        """
        CPU lazy transform for converting arrays to HDF5.

        Parameters
        ----------
        X : DatasetArray or dask.array.Array
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetHDF5
            HDF5 dataset object.
        """
        return self._lazy_transform_generic(X, **kwargs)

    def _transform_gpu(self, X, **kwargs):
        """
        GPU transform for converting arrays to HDF5.

        Parameters
        ----------
        X : DatasetArray or array-like
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetHDF5
            HDF5 dataset object.
        """
        return self._transform_generic(X, **kwargs)

    def _transform_cpu(self, X, **kwargs):
        """
        CPU transform for converting arrays to HDF5.

        Parameters
        ----------
        X : DatasetArray or array-like
            Input array to convert.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DatasetHDF5
            HDF5 dataset object.
        """
        return self._transform_generic(X, **kwargs)


class ZarrToArray(Transform):
    """
    Transform Zarr data to array format.

    This class converts Zarr dataset objects back to array format
    for further processing or storage.

    Parameters
    ----------
    chunks : tuple or None
        Chunk size for the output array. If None, uses existing chunks.
    save : bool
        Whether to save the converted array to disk (default: True).
    filename : str or None
        Output filename for the array. If None, uses original filename.
    """
    def __init__(self, chunks=None, save=True, filename=None):
        """
        Initialize ZarrToArray transform.

        Parameters
        ----------
        chunks : tuple or None
            Chunk size for the output array.
        save : bool
            Whether to save the converted array to disk.
        filename : str or None
            Output filename for the array.
        """
        self.chunks = chunks
        self.save = save
        self.filename = filename

    @staticmethod
    def _convert_filename(url):
        """
        Convert filename from Zarr format to array format by replacing .zarr with .npy.

        Parameters
        ----------
        url : str
            Original Zarr filename.

        Returns
        -------
        str
            Filename with .npy extension.
        """
        if url.endswith(".zarr"):
            return url.replace(".zarr", ".npy")
        return url + ".npy"

    def transform(self, X):
        """
        Transform a Zarr dataset to array format.

        Parameters
        ----------
        X : DatasetZarr
            Input Zarr dataset to convert to array format.

        Returns
        -------
        array-like
            Array representation of the Zarr data.

        Raises
        ------
        Exception
            If X is not a Zarr dataset or if no valid path is provided when saving.
        """
        # XXX: Avoid circular dependency
        from dasf.datasets.base import DatasetZarr

        if issubclass(X.__class__, DatasetZarr):
            if self.save:
                if self.filename:
                    url = self.filename
                elif hasattr(X, '_root_file'):
                    url = X._root_file
                else:
                    raise Exception("Array requires a valid path to convert to Array.")

                url = self._convert_filename(url)

                np.save(url, X._data)

            # This is just a place holder
            return X._data
        else:
            raise Exception("Input is not a Zarr dataset.")


class ArraysToDataFrame(Transform):
    """
    Transform multiple arrays into a DataFrame.

    This class converts multiple array-like objects into a single DataFrame
    by flattening and stacking the arrays as columns.
    """
    def _build_dataframe(self, data, columns, xp, df):
        """
        Build a DataFrame from multiple arrays.

        Parameters
        ----------
        data : list of array-like
            List of arrays to be combined into a DataFrame.
        columns : list of str
            Column names for the DataFrame.
        xp : module
            Array module (numpy or cupy).
        df : module
            DataFrame module (pandas or cudf).

        Returns
        -------
        DataFrame
            DataFrame with flattened arrays as columns.
        """
        data = [d.flatten() for d in data]
        stacked_data = xp.stack(data, axis=1)
        return df.DataFrame(
            stacked_data,
            columns=columns
        )

    def _lazy_transform(self, xp, df, **kwargs):
        """
        Lazy transform for converting arrays to DataFrame using Dask.

        Parameters
        ----------
        xp : module
            Array module (numpy or cupy).
        df : module
            DataFrame module (pandas or cudf).
        **kwargs
            Keyword arguments where keys are column names and values are arrays.

        Returns
        -------
        dask.dataframe.DataFrame
            Dask DataFrame with arrays as columns.
        """
        X = list(kwargs.values())
        y = list(kwargs.keys())
        assert len(X) == len(y), "Data and labels should have the same length."

        meta = ddf.utils.make_meta([
            (col, data.dtype)
            for col, data in zip(y, X)
        ],
            parent_meta=None if df == pd else cudf.DataFrame
        )

        lazy_dataframe_build = dask.delayed(self._build_dataframe)
        data_chunks = [x.to_delayed().ravel() for x in X]
        partial_dataframes = [
            ddf.from_delayed(lazy_dataframe_build(data=mapped_chunks,
                                                  columns=y,
                                                  xp=xp,
                                                  df=df), meta=meta)
            for mapped_chunks in zip(*data_chunks)
        ]

        return ddf.concat(partial_dataframes)

    def _lazy_transform_cpu(self, X=None, **kwargs):
        """
        CPU lazy transform for converting arrays to DataFrame.

        Parameters
        ----------
        X : array-like, optional
            Unused parameter for compatibility.
        **kwargs
            Keyword arguments where keys are column names and values are arrays.

        Returns
        -------
        dask.dataframe.DataFrame
            Dask DataFrame with arrays as columns.
        """
        return self._lazy_transform(np, pd, **kwargs)

    def _lazy_transform_gpu(self, X=None, **kwargs):
        """
        GPU lazy transform for converting arrays to DataFrame.

        Parameters
        ----------
        X : array-like, optional
            Unused parameter for compatibility.
        **kwargs
            Keyword arguments where keys are column names and values are arrays.

        Returns
        -------
        dask.dataframe.DataFrame
            Dask DataFrame with arrays as columns.
        """
        return self._lazy_transform(cp, cudf, **kwargs)

    def _transform(self, xp, df, **kwargs):
        """
        Transform arrays to DataFrame.

        Parameters
        ----------
        xp : module
            Array module (numpy or cupy).
        df : module
            DataFrame module (pandas or cudf).
        **kwargs
            Keyword arguments where keys are column names and values are arrays.

        Returns
        -------
        DataFrame
            DataFrame with arrays as columns.
        """
        X = list(kwargs.values())
        y = list(kwargs.keys())
        assert len(X) == len(y), "Data and labels should have the same length."

        return self._build_dataframe(data=X, columns=y, xp=xp, df=df)

    def _transform_cpu(self, X=None, **kwargs):
        """
        CPU transform for converting arrays to DataFrame.

        Parameters
        ----------
        X : array-like, optional
            Unused parameter for compatibility.
        **kwargs
            Keyword arguments where keys are column names and values are arrays.

        Returns
        -------
        pandas.DataFrame
            DataFrame with arrays as columns.
        """
        return self._transform(np, pd, **kwargs)

    def _transform_gpu(self, X=None, **kwargs):
        """
        GPU transform for converting arrays to DataFrame.

        Parameters
        ----------
        X : array-like, optional
            Unused parameter for compatibility.
        **kwargs
            Keyword arguments where keys are column names and values are arrays.

        Returns
        -------
        cudf.DataFrame
            DataFrame with arrays as columns.
        """
        return self._transform(cp, cudf, **kwargs)
