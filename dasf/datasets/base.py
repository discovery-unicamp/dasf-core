#!/usr/bin/env python3

import os
import zarr
import h5py
import dask

import numpy as np
import numpy.lib.format
import pandas as pd
import dask.array as da
import dask.dataframe as ddf
import xarray as xr

from numbers import Number

try:
    import cupy as cp
    import cudf
    import dask_cudf as dcudf
    # This is just to enable Xarray Cupy capabilities
    import cupy_xarray as cx   # noqa
except ImportError:
    pass

from pathlib import Path

from dasf.utils.funcs import human_readable_size
from dasf.utils.decorators import task_handler
from dasf.utils.types import is_array
from dasf.utils.types import is_dask_array
from dasf.transforms.base import TargeteredTransform


class Dataset(TargeteredTransform):
    def __init__(self, 
                 name: str,
                 download: bool = False,
                 root: str = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Dataset internals
        self._name = name
        self._download = download
        self._root = root
        self._metadata = dict()
        self._data = None
        self._chunks = None

        self.__set_dataset_cache_dir()

        self.download()

    def __set_dataset_cache_dir(self):
        self._cache_dir = os.path.abspath(str(Path.home()) + "/.cache/dasf/datasets/")
        os.makedirs(self._cache_dir, exist_ok=True)

        if self._root is None:
            self._root = self._cache_dir

    def download(self):
        if self._download:
            raise NotImplementedError("Function download() needs to be defined")

    def __len__(self) -> int:
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return len(self._data)

    def __getitem__(self, idx):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return self._data.__getitem__(idx)


class DatasetArray(Dataset):
    """Class representing an dataset wich is defined as an array of a defined
    shape.

    Parameters
    ----------
    name : str
        Symbolic name of the dataset.
    download : bool
        If the dataset must be downloaded (the default is False).
    root : str
        Root download directory (the default is None).
    chunks : Any
        Number of blocks of the array (the default is "auto").

    """
    def __init__(self,
                 name: str,
                 download: bool = False,
                 root: str = None,
                 chunks="auto"):

        Dataset.__init__(self, name, download, root)

        self._chunks = chunks

        self._root_file = root

        if root is not None:
            if not os.path.isfile(root):
                raise Exception("Array requires a root=filename.")

            self._root = os.path.dirname(root)

    def __operator_check__(self, other):
        assert self._data is not None, "Data is not loaded yet."
        if isinstance(other, DatasetArray):
            return other._data
        return other

    def __repr__(self):
        return repr(self._data)

    def __array__(self, dtype=None):
        assert self._data is not None, "Data is not loaded yet."
        return self._data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        assert self._data is not None, "Data is not loaded yet."
        if method == '__call__':
            scalars = []

            for input in inputs:
                if isinstance(input, Number):
                    scalars.append(input)
                elif isinstance(input, self.__class__):
                    scalars.append(input._data)
                else:
                    return NotImplemented

            self.__class__(name=self._name, chunks=self._chunks)
            self._data = ufunc(*scalars, **kwargs)
            return self
        else:
            return NotImplemented

    def __check_op_input(self, in_data):
        if is_array(in_data) or is_dask_array(in_data):
            return in_data
        elif isinstance(in_data, self.__class__):
            return in_data._data
        raise TypeError("Data is incompatible with Array")

    def __add__(self, other):
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data + data

    def __sub__(self, other):
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data - data

    def __mul__(self, other):
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data * data

    def __div__(self, other):
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data / data

    def __copy_attrs_from_data(self):
        self._metadata["type"] = type(self._data)

        attrs = dir(self._data)
        for attr in attrs:
            if not attr.startswith("__") and callable(getattr(self._data, attr)):
                if not hasattr(self, attr):
                    self.__dict__[attr] = getattr(self._data, attr)

    def __npy_header(self):
        """Read an array header from a filelike object.

        """
        with open(self._root_file, 'rb') as fobj:
            version = numpy.lib.format.read_magic(fobj)
            func_name = "read_array_header_" + "_".join(str(v) for v in version)
            func = getattr(numpy.lib.format, func_name)
            return func(fobj)

    def _lazy_load(self, xp, **kwargs):
        """Lazy load the dataset using an CPU dask container.

        Parameters
        ----------
        xp : type
            Library used to load the file. It must follow numpy library.
        **kwargs : type
            Additional keyworkded argumnts to the load.

        Returns
        -------
        Any
            The data (or a Future load object, for `_lazy` operations).

        """
        npy_shape = self.shape

        local_data = dask.delayed(xp.load)(self._root_file, **kwargs)

        local_data = da.from_delayed(local_data, shape=npy_shape, dtype=xp.float32)
        if isinstance(self._chunks, tuple):
            local_data = local_data.rechunk(self._chunks)

        return local_data

    def _load(self, xp, **kwargs):
        """Load data using CPU container.

        Parameters
        ----------
        xp : Module
            A module that load data (implement `load` function)
        **kwargs : type
            Additional `kwargs` to `xp.load` function.

        """

        return xp.load(self._root_file, **kwargs)

    def _load_meta(self) -> dict:
        """Load metadata to inspect.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        assert self._root_file is not None, ("There is no temporary file to "
                                             "inspect")
        assert os.path.isfile(self._root_file), ("The root variable should "
                                                 "be a NPY file")

        return self.inspect_metadata()

    def _lazy_load_gpu(self):
        """Load data with GPU container + DASK. (It does not load immediattly)

        """
        self._metadata = self._load_meta()
        self._data = self._lazy_load(cp)
        self.__copy_attrs_from_data()
        return self

    def _lazy_load_cpu(self):
        """Load data with CPU container + DASK. (It does not load immediattly)

        """
        self._metadata = self._load_meta()
        self._data = self._lazy_load(np)
        self.__copy_attrs_from_data()
        return self

    def _load_gpu(self):
        """Load data with GPU container (e.g. cupy).

        """
        self._metadata = self._load_meta()
        self._data = self._load(cp)
        self.__copy_attrs_from_data()
        return self

    def _load_cpu(self):
        """Load data with CPU container (e.g. numpy).

        """
        self._metadata = self._load_meta()
        self._data = self._load(np)
        self.__copy_attrs_from_data()
        return self

    @task_handler
    def load(self):
        ...

    @property
    def shape(self) -> tuple:
        """Returns the shape of an array.

        Returns
        -------
        tuple
            A tuple with the shape.

        """
        return self.__npy_header()[0]

    def inspect_metadata(self):
        """Return a dictionary with all metadata information from data.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        array_file_size = human_readable_size(
            os.path.getsize(self._root_file),
            decimal=2
        )

        npy_shape = self.shape

        return {
            "size": array_file_size,
            "file": self._root_file,
            "shape": npy_shape,
            "block": {"chunks": self._chunks},
        }


class DatasetZarr(Dataset):
    def __init__(self, name, download=False, root=None, chunks=None):

        Dataset.__init__(self, name, download, root)

        self._chunks = chunks

        self._root_file = root

        if root is not None:
            if not os.path.isfile(root):
                self._root = root
            else:
                self._root = os.path.dirname(root)

    def _lazy_load(self, xp):
        return da.from_zarr(self._root_file, chunks=self._chunks).map_blocks(xp.asarray)

    def _load(self):
        return zarr.open(self._root_file, mode='r')

    def _lazy_load_cpu(self):
        self._metadata = self._load_meta()
        self._data = self._lazy_load(np)
        return self

    def _lazy_load_gpu(self):
        self._metadata = self._load_meta()
        self._data = self._lazy_load(cp)
        return self

    def _load_cpu(self):
        self._metadata = self._load_meta()
        self._data = self._load()
        return self

    def _load_gpu(self):
        self._metadata = self._load_meta()
        self._data = cp.asarray(self._load())
        return self

    @task_handler
    def load(self):
        ...

    def _load_meta(self):
        assert self._root_file is not None, "There is no temporary file to inspect"

        return self.inspect_metadata()

    def inspect_metadata(self):
        z = zarr.open(self._root_file, mode='r')

        info = dict()
        for k, v in z.info_items():
            info[k] = v

        if isinstance(self._chunks, bool) and self._chunks:
            self._chunks = info["Chunk shape"]

        return {
            "size": human_readable_size(
                int(info["No. bytes"].split(' ')[0])
            ),
            "compressor": info["Compressor"],
            "type": info["Store type"],
            "file": self._root_file,
            "shape": info["Shape"],
            "block": {"chunks": self._chunks},
        }


class DatasetHDF5(Dataset):
    def __init__(self, name, download=False, root=None, chunks="auto", dataset_path=None):

        Dataset.__init__(self, name, download, root)

        self._chunks = chunks

        self._root_file = root

        self._dataset_path = dataset_path

        if root is not None:
            if not os.path.isfile(root):
                raise Exception("HDF5 requires a root=filename.")

            self._root = os.path.dirname(root)

        if dataset_path is None:
            raise Exception("HDF5 requires a path.")

    def _lazy_load(self, xp):
        f = h5py.File(self._root_file)
        data = f[self._dataset_path]
        return da.from_array(data, chunks=self._chunks, meta=xp.array(()))

    def _load(self):
        f = h5py.File(self._root_file)
        return f[self._dataset_path]

    def _lazy_load_cpu(self):
        self._metadata = self._load_meta()
        self._data = self._lazy_load(np)
        return self

    def _lazy_load_gpu(self):
        self._metadata = self._load_meta()
        self._data = self._lazy_load(cp)
        return self

    def _load_cpu(self):
        self._metadata = self._load_meta()
        self._data = self._load()
        return self

    def _load_gpu(self):
        self._metadata = self._load_meta()
        self._data = cp.asarray(self._load())
        return self

    @task_handler
    def load(self):
        ...

    def _load_meta(self):
        assert self._root_file is not None, "There is no temporary file to inspect"
        assert self._dataset_path is not None, "There is no path to fetch data"

        return self.inspect_metadata()

    def inspect_metadata(self):
        f = h5py.File(self._root_file)
        data = f[self._dataset_path]

        array_file_size = human_readable_size(
            data.size, decimal=2
        )

        return {
            "size": array_file_size,
            "file": self._root_file,
            "shape": data.shape,
            "block": {"chunks": self._chunks},
        }


class DatasetXarray(Dataset):
    def __init__(self, name, download=False, root=None, chunks=None, data_var=None):
        Dataset.__init__(self, name, download, root)

        self._chunks = chunks

        self._root_file = root

        self._data_var = data_var

        if chunks and not isinstance(chunks, dict):
            raise Exception("Chunks should be a dict.")

        if root is not None:
            if not os.path.isfile(root):
                raise Exception("HDF5 requires a root=filename.")

            self._root = os.path.dirname(root)

    def _lazy_load_cpu(self):
        assert self._chunks is not None, "Lazy operations require chunks"

        if self._data_var:
            self._data = xr.open_dataset(self._root_file,
                                         chunks=self._chunks)
        else:
            self._data = xr.open_dataarray(self._root_file,
                                           chunks=self._chunks)
        self._metadata = self._load_meta()

    def _lazy_load_gpu(self):
        assert self._chunks is not None, "Lazy operations require chunks"

        if self._data_var:
            self._data = xr.open_dataset(self._root_file,
                                         chunks=self._chunks).as_cupy()
        else:
            self._data = xr.open_dataarray(self._root_file,
                                           chunks=self._chunks).as_cupy()
        self._metadata = self._load_meta()

    def _load_cpu(self):
        if self._data_var:
            self._data = xr.open_dataset(self._root_file)
        else:
            self._data = xr.open_dataarray(self._root_file)
        self._data.load()
        self._metadata = self._load_meta()

    def _load_gpu(self):
        if self._data_var:
            self._data = xr.open_dataset(self._root_file).as_cupy()
        else:
            self._data = xr.open_dataarray(self._root_file).as_cupy()
        self._data.load()
        self._metadata = self._load_meta()

    @task_handler
    def load(self):
        ...

    def _load_meta(self):
        assert self._root_file is not None, "There is no temporary file to inspect"

        return self.inspect_metadata()

    def inspect_metadata(self):
        array_file_size = human_readable_size(
            os.path.getsize(self._root_file), decimal=2
        )

        return {
            "size": array_file_size,
            "file": self._root_file,
            "coords": tuple(self._data.coords),
            "attrs": self._data.attrs,
            "block": {"chunks": self._chunks},
        }

    def __len__(self):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        if self._data_var:
            return len(self._data[self._data_var])

        return len(self._data)

    def __getitem__(self, idx):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        # Always slice a DataArray
        if self._data_var:
            return self._data[self._data_var].data[idx]

        return self._data.data[idx]


class DatasetLabeled(Dataset):
    """A class representing a labeled dataset. Each item is a 2-element tuple,
    where the first element is a array of data and the second element is the
    respective label. The items can be accessed from `dataset[x]`.

    Parameters
    ----------
    name : str
        Symbolic name of the dataset.
    download : bool
        If the dataset must be downloaded (the default is False).
    root : str
        Root download directory (the default is None).
    chunks : Any
        Number of blocks of the array (the default is "auto").

    Attributes
    ----------
    __chunks : type
        Description of attribute `__chunks`.

    """
    def __init__(self,
                 name: str,
                 download: bool = False,
                 root: str = None,
                 chunks="auto"):

        Dataset.__init__(self, name, download, root)

        self._chunks = chunks

    def download(self):
        """Download the dataset.

        """
        if hasattr(self, "_train") and hasattr(self._train, "download"):
            self._train.download()

        if hasattr(self, "_val") and hasattr(self._val, "download"):
            self._val.download()

    def inspect_metadata(self) -> dict:
        """Return a dictionary with all metadata information from data
        (train and labels).

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        metadata_train = self._train.inspect_metadata()
        metadata_val = self._val.inspect_metadata()

        assert (
            metadata_train["shape"] == metadata_val["shape"]
        ), "Train and Labels should have same shape: " + str(
            metadata_train["shape"]
        ) + " != " + str(
            metadata_val["shape"]
        )

        return {"train": metadata_train, "labels": metadata_val}

    def _lazy_load(self, xp, **kwargs):
        """Lazy load the dataset using an CPU dask container.

        Parameters
        ----------
        xp : type
            Library used to load the file. It must follow numpy library.
        **kwargs : type
            Additional keyworkded argumnts to the load.

        Returns
        -------
        Tuple
            A Future object that will return a tuple: (data, label).

        """
        local_data = self._train._lazy_load(xp)
        local_labels = self._val._lazy_load(xp)

        return (local_data, local_labels)

    def _load(self, xp, **kwargs):
        """Load data using CPU container.

        Parameters
        ----------
        xp : Module
            A module that load data (implement `load` function)
        **kwargs : type
            Additional `kwargs` to `xp.load` function.

        Returns
        -------
        Tuple
            A 2-element tuple: (data, label)

        """
        local_data = self._train._load(xp)
        local_labels = self._val._load(xp)

        return (local_data, local_labels)

    def _load_meta(self):
        """Load metadata to inspect.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        assert self._train._root_file is not None, (
            "There is no temporary file to inspect"
        )
        assert self._val._root_file is not None, (
            "There is no temporary file to inspect"
        )
        assert os.path.isfile(self._train._root_file), (
            "The root variable should be a file"
        )
        assert os.path.isfile(self._val._root_file), (
            "The root variable should be a file"
        )

        return self.inspect_metadata()

    def _lazy_load_gpu(self):
        """Load data with GPU container + DASK. (It does not load immediattly)

        """
        self._metadata = self._load_meta()
        self._data, self._labels = self._lazy_load(cp)

    def _lazy_load_cpu(self):
        """Load data with CPU container + DASK. (It does not load immediattly)

        """
        self._metadata = self._load_meta()
        self._data, self._labels = self._lazy_load(np)

    def _load_gpu(self):
        """Load data with GPU container (e.g. cupy).

        """
        self._metadata = self._load_meta()
        self._data, self._labels = self._load(cp)

    def _load_cpu(self):
        """Load data with CPU container (e.g. numpy).

        """
        self._metadata = self._load_meta()
        self._data, self._labels = self._load(np)

    @task_handler
    def load(self):
        ...

    def __getitem__(self, idx):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return (self._data.__getitem__(idx), self._labels.__getitem__(idx))


class DatasetDataFrame(Dataset):
    def __init__(self, name, download=True, root=None, chunks="auto"):

        Dataset.__init__(self, name, download, root)

        self._chunks = chunks

        self._root_file = root

        if root is not None:
            if not os.path.isfile(root):
                raise Exception("DataFrame requires a root=filename.")

            self._root = os.path.dirname(root)

    def _load_meta(self):
        assert self._root_file is not None, (
            "There is no temporary file to inspect"
        )

        return self.inspect_metadata()

    def inspect_metadata(self):
        df_file_size = human_readable_size(
            os.stat(self._root_file).st_size, decimal=2
        )

        return {
            "size": df_file_size,
            "file": self._root_file,
            "type": type(self._data),
            "shape": self.shape,
            "columns": list(self._data.columns),
            "block": {"chunks": self._chunks},
        }

    def _lazy_load_gpu(self):
        self._data = dcudf.read_csv(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _lazy_load_cpu(self):
        self._data = ddf.read_csv(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _load_gpu(self):
        self._data = cudf.read_csv(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _load_cpu(self):
        self._data = pd.read_csv(self._root_file)
        self._metadata = self._load_meta()
        return self

    @task_handler
    def load(self):
        ...

    @property
    def shape(self):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return self._data.shape

    def __len__(self):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return len(self._data)

    def __getitem__(self, idx):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return self._data.iloc[idx]


class DatasetParquet(DatasetDataFrame):
    def __init__(self, name, download=True, root=None, chunks="auto"):

        DatasetDataFrame.__init__(self, name, download, root, chunks)

    def _lazy_load_gpu(self):
        self._data = dcudf.read_parquet(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _lazy_load_cpu(self):
        self._data = ddf.read_parquet(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _load_gpu(self):
        self._data = cudf.read_parquet(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _load_cpu(self):
        self._data = pd.read_parquet(self._root_file)
        self._metadata = self._load_meta()
        return self
