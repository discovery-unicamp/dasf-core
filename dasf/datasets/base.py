#!/usr/bin/env python3

import os
import json
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
except ImportError: # pragma: no cover
    pass

try:
    import numcodecs

    from kvikio.nvcomp_codec import NvCompBatchCodec
    from kvikio.zarr import GDSStore
except ImportError: # pragma: no cover
    pass

from pathlib import Path

from dasf.utils.funcs import human_readable_size
from dasf.utils.funcs import is_kvikio_supported
from dasf.utils.funcs import is_gds_supported
from dasf.utils.funcs import is_kvikio_compat_mode
from dasf.utils.funcs import is_nvcomp_codec_supported
from dasf.utils.decorators import task_handler
from dasf.utils.types import is_array
from dasf.utils.types import is_dask_array
from dasf.transforms.base import TargeteredTransform


class Dataset(TargeteredTransform):
    """Class representing a generic dataset based on a TargeteredTransform
    object.

    Parameters
    ----------
    name : str
        Symbolic name of the dataset.
    download : bool
        If the dataset must be downloaded (the default is False).
    root : str
        Root download directory (the default is None).
    *args : type
        Additional arguments without keys.
    **kwargs : type
        Additional keyworkded arguments.

    """
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
        self._metadata = {}
        self._data = None
        self._chunks = None

        self.__set_dataset_cache_dir()

        self.download()

    def __set_dataset_cache_dir(self):
        """Generate cached directory in $HOME to store dataset(s).

        """
        self._cache_dir = os.path.abspath(str(Path.home()) + "/.cache/dasf/datasets/")
        os.makedirs(self._cache_dir, exist_ok=True)

        if self._root is None:
            self._root = self._cache_dir

    def download(self):
        """Skeleton of the download method.

        """
        if self._download:
            raise NotImplementedError("Function download() needs to be defined")

    def __len__(self) -> int:
        """Return internal data length.

        """
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return len(self._data)

    def __getitem__(self, idx):
        """Generic __getitem__() function based on internal data.

        Parameters
        ----------
        idx : Any
            Key of the fetched data. It can be an integer or a tuple.

        """
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
        """Return a class representation based on internal array.

        """
        return repr(self._data)

    def __array__(self, dtype=None):
        assert self._data is not None, "Data is not loaded yet."
        return self._data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        assert self._data is not None, "Data is not loaded yet."
        if method == '__call__':
            scalars = []

            for inp in inputs:
                if isinstance(inp, Number):
                    scalars.append(inp)
                elif isinstance(inp, self.__class__):
                    scalars.append(inp._data)
                else:
                    return NotImplemented

            self.__class__(name=self._name, chunks=self._chunks)
            self._data = ufunc(*scalars, **kwargs)
            return self
        return NotImplemented

    def __check_op_input(self, in_data):
        """Return the proper type of data for operation

          >>> Result = DatasetArray + Numpy; or
          >>> Result = DatasetArray + DatasetArray

        Parameters
        ----------
        in_data : Any
            Input data to be analyzed.

        Returns
        -------
        data : Any
            A data representing the internal array or the class itself.

        """
        if is_array(in_data) or is_dask_array(in_data):
            return in_data
        if isinstance(in_data, self.__class__):
            return in_data._data
        raise TypeError("Data is incompatible with Array")

    def __add__(self, other):
        """Internal function of adding two array datasets.

        Parameters
        ----------
        other : Any
            A data representing an array or a DatasetArray.

        Returns
        -------
        DatasetArry
            A sum with two arrays.

        """
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data + data

    def __sub__(self, other):
        """Internal function of subtracting two array datasets.

        Parameters
        ----------
        other : Any
            A data representing an array or a DatasetArray.

        Returns
        -------
        DatasetArry
            A subtraction of two arrays.

        """
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data - data

    def __mul__(self, other):
        """Internal function of multiplication two array datasets.

        Parameters
        ----------
        other : Any
            A data representing an array or a DatasetArray.

        Returns
        -------
        DatasetArry
            A multiplication of two arrays.

        """
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data * data

    def __div__(self, other):
        """Internal function of division two array datasets.

        Parameters
        ----------
        other : Any
            A data representing an array or a DatasetArray.

        Returns
        -------
        DatasetArry
            A division of two arrays.

        """
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data / data

    def __copy_attrs_from_data(self):
        """Extends metadata to new transformed object (after operations).

        """
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
            Additional keyworkded arguments to the load.

        Returns
        -------
        Any
            The data (or a Future load object, for `_lazy` operations).

        """
        npy_shape = self.shape

        local_data = dask.delayed(xp.load)(self._root_file, **kwargs)

        local_data = da.from_delayed(local_data, shape=npy_shape, dtype=xp.float32, meta=xp.array(()))
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
        """Placeholder for load function.

        """
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

    def inspect_metadata(self) -> dict:
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
    """Class representing an dataset wich is defined as a Zarr array of a
    defined shape.

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
                 backend: str = None,
                 chunks=None):

        Dataset.__init__(self, name, download, root)

        self._backend = backend
        self._chunks = chunks

        self._root_file = root

        if root is not None:
            if not os.path.isfile(root):
                self._root = root
            else:
                self._root = os.path.dirname(root)

    def _lazy_load(self, xp, **kwargs):
        """Lazy load the dataset using an CPU dask container.

        Parameters
        ----------
        xp : type
            Library used to load the file. It must follow numpy library.
        **kwargs : type
            Additional keyworkded arguments to the load.

        Returns
        -------
        Any
            The data (or a Future load object, for `_lazy` operations).

        """
        if (self._backend == "kvikio" and is_kvikio_supported() and
            (is_gds_supported() or is_kvikio_compat_mode())
            and is_nvcomp_codec_supported()):
            store = GDSStore(self._root_file)
            meta = json.loads(store[".zarray"])
            meta["compressor"] = NvCompBatchCodec("lz4").get_config()
            store[".zarray"] = json.dumps(meta).encode()

            array = zarr.open_array(store, meta_array=xp.empty(()))
            return da.from_zarr(array, chunks=array.chunks).map_blocks(xp.asarray)

        return da.from_zarr(self._root_file, chunks=self._chunks).map_blocks(xp.asarray)

    def _load(self, xp, **kwargs):
        """Load data using CPU container.

        Parameters
        ----------
        xp : Module
            A module that load data (implement `load` function)
        **kwargs : type
            Additional `kwargs` to `xp.load` function.

        """
        return zarr.open(self._root_file, mode='r', meta_array=xp.empty(()))

    def _lazy_load_cpu(self):
        """Load data with CPU container + DASK. (It does not load immediattly)

        """
        self._metadata = self._load_meta()
        self._data = self._lazy_load(np)
        self.__copy_attrs_from_data()
        return self

    def _lazy_load_gpu(self):
        """Load data with GPU container + DASK. (It does not load immediattly)

        """
        self._metadata = self._load_meta()
        self._data = self._lazy_load(cp)
        self.__copy_attrs_from_data()
        return self

    def _load_cpu(self):
        """Load data with CPU container (e.g. numpy).

        """
        self._metadata = self._load_meta()
        self._data = self._load(np)
        self.__copy_attrs_from_data()
        return self

    def _load_gpu(self):
        """Load data with GPU container (e.g. cupy).

        """
        self._metadata = self._load_meta()
        self._data = self._load(cp)
        self.__copy_attrs_from_data()
        return self

    @task_handler
    def load(self):
        """Placeholder for load function.

        """
        ...

    def _load_meta(self) -> dict:
        """Load metadata to inspect.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        assert self._root_file is not None, "There is no temporary file to inspect"

        return self.inspect_metadata()

    def __read_zarray(self, key):
        """Returns the value of ZArray JSON metadata.

        """
        if self._root_file and os.path.isdir(self._root_file):
            zarray = os.path.abspath(self._root_file + "/.zarray")
            if os.path.exists(zarray):
                try:
                    with open(zarray) as fz:
                        meta = json.load(fz)
                        return meta[key]
                except Exception:
                    pass
        return None

    @property
    def shape(self) -> tuple:
        """Returns the shape of an array.

        Returns
        -------
        tuple
            A tuple with the shape.

        """
        if not self._data:
            shape = self.__read_zarray("shape")
            if shape is not None:
                return tuple(shape)
            return tuple()

        return self._data.shape

    @property
    def chunksize(self):
        """Returns the chunksize of an array.

        Returns
        -------
        tuple
            A tuple with the chunksize.

        """
        if not self._data:
            chunks = self.__read_zarray("chunks")
            if chunks is not None:
                return tuple(chunks)
            return tuple()

        return self._data.chunksize

    def inspect_metadata(self) -> dict:
        """Return a dictionary with all metadata information from data.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        z = zarr.open(self._root_file, mode='r')

        info = {}
        for k, v in z.info_items():
            info[k] = v

        if isinstance(self._chunks, bool) and self._chunks:
            self._chunks = info["Chunk shape"]

        if self._chunks is None:
            self._chunks = self.chunksize

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

    def __repr__(self):
        """Return a class representation based on internal array.

        """
        return repr(self._data)

    def __check_op_input(self, in_data):
        """Return the proper type of data for operation

          >>> Result = DatasetZarr + Numpy; or
          >>> Result = DatasetZarr + DatasetZarr

        Parameters
        ----------
        in_data : Any
            Input data to be analyzed.

        Returns
        -------
        data : Any
            A data representing the internal array or the class itself.

        """
        if is_array(in_data) or is_dask_array(in_data):
            return in_data
        elif isinstance(in_data, self.__class__):
            return in_data._data
        raise TypeError("Data is incompatible with Array")

    def __add__(self, other):
        """Internal function of adding two array datasets.

        Parameters
        ----------
        other : Any
            A data representing an array or a DatasetArray.

        Returns
        -------
        DatasetArry
            A sum with two arrays.

        """
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data + data

    def __sub__(self, other):
        """Internal function of subtracting two array datasets.

        Parameters
        ----------
        other : Any
            A data representing an array or a DatasetArray.

        Returns
        -------
        DatasetArry
            A subtraction of two arrays.

        """
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data - data

    def __mul__(self, other):
        """Internal function of multiplication two array datasets.

        Parameters
        ----------
        other : Any
            A data representing an array or a DatasetArray.

        Returns
        -------
        DatasetArry
            A multiplication of two arrays.

        """
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data * data

    def __div__(self, other):
        """Internal function of division two array datasets.

        Parameters
        ----------
        other : Any
            A data representing an array or a DatasetArray.

        Returns
        -------
        DatasetArry
            A division of two arrays.

        """
        assert self._data is not None, "Data is not loaded yet."
        data = self.__check_op_input(other)
        return self._data / data

    def __copy_attrs_from_data(self):
        """Extends metadata to new transformed object (after operations).

        """
        self._metadata["type"] = type(self._data)

        attrs = dir(self._data)
        for attr in attrs:
            if not attr.startswith("__") and callable(getattr(self._data, attr)):
                if not hasattr(self, attr):
                    self.__dict__[attr] = getattr(self._data, attr)


class DatasetHDF5(Dataset):
    """Class representing an dataset wich is defined as a HDF5 dataset of a
    defined shape.

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
    dataset_path : str
        Relative path of the internal HDF5 dataset (the default is None).

    """
    def __init__(self,
                 name: str,
                 download: str = False,
                 root: str = None,
                 chunks="auto",
                 dataset_path: str = None):

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

    def _lazy_load(self, xp, **kwargs):
        """Lazy load the dataset using an CPU dask container.

        Parameters
        ----------
        xp : type
            Library used to load the file. It must follow numpy library.
        **kwargs : type
            Additional keyworkded arguments to the load.

        Returns
        -------
        Any
            The data (or a Future load object, for `_lazy` operations).

        """
        f = h5py.File(self._root_file)
        data = f[self._dataset_path]
        return da.from_array(data, chunks=self._chunks, meta=xp.array(()))

    def _load(self, xp=None, **kwargs):
        """Load data using CPU container.

        Parameters
        ----------
        xp : Module
            A module that load data (implement `load` function) (placeholder).
        **kwargs : type
            Additional `kwargs` to `xp.load` function.

        """
        f = h5py.File(self._root_file)
        return f[self._dataset_path]

    def _lazy_load_cpu(self):
        """Load data with CPU container + DASK. (It does not load immediattly)

        """
        self._metadata = self._load_meta()
        self._data = self._lazy_load(np)
        return self

    def _lazy_load_gpu(self):
        """Load data with GPU container + DASK. (It does not load immediattly)

        """
        self._metadata = self._load_meta()
        self._data = self._lazy_load(cp)
        return self

    def _load_cpu(self):
        """Load data with CPU container (e.g. numpy).

        """
        self._metadata = self._load_meta()
        self._data = self._load()
        return self

    def _load_gpu(self):
        """Load data with GPU container (e.g. cupy).

        """
        self._metadata = self._load_meta()
        self._data = cp.asarray(self._load())
        return self

    @task_handler
    def load(self):
        """Placeholder for load function.

        """
        ...

    def _load_meta(self) -> dict:
        """Load metadata to inspect.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        assert self._root_file is not None, "There is no temporary file to inspect"
        assert self._dataset_path is not None, "There is no path to fetch data"

        return self.inspect_metadata()

    def inspect_metadata(self) -> dict:
        """Return a dictionary with all metadata information from data.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
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
    """Class representing an dataset wich is defined as a Xarray dataset of a
    defined shape.

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
    data_var : Any
        Key (or index) of the internal Xarray dataset (the default is None).

    """
    def __init__(self,
                 name: str,
                 download: bool = False,
                 root: str = None,
                 chunks=None,
                 data_var=None):
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
        """Load data with CPU container + DASK. (It does not load immediattly)

        """
        assert self._chunks is not None, "Lazy operations require chunks"

        if self._data_var:
            self._data = xr.open_dataset(self._root_file,
                                         chunks=self._chunks)
        else:
            self._data = xr.open_dataarray(self._root_file,
                                           chunks=self._chunks)
        self._metadata = self._load_meta()

    def _lazy_load_gpu(self):
        """Load data with GPU container + DASK. (It does not load immediattly)

        """
        assert self._chunks is not None, "Lazy operations require chunks"

        if self._data_var:
            self._data = xr.open_dataset(self._root_file,
                                         chunks=self._chunks).as_cupy()
        else:
            self._data = xr.open_dataarray(self._root_file,
                                           chunks=self._chunks).as_cupy()
        self._metadata = self._load_meta()

    def _load_cpu(self):
        """Load data with CPU container (e.g. numpy).

        """
        if self._data_var:
            self._data = xr.open_dataset(self._root_file)
        else:
            self._data = xr.open_dataarray(self._root_file)
        self._data.load()
        self._metadata = self._load_meta()

    def _load_gpu(self):
        """Load data with GPU container (e.g. cupy).

        """
        if self._data_var:
            self._data = xr.open_dataset(self._root_file).as_cupy()
        else:
            self._data = xr.open_dataarray(self._root_file).as_cupy()
        self._data.load()
        self._metadata = self._load_meta()

    @task_handler
    def load(self):
        """Placeholder for load function.

        """
        ...

    def _load_meta(self) -> dict:
        """Load metadata to inspect.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        assert self._root_file is not None, "There is no temporary file to inspect"

        return self.inspect_metadata()

    def inspect_metadata(self) -> dict:
        """Return a dictionary with all metadata information from data.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
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

    def __len__(self) -> int:
        """Return internal data length.

        """
        if self._data is None:
            raise Exception("Data is not loaded yet")

        if self._data_var:
            return len(self._data[self._data_var])

        return len(self._data)

    def __getitem__(self, idx):
        """A __getitem__() function based on internal Xarray data.

        Parameters
        ----------
        idx : Any
            Key of the fetched data. It can be an integer or a tuple.

        """
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

    def _lazy_load(self, xp, **kwargs) -> tuple:
        """Lazy load the dataset using an CPU dask container.

        Parameters
        ----------
        xp : type
            Library used to load the file. It must follow numpy library.
        **kwargs : type
            Additional keyworkded arguments to the load.

        Returns
        -------
        Tuple
            A Future object that will return a tuple: (data, label).

        """
        local_data = self._train._lazy_load(xp)
        local_labels = self._val._lazy_load(xp)

        return (local_data, local_labels)

    def _load(self, xp, **kwargs) -> tuple:
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

    def _load_meta(self) -> dict:
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
        """Placeholder for load function.

        """
        ...

    def __getitem__(self, idx):
        """A __getitem__() function for data and labeled data together.

        Parameters
        ----------
        idx : Any
            Key of the fetched data. It can be an integer or a tuple.

        """
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return (self._data.__getitem__(idx), self._labels.__getitem__(idx))


class DatasetDataFrame(Dataset):
    """Class representing an dataset wich is defined as a dataframe.

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
                 download: bool = True,
                 root: str = None,
                 chunks="auto"):

        Dataset.__init__(self, name, download, root)

        self._chunks = chunks

        self._root_file = root

        if root is not None:
            if not os.path.isfile(root):
                raise Exception("DataFrame requires a root=filename.")

            self._root = os.path.dirname(root)

    def _load_meta(self) -> dict:
        """Load metadata to inspect.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        assert self._root_file is not None, (
            "There is no temporary file to inspect"
        )

        return self.inspect_metadata()

    def inspect_metadata(self) -> dict:
        """Return a dictionary with all metadata information from data.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
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
        """Load data with GPU container + DASK. (It does not load immediattly)

        """
        self._data = dcudf.read_csv(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _lazy_load_cpu(self):
        """Load data with CPU container + DASK. (It does not load immediattly)

        """
        self._data = ddf.read_csv(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _load_gpu(self):
        """Load data with GPU container (e.g. CuDF).

        """
        self._data = cudf.read_csv(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _load_cpu(self):
        """Load data with CPU container (e.g. pandas).

        """
        self._data = pd.read_csv(self._root_file)
        self._metadata = self._load_meta()
        return self

    @task_handler
    def load(self):
        """Placeholder for load function.

        """
        ...

    @property
    def shape(self) -> tuple:
        """Returns the shape of an array.

        Returns
        -------
        tuple
            A tuple with the shape.

        """
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return self._data.shape

    def __len__(self) -> int:
        """Return internal data length.
        """
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return len(self._data)

    def __getitem__(self, idx):
        """A __getitem__() function based on internal dataframe.

        Parameters
        ----------
        idx : Any
            Key of the fetched data. It can be an integer or a tuple.

        """
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return self._data.iloc[idx]


class DatasetParquet(DatasetDataFrame):
    """Class representing an dataset wich is defined as a Parquet.

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
                 download: bool = True,
                 root: str = None,
                 chunks="auto"):

        DatasetDataFrame.__init__(self, name, download, root, chunks)

    def _lazy_load_gpu(self):
        """Load data with GPU container + DASK. (It does not load immediattly)

        """
        self._data = dcudf.read_parquet(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _lazy_load_cpu(self):
        """Load data with CPU container + DASK. (It does not load immediattly)

        """
        self._data = ddf.read_parquet(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _load_gpu(self):
        """Load data with GPU container (e.g. CuDF).

        """
        self._data = cudf.read_parquet(self._root_file)
        self._metadata = self._load_meta()
        return self

    def _load_cpu(self):
        """Load data with CPU container (e.g. pandas).

        """
        self._data = pd.read_parquet(self._root_file)
        self._metadata = self._load_meta()
        return self
