#!/usr/bin/env python3

import os
import dask

import numpy as np
import numpy.lib.format
import dask.array as da

try:
    import cupy as cp
except ImportError:
    pass

from enum import Enum
from pathlib import Path

from dasf.utils import utils
from dasf.utils.generators import generate_load
from dasf.pipeline import ParameterOperator


class DatasetType(Enum):
    none = "none"
    cmp_gathers = "CMP Gathers"
    surface_seismic = "Surface Seismic"
    borehole_seismic = "Borehole Seismic"
    fourd_far_stack = "4D Far Stack"
    fourd_near_stack = "4D Near Stack"
    fourd_mid_stack = "4D Mid Stack"
    fourd_full_stack = "4D Full Stack"
    far_stack = "Far Stack"
    near_stack = "Near Stack"
    mid_stack = "Mid Stack"
    full_stack = "Full Stack"
    prestack_seismic = "Prestack Seismic"
    poststack_seismic = "Poststack Seismic"
    migrated_volume = "Migrated Volume"

    def __str__(self):
        return self.value


class Dataset(object):
    """An abstract class representing a generic map-style dataset which
    implement the getitem and len protocols. All datasets that represent a map
    from keys to data samples should subclass it. Datasets subclassed from this
    class can be acessed using the subscription syntax, e.g.: `dataset[index]`.

    Parameters
    ----------
    name : str
        Symbolic name of the dataset.
    subtype : DatasetType
        The type of the seismic data.
    download : bool
        If the dataset must be downloaded (the default is False).
    root : str
        Root download directory (the default is None).

    """

    def __init__(self,
                 name: str,
                 subtype=DatasetType.none,
                 download: bool = False,
                 root: str = None):

        # Dataset internals
        self._name = name
        self._subtype = subtype
        self._download = download
        self._root = root
        self._metadata = dict()
        self._data = None

        # Internal chunks division (for Dask Types)
        self.__chunks = None

        assert type(self._subtype) == DatasetType

        self.__set_dataset_cache_dir()

        self.download()

    def __set_dataset_cache_dir(self):
        """Set the cache directory. The dataset will first be read from the
        cache directory, if the file exists.

        """
        self._cache_dir = os.path.abspath(str(Path.home()) +
                                          "/.cache/dasf/datasets/")
        os.makedirs(self._cache_dir, exist_ok=True)

        if self._root is None:
            self._root = self._cache_dir

    def download(self):
        """Download the dataset.

        """
        if self._download:
            raise NotImplementedError("Function download() needs to be "
                                      "defined")

    def __len__(self):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return len(self._data)

    def __getitem__(self, idx):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return self._data.__getitem__(idx)


class DatasetLoader(ParameterOperator):
    """Class to load dataset data and generate an array. The data is loaded
    depending on the implementation.

    - `cpu` allows loading `numpy.ndarray`.
    - `gpu` allows loading `cupy.ndarray`.
    - `lazy_cpu` allows loading `dask.array` data (as numpy).
    - `lazy_gpu` allows loading `dask.array` data (as cupy).

    Each implementation defines how load is executed.

    Parameters
    ----------
    dataset : Dataset
        The dataset object to load.
    replicate : bool
        Not used (the default is False).

    """

    def __init__(self, dataset: Dataset, replicate: bool = False):
        ParameterOperator.__init__(self, name=dataset._name)

        self.__dataset = dataset

    def run_lazy_cpu(self):
        """Load data using a DASK. Usually, this operation will append a future
        load to a DASK graph using an DASK CPU container.

        """
        self.__dataset.download()

        if hasattr(self.__dataset, 'lazy_load_cpu'):
            self.__dataset.lazy_load_cpu()
            return self.__dataset

    def run_cpu(self):
        """Load data using an CPU container (e.g. numpy).

        """
        self.__dataset.download()

        if hasattr(self.__dataset, 'load_cpu'):
            self.__dataset.load_cpu()
            return self.__dataset

    def run_lazy_gpu(self):
        """Load data using a DASK. Usually, this operation will append a future
        load to a DASK graph using an DASK GPU container.

        """
        self.__dataset.download()

        if hasattr(self.__dataset, 'lazy_load_gpu'):
            self.__dataset.lazy_load_gpu()
            return self.__dataset

    def run_gpu(self):
        """Load data using an GPU container (e.g. cupy).

        """
        self.__dataset.download()

        if hasattr(self.__dataset, 'load_gpu'):
            self.__dataset.load_gpu()
            return self.__dataset


@generate_load
class DatasetArray(Dataset):
    """Class representing an dataset wich is defined as an array of a defined
    shape.

    Parameters
    ----------
    name : str
        Symbolic name of the dataset.
    subtype : DatasetType
        The type of the seismic data.
    download : bool
        If the dataset must be downloaded (the default is False).
    root : str
        Root download directory (the default is None).
    chunks : Any
        Number of blocks of the array (the default is "auto").

    """
    def __init__(self,
                 name: str,
                 subtype = None,
                 download: bool = False,
                 root: str = None,
                 chunks="auto"):

        Dataset.__init__(self, name, subtype, download, root)

        self.__chunks = chunks

        self._root_file = root

        if root is not None:
            if not os.path.isfile(root):
                raise Exception("Array requires a root=filename.")

            self._root = os.path.dirname(root)

    def __npy_header(self):
        """Read an array header from a filelike object.

        """
        with open(self._root_file, 'rb') as fobj:
            version = numpy.lib.format.read_magic(fobj)
            func_name = ('read_array_header_' +
                         '_'.join(str(v) for v in version))
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

        local_data = da.from_delayed(local_data, shape=npy_shape,
                                     dtype=xp.float32)
        if isinstance(self.__chunks, tuple):
            local_data = local_data.rechunk(self.__chunks)

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

    def _lazy_load_cpu(self):
        """Load data with CPU container + DASK. (It does not load immediattly)

        """
        self._metadata = self._load_meta()
        self._data = self._lazy_load(np)

    def _load_gpu(self):
        """Load data with GPU container (e.g. cupy).

        """
        self._metadata = self._load_meta()
        self._data = self._load(cp)

    def _load_cpu(self):
        """Load data with CPU container (e.g. numpy).

        """
        self._metadata = self._load_meta()
        self._data = self._load(np)

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
        array_file_size = \
            utils.human_readable_size(os.path.getsize(self._root_file),
                                      decimal=2)

        npy_shape = self.shape

        return {
            'size': array_file_size,
            'file': self._root_file,
            'subtype': self._subtype,
            'shape': npy_shape,
            'block': {
               "chunks": self.__chunks
            }
        }


@generate_load
class DatasetLabeled(Dataset):
    """A class representing a labeled dataset. Each item is a 2-element tuple,
    where the first element is a array of data and the second element is the
    respective label. The items can be accessed from `dataset[x]`.

    Parameters
    ----------
    name : str
        Symbolic name of the dataset.
    subtype : DatasetType
        The type of the seismic data.
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
                 subtype = None,
                 download: bool = False,
                 root: str= None,
                 chunks = "auto"):

        Dataset.__init__(self, name, subtype, download, root)

        self.__chunks = chunks

    def download(self):
        """Download the dataset.

        """
        if hasattr(self, '_train') and hasattr(self._train, 'download'):
            self._train.download()

        if hasattr(self, '_val') and hasattr(self._val, 'download'):
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

        assert metadata_train['shape'] == \
               metadata_val['shape'], ("Train and Labels should have same "
                                       "shape: " +
                                       str(metadata_train['shape']) +
                                       " != " +
                                       str(metadata_val['shape']))

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

    def _load_meta(self) -> dict:
        """Load metadata to inspect.

        Returns
        -------
        dict
            A dictionary with metadata information.

        """
        assert self._train._root_file is not None, ("There is no temporary "
                                                    "file to inspect")
        assert self._val._root_file is not None, ("There is no temporary "
                                                  "file to inspect")
        assert os.path.isfile(self._train._root_file), ("The root variable "
                                                        "should be a file")
        assert os.path.isfile(self._val._root_file), ("The root variable "
                                                      "should be a file")

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

    def __getitem__(self, idx):
        return (self._data.__getitem__(idx),
                self._labels.__getitem__(idx))
