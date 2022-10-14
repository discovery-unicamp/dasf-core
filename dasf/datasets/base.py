#!/usr/bin/env python3

import os
import dask

import numpy as np
import numpy.lib.format
import dask.array as da

from numbers import Number

try:
    import cupy as cp
except ImportError:
    pass

from pathlib import Path

from dasf.utils import utils
from dasf.utils.decorators import task_handler
from dasf.utils.types import is_array
from dasf.utils.types import is_dask_array


class Dataset(object):
    def __init__(self, name, download=False, root=None):

        # Dataset internals
        self._name = name
        self._download = download
        self._root = root
        self._metadata = dict()
        self._data = None

        # Internal chunks division (for Dask Types)
        self.__chunks = None

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

    def __len__(self):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return len(self._data)

    def __getitem__(self, idx):
        if self._data is None:
            raise Exception("Data is not loaded yet")

        return self._data.__getitem__(idx)


class DatasetArray(Dataset):
    def __init__(self, name, download=False, root=None, chunks="auto"):

        Dataset.__init__(self, name, download, root)

        self.__chunks = chunks

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

            self.__class__(name=self._name, chunks=self.__chunks)
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
        attrs = dir(self._data)
        for attr in attrs:
            if not attr.startswith("__") and callable(getattr(self._data, attr)):
                if not hasattr(self, attr):
                    self.__dict__[attr] = getattr(self._data, attr)

    def __npy_header(self):
        with open(self._root_file, "rb") as fobj:
            version = numpy.lib.format.read_magic(fobj)
            func_name = "read_array_header_" + "_".join(str(v) for v in version)
            func = getattr(numpy.lib.format, func_name)
            return func(fobj)

    def _lazy_load(self, xp, **kwargs):
        npy_shape = self.shape

        local_data = dask.delayed(xp.load)(self._root_file, **kwargs)

        local_data = da.from_delayed(local_data, shape=npy_shape, dtype=xp.float32)
        if isinstance(self.__chunks, tuple):
            local_data = local_data.rechunk(self.__chunks)

        return local_data

    def _load(self, xp, **kwargs):
        return xp.load(self._root_file, **kwargs)

    def _load_meta(self):
        assert self._root_file is not None, "There is no temporary file to " "inspect"
        assert os.path.isfile(self._root_file), (
            "The root variable should " "be a NPY file"
        )

        return self.inspect_metadata()

    def _lazy_load_gpu(self):
        self._metadata = self._load_meta()
        self._data = self._lazy_load(cp)
        self.__copy_attrs_from_data()
        return self

    def _lazy_load_cpu(self):
        self._metadata = self._load_meta()
        self._data = self._lazy_load(np)
        self.__copy_attrs_from_data()
        return self

    def _load_gpu(self):
        self._metadata = self._load_meta()
        self._data = self._load(cp)
        self.__copy_attrs_from_data()
        return self

    def _load_cpu(self):
        self._metadata = self._load_meta()
        self._data = self._load(np)
        self.__copy_attrs_from_data()
        return self

    @task_handler
    def load(self):
        ...

    @property
    def shape(self):
        return self.__npy_header()[0]

    def inspect_metadata(self):
        array_file_size = utils.human_readable_size(
            os.path.getsize(self._root_file), decimal=2
        )

        npy_shape = self.shape

        return {
            "size": array_file_size,
            "file": self._root_file,
            "shape": npy_shape,
            "block": {"chunks": self.__chunks},
        }


class DatasetLabeled(Dataset):
    def __init__(self, name, download=False, root=None, chunks="auto"):

        Dataset.__init__(self, name, download, root)

        self.__chunks = chunks

    def download(self):
        if hasattr(self, "_train") and hasattr(self._train, "download"):
            self._train.download()

        if hasattr(self, "_val") and hasattr(self._val, "download"):
            self._val.download()

    def inspect_metadata(self):
        metadata_train = self._train.inspect_metadata()
        metadata_val = self._val.inspect_metadata()

        assert (
            metadata_train["shape"] == metadata_val["shape"]
        ), "Train and Labels should have same " "shape: " + str(
            metadata_train["shape"]
        ) + " != " + str(
            metadata_val["shape"]
        )

        return {"train": metadata_train, "labels": metadata_val}

    def _lazy_load(self, xp, **kwargs):
        local_data = self._train._lazy_load(xp)
        local_labels = self._val._lazy_load(xp)

        return (local_data, local_labels)

    def _load(self, xp, **kwargs):
        local_data = self._train._load(xp)
        local_labels = self._val._load(xp)

        return (local_data, local_labels)

    def _load_meta(self):
        assert self._train._root_file is not None, (
            "There is no temporary " "file to inspect"
        )
        assert self._val._root_file is not None, (
            "There is no temporary " "file to inspect"
        )
        assert os.path.isfile(self._train._root_file), (
            "The root variable " "should be a file"
        )
        assert os.path.isfile(self._val._root_file), (
            "The root variable " "should be a file"
        )

        return self.inspect_metadata()

    def _lazy_load_gpu(self):
        self._metadata = self._load_meta()
        self._data, self._labels = self._lazy_load(cp)

    def _lazy_load_cpu(self):
        self._metadata = self._load_meta()
        self._data, self._labels = self._lazy_load(np)

    def _load_gpu(self):
        self._metadata = self._load_meta()
        self._data, self._labels = self._load(cp)

    def _load_cpu(self):
        self._metadata = self._load_meta()
        self._data, self._labels = self._load(np)

    @task_handler
    def load(self):
        ...

    def __getitem__(self, idx):
        return (self._data.__getitem__(idx), self._labels.__getitem__(idx))
