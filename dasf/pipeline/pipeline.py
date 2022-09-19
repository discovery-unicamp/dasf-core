#!/usr/bin/env python3

import uuid
import GPUtil
import inspect
import prefect

import numpy as np
import dask.array as da

import pandas as pd
import dask.dataframe as ddf

import networkx as nx

from prefect import flow

from dasf.utils import utils
from dasf.pipeline.types import TaskExecutorType

try:
    import cupy as cp
    import cudf
except ImportError:
    pass


class Pipeline2:
    def __init__(self, name, executor=None):
        self._name = name
        self._executor = executor

        self._dag = nx.DiGraph()
        self._dag_table = dict()

    def __add_into_dag(self, obj, parameters=None, itself=None):
        key = hash(obj)

        if not key in self._dag_table:
            self._dag.add_node(key)
            self._dag_table[key] = dict()
            self._dag_table[key]["fn"] = obj
            self._dag_table[key]["parameters"] = None

        if parameters and isinstance(parameters, dict):
            if self._dag_table[key]["parameters"] is None:
                self._dag_table[key]["parameters"] = parameters
            else:
                self._dag_table[key]["parameters"].update(parameters)

            # If we are adding a object which require parameters,
            # we need to make sure they are mapped into DAG.
            for k, v in parameters.items():
                self.add(v)
                self._dag.add_edge(hash(v), key)


    def add(self, obj, **kwargs):
        from dasf.datasets.base import Dataset
        from dasf.transforms.transforms import Transform

        if inspect.isfunction(obj) and callable(obj):
            self.__add_into_dag(obj, kwargs)
        elif inspect.ismethod(obj):
            self.__add_into_dag(obj, kwargs, obj.__self__)
        elif issubclass(obj.__class__, Transform) and hasattr(obj, 'transform'):
            self.__add_into_dag(obj.transform, kwargs, obj)
        elif issubclass(obj.__class__, Dataset) and hasattr(obj, 'load'):
            self.__add_into_dag(obj.load, kwargs, obj)
        elif hasattr(obj, 'fit'):
            self.__add_into_dag(obj.fit, kwargs, obj)
        else:
            raise ValueError('This object is not a function, method or a '
                             'transformer object.')

        return self

    def run(self):
        if not nx.is_directed_acyclic_graph(self._dag):
            raise Exception("Pipeline has not a DAG format. Review it.")

        @flow
        def run_flow():
            func_keys = list(nx.topological_sort(self._dag))

            for fn_key in func_keys:


class Operator:
    pass


class BlockOperator(Operator):
    def __init__(
        self,
        name,
        function,
        slug=None,
        checkpoint=False,
        local=None,
        gpu=None,
        depth=None,
        boundary=None,
        trim=True,
        output_chunk=None,
    ):

        super().__init__(
            name=name, slug=slug, checkpoint=checkpoint, local=local, gpu=gpu
        )

        self.function = function
        self.depth = depth
        self.boundary = boundary
        self.trim = trim
        self.output_chunk = output_chunk

        if (
            self.boundary is None
            and self.depth is not None
            or self.boundary is not None
            and self.depth is None
        ):
            raise Exception("Both boundary and depth should be passed " "together")

    def run(self, X, **kwargs):
        if utils.is_executor_gpu(self.dtype) and utils.is_gpu_supported():
            dtype = cp.float32
        else:
            dtype = np.float32

        if (
            isinstance(X, da.core.Array)
            or isinstance(X, ddf.core.DataFrame)
            and utils.is_executor_cluster(self.dtype)
        ):
            drop_axis, new_axis = utils.block_chunk_reduce(X, self.output_chunk)

            if self.depth and self.boundary:
                if self.trim:
                    new_data = X.map_overlap(
                        self.function,
                        **kwargs,
                        dtype=dtype,
                        depth=self.depth,
                        boundary=self.boundary
                    )
                else:
                    data_blocks = da.overlap.overlap(
                        X, depth=self.depth, boundary=self.boundary
                    )

                    new_data = data_blocks.map_blocks(
                        self.function,
                        dtype=dtype,
                        drop_axis=drop_axis,
                        new_axis=new_axis,
                        **kwargs
                    )
            else:
                if isinstance(X, da.core.Array):
                    new_data = X.map_blocks(
                        self.function,
                        dtype=dtype,
                        drop_axis=drop_axis,
                        new_axis=new_axis,
                        **kwargs
                    )
                elif isinstance(X, ddf.core.DataFrame):
                    new_data = X.map_partitions(self.function, **kwargs)

            return new_data
        else:
            return self.function(X, **kwargs)
