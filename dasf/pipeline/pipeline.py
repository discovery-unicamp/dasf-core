#!/usr/bin/env python3

import uuid
import time
import GPUtil
import inspect
import graphviz

import numpy as np
import dask.array as da

import pandas as pd
import dask.dataframe as ddf

import networkx as nx

from dasf.utils import utils
from dasf.utils.logging import init_logging
from dasf.pipeline.types import TaskExecutorType

try:
    import cupy as cp
    import cudf
except ImportError:
    pass


class Pipeline:
    def __init__(self, name, executor=None, verbose=False):
        self._name = name
        self._executor = executor
        self._verbose = verbose

        self._dag = nx.DiGraph()
        self._dag_table = dict()
        self._dag_g = graphviz.Digraph(name, format="png")

        self._logger = init_logging()

    def __add_into_dag(self, obj, parameters=None, itself=None):
        key = hash(obj)

        if not key in self._dag_table:
            self._dag.add_node(key)
            self._dag_table[key] = dict()
            self._dag_table[key]["fn"] = obj
            self._dag_table[key]["name"] = obj.__qualname__
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
                self._dag_g.edge(v.__qualname__, obj.__qualname__, label=k)

    def add(self, obj, **kwargs):
        from dasf.datasets.base import Dataset
        from dasf.transforms.transforms import Transform

        if inspect.isfunction(obj) and callable(obj):
            self.__add_into_dag(obj, kwargs)
        elif inspect.ismethod(obj):
            self.__add_into_dag(obj, kwargs, obj.__self__)
        elif issubclass(obj.__class__, Transform) and hasattr(obj, "transform"):
            self.__add_into_dag(obj.transform, kwargs, obj)
        elif issubclass(obj.__class__, Dataset) and hasattr(obj, "load"):
            self.__add_into_dag(obj.load, kwargs, obj)
        elif hasattr(obj, "fit"):
            self.__add_into_dag(obj.fit, kwargs, obj)
        else:
            raise NotImplementedError(
                f"There is no implementation of {wrapper_func_attr} nor "
                 "{func_name}"
            )

        return self

    def visualize(self, filename=None):
        if utils.is_notebook():
            return self._dag_g
        return self._dag_g.view(filename)

    def run(self):
        if not nx.is_directed_acyclic_graph(self._dag):
            raise Exception("Pipeline has not a DAG format. Review it.")

        if self._executor and not hasattr(self._executor, "run"):
            raise Exception(
                f"Executor {self._executor.__name__} has not a run() method."
            )

        if self._executor:
            while True:
                if self._executor.is_connected:
                    break
                time.sleep(2)

        fn_keys = list(nx.topological_sort(self._dag))

        ret = None
        failed = False

        self._logger.info(f"Beginning pipeline run for '{self._name}'")

        for fn_key in fn_keys:
            func = self._dag_table[fn_key]["fn"]
            params = self._dag_table[fn_key]["parameters"]
            name = self._dag_table[fn_key]["name"]

            new_params = dict()
            if params:
                for k, v in params.items():
                    req_key = hash(v)

                    new_params[k] = self._dag_table[req_key]["ret"]

            if self._executor:
                self._executor.pre_run()

            self._logger.info(f"Task '{name}': Starting task run...")

            try:
                if len(new_params) > 0:
                    if self._executor:
                        ret = self._executor.run(fn=func, **new_params)
                    else:
                        ret = func(**new_params)
                else:
                    if self._executor:
                        ret = self._executor.run(fn=func)
                    else:
                        ret = func()
            except Exception as e:
                self._logger.exception(str(e))
                failed = True
                break

            self._logger.info(f"Task '{name}': Finished task run")

            self._dag_table[fn_key]["ret"] = ret

        if failed:
            self._logger.info(f"Pipeline failed at '{name}'")
        else:
            self._logger.info("Pipeline run successfully")

        return ret


class BlockOperator:
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
                        boundary=self.boundary,
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
                        **kwargs,
                    )
            else:
                if isinstance(X, da.core.Array):
                    new_data = X.map_blocks(
                        self.function,
                        dtype=dtype,
                        drop_axis=drop_axis,
                        new_axis=new_axis,
                        **kwargs,
                    )
                elif isinstance(X, ddf.core.DataFrame):
                    new_data = X.map_partitions(self.function, **kwargs)

            return new_data
        else:
            return self.function(X, **kwargs)
