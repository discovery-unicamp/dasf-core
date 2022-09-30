#!/usr/bin/env python3

import inspect
import graphviz

import numpy as np
import dask.array as da

import dask.dataframe as ddf

import networkx as nx

from dasf.utils import utils
from dasf.utils.logging import init_logging

try:
    import cupy as cp
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

    def __add_into_dag(self, obj, func_name, parameters=None, itself=None):
        key = hash(obj)

        if key not in self._dag_table:
            self._dag.add_node(key)
            self._dag_table[key] = dict()
            self._dag_table[key]["fn"] = obj
            self._dag_table[key]["name"] = func_name
            self._dag_table[key]["parameters"] = None
            self._dag_table[key]["ret"] = None

        if parameters and isinstance(parameters, dict):
            if self._dag_table[key]["parameters"] is None:
                self._dag_table[key]["parameters"] = parameters
            else:
                self._dag_table[key]["parameters"].update(parameters)

            # If we are adding a object which require parameters,
            # we need to make sure they are mapped into DAG.
            for k, v in parameters.items():
                dep_obj, dep_func_name, _ = self.__inspect_element(v)
                self.add(dep_obj)
                self._dag.add_edge(hash(dep_obj), key)
                self._dag_g.edge(dep_func_name, func_name, label=k)

    def __inspect_element(self, obj):
        from dasf.datasets.base import Dataset
        from dasf.transforms.base import Transform, Fit

        def generate_name(class_name, func_name):
            return ("%s.%s" % (class_name, func_name))

        if inspect.isfunction(obj) and callable(obj):
            return (obj,
                    obj.__qualname__,
                    None)
        elif inspect.ismethod(obj):
            return (obj,
                    generate_name(obj.__self__.__class__.__name__,
                                  obj.__name__),
                    obj.__self__)
        elif issubclass(obj.__class__, Transform) and hasattr(obj, "transform"):
            return (obj.transform,
                    generate_name(obj.__class__.__name__,
                                  "transform"),
                    obj)
        elif issubclass(obj.__class__, Dataset) and hasattr(obj, "load"):
            return (obj.load,
                    generate_name(obj.__class__.__name__,
                                  "load"),
                    obj)
        elif issubclass(obj.__class__, Fit) and hasattr(obj, "fit"):
            return (obj.fit,
                    generate_name(obj.__class__.__name__,
                                  "fit"),
                    obj)
        else:
            raise ValueError(
                f"This object {obj.__name__} is not a function, method "
                "or a transformer object."
            )

    def add(self, obj, **kwargs):
        obj, func_name, objref = self.__inspect_element(obj)
        self.__add_into_dag(obj, func_name, kwargs, objref)

        return self

    def visualize(self, filename=None):
        if utils.is_notebook():
            return self._dag_g
        return self._dag_g.view(filename)

    def __execute(self, func, params, name):
        ret = None

        new_params = dict()
        if params:
            for k, v in params.items():
                dep_obj, *_ = self.__inspect_element(v)
                req_key = hash(dep_obj)

                new_params[k] = self._dag_table[req_key]["ret"]

        if len(new_params) > 0:
            if self._executor:
                ret = self._executor.execute(fn=func, **new_params)
            else:
                ret = func(**new_params)
        else:
            if self._executor:
                ret = self._executor.execute(fn=func)
            else:
                ret = func()

        return ret

    def get_result_from(self, obj):
        _, obj_name, *_ = self.__inspect_element(obj)

        for key in self._dag_table:
            if self._dag_table[key]["name"] == obj_name:
                if not self._dag_table[key]["ret"]:
                    raise Exception("Pipeline was not executed yet.")
                return self._dag_table[key]["ret"]

        raise Exception(f"Function {obj_name} was not added into pipeline.")

    def run(self):
        if not nx.is_directed_acyclic_graph(self._dag):
            raise Exception("Pipeline has not a DAG format.")

        if self._executor and not hasattr(self._executor, "execute"):
            raise Exception(
                f"Executor {self._executor.__name__} has not a execute() "
                "method."
            )

        if self._executor:
            if not self._executor.is_connected:
                raise Exception("Executor is not connected.")

        fn_keys = list(nx.topological_sort(self._dag))

        self._logger.info(f"Beginning pipeline run for '{self._name}'")

        if self._executor:
            self._executor.pre_run(self)

        ret = None
        failed = False

        for fn_key in fn_keys:
            func = self._dag_table[fn_key]["fn"]
            params = self._dag_table[fn_key]["parameters"]
            name = self._dag_table[fn_key]["name"]

            if not failed:
                self._logger.info(f"Task '{name}': Starting task run...")
            else:
                self._logger.error(f"Task '{name}': Starting task run...")

            try:
                if not failed:
                    # Execute DAG node only if there is no error during the
                    # execution. Otherwise, skip it.
                    self._dag_table[fn_key]["ret"] = self.__execute(func,
                                                                    params,
                                                                    name)
            except Exception as e:
                failed = True
                err = str(e)
                self._logger.exception(f"Task '{name}': Failed with:\n{err}")

            if not failed:
                self._logger.info(f"Task '{name}': Finished task run")
            else:
                self._logger.error(f"Task '{name}': Finished task run")

        if failed:
            self._logger.info(f"Pipeline failed at '{name}'")
        else:
            self._logger.info("Pipeline run successfully")

        if self._executor:
            self._executor.post_run(self)

        return ret
