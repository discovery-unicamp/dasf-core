#!/usr/bin/env python3

import inspect
from typing import List

import graphviz
import networkx as nx

from dasf.utils.logging import init_logging


class PipelinePlugin:
    def on_pipeline_start(self, fn_keys):
        pass

    def on_pipeline_end(self):
        pass

    def on_task_start(self, func, params, name):
        pass

    def on_task_end(self, func, params, name, ret):
        pass

    def on_task_error(self, func, params, name, exception):
        pass


class Pipeline:
    def __init__(self,
                 name,
                 executor=None,
                 verbose=False,
                 callbacks: List[PipelinePlugin] = None):
        from dasf.pipeline.executors.wrapper import LocalExecutor

        self._name = name
        self._executor = executor if executor is not None else LocalExecutor()
        self._verbose = verbose

        self._dag = nx.DiGraph()
        self._dag_table = dict()
        self._dag_g = graphviz.Digraph(name, format="png")

        self._logger = init_logging()
        self._callbacks = callbacks or []

    def register_plugin(self, plugin):
        if isinstance(plugin, PipelinePlugin):
            self._callbacks.append(plugin)
        else:
            self._executor.register_plugin(plugin)

    def info(self):
        print(self._executor.info)

    def execute_callbacks(self, func_name: str, *args, **kwargs):
        for callback in self._callbacks:
            getattr(callback, func_name)(*args, **kwargs)

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
                if not self._dag.has_node(str(key)):
                    self._dag_g.node(str(key), func_name)

                if not self._dag.has_node(str(hash(dep_obj))):
                    self._dag_g.node(str(hash(dep_obj)), dep_func_name)

                self._dag.add_edge(hash(dep_obj), key)

                self._dag_g.edge(str(hash(dep_obj)), str(key), label=k)

    def __inspect_element(self, obj):
        from dasf.datasets.base import Dataset
        from dasf.ml.inference.loader.base import BaseLoader
        from dasf.transforms.base import Fit, Transform

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
        elif issubclass(obj.__class__, Dataset) and hasattr(obj, "load"):
            # (Disabled) Register dataset for reusability
            # obj = self.__register_dataset(obj)

            return (obj.load,
                    generate_name(obj.__class__.__name__,
                                  "load"),
                    obj)
        elif issubclass(obj.__class__, Fit) and hasattr(obj, "fit"):
            return (obj.fit,
                    generate_name(obj.__class__.__name__,
                                  "fit"),
                    obj)
        elif issubclass(obj.__class__, BaseLoader) and hasattr(obj, "load"):
            return (obj.load,
                    generate_name(obj.__class__.__name__,
                                  "load"),
                    obj)
        elif issubclass(obj.__class__, Transform) and hasattr(obj, "transform"):
            return (obj.transform,
                    generate_name(obj.__class__.__name__,
                                  "transform"),
                    obj)
        else:
            raise ValueError(
                f"This object {obj.__class__.__name__} is not a function, "
                "method or a transformer object."
            )

    def add(self, obj, **kwargs):
        obj, func_name, objref = self.__inspect_element(obj)
        self.__add_into_dag(obj, func_name, kwargs, objref)

        return self

    def visualize(self, filename=None):
        from dasf.utils.funcs import is_notebook

        if is_notebook():
            return self._dag_g
        return self._dag_g.view(filename)

    def __register_dataset(self, dataset):
        key = str(hash(dataset.load))
        kwargs = {key: dataset}

        if not self._executor.has_dataset(key):
            return self._executor.register_dataset(**kwargs)

        return self._executor.get_dataset(key)

    def __execute(self, func, params, name):
        ret = None

        new_params = dict()
        if params:
            for k, v in params.items():
                dep_obj, *_ = self.__inspect_element(v)
                req_key = hash(dep_obj)

                new_params[k] = self._dag_table[req_key]["ret"]

        if len(new_params) > 0:
            ret = self._executor.execute(fn=func, **new_params)
        else:
            ret = self._executor.execute(fn=func)

        return ret

    def get_result_from(self, obj):
        _, obj_name, *_ = self.__inspect_element(obj)

        for key in self._dag_table:
            if self._dag_table[key]["name"] == obj_name:
                if self._dag_table[key]["ret"] is None:
                    raise Exception("Pipeline was not executed yet.")
                return self._dag_table[key]["ret"]

        raise Exception(f"Function {obj_name} was not added into pipeline.")

    def run(self):
        if not nx.is_directed_acyclic_graph(self._dag):
            raise Exception("Pipeline has not a DAG format.")

        if not hasattr(self._executor, "execute"):
            raise Exception(
                f"Executor {self._executor.__class__.__name__} has not a execute() "
                "method."
            )

        if not self._executor.is_connected:
            raise Exception("Executor is not connected.")

        fn_keys = list(nx.topological_sort(self._dag))

        self._logger.info(f"Beginning pipeline run for '{self._name}'")
        self.execute_callbacks("on_pipeline_start", fn_keys)

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
                    self.execute_callbacks("on_task_start", func=func,
                                           params=params, name=name)

                    result = self.__execute(func, params, name)
                    self._dag_table[fn_key]["ret"] = result

                    self.execute_callbacks("on_task_end", func=func,
                                           params=params, name=name,
                                           ret=result)

            except Exception as e:
                self.execute_callbacks("on_task_error", func=func,
                                       params=params, name=name, exception=e)
                failed = True
                failed_at = name

                err = str(e)
                self._logger.exception(f"Task '{name}': Failed with:\n{err}")

            if not failed:
                self._logger.info(f"Task '{name}': Finished task run")
            else:
                self._logger.error(f"Task '{name}': Finished task run")

        if failed:
            self._logger.info(f"Pipeline failed at '{failed_at}'")
        else:
            self._logger.info("Pipeline run successfully")

        self._executor.post_run(self)

        self.execute_callbacks("on_pipeline_end")
        return ret
