#!/usr/bin/env python3

"""
Pipeline module for creating and executing computational pipelines.

This module provides the Pipeline class and PipelinePlugin base class for
creating directed acyclic graphs (DAGs) of computational tasks and executing
them with various executors. The pipeline supports dependency resolution,
error handling, visualization, and plugin callbacks.
"""

import inspect
from typing import List

import graphviz
import networkx as nx

from dasf.utils.logging import init_logging


class PipelinePlugin:
    """A base class for pipeline plugins."""
    def on_pipeline_start(self, fn_keys):
        """
        Called when the pipeline starts.

        Parameters
        ----------
        fn_keys : list
            A list of function keys.
        """
        pass

    def on_pipeline_end(self):
        """Called when the pipeline ends."""
        pass

    def on_task_start(self, func, params, name):
        """
        Called when a task starts.

        Parameters
        ----------
        func : function
            The function to be executed.
        params : dict
            The parameters of the function.
        name : str
            The name of the task.
        """
        pass

    def on_task_end(self, func, params, name, ret):
        """
        Called when a task ends.

        Parameters
        ----------
        func : function
            The function to be executed.
        params : dict
            The parameters of the function.
        name : str
            The name of the task.
        ret : object
            The return value of the task.
        """
        pass

    def on_task_error(self, func, params, name, exception):
        """
        Called when a task fails.

        Parameters
        ----------
        func : function
            The function to be executed.
        params : dict
            The parameters of the function.
        name : str
            The name of the task.
        exception : Exception
            The exception raised by the task.
        """
        pass


class Pipeline:
    """
    A class for creating and executing pipelines.

    Parameters
    ----------
    name : str
        The name of the pipeline.
    executor : object, optional
        The executor to use for running the pipeline, by default None.
    verbose : bool, optional
        Whether to print verbose output, by default False.
    callbacks : List[PipelinePlugin], optional
        A list of plugins to use with the pipeline, by default None.
    """
    def __init__(self,
                 name,
                 executor=None,
                 verbose=False,
                 callbacks: List[PipelinePlugin] = None):
        """
        Constructor for the Pipeline class.

        Parameters
        ----------
        name : str
            The name of the pipeline.
        executor : object, optional
            The executor to use for running the pipeline, by default None.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        callbacks : List[PipelinePlugin], optional
            A list of plugins to use with the pipeline, by default None.
        """
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
        """
        Register a plugin with the pipeline.

        Parameters
        ----------
        plugin : PipelinePlugin or other
            The plugin to register. If it's a PipelinePlugin, it will be added
            to the callbacks list. Otherwise, it will be registered with the executor.
        """
        if isinstance(plugin, PipelinePlugin):
            self._callbacks.append(plugin)
        else:
            self._executor.register_plugin(plugin)

    def info(self):
        """
        Print information about the executor.

        Displays the current executor's information by calling its info property.
        """
        print(self._executor.info)

    def execute_callbacks(self, func_name: str, *args, **kwargs):
        """
        Execute a method on all registered callbacks.

        Parameters
        ----------
        func_name : str
            The name of the method to call on each callback.
        *args
            Positional arguments to pass to the callback method.
        **kwargs
            Keyword arguments to pass to the callback method.
        """
        for callback in self._callbacks:
            getattr(callback, func_name)(*args, **kwargs)

    def __add_into_dag(self, obj, func_name, parameters=None, itself=None):
        """
        Add an object and its dependencies into the DAG.

        Parameters
        ----------
        obj : callable
            The function or callable object to add to the DAG.
        func_name : str
            The name to use for this function in the DAG.
        parameters : dict, optional
            Parameters for the function that may contain dependencies.
        itself : object, optional
            Reference to the object instance if obj is a bound method.
        """
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
        """
        Inspect an object to determine its type and extract relevant information.

        Parameters
        ----------
        obj : object
            The object to inspect (function, method, or class instance).

        Returns
        -------
        tuple
            A tuple containing (callable, function_name, object_reference).

        Raises
        ------
        ValueError
            If the object type is not supported.
        """
        from dasf.datasets.base import Dataset
        from dasf.ml.inference.loader.base import BaseLoader
        from dasf.transforms.base import Fit, Transform

        def generate_name(class_name, func_name):
            """
            Generate a qualified name for a class method.

            Parameters
            ----------
            class_name : str
                The name of the class.
            func_name : str
                The name of the function/method.

            Returns
            -------
            str
                A qualified name in the format "ClassName.method_name".
            """
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
        """
        Add an object to the pipeline DAG.

        Parameters
        ----------
        obj : object
            The object to add to the pipeline (function, method, or transform).
        **kwargs
            Keyword arguments that represent dependencies for this object.

        Returns
        -------
        Pipeline
            Returns self for method chaining.
        """
        obj, func_name, objref = self.__inspect_element(obj)
        self.__add_into_dag(obj, func_name, kwargs, objref)

        return self

    def visualize(self, filename=None):
        """
        Generate a visual representation of the pipeline DAG.

        Parameters
        ----------
        filename : str, optional
            The filename to save the visualization. If None and not in a notebook,
            a temporary file will be used.

        Returns
        -------
        graphviz.Digraph or str
            In Jupyter notebooks, returns the graphviz object for inline display.
            Otherwise, returns the path to the saved visualization file.
        """
        from dasf.utils.funcs import is_notebook

        if is_notebook():
            return self._dag_g
        return self._dag_g.view(filename)

    def __register_dataset(self, dataset):
        """
        Register a dataset with the executor for reusability.

        Parameters
        ----------
        dataset : Dataset
            The dataset object to register.

        Returns
        -------
        Dataset
            The registered dataset object from the executor.
        """
        key = str(hash(dataset.load))
        kwargs = {key: dataset}

        if not self._executor.has_dataset(key):
            return self._executor.register_dataset(**kwargs)

        return self._executor.get_dataset(key)

    def __execute(self, func, params, name):
        """
        Execute a function with resolved dependencies.

        Parameters
        ----------
        func : callable
            The function to execute.
        params : dict or None
            Parameters containing dependency references.
        name : str
            The name of the task being executed.

        Returns
        -------
        object
            The result of the function execution.
        """
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
        """
        Get the result from a specific object in the pipeline.

        Parameters
        ----------
        obj : object
            The object whose result should be retrieved.

        Returns
        -------
        object
            The result produced by the specified object during pipeline execution.

        Raises
        ------
        Exception
            If the pipeline hasn't been executed yet or if the object
            wasn't added to the pipeline.
        """
        _, obj_name, *_ = self.__inspect_element(obj)

        for key in self._dag_table:
            if self._dag_table[key]["name"] == obj_name:
                if self._dag_table[key]["ret"] is None:
                    raise Exception("Pipeline was not executed yet.")
                return self._dag_table[key]["ret"]

        raise Exception(f"Function {obj_name} was not added into pipeline.")

    def run(self):  # noqa: C901
        """
        Execute the pipeline by running all tasks in topological order.

        Executes all tasks in the pipeline according to their dependencies,
        handling errors and calling appropriate callbacks throughout the process.

        Returns
        -------
        object or None
            The result of the pipeline execution, typically the result of
            the last task executed.

        Raises
        ------
        Exception
            If the pipeline is not a directed acyclic graph, if the executor
            doesn't have an execute method, or if the executor is not connected.
        """
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
