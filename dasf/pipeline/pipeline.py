#!/usr/bin/env python3

import uuid
import GPUtil
import inspect
import prefect

import numpy as np
import dask.array as da

import pandas as pd
import dask.dataframe as ddf

from prefect import Task, Flow

from dasf.utils import utils
from dasf.pipeline.types import TaskExecutorType
from prefect.executors.local import LocalExecutor

try:
    import cupy as cp
    import cudf
except ImportError:
    pass


class DAG:
    key = None
    fn = None
    parameters = []
    itself = None

    def set(self, fn, parameters={}, itself=None):
        self.key = hash(fn)
        self.fn = fn
        for k, v in parameters.items():
            self.parameters.append((k, v))
        self.itself = itself

    def set_parameters(self, parameters):
        for k, v in parameters.items():
            self.parameters.append((k, v))


class Pipeline2:
    def __init__(self, name, executor=None):
        self._name = name
        self._executor = executor

        self._dag = []

    def __call(self, func):
        pass

    def __dag_exists(self, item):
        for dag in self._dag:
            if item.key == dag.key:
                return True
        return False

    def __add_parameters_into_dag(self, fn, itself=None):
        item = DAG()
        item.set(fn=fn, itself=itself)
        self._dag.append(item)

    def __add_into_dag(self, fn, parameters={}, itself=None):
        item = DAG()
        item.set(fn=fn, parameters=parameters, itself=itself)
        if not self.__dag_exists(item):
            self._dag.append(item)
        else:
            

    def __recursive_call(self, dag):
        key = hash(dag["fn"])

        for key, value in self._dag.items()
            if len(self._dag[key]["parameters"]) == 0:
                self._dag[key]["return"] = self._dag[key]["fn"]()
                return
            else:
                new_kwargs = dict()
                for parameter in self._dag[key]["parameters"]:
                    parameter_fn = self._dag[key]["parameters"][parameter]
                    print(parameter_fn)
                    if not hash(parameter_fn) in self._dag:
                        raise Exception('Did you include all the parameters '
                                        'or defined the DAG properly?')

    def add_parameters(self, parameters):
        if isinstance(parameters, list):
            for parameter in parameters:
                if inspect.isfunction(parameter) and callable(parameter):
                    self.__add_into_dag(parameter)
                elif inspect.ismethod(parameter):
                    self.__add_into_dag(parameter, itself=parameter.__self__)
                elif hasattr(parameter, 'load'):
                    self.__add_into_dag(parameter.load, itself=parameter)
                else:
                    raise ValueError('This object is not a parameter object.')
        else:
            if inspect.isfunction(parameters) and callable(parameters):
                self.__add_into_dag(parameters)
            elif inspect.ismethod(parameters):
                self.__add_into_dag(parameters, itself=parameter.__self__)
            elif hasattr(parameters, 'load'):
                self.__add_into_dag(parameters.load, itself=parameters)
            else:
                raise ValueError('This object is not a parameter object.')

    def add(self, obj, **kwargs):
        from dasf.transforms.transforms import Transform

        if inspect.isfunction(obj) and callable(obj):
            self.__add_into_dag(obj, kwargs)
        elif inspect.ismethod(obj):
            self.__add_into_dag(obj, kwargs, obj.__self__)
        elif issubclass(obj.__class__, Transform) and hasattr(obj, 'transform'):
            self.__add_into_dag(obj.transform, kwargs, obj)
        elif hasattr(obj, 'fit'):
            self.__add_into_dag(obj.fit, kwargs, obj)
        else:
            raise ValueError('This object is not a function, method or a '
                             'transformer object.')

        return self

    def run(self):
        self.__recursive_call(list(self._dag.values())[0])

        print(self._dag)



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


class WrapperLocalExecutor(LocalExecutor):
    def __init__(self, disable_gpu=False):
        super().__init__()

        self.ngpus = len(GPUtil.getGPUs())
        self.client = None

        if self.ngpus > 0 and not disable_gpu:
            self.dtype = utils.set_executor_gpu()
        else:
            self.dtype = utils.set_executor_default()


class ComputePipeline(Flow):
    def __init__(self, name, executor=None):
        super().__init__(name)

        self.cparameters = dict()

        self.executor = executor

        if self.executor is None:
            self.executor = WrapperLocalExecutor()

    def __check_task_pipes(self, task1, task2):
        if not task2.task_inputs:
            raise NotImplementedError

        for key2 in task2.task_inputs:
            key1 = next(iter(task1.task_output))
            value1 = task1.task_output[key1]
            if key1 == key2 and value1 == task2.task_inputs[key2]:
                return key2
        return None

    def all_upstream_tasks(self):
        tasks = list()
        for k, v in self.all_upstream_edges().items():
            if not v and not issubclass(k.__class__, Parameter):
                tasks.append(k)
        return tasks

    def add_parameters(self, parameters):
        if isinstance(parameters, list):
            for parameter in parameters:
                self.cparameters[parameter.name] = parameter
        else:
            self.cparameters[parameters.name] = parameters

    def add_edge(self, task1, task2, key):
        # key = self.__check_task_pipes(task1, task2)

        if self.executor and hasattr(self.executor, "dtype"):
            # Check if they have setup method.
            # The tasks can be Parameters, Stages and Multi tasks.
            if hasattr(task1, "setup"):
                task1.setup(self.executor)
            if hasattr(task2, "setup"):
                task2.setup(self.executor)

        super().add_edge(task1, task2, key=key)

    def add(self, task, **kwargs):
        if isinstance(task, list):
            for t in task:
                if hasattr(t, "setup"):
                    t.setup(self.executor)

                for arg in kwargs:
                    if hasattr(kwargs[arg], "setup"):
                        kwargs[arg].setup(self.executor)

                    super().add_edge(kwargs[arg], t, key=arg)
        else:
            if self.executor and hasattr(self.executor, "dtype"):
                # Check if they have setup method.
                # The tasks can be Parameters, Stages and Multi tasks.
                if hasattr(task, "setup"):
                    task.setup(self.executor)

            for arg in kwargs:
                if hasattr(kwargs[arg], "setup"):
                    kwargs[arg].setup(self.executor)

                super().add_edge(kwargs[arg], task, key=arg)

        return self

    def run(self, run_on_schedule=None, runner_cls=None, **kwargs):
        return super().run(
            executor=self.executor,
            parameters=self.cparameters,
            run_on_schedule=run_on_schedule,
            runner_cls=runner_cls,
            **kwargs
        )
