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


class Pipeline2:
    def __init__(self, name, executor=None):
        self._name = name
        self._executor = executor

        self._dag = dict()

    def __call(self, func):
        pass

    def __add_into_dag(self, obj, fn, reqs, itself=None):
        self._dag[hash(obj)] = dict()
        self._dag[hash(obj)]["fn"] = fn
        self._dag[hash(obj)]["reqs"] = reqs
        if itself:
            self._dag[hash(obj)]["reqs"]["self"] = itself

    def add(self, obj, **kwargs):
        from dasf.transforms.transforms import Transform

        if inspect.isfunction(obj) and callable(obj):
            self.__add_into_dag(obj, obj, kwargs)
        elif inspect.ismethod(obj):
            self.__add_into_dag(obj, obj, kwargs, obj.__self__)
        else:
            if issubclass(obj, Transform) and hasattr(obj, 'transform'):
                self.__add_into_dag(obj, obj.transform, kwargs, obj)
            elif hasattr(obj, 'fit'):
                self.__add_into_dag(obj, obj.fit, kwargs, obj)
            else:
                raise ValueError('This object is not a function, method or a '
                                 'transformer object.')

        return self


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
