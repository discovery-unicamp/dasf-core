#!/usr/bin/env python3

import uuid
import GPUtil
import prefect

import numpy as np
import dask.array as da

import pandas as pd
import dask.dataframe as ddf

from prefect import Parameter, Task, Flow
from prefect.engine.signals import LOOP

from dasf.utils import utils
from dasf.pipeline.types import TaskExecutorType
from prefect.executors.local import LocalExecutor

try:
    import cupy as cp
    import cudf
except ImportError:
    pass


class ParameterOperator(Parameter):
    """ """

    def __init__(self, name, local=None, gpu=None):
        super().__init__(name=name)

        # Starting attributes with task_* avoid
        # conflicts with parent object.
        self.task_output = None

        self.local = local
        self.gpu = gpu

        # helpers
        self.xp = None
        self.df = None

    def set_output(self, dtype):
        self.task_output = {"output": dtype}

    def setup_cpu(self, executor):
        self.xp = np
        self.df = pd

    def setup_mcpu(self, executor):
        self.xp = da
        self.df = ddf

    def setup_gpu(self, executor):
        self.xp = cp
        self.df = cudf

    def setup_mgpu(self, executor):
        self.xp = da
        self.df = ddf

    def setup(self, executor):
        if hasattr(executor, "client"):
            self.client = executor.client

        self.dtype = utils.return_local_and_gpu(executor, self.local, self.gpu)

        if self.dtype == TaskExecutorType.single_cpu:
            return self.setup_cpu(executor)
        elif self.dtype == TaskExecutorType.multi_cpu:
            return self.setup_mcpu(executor)
        elif self.dtype == TaskExecutorType.single_gpu:
            return self.setup_gpu(executor)
        elif self.dtype == TaskExecutorType.multi_gpu:
            return self.setup_mgpu(executor)


class Operator(Task):
    """ """

    def __init__(self, name, slug=None, checkpoint=False, local=None, gpu=None):

        if slug is None:
            slug = str(uuid.uuid4())

        super().__init__(slug=slug, name=name)

        self.__checkpoint = checkpoint

        # Starting attributes with task_* avoid
        # conflicts with parent object.
        self.task_inputs = None
        self.task_output = None

        self.client = None
        self.dtype = TaskExecutorType.single_cpu

        self.local = local
        self.gpu = gpu

        # helpers
        self.xp = None
        self.df = None

    def set_inputs(self, **kwargs):
        self.task_inputs = kwargs

    def set_output(self, dtype):
        self.task_output = {"output": dtype}

    def set_checkpoint(self, checkpoint):
        self.__checkpoint = checkpoint

    def get_checkpoint(self):
        return self.__checkpoint

    def setup_cpu(self, executor):
        self.xp = np
        self.df = pd

    def setup_lazy_cpu(self, executor):
        self.xp = da
        self.df = ddf

    def setup_gpu(self, executor):
        self.xp = cp
        self.df = cudf

    def setup_lazy_gpu(self, executor):
        self.xp = da
        self.df = ddf

    def setup(self, executor):
        if hasattr(executor, "client"):
            self.client = executor.client

        self.dtype = utils.return_local_and_gpu(executor, self.local, self.gpu)

        if self.dtype == TaskExecutorType.single_cpu:
            return self.setup_cpu(executor)
        elif self.dtype == TaskExecutorType.multi_cpu:
            return self.setup_lazy_cpu(executor)
        elif self.dtype == TaskExecutorType.single_gpu:
            return self.setup_gpu(executor)
        elif self.dtype == TaskExecutorType.multi_gpu:
            return self.setup_lazy_gpu(executor)

    def run_cpu(self, **kwargs):
        pass

    def run_lazy_cpu(self, **kwargs):
        pass

    def run_gpu(self, **kwargs):
        pass

    def run_lazy_gpu(self, **kwargs):
        pass

    def run(self, **kwargs):
        if self.dtype == TaskExecutorType.single_cpu:
            return self.run_cpu(**kwargs)
        elif self.dtype == TaskExecutorType.multi_cpu:
            return self.run_lazy_cpu(**kwargs)
        elif self.dtype == TaskExecutorType.single_gpu:
            return self.run_gpu(**kwargs)
        elif self.dtype == TaskExecutorType.multi_gpu:
            return self.run_lazy_gpu(**kwargs)


class BatchPipeline(Task):
    def __init__(self, name, slug=None):
        if slug is None:
            slug = str(uuid.uuid4())

        super().__init__(slug=slug, name=name)

        self.pipeline = None

    def add_pipeline(self, pipeline):
        assert self.pipeline is None, "Pipeline is already defined"

        self.pipeline = pipeline

    def run(self, data):
        assert self.pipeline is not None, "Pipeline is not defined"

        batch_len = len(data)

        iterator = 0

        index = prefect.context.get("task_loop_result", 0)

        tasks = self.pipeline.all_upstream_tasks()

        data_param = Parameter("data")
        for task in tasks:
            self.pipeline.add_edge(data_param, task, "data")

        self.pipeline.cparameters["data"] = data[index]

        result = self.pipeline.run()

        iterator += data.batch_size * (index + 1)
        index += 1

        # Loop
        if iterator >= batch_len:
            return result
        raise LOOP(result=index)


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
