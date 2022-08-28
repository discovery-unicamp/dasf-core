#!/usr/bin/env python3

import GPUtil

from dasf.pipeline.types import TaskExecutorType
from prefect.executors.local import LocalExecutor


class PrefectPipelineExecutor(LocalExecutor):
    """
    """
    @property
    def dtype(self):
        return TaskExecutorType.single_gpu \
            if len(GPUtil.getGPUs()) > 0 else TaskExecutorType.single_cpu
