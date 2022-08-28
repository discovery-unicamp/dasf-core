#!/usr/bin/env python3

from enum import Enum


class TaskExecutorType(Enum):
    single_cpu = 0
    multi_cpu = 1
    single_gpu = 2
    multi_gpu = 3
