#!/usr/bin/env python3

from enum import IntEnum, auto


class TaskExecutorType(IntEnum):
    single_cpu = auto()
    multi_cpu = auto()
    single_gpu = auto()
    multi_gpu = auto()
