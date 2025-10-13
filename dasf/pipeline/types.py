#!/usr/bin/env python3
"""Type definitions module."""

from enum import IntEnum, auto


class TaskExecutorType(IntEnum):
    """This class defines the type of task executor."""
    single_cpu = auto()
    multi_cpu = auto()
    single_gpu = auto()
    multi_gpu = auto()
