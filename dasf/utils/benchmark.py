#!/usr/bin/env python3

"""
Benchmarking utilities for performance and memory profiling.

This module provides classes for timing and memory profiling of code execution
with support for multiple backends including cProfile, perf_counter, memray,
and memory_profiler.
"""

import timeit
from time import perf_counter

try:
    import memray
    USE_MEMRAY = True
except ImportError:  # pragma: no cover
    USE_MEMRAY = False

import cProfile
from pstats import Stats

try:
    from functools import partial

    from memory_profiler import LineProfiler, choose_backend, memory_usage, show_results

    USE_MEM_PROF = True
except ImportError:  # pragma: no cover
    USE_MEM_PROF = False


class TimeBenchmark:
    """
    A class for timing and profiling code execution.

    Supports multiple backends for performance measurement including
    cProfile for detailed profiling and perf_counter for simple timing.

    Parameters
    ----------
    backend : str, optional
        The timing backend to use. Options are:
        - "cprofile": Python's cProfile for detailed profiling
        - "perf_counter": Simple timing using time.perf_counter()
        Default is "cprofile".
    """
    def __init__(self, backend="cprofile"):
        """
        Initialize the TimeBenchmark instance.

        Parameters
        ----------
        backend : str, optional
            The timing backend to use. Options are:
            - "cprofile": Python's cProfile for detailed profiling
            - "perf_counter": Simple timing using time.perf_counter()
            - "timeit": Use timeit module for repeated measurements
            Default is "cprofile".
        """
        self.__backend = backend

    def __enter__(self):
        """
        Enter the timing context manager.

        Returns
        -------
        TimeBenchmark
            Self for use in context manager.
        """
        if self.__backend == "cprofile":
            self.__pr = cProfile.Profile()
            self.__pr.enable()
        elif self.__backend == "perf_counter":
            self.__start = perf_counter()
            self.__end = 0.0
        else:
            print("There is no available backend")
        return self

    def __exit__(self, *args, **kwargs):
        """
        Exit the timing context manager and display results.

        Parameters
        ----------
        *args
            Exception type, value, and traceback (if any).
        **kwargs
            Additional keyword arguments.
        """
        if self.__backend == "cprofile":
            self.__pr.disable()
            p = Stats(self.__pr)

            p.strip_dirs().sort_stats('cumulative').print_stats(10)
        elif self.__backend == "perf_counter":
            self.__end = perf_counter()
            print("Time spent:", self.__end - self.__start)

    def run(self, function, *args, **kwargs):
        """
        Run a function with timing/profiling.

        Parameters
        ----------
        function : callable
            The function to benchmark.
        *args
            Positional arguments to pass to the function.
        **kwargs
            Keyword arguments to pass to the function.
        """
        if self.__backend == "cprofile":
            pr = cProfile.Profile()
            pr.enable()

            function(*args, **kwargs)

            pr.disable()
            p = Stats(pr)

            p.strip_dirs().sort_stats('cumulative').print_stats(10)

            self.teardown()
        elif self.__backend == "timeit":
            timeit.repeat("function(*args, **kwargs)", setup="self.setup()")

            self.teardown()
        else:
            print("There is no available backend")


class MemoryBenchmark:
    """
    A class for memory profiling and benchmarking.

    Supports multiple backends for memory measurement including memray
    and memory_profiler with various configuration options.

    Parameters
    ----------
    backend : str, optional
        The memory profiling backend to use. Options are:
        - "memray": Native memory profiler (default)
        - "memory_profiler": Line-by-line memory profiler
    debug : bool, optional
        Whether to enable debug mode for detailed profiling (default is False).
    output_file : str, optional
        File to save profiling results to (default is None).
    *args
        Additional positional arguments for the backend.
    **kwargs
        Additional keyword arguments for the backend.
    """
    def __init__(self, backend="memray", debug=False, output_file=None, *args, **kwargs):
        """
        Initialize the MemoryBenchmark instance.

        Parameters
        ----------
        backend : str, optional
            The memory profiling backend to use. Options are:
            - "memray": Native memory profiler (default)
            - "memory_profiler": Line-by-line memory profiler
        debug : bool, optional
            Whether to enable debug mode for detailed profiling (default is False).
        output_file : str, optional
            File to save profiling results to (default is None).
        *args
            Additional positional arguments for the backend.
        **kwargs
            Additional keyword arguments for the backend.
        """
        self.__backend = backend
        self.__debug = debug
        self.__output_file = output_file
        self.__args = args
        self.__kwargs = kwargs

    def __enter__(self):
        """
        Enter the memory profiling context manager.

        Returns
        -------
        object
            The memory tracker instance for memray backend.

        Raises
        ------
        Exception
            If the backend does not support context manager usage.
        """
        if self.__backend == "memray" and USE_MEMRAY:
            self.__memray = memray.Tracker(*self.__args, **self.__kwargs)

            return self.__memray.__enter__()
        else:
            raise Exception(f"The backend {self.__backend} does not support context "
                            "manager")

    def __exit__(self, *args, **kwargs):
        """
        Exit the memory profiling context manager.

        Parameters
        ----------
        *args
            Exception type, value, and traceback (if any).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        object
            Result from the underlying tracker's __exit__ method.
        """
        if self.__backend == "memray" and USE_MEMRAY:
            return self.__memray.__exit__(*args, **kwargs)

    def run(self, function, *args, **kwargs):
        """
        Run a function with memory profiling.

        Parameters
        ----------
        function : callable
            The function to profile.
        *args
            Positional arguments to pass to the function.
        **kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        object
            The return value of the profiled function, or memory usage data
            depending on the backend configuration.
        """
        if self.__backend == "memory_profiler" and USE_MEM_PROF:
            if self.__debug:
                # profile = LineProfiler(include_children=True)

                get_prof = partial(LineProfiler, backend=choose_backend("psutil"))
                show_results_bound = partial(
                    show_results, precision=4
                )

                prof = get_prof()
                vals = prof(function)(*args, **kwargs)
                show_results_bound(prof)
            else:
                vals = memory_usage((function, args, kwargs), *self.__args,
                                    **self.__kwargs)

            self.teardown()

            return vals
        elif self.__backend == "memray" and USE_MEMRAY:
            with memray.Tracker(*self.__args, **self.__kwargs):
                ret = function(*args, **kwargs)

            self.teardown()

            return ret
        else:
            print(f"The backend {self.__backend} is not supported")
