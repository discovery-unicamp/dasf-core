#!/usr/bin/env python3

import timeit

from time import perf_counter

try:
    import memray
    USE_MEMRAY = True
except ImportError:
    USE_MEMRAY = False

import cProfile
from pstats import Stats

try:
    from functools import partial
    from memory_profiler import show_results, LineProfiler
    from memory_profiler import memory_usage, choose_backend

    USE_MEM_PROF = True
except ImportError:
    USE_MEM_PROF = False


class TimeBenchmark:
    def __init__(self, backend="cprofile"):
        self.__backend = backend

    def __enter__(self):
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
        if self.__backend == "cprofile":
            self.__pr.disable()
            p = Stats(self.__pr)

            p.strip_dirs().sort_stats('cumulative').print_stats(10)
        elif self.__backend == "perf_counter":
            self.__end = perf_counter()
            print("Time spent:", self.__end - self.__start)

    def run(self, function, *args, **kwargs):
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
    def __init__(self, backend="memray", debug=False, output_file=None, *args, **kwargs):
        self.__backend = backend
        self.__debug = debug
        self.__output_file = output_file
        self.__args = args
        self.__kwargs = kwargs

    def __enter__(self):
        if self.__backend == "memray" and USE_MEMRAY:
            self.__memray = memray.Tracker(*self.__args, **self.__kwargs)

            return self.__memray.__enter__()
        else:
            raise Exception(f"The backend {self.__backend} does not support context "
                            "manager")

    def __exit__(self, *args, **kwargs):
        if self.__backend == "memray" and USE_MEMRAY:
            return self.__memray.__exit__(*args, **kwargs)

    def run(self, function, *args, **kwargs):
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
