#!/usr/bin/env python3


class Executor:
    @property
    def ngpus(self) -> int:
        return 0

    @property
    def is_connected(self) -> bool:
        return False

    def pre_run(self, pipeline):
        pass

    def post_run(self, pipeline):
        pass

    def execute(self, fn, *args, **kwargs):
        ...
