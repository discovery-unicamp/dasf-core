#!/usr/bin/env python3


class Executor:
    @property
    def is_connected(self) -> bool:
        return False

    @property
    def info(self) -> str:
        return "This executor has no info to show."

    def has_dataset(self, key) -> bool:
        return False

    def register_dataset(self, **kwargs):
        dataset = list(kwargs.values())

        if len(dataset) != 1:
            raise Exception(f"This function requires one dataset only. "
                            "We found {len(dataset)}.")

        return dataset.pop()

    def get_dataset(self, key):
        raise NotImplementedError("This function needs to be specialized for "
                                  "every executor.")

    def register_plugin(self, plugin):
        raise Exception("This executor does not accept plugins.")

    def pre_run(self, pipeline):
        pass

    def post_run(self, pipeline):
        pass

    def execute(self, fn, *args, **kwargs):
        ...

    def shutdown(self):
        pass
