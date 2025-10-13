#!/usr/bin/env python3
"""A base module for DASF executors."""


class Executor:
    """This class defines the base API for a pipeline executor."""

    @property
    def is_connected(self) -> bool:
        """Returns true if the executor is connected to a backend."""
        return False

    @property
    def info(self) -> str:
        """Returns a string with the executor information."""
        return "This executor has no info to show."

    def has_dataset(self, key) -> bool:
        """Returns true if a dataset is registered in the executor."""
        return False

    def register_dataset(self, **kwargs):
        """Registers a dataset in the executor.

        Raises
        ------
        Exception
            If more than one dataset is provided.
        """
        dataset = list(kwargs.values())

        if len(dataset) != 1:
            raise Exception("This function requires one dataset only. "
                            f"We found {len(dataset)}.")

        return dataset.pop()

    def get_dataset(self, key):
        """Gets a dataset from the executor.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("This function needs to be specialized for "
                                  "every executor.")

    def register_plugin(self, plugin):
        """Registers a plugin in the executor.

        Raises
        ------
        Exception
            If the executor does not accept plugins.
        """
        raise Exception("This executor does not accept plugins.")

    def pre_run(self, pipeline):
        """Executes before the pipeline starts.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to be executed.
        """
        pass

    def post_run(self, pipeline):
        """Executes after the pipeline finishes.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline that was executed.
        """
        pass

    def execute(self, fn, *args, **kwargs):
        """Executes a function in the executor.

        Parameters
        ----------
        fn : function
            The function to be executed.
        args : list
            The arguments of the function.
        kwargs : dict
            The keyword arguments of the function.
        """
        ...

    def shutdown(self):
        """Shutdowns the executor."""
        pass
