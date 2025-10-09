#!/usr/bin/python3

""" Definition of the generic operators of the pipeline. """

import inspect
from uuid import uuid4

import dask.array as da
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    pass


from dasf.utils.decorators import task_handler
from dasf.utils.funcs import block_chunk_reduce
from dasf.utils.types import (
    is_dask_array,
    is_dask_cpu_dataframe,
    is_dask_dataframe,
    is_dask_gpu_dataframe,
)


class Operator:
    """
    Class representing a generic Operator of the pipeline.
    """

    def get_uuid(self):
        """
        Return the UUID representation of the Operator.
        """
        if not hasattr(self, "_uuid"):
            self._uuid = uuid4()
        return self._uuid


class Fit(Operator):
    """
    Class representing a Fit operation of the pipeline.
    """

    def _lazy_fit_cpu(self, X, y=None, **kwargs):
        """
        Respective lazy fit mocked function for CPUs.
        """
        raise NotImplementedError

    def _lazy_fit_gpu(self, X, y=None, **kwargs):
        """
        Respective lazy fit mocked function for GPUs.
        """
        raise NotImplementedError

    def _fit_cpu(self, X, y=None, **kwargs):
        """
        Respective immediate fit mocked function for local CPU(s).
        """
        raise NotImplementedError

    def _fit_gpu(self, X, y=None, **kwargs):
        """
        Respective immediate fit mocked function for local GPU(s).
        """
        raise NotImplementedError

    @task_handler
    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        Generic fit funtion according executor.
        """
        ...

    @staticmethod
    def fit_from_model(model, X, y, sample_weight=None, **kwargs):
        """
        Return the model of a previous created object.
        """
        return model.fit(X=X, y=y, sample_weight=sample_weight, **kwargs)


class FitPredict(Operator):
    """
    Class representing a Fit with Predict operation of the pipeline.
    """

    def _lazy_fit_predict_cpu(self, X, y=None, **kwargs):
        """
        Respective lazy fit with predict mocked function for CPUs.
        """
        raise NotImplementedError

    def _lazy_fit_predict_gpu(self, X, y=None, **kwargs):
        """
        Respective lazy fit with predict mocked function for GPUs.
        """
        raise NotImplementedError

    def _fit_predict_cpu(self, X, y=None, **kwargs):
        """
        Respective immediate fit with predict mocked function for local CPU(s).
        """
        raise NotImplementedError

    def _fit_predict_gpu(self, X, y=None, **kwargs):
        """
        Respective immediate fit with predict mocked function for local GPU(s).
        """
        raise NotImplementedError

    @task_handler
    def fit_predict(self, X, y=None, **kwargs):
        """
        Generic fit with predict funtion according executor.
        """
        ...

    @staticmethod
    def fit_predict_from_model(model, X, y, sample_weight=None, **kwargs):
        """
        Return the model of a previous created object.
        """
        return model.fit_predict(X=X, y=y, sample_weight=sample_weight,
                                 **kwargs)


class FitTransform(Operator):
    """
    Class representing a Fit with Transform operation of the pipeline.
    """

    def _lazy_fit_transform_cpu(self, X, y=None, **kwargs):
        """
        Respective lazy fit with transform mocked function for CPUs.
        """
        raise NotImplementedError

    def _lazy_fit_transform_gpu(self, X, y=None, **kwargs):
        """
        Respective lazy fit with transform mocked function for GPUs.
        """
        raise NotImplementedError

    def _fit_transform_cpu(self, X, y=None, **kwargs):
        """
        Respective immediate fit with transform mocked function for local CPU(s).
        """
        raise NotImplementedError

    def _fit_transform_gpu(self, X, y=None, **kwargs):
        """
        Respective immediate fit with transform mocked function for local GPU(s).
        """
        raise NotImplementedError

    @task_handler
    def fit_transform(self, X, y=None, **kwargs):
        """
        Generic fit with transform funtion according executor.
        """
        ...

    @staticmethod
    def fit_transform_from_model(model, X, y, sample_weight=None, **kwargs):
        """
        Return the model of a previous created object.
        """
        return model.fit_transform(X=X, y=y, sample_weight=sample_weight,
                                   **kwargs)


class Predict(Operator):
    """
    Class representing a Predict operation of the pipeline.
    """

    def _lazy_predict_cpu(self, X, sample_weight=None, **kwargs):
        """
        Respective lazy predict mocked function for CPUs.
        """
        raise NotImplementedError

    def _lazy_predict_gpu(self, X, sample_weight=None, **kwargs):
        """
        Respective lazy predict mocked function for GPUs.
        """
        raise NotImplementedError

    def _predict_cpu(self, X, sample_weight=None, **kwargs):
        """
        Respective immediate predict mocked function for local CPU(s).
        """
        raise NotImplementedError

    def _predict_gpu(self, X, sample_weight=None, **kwargs):
        """
        Respective immediate predict mocked function for local GPU(s).
        """
        raise NotImplementedError

    @task_handler
    def predict(self, X, sample_weight=None, **kwargs):
        """
        Generic predict funtion according executor.
        """
        ...

    @staticmethod
    def predict_from_model(model, X, sample_weight=None, **kwargs):
        """
        Return the model of a previous created object.
        """
        if 'sample_weight' not in inspect.signature(model.predict).parameters:
            return model.predict(X=X, **kwargs)
        return model.predict(X=X, sample_weight=sample_weight, **kwargs)


class GetParams(Operator):
    """
    Class representing a Get Parameters operation of the pipeline.
    """

    def _lazy_get_params_cpu(self, deep=True, **kwargs):
        """
        Respective lazy get_params mocked function for CPUs.
        """
        raise NotImplementedError

    def _lazy_get_params_gpu(self, deep=True, **kwargs):
        """
        Respective lazy get_params mocked function for GPUs.
        """
        raise NotImplementedError

    def _get_params_cpu(self, deep=True, **kwargs):
        """
        Respective immediate get_params mocked function for local CPU(s).
        """
        raise NotImplementedError

    def _get_params_gpu(self, deep=True, **kwargs):
        """
        Respective immediate get_params mocked function for local GPU(s).
        """
        raise NotImplementedError

    @task_handler
    def get_params(self, deep=True, **kwargs):
        """
        Generic get_params funtion according executor.
        """
        ...


class SetParams(Operator):
    """
    Class representing a Set Parameters operation of the pipeline.
    """

    def _lazy_set_params_cpu(self, **params):
        """
        Respective lazy set_params mocked function for CPUs.
        """
        raise NotImplementedError

    def _lazy_set_params_gpu(self, **params):
        """
        Respective lazy set_params mocked function for GPUs.
        """
        raise NotImplementedError

    def _set_params_cpu(self, **params):
        """
        Respective immediate set_params mocked function for local CPU(s).
        """
        raise NotImplementedError

    def _set_params_gpu(self, **params):
        """
        Respective immediate set_params mocked function for local GPU(s).
        """
        raise NotImplementedError

    @task_handler
    def set_params(self, **params):
        """
        Generic set_params funtion according executor.
        """
        ...


class Transform(Operator):
    """
    Class representing a Transform operation of the pipeline.
    """

    def _lazy_transform_cpu(self, X, **kwargs):
        """
        Respective lazy transform mocked function for CPUs.
        """
        raise NotImplementedError

    def _lazy_transform_gpu(self, X, **kwargs):
        """
        Respective lazy transform mocked function for GPUs.
        """
        raise NotImplementedError

    def _transform_cpu(self, X, **kwargs):
        """
        Respective immediate transform mocked function for local CPU(s).
        """
        raise NotImplementedError

    def _transform_gpu(self, X, **kwargs):
        """
        Respective immediate transform mocked function for local GPU(s).
        """
        raise NotImplementedError

    @task_handler
    def transform(self, X, **kwargs):
        """
        Generic transform funtion according executor.
        """
        ...

    @staticmethod
    def transform_from_model(model, X, **kwargs):
        """
        Return the model of a previous created object.
        """
        return model.transform(X=X, **kwargs)


class TargeteredTransform(Transform):
    """
    Class representing a Targetered Transform operation of the pipeline.

    This specific transform operates according the parameters of the
    constructor.

    Parameters
    ----------
    run_local : bool
        Define that the operator will run locally and not distributed.
    run_gpu : bool
        Define if the operator will use GPU(s) or not.

    """

    def __init__(self, run_local=None, run_gpu=None):
        """ Constructor of the class TargeteredTransform. """
        super().__init__()

        self._run_local = run_local
        self._run_gpu = run_gpu


class MappedTransform(Transform):
    """Class representing a MappedTransform based on Transform
    object.

    This object refers to any operation that can be done in blocks.
    In special, for Dask chunks. There are several ways of doing
    that. This class tries to simplify how the functions are applied
    into a block.

    Parameters
    ----------
    function : Callable
        A function that will be applied in a block.
    depth : tuple
        The value of the boundary elements per axis (the default is
        None).
    boundary : str
        The type of the boundary. See Dask boundaries for more
        examples (the default is None).
    trim : bool
        Option to trim the data after an overlap (the default is
        True).
    output_chunk : tuple
        New shape of the output after computing the function (the
        default is None).
    drop_axis : tuple
        Which axis should be deleted after computing the function
        (the default is None).
    new_axis : tuple
        Which axis represent a new axis after computing the function
        (the default is None).

    """

    def __init__(
        self,
        function,
        depth=None,
        boundary=None,
        trim=True,
        output_chunk=None,
        drop_axis=None,
        new_axis=None,
    ):

        self.function = function
        self.depth = depth
        self.boundary = boundary
        self.trim = trim
        self.output_chunk = output_chunk
        self.drop_axis = drop_axis
        self.new_axis = new_axis

        if (
            self.boundary is None
            and self.depth is not None
            or self.boundary is not None
            and self.depth is None
        ):
            raise Exception("Both boundary and depth should be passed "
                            "together")

    def __lazy_transform_generic(self, X, xp, **kwargs):
        if self.drop_axis is not None or self.new_axis is not None:
            drop_axis, new_axis = self.drop_axis, self.new_axis
        else:
            drop_axis, new_axis = block_chunk_reduce(X, self.output_chunk)

        if self.output_chunk is None:
            __output_chunk = X.chunks
        else:
            __output_chunk = self.output_chunk

        if self.depth and self.boundary:
            if self.trim:
                new_data = X.map_overlap(
                    self.function,
                    **kwargs,
                    dtype=X.dtype,
                    depth=self.depth,
                    boundary=self.boundary,
                    meta=xp.array(()),
                )
            else:
                data_blocks = da.overlap.overlap(
                    X, depth=self.depth, boundary=self.boundary
                )

                new_data = data_blocks.map_blocks(
                    self.function,
                    dtype=X.dtype,
                    drop_axis=drop_axis,
                    new_axis=new_axis,
                    chunks=__output_chunk,
                    meta=xp.array(()),
                    **kwargs,
                )
        else:
            if is_dask_array(X):
                new_data = X.map_blocks(
                    self.function,
                    dtype=X.dtype,
                    drop_axis=drop_axis,
                    new_axis=new_axis,
                    chunks=__output_chunk,
                    meta=xp.array(()),
                    **kwargs,
                )
            elif is_dask_dataframe(X):
                new_data = X.map_partitions(self.function, **kwargs)

        return new_data

    def _lazy_transform_cpu(self, X, **kwargs):
        return self.__lazy_transform_generic(X, xp=np, **kwargs)

    def _lazy_transform_gpu(self, X, **kwargs):
        return self.__lazy_transform_generic(X, xp=cp, **kwargs)

    def _transform_cpu(self, X, **kwargs):
        return self.function(X, **kwargs)

    def _transform_gpu(self, X, **kwargs):
        return self.function(X, **kwargs)

    @task_handler
    def transform(self, X, **kwargs):
        ...


class ReductionTransform(Transform):
    """Class representing a Reduction based on Transform
    object.

    This is a simple MapReduction operation using Dask.

    Parameters
    ----------
    output_size : tuple
        The size of the new output.
    func_aggregate : Callable
        The function called to aggregate the result of each chunk.
    func_chunk : Callable
        The function applied in each chunk.
    func_combine : Callable
        The function to combine each reduction of aggregate (the
        default is None).

    """
    def __init__(self, output_size, func_aggregate, func_chunk, func_combine=None):
        self.output_size = output_size

        self.func_aggregate = func_aggregate
        self.func_chunk = func_chunk
        self.func_combine = func_combine

    def _operation_aggregate_cpu(self, block, axis=None, keepdims=False):
        return self.func_aggregate(block, axis, keepdims, xp=np)

    def _operation_aggregate_gpu(self, block, axis=None, keepdims=False):
        return self.func_aggregate(block, axis, keepdims, xp=cp)

    def _operation_combine_cpu(self, block, axis=None, keepdims=False):
        return self.func_combine(block, axis, keepdims, xp=np)

    def _operation_combine_gpu(self, block, axis=None, keepdims=False):
        return self.func_combine(block, axis, keepdims, xp=cp)

    def _operation_chunk_cpu(self, block, axis=None, keepdims=False):
        return self.func_chunk(block, axis, keepdims, xp=np)

    def _operation_chunk_gpu(self, block, axis=None, keepdims=False):
        return self.func_chunk(block, axis, keepdims, xp=cp)

    def _lazy_transform_cpu(self, X, *args, **kwargs):
        if self.func_combine is not None:
            kwargs['combine'] = self._operation_combine_cpu

        if is_dask_cpu_dataframe(X):
            return X.reduction(chunk=self._operation_chunk_cpu,
                               aggregate=self._operation_aggregate_cpu,
                               meta=self.output_size,
                               *args,
                               **kwargs)
        else:
            return da.reduction(X,
                                chunk=self._operation_chunk_cpu,
                                aggregate=self._operation_aggregate_cpu,
                                dtype=X.dtype,
                                meta=np.array(self.output_size,
                                              dtype=X.dtype),
                                *args,
                                **kwargs)

    def _lazy_transform_gpu(self, X, *args, **kwargs):
        if self.func_combine is not None:
            kwargs['combine'] = self._operation_combine_gpu

        if is_dask_gpu_dataframe(X):
            return X.reduction(chunk=self._operation_chunk_gpu,
                               aggregate=self._operation_aggregate_gpu,
                               meta=self.output_size,
                               *args,
                               **kwargs)
        else:
            return da.reduction(X,
                                chunk=self._operation_chunk_gpu,
                                aggregate=self._operation_aggregate_gpu,
                                dtype=X.dtype,
                                meta=cp.array(self.output_size,
                                              dtype=X.dtype),
                                *args,
                                **kwargs)

    def _transform_cpu(self, X, *args, **kwargs):
        return self.func_chunk(block=X, xp=np, *args, **kwargs)

    def _transform_gpu(self, X, *args, **kwargs):
        return self.func_chunk(block=X, xp=cp, *args, **kwargs)

    @task_handler
    def transform(self, X, *args, **kwargs):
        ...
