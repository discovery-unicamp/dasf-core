#!/usr/bin/python3

import inspect
import numpy as np
import dask.array as da
from uuid import uuid4

try:
    import cupy as cp
    import dask_cudf as dcudf
except ImportError: # pragma: no cover
    pass

from dasf.utils.decorators import task_handler
from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_dataframe
from dasf.utils.types import is_dask_cpu_dataframe
from dasf.utils.types import is_dask_gpu_dataframe
from dasf.utils.funcs import block_chunk_reduce

class Operator:
    def get_uuid(self):
        if not hasattr(self, "_uuid"):
            self._uuid = uuid4()
        return self._uuid

class Fit(Operator):
    def _lazy_fit_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _lazy_fit_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    @task_handler
    def fit(self, X, y, sample_weight=None, **kwargs):
        ...

    @staticmethod
    def fit_from_model(model, X, y, sample_weight=None, **kwargs):
        return model.fit(X=X, y=y, sample_weight=sample_weight, **kwargs)


class FitPredict(Operator):
    def _lazy_fit_predict_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _lazy_fit_predict_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_predict_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_predict_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    @task_handler
    def fit_predict(self, X, y=None, **kwargs):
        ...

    @staticmethod
    def fit_predict_from_model(model, X, y, sample_weight=None, **kwargs):
        return model.fit_predict(X=X, y=y, sample_weight=sample_weight,
                                 **kwargs)


class FitTransform(Operator):
    def _lazy_fit_transform_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _lazy_fit_transform_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_transform_cpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    def _fit_transform_gpu(self, X, y=None, **kwargs):
        raise NotImplementedError

    @task_handler
    def fit_transform(self, X, y=None, **kwargs):
        ...

    @staticmethod
    def fit_transform_from_model(model, X, y, sample_weight=None, **kwargs):
        return model.fit_transform(X=X, y=y, sample_weight=sample_weight,
                                   **kwargs)


class Predict(Operator):
    def _lazy_predict_cpu(self, X, sample_weight=None, **kwargs):
        raise NotImplementedError

    def _lazy_predict_gpu(self, X, sample_weight=None, **kwargs):
        raise NotImplementedError

    def _predict_cpu(self, X, sample_weight=None, **kwargs):
        raise NotImplementedError

    def _predict_gpu(self, X, sample_weight=None, **kwargs):
        raise NotImplementedError

    @task_handler
    def predict(self, X, sample_weight=None, **kwargs):
        ...

    @staticmethod
    def predict_from_model(model, X, sample_weight=None, **kwargs):
        if 'sample_weight' not in inspect.signature(model.predict).parameters:
            return model.predict(X=X, **kwargs)
        return model.predict(X=X, sample_weight=sample_weight, **kwargs)


class GetParams(Operator):
    def _lazy_get_params_cpu(self, deep=True, **kwargs):
        raise NotImplementedError

    def _lazy_get_params_gpu(self, deep=True, **kwargs):
        raise NotImplementedError

    def _get_params_cpu(self, deep=True, **kwargs):
        raise NotImplementedError

    def _get_params_gpu(self, deep=True, **kwargs):
        raise NotImplementedError

    @task_handler
    def get_params(self, deep=True, **kwargs):
        ...


class SetParams(Operator):
    def _lazy_set_params_cpu(self, **params):
        raise NotImplementedError

    def _lazy_set_params_gpu(self, **params):
        raise NotImplementedError

    def _set_params_cpu(self, **params):
        raise NotImplementedError

    def _set_params_gpu(self, **params):
        raise NotImplementedError

    @task_handler
    def set_params(self, **params):
        ...


class Transform(Operator):
    def _lazy_transform_cpu(self, X, **kwargs):
        raise NotImplementedError

    def _lazy_transform_gpu(self, X, **kwargs):
        raise NotImplementedError

    def _transform_cpu(self, X, **kwargs):
        raise NotImplementedError

    def _transform_gpu(self, X, **kwargs):
        raise NotImplementedError

    @task_handler
    def transform(self, X, **kwargs):
        ...

    @staticmethod
    def transform_from_model(model, X, **kwargs):
        return model.transform(X=X, **kwargs)


class TargeteredTransform(Transform):
    def __init__(self, run_local=None, run_gpu=None):
        super().__init__()

        self._run_local = run_local
        self._run_gpu = run_gpu


class MappedTransform(Transform):
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
