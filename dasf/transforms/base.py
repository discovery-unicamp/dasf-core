#!/usr/bin/python3

import inspect
import dask.array as da

from dasf.utils.decorators import task_handler
from dasf.utils.types import is_dask_array
from dasf.utils.types import is_dask_dataframe
from dasf.utils.funcs import block_chunk_reduce


class Fit:
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


class FitPredict:
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


class FitTransform:
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


class Predict:
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


class GetParams:
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


class SetParams:
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


class Transform:
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
    ):

        self.function = function
        self.depth = depth
        self.boundary = boundary
        self.trim = trim
        self.output_chunk = output_chunk

        if (
            self.boundary is None
            and self.depth is not None
            or self.boundary is not None
            and self.depth is None
        ):
            raise Exception("Both boundary and depth should be passed "
                            "together")

    def __lazy_transform_generic(self, X, **kwargs):
        drop_axis, new_axis = block_chunk_reduce(X, self.output_chunk)

        if self.depth and self.boundary:
            if self.trim:
                new_data = X.map_overlap(
                    self.function,
                    **kwargs,
                    dtype=X.dtype,
                    depth=self.depth,
                    boundary=self.boundary,
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
                    **kwargs,
                )
        else:
            if is_dask_array(X):
                new_data = X.map_blocks(
                    self.function,
                    dtype=X.dtype,
                    drop_axis=drop_axis,
                    new_axis=new_axis,
                    **kwargs,
                )
            elif is_dask_dataframe(X):
                new_data = X.map_partitions(self.function, **kwargs)

        return new_data

    def _lazy_transform_cpu(self, X, **kwargs):
        return self.__lazy_transform_generic(X, **kwargs)

    def _lazy_transform_gpu(self, X, **kwargs):
        return self.__lazy_transform_generic(X, **kwargs)

    def _transform_cpu(self, X, **kwargs):
        return self.function(X, **kwargs)

    def _transform_gpu(self, X, **kwargs):
        return self.function(X, **kwargs)

    @task_handler
    def transform(self, X, **kwargs):
        ...
