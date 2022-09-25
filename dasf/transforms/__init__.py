#!/usr/bin/python3

from dasf.transforms.transforms import Transform  # noqa
from dasf.transforms.transforms import ArraysToDataFrame  # noqa
from dasf.transforms.operations import SliceArray, SliceArrayByPercent  # noqa
from dasf.transforms.operations import Reshape  # noqa
from dasf.transforms.memory import PersistDaskData, LoadDaskData  # noqa
from dasf.utils.decorators import task_handler


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
    def fit_predict(self, X, y, **kwargs):
        ...


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


__all__ = [
    "Transform",
    "ArraysToDataFrame",
    "SliceArray",
    "SliceArrayByPercent",
    "Reshape",
    "PersistDaskData",
    "LoadDaskData",
]
