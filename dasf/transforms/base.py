#!/usr/bin/python3

import inspect

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
        return model.fit_predict(X=X, y=y, sample_weight=sample_weight, **kwargs)


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
        return model.fit_transform(X=X, y=y, sample_weight=sample_weight, **kwargs)


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
        if 'sample_weight' not in inspect.signature(model.predict):
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
