#!/usr/bin/env python3

from dasf.transforms.transforms import _Fit
from dasf.transforms.transforms import _FitPredict
from dasf.transforms.transforms import _FitTransform
from dasf.transforms.transforms import _GetParams
from dasf.transforms.transforms import _SetParams


class MixtureClassifier(_Fit, _FitPredict, _FitTransform,
                        _GetParams, _SetParams):
    def fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def fit_predict(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        raise NotImplementedError

    def get_params(deep=True):
        raise NotImplementedError

    def set_params(**params):
        raise NotImplementedError
