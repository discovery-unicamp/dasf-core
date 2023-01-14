#!/usr/bin/env python3

from dasf.transforms.base import Fit
from dasf.transforms.base import FitPredict
from dasf.transforms.base import FitTransform
from dasf.transforms.base import GetParams
from dasf.transforms.base import SetParams


class MixtureClassifier(Fit, FitPredict, FitTransform,
                        GetParams, SetParams):
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
