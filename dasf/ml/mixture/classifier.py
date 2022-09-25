#!/usr/bin/env python3

from dasf.transforms import Fit
from dasf.transforms import FitPredict
from dasf.transforms import FitTransform
from dasf.transforms import GetParams
from dasf.transforms import SetParams


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
