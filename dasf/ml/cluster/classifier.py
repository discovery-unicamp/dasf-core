#!/usr/bin/env python3


class ClusterClassifier:
    def fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def fit_predict(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        raise NotImplementedError

    def predict(self, X, sample_weight=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def get_params(deep=True):
        raise NotImplementedError

    def set_params(**params):
        raise NotImplementedError
