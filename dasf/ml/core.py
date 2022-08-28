#!/usr/bin/env python3

import os
import pickle

from pathlib import Path

from dasf.pipeline import Operator


class ML(Operator):
    def __init__(self, name, checkpoint=False, method=None, **kwargs):
        # Machine Learning Algorithm Fit
        super().__init__(name=name, checkpoint=checkpoint, **kwargs)

        self._cached_dir = os.path.abspath(str(Path.home()) +
                                           "/.cache/dasf/ml/")
        os.makedirs(self._cached_dir, exist_ok=True)

        if method is None:
            self._tmp = os.path.abspath(self._cached_dir + "/" +
                                        type(self).__name__.lower())
        else:
            self._tmp = os.path.abspath(self._cached_dir + "/" +
                                        method)

        self.__checkpoint = checkpoint

    def dump(self, model):
        print(type(model))
        with open(self._tmp, "wb") as fh:
            pickle.dump(model, fh)

    def load(self, model):
        with open(self._tmp, "rb") as fh:
            return pickle.load(fh)
        return None


class FitInternal(ML):
    def __init__(self, name, checkpoint=False, method=None, **kwargs):
        # Machine Learning Algorithm Fit
        super().__init__(name=name, checkpoint=checkpoint,
                         method=method, **kwargs)

    def run(self, model, X, y=None, sample_weight=None):
        if self.get_checkpoint() and os.path.exists(self._tmp):
            model = self.load(model)

        result = model.fit(X, y)

        if self.get_checkpoint():
            self.dump(result)

        return result


class FitPredictInternal(ML):
    def __init__(self, name, checkpoint=False, method=None, **kwargs):
        # Machine Learning Algorithm Fit Predict
        super().__init__(name=name, checkpoint=checkpoint,
                         method=method, **kwargs)

    def run(self, model, X, y=None, sample_weight=None):
        if self.get_checkpoint() and os.path.exists(self._tmp):
            model = self.load(model)

        result = model.fit_predict(X, y, sample_weight)

        return result


class FitTransformInternal(ML):
    def __init__(self, name, checkpoint=False, method=None, **kwargs):
        # Machine Learning Algorithm Fit Transform
        super().__init__(name=name, checkpoint=checkpoint,
                         method=method, **kwargs)

    def run(self, model, X, y=None):
        if self.get_checkpoint() and os.path.exists(self._tmp):
            model = self.load(model)

        result = model.fit_transform(X, y)

        if self.get_checkpoint():
            self.dump(result)

        return result


class PredictInternal(ML):
    def __init__(self, name, checkpoint=False, method=None, **kwargs):
        # Machine Learning Algorithm Predict
        super().__init__(name=name, checkpoint=checkpoint,
                         method=method, **kwargs)

    def run(self, model, X, sample_weight=None):
        if self.get_checkpoint() and os.path.exists(self._tmp):
            model = self.load(model)

        return model.predict(X)


class TransformInternal(ML):
    def __init__(self, name, checkpoint=False, method=None, **kwargs):
        # Machine Learning Algorithm Predict
        super().__init__(name=name, checkpoint=checkpoint,
                         method=method, **kwargs)

    def run(self, model, X, copy=False):
        return model.transform(X, copy)
