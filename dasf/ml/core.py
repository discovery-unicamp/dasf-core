#!/usr/bin/env python3

import os
import pickle

from pathlib import Path

from dasf.pipeline import Operator


class MLGeneric(Operator):
    def __init__(self, name, checkpoint=False, **kwargs):
        # Machine Learning Algorithm
        super().__init__(name=name, checkpoint=checkpoint, **kwargs)

        self._cached_dir = os.path.abspath(str(Path.home()) +
                                           "/.cache/dasf/ml/")
        os.makedirs(self._cached_dir, exist_ok=True)

        self._tmp = os.path.abspath(self._cached_dir + "/" +
                                    name.lower())

        self.__checkpoint = checkpoint

    def dump(self, model):
        if self.get_checkpoint():
            with open(self._tmp, "wb") as fh:
                pickle.dump(model, fh)

    def load(self, model):
        if self.get_checkpoint() and os.path.exists(self._tmp):
            with open(self._tmp, "rb") as fh:
                return pickle.load(fh)
        return model


class FitInternal(MLGeneric):
    def __init__(self, name, checkpoint=False, **kwargs):
        # Machine Learning Algorithm Fit
        super().__init__(name=name, checkpoint=checkpoint, **kwargs)

    def run_generic(self, func, model, X, y=None, sample_weight=None):
        model = self.load(model)

        if hasattr(model, func):
            result = getattr(model, func)(X, y, sample_weight)
        else:
            result = model.fit(X, y, sample_weight)

        self.dump(model)

        return result

    def run_lazy_cpu(self, model, X, y=None, sample_weight=None):
        return self.run_generic(model=model, X=X, y=y,
                                sample_weight=sample_weight,
                                func="_lazy_fit_cpu")

    def run_cpu(self, model, X, y=None, sample_weight=None):
        return self.run_generic(model=model, X=X, y=y,
                                sample_weight=sample_weight,
                                func="_fit_cpu")

    def run_lazy_gpu(self, model, X, y=None, sample_weight=None):
        return self.run_generic(model=model, X=X, y=y,
                                sample_weight=sample_weight,
                                func="_lazy_fit_gpu")

    def run_gpu(self, model, X, y=None, sample_weight=None):
        return self.run_generic(model=model, X=X, y=y,
                                sample_weight=sample_weight,
                                func="_fit_gpu")


class FitPredictInternal(MLGeneric):
    def __init__(self, name, checkpoint=False, **kwargs):
        # Machine Learning Algorithm Fit Predict
        super().__init__(name=name, checkpoint=checkpoint, **kwargs)

    def run_generic(self, func, model, X, y=None, sample_weight=None):
        model = self.load(model)

        if hasattr(model, func):
            return getattr(model, func)(X, y, sample_weight)
        else:
            return model.fit_predict(X, y, sample_weight)

    def run_lazy_cpu(self, model, X, y=None, sample_weight=None):
        return self.run_generic(model=model, X=X, y=y,
                                sample_weight=sample_weight,
                                func="_lazy_fit_predict_cpu")

    def run_cpu(self, model, X, y=None, sample_weight=None):
        return self.run_generic(model=model, X=X, y=y,
                                sample_weight=sample_weight,
                                func="_fit_predict_cpu")

    def run_lazy_gpu(self, model, X, y=None, sample_weight=None):
        return self.run_generic(model=model, X=X, y=y,
                                sample_weight=sample_weight,
                                func="_lazy_fit_predict_gpu")

    def run_gpu(self, model, X, y=None, sample_weight=None):
        return self.run_generic(model=model, X=X, y=y,
                                sample_weight=sample_weight,
                                func="_fit_predict_gpu")


class FitTransformInternal(MLGeneric):
    def __init__(self, name, checkpoint=False, **kwargs):
        # Machine Learning Algorithm Fit Transform
        super().__init__(name=name, checkpoint=checkpoint, **kwargs)

    def run_generic(self, func, model, X, y=None):
        model = self.load(model)

        if hasattr(model, func):
            result = getattr(model, func)(X, y)
        else:
            result = model.fit_transform(X, y)

        self.dump(result)

        return result

    def run_lazy_cpu(self, model, X, y=None):
        return self.run_generic(model=model, X=X, y=y,
                                func="_lazy_fit_transform_cpu")

    def run_cpu(self, model, X, y=None):
        return self.run_generic(model=model, X=X, y=y,
                                func="_fit_transform_cpu")

    def run_lazy_gpu(self, model, X, y=None):
        return self.run_generic(model=model, X=X, y=y,
                                func="_lazy_fit_transform_gpu")

    def run_gpu(self, model, X, y=None):
        return self.run_generic(model=model, X=X, y=y,
                                func="_fit_transform_gpu")


class PredictInternal(MLGeneric):
    def __init__(self, name, checkpoint=False, **kwargs):
        # Machine Learning Algorithm Predict
        super().__init__(name=name, checkpoint=checkpoint, **kwargs)

    def run_generic(self, func, model, X, sample_weight=None):
        model = self.load(model)

        if hasattr(model, func):
            return getattr(model, func)(X, sample_weight)
        else:
            return model.predict(X, sample_weight)

    def run_lazy_cpu(self, model, X, sample_weight=None):
        return self.run_generic(model=model, X=X, sample_weight=sample_weight,
                                func="_lazy_predict_cpu")

    def run_cpu(self, model, X, sample_weight=None):
        return self.run_generic(model=model, X=X, sample_weight=sample_weight,
                                func="_predict_cpu")

    def run_lazy_gpu(self, model, X, sample_weight=None):
        return self.run_generic(model=model, X=X, sample_weight=sample_weight,
                                func="_lazy_predict_gpu")

    def run_gpu(self, model, X, sample_weight=None):
        return self.run_generic(model=model, X=X, sample_weight=sample_weight,
                                func="_fit_predict_gpu")


class TransformInternal(MLGeneric):
    def __init__(self, name, checkpoint=False, **kwargs):
        # Machine Learning Algorithm Predict
        super().__init__(name=name, checkpoint=checkpoint, **kwargs)

    def run_generic(self, func, model, X, copy=False):
        if hasattr(model, func):
            return getattr(model, func)(X, copy)
        else:
            return model.transform(X, copy)

    def run_lazy_cpu(self, model, X, copy=False):
        return self.run_generic(model=model, X=X, copy=copy,
                                func="_lazy_transform_cpu")

    def run_cpu(self, model, X, copy=False):
        return self.run_generic(model=model, X=X, copy=copy,
                                func="_transform_cpu")

    def run_lazy_gpu(self, model, X, copy=False):
        return self.run_generic(model=model, X=X, copy=copy,
                                func="_lazy_transform_gpu")

    def run_gpu(self, model, X, copy=False):
        return self.run_generic(model=model, X=X, copy=copy,
                                func="_transform_gpu")
