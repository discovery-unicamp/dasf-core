#!/usr/bin/env python3


import os
import pickle

from pathlib import Path

from sklearn.cluster import KMeans as KMeans_CPU
from dask_ml.cluster import KMeans as KMeans_MCPU

from dasf.ml.core import FitInternal, FitPredictInternal, PredictInternal
from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.utils import is_gpu_supported
from dasf.utils.generators import generate_fit
from dasf.utils.generators import generate_predict
from dasf.utils.generators import generate_fit_predict
from dasf.pipeline import ParameterOperator
from dasf.pipeline import Operator

try:
    from cuml.cluster import KMeans as KMeans_GPU
    from cuml.dask.cluster import KMeans as KMeans_MGPU
except ImportError:
    pass


@generate_fit
@generate_predict
@generate_fit_predict
class KMeans(ClusterClassifier):
    def __init__(self, n_clusters, random_state=0, max_iter=300):

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

        self.__kmeans_cpu = KMeans_CPU(n_clusters=self.n_clusters,
                                       random_state=self.random_state,
                                       max_iter=self.max_iter)

        self.__kmeans_mcpu = KMeans_MCPU(n_clusters=self.n_clusters,
                                         random_state=self.random_state,
                                         max_iter=self.max_iter)

        if is_gpu_supported():
            self.__kmeans_gpu = KMeans_GPU(n_clusters=self.n_clusters,
                                           random_state=self.random_state,
                                           max_iter=self.max_iter)

            # XXX: KMeans in Multi GPU requires a Client instance,
            # skip if not present.
            try:
                self.__kmeans_mgpu = \
                    KMeans_MGPU(n_clusters=self.n_clusters,
                                random_state=self.random_state,
                                max_iter=self.max_iter)
            except ValueError:
                self.__kmeans_mgpu = None

    def _lazy_fit_cpu(self, X, y=None, sample_weight=None):
        return self.__kmeans_mcpu.fit(X=X, y=y)

    def _lazy_fit_gpu(self, X, y=None, sample_weight=None):
        if self.__kmeans_mgpu is None:
            raise NotImplementedError
        return self.__kmeans_mgpu.fit(X=X, sample_weight=sample_weight)

    def _fit_cpu(self, X, y=None, sample_weight=None):
        return self.__kmeans_cpu.fit(X=X, y=y, sample_weight=sample_weight)

    def _fit_gpu(self, X, y=None, sample_weight=None):
        return self.__kmeans_gpu.fit(X=X, sample_weight=sample_weight)

    def _lazy_fit_predict_cpu(self, X, y=None, sample_weight=None):
        return self.__kmeans_mcpu.fit_predict(X, y, sample_weight)

    def _lazy_fit_predict_gpu(self, X, y=None, sample_weight=None):
        if self.__kmeans_mgpu is None:
            raise NotImplementedError
        return self.__kmeans_mgpu.fit_predict(X, y, sample_weight)

    def _fit_predict_cpu(self, X, y=None, sample_weight=None):
        return self.__kmeans_cpu.fit_predict(X, y, sample_weight)

    def _fit_predict_gpu(self, X, y=None, sample_weight=None):
        return self.__kmeans_gpu.fit_predict(X, y, sample_weight)

    def _lazy_predict_cpu(self, X, sample_weight=None):
        return self.__kmeans_mcpu.predict(X)

    def _lazy_predict_gpu(self, X, sample_weight=None):
        if self.__kmeans_mgpu is None:
            raise NotImplementedError
        return self.__kmeans_mgpu.predict(X, sample_weight)

    def _predict_cpu(self, X, sample_weight=None):
        return self.__kmeans_cpu.predict(X, sample_weight)

    def _predict_gpu(self, X, sample_weight=None):
        return self.__kmeans_gpu.predict(X, sample_weight)


class KMeansOperator(ParameterOperator):
    def __init__(self, n_clusters, random_state=0, max_iter=300,
                 checkpoint=False):
        super().__init__(name=type(self).__name__)

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

        self.checkpoint = checkpoint

        self.__kmeans_cpu = KMeans_CPU(n_clusters=self.n_clusters,
                                       random_state=self.random_state,
                                       max_iter=self.max_iter)

        self.__kmeans_mcpu = KMeans_MCPU(n_clusters=self.n_clusters,
                                         random_state=self.random_state,
                                         max_iter=self.max_iter)

        if is_gpu_supported():
            self.__kmeans_gpu = KMeans_GPU(n_clusters=self.n_clusters,
                                           random_state=self.random_state,
                                           max_iter=self.max_iter)

            self.__kmeans_mgpu = KMeans_MGPU(n_clusters=self.n_clusters,
                                             random_state=self.random_state,
                                             max_iter=self.max_iter)

        self.kmeans = None

        self.fit = KMeansFit(checkpoint=self.checkpoint)
        self.predict = KMeansPredict(checkpoint=self.checkpoint)
        self.predict2 = KMeansPredict2(checkpoint=self.checkpoint)
        self.fit_predict = KMeansFitPredict(checkpoint=self.checkpoint)

    def setup_cpu(self, executor):
        self.kmeans = self.__kmeans_cpu

    def setup_mcpu(self, executor):
        self.kmeans = self.__kmeans_mcpu

    def setup_gpu(self, executor):
        self.kmeans = self.__kmeans_gpu

    def setup_mgpu(self, executor):
        self.kmeans = self.__kmeans_mgpu

    def run(self):
        return self.kmeans


class KMeansFit(FitInternal):
    def __init__(self, checkpoint=False):
        super().__init__(name="KMeansFit", checkpoint=checkpoint,
                         method="kmeans")

    def dump(self, model):
        with open(self._tmp, "wb") as fh:
            pickle.dump(model.cluster_centers_, fh)

    def load(self, model):
        with open(self._tmp, "rb") as fh:
            model.cluster_centers_ = pickle.load(fh)

        return model


class KMeansFitPredict(FitPredictInternal):
    def __init__(self, checkpoint=False):
        super().__init__(name="KMeansFitPredict", checkpoint=checkpoint,
                         method="kmeans")

    def load(self, model):
        with open(self._tmp, "rb") as fh:
            model.cluster_centers_ = pickle.load(fh)
        return model


class KMeansPredict(PredictInternal):
    def __init__(self, checkpoint=False):
        super().__init__(name="KMeansPredict", checkpoint=checkpoint,
                         method="kmeans")

    def load(self, model):
        with open(self._tmp, "rb") as fh:
            model.cluster_centers_ = pickle.load(fh)
        return model


class KMeansPredict2(Operator):
    def __init__(self, checkpoint=False):
        super().__init__(name="KMeansPredict2", checkpoint=checkpoint)

        self._cached_dir = os.path.abspath(str(Path.home()) +
                                           "/.cache/dasf/ml/")
        os.makedirs(self._cached_dir, exist_ok=True)

        self._tmp = os.path.abspath(self._cached_dir + "/kmeans")

        self.__checkpoint = checkpoint

    def load(self, model):
        with open(self._tmp, "rb") as fh:
            model.cluster_centers_ = pickle.load(fh)
        return model

    def run(self, model, X):
        if self.get_checkpoint() and os.path.exists(self._tmp):
            model = self.load(model)

        def __predict(block, kmeans_model):
            return kmeans_model.predict(block)

        return X.map_blocks(__predict, model, chunks=(X.chunks[0], ),
                            drop_axis=[1], dtype=X.dtype)
