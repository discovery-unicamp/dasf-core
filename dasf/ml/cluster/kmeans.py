#!/usr/bin/env python3

from sklearn.cluster import KMeans as KMeans_CPU
from dask_ml.cluster import KMeans as KMeans_MCPU

from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.utils import is_gpu_supported
from dasf.utils.decorators import task_handler

try:
    from cuml.cluster import KMeans as KMeans_GPU
    from cuml.dask.cluster import KMeans as KMeans_MGPU
except ImportError:
    pass


class KMeans(ClusterClassifier):
    def __init__(self, n_clusters, random_state=0, max_iter=300):

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

        self.__kmeans_cpu = KMeans_CPU(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

        self.__kmeans_mcpu = KMeans_MCPU(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

        if is_gpu_supported():
            self.__kmeans_gpu = KMeans_GPU(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                max_iter=self.max_iter,
            )

            # XXX: KMeans in Multi GPU requires a Client instance,
            # skip if not present.
            try:
                self.__kmeans_mgpu = KMeans_MGPU(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    max_iter=self.max_iter,
                )
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

    def _lazy_predict2_cpu(self, X, sample_weight=None):
        def __predict(block):
            return self._predict_cpu.predict(block, sample_weight=sample_weight)

        return X.map_blocks(
            __predict, chunks=(X.chunks[0],), drop_axis=[1], dtype=X.dtype
        )

    def _lazy_predict2_gpu(self, X, sample_weight=None):
        def __predict(block):
            return self._predict_gpu.predict(block, sample_weight=sample_weight)

        return X.map_blocks(
            __predict, chunks=(X.chunks[0],), drop_axis=[1], dtype=X.dtype
        )

    def _predict2_cpu(self, X, sample_weight=None):
        raise NotImplementedError("Method available only for Dask.")

    def _predict2_gpu(self, X, sample_weight=None):
        raise NotImplementedError("Method available only for Dask.")

    @task_handler
    def predict2(self, sample_weight=None):
        ...
