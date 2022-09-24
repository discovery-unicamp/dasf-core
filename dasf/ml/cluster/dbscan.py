#!/usr/bin/env python3

from sklearn.cluster import DBSCAN as DBSCAN_CPU

from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.utils import is_gpu_supported

try:
    from cuml.cluster import DBSCAN as DBSCAN_GPU
    from cuml.dask.cluster import DBSCAN as DBSCAN_MGPU
except ImportError:
    pass


class DBSCAN(ClusterClassifier):
    def __init__(self, eps=0.5, leaf_size=40, metric="euclidean",
                 min_samples=None, p=None):

        self.eps = eps
        self.leaf_size = leaf_size
        self.metric = metric
        self.min_samples = min_samples
        self.p = p

        self.__dbscan_cpu = DBSCAN_CPU(
            eps=self.eps,
            leaf_size=self.leaf_size,
            metric=self.metric,
            min_samples=self.min_samples,
            p=self.p,
        )

        if is_gpu_supported():
            self.__dbscan_gpu = DBSCAN_GPU(
                eps=self.eps,
                leaf_size=self.leaf_size,
                metric=self.metric,
                min_samples=self.min_samples,
                p=self.p,
            )

            self.__dbscan_mgpu = DBSCAN_MGPU(
                eps=self.eps,
                leaf_size=self.leaf_size,
                metric=self.metric,
                min_samples=self.min_samples,
                p=self.p,
            )

    def _lazy_fit_gpu(self, X, y=None, out_dtype="int32"):
        self.__dbscan_mgpu.fit(X=X, out_dtype=out_dtype)

    def _fit_cpu(self, X, y=None, sample_weight=None):
        self.__dbscan_cpu.fit(X=X, y=y, sample_weight=sample_weight)

    def _fit_gpu(self, X, y=None, out_dtype="int32"):
        self.__dbscan_gpu.fit(X=X, out_dtype=out_dtype)

    def _lazy_fit_predict_gpu(self, X, y=None, out_dtype="int32"):
        self.__dbscan_mgpu.fit_predict(X=X, out_dtype=out_dtype)

    def _fit_predict_cpu(self, X, y=None, sample_weight=None):
        self.__dbscan_cpu.fit_predict(X=X, y=y, sample_weight=sample_weight)

    def _fit_predict_gpu(self, X, y=None, out_dtype="int32"):
        self.__dbscan_gpu.fit_predict(X=X, out_dtype=out_dtype)
