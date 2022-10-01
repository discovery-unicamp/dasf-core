#!/usr/bin/env python3

# XXX: Import due to CVE-2022-21797
import joblib
from packaging import version

from hdbscan import HDBSCAN as HDBSCAN_CPU

from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.utils import is_gpu_supported

try:
    from cuml.cluster import HDBSCAN as HDBSCAN_GPU
except ImportError:
    pass


class HDBSCAN(ClusterClassifier):
    def __init__(
        self,
        alpha=1.0,
        gen_min_span_tree=False,
        leaf_size=40,
        metric="euclidean",
        min_cluster_size=5,
        min_samples=None,
        p=None,
        **kwargs
    ):

        self.alpha = alpha
        self.gen_min_span_tree = gen_min_span_tree
        self.leaf_size = leaf_size
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.p = p

        if version.parse(joblib.__version__) >= version.parse("1.2.0"):
            self.__hdbscan_cpu = HDBSCAN_CPU(
                alpha=self.alpha,
                gen_min_span_tree=self.gen_min_span_tree,
                leaf_size=self.leaf_size,
                metric=self.metric,
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                p=self.p,
            )

            if is_gpu_supported():
                self.__hdbscan_gpu = HDBSCAN_GPU(
                    alpha=self.alpha,
                    gen_min_span_tree=self.gen_min_span_tree,
                    metric=self.metric,
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    p=self.p,
                )
        else:
            self.__hdbscan_cpu = None
            self.__hdbscan_gpu = None

    def _fit_cpu(self, X, y=None):
        if not self.__hdbscan_cpu:
            raise NotImplementedError
        self.__hdbscan_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None, convert_dtype=True):
        if not self.__hdbscan_gpu:
            raise NotImplementedError
        self.__hdbscan_gpu.fit(X=X, y=y, convert_dtype=convert_dtype)

    def _fit_predict_cpu(self, X, y=None):
        if not self.__hdbscan_cpu:
            raise NotImplementedError
        self.__hdbscan_cpu.fit_predict(X=X, y=y)

    def _fit_predict_gpu(self, X, y=None):
        if not self.__hdbscan_gpu:
            raise NotImplementedError
        self.__hdbscan_gpu.fit_predict(X=X, y=y)
