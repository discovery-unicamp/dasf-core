#!/usr/bin/env python3

import os

from hdbscan import HDBSCAN as HDBSCAN_CPU

from dasf.ml.core import FitInternal, FitPredictInternal
from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.utils import is_gpu_supported
from dasf.utils.generators import generate_fit
from dasf.utils.generators import generate_fit_predict
from dasf.pipeline import ParameterOperator

try:
    from cuml.cluster import HDBSCAN as HDBSCAN_GPU
except ImportError:
    pass


@generate_fit
@generate_fit_predict
class HDBSCAN(ClusterClassifier):
    def __init__(self, alpha=1.0, gen_min_span_tree=False, leaf_size=40,
                 metric='euclidean', min_cluster_size=5, min_samples=None,
                 p=None, **kwargs):

        self.alpha = alpha
        self.gen_min_span_tree = gen_min_span_tree
        self.leaf_size = leaf_size
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.p = p

        self.__hdbscan_cpu = \
            HDBSCAN_CPU(alpha=self.alpha,
                        gen_min_span_tree=self.gen_min_span_tree,
                        leaf_size=self.leaf_size,
                        metric=self.metric,
                        min_cluster_size=self.min_cluster_size,
                        min_samples=self.min_samples,
                        p=self.p)

        if is_gpu_supported():
            self.__hdbscan_gpu = \
                HDBSCAN_GPU(alpha=self.alpha,
                            gen_min_span_tree=self.gen_min_span_tree,
                            metric=self.metric,
                            min_cluster_size=self.min_cluster_size,
                            min_samples=self.min_samples,
                            p=self.p)

    def _fit_cpu(self, X, y=None):
        self.__hdbscan_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None, convert_dtype=True):
        self.__hdbscan_gpu.fit(X=X, y=y, convert_dtype=convert_dtype)

    def _fit_predict_cpu(self, X, y=None):
        self.__hdbscan_cpu.fit_predict(X=X, y=y)

    def _fit_predict_gpu(self, X, y=None):
        self.__hdbscan_gpu.fit_predict(X=X, y=y)


class HDBSCANOperator(ParameterOperator):
    def __init__(self, alpha=1.0, gen_min_span_tree=False, leaf_size=40,
                 metric='euclidean', min_cluster_size=5, min_samples=None,
                 p=None, **kwargs):
        super().__init__(name=type(self).__name__, **kwargs)

        self.alpha = alpha
        self.gen_min_span_tree = gen_min_span_tree
        self.leaf_size = leaf_size
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.p = p

        self.__hdbscan_cpu = \
            HDBSCAN_CPU(alpha=self.alpha,
                        gen_min_span_tree=self.gen_min_span_tree,
                        leaf_size=self.leaf_size,
                        metric=self.metric,
                        min_cluster_size=self.min_cluster_size,
                        min_samples=self.min_samples,
                        p=self.p)

        if is_gpu_supported():
            self.__hdbscan_gpu = \
                HDBSCAN_GPU(alpha=self.alpha,
                            gen_min_span_tree=self.gen_min_span_tree,
                            metric=self.metric,
                            min_cluster_size=self.min_cluster_size,
                            min_samples=self.min_samples,
                            p=self.p)

        # Select CPU as default to initialize the attribute
        self.hdbscan = self.__hdbscan_cpu

        self.fit = HDBSCANFit(**kwargs)
        self.fit_predict = HDBSCANFitPredict(**kwargs)

    def setup_cpu(self, executor):
        self.hdbscan = self.__hdbscan_cpu

    def setup_mcpu(self, executor):
        raise NotImplementedError

    def setup_gpu(self, executor):
        self.hdbscan = self.__hdbscan_gpu

    def setup_mgpu(self, executor):
        raise NotImplementedError

    def run(self):
        return self.hdbscan


class HDBSCANFit(FitInternal):
    def __init__(self, **kwargs):
        super().__init__(name="HDBSCANFit", **kwargs)


class HDBSCANFitPredict(FitPredictInternal):
    def __init__(self, **kwargs):
        super().__init__(name="HDBSCANFitPredict", **kwargs)

    # XXX: Fix issue of missing labels for RAPIDS AI HDBSCAN
    def run(self, model, X, y=None, sample_weight=None):
        if self.get_checkpoint() and os.path.exists(self._tmp):
            model = self.load()

        if y is None:
            result = model.fit_predict(X, sample_weight)
        else:
            result = model.fit_predict(X, y, sample_weight)

        if self.get_checkpoint():
            self.dump(result)

        return result
