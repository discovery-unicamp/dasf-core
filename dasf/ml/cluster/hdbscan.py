#!/usr/bin/env python3

from hdbscan import HDBSCAN as HDBSCAN_CPU

from dasf.ml.core import FitInternal, FitPredictInternal
from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.utils import is_gpu_supported
from dasf.pipeline import ParameterOperator

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

    def _fit_cpu(self, X, y=None):
        self.__hdbscan_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None, convert_dtype=True):
        self.__hdbscan_gpu.fit(X=X, y=y, convert_dtype=convert_dtype)

    def _fit_predict_cpu(self, X, y=None):
        self.__hdbscan_cpu.fit_predict(X=X, y=y)

    def _fit_predict_gpu(self, X, y=None):
        self.__hdbscan_gpu.fit_predict(X=X, y=y)


class HDBSCANOp(ParameterOperator):
    def __init__(
        self,
        alpha=1.0,
        gen_min_span_tree=False,
        leaf_size=40,
        metric="euclidean",
        min_cluster_size=5,
        min_samples=None,
        p=None,
        checkpoint=False,
        **kwargs
    ):
        super().__init__(name="HDBSCAN")

        self._operator = HDBSCAN(
            alpha=alpha,
            gen_min_span_tree=gen_min_span_tree,
            leaf_size=leaf_size,
            metric=metric,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            p=p,
            **kwargs
        )

        self.fit = HDBSCANFitOp(checkpoint=checkpoint, **kwargs)
        self.fit_predict = HDBSCANFitPredictOp(checkpoint=checkpoint, **kwargs)

    def run(self):
        return self._operator


class HDBSCANFitOp(FitInternal):
    def __init__(self, checkpoint=False, **kwargs):
        super().__init__(name="HDBSCANFit", checkpoint=checkpoint, **kwargs)

    def dump(self, model):
        # TODO: Check how this algorithm can be saved
        return model

    def load(self, model):
        # TODO: Check how this algorithm can be restored
        return model


class HDBSCANFitPredictOp(FitPredictInternal):
    def __init__(self, checkpoint=False, **kwargs):
        super().__init__(name="HDBSCANFitPredict", checkpoint=checkpoint, **kwargs)

    def load(self, model):
        # TODO: Check how this algorithm can be restored
        return model
