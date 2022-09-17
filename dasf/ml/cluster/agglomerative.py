#!/usr/bin/env python3

from sklearn.cluster import (
    AgglomerativeClustering as AgglomerativeClustering_CPU,
)  # noqa

from dasf.ml.core import FitInternal, FitPredictInternal
from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.utils import is_gpu_supported
from dasf.utils.generators import generate_fit
from dasf.utils.generators import generate_fit_predict

try:
    from cuml import AgglomerativeClustering as AgglomerativeClustering_GPU
except ImportError:
    pass


@generate_fit
@generate_fit_predict
class AgglomerativeClustering(ClusterClassifier):
    def __init__(
        self,
        n_clusters=2,
        affinity="euclidean",
        connectivity=None,
        linkage="single",
        memory=None,
        compute_full_tree="auto",
        distance_threshold=None,
        compute_distances=False,
        handle=None,
        verbose=False,
        n_neighbors=10,
        output_type=None,
    ):

        self.__agg_cluster_cpu = AgglomerativeClustering_CPU(
            n_clusters=n_clusters,
            affinity=affinity,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
        )

        if is_gpu_supported():
            if connectivity is None:
                connectivity = "knn"

            self.__agg_cluster_gpu = AgglomerativeClustering_GPU(
                n_clusters=n_clusters,
                affinity=affinity,
                linkage=linkage,
                handle=handle,
                verbose=verbose,
                connectivity=connectivity,
                n_neighbors=n_neighbors,
                output_type=output_type,
            )

    def _fit_cpu(self, X, y=None, convert_dtype=True):
        return self.__agg_cluster_cpu.fit(X, y)

    def _fit_gpu(self, X, y=None, convert_dtype=True):
        return self.__agg_cluster_gpu.fit(X, y, convert_dtype=convert_dtype)

    def _fit_predict_cpu(self, X, y=None):
        return self.__agg_cluster_cpu.fit_predict(X, y)

    def _fit_predict_gpu(self, X, y=None):
        return self.__agg_cluster_gpu.fit_predict(X, y)


class AgglomerativeClusteringOp:
    def __init__(
        self,
        n_clusters=2,
        affinity="euclidean",
        connectivity=None,
        linkage="single",
        memory=None,
        compute_full_tree="auto",
        distance_threshold=None,
        compute_distances=False,
        handle=None,
        verbose=False,
        n_neighbors=10,
        output_type=None,
        checkpoint=False,
    ):
        self._operator = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            linkage=linkage,
            handle=handle,
            verbose=verbose,
            connectivity=connectivity,
            n_neighbors=n_neighbors,
            output_type=output_type,
        )

        self.fit = AgglomerativeClusteringFitOp(checkpoint=checkpoint)
        self.fit_predict = AgglomerativeClusteringFitPredictOp(checkpoint=checkpoint)

    def run(self):
        return self._operator


class AgglomerativeClusteringFitOp(FitInternal):
    def __init__(self, checkpoint=False):
        super().__init__(name="AgglomerativeClusteringFit", checkpoint=checkpoint)

    def dump(self, model):
        # TODO: Check how this algorithm can be saved
        return model

    def load(self, model):
        # TODO: Check how this algorithm can be restored
        return model


class AgglomerativeClusteringFitPredictOp(FitPredictInternal):
    def __init__(self, checkpoint=False):
        super().__init__(
            name="AgglomerativeClusteringFitPredict", checkpoint=checkpoint
        )

    def load(self, model):
        # TODO: Check how this algorithm can be restored
        return model
