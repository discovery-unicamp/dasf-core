#!/usr/bin/env python3

from sklearn.cluster import (
    AgglomerativeClustering as AgglomerativeClustering_CPU,
)  # noqa

from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.funcs import is_gpu_supported

try:
    from cuml import AgglomerativeClustering as AgglomerativeClustering_GPU
except ImportError:
    pass


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
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_clusters = n_clusters
        self.affinity = affinity
        self.connectivity = connectivity
        self.linkage = linkage
        self.memory = memory
        self.compute_full_tree = compute_full_tree
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances
        self.handle = handle
        self.verbose = verbose
        self.n_neighbors = n_neighbors
        self.output_type = output_type

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
