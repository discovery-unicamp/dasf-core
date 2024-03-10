#!/usr/bin/env python3

from sklearn.cluster import (  # noqa
    AgglomerativeClustering as AgglomerativeClustering_CPU,
)

from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.funcs import is_gpu_supported

try:
    from cuml import AgglomerativeClustering as AgglomerativeClustering_GPU
except ImportError:
    pass


class AgglomerativeClustering(ClusterClassifier):
    """

    Agglomerative Clustering

    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    affinity : str or callable, default='euclidean'
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed".
        If linkage is "ward", only "euclidean" is accepted.
        If "precomputed", a distance matrix (instead of a similarity matrix)
        is needed as input for the fit method.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        kneighbors_graph. Default is ``None``, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at ``n_clusters``. This is
        useful to decrease computation time if the number of clusters is not
        small compared to the number of samples. This option is useful only
        when specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - 'ward' minimizes the variance of the clusters being merged.
        - 'average' uses the average of the distances of each observation of
          the two sets.
        - 'complete' or 'maximum' linkage uses the maximum distances between
          all observations of the two sets.
        - 'single' uses the minimum of the distances between all observations
          of the two sets.

        .. versionadded:: 0.20
            Added the 'single' option

    distance_threshold : float, default=None
        The linkage distance threshold above which, clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

    n_neighbors : int, default = 15
        The number of neighbors to compute when connectivity = "knn"

    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of the
        estimator. If None, it'll inherit the output type set at the module
        level, cuml.global_settings.output_type. See Output Data Type
        Configuration for more info.

    Examples
    --------
    >>> from dasf.ml.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering
    AgglomerativeClustering()

    For further informations see:
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    - https://docs.rapids.ai/api/cuml/stable/api.html#agglomerative-clustering

    """
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
