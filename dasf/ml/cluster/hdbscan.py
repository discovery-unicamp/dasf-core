#!/usr/bin/env python3

""" HDBSCAN algorithm module. """

from sklearn.cluster import HDBSCAN as HDBSCAN_CPU

from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.funcs import is_gpu_supported

try:
    from cuml.cluster import HDBSCAN as HDBSCAN_GPU
except ImportError:
    pass


class HDBSCAN(ClusterClassifier):
    """
    Perform HDBSCAN clustering from vector array or distance matrix.

    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
    with Noise. Performs DBSCAN over varying epsilon values and integrates
    the result to find a clustering that gives the best stability over epsilon.
    This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN),
    and be more robust to parameter selection.

    Parameters
    ----------
    min_cluster_size : int, optional (default=5)
        The minimum size of clusters; single linkage splits that contain
        fewer points than this will be considered points "falling out" of a
        cluster rather than a cluster splitting into two new clusters.

    min_samples : int, optional (default=None)
        The number of samples in a neighbourhood for a point to be
        considered a core point.

    metric : string, or callable, optional (default='euclidean')
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    p : int, optional (default=None)
        p value to use if using the minkowski metric.

    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.
        See [3]_ for more information.

    cluster_selection_epsilon: float, optional (default=0.0)
                A distance threshold. Clusters below this value will be merged.
        See [5]_ for more information.

    algorithm : string, optional (default='best')
        Exactly which algorithm to use; hdbscan has variants specialised
        for different characteristics of the data. By default this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
            * ``best``
            * ``generic``
            * ``prims_kdtree``
            * ``prims_balltree``
            * ``boruvka_kdtree``
            * ``boruvka_balltree``

    leaf_size: int, optional (default=40)
        If using a space tree algorithm (kdtree, or balltree) the number
        of points ina leaf node of the tree. This does not alter the
        resulting clustering, but may have an effect on the runtime
        of the algorithm.

    memory : Instance of joblib.Memory or string (optional)
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    approx_min_span_tree : bool, optional (default=True)
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.

    gen_min_span_tree: bool, optional (default=False)
        Whether to generate the minimum spanning tree with regard
        to mutual reachability distance for later analysis.

    core_dist_n_jobs : int, optional (default=4)
        Number of parallel jobs to run in core distance computations (if
        supported by the specific algorithm). For ``core_dist_n_jobs``
        below -1, (n_cpus + 1 + core_dist_n_jobs) are used.

    cluster_selection_method : string, optional (default='eom')
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass algorithm
        to find the most persistent clusters. Alternatively you can instead
        select the clusters at the leaves of the tree -- this provides the
        most fine grained and homogeneous clusters. Options are:
            * ``eom``
            * ``leaf``

    allow_single_cluster : bool, optional (default=False)
        By default HDBSCAN* will not produce a single cluster, setting this
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.

    prediction_data : boolean, optional
        Whether to generate extra cached data for predicting labels or
        membership vectors few new unseen points later. If you wish to
        persist the clustering object for later re-use you probably want
        to set this to True.
        (default False)

    match_reference_implementation : bool, optional (default=False)
        There exist some interpretational differences between this
        HDBSCAN* implementation and the original authors reference
        implementation in Java. This can result in very minor differences
        in clustering results. Setting this flag to True will, at a some
        performance cost, ensure that the clustering results match the
        reference implementation.

    connectivity : {'pairwise', 'knn'}, default='knn'
        The type of connectivity matrix to compute.
            * 'pairwise' will compute the entire fully-connected graph of
            pairwise distances between each set of points. This is the fastest
            to compute and can be very fast for smaller datasets but requires
            O(n^2) space.

            * 'knn' will sparsify the fully-connected connectivity matrix to
            save memory and enable much larger inputs. "n_neighbors” will
            control the amount of memory used and the graph will be connected
            automatically in the event "n_neighbors” was not large enough to
            connect it.

    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of the
        estimator. If None, it'll inherit the output type set at the module
        level, cuml.global_settings.output_type. See Output Data Type
        Configuration for more info.

    Examples
    --------
    >>> from dasf.ml.cluster import HDBSCAN
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3],
    ...               [8, 7], [8, 8], [25, 80]])
    >>> clustering = HDBSCAN(min_cluster_size=30, min_samples=2).fit(X)
    >>> clustering
    HDBSCAN(min_cluster_size=30, min_samples=2)

    For further informations see:
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    - https://docs.rapids.ai/api/cuml/stable/api.html#dbscan
    - https://docs.rapids.ai/api/cuml/stable/api.html#dbscan-clustering

    References
    ----------

    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates.
       In Pacific-Asia Conference on Knowledge Discovery and Data Mining
       (pp. 160-172). Springer Berlin Heidelberg.

    .. [2] Campello, R. J., Moulavi, D., Zimek, A., & Sander, J. (2015).
       Hierarchical density estimates for data clustering, visualization,
       and outlier detection. ACM Transactions on Knowledge Discovery
       from Data (TKDD), 10(1), 5.

    .. [3] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
       cluster tree. In Advances in Neural Information Processing Systems
       (pp. 343-351).

    .. [4] Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and
       Sander, J., 2014. Density-Based Clustering Validation. In SDM
       (pp. 839-847).

    .. [5] Malzer, C., & Baum, M. (2019). A Hybrid Approach To Hierarchical
           Density-based Cluster Selection. arxiv preprint 1911.02282.

    """
    def __init__(
        self,
        alpha=1.0,
        gen_min_span_tree=False,
        leaf_size=40,
        metric="euclidean",
        min_cluster_size=5,
        min_samples=None,
        p=None,
        algorithm='auto',
        approx_min_span_tree=True,
        core_dist_n_jobs=4,
        cluster_selection_method='eom',
        allow_single_cluster=False,
        prediction_data=False,
        match_reference_implementation=False,
        connectivity='knn',
        output_type=None,
        verbose=0,
        **kwargs
    ):
        """ Constructor of the class HDBSCAN. """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gen_min_span_tree = gen_min_span_tree
        self.leaf_size = leaf_size
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.p = p
        self.algorithm = algorithm
        self.approx_min_span_tree = approx_min_span_tree
        self.core_dist_n_jobs = core_dist_n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.prediction_data = prediction_data
        self.match_reference_implementation = match_reference_implementation
        self.connectivity = connectivity
        self.output_type = output_type
        self.verbose = verbose

        self.__hdbscan_cpu = HDBSCAN_CPU(
            alpha=alpha,
            leaf_size=leaf_size,
            metric=metric,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            algorithm=algorithm,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
        )

        if is_gpu_supported():
            self.__hdbscan_gpu = HDBSCAN_GPU(
                alpha=alpha,
                gen_min_span_tree=gen_min_span_tree,
                metric=metric,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                p=p,
                cluster_selection_method=cluster_selection_method,
                allow_single_cluster=allow_single_cluster,
                verbose=verbose,
                connectivity=connectivity,
                prediction_data=prediction_data,
                output_type=output_type
            )

    def _fit_cpu(self, X, y=None):
        """
        Perform HDBSCAN clustering from features or distance matrix using CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return self.__hdbscan_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None, convert_dtype=True):
        """
        Perform HDBSCAN clustering from features or distance matrix using GPU only
        (from CuML).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return self.__hdbscan_gpu.fit(X=X, y=y, convert_dtype=convert_dtype)

    def _fit_predict_cpu(self, X, y=None):
        """
        Performs clustering on X and returns cluster labels using only CPU.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        y : ndarray, shape (n_samples, )
            cluster labels
        """
        return self.__hdbscan_cpu.fit_predict(X=X, y=y)

    def _fit_predict_gpu(self, X, y=None):
        """
        Performs clustering on X and returns cluster labels using only GPU
        (from CuML).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        y : ndarray, shape (n_samples, )
            cluster labels
        """
        return self.__hdbscan_gpu.fit_predict(X=X, y=y)
