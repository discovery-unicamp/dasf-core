""" DBSCAN algorithm module. """
#!/usr/bin/env python3

from sklearn.cluster import DBSCAN as DBSCAN_CPU

from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.funcs import is_gpu_supported

try:
    from cuml.cluster import DBSCAN as DBSCAN_GPU
    from cuml.dask.cluster import DBSCAN as DBSCAN_MGPU
except ImportError:
    pass


class DBSCAN(ClusterClassifier):
    """
    Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : string, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`Glossary <sparse graph>`, in which
        case only "nonzero" elements may be considered neighbors for DBSCAN.

        .. versionadded:: 0.17
           metric *precomputed* to accept precomputed sparse matrix.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=None
        The power of the Minkowski metric to be used to calculate distance
        between points. If None, then ``p=2`` (equivalent to the Euclidean
        distance).

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of the
        estimator. If None, it'll inherit the output type set at the module
        level, cuml.global_settings.output_type. See Output Data Type
        Configuration for more info.

    calc_core_sample_indices(optional) : boolean, default = True
        Indicates whether the indices of the core samples should be calculated.
        The the attribute `core_sample_indices_` will not be used, setting this
        to False will avoid unnecessary kernel launches.


    Examples
    --------
    >>> from dasf.ml.cluster import DBSCAN
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3],
    ...               [8, 7], [8, 8], [25, 80]])
    >>> clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    >>> clustering
    DBSCAN(eps=3, min_samples=2)

    For further informations see:
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    - https://docs.rapids.ai/api/cuml/stable/api.html#dbscan
    - https://docs.rapids.ai/api/cuml/stable/api.html#dbscan-clustering

    See Also
    --------
    OPTICS : A similar clustering at multiple values of eps. Our implementation
        is optimized for memory usage.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
    DBSCAN revisited, revisited: why and how you should (still) use DBSCAN.
    ACM Transactions on Database Systems (TODS), 42(3), 19.

    """
    def __init__(
        self,
        eps=0.5,
        leaf_size=40,
        metric="euclidean",
        min_samples=5,
        p=None,
        output_type=None,
        calc_core_sample_indices=True,
        verbose=False,
        **kwargs
    ):
        """ Constructor of the class DBSCAN. """
        super().__init__(**kwargs)

        self.eps = eps
        self.leaf_size = leaf_size
        self.metric = metric
        self.min_samples = min_samples
        self.p = p
        self.output_type = output_type
        self.calc_core_sample_indices = calc_core_sample_indices
        self.verbose = verbose

        self.__dbscan_cpu = DBSCAN_CPU(
            eps=self.eps,
            leaf_size=self.leaf_size,
            metric=self.metric,
            min_samples=self.min_samples,
            p=self.p,
        )

        if is_gpu_supported():
            self.__dbscan_gpu = DBSCAN_GPU(
                min_samples=self.min_samples,
                output_type=output_type,
                calc_core_sample_indices=calc_core_sample_indices,
            )

            try:
                self.__dbscan_mgpu = DBSCAN_MGPU(
                    min_samples=self.min_samples,
                    output_type=output_type,
                    calc_core_sample_indices=calc_core_sample_indices,
                )
            except ValueError:
                self.__dbscan_mgpu = None

    def _lazy_fit_gpu(self, X, y=None, out_dtype="int32"):
        """
        Perform DBSCAN clustering from features, or distance matrix using Dask
        with GPUs only (from CuML).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        if self.__dbscan_mgpu is None:
            raise NotImplementedError
        return self.__dbscan_mgpu.fit(X=X, out_dtype=out_dtype)

    def _fit_cpu(self, X, y=None, sample_weight=None):
        """
        Perform DBSCAN clustering from features, or distance matrix using CPU
        only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        return self.__dbscan_cpu.fit(X=X, y=y, sample_weight=sample_weight)

    def _fit_gpu(self, X, y=None, out_dtype="int32"):
        """
        Perform DBSCAN clustering from features, or distance matrix using GPU
        only (from CuML).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        return self.__dbscan_gpu.fit(X=X, out_dtype=out_dtype)

    def _lazy_fit_predict_gpu(self, X, y=None, out_dtype="int32"):
        """
        Compute clusters from a data or distance matrix and predict labels
        using Dask and GPUs (from CuML).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        if self.__dbscan_mgpu is None:
            raise NotImplementedError
        return self.__dbscan_mgpu.fit_predict(X=X, out_dtype=out_dtype)

    def _fit_predict_cpu(self, X, y=None, sample_weight=None):
        """
        Compute clusters from a data or distance matrix and predict labels
        using CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        return self.__dbscan_cpu.fit_predict(X=X, y=y, sample_weight=sample_weight)

    def _fit_predict_gpu(self, X, y=None, out_dtype="int32"):
        """
        Compute clusters from a data or distance matrix and predict labels
        using GPU only (from CuML).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        return self.__dbscan_gpu.fit_predict(X=X, out_dtype=out_dtype)
