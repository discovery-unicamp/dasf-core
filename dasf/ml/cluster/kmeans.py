#!/usr/bin/env python3

""" K-Means algorithm module. """

from dask_ml.cluster import KMeans as KMeans_MCPU
from sklearn.cluster import KMeans as KMeans_CPU

from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.decorators import task_handler
from dasf.utils.funcs import is_gpu_supported

try:
    import GPUtil
    if len(GPUtil.getGPUs()) == 0:  # check if GPU are available in current env
        raise ImportError("There is no GPU available here")

    from cuml.cluster import KMeans as KMeans_GPU
    from cuml.dask.cluster import KMeans as KMeans_MGPU
except ImportError:
    pass


class KMeans(ClusterClassifier):
    """
    K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape (n_clusters, n_features), default='k-means++'

        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    precompute_distances : {'auto', True, False}, default='auto'
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision. IMPORTANT: This is used only in Dask ML version.

        True : always precompute distances.

        False : never precompute distances.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    n_jobs : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center. IMPORTANT: This is used only in Dask ML version.

        ``None`` or ``-1`` means using all processors.

    init_max_iter : int, default=None
        Number of iterations for init step.

    algorithm : {“lloyd”, “elkan”}, default=”lloyd”
        K-means algorithm to use. The classical EM-style algorithm is "lloyd".
        The "elkan" variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However
        it’s more memory intensive due to the allocation of an extra array of
        shape (n_samples, n_clusters).

        .. versionchanged:: 0.18
            Added Elkan algorithm

    oversampling_factor : int, default=2
        The amount of points to sample in scalable k-means++ initialization
        for potential centroids. Increasing this value can lead to better
        initial centroids at the cost of memory. The total number of centroids
        sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.

    max_samples_per_batch : int, default=32768
        The number of data samples to use for batches of the pairwise distance
        computation. This computation is done throughout both fit predict. The
        default should suit most cases. The total number of elements in the
        batched pairwise distance computation is max_samples_per_batch *
        n_clusters. It might become necessary to lower this number when
        n_clusters becomes prohibitively large.

    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of the
        estimator. If None, it'll inherit the output type set at the module
        level, cuml.global_settings.output_type. See Output Data Type
        Configuration for more info.

    See Also
    --------
    MiniBatchKMeans : Alternative online implementation that does incremental
        updates of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), where n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from dasf.ml.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)

    For further informations see:
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    - https://ml.dask.org/modules/generated/dask_ml.cluster.KMeans.html
    - https://docs.rapids.ai/api/cuml/stable/api.html#k-means-clustering
    - https://docs.rapids.ai/api/cuml/stable/api.html#cuml.dask.cluster.KMeans

    """
    def __init__(
        self,
        n_clusters=8,
        init=None,
        n_init=None,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm='lloyd',
        oversampling_factor=2.0,
        n_jobs=1,
        init_max_iter=None,
        max_samples_per_batch=32768,
        precompute_distances='auto',
        output_type=None,
        **kwargs
    ):
        """ Constructor of the class KMeans. """
        super().__init__(**kwargs)

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.verbose = verbose
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.oversampling_factor = oversampling_factor
        self.n_jobs = n_jobs
        self.init_max_iter = init_max_iter
        self.max_samples_per_batch = max_samples_per_batch
        self.precompute_distances = precompute_distances
        self.output_type = output_type

        # Estimator for CPU operations
        self.__kmeans_cpu = KMeans_CPU(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            init=("k-means++" if init is None else init),
            n_init=(10 if n_init is None else n_init),
            tol=tol,
            verbose=verbose,
            copy_x=copy_x,
            algorithm=algorithm,
        )

        # Estimator for Dask ML operations
        self.__kmeans_mcpu = KMeans_MCPU(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            init=("k-means||" if init is None else init),
            tol=tol,
            oversampling_factor=oversampling_factor,
            algorithm=algorithm,
            n_jobs=n_jobs,
            init_max_iter=init_max_iter,
            copy_x=copy_x,
            precompute_distances=precompute_distances,
        )

        if is_gpu_supported():
            # Estimator for CuML operations
            self.__kmeans_gpu = KMeans_GPU(
                n_clusters=n_clusters,
                random_state=(1 if random_state is None else random_state),
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                init=("scalable-k-means++" if init is None else init),
                oversampling_factor=oversampling_factor,
                max_samples_per_batch=max_samples_per_batch,
            )

            # XXX: KMeans in Multi GPU requires a Client instance,
            # skip if not present.
            try:
                self.__kmeans_mgpu = KMeans_MGPU(
                    n_clusters=n_clusters,
                    random_state=(1 if random_state is None else random_state),
                    max_iter=max_iter,
                    tol=tol,
                    verbose=verbose,
                    init=("scalable-k-means++" if init is None else init),
                    oversampling_factor=oversampling_factor,
                    max_samples_per_batch=max_samples_per_batch,
                )
            except ValueError:
                self.__kmeans_mgpu = None

    def _lazy_fit_cpu(self, X, y=None, sample_weight=None):
        """
        Compute Dask k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it&apos;s not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return self.__kmeans_mcpu.fit(X=X, y=y)

    def _lazy_fit_gpu(self, X, y=None, sample_weight=None):
        """
        Compute Dask CuML k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it&apos;s not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.__kmeans_mgpu is None:
            raise NotImplementedError
        return self.__kmeans_mgpu.fit(X=X, sample_weight=sample_weight)

    def _fit_cpu(self, X, y=None, sample_weight=None):
        """
        Compute Scikit Learn k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it&apos;s not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return self.__kmeans_cpu.fit(X=X, y=y, sample_weight=sample_weight)

    def _fit_gpu(self, X, y=None, sample_weight=None):
        """
        Compute CuML k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it&apos;s not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return self.__kmeans_gpu.fit(X=X, sample_weight=sample_weight)

    def _lazy_fit_predict_cpu(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample using
        Dask ML.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        local_kmeans = self.__kmeans_mcpu.fit(X=X, y=y)
        return local_kmeans.predict(X=X)

    def _lazy_fit_predict_gpu(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample using
        Dask CuML.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.__kmeans_mgpu is None:
            raise NotImplementedError
        return self.__kmeans_mgpu.fit_predict(X, y, sample_weight)

    def _fit_predict_cpu(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample using
        Scikit Learn.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.__kmeans_cpu.fit_predict(X)

    def _fit_predict_gpu(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample using
        CuML.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.__kmeans_gpu.fit_predict(X=X, sample_weight=sample_weight)

    def _lazy_predict_cpu(self, X, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to using Dask ML.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.__kmeans_mcpu.predict(X)

    def _lazy_predict_gpu(self, X, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to using Dask
        CuML.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.__kmeans_mgpu is None:
            raise NotImplementedError
        return self.__kmeans_mgpu.predict(X, sample_weight)

    def _predict_cpu(self, X, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to using Scikit
        Learn.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.__kmeans_cpu.predict(X)

    def _predict_gpu(self, X, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to using CuML.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.__kmeans_gpu.predict(X, sample_weight)

    def _lazy_predict2_cpu(self, X, sample_weight=None):
        """
        A block predict using Scikit Learn variant but for Dask.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        def __predict(block):
            """Block function to predict data per block."""
            return self._predict_cpu.predict(block, sample_weight=sample_weight)

        return X.map_blocks(
            __predict, chunks=(X.chunks[0],), drop_axis=[1], dtype=X.dtype
        )

    def _lazy_predict2_gpu(self, X, sample_weight=None):
        """
        A block predict using CuML variant but for Dask.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        def __predict(block):
            """Block function to predict data per block."""
            return self._predict_gpu.predict(block, sample_weight=sample_weight)

        return X.map_blocks(
            __predict, chunks=(X.chunks[0],), drop_axis=[1], dtype=X.dtype
        )

    def _predict2_cpu(self, X, sample_weight=None, compat=True):
        """
        A block predict using Scikit Learn variant as a placeholder.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        compat : bool
            There is no version for single CPU/GPU for predict2. This
            compatibility parameter uses the original predict method.
            Otherwise, it raises an exception.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if compat:
            return self._predict_cpu.predict(X, sample_weight=sample_weight)

        raise NotImplementedError("Method available only for Dask.")

    def _predict2_gpu(self, X, sample_weight=None, compat=True):
        """
        A block predict using CuML variant as a placeholder.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        compat : bool
            There is no version for single CPU/GPU for predict2. This
            compatibility parameter uses the original predict method.
            Otherwise, it raises an exception.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if compat:
            return self._predict_gpu.predict(X, sample_weight=sample_weight)

        raise NotImplementedError("Method available only for Dask.")

    @task_handler
    def predict2(self, sample_weight=None):
        """
        Generic predict2 funtion according executor (for some ML methods only).
        """
        ...
