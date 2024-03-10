#!/usr/bin/env python3

from dask_ml.cluster import SpectralClustering as SpectralClustering_MCPU
from sklearn.cluster import SpectralClustering as SpectralClustering_CPU

from dasf.ml.cluster.classifier import ClusterClassifier


class SpectralClustering(ClusterClassifier):
    """
    Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex, or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster, such as when clusters are
    nested circles on the 2D plane.

    If the affinity matrix is the adjacency matrix of a graph, this method
    can be used to find normalized graph cuts.

    When calling ``fit``, an affinity matrix is constructed using either
    a kernel function such the Gaussian (aka RBF) kernel with Euclidean
    distance ``d(X, X)``::

            np.exp(-gamma * d(X,X) ** 2)

    or a k-nearest neighbors connectivity matrix.

    Alternatively, a user-provided affinity matrix can be specified by
    setting ``affinity='precomputed'``.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    ----------
    n_clusters : int, default=8
        The dimension of the projection subspace.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities. If None, then ``'arpack'`` is
        used.

    n_components : int, default=n_clusters
        Number of eigenvectors to use for the spectral embedding

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization of the
        lobpcg eigenvectors decomposition when ``eigen_solver='amg'`` and by
        the K-Means initialization. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia. Only used if
        ``assign_labels='kmeans'``.

    gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
        Ignored for ``affinity='nearest_neighbors'``.

    affinity : str or callable, default='rbf'
        How to construct the affinity matrix.
         - 'nearest_neighbors': construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'rbf': construct the affinity matrix using a radial basis function
           (RBF) kernel.
         - 'precomputed': interpret ``X`` as a precomputed affinity matrix,
           where larger values indicate greater similarity between instances.
         - 'precomputed_nearest_neighbors': interpret ``X`` as a sparse graph
           of precomputed distances, and construct a binary affinity matrix
           from the ``n_neighbors`` nearest neighbors of each instance.
         - one of the kernels supported by
           :func:`~sklearn.metrics.pairwise_kernels`.

        Only kernels that produce similarity scores (non-negative values that
        increase with similarity) should be used. This property is not checked
        by the clustering algorithm.

    n_neighbors : int, default=10
        Number of neighbors to use when constructing the affinity matrix using
        the nearest neighbors method. Ignored for ``affinity='rbf'``.

    eigen_tol : float, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when ``eigen_solver='arpack'``.

    assign_labels : {'kmeans', 'discretize'}, default='kmeans'
        The strategy for assigning labels in the embedding space. There are two
        ways to assign labels after the Laplacian embedding. k-means is a
        popular choice, but it can be sensitive to initialization.
        Discretization is another approach which is less sensitive to random
        initialization.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict of str to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    n_jobs : int, default=None
        The number of parallel jobs to run when `affinity='nearest_neighbors'`
        or `affinity='precomputed_nearest_neighbors'`. The neighbors search
        will be done in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        Verbosity mode.

        .. versionadded:: 0.24

    persist_embedding : bool
        Whether to persist the intermediate n_samples x n_components array used
        for clustering.

    kmeans_params : dictionary of string to any, optional
        Keyword arguments for the KMeans clustering used for the final
        clustering.

    Examples
    --------
    >>> from dasf.ml.cluster import SpectralClustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralClustering(n_clusters=2,
    ...         assign_labels='discretize',
    ...         random_state=0).fit(X)
    >>> clustering
    SpectralClustering(assign_labels='discretize', n_clusters=2,
        random_state=0)

    For further informations see:
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
    - https://ml.dask.org/modules/generated/dask_ml.cluster.SpectralClustering.html

    Notes
    -----
    A distance matrix for which 0 indicates identical elements and high values
    indicate very dissimilar elements can be transformed into an affinity /
    similarity matrix that is well-suited for the algorithm by
    applying the Gaussian (aka RBF, heat) kernel::

        np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

    where ``delta`` is a free parameter representing the width of the Gaussian
    kernel.

    An alternative is to take a symmetric version of the k-nearest neighbors
    connectivity matrix of the points.

    If the pyamg package is installed, it is used: this greatly
    speeds up computation.

    References
    ----------

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

    """
    def __init__(
        self,
        n_clusters=8,
        eigen_solver=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol=0.0,
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None,
        n_components=None,
        persist_embedding=False,
        kmeans_params=None,
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.n_components = n_components
        self.persist_embedding = persist_embedding
        self.kmeans_params = kmeans_params
        self.verbose = verbose

        self.__sc_cpu = SpectralClustering_CPU(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            random_state=random_state,
            n_init=n_init,
            gamma=gamma,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_tol=eigen_tol,
            assign_labels=assign_labels,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_jobs=n_jobs,
            n_components=n_components,
            verbose=verbose
        )

        # If n_components is set to None, use default
        n_components = 100 if n_components is None else n_components

        self.__sc_mcpu = SpectralClustering_MCPU(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            random_state=random_state,
            n_init=n_init,
            gamma=gamma,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_tol=eigen_tol,
            assign_labels=assign_labels,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_jobs=n_jobs,
            n_components=n_components,
            persist_embedding=persist_embedding,
            kmeans_params=kmeans_params,
        )

    def _fit_cpu(self, X, y=None, sample_weight=None):
        return self.__sc_cpu.fit(X=X, y=y)

    def _lazy_fit_predict_cpu(self, X, y=None, sample_weight=None):
        return self.__sc_mcpu.fit_predict(X=X)

    def _fit_predict_cpu(self, X, y=None, sample_weight=None):
        return self.__sc_cpu.fit_predict(X)
