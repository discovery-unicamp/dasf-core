#!/usr/bin/env python3

""" NearestNeighbors algorithm module. """

from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_CPU
from sklearn.neighbors import NearestNeighbors as NearestNeighbors_CPU

from dasf.transforms.base import Fit, GetParams, Predict, SetParams, Transform
from dasf.utils.decorators import task_handler
from dasf.utils.funcs import is_gpu_supported

try:
    from cuml.neighbors import KNeighborsClassifier as KNeighborsClassifier_GPU
    from cuml.neighbors import NearestNeighbors as NearestNeighbors_GPU
except ImportError:
    pass


class KNeighbors(Transform):
    """
    Class representing a KNeighbors operation of the pipeline.
    """

    def _kneighbors_cpu(self, X, **kwargs):
        """
        Respective immediate kneighbors mocked function for local CPU(s).
        """
        raise NotImplementedError

    def _kneighbors_gpu(self, X, **kwargs):
        """
        Respective immediate kneighbors mocked function for local GPU(s).
        """
        raise NotImplementedError

    def _lazy_kneighbors_cpu(self, X, **kwargs):
        """
        Respective lazy kneighbors mocked function for CPUs.
        """
        raise NotImplementedError

    def _lazy_kneighbors_gpu(self, X, **kwargs):
        """
        Respective lazy kneighbors mocked function for GPUs.
        """
        raise NotImplementedError

    @task_handler
    def kneighbors(self, X, **kwargs):
        """
        Generic kneighbors funtion according executor.
        """
        ...


class NearestNeighbors(Fit, GetParams, SetParams, KNeighbors):
    """
    Unsupervised learner for implementing neighbor searches.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    radius : float, default=1.0
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    p : float (positive), default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    effective_metric_ : str
        Metric used to compute distances to neighbors.

    effective_metric_params_ : dict
        Parameters for the metric used to compute distances to neighbors.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        Number of samples in the fitted data.

    See Also
    --------
    KNeighborsClassifier : Classifier implementing the k-nearest neighbors
        vote.
    RadiusNeighborsClassifier : Classifier implementing a vote among neighbors
        within a given radius.
    KNeighborsRegressor : Regression based on k-nearest neighbors.
    RadiusNeighborsRegressor : Regression based on neighbors within a fixed
        radius.
    BallTree : Space partitioning data structure for organizing points in a
        multi-dimensional space, used for nearest neighbor search.

    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.

    https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
    """
    def __init__(self, n_neighbors=5, radius=1.0, algorithm='auto',
                 leaf_size=30, metric='minkowski', p=2,
                 metric_params=None, n_jobs=None, handle=None, verbose=False,
                 output_type=None, **kwargs):
        """ Constructor of the class NearestNeighbors. """
        self.__nn_cpu = NearestNeighbors_CPU(n_neighbors=n_neighbors,
                                             radius=radius,
                                             algorithm=algorithm,
                                             leaf_size=leaf_size,
                                             metric=metric, p=p,
                                             metric_params=metric_params,
                                             n_jobs=n_jobs, **kwargs)

        if is_gpu_supported():
            self.__nn_gpu = NearestNeighbors_GPU(n_neighbors=n_neighbors,
                                                 radius=radius,
                                                 algorithm=algorithm,
                                                 leaf_size=leaf_size,
                                                 metric=metric, p=p,
                                                 metric_params=metric_params,
                                                 n_jobs=n_jobs,
                                                 handle=handle,
                                                 verbose=verbose,
                                                 output_type=output_type,
                                                 **kwargs)
        else:
            self.__nn_gpu = None

    def _fit_cpu(self, X, y=None, **kwargs):
        """
        Fit the nearest neighbors estimator from the training dataset using
        CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : NearestNeighbors
            The fitted nearest neighbors estimator.
        """
        return self.__nn_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None, **kwargs):
        """
        Fit the nearest neighbors estimator from the training dataset using
        GPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : NearestNeighbors
            The fitted nearest neighbors estimator.
        """
        if self.__nn_gpu is None:
            raise NotImplementedError("GPU is not supported")
        return self.__nn_gpu.fit(X=X, **kwargs)

    def _get_params_cpu(self, deep=True, **kwargs):
        """
        Get parameters for this estimator using CPU only.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self.__nn_cpu.get_params(deep=deep)

    def _set_params_cpu(self, **params):
        """
        Set the parameters of this estimator using CPU only.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        return self.__nn_cpu.set_params(**params)

    def _kneighbors_cpu(self, X, **kwargs):
        """
        Find the K-neighbors of a point using CPU.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features), \
            or (n_queries, n_indexed) if metric == 'precomputed', default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True.

        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.

        Examples
        --------
        In the following example, we construct a NearestNeighbors
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]
        """
        return self.__nn_cpu.kneighbors(X=X, **kwargs)

    def _kneighbors_gpu(self, X, **kwargs):
        """
        Find the K-neighbors of a point using GPU.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features), \
            or (n_queries, n_indexed) if metric == 'precomputed', default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True.

        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        if self.__nn_gpu is None:
            raise NotImplementedError("GPU is not supported")
        return self.__nn_gpu.kneighbors(X=X, **kwargs)


class KNeighborsClassifier(Fit, Predict):
    """Classifier implementing the k-nearest neighbors vote.

    Read more in the :ref:`User Guide <classification>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    weights : {'uniform', 'distance'}, callable or None, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Refer to the example entitled
        :ref:`sphx_glr_auto_examples_neighbors_plot_classification.py`
        showing the impact of the `weights` parameter on the decision
        boundary.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p : float, default=2
        Power parameter for the Minkowski metric. When p = 1, this is equivalent
        to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
        For arbitrary p, minkowski_distance (l_p) is used. This parameter is expected
        to be positive.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels known to the classifier

    effective_metric_ : str or callble
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        Number of samples in the fitted data.

    outputs_2d_ : bool
        False when `y`'s shape is (n_samples, ) or (n_samples, 1) during fit
        otherwise True.

    See Also
    --------
    RadiusNeighborsClassifier: Classifier based on neighbors within a fixed radius.
    KNeighborsRegressor: Regression based on k-nearest neighbors.
    RadiusNeighborsRegressor: Regression based on neighbors within a fixed radius.
    NearestNeighbors: Unsupervised learner for implementing neighbor searches.

    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.

    .. warning::

       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances
       but different labels, the results will depend on the ordering of the
       training data.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> neigh = KNeighborsClassifier(n_neighbors=3)
    >>> neigh.fit(X, y)
    KNeighborsClassifier(...)
    >>> print(neigh.predict([[1.1]]))
    [0]
    >>> print(neigh.predict_proba([[0.9]]))
    [[0.666... 0.333...]]
    """
    def __init__(self, n_neighbors=5, weights="uniform", algorithm="auto",
                 leaf_size=30, p=2, metric="minkowski", metric_params=None,
                 n_jobs=None):
        """ Constructor of the class KNeighborsClassifier. """
        self.__knn_cpu = KNeighborsClassifier_CPU(n_neighbors=n_neighbors,
                                                  weights=weights,
                                                  algorithm=algorithm,
                                                  leaf_size=leaf_size,
                                                  p=p,
                                                  metric=metric,
                                                  metric_params=metric_params,
                                                  n_jobs=n_jobs)

        if is_gpu_supported():
            self.__knn_gpu = KNeighborsClassifier_GPU(n_neighbors=n_neighbors,
                                                      weights=weights,
                                                      algorithm=algorithm,
                                                      metric=metric)
        else:
            self.__knn_gpu = None

    def _fit_cpu(self, X, y=None, **kwargs):
        """
        Fit the k-nearest neighbors estimator from the training dataset using
        CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : NearestNeighbors
            The fitted nearest neighbors estimator.
        """
        return self.__knn_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None, **kwargs):
        """
        Fit the k-nearest neighbors estimator from the training dataset using
        GPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : NearestNeighbors
            The fitted nearest neighbors estimator.
        """
        if self.__knn_gpu is None:
            raise NotImplementedError("GPU is not supported")
        return self.__knn_gpu.fit(X=X, **kwargs)

    def _predict_cpu(self, X, **kwargs):
        """
        Predict the class labels for the provided dataset using CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed', or None
            Test samples. If `None`, predictions for all indexed points are
            returned; in this case, points are not considered their own
            neighbors.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        return self.__knn_cpu.predict(X=X)

    def _predict_gpu(self, X, y=None, **kwargs):
        """
        Predict the class labels for the provided dataset using GPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed', or None
            Test samples. If `None`, predictions for all indexed points are
            returned; in this case, points are not considered their own
            neighbors.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        if self.__knn_gpu is None:
            raise NotImplementedError("GPU is not supported")
        return self.__knn_gpu.predict(X=X, **kwargs)
