#!/usr/bin/env python3

""" Module representing all types of dataset generators. """

from dask_ml.datasets import make_blobs as make_blobs_MCPU
from sklearn.datasets import make_blobs as make_blobs_CPU

from dask_ml.datasets import make_classification as make_classification_MCPU
from sklearn.datasets import make_classification as make_classification_CPU

from dask_ml.datasets import make_regression as make_regression_MCPU
from sklearn.datasets import make_regression as make_regression_CPU

try:
    from numba import cuda
    assert len(cuda.gpus) != 0 # check if GPU are available in current env
    import cupy as cp

    from cuml.dask.datasets import make_blobs as make_blobs_MGPU
    from cuml.datasets import make_blobs as make_blobs_GPU

    from cuml.dask.datasets import make_classification as make_classification_MGPU
    from cuml.datasets import make_classification as make_classification_GPU
    
    from cuml.dask.datasets import make_regression as make_regression_MGPU
    from cuml.datasets import make_regression as make_regression_GPU
except:  # pragma: no cover
    pass

from dasf.utils.funcs import is_dask_gpu_supported, is_dask_supported, is_gpu_supported
from dasf.utils.types import is_cpu_array


class make_blobs:
    """Generate isotropic Gaussian blobs for clustering.

    For an example of usage, see
    :ref:`sphx_glr_auto_examples_datasets_plot_random_dataset.py`.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int or array-like, default=100
        If int, it is the total number of points equally divided among
        clusters.
        If array-like, each element of the sequence indicates
        the number of samples per cluster.

        .. versionchanged:: v0.20
            one can now pass an array-like to the ``n_samples`` parameter

    n_features : int, default=2
        The number of features for each sample.

    centers : int or array-like of shape (n_centers, n_features), default=None
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.

    cluster_std : float or array-like of float, default=1.0
        The standard deviation of the clusters.

    center_box : tuple of float (min, max), default=(-10.0, 10.0)
        The bounding box for each cluster center when centers are
        generated at random.

    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_centers : bool, default=False
        If True, then return the centers of each cluster.

        .. versionadded:: 0.23

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each sample.

    centers : ndarray of shape (n_centers, n_features)
        The centers of each cluster. Only returned if
        ``return_centers=True``.

    See Also
    --------
    make_classification : A more intricate variant.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
    ...                   random_state=0)
    >>> print(X.shape)
    (10, 2)
    >>> y
    array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
    >>> X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2,
    ...                   random_state=0)
    >>> print(X.shape)
    (10, 2)
    >>> y
    array([0, 1, 2, 0, 2, 2, 2, 1, 1, 0])
    """
    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        if kwargs is None:
            return instance
        else:
            return instance(**kwargs)

    def _lazy_make_blobs_cpu(self, **kwargs):
        return make_blobs_MCPU(**kwargs)

    def _lazy_make_blobs_gpu(self, **kwargs):
        return make_blobs_MGPU(**kwargs)

    def _make_blobs_cpu(self, **kwargs):
        return make_blobs_CPU(**kwargs)

    def _make_blobs_gpu(self, **kwargs):
        return make_blobs_GPU(**kwargs)

    def __call__(self, **kwargs):
        if is_dask_gpu_supported():
            if "centers" in kwargs and is_cpu_array(kwargs["centers"]):
                kwargs["centers"] = cp.asarray(kwargs["centers"])
            return self._lazy_make_blobs_gpu(**kwargs)
        elif is_dask_supported():
            return self._lazy_make_blobs_cpu(**kwargs)
        elif is_gpu_supported():
            if "centers" in kwargs and is_cpu_array(kwargs["centers"]):
                kwargs["centers"] = cp.asarray(kwargs["centers"])
            return self._make_blobs_gpu(**kwargs)
        else:
            return self._make_blobs_cpu(**kwargs)


class make_classification:
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    For an example of usage, see
    :ref:`sphx_glr_auto_examples_datasets_plot_random_dataset.py`.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=20
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.

    n_informative : int, default=2
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, default=2
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, default=0
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, default=2
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, default=2
        The number of clusters per class.

    weights : array-like of shape (n_classes,) or (n_classes - 1,),\
              default=None
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if ``len(weights) == n_classes - 1``,
        then the last class weight is automatically inferred.
        More than ``n_samples`` samples may be returned if the sum of
        ``weights`` exceeds 1. Note that the actual class proportions will
        not exactly match ``weights`` when ``flip_y`` isn't 0.

    flip_y : float, default=0.01
        The fraction of samples whose class is assigned randomly. Larger
        values introduce noise in the labels and make the classification
        task harder. Note that the default setting flip_y > 0 might lead
        to less than ``n_classes`` in y in some cases.

    class_sep : float, default=1.0
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : bool, default=True
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, ndarray of shape (n_features,) or None, default=0.0
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, ndarray of shape (n_features,) or None, default=1.0
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : bool, default=True
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for class membership of each sample.

    See Also
    --------
    make_blobs : Simplified variant.
    make_multilabel_classification : Unrelated generator for multilabel tasks.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(random_state=42)
    >>> X.shape
    (100, 20)
    >>> y.shape
    (100,)
    >>> list(y[:5])
    [0, 0, 1, 1, 0]
    """
    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        if kwargs is None:
            return instance
        else:
            return instance(**kwargs)

    def _lazy_make_classification_cpu(self, **kwargs):
        return make_classification_MCPU(**kwargs)

    def _lazy_make_classification_gpu(self, **kwargs):
        return make_classification_MGPU(**kwargs)

    def _make_classification_cpu(self, **kwargs):
        return make_classification_CPU(**kwargs)

    def _make_classification_gpu(self, **kwargs):
        return make_classification_GPU(**kwargs)

    def __call__(self, **kwargs):
        if is_dask_gpu_supported():
            return self._lazy_make_classification_gpu(**kwargs)
        elif is_dask_supported():
            return self._lazy_make_classification_cpu(**kwargs)
        elif is_gpu_supported():
            return self._make_classification_gpu(**kwargs)
        else:
            return self._make_classification_cpu(**kwargs)


class make_regression:
    """Generate a random regression problem.

    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile. See :func:`make_low_rank_matrix` for
    more details.

    The output is generated by applying a (potentially biased) random linear
    regression model with `n_informative` nonzero regressors to the previously
    generated input and some gaussian centered noise with some adjustable
    scale.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=100
        The number of features.

    n_informative : int, default=10
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.

    n_targets : int, default=1
        The number of regression targets, i.e., the dimension of the y output
        vector associated with a sample. By default, the output is a scalar.

    bias : float, default=0.0
        The bias term in the underlying linear model.

    effective_rank : int, default=None
        If not None:
            The approximate number of singular vectors required to explain most
            of the input data by linear combinations. Using this kind of
            singular spectrum in the input allows the generator to reproduce
            the correlations often observed in practice.
        If None:
            The input set is well conditioned, centered and gaussian with
            unit variance.

    tail_strength : float, default=0.5
        The relative importance of the fat noisy tail of the singular values
        profile if `effective_rank` is not None. When a float, it should be
        between 0 and 1.

    noise : float, default=0.0
        The standard deviation of the gaussian noise applied to the output.

    shuffle : bool, default=True
        Shuffle the samples and the features.

    coef : bool, default=False
        If True, the coefficients of the underlying linear model are returned.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The output values.

    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        The coefficient of the underlying linear model. It is returned only if
        coef is True.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=5, n_features=2, noise=1, random_state=42)
    >>> X
    array([[ 0.4967..., -0.1382... ],
        [ 0.6476...,  1.523...],
        [-0.2341..., -0.2341...],
        [-0.4694...,  0.5425...],
        [ 1.579...,  0.7674...]])
    >>> y
    array([  6.737...,  37.79..., -10.27...,   0.4017...,   42.22...])
    """
    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        if kwargs is None:
            return instance
        else:
            return instance(**kwargs)

    def _lazy_make_regression_cpu(self, **kwargs):
        return make_regression_MCPU(**kwargs)

    def _lazy_make_regression_gpu(self, **kwargs):
        return make_regression_MGPU(**kwargs)

    def _make_regression_cpu(self, **kwargs):
        return make_regression_CPU(**kwargs)

    def _make_regression_gpu(self, **kwargs):
        return make_regression_GPU(**kwargs)

    def __call__(self, **kwargs):
        if is_dask_gpu_supported():
            return self._lazy_make_regression_gpu(**kwargs)
        elif is_dask_supported():
            return self._lazy_make_regression_cpu(**kwargs)
        elif is_gpu_supported():
            return self._make_regression_gpu(**kwargs)
        else:
            return self._make_regression_cpu(**kwargs)
