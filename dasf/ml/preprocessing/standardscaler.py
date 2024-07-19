#!/usr/bin/env python3

""" StandardScaler algorithm module. """

from dask_ml.preprocessing import StandardScaler as StandardScaler_MCPU
from sklearn.preprocessing import StandardScaler as StandardScaler_CPU

from dasf.utils.funcs import is_gpu_supported

try:
    from cuml.preprocessing import StandardScaler as StandardScaler_GPU
except ImportError:
    pass

from dasf.transforms.base import Fit, FitTransform, TargeteredTransform


class StandardScaler(Fit, FitTransform, TargeteredTransform):
    """
    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    than others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    `StandardScaler` is sensitive to outliers, and the features may scale
    differently from each other in the presence of outliers. For an example
    visualization, refer to :ref:`Compare StandardScaler with other scalers
    <plot_all_scaling_standard_scaler_section>`.

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve zero mean and unit
        variance. Generally this is calculated using `np.sqrt(var_)`. If a
        variance is zero, we can't achieve unit variance, and the data is left
        as-is, giving a scaling factor of 1. `scale_` is equal to `None`
        when `with_std=False`.

        .. versionadded:: 0.17
           *scale_*

    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False`` and ``with_std=False``.

    var_ : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_mean=False`` and
        ``with_std=False``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_seen_ : int or ndarray of shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are no missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array of dtype int. If
        `sample_weights` are used it will be a float (if no missing data)
        or an array of dtype float that sums the weights seen so far.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.
    """
    def __init__(self, copy=True, with_mean=True, with_std=True, **kwargs):
        """ Constructor of the class StandardScaler. """
        TargeteredTransform.__init__(self, **kwargs)

        self.__std_scaler_cpu = StandardScaler_CPU(
            copy=copy, with_mean=with_mean, with_std=with_std
        )

        self.__std_scaler_dask = StandardScaler_MCPU(
            copy=copy, with_mean=with_mean, with_std=with_std
        )

        if is_gpu_supported():
            self.__std_scaler_gpu = StandardScaler_GPU(
                copy=copy, with_mean=with_mean, with_std=with_std
            )
        else:
            self.__std_scaler_gpu = None

    def _lazy_fit_cpu(self, X, y=None, sample_weight=None):
        """
        Compute the mean and std to be used for later scaling using Dask with
        CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        self.__std_scaler_dask = self.__std_scaler_dask.fit(X=X, y=y)
        return self

    def _lazy_fit_gpu(self, X, y=None, sample_weight=None):
        """
        Compute the mean and std to be used for later scaling using Dask with
        GPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        self.__std_scaler_dask = self.__std_scaler_dask.fit(X=X, y=y)
        return self

    def _fit_cpu(self, X, y=None, sample_weight=None):
        """
        Compute the mean and std to be used for later scaling using CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        self.__std_scaler_cpu = self.__std_scaler_cpu.fit(X=X, y=y)
        return self

    def _fit_gpu(self, X, y=None, sample_weight=None):
        """
        Compute the mean and std to be used for later scaling using CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        if self.__std_scaler_gpu is None:
            raise NotImplementedError("GPU is not supported")
        self.__std_scaler_gpu = self.__std_scaler_gpu.fit(X=X, y=y)
        return self

    def _lazy_fit_transform_cpu(self, X, y=None):
        """
        Fit to data, then transform it using Dask with CPUs only.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.__std_scaler_dask.fit_transform(X=X, y=y)

    def _lazy_fit_transform_gpu(self, X, y=None):
        """
        Fit to data, then transform it using Dask with GPUs only.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.__std_scaler_dask.fit_transform(X=X, y=y)

    def _fit_transform_cpu(self, X, y=None):
        """
        Fit to data, then transform it using CPU only.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return self.__std_scaler_cpu.fit_transform(X=X, y=y)

    def _fit_transform_gpu(self, X, y=None):
        """
        Fit to data, then transform it using GPU only.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        if self.__std_scaler_gpu is None:
            raise NotImplementedError("GPU is not supported")
        ret = self.__std_scaler_gpu.fit(X=X, y=y)
        return ret.transform(X=X)

    def _lazy_partial_fit_cpu(self, X, y=None, sample_weight=None):
        """
        Online computation of mean and std on X for later scaling using Dask
        with CPUs only.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        self.__std_scaler_dask = self.__std_scaler_dask.partial_fit(X=X, y=y)
        return self

    def _lazy_partial_fit_gpu(self, X, y=None, sample_weight=None):
        """
        Online computation of mean and std on X for later scaling using Dask
        with GPUs only.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        self.__std_scaler_dask = self.__std_scaler_dask.partial_fit(X=X, y=y)
        return self

    def _partial_fit_cpu(self, X, y=None, sample_weight=None):
        """
        Online computation of mean and std on X for later scaling CPU only.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        self.__std_scaler_cpu = self.__std_scaler_cpu.partial_fit(X=X, y=y)
        return self

    def _partial_fit_gpu(self, X, y=None, sample_weight=None):
        """
        Online computation of mean and std on X for later scaling using GPU only.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        if self.__std_scaler_gpu is None:
            raise NotImplementedError("GPU is not supported")
        self.__std_scaler_gpu = self.__std_scaler_gpu.partial_fit(X=X, y=y)
        return self

    def _lazy_transform_cpu(self, X, copy=None):
        """
        Perform standardization by centering and scaling using Dask with CPUs
        only.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        return self.__std_scaler_dask.transform(X=X, copy=copy)

    def _lazy_transform_gpu(self, X, copy=None):
        """
        Perform standardization by centering and scaling using Dask with GPUs
        only.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        return self.__std_scaler_dask.transform(X=X, copy=copy)

    def _transform_cpu(self, X, copy=None):
        """
        Perform standardization by centering and scaling using CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        return self.__std_scaler_cpu.transform(X=X, copy=copy)

    def _transform_gpu(self, X, copy=None):
        """
        Perform standardization by centering and scaling using GPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        if self.__std_scaler_gpu is None:
            raise NotImplementedError("GPU is not supported")
        return self.__std_scaler_gpu.transform(X=X, copy=copy)

    def _lazy_inverse_transform_cpu(self, X, copy=None):
        """
        Undo the scaling of X according to feature_range using Dask with CPUs
        only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.__std_scaler_dask.inverse_transform(X=X, copy=copy)

    def _lazy_inverse_transform_gpu(self, X, copy=None):
        """
        Undo the scaling of X according to feature_range using Dask with GPUs
        only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.__std_scaler_dask.inverse_transform(X=X, copy=copy)

    def _inverse_transform_cpu(self, X, copy=None):
        """
        Undo the scaling of X according to feature_range using CPU only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.__std_scaler_cpu.inverse_transform(X=X, copy=copy)

    def _inverse_transform_gpu(self, X, copy=None):
        """
        Undo the scaling of X according to feature_range using GPU only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        if self.__std_scaler_gpu is None:
            raise NotImplementedError("GPU is not supported")
        return self.__std_scaler_gpu.inverse_transform(X=X, copy=copy)
