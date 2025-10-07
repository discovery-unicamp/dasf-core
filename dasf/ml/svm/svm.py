#!/usr/bin/env python3

""" Support Vector Machine algorithms module. """

from sklearn.svm import SVC as SVC_CPU
from sklearn.svm import SVR as SVR_CPU
from sklearn.svm import LinearSVC as LinearSVC_CPU
from sklearn.svm import LinearSVR as LinearSVR_CPU

try:
    import GPUtil
    if len(GPUtil.getGPUs()) == 0:  # check if GPU are available in current env
        raise ImportError("There is no GPU available here")

    from cuml.svm import SVC as SVC_GPU
    from cuml.svm import SVR as SVR_GPU
    from cuml.svm import LienarSVC as LinearSVC_GPU
    from cuml.svm import LienarSVR as LinearSVR_GPU
except ImportError:
    pass

from dasf.transforms.base import Fit, GetParams, Predict, SetParams
from dasf.utils.funcs import is_gpu_supported


class SVC(Fit, Predict, GetParams, SetParams):
    """
    C-Support Vector Classification.

    The implementation is based on libsvm. The fit time scales at least
    quadratically with the number of samples and may be impractical
    beyond tens of thousands of samples. For large datasets
    consider using :class:`~sklearn.svm.LinearSVC` or
    :class:`~sklearn.linear_model.SGDClassifier` instead, possibly after a
    :class:`~sklearn.kernel_approximation.Nystroem` transformer or
    other :ref:`kernel_approximation`.

    The multiclass support is handled according to a one-vs-one scheme.

    For details on the precise mathematical formulation of the provided
    kernel functions and how `gamma`, `coef0` and `degree` affect each
    other, see the corresponding section in the narrative documentation:
    :ref:`svm_kernels`.

    To learn how to tune SVC's hyperparameters, see the following example:
    :ref:`sphx_glr_auto_examples_model_selection_plot_nested_cross_validation_iris.py`

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty. For an intuitive visualization of the effects
        of scaling the regularization parameter C, see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
        Specifies the kernel type to be used in the algorithm. If
        none is given, 'rbf' will be used. If a callable is given it is used to
        pre-compute the kernel matrix from data matrices; that matrix should be
        an array of shape ``(n_samples, n_samples)``. For an intuitive
        visualization of different kernel types see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_kernels.py`.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    class_weight : dict or 'balanced', default=None
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, note that
        internally, one-vs-one ('ovo') is always used as a multi-class strategy
        to train models; an ovr matrix is only constructed from the ovo matrix.
        The parameter is ignored for binary classification.

    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        :term:`predict` will break ties according to the confidence values of
        :term:`decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C for each class.
        Computed based on the ``class_weight`` parameter.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (n_classes -1, n_SV)
        Dual coefficients of the support vector in the decision
        function (see :ref:`sgd_mathematical_formulation`), multiplied by
        their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the :ref:`multi-class section of the User Guide
        <svm_multi_class>` for details.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : ndarray of shape (n_classes * (n_classes - 1) // 2,)
        Number of iterations run by the optimization routine to fit the model.
        The shape of this attribute depends on the number of models optimized
        which in turn depends on the number of classes.

    support_ : ndarray of shape (n_SV)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors. An empty array if kernel is precomputed.

    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.

    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
        If `probability=True`, it corresponds to the parameters learned in
        Platt scaling to produce probability estimates from decision values.
        If `probability=False`, it's an empty array. Platt scaling uses the
        logistic function
        ``1 / (1 + exp(decision_value * probA_ + probB_))``
        where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
        more information on the multiclass case and training procedure see
        section 8 of [1]_.

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.
    """
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        nochange_steps=1000,
        random_state=None,
    ):

        self.__svc_cpu = SVC_CPU(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

        if is_gpu_supported():
            self.__svc_gpu = SVC_GPU(
                C=C,
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                coef0=coef0,
                tol=tol,
                cache_size=cache_size,
                class_weight=class_weight,
                verbose=verbose,
                max_iter=max_iter,
                random_state=random_state,
                multiclass_strategy=decision_function_shape,
                probability=probability,
                output_type="input",
            )

    def _fit_cpu(self, X, y, sample_weight=None):
        return self.__svc_cpu.fit(X, y, sample_weight)

    def _fit_gpu(self, X, y, sample_weight=None):
        return self.__svc_gpu.fit(X, y, sample_weight)

    def _predict_cpu(self, X):
        return self.__svc_cpu.predict(X)

    def _predict_gpu(self, X):
        return self.__svc_gpu.predict(X)

    def _get_params_cpu(self, deep=True):
        return self.__svc_cpu.get_params(deep=deep)

    def _set_params_cpu(self, **params):
        return self.__svc_cpu.set_params(**params)


class SVR(Fit, Predict):
    """
    Epsilon-Support Vector Regression.

    The free parameters in the model are C and epsilon.

    The implementation is based on libsvm. The fit time complexity
    is more than quadratic with the number of samples which makes it hard
    to scale to datasets with more than a couple of 10000 samples. For large
    datasets consider using :class:`~sklearn.svm.LinearSVR` or
    :class:`~sklearn.linear_model.SGDRegressor` instead, possibly after a
    :class:`~sklearn.kernel_approximation.Nystroem` transformer or
    other :ref:`kernel_approximation`.

    Read more in the :ref:`User Guide <svm_regression>`.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        The penalty is a squared l2. For an intuitive visualization of the
        effects of scaling the regularization parameter C, see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`.

    epsilon : float, default=0.1
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value. Must be non-negative.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (1, n_SV)
        Coefficients of the support vector in the decision function.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    intercept_ : ndarray of shape (1,)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run by the optimization routine to fit the model.

        .. versionadded:: 1.1

    n_support_ : ndarray of shape (1,), dtype=int32
        Number of support vectors.

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.

    support_ : ndarray of shape (n_SV,)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.
    """
    def __init__(
        self,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=0.001,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
        nochange_steps=1000,
    ):
        """ Constructor of the class SVR. """
        self.__svr_cpu = SVR_CPU(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            epsilon=epsilon,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

        if is_gpu_supported():
            self.__svr_gpu = SVR_GPU(
                C=C,
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                coef0=coef0,
                tol=tol,
                epsilon=epsilon,
                shrinking=shrinking,
                cache_size=cache_size,
                verbose=verbose,
                max_iter=max_iter,
                nochange_steps=nochange_steps,
                output_type="input",
            )
        else:
            self.__svr_gpu = None

    def _fit_cpu(self, X, y, sample_weight=None):
        return self.__svr_cpu.fit(X, y, sample_weight)

    def _fit_gpu(self, X, y, sample_weight=None):
        return self.__svr_gpu.fit(X, y, sample_weight)

    def _predict_cpu(self, X):
        return self.__svr_cpu.predict(X)

    def _predict_gpu(self, X):
        return self.__svr_gpu.predict(X)


class LinearSVC(Fit, Predict, GetParams, SetParams):
    """
    Linear Support Vector Classification.

    Similar to SVC with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.

    The main differences between :class:`~sklearn.svm.LinearSVC` and
    :class:`~sklearn.svm.SVC` lie in the loss function used by default, and in
    the handling of intercept regularization between those two implementations.

    This class supports both dense and sparse input and the multiclass support
    is handled according to a one-vs-the-rest scheme.

    Read more in the :ref:`User Guide <svm_classification>`.

    Parameters
    ----------
    penalty : {'l1', 'l2'}, default='l2'
        Specifies the norm used in the penalization. The 'l2'
        penalty is the standard used in SVC. The 'l1' leads to ``coef_``
        vectors that are sparse.

    loss : {'hinge', 'squared_hinge'}, default='squared_hinge'
        Specifies the loss function. 'hinge' is the standard SVM loss
        (used e.g. by the SVC class) while 'squared_hinge' is the
        square of the hinge loss. The combination of ``penalty='l1'``
        and ``loss='hinge'`` is not supported.

    dual : "auto" or bool, default="auto"
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.
        `dual="auto"` will choose the value of the parameter automatically,
        based on the values of `n_samples`, `n_features`, `loss`, `multi_class`
        and `penalty`. If `n_samples` < `n_features` and optimizer supports
        chosen `loss`, `multi_class` and `penalty`, then dual will be set to True,
        otherwise it will be set to False.

        .. versionchanged:: 1.3
           The `"auto"` option is added in version 1.3 and will be the default
           in version 1.5.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        For an intuitive visualization of the effects of scaling
        the regularization parameter C, see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`.

    multi_class : {'ovr', 'crammer_singer'}, default='ovr'
        Determines the multi-class strategy if `y` contains more than
        two classes.
        ``"ovr"`` trains n_classes one-vs-rest classifiers, while
        ``"crammer_singer"`` optimizes a joint objective over all classes.
        While `crammer_singer` is interesting from a theoretical perspective
        as it is consistent, it is seldom used in practice as it rarely leads
        to better accuracy and is more expensive to compute.
        If ``"crammer_singer"`` is chosen, the options loss, penalty and dual
        will be ignored.

    fit_intercept : bool, default=True
        Whether or not to fit an intercept. If set to True, the feature vector
        is extended to include an intercept term: `[x_1, ..., x_n, 1]`, where
        1 corresponds to the intercept. If set to False, no intercept will be
        used in calculations (i.e. data is expected to be already centered).

    intercept_scaling : float, default=1.0
        When `fit_intercept` is True, the instance vector x becomes ``[x_1,
        ..., x_n, intercept_scaling]``, i.e. a "synthetic" feature with a
        constant value equal to `intercept_scaling` is appended to the instance
        vector. The intercept becomes intercept_scaling * synthetic feature
        weight. Note that liblinear internally penalizes the intercept,
        treating it like any other term in the feature vector. To reduce the
        impact of the regularization on the intercept, the `intercept_scaling`
        parameter can be set to a value greater than 1; the higher the value of
        `intercept_scaling`, the lower the impact of regularization on it.
        Then, the weights become `[w_x_1, ..., w_x_n,
        w_intercept*intercept_scaling]`, where `w_x_1, ..., w_x_n` represent
        the feature weights and the intercept weight is scaled by
        `intercept_scaling`. This scaling allows the intercept term to have a
        different regularization behavior compared to the other features.

    class_weight : dict or 'balanced', default=None
        Set the parameter C of class i to ``class_weight[i]*C`` for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        the dual coordinate descent (if ``dual=True``). When ``dual=False`` the
        underlying implementation of :class:`LinearSVC` is not random and
        ``random_state`` has no effect on the results.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 \
            else (n_classes, n_features)
        Weights assigned to the features (coefficients in the primal
        problem).

        ``coef_`` is a readonly property derived from ``raw_coef_`` that
        follows the internal memory layout of liblinear.

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    classes_ : ndarray of shape (n_classes,)
        The unique classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : int
        Maximum number of iterations run across all classes.
    """
    def __init__(
        self,
        epsilon=0.0,
        tol=0.0001,
        C=1.0,
        loss="epsilon_insensitive",
        fit_intercept=True,
        intercept_scaling=1.0,
        dual=True,
        verbose=0,
        random_state=None,
        max_iter=1000,
        handle=None,
        penalty="l2",
        penalized_intercept=False,
        linesearch_max_iter=100,
        lbfgs_memory=5,
        grad_tol=0.0001,
        change_tol=1e-05,
        multi_class="ovr",
    ):
        """ Constructor of the class LinearSVC. """
        self.__linear_svc_cpu = LinearSVC_CPU(
            epsilon=epsilon,
            tol=tol,
            C=C,
            loss=loss,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            dual=dual,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
        )

        if is_gpu_supported():
            self.__linear_svc_gpu = LinearSVC_GPU(
                tol=tol,
                C=C,
                loss=loss,
                fit_intercept=fit_intercept,
                intercept_scaling=intercept_scaling,
                dual=dual,
                verbose=verbose,
                random_state=random_state,
                max_iter=max_iter,
                handle=handle,
                penalty=penalty,
                penalized_intercept=penalized_intercept,
                linesearch_max_iter=linesearch_max_iter,
                lbfgs_memory=lbfgs_memory,
                grad_tol=grad_tol,
                change_tol=change_tol,
                multi_class=multi_class,
            )

    def _fit_cpu(self, X, y, sample_weight=None):
        """
        Fit the model according to the given training data using CPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        return self.__linear_svc_cpu.fit(X, y, sample_weight)

    def _fit_gpu(self, X, y, sample_weight=None):
        """
        Fit the model according to the given training data using GPU only.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        return self.__linear_svc_gpu.fit(X, y, sample_weight)

    def _predict_cpu(self, X):
        return self.__linear_svc_cpu.predict(X)

    def _predict_gpu(self, X):
        return self.__linear_svc_gpu.predict(X)


class LinearSVR(Fit, Predict):
    """
    Linear Support Vector Regression.

    Similar to SVR with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better to large numbers of
    samples.

    The main differences between :class:`~sklearn.svm.LinearSVR` and
    :class:`~sklearn.svm.SVR` lie in the loss function used by default, and in
    the handling of intercept regularization between those two implementations.

    This class supports both dense and sparse input.

    Read more in the :ref:`User Guide <svm_regression>`.

    Parameters
    ----------
    epsilon : float, default=0.0
        Epsilon parameter in the epsilon-insensitive loss function. Note
        that the value of this parameter depends on the scale of the target
        variable y. If unsure, set ``epsilon=0``.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    loss : {'epsilon_insensitive', 'squared_epsilon_insensitive'}, \
            default='epsilon_insensitive'
        Specifies the loss function. The epsilon-insensitive loss
        (standard SVR) is the L1 loss, while the squared epsilon-insensitive
        loss ('squared_epsilon_insensitive') is the L2 loss.

    fit_intercept : bool, default=True
        Whether or not to fit an intercept. If set to True, the feature vector
        is extended to include an intercept term: `[x_1, ..., x_n, 1]`, where
        1 corresponds to the intercept. If set to False, no intercept will be
        used in calculations (i.e. data is expected to be already centered).

    intercept_scaling : float, default=1.0
        When `fit_intercept` is True, the instance vector x becomes `[x_1, ...,
        x_n, intercept_scaling]`, i.e. a "synthetic" feature with a constant
        value equal to `intercept_scaling` is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight.
        Note that liblinear internally penalizes the intercept, treating it
        like any other term in the feature vector. To reduce the impact of the
        regularization on the intercept, the `intercept_scaling` parameter can
        be set to a value greater than 1; the higher the value of
        `intercept_scaling`, the lower the impact of regularization on it.
        Then, the weights become `[w_x_1, ..., w_x_n,
        w_intercept*intercept_scaling]`, where `w_x_1, ..., w_x_n` represent
        the feature weights and the intercept weight is scaled by
        `intercept_scaling`. This scaling allows the intercept term to have a
        different regularization behavior compared to the other features.

    dual : "auto" or bool, default="auto"
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.
        `dual="auto"` will choose the value of the parameter automatically,
        based on the values of `n_samples`, `n_features` and `loss`. If
        `n_samples` < `n_features` and optimizer supports chosen `loss`,
        then dual will be set to True, otherwise it will be set to False.

    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features) if n_classes == 2 \
            else (n_classes, n_features)
        Weights assigned to the features (coefficients in the primal
        problem).

        `coef_` is a readonly property derived from `raw_coef_` that
        follows the internal memory layout of liblinear.

    intercept_ : ndarray of shape (1) if n_classes == 2 else (n_classes)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : int
        Maximum number of iterations run across all classes.
    """
    def __init__(
        self,
        epsilon=0.0,
        tol=0.0001,
        C=1.0,
        loss="epsilon_insensitive",
        fit_intercept=True,
        intercept_scaling=1.0,
        dual=True,
        verbose=0,
        random_state=None,
        max_iter=1000,
        handle=None,
        penalty="l2",
        penalized_intercept=False,
        linesearch_max_iter=100,
        lbfgs_memory=5,
        grad_tol=0.0001,
        change_tol=1e-05,
        multi_class="ovr",
    ):
        """ Constructor of the class LinearSVR. """
        self.__linear_svr_cpu = LinearSVR_CPU(
            epsilon=epsilon,
            tol=tol,
            C=C,
            loss=loss,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            dual=dual,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
        )

        if is_gpu_supported():
            self.__linear_svr_gpu = LinearSVR_GPU(
                tol=tol,
                C=C,
                loss=loss,
                fit_intercept=fit_intercept,
                intercept_scaling=intercept_scaling,
                dual=dual,
                verbose=verbose,
                random_state=random_state,
                max_iter=max_iter,
                handle=handle,
                penalty=penalty,
                penalized_intercept=penalized_intercept,
                linesearch_max_iter=linesearch_max_iter,
                lbfgs_memory=lbfgs_memory,
                grad_tol=grad_tol,
                change_tol=change_tol,
                multi_class=multi_class,
            )

    def _fit_cpu(self, X, y, sample_weight=None):
        return self.__linear_svr_cpu.fit(X, y, sample_weight)

    def _fit_gpu(self, X, y, sample_weight=None):
        return self.__linear_svr_gpu.fit(X, y, sample_weight)

    def _predict_cpu(self, X):
        return self.__linear_svr_cpu.predict(X)

    def _predict_gpu(self, X):
        return self.__linear_svr_gpu.predict(X)
