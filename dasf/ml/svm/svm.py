#!/usr/bin/env python3

from sklearn.svm import SVC as SVC_CPU
from sklearn.svm import SVR as SVR_CPU
from sklearn.svm import LinearSVC as LinearSVC_CPU
from sklearn.svm import LinearSVR as LinearSVR_CPU

try:
    from cuml.svm import SVC as SVC_GPU
    from cuml.svm import SVR as SVR_GPU
    from cuml.svm import LienarSVC as LinearSVC_GPU
    from cuml.svm import LienarSVR as LinearSVR_GPU
except ImportError:
    pass

from dasf.utils.utils import is_gpu_supported
from dasf.transforms import Fit
from dasf.transforms import Predict
from dasf.transforms import GetParams
from dasf.transforms import SetParams


class SVC(Fit, Predict, GetParams, SetParams):
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

    def _fit_cpu(self, X, y, sample_weight=None):
        return self.__svr_cpu.fit(X, y, sample_weight)

    def _fit_gpu(self, X, y, sample_weight=None):
        return self.__svr_gpu.fit(X, y, sample_weight)

    def _predict_cpu(self, X):
        return self.__svr_cpu.predict(X)

    def _predict_gpu(self, X):
        return self.__svr_gpu.predict(X)


class LinearSVC(Fit, Predict, GetParams, SetParams):
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
        return self.__linear_svc_cpu.fit(X, y, sample_weight)

    def _fit_gpu(self, X, y, sample_weight=None):
        return self.__linear_svc_gpu.fit(X, y, sample_weight)

    def _predict_cpu(self, X):
        return self.__linear_svc_cpu.predict(X)

    def _predict_gpu(self, X):
        return self.__linear_svc_gpu.predict(X)


class LinearSVR(Fit, Predict):
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
