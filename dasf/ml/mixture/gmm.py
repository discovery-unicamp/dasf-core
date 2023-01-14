#!/usr/bin/env python3

from sklearn.mixture import GaussianMixture as GaussianMixture_CPU

from dasf.ml.mixture.classifier import MixtureClassifier


class GaussianMixture(MixtureClassifier):
    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=0.001,
        reg_covar=1e-06,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10
    ):

        self.__gmm_cpu = GaussianMixture_CPU(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

    def _fit_cpu(self, X, y=None):
        return self.__gmm_cpu.fit(X=X, y=y)

    def _fit_predict_cpu(self, X, y=None):
        return self.__gmm_cpu.fit_predict(X=X, y=y)

    def _predict_cpu(self, X, y=None):
        return self.__gmm_cpu.predict(X=X)

    def _set_params_cpu(self, **params):
        return self.__gmm_cpu.set_params(**params)

    def _get_params_cpu(self, deep=True):
        return self.__gmm_cpu.get_params(deep=deep)
