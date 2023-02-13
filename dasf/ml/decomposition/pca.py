#!/usr/bin/env python3

from sklearn.decomposition import PCA as PCA_CPU
from dask_ml.decomposition import PCA as PCA_MCPU

from dasf.utils.funcs import is_dask_supported
from dasf.utils.funcs import is_gpu_supported
from dasf.transforms.base import Fit, FitTransform
from dasf.transforms.base import Transform
from dasf.transforms.base import TargeteredTransform

try:
    from cuml.decomposition import PCA as PCA_GPU
    from cuml.dask.decomposition import PCA as PCA_MGPU
except ImportError:
    pass


class PCA(Fit, FitTransform, Transform, TargeteredTransform):
    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
        *args,
        **kwargs,
    ):
        TargeteredTransform.__init__(self, *args, **kwargs)

        self.__pca_cpu = PCA_CPU(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )

        self.__pca_mcpu = PCA_MCPU(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )
        if is_gpu_supported():
            try:
                self.__pca_gpu = PCA_GPU(
                    n_components=n_components,
                    copy=copy,
                    whiten=whiten,
                    svd_solver=svd_solver,
                    tol=tol,
                    iterated_power=iterated_power,
                    random_state=random_state,
                )
            except TypeError:
                self.__pca_gpu = None

            # XXX: PCA in Multi GPU requires a Client instance,
            # skip if not present.
            try:
                self.__pca_mgpu = PCA_MGPU(
                    n_components=n_components,
                    copy=copy,
                    whiten=whiten,
                    svd_solver=svd_solver,
                    tol=tol,
                    iterated_power=iterated_power,
                    random_state=random_state,
                )
            except ValueError:
                self.__pca_mgpu = None

    def _lazy_fit_cpu(self, X, y=None, sample_weights=None):
        return self.__pca_mcpu.fit(X)

    def _lazy_fit_gpu(self, X, y=None, sample_weights=None):
        if self.__pca_mgpu is None:
            raise NotImplementedError
        return self.__pca_mgpu.fit(X)

    def _fit_cpu(self, X, y=None, sample_weights=None):
        return self.__pca_cpu.fit(X)

    def _fit_gpu(self, X, y=None, sample_weights=None):
        if self.__pca_gpu is None:
            raise NotImplementedError
        return self.__pca_gpu.fit(X)

    def _lazy_fit_transform_cpu(self, X, y=None, sample_weights=None):
        return self.__pca_mcpu.fit_transform(X, y)

    def _lazy_fit_transform_gpu(self, X, y=None, sample_weights=None):
        if self.__pca_mgpu is None:
            raise NotImplementedError
        return self.__pca_mgpu.fit_transform(X, y)

    def _fit_transform_cpu(self, X, y=None, sample_weights=None):
        return self.__pca_cpu.fit_transform(X, y)

    def _fit_transform_gpu(self, X, y=None, sample_weights=None):
        if self.__pca_gpu is None:
            raise NotImplementedError
        return self.__pca_gpu.fit_transform(X, y)

    def _lazy_transform_cpu(self, X, y=None, sample_weights=None):
        return self.__pca_mcpu.transform(X)

    def _lazy_transform_gpu(self, X, y=None, sample_weights=None):
        if self.__pca_mgpu is None:
            raise NotImplementedError
        return self.__pca_mgpu.transform(X)

    def _transform_cpu(self, X, y=None, sample_weights=None):
        return self.__pca_cpu.transform(X)

    def _transform_gpu(self, X, y=None, sample_weights=None):
        if self.__pca_gpu is None:
            raise NotImplementedError
        return self.__pca_gpu.transform(X)

    def _get_covariance_cpu(self):
        return self.__pca_cpu.get_covariance()

    def get_covariance(self):
        if not is_dask_supported() and not is_gpu_supported():
            return self._get_covariance_cpu()
        else:
            raise NotImplementedError

    def _get_precision_cpu(self):
        return self.__pca_cpu.get_precision()

    def get_precision(self):
        if not is_dask_supported() and not is_gpu_supported():
            return self._get_precision_cpu()
        else:
            raise NotImplementedError
