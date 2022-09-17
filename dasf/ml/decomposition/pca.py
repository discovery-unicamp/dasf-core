#!/usr/bin/env python3

from sklearn.decomposition import PCA as PCA_CPU
from dask_ml.decomposition import PCA as PCA_MCPU

from dasf.ml import Fit, FitTransform
from dasf.utils.utils import get_full_qualname
from dasf.utils.utils import is_gpu_supported
from dasf.pipeline.types import TaskExecutorType

try:
    from cuml.decomposition import PCA as PCA_GPU
    from cuml.dask.decomposition import PCA as PCA_MGPU
except ImportError:
    pass


class PCAFit(Fit):
    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None
    ):
        super().__init__(name="PCAFit")

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
            self.__pca_gpu = PCA_GPU(
                n_components=n_components,
                copy=copy,
                whiten=whiten,
                svd_solver=svd_solver,
                tol=tol,
                iterated_power=iterated_power,
                random_state=random_state,
            )
            self.__pca_mgpu = PCA_MGPU(
                n_components=n_components,
                copy=copy,
                whiten=whiten,
                svd_solver=svd_solver,
                tol=tol,
                iterated_power=iterated_power,
                random_state=random_state,
            )

        # Select CPU as default to initialize the attribute
        self.pca = self.__pca_cpu

        if is_gpu_supported():
            self.set_output(
                [
                    get_full_qualname(self.__pca_cpu),
                    get_full_qualname(self.__pca_mcpu),
                    get_full_qualname(self.__pca_gpu),
                    get_full_qualname(self.__pca_mgpu),
                ]
            )
        else:
            self.set_output(
                [get_full_qualname(self.__pca_cpu), get_full_qualname(self.__pca_mcpu)]
            )

    def define_executor(self, executor):
        if executor.dtype == TaskExecutorType.single_cpu:
            self.pca = self.__pca_cpu
        elif executor.dtype == TaskExecutorType.multi_cpu:
            self.pca = self.__pca_mcpu
        elif is_gpu_supported():
            if executor.dtype == TaskExecutorType.single_gpu:
                self.pca = self.__pca_gpu
            elif executor.dtype == TaskExecutorType.multi_gpu:
                self.pca = self.__pca_mgpu

    def run(self, data):
        return self.pca.fit(data)


class PCAFitTransform(FitTransform):
    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        super().__init__(name="PCAFitTransform")

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
            self.__pca_gpu = PCA_GPU(
                n_components=n_components,
                copy=copy,
                whiten=whiten,
                svd_solver=svd_solver,
                tol=tol,
                iterated_power=iterated_power,
                random_state=random_state,
            )
            self.__pca_mgpu = PCA_MGPU(
                n_components=n_components,
                copy=copy,
                whiten=whiten,
                svd_solver=svd_solver,
                tol=tol,
                iterated_power=iterated_power,
                random_state=random_state,
            )

        # Select CPU as default to initialize the attribute
        self.pca = self.__pca_cpu

        if is_gpu_supported():
            self.set_output(
                [
                    get_full_qualname(self.__pca_cpu),
                    get_full_qualname(self.__pca_mcpu),
                    get_full_qualname(self.__pca_gpu),
                    get_full_qualname(self.__pca_mgpu),
                ]
            )
        else:
            self.set_output(
                [get_full_qualname(self.__pca_cpu), get_full_qualname(self.__pca_mcpu)]
            )

    def define_executor(self, executor):
        if executor.dtype == TaskExecutorType.single_cpu:
            self.pca = self.__pca_cpu
        elif executor.dtype == TaskExecutorType.multi_cpu:
            self.pca = self.__pca_mcpu
        elif is_gpu_supported():
            if executor.dtype == TaskExecutorType.single_gpu:
                self.pca = self.__pca_gpu
            elif executor.dtype == TaskExecutorType.multi_gpu:
                self.pca = self.__pca_mgpu

    def run(self, data, labels=None):
        return self.pca.fit_transform(data, labels)
