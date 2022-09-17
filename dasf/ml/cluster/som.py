#!/usr/bin/env python3


import numpy as np

from xpysom import XPySom

from dasf.ml.core import FitInternal, PredictInternal
from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.utils import is_dask_gpu_supported
from dasf.utils.utils import is_dask_supported
from dasf.utils.utils import is_gpu_supported
from dasf.utils.generators import generate_fit
from dasf.utils.generators import generate_predict
from dasf.utils.generators import generate_fit_predict

try:
    import cupy as cp
except ImportError:
    pass


@generate_fit
@generate_predict
@generate_fit_predict
class SOM(ClusterClassifier):
    def __init__(
        self,
        x,
        y,
        input_len,
        num_epochs=100,
        sigma=0,
        sigmaN=1,
        learning_rate=0.5,
        learning_rateN=0.01,
        decay_function="exponential",
        neighborhood_function="gaussian",
        std_coeff=0.5,
        topology="rectangular",
        activation_distance="euclidean",
        random_seed=None,
        n_parallel=0,
        compact_support=False,
    ):

        self.x = x
        self.y = y
        self.input_len = input_len
        self.num_epochs = num_epochs
        self.sigma = sigma
        self.sigmaN = sigmaN
        self.learning_rate = learning_rate
        self.learning_rateN = learning_rateN
        self.decay_function = decay_function
        self.neighborhood_function = neighborhood_function
        self.std_coeff = std_coeff
        self.topology = topology
        self.activation_distance = activation_distance
        self.random_seed = random_seed
        self.n_parallel = n_parallel
        self.compact_support = compact_support

        self.__som_cpu = XPySom(
            x=self.x,
            y=self.y,
            input_len=self.input_len,
            sigma=self.sigma,
            sigmaN=self.sigmaN,
            learning_rate=self.learning_rate,
            learning_rateN=self.learning_rateN,
            decay_function=self.decay_function,
            neighborhood_function=self.neighborhood_function,
            std_coeff=self.std_coeff,
            topology=self.topology,
            activation_distance=self.activation_distance,
            random_seed=self.random_seed,
            n_parallel=self.n_parallel,
            compact_support=self.compact_support,
            xp=np,
        )

        self.__som_mcpu = XPySom(
            x=self.x,
            y=self.y,
            input_len=self.input_len,
            sigma=self.sigma,
            sigmaN=self.sigmaN,
            learning_rate=self.learning_rate,
            learning_rateN=self.learning_rateN,
            decay_function=self.decay_function,
            neighborhood_function=self.neighborhood_function,
            std_coeff=self.std_coeff,
            topology=self.topology,
            activation_distance=self.activation_distance,
            random_seed=self.random_seed,
            n_parallel=self.n_parallel,
            compact_support=self.compact_support,
            xp=np,
            use_dask=True,
        )

        if is_gpu_supported():
            self.__som_gpu = XPySom(
                x=self.x,
                y=self.y,
                input_len=self.input_len,
                sigma=self.sigma,
                sigmaN=self.sigmaN,
                learning_rate=self.learning_rate,
                learning_rateN=self.learning_rateN,
                decay_function=self.decay_function,
                neighborhood_function=self.neighborhood_function,
                std_coeff=self.std_coeff,
                topology=self.topology,
                activation_distance=self.activation_distance,
                random_seed=self.random_seed,
                n_parallel=self.n_parallel,
                compact_support=self.compact_support,
                xp=cp,
            )

            self.__som_mgpu = XPySom(
                x=self.x,
                y=self.y,
                input_len=self.input_len,
                sigma=self.sigma,
                sigmaN=self.sigmaN,
                learning_rate=self.learning_rate,
                learning_rateN=self.learning_rateN,
                decay_function=self.decay_function,
                neighborhood_function=self.neighborhood_function,
                std_coeff=self.std_coeff,
                topology=self.topology,
                activation_distance=self.activation_distance,
                random_seed=self.random_seed,
                n_parallel=self.n_parallel,
                compact_support=self.compact_support,
                xp=cp,
                use_dask=True,
            )

    def _lazy_fit_cpu(self, X, y=None, sample_weight=None):
        self.__som = self.__som_mcpu
        return self.__som_mcpu.train(X, self.num_epochs)

    def _lazy_fit_gpu(self, X, y=None, sample_weight=None):
        self.__som = self.__som_mgpu
        return self.__som_mgpu.train(X, self.num_epochs)

    def _fit_cpu(self, X, y=None, sample_weight=None):
        self.__som = self.__som_cpu
        return self.__som_cpu.train(X, self.num_epochs)

    def _fit_gpu(self, X, y=None, sample_weight=None):
        self.__som = self.__som_gpu
        return self.__som_gpu.train(X, self.num_epochs)

    def _lazy_fit_predict_cpu(self, X, y=None, sample_weight=None):
        self.__som = self.__som_mcpu
        return self.__som_mcpu.train(X, self.num_epochs).predict(X)

    def _lazy_fit_predict_gpu(self, X, y=None, sample_weight=None):
        self.__som = self.__som_mgpu
        return self.__som_mgpu.train(X, self.num_epochs).predict(X)

    def _fit_predict_cpu(self, X, y=None, sample_weight=None):
        self.__som = self.__som_cpu
        return self.__som_cpu.train(X, self.num_epochs).predict(X)

    def _fit_predict_gpu(self, X, y=None, sample_weight=None):
        self.__som = self.__som_gpu
        return self.__som_gpu.train(X, self.num_epochs).predict(X)

    def _lazy_predict_cpu(self, X, sample_weight=None):
        return self.__som_mcpu.predict(X)

    def _lazy_predict_gpu(self, X, sample_weight=None):
        return self.__som_mgpu.predict(X)

    def _predict_cpu(self, X, sample_weight=None):
        return self.__som_cpu.predict(X)

    def _predict_gpu(self, X, sample_weight=None):
        return self.__som_gpu.predict(X)

    def _lazy_quantization_error_cpu(self, X):
        return self.__som_mcpu.quantization_error(X)

    def _lazy_quantization_error_gpu(self, X):
        return self.__som_mgpu.quantization_error(X)

    def _quantization_error_cpu(self, X):
        return self.__som_cpu.quantization_error(X)

    def _quantization_error_gpu(self, X):
        return self.__som_gpu.quantization_error(X)

    def quantization_error(self, X):
        if is_dask_gpu_supported():
            self._lazy_quantization_error_gpu(X)
        elif is_dask_supported():
            self._lazy_quantization_error_cpu
        elif is_gpu_supported():
            self._quantization_error_gpu(X)
        else:
            self._quantization_error_cpu(X)


class SOMOp:
    def __init__(
        self,
        x,
        y,
        input_len,
        num_epochs=100,
        sigma=0,
        sigmaN=1,
        learning_rate=0.5,
        learning_rateN=0.01,
        decay_function="exponential",
        neighborhood_function="gaussian",
        std_coeff=0.5,
        topology="rectangular",
        activation_distance="euclidean",
        random_seed=None,
        n_parallel=0,
        compact_support=False,
        checkpoint=False,
    ):
        self._operator = SOM(
            x=x,
            y=y,
            input_len=input_len,
            num_epochs=num_epochs,
            sigma=sigma,
            sigmaN=sigmaN,
            learning_rate=learning_rate,
            learning_rateN=learning_rateN,
            decay_function=decay_function,
            neighborhood_function=neighborhood_function,
            std_coeff=std_coeff,
            topology=topology,
            activation_distance=activation_distance,
            random_seed=random_seed,
            n_parallel=n_parallel,
            compact_support=compact_support,
        )

        self.fit = SOMFitOp(checkpoint=checkpoint)
        self.predict = SOMPredictOp(checkpoint=checkpoint)

    def run(self):
        return self._operator


class SOMFitOp(FitInternal):
    def __init__(self, checkpoint=False):
        super().__init__(name="SOMFit", checkpoint=checkpoint)

    def dump(self, model):
        # TODO: Check how this algorithm can be saved
        return model

    def load(self, model):
        # TODO: Check how this algorithm can be restored
        return model


class SOMPredictOp(PredictInternal):
    def __init__(self, checkpoint=False):
        super().__init__(name="SOMPredict", checkpoint=checkpoint)

    def load(self, model):
        # TODO: Check how this algorithm can be restored
        return model
