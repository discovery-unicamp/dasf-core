#!/usr/bin/env python3

import numpy as np
from xpysom_dask import XPySom

from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.decorators import task_handler
from dasf.utils.funcs import is_gpu_supported

try:
    import cupy as cp
except ImportError:
    pass


class SOM(ClusterClassifier):
    """
    Initializes a Self Organizing Maps.

    A rule of thumb to set the size of the grid for a dimensionality
    reduction task is that it should contain 5*sqrt(N) neurons
    where N is the number of samples in the dataset to analyze.

    E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
    hence a map 8-by-8 should perform well.

    Parameters
    ----------
    x : int
        x dimension of the SOM.

    y : int
        y dimension of the SOM.

    input_len : int
        Number of the elements of the vectors in input.

    sigma : float, default=min(x,y)/2
        Spread of the neighborhood function, needs to be adequate
        to the dimensions of the map.

    sigmaN : float, default=0.01
        Spread of the neighborhood function at last iteration.

    learning_rate : float, default=0.5
        initial learning rate.

    learning_rateN : float, default=0.01
        final learning rate

    decay_function : string, default='exponential'
        Function that reduces learning_rate and sigma at each iteration.
        Possible values: 'exponential', 'linear', 'aymptotic'

    neighborhood_function : string, default='gaussian'
        Function that weights the neighborhood of a position in the map.
        Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

    topology : string, default='rectangular'
        Topology of the map.
        Possible values: 'rectangular', 'hexagonal'

    activation_distance : string, default='euclidean'
        Distance used to activate the map.
        Possible values: 'euclidean', 'cosine', 'manhattan'

    random_seed : int, default=None
        Random seed to use.

    n_parallel : uint, default=#max_CUDA_threads or 500*#CPUcores
        Number of samples to be processed at a time. Setting a too low
        value may drastically lower performance due to under-utilization,
        setting a too high value increases memory usage without granting 
        any significant performance benefit.

    xp : numpy or cupy, default=cupy if can be imported else numpy
        Use numpy (CPU) or cupy (GPU) for computations.

    std_coeff: float, default=0.5
        Used to calculate gausssian exponent denominator:
        d = 2*std_coeff**2*sigma**2

    compact_support: bool, default=False
        Cut the neighbor function to 0 beyond neighbor radius sigma

    Examples
    --------
    >>> from dasf.ml.cluster import SOM
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> som = SOM(x=3, y=2, input_len=2,
    ...           num_epochs=100).fit(X)
    >>> som
    SOM(x=3, y=2, input_len=2, num_epochs=100)

    """
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
        **kwargs
    ):
        super().__init__(**kwargs)

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

    @task_handler
    def quantization_error(self, X):
        ...
