#!/usr/bin/env python3

from sklearn.cluster import DBSCAN as DBSCAN_CPU

from dasf.ml.core import FitInternal, FitPredictInternal
from dasf.ml.cluster.classifier import ClusterClassifier
from dasf.utils.utils import get_full_qualname
from dasf.utils.utils import is_gpu_supported
from dasf.pipeline import ParameterOperator

try:
    from cuml.cluster import DBSCAN as DBSCAN_GPU
    from cuml.dask.cluster import DBSCAN as DBSCAN_MGPU
except ImportError:
    pass


class DBSCAN(ClusterClassifier):
    def __init__(self, eps=0.5, leaf_size=40, metric='euclidean',
                 min_samples=None, p=None):

        self.eps = eps
        self.leaf_size = leaf_size
        self.metric = metric
        self.min_samples = min_samples
        self.p = p

        self.__dbscan_cpu = DBSCAN_CPU(eps=self.eps,
                                       leaf_size=self.leaf_size,
                                       metric=self.metric,
                                       min_samples=self.min_samples,
                                       p=self.p)

        if is_gpu_supported():
            self.__dbscan_gpu = DBSCAN_GPU(eps=self.eps,
                                           leaf_size=self.leaf_size,
                                           metric=self.metric,
                                           min_samples=self.min_samples,
                                           p=self.p)

            self.__dbscan_mgpu = DBSCAN_MGPU(eps=self.eps,
                                             leaf_size=self.leaf_size,
                                             metric=self.metric,
                                             min_samples=self.min_samples,
                                             p=self.p)

    def _lazy_fit_gpu(self, X, y=None, out_dtype='int32'):
        self.__dbscan_mgpu.fit(X=X, out_dtype=out_dtype)

    def _fit_cpu(self, X, y=None, sample_weight=None):
        self.__dbscan_cpu.fit(X=X, y=y, sample_weight=sample_weight)

    def _fit_gpu(self, X, y=None, out_dtype='int32'):
        self.__dbscan_gpu.fit(X=X, out_dtype=out_dtype)

    def _lazy_fit_predict_gpu(self, X, y=None, out_dtype='int32'):
        self.__dbscan_mgpu.fit_predict(X=X, out_dtype=out_dtype)

    def _fit_predict_cpu(self, X, y=None, sample_weight=None):
        self.__dbscan_cpu.fit_predict(X=X, y=y, sample_weight=sample_weight)

    def _fit_predict_gpu(self, X, y=None, out_dtype='int32'):
        self.__dbscan_gpu.fit_predict(X=X, out_dtype=out_dtype)


class DBSCANOperator(ParameterOperator):
    def __init__(self, eps=0.5, leaf_size=40, metric='euclidean',
                 min_samples=None, p=None):
        super().__init__(name=type(self).__name__)

        self.eps = eps
        self.leaf_size = leaf_size
        self.metric = metric
        self.min_samples = min_samples
        self.p = p

        self.__dbscan_cpu = DBSCAN_CPU(eps=self.eps,
                                       leaf_size=self.leaf_size,
                                       metric=self.metric,
                                       min_samples=self.min_samples,
                                       p=self.p)

        if is_gpu_supported():
            self.__dbscan_gpu = DBSCAN_GPU(eps=self.eps,
                                           leaf_size=self.leaf_size,
                                           metric=self.metric,
                                           min_samples=self.min_samples,
                                           p=self.p)

            self.__dbscan_mgpu = DBSCAN_MGPU(eps=self.eps,
                                             leaf_size=self.leaf_size,
                                             metric=self.metric,
                                             min_samples=self.min_samples,
                                             p=self.p)

        # Select CPU as default to initialize the attribute
        self.dbscan = self.__dbscan_cpu

        if is_gpu_supported():
            self.set_output([get_full_qualname(self.__dbscan_cpu),
                             get_full_qualname(self.__dbscan_gpu),
                             get_full_qualname(self.__dbscan_mgpu)])
        else:
            self.set_output([get_full_qualname(self.__dbscan_cpu)])

        self.fit = DBSCANFit()
        self.fit_predict = DBSCANFitPredict()

    def setup_cpu(self, executor):
        self.dbscan = self.__dbscan_cpu

    def setup_mcpu(self, executor):
        raise NotImplementedError

    def setup_gpu(self, executor):
        self.dbscan = self.__dbscan_gpu

    def setup_mgpu(self, executor):
        self.dbscan = self.__dbscan_mgpu

    def run(self):
        return self.dbscan


class DBSCANFit(FitInternal):
    def __init__(self):
        super().__init__(name="DBSCANFit")


class DBSCANFitPredict(FitPredictInternal):
    def __init__(self):
        super().__init__(name="DBSCANFitPredict")
