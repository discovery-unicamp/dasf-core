#!/usr/bin/env python3

import unittest
import numpy as np
import dask.array as da

try:
    import cupy as cp

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
except ImportError:
    pass

from sklearn.datasets import make_blobs

from dasf.ml.cluster import KMeans
from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_gpu_array
from dasf.utils.types import is_dask_cpu_array
from dasf.utils.types import is_dask_gpu_array
from dasf.utils.funcs import is_gpu_supported


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.centers = 6
        self.size = 1000

        # Lets use distant samples for algorithm coherence check
        cluster_std = np.ones(self.centers) * 0.075

        self.X, self.y, self.centroids = make_blobs(n_samples=self.size,
                                                    n_features=2,
                                                    cluster_std=cluster_std,
                                                    centers=self.centers,
                                                    return_centers=True)

    def __match_randomly_labels_created(self, y1, y2):
        y2 = (y2 * -1) - 1

        for i in range(len(y1)):
            if y2[i] < 0:
                y2[y2 == y2[i]] = y1[i]

            if not np.any(y2[y2 < 0]):
                break

        return y1, y2

    def test_kmeans_cpu(self):
        kmeans = KMeans(n_clusters=self.centers, max_iter=300)

        kmeans._fit_cpu(self.X)

        y = kmeans._predict_cpu(self.X)

        self.assertTrue(is_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y, self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))

    def test_kmeans_mcpu(self):
        kmeans = KMeans(n_clusters=self.centers, max_iter=300)

        da_X = da.from_array(self.X)

        kmeans._lazy_fit_cpu(da_X)

        y = kmeans._lazy_predict_cpu(da_X)

        self.assertTrue(is_dask_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y.compute(), self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_kmeans_gpu(self):
        kmeans = KMeans(n_clusters=self.centers, max_iter=100)

        cp_X = cp.asarray(self.X)

        kmeans._fit_gpu(cp_X)

        y = kmeans._predict_gpu(cp_X)

        self.assertTrue(is_gpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y.get(), self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_kmeans_mgpu(self):
        # KMeans for Multi GPUs requires a client
        client = Client(LocalCUDACluster())

        kmeans = KMeans(n_clusters=self.centers, max_iter=100)

        da_X = da.from_array(cp.asarray(self.X))

        kmeans._lazy_fit_gpu(da_X)

        y = kmeans._lazy_predict_gpu(da_X)

        self.assertTrue(is_dask_gpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y.compute().get(), self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))
