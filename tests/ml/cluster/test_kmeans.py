#!/usr/bin/env python3

import unittest

import dask.array as da
import numpy as np

try:
    import cupy as cp
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
except ImportError:
    pass

from unittest.mock import Mock, patch
from sklearn.datasets import make_blobs

from dasf.ml.cluster import KMeans
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import (
    is_cpu_array,
    is_dask_cpu_array,
    is_dask_gpu_array,
    is_gpu_array,
)


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.size = 1000
        self.centers = 3
        random_state = 42

        self.X, self.y, self.centroids = make_blobs(n_samples=self.size,
                                                    centers=self.centers,
                                                    n_features=2,
                                                    return_centers=True,
                                                    random_state=random_state)

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
        with LocalCUDACluster() as cluster:
            client = Client(cluster)

            kmeans = KMeans(n_clusters=self.centers, max_iter=100)

            da_X = da.from_array(cp.asarray(self.X))

            kmeans._lazy_fit_gpu(da_X)

            y = kmeans._lazy_predict_gpu(da_X)

            self.assertTrue(is_dask_gpu_array(y))

            y1, y2 = self.__match_randomly_labels_created(y.compute().get(), self.y)

            self.assertTrue(np.array_equal(y1, y2, equal_nan=True))

            # Compute everything to gracefully shutdown
            client.close()

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_kmeans_mcpu_local(self):
        kmeans = KMeans(n_clusters=self.centers, max_iter=300, run_local=True)

        da_X = da.from_array(self.X)

        kmeans.fit(da_X)

        y = kmeans.predict(da_X)

        self.assertTrue(is_cpu_array(y))
