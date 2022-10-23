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

from dasf.ml.cluster import SpectralClustering
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

    def test_spectral_cpu(self):
        sc = SpectralClustering(n_clusters=self.centers)

        y = sc._fit_predict_cpu(self.X)

        self.assertTrue(is_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y, self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))

    def test_spectral_mcpu(self):
        sc = SpectralClustering(n_clusters=self.centers)

        da_X = da.from_array(self.X)

        y = sc._lazy_fit_predict_cpu(da_X)

        self.assertTrue(is_dask_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y.compute(), self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))
