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

from dasf.ml.cluster import AgglomerativeClustering
from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_gpu_array
from dasf.utils.types import is_dask_cpu_array
from dasf.utils.types import is_dask_gpu_array
from dasf.utils.funcs import is_gpu_supported


class TestAgglomerativeClustering(unittest.TestCase):
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

    def test_agglomerative_cpu(self):
        sc = AgglomerativeClustering(n_clusters=self.centers)

        y = sc._fit_predict_cpu(self.X)

        self.assertTrue(is_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y, self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_agglomerative_gpu(self):
        # For GPUs we need to specify which data we are handling with `output_type`.
        sc = AgglomerativeClustering(n_clusters=self.centers, output_type='cupy')

        cp_X = cp.asarray(self.X)

        y = sc._fit_predict_gpu(cp_X)

        self.assertTrue(is_gpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y.get(), self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))
