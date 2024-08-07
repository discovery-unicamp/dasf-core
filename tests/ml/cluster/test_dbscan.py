#!/usr/bin/env python3

import unittest

import numpy as np

try:
    import cupy as cp
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
except ImportError:
    pass

from sklearn.datasets import make_blobs

from dasf.ml.cluster import DBSCAN
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import is_cpu_array, is_gpu_array


class TestDBSCAN(unittest.TestCase):
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

    def test_dbscan_cpu(self):
        sc = DBSCAN()

        y = sc._fit_predict_cpu(self.X)

        self.assertTrue(is_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y, self.y)

        self.assertTrue(float(len(np.where(y1 != y2)[0])/len(y1))*100 < 5.0)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_dbscan_gpu(self):
        # For GPUs we need to specify which data we are handling with `output_type`.
        sc = DBSCAN(output_type='cupy')

        y = sc._fit_predict_gpu(self.X)

        self.assertTrue(is_gpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y.get(), self.y)

        self.assertTrue(float(len(np.where(y1 != y2)[0])/len(y1))*100 < 5.0)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_dbscan_mgpu(self):
        # KMeans for Multi GPUs requires a client
        with LocalCUDACluster() as cluster:
            client = Client(cluster)

            # For GPUs we need to specify which data we are handling with `output_type`.
            sc = DBSCAN(output_type='cupy')

            cp_X = cp.asarray(self.X)

            y = sc._lazy_fit_predict_gpu(cp_X)

            self.assertTrue(is_gpu_array(y))

            y1, y2 = self.__match_randomly_labels_created(y.get(), self.y)

            self.assertTrue(float(len(np.where(y1 != y2)[0])/len(y1))*100 < 5.0)

            # Compute everything to gracefully shutdown
            client.close()
