#!/usr/bin/env python3

import unittest
import numpy as np

from parameterized import parameterized_class

try:
    import cupy as cp
except ImportError:
    pass

from sklearn.datasets import make_blobs, make_moons

from dasf.ml.cluster import HDBSCAN
from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_gpu_array
from dasf.utils.funcs import is_gpu_supported


def generate_blobs():
    blobs = []

    X, y, c = make_blobs(n_samples=1000,
                         centers=3,
                         n_features=2,
                         return_centers=True,
                         random_state=42)

    blobs.append({'X': X, 'y': y, 'centroids': c})

    X, y, c = make_blobs(n_samples=1000,
                         return_centers=True,
                         random_state=30)

    blobs.append({'X': X, 'y': y, 'centroids': c})

#    X, y, c = make_blobs(n_samples=4000,
#                         centers=[(-0.75,2.25),
#                                  (1.0, 2.0),
#                                  (1.0, 1.0),
#                                  (2.0, -0.5),
#                                  (-1.0, -1.0),
#                                  (0.0, 0.0)],
#                         cluster_std=0.5,
#                         return_centers=True,
#                         random_state=12)
#
#    blobs.append({'id': 3, 'X': X, 'y': y, 'centroids': c})

    X, y, c = make_blobs(n_samples=2000,
                         n_features=10,
                         return_centers=True,
                         random_state=10)

    blobs.append({'X': X, 'y': y, 'centroids': c})

    X, y = make_moons(n_samples=3000,
                         noise=0.1,
                         random_state=42)

    blobs.append({'X': X, 'y': y, 'centroids': []})

    return blobs


class TestHDBSCAN(unittest.TestCase):
    def setUp(self):
        size = 1000
        centers = 3
        random_state = 42

        self.X, self.y, self.centroids = make_blobs(n_samples=size,
                                                    centers=centers,
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

    def test_hdbscan_cpu_only_fit(self):
        sc = HDBSCAN()

        y = sc._fit_cpu(self.X)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_hdbscan_gpu_only_fit(self):
        sc = HDBSCAN()

        cp_X = cp.asarray(self.X)

        y = sc._fit_gpu(cp_X)

    def test_hdbscan_cpu(self):
        sc = HDBSCAN()

        y = sc._fit_predict_cpu(self.X)

        self.assertTrue(is_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y, self.y)

        self.assertTrue(float(len(np.where(y1 != y2)[0])/len(y1))*100 < 5.0)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_hdbscan_gpu(self):
        sc = HDBSCAN()

        cp_X = cp.asarray(self.X)

        y = sc._fit_predict_gpu(cp_X)

        self.assertTrue(is_gpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y.get(), self.y)

        self.assertTrue(float(len(np.where(y1 != y2)[0])/len(y1))*100 < 5.0)


@parameterized_class(generate_blobs())
class TestHDBSCANMatches(unittest.TestCase):
    def __match_randomly_labels_created(self, y1, y2):
        y2 = (y2 * -1) - 1

        for i in range(len(y1)):
            if y2[i] < 0:
                y2[y2 == y2[i]] = y1[i]

            if not np.any(y2[y2 < 0]):
                break

        return y1, y2

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_hdbscan(self):
        sc_cpu = HDBSCAN()
        sc_gpu = HDBSCAN()

        cp_X = cp.asarray(self.X)

        y_cpu = sc_cpu._fit_predict_cpu(self.X)
        y_gpu = sc_gpu._fit_predict_gpu(cp_X)

        y1, y2 = self.__match_randomly_labels_created(y_gpu.get(), y_cpu)

        self.assertTrue(float(len(np.where(y1 != y2)[0])/len(y1))*100 < 5.0)
