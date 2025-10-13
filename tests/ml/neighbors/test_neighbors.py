#!/usr/bin/env python3

import unittest

import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

from unittest.mock import Mock, patch
from sklearn.datasets import make_blobs

from dasf.ml.neighbors import KNeighborsClassifier, NearestNeighbors
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import is_cpu_array, is_gpu_array


class TestNearestNeighbors(unittest.TestCase):
    def setUp(self):
        self.size = 10
        self.centers = 3
        random_state = 42

        self.X, self.y, self.centroids = make_blobs(n_samples=self.size,
                                                    centers=self.centers,
                                                    n_features=2,
                                                    return_centers=True,
                                                    random_state=random_state)

        self.dists = np.array([[0, 3], [1, 3], [2, 7], [3, 1], [4, 9],
                               [5, 9], [6, 5], [7, 2], [8, 7], [9, 5]])

    def test_nearestneighbors_cpu(self):
        nn = NearestNeighbors(n_neighbors=2)

        nn._fit_cpu(self.X)

        idxs, dists = nn._kneighbors_cpu(X=self.X)

        self.assertTrue(is_cpu_array(idxs))
        self.assertTrue(is_cpu_array(dists))

        self.assertTrue(np.array_equal(dists, self.dists, equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_nearestneighbors_gpu(self):
        nn = NearestNeighbors(n_neighbors=2, output_type='cupy')

        cp_X = cp.asarray(self.X)

        nn._fit_gpu(cp_X)

        idxs, dists = nn._kneighbors_gpu(X=self.X)

        self.assertTrue(is_gpu_array(idxs))
        self.assertTrue(is_gpu_array(dists))

        self.assertTrue(np.array_equal(dists.get(), self.dists, equal_nan=True))

    @patch('dasf.ml.neighbors.neighbors.is_gpu_supported', Mock(return_value=False))
    def test_nearestneighbors_fit_gpu_no_gpu(self):
        nn = NearestNeighbors(n_neighbors=2, output_type='cupy')

        with self.assertRaises(NotImplementedError) as context:
            nn._fit_gpu(self.X)

        self.assertTrue('GPU is not supported' in str(context.exception))


class TestKNeighborsClassifier(unittest.TestCase):
    def setUp(self):
        self.size = 1000
        self.centers = 3
        random_state = 42

        self.X, self.y, self.centroids = make_blobs(n_samples=self.size,
                                                    centers=self.centers,
                                                    n_features=2,
                                                    return_centers=True,
                                                    random_state=random_state)

        self.dists = np.array([[0, 3], [1, 3], [2, 7], [3, 1], [4, 9],
                               [5, 9], [6, 5], [7, 2], [8, 7], [9, 5]])

    def __match_randomly_labels_created(self, y1, y2):
        y2 = (y2 * -1) - 1

        for i in range(len(y1)):
            if y2[i] < 0:
                y2[y2 == y2[i]] = y1[i]

            if not np.any(y2[y2 < 0]):
                break

        return y1, y2

    def test_knearestneighbors_cpu(self):
        knn = KNeighborsClassifier(n_neighbors=3)

        knn._fit_cpu(self.X, self.y)

        y = knn._predict_cpu(self.X)

        self.assertTrue(is_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y, self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_knearestneighbors_gpu(self):
        knn = KNeighborsClassifier(n_neighbors=3, output_type='cupy')

        cp_X = cp.asarray(self.X)
        cp_y = cp.asarray(self.y)

        knn._fit_gpu(cp_X, cp_y)

        y = knn._predict_gpu(self.X)

        self.assertTrue(is_gpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y.get(), self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))

    @patch('dasf.ml.neighbors.neighbors.is_gpu_supported', Mock(return_value=False))
    def test_knearestneighbors_fit_gpu_no_gpu(self):
        knn = KNeighborsClassifier(n_neighbors=3, output_type='cupy')

        with self.assertRaises(NotImplementedError) as context:
            knn._fit_gpu(self.X)

        self.assertTrue('GPU is not supported' in str(context.exception))

    @patch('dasf.ml.neighbors.neighbors.is_gpu_supported', Mock(return_value=False))
    def test_knearestneighbors_fit_predict_gpu_no_gpu(self):
        knn = KNeighborsClassifier(n_neighbors=3, output_type='cupy')

        with self.assertRaises(NotImplementedError) as context:
            _ = knn._predict_gpu(self.X)

        self.assertTrue('GPU is not supported' in str(context.exception))
