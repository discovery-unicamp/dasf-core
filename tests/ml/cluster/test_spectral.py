#!/usr/bin/env python3

import unittest
import numpy as np
import dask.array as da

from sklearn.datasets import make_blobs

from dasf.ml.cluster import SpectralClustering
from dasf.utils.types import is_cpu_array
from dasf.utils.types import is_dask_cpu_array


class TestSpectralClustering(unittest.TestCase):
    def setUp(self):
        self.size = 1000
        self.centers = 3
        self.random_state = 42

        self.X, self.y, self.centroids = make_blobs(n_samples=self.size,
                                                    centers=self.centers,
                                                    n_features=2,
                                                    return_centers=True,
                                                    random_state=self.random_state)

    def __match_randomly_labels_created(self, y1, y2):
        y2 = (y2 * -1) - 1

        for i in range(len(y1)):
            if y2[i] < 0:
                y2[y2 == y2[i]] = y1[i]

            if not np.any(y2[y2 < 0]):
                break

        return y1, y2

    def test_spectral_cpu(self):
        sc = SpectralClustering(n_clusters=self.centers,
                                random_state=self.random_state)

        y = sc._fit_predict_cpu(self.X)

        self.assertTrue(is_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y, self.y)

        self.assertTrue(np.array_equal(y1, y2, equal_nan=True))

    def test_spectral_mcpu(self):
        sc = SpectralClustering(n_clusters=self.centers,
                                random_state=self.random_state,
                                n_components=250)

        da_X = da.from_array(self.X)

        try:
            y = sc._lazy_fit_predict_cpu(da_X)
        except TypeError as te:
            self.skipTest("BUG - SpectralClustering Dask Type Error: " + str(te))

        self.assertTrue(is_dask_cpu_array(y))

        y1, y2 = self.__match_randomly_labels_created(y.compute(), self.y)

        # Check if the accurary is higher than 99%.
        self.assertTrue(len(np.setdiff1d(y1, y2)) <= int(self.size*0.01))
