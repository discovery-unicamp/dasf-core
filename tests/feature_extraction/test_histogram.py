#!/usr/bin/python3

import unittest

import dask.array as da
import numpy as np

from dasf.feature_extraction import Histogram

try:
    import cupy as cp
except ImportError:
    pass

class TestHistogram(unittest.TestCase):
    def setUp(self):
        self._data = np.array([0, 8, 6, 9, 7, 3, 5, 2, 3, 4, 1, 3, 2, 3, 4, 3, 2, 1, 9, 2])

    def test_histogram_cpu(self):
        data = self._data

        hist = Histogram(bins=5)

        bins = hist._transform_cpu(X=data)

        self.assertEqual(bins[0].shape, (5,))
        self.assertTrue(all([a == b for a, b in zip(bins[0], np.array([3, 9, 3, 2, 3]))]))

    def test_histogram_gpu(self):
        data = cp.array(self._data)

        hist = Histogram(bins=5)

        bins = hist._transform_gpu(X=data)

        self.assertEqual(bins[0].shape, (5,))
        self.assertTrue(all([a == b for a, b in zip(bins[0], cp.array([3, 9, 3, 2, 3]))]))

    def test_histogram_mcpu(self):
        data = da.from_array(self._data)

        hist = Histogram(bins=5, range=[0, 9])

        bins = hist._lazy_transform_cpu(X=data)

        self.assertEqual(bins[0].shape, (5,))
        self.assertTrue(all([a == b for a, b in zip(bins[0], np.array([3, 9, 3, 2, 3]))]))

    def test_histogram_mgpu(self):
        data = da.from_array(cp.array(self._data))

        hist = Histogram(bins=5, range=[0, 9])

        bins = hist._lazy_transform_gpu(X=data)

        self.assertEqual(bins[0].shape, (5,))
        self.assertTrue(all([a == b for a, b in zip(bins[0], cp.array([3, 9, 3, 2, 3]))]))

    def test_histogram_dask_without_range(self):
        data = da.from_array(self._data)

        hist = Histogram(bins=5)

        self.assertRaises(ValueError, hist._lazy_transform_gpu, X=data)
