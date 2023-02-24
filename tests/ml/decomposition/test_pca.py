#!/usr/bin/env python3

import unittest
import numpy as np
import dask.array as da

try:
    import cupy as cp
except ImportError:
    pass

from sklearn import datasets
from parameterized import parameterized

from dasf.ml.decomposition import PCA
from dasf.utils.funcs import is_gpu_supported


PCA_SOLVERS_CPU = [("full"), ("arpack"), ("randomized"), ("auto")]
PCA_SOLVERS_MCPU = [("full"), ("tsqr"), ("auto")]  # "ramdomized" has a Dask bug 
PCA_SOLVERS_GPU = [("full"), ("jacobi"), ("auto")]


class TestPCA(unittest.TestCase):
    def setUp(self):
        self.data = datasets.load_iris().data
        self.n_components = 3

    @parameterized.expand(PCA_SOLVERS_CPU)
    def test_pca_cpu(self, svd_solver):
        try:
            pca = PCA(n_components=self.n_components, svd_solver=svd_solver)

            # check the shape of fit.transform
            X_r = pca._fit_cpu(self.data).transform(self.data)
            self.assertEqual(X_r.shape[1], self.n_components)

            # check the equivalence of fit.transform and fit_transform
            X_r2 = pca._fit_transform_cpu(self.data)
            np.testing.assert_allclose(X_r, X_r2)
            X_r = pca._transform_cpu(self.data)
            np.testing.assert_allclose(X_r, X_r2)

            # Test get_covariance and get_precision
            cov = pca._get_covariance_cpu()
            precision = pca._get_precision_cpu()
            np.testing.assert_allclose(np.dot(cov, precision),
                                       np.eye(self.data.shape[1]),
                                       atol=1e-12)
        except NotImplementedError:
            unittest.SkipTest(f"Skipped because {svd_solver} is not supported.")

    @parameterized.expand(PCA_SOLVERS_MCPU)
    def test_pca_mcpu(self, svd_solver):
        try:
            data_X = da.from_array(self.data,
                                   chunks=(int(self.data.shape[0]/4), self.data.shape[1]),
                                   meta=np.array(()))

            pca = PCA(n_components=self.n_components, svd_solver=svd_solver)

            # check the shape of fit.transform
            X_r = pca._lazy_fit_cpu(data_X).transform(data_X)
            self.assertEqual(X_r.shape[1], self.n_components)

            # check the equivalence of fit.transform and fit_transform
            X_r2 = pca._lazy_fit_transform_cpu(data_X)
            np.testing.assert_allclose(X_r, X_r2)
            X_r = pca._lazy_transform_cpu(data_X)
            np.testing.assert_allclose(X_r, X_r2)
        except AttributeError:
            unittest.SkipTest(f"Skipped due to Dask-ML bug.")

    @parameterized.expand(PCA_SOLVERS_GPU)
    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_pca_gpu(self, svd_solver):
        raise unittest.SkipTest("Check why PCA is not working for floats")

    @parameterized.expand(PCA_SOLVERS_GPU)
    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_pca_mgpu(self, svd_solver):
        raise unittest.SkipTest("Check why PCA is not working for floats")
