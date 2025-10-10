#!/usr/bin/env python3

import unittest

import dask.array as da
import numpy as np
from dask.distributed import Client

try:
    import cupy as cp
    from dask_cuda import LocalCUDACluster
except ImportError:
    pass

from unittest.mock import Mock, patch
from parameterized import parameterized
from sklearn import datasets

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
            unittest.SkipTest("Skipped due to Dask-ML bug.")

    @parameterized.expand(PCA_SOLVERS_GPU)
    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_pca_gpu(self, svd_solver):
        try:
            data_X = cp.asarray(self.data)

            pca = PCA(n_components=self.n_components, svd_solver=svd_solver)

            # check the shape of fit.transform
            X_r = pca._fit_gpu(data_X).transform(data_X)
            self.assertEqual(X_r.shape[1], self.n_components)

            # check the equivalence of fit.transform and fit_transform
            X_r2 = pca._fit_transform_gpu(data_X)
            np.testing.assert_allclose(X_r.get(), X_r2.get())
            X_r = pca._transform_gpu(data_X)
            np.testing.assert_allclose(X_r.get(), X_r2.get())
        except AttributeError:
            unittest.SkipTest("Skipped due to Dask-ML bug.")

    @parameterized.expand(PCA_SOLVERS_GPU)
    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_pca_mgpu(self, svd_solver):
        try:
            with LocalCUDACluster() as cluster:
                _ = Client(cluster)

                data_X = da.from_array(self.data,
                                       chunks=(int(self.data.shape[0]/4),
                                               self.data.shape[1]),
                                       meta=cp.array(()))

                pca = PCA(n_components=self.n_components, svd_solver=svd_solver)

                # check the shape of fit.transform
                X_r = pca._lazy_fit_gpu(data_X).transform(data_X)
                self.assertEqual(X_r.compute().shape[1], self.n_components)

                # check the equivalence of fit.transform and fit_transform
                X_r2 = pca._lazy_fit_transform_gpu(data_X)
                np.testing.assert_allclose(X_r.compute().get(), X_r2.compute().get())
                X_r = pca._lazy_transform_gpu(data_X)
                np.testing.assert_allclose(X_r.compute().get(), X_r2.compute().get())
        except AttributeError:
            unittest.SkipTest("Skipped due to Dask-ML bug.")

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_pca_mgpu_without_dask_client(self):
        with self.assertRaises(NotImplementedError):
            data_X = da.from_array(self.data,
                                   chunks=(int(self.data.shape[0]/4), self.data.shape[1]),
                                   meta=cp.array(()))

            pca = PCA(n_components=self.n_components)

            # check the shape of fit.transform
            _ = pca._lazy_fit_gpu(data_X).transform(data_X)

    @patch('dasf.ml.decomposition.pca.is_gpu_supported', Mock(return_value=False))
    def test_pca_cpu_cov_and_prec(self):
        pca = PCA(n_components=self.n_components)

        # check the shape of fit.transform
        X_r = pca._fit_cpu(self.data).transform(self.data)
        self.assertEqual(X_r.shape[1], self.n_components)

        # check the equivalence of fit.transform and fit_transform
        X_r2 = pca._fit_transform_cpu(self.data)
        np.testing.assert_allclose(X_r, X_r2)
        X_r = pca._transform_cpu(self.data)
        np.testing.assert_allclose(X_r, X_r2)

        # Test get_covariance and get_precision
        cov = pca.get_covariance()
        precision = pca.get_precision()
        np.testing.assert_allclose(np.dot(cov, precision),
                                   np.eye(self.data.shape[1]),
                                   atol=1e-12)

    @patch('dasf.ml.decomposition.pca.is_gpu_supported', Mock(return_value=True))
    def test_pca_cpu_cov_exception(self):
        with self.assertRaises(NotImplementedError):
            pca = PCA(n_components=self.n_components)

            # check the shape of fit.transform
            X_r = pca._fit_cpu(self.data).transform(self.data)
            self.assertEqual(X_r.shape[1], self.n_components)

            # check the equivalence of fit.transform and fit_transform
            X_r2 = pca._fit_transform_cpu(self.data)
            np.testing.assert_allclose(X_r, X_r2)
            X_r = pca._transform_cpu(self.data)
            np.testing.assert_allclose(X_r, X_r2)

            # Test get_covariance and get_precision
            _ = pca.get_covariance()

    @patch('dasf.ml.decomposition.pca.is_gpu_supported', Mock(return_value=True))
    def test_pca_cpu_prec_exception(self):
        with self.assertRaises(NotImplementedError):
            pca = PCA(n_components=self.n_components)

            # check the shape of fit.transform
            X_r = pca._fit_cpu(self.data).transform(self.data)
            self.assertEqual(X_r.shape[1], self.n_components)

            # check the equivalence of fit.transform and fit_transform
            X_r2 = pca._fit_transform_cpu(self.data)
            np.testing.assert_allclose(X_r, X_r2)
            X_r = pca._transform_cpu(self.data)
            np.testing.assert_allclose(X_r, X_r2)

            # Test get_covariance and get_precision
            _ = pca.get_precision()

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    @patch('dasf.ml.decomposition.pca.is_gpu_supported', Mock(return_value=False))
    def test_pca_gpu_fit_exception(self):
        with self.assertRaises(NotImplementedError):
            data_X = cp.asarray(self.data)

            pca = PCA(n_components=self.n_components)

            # check the shape of fit.transform
            _ = pca._fit_gpu(data_X).transform(data_X)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    @patch('dasf.ml.decomposition.pca.is_gpu_supported', Mock(return_value=False))
    def test_pca_gpu_fit_transform_exception(self):
        with self.assertRaises(NotImplementedError):
            data_X = cp.asarray(self.data)

            pca = PCA(n_components=self.n_components)

            # check the equivalence of fit.transform and fit_transform
            _ = pca._fit_transform_gpu(data_X)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    @patch('dasf.ml.decomposition.pca.is_gpu_supported', Mock(return_value=False))
    def test_pca_gpu_transform_exception(self):
        with self.assertRaises(NotImplementedError):
            data_X = cp.asarray(self.data)

            pca = PCA(n_components=self.n_components)

            _ = pca._transform_gpu(data_X)
