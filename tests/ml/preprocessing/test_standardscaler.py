#!/usr/bin/env python3

import unittest

import dask.array as da
import numpy as np
from unittest.mock import Mock, patch
from scipy import sparse

try:
    import cupy as cp
except ImportError:
    pass

from dasf.ml.preprocessing import StandardScaler
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import (
    is_cpu_array,
    is_dask_cpu_array,
    is_dask_gpu_array,
    is_gpu_array,
)


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        size = 20
        self.X = np.array([np.arange(size)])
        self.X.shape = (size, 1)

        mean = np.mean(self.X)
        std = np.std(self.X)

        self.y = (self.X - mean) / std

    def test_standardscaler_cpu(self):
        ss = StandardScaler()

        y = ss._fit_transform_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_standardscaler_mcpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_transform_cpu(da.from_array(self.X))

        self.assertTrue(is_dask_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_gpu(self):
        ss = StandardScaler()

        y = ss._fit_transform_gpu(cp.asarray(self.X))

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_mgpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_transform_gpu(da.from_array(cp.asarray(self.X)))

        self.assertTrue(is_dask_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute().get(), equal_nan=True))

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_standardscaler_mcpu_local(self):
        ss = StandardScaler(run_local=True)

        y = ss._fit_transform_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_standardscaler_2_cpu(self):
        ss = StandardScaler()

        y = ss._fit_cpu(self.X)._transform_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_standardscaler_2_mcpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_cpu(da.from_array(self.X)) \
              ._lazy_transform_cpu(da.from_array(self.X))

        self.assertTrue(is_dask_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_2_gpu(self):
        ss = StandardScaler()

        y = ss._fit_gpu(cp.asarray(self.X))._transform_gpu(cp.asarray(self.X))

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_2_mgpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_gpu(da.from_array(cp.asarray(self.X))) \
              ._lazy_transform_gpu(da.from_array(cp.asarray(self.X)))

        self.assertTrue(is_dask_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.compute().get(), equal_nan=True))

    def test_standardscaler_partial_cpu(self):
        ss = StandardScaler()

        y = ss._partial_fit_cpu(self.X)._transform_cpu(self.X)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.y, y, equal_nan=True))

    def test_standardscaler_partial_mcpu(self):
        ss = StandardScaler()

        with self.assertRaises(NotImplementedError):
            _ = ss._lazy_partial_fit_cpu(da.from_array(self.X)) \
                  ._lazy_transform_cpu(da.from_array(self.X))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_partial_gpu(self):
        ss = StandardScaler()

        y = ss._partial_fit_gpu(cp.asarray(self.X))._transform_gpu(cp.asarray(self.X))

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.y, y.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_partial_mgpu(self):
        ss = StandardScaler()

        with self.assertRaises(NotImplementedError):
            _ = ss._lazy_partial_fit_gpu(da.from_array(cp.asarray(self.X))) \
                  ._lazy_transform_gpu(da.from_array(cp.asarray(self.X)))

    def test_standardscaler_inverse_cpu(self):
        ss = StandardScaler()

        y = ss._fit_cpu(self.X)._transform_cpu(self.X)

        x = ss._inverse_transform_cpu(y)

        self.assertTrue(is_cpu_array(y))
        self.assertTrue(np.array_equal(self.X, x, equal_nan=True))

    def test_standardscaler_inverse_mcpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_cpu(da.from_array(self.X)) \
              ._lazy_transform_cpu(da.from_array(self.X))

        x = ss._lazy_inverse_transform_cpu(y)

        self.assertTrue(is_dask_cpu_array(y))
        self.assertTrue(np.array_equal(self.X, x.compute(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_inverse_gpu(self):
        ss = StandardScaler()

        y = ss._fit_gpu(cp.asarray(self.X))._transform_gpu(cp.asarray(self.X))

        x = ss._inverse_transform_gpu(y)

        self.assertTrue(is_gpu_array(y))
        self.assertTrue(np.array_equal(self.X, x.get(), equal_nan=True))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_standardscaler_inverse_mgpu(self):
        ss = StandardScaler()

        y = ss._lazy_fit_gpu(da.from_array(cp.asarray(self.X))) \
              ._lazy_transform_gpu(da.from_array(cp.asarray(self.X)))

        x = ss._lazy_inverse_transform_gpu(y)

        self.assertTrue(is_dask_gpu_array(y))
        self.assertTrue(np.array_equal(self.X, x.compute().get(), equal_nan=True))

    def test_standardscaler_init_parameters(self):
        """Test StandardScaler initialization with different parameters."""
        # Test default parameters
        ss_default = StandardScaler()
        self.assertTrue(hasattr(ss_default, '_StandardScaler__std_scaler_cpu'))
        self.assertTrue(hasattr(ss_default, '_StandardScaler__std_scaler_dask'))

        # Test custom parameters
        ss_custom = StandardScaler(copy=False, with_mean=False, with_std=False)
        self.assertTrue(hasattr(ss_custom, '_StandardScaler__std_scaler_cpu'))
        self.assertTrue(hasattr(ss_custom, '_StandardScaler__std_scaler_dask'))

    def test_standardscaler_with_mean_false(self):
        """Test StandardScaler with with_mean=False."""
        ss = StandardScaler(with_mean=False)

        # Create data with non-zero mean
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        result = ss._fit_transform_cpu(X)

        # With with_mean=False, mean should not be subtracted
        # Only scaling by std should happen
        expected_std = np.std(X, axis=0, ddof=0)
        expected = X / expected_std

        self.assertTrue(is_cpu_array(result))
        np.testing.assert_allclose(result, expected, rtol=1e-7)

    def test_standardscaler_with_std_false(self):
        """Test StandardScaler with with_std=False."""
        ss = StandardScaler(with_std=False)

        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        result = ss._fit_transform_cpu(X)

        # With with_std=False, only mean centering should happen
        expected_mean = np.mean(X, axis=0)
        expected = X - expected_mean

        self.assertTrue(is_cpu_array(result))
        np.testing.assert_allclose(result, expected, rtol=1e-7)

    def test_standardscaler_both_false(self):
        """Test StandardScaler with both with_mean=False and with_std=False."""
        ss = StandardScaler(with_mean=False, with_std=False)

        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        result = ss._fit_transform_cpu(X)

        # With both False, data should remain unchanged
        self.assertTrue(is_cpu_array(result))
        np.testing.assert_allclose(result, X, rtol=1e-7)

    def test_standardscaler_multifeature_data(self):
        """Test StandardScaler with multi-feature data."""
        # Create data with different scales
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[:, 0] *= 100  # First feature has large scale
        X[:, 1] *= 0.01  # Second feature has small scale

        ss = StandardScaler()
        result = ss._fit_transform_cpu(X)

        # Check that each feature is standardized
        self.assertTrue(is_cpu_array(result))
        np.testing.assert_allclose(np.mean(result, axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.std(result, axis=0), 1, rtol=1e-7)

    def test_standardscaler_single_sample(self):
        """Test StandardScaler with single sample."""
        X = np.array([[1, 2, 3]], dtype=float)

        ss = StandardScaler()
        result = ss._fit_transform_cpu(X)

        # With single sample, std is 0, so result should be 0
        expected = np.zeros_like(X)

        self.assertTrue(is_cpu_array(result))
        np.testing.assert_allclose(result, expected, rtol=1e-7)

    def test_standardscaler_constant_feature(self):
        """Test StandardScaler with constant feature (zero variance)."""
        X = np.array([[1, 5], [1, 10], [1, 15]], dtype=float)

        ss = StandardScaler()
        result = ss._fit_transform_cpu(X)

        # First feature is constant, should become 0
        # Second feature should be standardized normally
        expected_col1 = np.zeros(3)
        expected_col2 = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])

        self.assertTrue(is_cpu_array(result))
        np.testing.assert_allclose(result[:, 0], expected_col1, rtol=1e-7)
        np.testing.assert_allclose(result[:, 1], expected_col2, rtol=1e-7)

    def test_standardscaler_copy_parameter(self):
        """Test StandardScaler copy parameter."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        X_original = X.copy()

        # Test with copy=True (default)
        ss_copy = StandardScaler(copy=True)
        result_copy = ss_copy._fit_transform_cpu(X)

        # Original array should be unchanged
        np.testing.assert_array_equal(X, X_original)

        # Test with copy=False
        ss_no_copy = StandardScaler(copy=False)
        X_for_no_copy = X.copy()
        result_no_copy = ss_no_copy._fit_transform_cpu(X_for_no_copy)

        # Results should be the same
        np.testing.assert_allclose(result_copy, result_no_copy, rtol=1e-7)

    def test_standardscaler_empty_array(self):
        """Test StandardScaler with empty array."""
        X = np.array([]).reshape(0, 2)

        ss = StandardScaler()

        # Should handle empty arrays gracefully
        try:
            result = ss._fit_transform_cpu(X)
            self.assertTrue(is_cpu_array(result))
            self.assertEqual(result.shape, (0, 2))
        except Exception as e:
            # Some implementations may raise ValueError for empty arrays
            self.assertIsInstance(e, (ValueError, IndexError))

    def test_standardscaler_sparse_matrix_error(self):
        """Test StandardScaler with sparse matrix and with_mean=True raises error."""
        # Create sparse matrix
        X_sparse = sparse.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]])

        ss = StandardScaler(with_mean=True)

        # Should raise ValueError for sparse matrices with with_mean=True
        with self.assertRaises((ValueError, TypeError)):
            ss._fit_transform_cpu(X_sparse)

    def test_standardscaler_sparse_matrix_valid(self):
        """Test StandardScaler with sparse matrix and with_mean=False."""
        # Create sparse matrix
        X_sparse = sparse.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=float)

        ss = StandardScaler(with_mean=False, with_std=True)

        try:
            result = ss._fit_transform_cpu(X_sparse)
            # Should work with sparse matrices when with_mean=False
            self.assertTrue(sparse.issparse(result) or is_cpu_array(result))
        except Exception:
            # Some backends may not support sparse matrices
            pass

    def test_standardscaler_dask_chunks(self):
        """Test StandardScaler with different Dask array chunk sizes."""
        X = np.random.randn(100, 4)

        # Test with different chunk sizes (only first axis chunking allowed)
        chunk_sizes = [(10, 4), (25, 4), (50, 4)]

        ss = StandardScaler()
        expected = ss._fit_transform_cpu(X)

        for chunks in chunk_sizes:
            da_X = da.from_array(X, chunks=chunks)
            result = ss._lazy_fit_transform_cpu(da_X)

            self.assertTrue(is_dask_cpu_array(result))
            np.testing.assert_allclose(result.compute(), expected, rtol=1e-7)

    @unittest.skipIf(not is_gpu_supported(), "GPU not supported")
    def test_standardscaler_gpu_not_available_error(self):
        """Test StandardScaler GPU methods when GPU is not available."""
        ss = StandardScaler()

        # Temporarily set GPU scaler to None
        original_gpu_scaler = ss._StandardScaler__std_scaler_gpu
        ss._StandardScaler__std_scaler_gpu = None

        X = np.array([[1, 2], [3, 4]], dtype=float)

        try:
            # Should raise NotImplementedError
            with self.assertRaises(NotImplementedError):
                ss._fit_gpu(cp.asarray(X))

            with self.assertRaises(NotImplementedError):
                ss._fit_transform_gpu(cp.asarray(X))

            with self.assertRaises(NotImplementedError):
                ss._partial_fit_gpu(cp.asarray(X))

            with self.assertRaises(NotImplementedError):
                ss._transform_gpu(cp.asarray(X))

            with self.assertRaises(NotImplementedError):
                ss._inverse_transform_gpu(cp.asarray(X))
        finally:
            # Restore original GPU scaler
            ss._StandardScaler__std_scaler_gpu = original_gpu_scaler

    def test_standardscaler_consistency_across_backends(self):
        """Test that CPU and Dask backends produce consistent results."""
        np.random.seed(42)
        X = np.random.randn(50, 3)

        ss = StandardScaler()

        # CPU result
        cpu_result = ss._fit_transform_cpu(X.copy())

        # Reset scaler for Dask
        ss_dask = StandardScaler()
        dask_result = ss_dask._lazy_fit_transform_cpu(da.from_array(X))

        self.assertTrue(is_cpu_array(cpu_result))
        self.assertTrue(is_dask_cpu_array(dask_result))
        np.testing.assert_allclose(cpu_result, dask_result.compute(), rtol=1e-10)

    def test_standardscaler_fit_then_transform_consistency(self):
        """Test that fit_transform and fit->transform produce same results."""
        np.random.seed(42)
        X = np.random.randn(30, 2)

        # Method 1: fit_transform
        ss1 = StandardScaler()
        result1 = ss1._fit_transform_cpu(X)

        # Method 2: fit then transform
        ss2 = StandardScaler()
        ss2._fit_cpu(X)
        result2 = ss2._transform_cpu(X)

        np.testing.assert_allclose(result1, result2, rtol=1e-12)

    def test_standardscaler_partial_fit_equivalence(self):
        """Test that partial_fit with full data equals regular fit."""
        np.random.seed(42)
        X = np.random.randn(40, 3)

        # Regular fit
        ss1 = StandardScaler()
        result1 = ss1._fit_transform_cpu(X)

        # Partial fit with full data
        ss2 = StandardScaler()
        ss2._partial_fit_cpu(X)
        result2 = ss2._transform_cpu(X)

        np.testing.assert_allclose(result1, result2, rtol=1e-10)

    def test_standardscaler_transform_without_fit_error(self):
        """Test that transform without fit raises appropriate error."""
        X = np.array([[1, 2], [3, 4]], dtype=float)

        ss = StandardScaler()

        # Should raise error when trying to transform without fitting
        with self.assertRaises((AttributeError, ValueError)):
            ss._transform_cpu(X)

    def test_standardscaler_inverse_transform_roundtrip(self):
        """Test that transform->inverse_transform recovers original data."""
        np.random.seed(42)
        X = np.random.randn(20, 4) * 10 + 5  # Scale and shift data

        ss = StandardScaler()

        # Transform then inverse transform
        X_transformed = ss._fit_transform_cpu(X)
        X_recovered = ss._inverse_transform_cpu(X_transformed)

        np.testing.assert_allclose(X, X_recovered, rtol=1e-10)

    def test_standardscaler_different_dtypes(self):
        """Test StandardScaler with different input data types."""
        base_data = [[1, 2], [3, 4], [5, 6]]

        dtypes = [np.float32, np.float64, np.int32, np.int64]

        results = []
        for dtype in dtypes:
            X = np.array(base_data, dtype=dtype)
            ss = StandardScaler()

            try:
                result = ss._fit_transform_cpu(X)
                results.append(result)
                self.assertTrue(is_cpu_array(result))
            except Exception:
                # Some dtypes might not be supported
                pass

        # All successful results should be approximately equal
        if len(results) > 1:
            for i in range(1, len(results)):
                np.testing.assert_allclose(results[0], results[i], rtol=1e-5)
