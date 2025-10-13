#!/usr/bin/env python3

import unittest

import dask.array as da
from unittest.mock import Mock, patch

try:
    import cupy as cp
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
except ImportError:
    pass

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from dasf.ml.xgboost import XGBRegressor
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import (
    is_cpu_array,
    is_dask_cpu_array,
    is_dask_gpu_array,
    is_gpu_array,
)


class TestXGBRegressor(unittest.TestCase):
    def setUp(self):
        self.n_samples = 1000
        self.n_features = 10
        random_state = 42

        self.X, self.y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=0.1,
            random_state=random_state
        )

    def test_xgb_regressor_cpu_fit_predict(self):
        xgb = XGBRegressor(n_estimators=10, random_state=42)

        xgb._fit_cpu(self.X, self.y)
        y_pred = xgb._predict_cpu(self.X)

        self.assertTrue(is_cpu_array(y_pred))
        self.assertEqual(len(y_pred), len(self.y))

        mse = mean_squared_error(self.y, y_pred)
        self.assertLess(mse, 1000.0)

    def test_xgb_regressor_mcpu_fit_predict(self):
        from dask.distributed import Client, LocalCluster

        with LocalCluster(processes=False) as cluster:
            with Client(cluster):
                xgb = XGBRegressor(n_estimators=10, random_state=42)

                da_X = da.from_array(self.X, chunks=(100, self.n_features))
                da_y = da.from_array(self.y, chunks=100)

                xgb._lazy_fit_cpu(da_X, da_y)
                y_pred = xgb._lazy_predict_cpu(da_X)

                self.assertTrue(is_dask_cpu_array(y_pred))
                y_pred_computed = y_pred.compute()
                self.assertEqual(len(y_pred_computed), len(self.y))

                mse = mean_squared_error(self.y, y_pred_computed)
                self.assertLess(mse, 1000.0)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_xgb_regressor_gpu_fit_predict(self):
        xgb = XGBRegressor(n_estimators=10, random_state=42)

        cp_X = cp.asarray(self.X)
        cp_y = cp.asarray(self.y)

        xgb._fit_gpu(cp_X, cp_y)
        y_pred = xgb._predict_gpu(cp_X)

        self.assertTrue(is_gpu_array(y_pred))
        y_pred_cpu = y_pred.get()
        self.assertEqual(len(y_pred_cpu), len(self.y))

        mse = mean_squared_error(self.y, y_pred_cpu)
        self.assertLess(mse, 1000.0)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_xgb_regressor_mgpu_fit_predict(self):
        with LocalCUDACluster() as cluster:
            client = Client(cluster)

            xgb = XGBRegressor(n_estimators=10, random_state=42)

            da_X = da.from_array(cp.asarray(self.X), chunks=(100, self.n_features))
            da_y = da.from_array(cp.asarray(self.y), chunks=100)

            xgb._lazy_fit_gpu(da_X, da_y)
            y_pred = xgb._lazy_predict_gpu(da_X)

            self.assertTrue(is_dask_gpu_array(y_pred))
            y_pred_cpu = y_pred.compute().get()
            self.assertEqual(len(y_pred_cpu), len(self.y))

            mse = mean_squared_error(self.y, y_pred_cpu)
            self.assertLess(mse, 1000.0)

            client.close()

    def test_xgb_regressor_parameters(self):
        xgb = XGBRegressor(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        xgb._fit_cpu(self.X, self.y)
        y_pred = xgb._predict_cpu(self.X)

        self.assertTrue(is_cpu_array(y_pred))
        self.assertEqual(len(y_pred), len(self.y))

    def test_xgb_regressor_small_dataset(self):
        X_small = self.X[:50]
        y_small = self.y[:50]

        xgb = XGBRegressor(n_estimators=5, random_state=42)

        xgb._fit_cpu(X_small, y_small)
        y_pred = xgb._predict_cpu(X_small)

        self.assertTrue(is_cpu_array(y_pred))
        self.assertEqual(len(y_pred), len(y_small))

    @patch('dasf.utils.decorators.is_gpu_supported', Mock(return_value=False))
    @patch('dasf.utils.decorators.is_dask_supported', Mock(return_value=True))
    @patch('dasf.utils.decorators.is_dask_gpu_supported', Mock(return_value=False))
    def test_xgb_regressor_mcpu_local(self):
        from dask.distributed import Client, LocalCluster

        with LocalCluster(processes=False) as cluster:
            with Client(cluster):
                xgb = XGBRegressor(n_estimators=10, random_state=42, run_local=True)

                da_X = da.from_array(self.X, chunks=(100, self.n_features))
                da_y = da.from_array(self.y, chunks=100)

                xgb.fit(da_X, da_y)
                y_pred = xgb.predict(da_X)

                # When using run_local=True with Dask, result might be a dask array
                if is_dask_cpu_array(y_pred):
                    y_pred = y_pred.compute()
                self.assertTrue(is_cpu_array(y_pred))
                self.assertEqual(len(y_pred), len(self.y))
