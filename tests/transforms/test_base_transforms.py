#!/usr/bin/env python3

import functools
import os
import shutil
import tempfile
import unittest

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

try:
    import cudf
    import cupy as cp
    import dask_cudf as dcudf
except ImportError:
    pass

from dasf.transforms.base import MappedTransform, ReductionTransform
from dasf.utils.funcs import is_gpu_supported
from dasf.utils.types import (
    is_cpu_array,
    is_cpu_dataframe,
    is_dask_cpu_array,
    is_dask_cpu_dataframe,
    is_dask_gpu_array,
    is_dask_gpu_dataframe,
    is_gpu_array,
    is_gpu_dataframe,
)


class TestMappedTransform(unittest.TestCase):
    def __internal_max_function(self, block, nxp=np, block_info=None):
        # Using `nxp` parameter to avoid conflicts
        return nxp.asarray([[nxp.max(block)]])

    def __internal_max_min_function(self, block, nxp=np, block_info=None):
        # Using `nxp` parameter to avoid conflicts
        return nxp.asarray([[nxp.min(block), nxp.max(block)]])

    def test_rechunk_max_min_cpu(self):
        X = np.random.random((40, 40, 40))

        mapped = MappedTransform(function=self.__internal_max_min_function,
                                 output_chunk=(1, 2))

        X_t = mapped._transform_cpu(X, nxp=np)

        # Numpy does not use blocks.
        self.assertEqual(X_t.shape, (1, 2))

        self.assertTrue(is_cpu_array(X_t))
        
    def test_rechunk_max_min_mcpu(self):
        X = da.random.random((40, 40, 40), chunks=(10, 40, 40))

        mapped = MappedTransform(function=self.__internal_max_min_function,
                                 output_chunk=(1, 2))

        X_t = mapped._lazy_transform_cpu(X, nxp=np)

        self.assertEqual(X_t.shape, (1, 2))
        # We split data in 4 chunks of axis 0. If the new chunk size is
        # (1, 2), it means that the final size is (1 * 4, 2).
        self.assertEqual(X_t.compute().shape, (1, 2))

        self.assertTrue(is_dask_cpu_array(X_t))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_rechunk_max_min_gpu(self):
        X = cp.random.random((40, 40, 40))

        mapped = MappedTransform(function=self.__internal_max_min_function,
                                 output_chunk=(1, 2))

        X_t = mapped._transform_gpu(X, nxp=cp)

        # Numpy does not use blocks.
        self.assertEqual(X_t.shape, (1, 2))

        self.assertTrue(is_gpu_array(X_t))

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_rechunk_max_min_mgpu(self):
        # We need to convert to Dask to use Cupy random.
        X_cp = cp.random.random((40, 40, 40))

        X = da.from_array(X_cp, chunks=(10, 40, 40), meta=cp.array(()))

        mapped = MappedTransform(function=self.__internal_max_min_function,
                                 output_chunk=(1, 2))

        X_t = mapped._lazy_transform_gpu(X, nxp=cp)

        self.assertEqual(X_t.shape, (1, 2))
        # We split data in 4 chunks of axis 0. If the new chunk size is
        # (1, 2), it means that the final size is (1 * 4, 2).
        self.assertEqual(X_t.compute().get().shape, (1, 2))


class TestReductionTransformArray(unittest.TestCase):
    def __internal_chunk_array(self, block, axis=None, keepdims=False, xp=np):
        # Using `xp` parameter to avoid conflicts
        return xp.array([xp.min(block), xp.max(block)])

    def __internal_aggregate_array(self, block, axis=None, keepdims=False, xp=np):
        # Using `xp` parameter to avoid conflicts
        block = xp.array(block)

        return xp.array([xp.min(block), xp.max(block)])

    def test_reduction_min_max_cpu(self):
        X = np.arange(40 * 40 * 40)
        X.shape = (40, 40, 40)

        reduction = ReductionTransform(func_aggregate=self.__internal_aggregate_array,
                                       func_chunk=self.__internal_chunk_array,
                                       output_size=[0, 0])

        X_t = reduction._transform_cpu(X)

        # Numpy does not use blocks.
        self.assertEqual(len(X_t), 2)
        self.assertTrue(is_cpu_array(X_t))

        self.assertEqual(X_t[0], 0)
        self.assertEqual(X_t[1], 40 * 40 * 40 - 1)

    def test_reduction_min_max_mcpu(self):
        X = np.arange(40 * 40 * 40)
        X.shape = (40, 40, 40)

        X = da.from_array(X, chunks=(10, 10, 10))

        reduction = ReductionTransform(func_aggregate=self.__internal_aggregate_array,
                                       func_chunk=self.__internal_chunk_array,
                                       output_size=[0, 0])

        X_t = reduction._lazy_transform_cpu(X, concatenate=False)

        self.assertTrue(is_dask_cpu_array(X_t))
        self.assertEqual(len(X_t.compute()), 2)

        self.assertEqual(X_t.compute()[0], 0)
        self.assertEqual(X_t.compute()[1], 40 * 40 * 40 - 1)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reduction_min_max_gpu(self):
        X = cp.arange(40 * 40 * 40)
        X.shape = (40, 40, 40)

        reduction = ReductionTransform(func_aggregate=self.__internal_aggregate_array,
                                       func_chunk=self.__internal_chunk_array,
                                       output_size=[0, 0])

        X_t = reduction._transform_gpu(X)

        # Numpy does not use blocks.
        self.assertEqual(len(X_t), 2)
        self.assertTrue(is_gpu_array(X_t))

        self.assertEqual(X_t[0].get(), 0)
        self.assertEqual(X_t[1].get(), 40 * 40 * 40 - 1)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reduction_min_max_mgpu(self):
        X = cp.arange(40 * 40 * 40)
        X.shape = (40, 40, 40)

        X = da.from_array(X, chunks=(10, 10, 10), meta=cp.array(()))

        reduction = ReductionTransform(func_aggregate=self.__internal_aggregate_array,
                                       func_chunk=self.__internal_chunk_array,
                                       output_size=[0, 0])

        X_t = reduction._lazy_transform_gpu(X, concatenate=False)

        self.assertTrue(is_dask_gpu_array(X_t))
        self.assertEqual(len(X_t.compute()), 2)

        self.assertEqual(X_t.compute()[0].get(), 0)
        self.assertEqual(X_t.compute()[1].get(), 40 * 40 * 40 - 1)


class TestReductionTransformDataFrame(unittest.TestCase):
    def _internal_chunk_max(self, row):
        if row['A'] > row['B'] and row['A'] > row['C']:
            return row['A']
        elif row['B'] > row['A'] and row['B'] > row['C']:
            return row['B']
        else:
            return row['C']

    def _internal_chunk_min(self, row):
        if row['A'] < row['B'] and row['A'] < row['C']:
            return row['A']
        elif row['B'] < row['A'] and row['B'] < row['C']:
            return row['B']
        else:
            return row['C']

    def _internal_aggregate_min_max(self, pds, xd):
        return xd.DataFrame({
            'min': [pds['min'].min()],
            'max': [pds['max'].max()]
        })

    def _internal_chunk_partition_cpu(self, block, axis=None, keepdims=False, xp=None):
        pds_max = block.apply(self._internal_chunk_min, axis=1)
        pds_min = block.apply(self._internal_chunk_max, axis=1)

        pds = pd.DataFrame({'min': pds_min, 'max': pds_max})
        return self._internal_aggregate_min_max(pds, xd=pd)

    def _internal_chunk_partition_gpu(self, block, axis=None, keepdims=False, xp=None):
        pds_max = block.apply(self._internal_chunk_min, axis=1)
        pds_min = block.apply(self._internal_chunk_max, axis=1)

        pds = cudf.DataFrame({'min': pds_min, 'max': pds_max})
        return self._internal_aggregate_min_max(pds, xd=cudf)

    def _internal_aggregate_series_cpu(self, block, axis=None, keepdims=False, xp=None):
        return self._internal_aggregate_min_max(block, xd=pd)

    def _internal_aggregate_series_gpu(self, block, axis=None, keepdims=False, xp=None):
        return self._internal_aggregate_min_max(block, xd=cudf)

    def test_reduction_min_max_cpu(self):
        df = pd.DataFrame({
            'A': range(0, 1000),
            'B': range(0, 1000),
            'C': range(0, 1000)
        })

        reduction = ReductionTransform(func_aggregate=self._internal_aggregate_series_cpu,
                                       func_chunk=self._internal_chunk_partition_cpu,
                                       output_size={'min': 'int64',
                                                    'max': 'int64'})

        X_t = reduction._transform_cpu(X=df)

        self.assertEqual(len(X_t.iloc[0]), 2)

        self.assertEqual(X_t['min'].iloc[0], 0)
        self.assertEqual(X_t['max'].iloc[0], 1000 - 1)

    def test_reduction_min_max_mcpu(self):
        df = pd.DataFrame({
            'A': range(0, 1000),
            'B': range(0, 1000),
            'C': range(0, 1000)
        })

        ddf = dd.from_pandas(df, npartitions=8)

        reduction = ReductionTransform(func_aggregate=self._internal_aggregate_series_cpu,
                                       func_chunk=self._internal_chunk_partition_cpu,
                                       output_size={'min': 'int64',
                                                    'max': 'int64'})

        X_t = reduction._lazy_transform_cpu(X=ddf, axis=[0])

        self.assertTrue(is_dask_cpu_dataframe(X_t))
        self.assertEqual(len(X_t.compute().iloc[0]), 2)

        self.assertEqual(X_t.compute().iloc[0]['min'], 0)
        self.assertEqual(X_t.compute().iloc[0]['max'], 1000 - 1)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reduction_min_max_gpu(self):
        df = cudf.DataFrame({
            'A': range(0, 1000),
            'B': range(0, 1000),
            'C': range(0, 1000)
        })

        reduction = ReductionTransform(func_aggregate=self._internal_aggregate_series_gpu,
                                       func_chunk=self._internal_chunk_partition_gpu,
                                       output_size={'min': 'int64',
                                                    'max': 'int64'})

        X_t = reduction._transform_gpu(X=df)

        self.assertEqual(len(X_t.iloc[0]), 2)

        self.assertEqual(X_t['min'].iloc[0], 0)
        self.assertEqual(X_t['max'].iloc[0], 1000 - 1)

    @unittest.skipIf(not is_gpu_supported(),
                     "not supported CUDA in this platform")
    def test_reduction_min_max_mgpu(self):
        df = cudf.DataFrame({
            'A': range(0, 1000),
            'B': range(0, 1000),
            'C': range(0, 1000)
        })

        ddf = dcudf.from_cudf(df, npartitions=8)

        reduction = ReductionTransform(func_aggregate=self._internal_aggregate_series_gpu,
                                       func_chunk=self._internal_chunk_partition_gpu,
                                       output_size={'min': 'int64',
                                                    'max': 'int64'})

        X_t = reduction._lazy_transform_gpu(X=ddf, axis=[0])

        self.assertTrue(is_dask_gpu_dataframe(X_t))
        self.assertEqual(len(X_t.compute().iloc[0]), 2)

        self.assertEqual(X_t.compute().iloc[0]['min'], 0)
        self.assertEqual(X_t.compute().iloc[0]['max'], 1000 - 1)
