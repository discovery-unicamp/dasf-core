#!/usr/bin/env python3

import unittest

import numpy as np

from mock import MagicMock

from dasf.pipeline import Pipeline
from dasf.datasets import DatasetArray
from dasf.transforms.base import Transform


class Dataset_A(DatasetArray):
    def load(self):
        self._data = np.arange(10)
        return self._data


class Transform_A(Transform):
    def transform(self, X):
        return X + 2


class Transform_B(Transform):
    def transform(self, X):
        return X - 2


class Transform_C(Transform):
    def transform(self, X):
        return X * 2


class Transform_D(Transform):
    def transform(self, X):
        return X / 2


class Transform_E(Transform):
    def transform_new(self, X):
        return X + 4


def transform_f(X):
    return X - 4


class TestPipeline(unittest.TestCase):
    def test_pipeline_creation(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        t_A = Transform_A()
        t_B = Transform_B()

        pipeline = Pipeline("Test Pipeline Creation")

        pipeline = pipeline.add(t_A, X=dataset_A) \
                           .add(t_B, X=t_A)

        with self.assertLogs('DASF', level='INFO') as plogs:
            pipeline.run()

            self.assertIn('Pipeline run successfully', plogs.output[-1])

            all_output = '\n'.join(plogs.output)

            self.assertIn('Dataset_A', all_output)
            self.assertIn('Transform_A', all_output)
            self.assertIn('Transform_B', all_output)

    def test_pipeline_creation_in_sequence(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        t_A = Transform_A()
        t_B = Transform_B()
        t_C = Transform_C()
        t_D = Transform_D()

        pipeline = Pipeline("Test Pipeline Creation 1")

        pipeline = pipeline.add(t_A, X=dataset_A) \
                           .add(t_B, X=t_A)

        with self.assertLogs('DASF', level='INFO') as plogs:
            pipeline.run()

            self.assertIn('Pipeline run successfully', plogs.output[-1])

            all_output = '\n'.join(plogs.output)

            self.assertIn('Dataset_A', all_output)
            self.assertIn('Transform_A', all_output)
            self.assertIn('Transform_B', all_output)
            self.assertNotIn('Transform_C', all_output)
            self.assertNotIn('Transform_D', all_output)

        pipeline = Pipeline("Test Pipeline Creation 2")

        pipeline = pipeline.add(t_C, X=dataset_A) \
                           .add(t_D, X=t_C)

        with self.assertLogs('DASF', level='INFO') as plogs:
            pipeline.run()

            self.assertIn('Pipeline run successfully', plogs.output[-1])

            all_output = '\n'.join(plogs.output)

            self.assertIn('Dataset_A', all_output)
            self.assertNotIn('Transform_A', all_output)
            self.assertNotIn('Transform_B', all_output)
            self.assertIn('Transform_C', all_output)
            self.assertIn('Transform_D', all_output)

    def test_pipeline_non_transformers(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        t_E = Transform_E()

        pipeline = Pipeline("Test Pipeline Creation Non Transformers")

        pipeline = pipeline.add(t_E.transform_new, X=dataset_A) \
                           .add(transform_f, X=t_E.transform_new)

        with self.assertLogs('DASF', level='INFO') as plogs:
            pipeline.run()

            self.assertIn('Pipeline run successfully', plogs.output[-1])

            all_output = '\n'.join(plogs.output)

            self.assertIn('Dataset_A', all_output)
            self.assertIn('transform_new', all_output)
            self.assertIn('transform_f', all_output)

    def test_pipeline_results(self):
        orig_data = np.arange(10)

        dataset_A = Dataset_A(name="Test Dataset A")

        t_A = Transform_A()
        t_B = Transform_B()

        pipeline = Pipeline("Test Pipeline Creation")

        pipeline = pipeline.add(t_A, X=dataset_A) \
                           .add(t_B, X=t_A)

        with self.assertLogs('DASF', level='INFO') as plogs:
            pipeline.run()

            self.assertIn('Pipeline run successfully', plogs.output[-1])

            t_A_r = pipeline.get_result_from(t_A)
            t_B_r = pipeline.get_result_from(t_B)

            self.assertTrue(np.array_equal(t_A_r, orig_data + 2))
            self.assertTrue(np.array_equal(t_B_r, orig_data))

    def test_dataset_registration(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        executor = MagicMock()
        executor.is_connected = True
        executor.pre_run.return_value = None
        executor.post_run.return_value = None
        executor.has_dataset = MagicMock(side_effect=[False, True])
        executor.register_dataset.return_value = None
        executor.get_dataset.return_value = dataset_A

        t_A = Transform_A()
        t_B = Transform_B()

        pipeline = Pipeline("Test Pipeline Creation", executor=executor)

        pipeline = pipeline.add(t_A, X=dataset_A) \
                           .add(t_B, X=t_A)

        pipeline.run()

        key = str(hash(dataset_A.load))
        kwargs = {key: dataset_A}

        executor.register_dataset.assert_called_once_with(**kwargs)
        executor.has_dataset.assert_called_with(key)
