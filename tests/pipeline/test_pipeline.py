#!/usr/bin/env python3

import unittest

import numpy as np
from unittest.mock import MagicMock

from dasf.datasets import DatasetArray
from dasf.pipeline import Pipeline
from dasf.pipeline.executors.base import Executor
from dasf.transforms.base import TargeteredTransform, Transform


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


class Transform_F(TargeteredTransform):
    def transform(self, X):
        return X - 4


class Transform_Fail(Transform):
    def transform(self, X):
        raise Exception('Throw an exception from a transformation.')
        return X


def transform_g(X):
    return X - 4


class TestExecutorDisconnected(Executor):
    @property
    def is_connected(self):
        return False


class TestNonExecutor:
    def __init__(self):
        pass


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
                           .add(transform_g, X=t_E.transform_new)

        with self.assertLogs('DASF', level='INFO') as plogs:
            pipeline.run()

            self.assertIn('Pipeline run successfully', plogs.output[-1])

            all_output = '\n'.join(plogs.output)

            self.assertIn('Dataset_A', all_output)
            self.assertIn('transform_new', all_output)
            self.assertIn('transform_g', all_output)

    def test_pipeline_targetered_transformers(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        t_F = Transform_F()

        pipeline = Pipeline("Test Pipeline Creation Non Transformers")

        pipeline = pipeline.add(t_F, X=dataset_A) \
                           .add(transform_g, X=t_F)

        with self.assertLogs('DASF', level='INFO') as plogs:
            pipeline.run()

            self.assertIn('Pipeline run successfully', plogs.output[-1])

            all_output = '\n'.join(plogs.output)

            self.assertIn('Dataset_A', all_output)
            self.assertIn('Transform_F', all_output)
            self.assertIn('transform_g', all_output)

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

    def test_pipeline_results_from_wrong_operator(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        t_A = Transform_A()
        t_B = Transform_B()
        t_C = Transform_C()

        pipeline = Pipeline("Test Pipeline Creation")

        pipeline = pipeline.add(t_A, X=dataset_A) \
                           .add(t_B, X=t_A)

        with self.assertRaises(Exception) as context, \
             self.assertLogs('DASF', level='INFO') as plogs:
            pipeline.run()

            self.assertIn('Pipeline run successfully', plogs.output[-1])

            _ = pipeline.get_result_from(t_C)

            self.assertTrue('was not added into pipeline.' in str(context.exception))

    def test_pipeline_results_not_run_exception(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        t_A = Transform_A()
        t_B = Transform_B()

        pipeline = Pipeline("Test Pipeline Creation")

        pipeline = pipeline.add(t_A, X=dataset_A) \
                           .add(t_B, X=t_A)

        with self.assertRaises(Exception) as context:
            _ = pipeline.get_result_from(t_A)
            _ = pipeline.get_result_from(t_B)

        self.assertTrue('Pipeline was not executed yet.' in str(context.exception))

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

        # XXX: Disable register dataset for now
        # key = str(hash(dataset_A.load))
        # kwargs = {key: dataset_A}

        # executor.register_dataset.assert_called_once_with(**kwargs)
        # executor.has_dataset.assert_called_with(key)
        raise unittest.SkipTest("Datasets are disabled for now")

    def test_pipeline_failure(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        t_A = Transform_A()
        t_B = Transform_B()
        t_F = Transform_Fail()

        pipeline = Pipeline("Test Pipeline Creation")

        pipeline = pipeline.add(t_F, X=dataset_A) \
                           .add(t_A, X=t_F) \
                           .add(t_B, X=t_A)

        with self.assertLogs('DASF', level='INFO') as plogs:
            pipeline.run()

            self.assertIn('Pipeline failed at \'Transform_Fail.transform\'',
                          plogs.output[-1])

            all_output = '\n'.join(plogs.output)

            self.assertIn('Dataset_A', all_output)
            self.assertIn('Transform_Fail', all_output)
            self.assertIn('Transform_A', all_output)
            self.assertIn('Transform_B', all_output)

            self.assertIn('ERROR', all_output)
            self.assertIn('Failed', all_output)

    def test_pipeline_loop(self):
        t_A = Transform_A()
        t_B = Transform_B()
        t_C = Transform_C()

        pipeline = Pipeline("Test Pipeline Loop")

        pipeline = pipeline.add(t_B, X=t_A) \
                           .add(t_C, X=t_B) \
                           .add(t_A, X=t_C)

        with self.assertRaises(Exception) as context:
            pipeline.run()

        self.assertTrue('Pipeline has not a DAG format.' in str(context.exception))

    def test_pipeline_executor_disconnected(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        t_A = Transform_A()
        t_B = Transform_B()

        executor = TestExecutorDisconnected()

        pipeline = Pipeline("Test Pipeline Creation", executor=executor)

        pipeline = pipeline.add(t_A, X=dataset_A) \
                           .add(t_B, X=t_A)

        with self.assertRaises(Exception) as context:
            pipeline.run()

        self.assertTrue('Executor is not connected.' in str(context.exception))

    def test_pipeline_non_executor(self):
        dataset_A = Dataset_A(name="Test Dataset A")

        t_A = Transform_A()
        t_B = Transform_B()

        executor = TestNonExecutor()

        pipeline = Pipeline("Test Pipeline Creation", executor=executor)

        pipeline = pipeline.add(t_A, X=dataset_A) \
                           .add(t_B, X=t_A)

        with self.assertRaises(Exception) as context:
            pipeline.run()

        self.assertTrue('Executor TestNonExecutor has not a execute() method.'
                        in str(context.exception))
