#!/usr/bin/env python3

import unittest

import numpy as np
from unittest.mock import MagicMock, patch

from dasf.datasets import DatasetArray
from dasf.pipeline import Pipeline, PipelinePlugin
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


class MockPipelinePlugin(PipelinePlugin):
    def __init__(self):
        self.pipeline_start_called = False
        self.pipeline_end_called = False
        self.task_start_calls = []
        self.task_end_calls = []
        self.task_error_calls = []

    def on_pipeline_start(self, fn_keys):
        self.pipeline_start_called = True
        self.fn_keys = fn_keys

    def on_pipeline_end(self):
        self.pipeline_end_called = True

    def on_task_start(self, func, params, name):
        self.task_start_calls.append((func, params, name))

    def on_task_end(self, func, params, name, ret):
        self.task_end_calls.append((func, params, name, ret))

    def on_task_error(self, func, params, name, exception):
        self.task_error_calls.append((func, params, name, exception))


class MockExecutorPlugin:
    def __init__(self):
        self.registered = False


class InvalidObject:
    def __init__(self):
        pass


class TestExecutorDisconnected(Executor):
    @property
    def is_connected(self):
        return False


class NonExecutor:
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

        executor = NonExecutor()

        pipeline = Pipeline("Test Pipeline Creation", executor=executor)

        pipeline = pipeline.add(t_A, X=dataset_A) \
                           .add(t_B, X=t_A)

        with self.assertRaises(Exception) as context:
            pipeline.run()

        self.assertTrue('Executor NonExecutor has not a execute() method.'
                        in str(context.exception))

    def test_pipeline_with_callbacks(self):
        """Test pipeline execution with PipelinePlugin callbacks"""
        dataset_A = Dataset_A(name="Test Dataset A")
        t_A = Transform_A()

        plugin = MockPipelinePlugin()
        pipeline = Pipeline("Test Pipeline with Callbacks", callbacks=[plugin])
        pipeline.add(t_A, X=dataset_A)

        with self.assertLogs('DASF', level='INFO'):
            pipeline.run()

        self.assertTrue(plugin.pipeline_start_called)
        self.assertTrue(plugin.pipeline_end_called)
        self.assertEqual(len(plugin.task_start_calls), 2)  # Dataset + Transform
        self.assertEqual(len(plugin.task_end_calls), 2)
        self.assertEqual(len(plugin.task_error_calls), 0)

    def test_pipeline_callbacks_on_error(self):
        """Test that callbacks are called even when tasks fail"""
        dataset_A = Dataset_A(name="Test Dataset A")
        t_fail = Transform_Fail()

        plugin = MockPipelinePlugin()
        pipeline = Pipeline("Test Pipeline Error Callbacks", callbacks=[plugin])
        pipeline.add(t_fail, X=dataset_A)

        with self.assertLogs('DASF', level='INFO'):
            pipeline.run()

        self.assertTrue(plugin.pipeline_start_called)
        self.assertTrue(plugin.pipeline_end_called)
        self.assertEqual(len(plugin.task_error_calls), 1)

        # Check error callback details
        func, params, name, exception = plugin.task_error_calls[0]
        self.assertEqual(name, "Transform_Fail.transform")
        self.assertIsInstance(exception, Exception)

    def test_register_pipeline_plugin(self):
        """Test registering PipelinePlugin"""
        pipeline = Pipeline("Test Plugin Registration")
        plugin = MockPipelinePlugin()

        self.assertEqual(len(pipeline._callbacks), 0)
        pipeline.register_plugin(plugin)
        self.assertEqual(len(pipeline._callbacks), 1)
        self.assertIn(plugin, pipeline._callbacks)

    def test_register_executor_plugin(self):
        """Test registering non-PipelinePlugin (passed to executor)"""
        executor = MagicMock()
        executor.is_connected = True
        pipeline = Pipeline("Test Executor Plugin", executor=executor)

        exec_plugin = MockExecutorPlugin()
        pipeline.register_plugin(exec_plugin)

        executor.register_plugin.assert_called_once_with(exec_plugin)

    def test_pipeline_info(self):
        """Test pipeline info method"""
        executor = MagicMock()
        executor.info = "Mock executor info"
        pipeline = Pipeline("Test Info", executor=executor)

        with patch('builtins.print') as mock_print:
            pipeline.info()
            mock_print.assert_called_once_with("Mock executor info")

    def test_pipeline_visualize_notebook(self):
        """Test pipeline visualization in notebook environment"""
        dataset_A = Dataset_A(name="Test Dataset A")
        t_A = Transform_A()

        pipeline = Pipeline("Test Visualization")
        pipeline.add(t_A, X=dataset_A)

        with patch('dasf.utils.funcs.is_notebook', return_value=True):
            result = pipeline.visualize()
            self.assertEqual(result, pipeline._dag_g)

    def test_pipeline_visualize_non_notebook(self):
        """Test pipeline visualization outside notebook"""
        dataset_A = Dataset_A(name="Test Dataset A")
        t_A = Transform_A()

        pipeline = Pipeline("Test Visualization")
        pipeline.add(t_A, X=dataset_A)

        with patch('dasf.utils.funcs.is_notebook', return_value=False):
            with patch.object(pipeline._dag_g, 'view') as mock_view:
                mock_view.return_value = "view_result"
                result = pipeline.visualize("test_filename")
                mock_view.assert_called_once_with("test_filename")
                self.assertEqual(result, "view_result")

    def test_pipeline_verbose_mode(self):
        """Test pipeline creation with verbose=True"""
        pipeline = Pipeline("Test Verbose", verbose=True)
        self.assertTrue(pipeline._verbose)

    def test_invalid_object_in_pipeline(self):
        """Test adding invalid object to pipeline"""
        pipeline = Pipeline("Test Invalid Object")
        invalid_obj = InvalidObject()

        with self.assertRaises(ValueError) as context:
            pipeline.add(invalid_obj)

        self.assertIn("is not a function, method or a transformer object",
                      str(context.exception))

    def test_pipeline_add_with_existing_parameters(self):
        """Test adding object with parameters that get updated"""
        dataset_A = Dataset_A(name="Test Dataset A")
        dataset_B = Dataset_A(name="Test Dataset B")
        t_A = Transform_A()

        pipeline = Pipeline("Test Parameter Update")
        pipeline.add(t_A, X=dataset_A)
        pipeline.add(t_A, Y=dataset_B)  # Add same transform with different params

        # Verify parameters were merged
        key = hash(t_A.transform)
        self.assertIn('X', pipeline._dag_table[key]["parameters"])
        self.assertIn('Y', pipeline._dag_table[key]["parameters"])

    def test_pipeline_execute_callbacks_method(self):
        """Test execute_callbacks method directly"""
        plugin1 = MockPipelinePlugin()
        plugin2 = MockPipelinePlugin()

        pipeline = Pipeline("Test Callbacks", callbacks=[plugin1, plugin2])
        pipeline.execute_callbacks("on_pipeline_start", ["key1", "key2"])

        self.assertTrue(plugin1.pipeline_start_called)
        self.assertTrue(plugin2.pipeline_start_called)
        self.assertEqual(plugin1.fn_keys, ["key1", "key2"])
        self.assertEqual(plugin2.fn_keys, ["key1", "key2"])

    def test_pipeline_get_result_edge_cases(self):
        """Test edge cases in get_result_from method"""
        dataset_A = Dataset_A(name="Test Dataset A")
        t_A = Transform_A()

        pipeline = Pipeline("Test Result Edge Cases")
        pipeline.add(t_A, X=dataset_A)

        # Test getting result before execution
        with self.assertRaises(Exception) as context:
            pipeline.get_result_from(t_A)
        self.assertIn("Pipeline was not executed yet", str(context.exception))

        # Execute pipeline
        with self.assertLogs('DASF', level='INFO'):
            pipeline.run()

        # Test getting result from non-existent object
        t_B = Transform_B()
        with self.assertRaises(Exception) as context:
            pipeline.get_result_from(t_B)
        self.assertIn("was not added into pipeline", str(context.exception))

    def test_pipeline_function_object(self):
        """Test adding regular function objects to pipeline"""
        dataset_A = Dataset_A(name="Test Dataset A")

        pipeline = Pipeline("Test Function Object")
        pipeline.add(transform_g, X=dataset_A)

        with self.assertLogs('DASF', level='INFO') as logs:
            pipeline.run()

            all_output = '\n'.join(logs.output)
            self.assertIn('transform_g', all_output)
            self.assertIn('Pipeline run successfully', logs.output[-1])

    def test_pipeline_with_method_object(self):
        """Test adding method objects to pipeline"""
        dataset_A = Dataset_A(name="Test Dataset A")
        t_E = Transform_E()

        pipeline = Pipeline("Test Method Object")
        pipeline.add(t_E.transform_new, X=dataset_A)

        with self.assertLogs('DASF', level='INFO') as logs:
            pipeline.run()

            all_output = '\n'.join(logs.output)
            self.assertIn('transform_new', all_output)

    def test_pipeline_default_executor(self):
        """Test pipeline with default LocalExecutor"""
        pipeline = Pipeline("Test Default Executor")

        # Should use LocalExecutor by default
        from dasf.pipeline.executors.wrapper import LocalExecutor
        self.assertIsInstance(pipeline._executor, LocalExecutor)

    def test_pipeline_empty_parameters(self):
        """Test adding object with no parameters"""
        dataset_A = Dataset_A(name="Test Dataset A")

        pipeline = Pipeline("Test Empty Parameters")
        pipeline.add(dataset_A)  # No parameters

        key = hash(dataset_A.load)
        self.assertIsNone(pipeline._dag_table[key]["parameters"])

    def test_pipeline_dag_visualization_nodes(self):
        """Test DAG visualization node creation"""
        dataset_A = Dataset_A(name="Test Dataset A")
        t_A = Transform_A()

        pipeline = Pipeline("Test DAG Nodes")
        pipeline.add(t_A, X=dataset_A)

        # Check that DAG graph has nodes
        self.assertGreater(len(pipeline._dag_g.body), 0)

    def test_pipeline_name_property(self):
        """Test pipeline name is stored correctly"""
        pipeline_name = "Test Pipeline Name"
        pipeline = Pipeline(pipeline_name)
        self.assertEqual(pipeline._name, pipeline_name)

    def test_pipeline_plugin_base_class(self):
        """Test PipelinePlugin base class methods"""
        plugin = PipelinePlugin()

        # All methods should be no-ops by default
        plugin.on_pipeline_start(["key1", "key2"])
        plugin.on_pipeline_end()
        plugin.on_task_start(None, None, "test")
        plugin.on_task_end(None, None, "test", "result")
        plugin.on_task_error(None, None, "test", Exception("test"))

    def test_pipeline_logger_initialization(self):
        """Test that pipeline initializes logger correctly"""
        pipeline = Pipeline("Test Logger")
        self.assertIsNotNone(pipeline._logger)

    def test_pipeline_dag_structure(self):
        """Test that DAG structure is maintained correctly"""
        dataset_A = Dataset_A(name="Test Dataset A")
        t_A = Transform_A()
        t_B = Transform_B()

        pipeline = Pipeline("Test DAG Structure")
        pipeline.add(t_A, X=dataset_A)
        pipeline.add(t_B, X=t_A)

        # Verify DAG has correct nodes and edges
        self.assertEqual(pipeline._dag.number_of_nodes(), 3)  # Dataset + 2 transforms
        self.assertEqual(pipeline._dag.number_of_edges(), 2)  # 2 dependencies

        # Verify topological order using networkx
        import networkx as nx
        topo_order = list(nx.topological_sort(pipeline._dag))
        dataset_key = hash(dataset_A.load)
        t_A_key = hash(t_A.transform)
        t_B_key = hash(t_B.transform)

        self.assertIn(dataset_key, topo_order)
        self.assertIn(t_A_key, topo_order)
        self.assertIn(t_B_key, topo_order)

    def test_pipeline_inspect_fit_object(self):
        """Test pipeline with Fit object"""
        # Create a simple fit object mock
        class MockFit:
            def fit(self, X):
                return X

        # Mock the issubclass check for Fit
        with patch('dasf.pipeline.pipeline.issubclass') as mock_issubclass:
            def side_effect(cls, base_cls):
                from dasf.transforms.base import Fit
                return base_cls == Fit

            mock_issubclass.side_effect = side_effect

            fit_obj = MockFit()
            pipeline = Pipeline("Test Fit Object")

            # This should work without error
            obj, func_name, objref = pipeline._Pipeline__inspect_element(fit_obj)
            self.assertEqual(func_name, "MockFit.fit")
            self.assertEqual(obj, fit_obj.fit)
            self.assertEqual(objref, fit_obj)

    def test_pipeline_inspect_loader_object(self):
        """Test pipeline with BaseLoader object"""
        class MockLoader:
            def load(self, X):
                return X

        # Mock the issubclass check for BaseLoader
        with patch('dasf.pipeline.pipeline.issubclass') as mock_issubclass:
            def side_effect(cls, base_cls):
                from dasf.ml.inference.loader.base import BaseLoader
                return base_cls == BaseLoader

            mock_issubclass.side_effect = side_effect

            loader_obj = MockLoader()
            pipeline = Pipeline("Test Loader Object")

            # This should work without error
            obj, func_name, objref = pipeline._Pipeline__inspect_element(loader_obj)
            self.assertEqual(func_name, "MockLoader.load")
            self.assertEqual(obj, loader_obj.load)
            self.assertEqual(objref, loader_obj)
