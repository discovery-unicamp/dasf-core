#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

import pandas as pd
import networkx as nx

from dasf.profile.analysis import TraceAnalyser, main, valid_analyses
from dasf.profile.profiler import CompleteEvent, InstantEvent
from dasf.profile.utils import MultiEventDatabase


class TestTraceAnalyser(unittest.TestCase):
    def setUp(self):
        self.mock_database = Mock(spec=MultiEventDatabase)
        
        # Create sample events for testing
        self.sample_events = [
            CompleteEvent(
                name="Compute",
                timestamp=100.0,
                duration=5.0,
                process_id="host1",
                thread_id="worker-host1-gpu0",
                args={
                    "key": "task-1",
                    "name": "add",
                    "dependencies": ["task-0"],
                    "dependents": ["task-2"],
                    "size": 1024,
                    "shape": (100, 100),
                    "type": "numpy.ndarray"
                }
            ),
            CompleteEvent(
                name="Compute", 
                timestamp=110.0,
                duration=3.0,
                process_id="host1",
                thread_id="worker-host1-gpu0",
                args={
                    "key": "task-2",
                    "name": "multiply",
                    "dependencies": ["task-1"],
                    "dependents": [],
                    "size": 2048,
                    "shape": (200, 200),
                    "type": "numpy.ndarray"
                }
            ),
            InstantEvent(
                name="Resource Usage",
                timestamp=102.0,
                process_id="host1",
                thread_id="worker-host1-gpu0",
                args={
                    "gpu_utilization": 85.5,
                    "gpu_memory_used": 4000000000
                }
            ),
            InstantEvent(
                name="Managed Memory",
                timestamp=105.0,
                process_id="host1", 
                thread_id="worker-host1-gpu0",
                args={
                    "key": "task-1",
                    "state": "memory",
                    "size": 1024,
                    "tasks": 2
                }
            )
        ]

    def test_trace_analyser_creation_with_processing(self):
        self.mock_database.__iter__ = Mock(return_value=iter(self.sample_events))
        
        analyser = TraceAnalyser(self.mock_database, process_trace_before=True)
        
        # When process_trace_before=True, database should be converted to list
        self.assertEqual(analyser._database, self.sample_events)

    def test_trace_analyser_creation_without_processing(self):
        analyser = TraceAnalyser(self.mock_database, process_trace_before=False)
        
        # When process_trace_before=False, database should remain as is
        self.assertEqual(analyser._database, self.mock_database)

    @patch('tqdm.tqdm')
    def test_create_annotated_task_graph(self, mock_tqdm):
        mock_tqdm.side_effect = lambda x, desc=None: x  # Pass through without tqdm
        
        self.mock_database.__iter__ = Mock(return_value=iter(self.sample_events))
        analyser = TraceAnalyser(self.mock_database)
        
        graph = analyser.create_annotated_task_graph()
        
        self.assertIsInstance(graph, nx.DiGraph)
        
        # Check nodes are added correctly
        self.assertIn("task-1", graph.nodes)
        self.assertIn("task-2", graph.nodes)
        
        # Check node attributes
        node1 = graph.nodes["task-1"]
        self.assertEqual(node1["name"], "add")
        self.assertEqual(node1["size"], 1024)
        self.assertEqual(node1["shape"], (100, 100))
        self.assertEqual(node1["duration"], 5.0)
        
        # Check edges (dependencies)
        self.assertTrue(graph.has_edge("task-1", "task-2"))
        
        # Check computed attributes
        self.assertIn("input_data_size", node1)
        self.assertIn("throughput", node1)

    @patch('tqdm.tqdm')
    def test_per_function_bottleneck(self, mock_tqdm):
        mock_tqdm.side_effect = lambda x, desc=None: x  # Pass through without tqdm
        
        self.mock_database.__iter__ = Mock(return_value=iter(self.sample_events))
        analyser = TraceAnalyser(self.mock_database)
        
        df = analyser.per_function_bottleneck()
        
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check expected columns
        expected_columns = [
            'Host', 'GPU', 'Function', 'Duration (s)', 
            'Percentage of total time (%)', 'Mean GPU Utilization (%)',
            'Mean GPU Memory Used (GB)', 'Mean Data Size (MB)',
            'Mean Throughput (MB/s)', 'Num Tasks (chunks)', 'Mean Task time (s)'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check that data is sorted by duration descending
        if len(df) > 1:
            durations = df['Duration (s)'].values
            self.assertTrue(all(durations[i] >= durations[i+1] for i in range(len(durations)-1)))

    @patch('tqdm.tqdm')
    def test_per_worker_task_balance(self, mock_tqdm):
        mock_tqdm.side_effect = lambda x, desc=None: x  # Pass through without tqdm
        
        self.mock_database.__iter__ = Mock(return_value=iter(self.sample_events))
        analyser = TraceAnalyser(self.mock_database)
        
        df = analyser.per_worker_task_balance()
        
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check expected columns
        self.assertIn('Time Interval (seconds from begin)', df.columns)
        
        # Check that dataframe is sorted by time interval
        if len(df) > 1:
            intervals = df['Time Interval (seconds from begin)'].values
            self.assertTrue(all(intervals[i] <= intervals[i+1] for i in range(len(intervals)-1)))

    @patch('tqdm.tqdm')
    def test_per_task_bottleneck(self, mock_tqdm):
        mock_tqdm.side_effect = lambda x, desc=None: x  # Pass through without tqdm
        
        self.mock_database.__iter__ = Mock(return_value=iter(self.sample_events))
        analyser = TraceAnalyser(self.mock_database)
        
        df = analyser.per_task_bottleneck()
        
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check expected columns
        expected_columns = [
            'Host', 'GPU', 'Task Key', 'Duration (s)', 
            'Percentage of total time (%)', 'Memory usage (Mb)'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_trace_analyser_with_empty_database(self):
        empty_events = []
        self.mock_database.__iter__ = Mock(return_value=iter(empty_events))
        
        analyser = TraceAnalyser(self.mock_database)
        
        # Should not raise exceptions with empty data
        graph = analyser.create_annotated_task_graph()
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)


class TestMainFunction(unittest.TestCase):
    def setUp(self):
        self.mock_database = Mock(spec=MultiEventDatabase)

    @patch('dasf.profile.analysis.pd.set_option')
    @patch('dasf.profile.analysis.TraceAnalyser')
    @patch('builtins.print')
    def test_main_function_with_all_analyses(self, mock_print, mock_analyser_class, mock_set_option):
        mock_analyser = Mock()
        mock_analyser_class.return_value = mock_analyser
        
        # Mock return values for analysis methods
        mock_df = Mock(spec=pd.DataFrame)
        mock_analyser.per_function_bottleneck.return_value = mock_df
        mock_analyser.per_task_bottleneck.return_value = mock_df
        mock_analyser.per_worker_task_balance.return_value = mock_df
        
        main(self.mock_database, analyses=valid_analyses, head=10)
        
        # Verify all analyses were called
        mock_analyser.per_function_bottleneck.assert_called_once()
        mock_analyser.per_task_bottleneck.assert_called_once()
        mock_analyser.per_worker_task_balance.assert_called_once()
        
        # Verify pandas options were set
        self.assertTrue(mock_set_option.called)
        
        # Verify completion message
        mock_print.assert_any_call("Analyses finished!")

    @patch('dasf.profile.analysis.pd.set_option')
    @patch('dasf.profile.analysis.TraceAnalyser')
    @patch('builtins.print')
    def test_main_function_with_specific_analysis(self, mock_print, mock_analyser_class, mock_set_option):
        mock_analyser = Mock()
        mock_analyser_class.return_value = mock_analyser
        
        mock_df = Mock(spec=pd.DataFrame)
        mock_analyser.per_function_bottleneck.return_value = mock_df
        
        main(self.mock_database, analyses=["function_bottleneck"], head=20)
        
        # Only function_bottleneck should be called
        mock_analyser.per_function_bottleneck.assert_called_once()
        mock_analyser.per_task_bottleneck.assert_not_called()
        mock_analyser.per_worker_task_balance.assert_not_called()

    @patch('dasf.profile.analysis.pd.set_option')
    @patch('dasf.profile.analysis.TraceAnalyser')
    @patch('dasf.profile.analysis.Path')
    def test_main_function_with_output_directory(self, mock_path, mock_analyser_class, mock_set_option):
        mock_analyser = Mock()
        mock_analyser_class.return_value = mock_analyser
        
        mock_df = Mock(spec=pd.DataFrame)
        mock_analyser.per_function_bottleneck.return_value = mock_df
        
        mock_output_path = Mock()
        mock_path.return_value = mock_output_path
        
        main(self.mock_database, output="test_output", analyses=["function_bottleneck"])
        
        # Verify output directory creation
        mock_path.assert_called_once_with("test_output")
        mock_output_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Verify CSV export
        mock_df.to_csv.assert_called_once()

    @patch('dasf.profile.analysis.pd.set_option')
    @patch('dasf.profile.analysis.TraceAnalyser')
    def test_main_function_default_analyses(self, mock_analyser_class, mock_set_option):
        mock_analyser = Mock()
        mock_analyser_class.return_value = mock_analyser
        
        # Mock return values
        mock_df = Mock(spec=pd.DataFrame)
        mock_analyser.per_function_bottleneck.return_value = mock_df
        mock_analyser.per_task_bottleneck.return_value = mock_df
        mock_analyser.per_worker_task_balance.return_value = mock_df
        
        # Call with analyses=None (should default to all analyses)
        main(self.mock_database, analyses=None)
        
        # All analyses should be called when analyses=None
        mock_analyser.per_function_bottleneck.assert_called_once()
        mock_analyser.per_task_bottleneck.assert_called_once()
        mock_analyser.per_worker_task_balance.assert_called_once()


class TestValidAnalyses(unittest.TestCase):
    def test_valid_analyses_list(self):
        expected_analyses = ["function_bottleneck", "task_bottleneck", "task_balance"]
        self.assertEqual(valid_analyses, expected_analyses)
        
    def test_valid_analyses_types(self):
        for analysis in valid_analyses:
            self.assertIsInstance(analysis, str)


class TestAnalysisIntegration(unittest.TestCase):
    def test_complete_analysis_workflow(self):
        # Create a more comprehensive set of test events
        events = [
            CompleteEvent(
                name="Compute",
                timestamp=100.0,
                duration=2.0,
                process_id="host1",
                thread_id="worker-host1-gpu0", 
                args={
                    "key": "task-1",
                    "name": "add",
                    "dependencies": [],
                    "dependents": ["task-2"],
                    "size": 1000,
                    "shape": (10, 10),
                    "type": "numpy.ndarray"
                }
            ),
            CompleteEvent(
                name="Compute",
                timestamp=103.0,
                duration=3.0,
                process_id="host1",
                thread_id="worker-host1-gpu0",
                args={
                    "key": "task-2", 
                    "name": "multiply",
                    "dependencies": ["task-1"],
                    "dependents": [],
                    "size": 2000,
                    "shape": (20, 20),
                    "type": "numpy.ndarray"
                }
            ),
            InstantEvent(
                name="Managed Memory",
                timestamp=105.0,
                process_id="host1",
                thread_id="worker-host1-gpu0",
                args={
                    "key": "task-1",
                    "state": "memory", 
                    "size": 1000,
                    "tasks": 1
                }
            )
        ]
        
        mock_database = Mock(spec=MultiEventDatabase)
        mock_database.__iter__ = Mock(return_value=iter(events))
        
        analyser = TraceAnalyser(mock_database)
        
        # Test that all analysis methods can run without errors
        with patch('tqdm.tqdm', side_effect=lambda x, desc=None: x):
            graph = analyser.create_annotated_task_graph()
            self.assertIsInstance(graph, nx.DiGraph)
            self.assertEqual(len(graph.nodes), 2)
            
            function_df = analyser.per_function_bottleneck()
            self.assertIsInstance(function_df, pd.DataFrame)
            
            task_df = analyser.per_task_bottleneck()
            self.assertIsInstance(task_df, pd.DataFrame)
            
            balance_df = analyser.per_worker_task_balance()
            self.assertIsInstance(balance_df, pd.DataFrame)
