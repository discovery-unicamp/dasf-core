#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch, MagicMock
import timeit
from time import perf_counter

import dasf.utils.benchmark
from dasf.utils.benchmark import TimeBenchmark, MemoryBenchmark


class TestTimeBenchmark(unittest.TestCase):
    def test_init_cprofile_backend(self):
        tb = TimeBenchmark(backend="cprofile")
        self.assertEqual(tb._TimeBenchmark__backend, "cprofile")

    def test_init_perf_counter_backend(self):
        tb = TimeBenchmark(backend="perf_counter")
        self.assertEqual(tb._TimeBenchmark__backend, "perf_counter")

    def test_init_default_backend(self):
        tb = TimeBenchmark()
        self.assertEqual(tb._TimeBenchmark__backend, "cprofile")

    @patch('cProfile.Profile')
    def test_enter_cprofile_backend(self, mock_profile):
        mock_pr = Mock()
        mock_profile.return_value = mock_pr
        
        tb = TimeBenchmark(backend="cprofile")
        result = tb.__enter__()
        
        mock_profile.assert_called_once()
        mock_pr.enable.assert_called_once()
        self.assertEqual(result, tb)

    @patch('dasf.utils.benchmark.perf_counter')
    def test_enter_perf_counter_backend(self, mock_perf_counter):
        mock_perf_counter.return_value = 123.456
        
        tb = TimeBenchmark(backend="perf_counter")
        result = tb.__enter__()
        
        mock_perf_counter.assert_called_once()
        self.assertEqual(tb._TimeBenchmark__start, 123.456)
        self.assertEqual(result, tb)

    @patch('builtins.print')
    def test_enter_invalid_backend(self, mock_print):
        tb = TimeBenchmark(backend="invalid")
        result = tb.__enter__()
        
        mock_print.assert_called_once_with("There is no available backend")
        self.assertEqual(result, tb)

    @patch('dasf.utils.benchmark.Stats')
    @patch('cProfile.Profile')
    def test_exit_cprofile_backend(self, mock_profile, mock_stats):
        mock_pr = Mock()
        mock_profile.return_value = mock_pr
        mock_stats_instance = Mock()
        mock_stats.return_value = mock_stats_instance
        mock_stats_instance.strip_dirs.return_value = mock_stats_instance
        mock_stats_instance.sort_stats.return_value = mock_stats_instance
        
        tb = TimeBenchmark(backend="cprofile")
        tb.__enter__()
        tb.__exit__()
        
        mock_pr.disable.assert_called_once()
        mock_stats.assert_called_once_with(mock_pr)
        mock_stats_instance.strip_dirs.assert_called_once()
        mock_stats_instance.sort_stats.assert_called_once_with('cumulative')
        mock_stats_instance.print_stats.assert_called_once_with(10)

    @patch('builtins.print')
    @patch('dasf.utils.benchmark.perf_counter')
    def test_exit_perf_counter_backend(self, mock_perf_counter, mock_print):
        mock_perf_counter.side_effect = [123.456, 125.789]
        
        tb = TimeBenchmark(backend="perf_counter")
        tb.__enter__()
        tb.__exit__()
        
        expected_time = 125.789 - 123.456
        mock_print.assert_called_once_with("Time spent:", expected_time)

    @patch('dasf.utils.benchmark.Stats')
    @patch('cProfile.Profile')
    def test_run_cprofile_backend(self, mock_profile, mock_stats):
        mock_pr = Mock()
        mock_profile.return_value = mock_pr
        mock_stats_instance = Mock()
        mock_stats.return_value = mock_stats_instance
        mock_stats_instance.strip_dirs.return_value = mock_stats_instance
        mock_stats_instance.sort_stats.return_value = mock_stats_instance
        
        def test_function(x, y):
            return x + y
        
        tb = TimeBenchmark(backend="cprofile")
        tb.teardown = Mock()  # Mock teardown method
        tb.run(test_function, 1, 2)
        
        mock_pr.enable.assert_called_once()
        mock_pr.disable.assert_called_once()
        tb.teardown.assert_called_once()

    @patch('timeit.repeat')
    def test_run_timeit_backend(self, mock_repeat):
        def test_function():
            return 42
        
        tb = TimeBenchmark(backend="timeit")
        tb.setup = Mock()  # Mock setup method
        tb.teardown = Mock()  # Mock teardown method
        tb.run(test_function)
        
        mock_repeat.assert_called_once()
        tb.teardown.assert_called_once()

    @patch('builtins.print')
    def test_run_invalid_backend(self, mock_print):
        def test_function():
            return 42
        
        tb = TimeBenchmark(backend="invalid")
        tb.run(test_function)
        
        mock_print.assert_called_once_with("There is no available backend")


class TestMemoryBenchmark(unittest.TestCase):
    def test_init_default_values(self):
        mb = MemoryBenchmark()
        self.assertEqual(mb._MemoryBenchmark__backend, "memray")
        self.assertFalse(mb._MemoryBenchmark__debug)
        self.assertIsNone(mb._MemoryBenchmark__output_file)

    def test_init_custom_values(self):
        mb = MemoryBenchmark(backend="memory_profiler", debug=True, output_file="test.prof")
        self.assertEqual(mb._MemoryBenchmark__backend, "memory_profiler")
        self.assertTrue(mb._MemoryBenchmark__debug)
        self.assertEqual(mb._MemoryBenchmark__output_file, "test.prof")

    @patch('dasf.utils.benchmark.USE_MEMRAY', True)
    @patch('memray.Tracker')
    def test_enter_memray_backend(self, mock_tracker):
        mock_tracker_instance = Mock()
        mock_tracker_instance.__enter__ = Mock(return_value="tracker_result")
        mock_tracker_instance.__exit__ = Mock()
        mock_tracker.return_value = mock_tracker_instance
        
        mb = MemoryBenchmark(backend="memray")
        result = mb.__enter__()
        
        mock_tracker.assert_called_once()
        mock_tracker_instance.__enter__.assert_called_once()
        self.assertEqual(result, "tracker_result")

    @patch('dasf.utils.benchmark.USE_MEMRAY', False)
    def test_enter_memray_backend_not_available(self):
        mb = MemoryBenchmark(backend="memray")
        
        with self.assertRaises(Exception) as context:
            mb.__enter__()
        
        self.assertIn("does not support context manager", str(context.exception))

    def test_enter_unsupported_backend(self):
        mb = MemoryBenchmark(backend="invalid")
        
        with self.assertRaises(Exception) as context:
            mb.__enter__()
        
        self.assertIn("does not support context manager", str(context.exception))

    @patch('dasf.utils.benchmark.USE_MEMRAY', True)
    @patch('memray.Tracker')
    def test_exit_memray_backend(self, mock_tracker):
        mock_tracker_instance = Mock()
        mock_tracker_instance.__enter__ = Mock()
        mock_tracker_instance.__exit__ = Mock(return_value="exit_result")
        mock_tracker.return_value = mock_tracker_instance
        
        mb = MemoryBenchmark(backend="memray")
        mb._MemoryBenchmark__memray = mock_tracker_instance
        result = mb.__exit__(None, None, None)
        
        mock_tracker_instance.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(result, "exit_result")

    @patch('dasf.utils.benchmark.USE_MEM_PROF', False)
    @patch('builtins.print')
    def test_run_memory_profiler_debug(self, mock_print):
        def test_function(x):
            return x * 2
        
        mb = MemoryBenchmark(backend="memory_profiler", debug=True)
        result = mb.run(test_function, 5)
        
        mock_print.assert_called_once_with("The backend memory_profiler is not supported")

    @patch('dasf.utils.benchmark.USE_MEM_PROF', False)
    @patch('builtins.print')
    def test_run_memory_profiler_no_debug(self, mock_print):
        def test_function(x, y):
            return x + y
        
        mb = MemoryBenchmark(backend="memory_profiler", debug=False, interval=0.1)
        result = mb.run(test_function, 1, y=2)
        
        mock_print.assert_called_once_with("The backend memory_profiler is not supported")

    @patch('dasf.utils.benchmark.USE_MEMRAY', True)
    @patch('memray.Tracker')
    def test_run_memray_backend(self, mock_tracker):
        mock_tracker_instance = Mock()
        mock_tracker_instance.__enter__ = Mock()
        mock_tracker_instance.__exit__ = Mock()
        mock_tracker.return_value = mock_tracker_instance
        
        def test_function():
            return "memray_result"
        
        mb = MemoryBenchmark(backend="memray")
        mb.teardown = Mock()
        result = mb.run(test_function)
        
        mock_tracker.assert_called()
        self.assertEqual(result, "memray_result")

    @patch('builtins.print')
    def test_run_unsupported_backend(self, mock_print):
        def test_function():
            return 42
        
        mb = MemoryBenchmark(backend="invalid")
        mb.run(test_function)
        
        mock_print.assert_called_once_with("The backend invalid is not supported")

    @patch('dasf.utils.benchmark.USE_MEM_PROF', False)
    @patch('builtins.print')
    def test_run_memory_profiler_not_available(self, mock_print):
        def test_function():
            return 42
        
        mb = MemoryBenchmark(backend="memory_profiler")
        mb.run(test_function)
        
        mock_print.assert_called_once_with("The backend memory_profiler is not supported")

    @patch('dasf.utils.benchmark.USE_MEMRAY', False)
    @patch('builtins.print')
    def test_run_memray_not_available(self, mock_print):
        def test_function():
            return 42
        
        mb = MemoryBenchmark(backend="memray")
        mb.run(test_function)
        
        mock_print.assert_called_once_with("The backend memray is not supported")
