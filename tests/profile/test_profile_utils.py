#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch

from dasf.pipeline import Pipeline
from dasf.profile.profiler import EventDatabase
try:
    from dasf.profile.utils import MultiEventDatabase, register_default_profiler
except ImportError:
    raise unittest.SkipTest("Module nvtx is not installed")


class TestMultiEventDatabase(unittest.TestCase):
    def setUp(self):
        self.mock_db1 = Mock(spec=EventDatabase)
        self.mock_db2 = Mock(spec=EventDatabase)
        self.mock_db3 = Mock(spec=EventDatabase)

    def test_multi_event_database_creation(self):
        databases = [self.mock_db1, self.mock_db2]
        multi_db = MultiEventDatabase(databases)

        self.assertEqual(multi_db._databases, databases)

    def test_multi_event_database_iteration(self):
        traces1 = ["trace1", "trace2"]
        traces2 = ["trace3", "trace4"]

        self.mock_db1.get_traces.return_value = traces1
        self.mock_db2.get_traces.return_value = traces2

        multi_db = MultiEventDatabase([self.mock_db1, self.mock_db2])

        result = list(multi_db)
        expected = traces1 + traces2

        self.assertEqual(result, expected)
        self.mock_db1.get_traces.assert_called_once()
        self.mock_db2.get_traces.assert_called_once()

    def test_multi_event_database_empty_databases(self):
        multi_db = MultiEventDatabase([])
        result = list(multi_db)
        self.assertEqual(result, [])

    def test_multi_event_database_single_database(self):
        traces = ["trace1", "trace2", "trace3"]
        self.mock_db1.get_traces.return_value = traces

        multi_db = MultiEventDatabase([self.mock_db1])
        result = list(multi_db)

        self.assertEqual(result, traces)
        self.mock_db1.get_traces.assert_called_once()

    def test_multi_event_database_string_representation(self):
        databases = [self.mock_db1, self.mock_db2, self.mock_db3]
        multi_db = MultiEventDatabase(databases)

        expected = "MultiEventDatabase with 3 databases"
        self.assertEqual(str(multi_db), expected)
        self.assertEqual(repr(multi_db), expected)

    def test_multi_event_database_with_one_database(self):
        databases = [self.mock_db1]
        multi_db = MultiEventDatabase(databases)

        expected = "MultiEventDatabase with 1 databases"
        self.assertEqual(str(multi_db), expected)


class TestRegisterDefaultProfiler(unittest.TestCase):
    def setUp(self):
        self.mock_pipeline = Mock(spec=Pipeline)

    @patch('dasf.profile.utils.time.time', return_value=1234567890)
    @patch('dasf.profile.utils.WorkerTaskPlugin')
    @patch('dasf.profile.utils.ResourceMonitor')
    @patch('dasf.profile.utils.atexit.register')
    @patch('builtins.print')
    def test_register_default_profiler_with_defaults(self,
                                                     mock_print,
                                                     mock_atexit,
                                                     mock_resource_monitor,
                                                     mock_worker_plugin,
                                                     mock_time):
        mock_worker_instance = Mock()
        mock_resource_instance = Mock()
        mock_worker_plugin.return_value = mock_worker_instance
        mock_resource_monitor.return_value = mock_resource_instance

        register_default_profiler(self.mock_pipeline)

        mock_worker_plugin.assert_called_once_with(
                name="default-1234567890-TracePlugin"
                )
        mock_resource_monitor.assert_called_once_with(
                name="default-1234567890-ResourceMonitor"
                )

        self.mock_pipeline.register_plugin.assert_called_once_with(mock_worker_instance)

        mock_atexit.assert_called_once()

        mock_print.assert_any_call(
                "Registered worker plugin: default-1234567890-TracePlugin"
                )
        mock_print.assert_any_call(
                "Registered resource plugin: default-1234567890-ResourceMonitor"
                )

    @patch('dasf.profile.utils.time.time', return_value=9876543210)
    @patch('dasf.profile.utils.WorkerTaskPlugin')
    @patch('dasf.profile.utils.ResourceMonitor')
    @patch('dasf.profile.utils.atexit.register')
    @patch('builtins.print')
    def test_register_default_profiler_with_custom_name(self,
                                                        mock_print,
                                                        mock_atexit,
                                                        mock_resource_monitor,
                                                        mock_worker_plugin,
                                                        mock_time):
        mock_worker_instance = Mock()
        mock_resource_instance = Mock()
        mock_worker_plugin.return_value = mock_worker_instance
        mock_resource_monitor.return_value = mock_resource_instance

        register_default_profiler(self.mock_pipeline, name="custom_profiler")

        mock_worker_plugin.assert_called_once_with(
                name="custom_profiler-9876543210-TracePlugin"
                )
        mock_resource_monitor.assert_called_once_with(
                name="custom_profiler-9876543210-ResourceMonitor"
                )

        mock_print.assert_any_call(
                "Registered worker plugin: custom_profiler-9876543210-TracePlugin"
                )
        mock_print.assert_any_call(
                "Registered resource plugin: custom_profiler-9876543210-ResourceMonitor"
                )

    @patch('dasf.profile.utils.time.time', return_value=1111111111)
    @patch('dasf.profile.utils.WorkerTaskPlugin')
    @patch('dasf.profile.utils.ResourceMonitor')
    @patch('dasf.profile.utils.atexit.register')
    @patch('builtins.print')
    def test_register_default_profiler_without_time_suffix(self,
                                                           mock_print,
                                                           mock_atexit,
                                                           mock_resource_monitor,
                                                           mock_worker_plugin,
                                                           mock_time):
        mock_worker_instance = Mock()
        mock_resource_instance = Mock()
        mock_worker_plugin.return_value = mock_worker_instance
        mock_resource_monitor.return_value = mock_resource_instance

        register_default_profiler(self.mock_pipeline,
                                  name="no_time",
                                  add_time_suffix=False)

        mock_worker_plugin.assert_called_once_with(name="no_time-TracePlugin")
        mock_resource_monitor.assert_called_once_with(name="no_time-ResourceMonitor")

        mock_print.assert_any_call("Registered worker plugin: no_time-TracePlugin")
        mock_print.assert_any_call("Registered resource plugin: no_time-ResourceMonitor")

    @patch('dasf.profile.utils.time.time', return_value=2222222222)
    @patch('dasf.profile.utils.WorkerTaskPlugin')
    @patch('dasf.profile.utils.ResourceMonitor')
    @patch('dasf.profile.utils.GPUAnnotationPlugin')
    @patch('dasf.profile.utils.atexit.register')
    @patch('builtins.print')
    def test_register_default_profiler_with_nvtx(self,
                                                 mock_print,
                                                 mock_atexit,
                                                 mock_gpu_plugin,
                                                 mock_resource_monitor,
                                                 mock_worker_plugin,
                                                 mock_time):
        mock_worker_instance = Mock()
        mock_resource_instance = Mock()
        mock_gpu_instance = Mock()
        mock_worker_plugin.return_value = mock_worker_instance
        mock_resource_monitor.return_value = mock_resource_instance
        mock_gpu_plugin.return_value = mock_gpu_instance

        register_default_profiler(self.mock_pipeline, enable_nvtx=True)

        mock_worker_plugin.assert_called_once_with(
                name="default-2222222222-TracePlugin"
                )
        mock_resource_monitor.assert_called_once_with(
                name="default-2222222222-ResourceMonitor"
                )
        mock_gpu_plugin.assert_called_once_with()

        # Should register both worker and GPU plugins
        expected_calls = [
            unittest.mock.call(mock_worker_instance),
            unittest.mock.call(mock_gpu_instance)
        ]
        self.mock_pipeline.register_plugin.assert_has_calls(expected_calls)

        mock_print.assert_any_call(
                "Registered worker plugin: default-2222222222-TracePlugin"
                )
        mock_print.assert_any_call(
                "Registered resource plugin: default-2222222222-ResourceMonitor"
                )
        mock_print.assert_any_call(
                "Registered GPU annotation plugin (NVTX)"
                )

    @patch('dasf.profile.utils.time.time', return_value=3333333333)
    @patch('dasf.profile.utils.WorkerTaskPlugin')
    @patch('dasf.profile.utils.ResourceMonitor')
    @patch('dasf.profile.utils.atexit.register')
    def test_register_default_profiler_atexit_close_function(
            self,
            mock_atexit,
            mock_resource_monitor, mock_worker_plugin,
            mock_time
            ):
        mock_worker_instance = Mock()
        mock_resource_instance = Mock()
        mock_worker_plugin.return_value = mock_worker_instance
        mock_resource_monitor.return_value = mock_resource_instance

        register_default_profiler(self.mock_pipeline)

        # Get the close function registered with atexit
        mock_atexit.assert_called_once()
        close_function = mock_atexit.call_args[0][0]

        # Call the close function and verify it stops the resource monitor
        close_function()
        mock_resource_instance.stop.assert_called_once()
