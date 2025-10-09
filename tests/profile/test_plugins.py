#!/usr/bin/env python3

import os
import socket
import time
import unittest
from unittest.mock import Mock, patch, MagicMock

from dasf.profile.plugins import WorkerTaskPlugin, ResourceMonitor, GPUAnnotationPlugin


class TestWorkerTaskPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = WorkerTaskPlugin(name="TestPlugin")
        self.mock_worker = Mock()
        self.mock_worker.name = "worker-123"
        self.mock_worker.data = {}
        self.mock_worker.state = Mock()
        self.mock_worker.state.tasks = {}
        self.mock_worker.state.nbytes = 1024

    @patch('socket.gethostname', return_value='test-host')
    @patch('dasf.profile.plugins.EventProfiler')
    def test_setup(self, mock_profiler, mock_hostname):
        mock_profiler_instance = Mock()
        mock_profiler.return_value = mock_profiler_instance
        
        self.plugin.setup(self.mock_worker)
        
        self.assertEqual(self.plugin.worker, self.mock_worker)
        self.assertEqual(self.plugin.hostname, 'test-host')
        self.assertEqual(self.plugin.worker_id, 'worker-test-host-worker-123')
        
        mock_profiler.assert_called_once_with(database_file="TestPlugin-test-host.msgpack")
        self.assertEqual(self.plugin.database, mock_profiler_instance)

    @patch('socket.gethostname', return_value='test-host')
    @patch('dasf.profile.plugins.EventProfiler')
    @patch('time.monotonic', return_value=100.0)
    def test_transition_to_memory_with_compute_startstop(self, mock_time, mock_profiler, mock_hostname):
        mock_profiler_instance = Mock()
        mock_profiler.return_value = mock_profiler_instance
        
        self.plugin.setup(self.mock_worker)
        
        # Setup task with startstops
        task_key = "test-task-123"
        mock_task = Mock()
        mock_task.startstops = [
            {"action": "compute", "start": 90.0, "stop": 95.0},
            {"action": "other", "start": 80.0, "stop": 85.0}
        ]
        mock_task.nbytes = 2048
        mock_task.dependencies = []
        mock_task.dependents = []
        
        self.mock_worker.state.tasks[task_key] = mock_task
        
        # Setup data object with shape and dtype
        mock_data = Mock()
        mock_data.shape = (100, 200)
        mock_data.dtype = "float32"
        self.mock_worker.data[task_key] = mock_data
        
        self.plugin.transition(task_key, "executing", "memory")
        
        # Verify complete event was recorded
        mock_profiler_instance.record_complete_event.assert_called_once()
        call_args = mock_profiler_instance.record_complete_event.call_args
        
        self.assertEqual(call_args[1]["name"], "Compute")
        self.assertEqual(call_args[1]["timestamp"], 100.0)
        self.assertEqual(call_args[1]["duration"], 5.0)  # 95.0 - 90.0
        self.assertEqual(call_args[1]["process_id"], "test-host")
        self.assertEqual(call_args[1]["thread_id"], "worker-test-host-worker-123")
        
        args = call_args[1]["args"]
        self.assertEqual(args["key"], task_key)
        self.assertEqual(args["size"], 2048)
        self.assertEqual(args["shape"], (100, 200))
        self.assertEqual(args["dtype"], "float32")

    @patch('socket.gethostname', return_value='test-host')
    @patch('dasf.profile.plugins.EventProfiler')
    @patch('time.monotonic', return_value=200.0)
    def test_transition_to_memory_records_managed_memory_event(self, mock_time, mock_profiler, mock_hostname):
        mock_profiler_instance = Mock()
        mock_profiler.return_value = mock_profiler_instance
        
        self.plugin.setup(self.mock_worker)
        
        task_key = "test-task-456"
        
        self.plugin.transition(task_key, "executing", "memory")
        
        # Verify instant event was recorded for managed memory
        mock_profiler_instance.record_instant_event.assert_called_once()
        call_args = mock_profiler_instance.record_instant_event.call_args
        
        self.assertEqual(call_args[1]["name"], "Managed Memory")
        self.assertEqual(call_args[1]["timestamp"], 200.0)
        self.assertEqual(call_args[1]["process_id"], "test-host")
        self.assertEqual(call_args[1]["thread_id"], "worker-test-host-worker-123")
        
        args = call_args[1]["args"]
        self.assertEqual(args["key"], task_key)
        self.assertEqual(args["state"], "memory")
        self.assertEqual(args["size"], 1024)
        self.assertEqual(args["tasks"], 0)

    @patch('socket.gethostname', return_value='test-host')
    @patch('dasf.profile.plugins.EventProfiler')
    @patch('time.monotonic', return_value=300.0)
    def test_transition_to_erred_records_managed_memory_event(self, mock_time, mock_profiler, mock_hostname):
        mock_profiler_instance = Mock()
        mock_profiler.return_value = mock_profiler_instance
        
        self.plugin.setup(self.mock_worker)
        
        task_key = "test-task-789"
        
        self.plugin.transition(task_key, "executing", "erred")
        
        # Verify instant event was recorded for managed memory even on error
        mock_profiler_instance.record_instant_event.assert_called_once()
        call_args = mock_profiler_instance.record_instant_event.call_args
        
        args = call_args[1]["args"]
        self.assertEqual(args["state"], "erred")


class TestResourceMonitor(unittest.TestCase):
    @patch('socket.gethostname', return_value='monitor-host')
    @patch('dasf.profile.plugins.EventProfiler')
    @patch('dasf.profile.plugins.SystemMonitor')
    @patch('dasf.profile.plugins.PeriodicCallback')
    def test_resource_monitor_creation(self, mock_callback, mock_system_monitor, 
                                     mock_profiler, mock_hostname):
        mock_profiler_instance = Mock()
        mock_system_monitor_instance = Mock()
        mock_callback_instance = Mock()
        
        mock_profiler.return_value = mock_profiler_instance
        mock_system_monitor.return_value = mock_system_monitor_instance
        mock_callback.return_value = mock_callback_instance
        
        monitor = ResourceMonitor(time=50, autostart=False, name="TestMonitor")
        
        self.assertEqual(monitor.time, 50)
        self.assertEqual(monitor.name, "TestMonitor")
        self.assertEqual(monitor.hostname, "monitor-host")
        
        mock_profiler.assert_called_once_with(database_file="TestMonitor-monitor-host.msgpack")
        mock_system_monitor.assert_called_once()
        mock_callback.assert_called_once_with(monitor.update, callback_time=50)
        
        # autostart=False, so start should not be called
        mock_callback_instance.start.assert_not_called()

    @patch('socket.gethostname', return_value='monitor-host')
    @patch('dasf.profile.plugins.EventProfiler')
    @patch('dasf.profile.plugins.SystemMonitor')
    @patch('dasf.profile.plugins.PeriodicCallback')
    def test_resource_monitor_autostart(self, mock_callback, mock_system_monitor, 
                                       mock_profiler, mock_hostname):
        mock_callback_instance = Mock()
        mock_callback.return_value = mock_callback_instance
        
        monitor = ResourceMonitor(autostart=True)
        
        mock_callback_instance.start.assert_called_once()

    @patch('socket.gethostname', return_value='monitor-host')
    @patch('dasf.profile.plugins.EventProfiler')
    @patch('dasf.profile.plugins.SystemMonitor')
    @patch('dasf.profile.plugins.PeriodicCallback')
    @patch('time.monotonic', return_value=400.0)
    def test_update_method(self, mock_time, mock_callback, mock_system_monitor, 
                          mock_profiler, mock_hostname):
        mock_profiler_instance = Mock()
        mock_system_monitor_instance = Mock()
        mock_profiler.return_value = mock_profiler_instance
        mock_system_monitor.return_value = mock_system_monitor_instance
        
        # Mock system monitor update result
        mock_system_monitor_instance.update.return_value = {
            "cpu_percent": 75.5,
            "memory_percent": 60.2,
            "disk_read": 1024,
            "disk_write": 2048
        }
        
        monitor = ResourceMonitor(autostart=False)
        result = monitor.update()
        
        # Verify update was called and result returned
        mock_system_monitor_instance.update.assert_called_once()
        self.assertEqual(result, {
            "cpu_percent": 75.5,
            "memory_percent": 60.2,
            "disk_read": 1024,
            "disk_write": 2048
        })
        
        # Verify instant event was recorded
        mock_profiler_instance.record_instant_event.assert_called_once()
        call_args = mock_profiler_instance.record_instant_event.call_args
        
        self.assertEqual(call_args[1]["name"], "Resource Usage")
        self.assertEqual(call_args[1]["timestamp"], 400.0)
        self.assertEqual(call_args[1]["process_id"], "monitor-host")
        self.assertEqual(call_args[1]["thread_id"], None)
        self.assertEqual(call_args[1]["args"], result)

    @patch('socket.gethostname', return_value='monitor-host')
    @patch('dasf.profile.plugins.EventProfiler')
    @patch('dasf.profile.plugins.SystemMonitor')
    @patch('dasf.profile.plugins.PeriodicCallback')
    def test_start_and_stop_methods(self, mock_callback, mock_system_monitor, 
                                   mock_profiler, mock_hostname):
        mock_profiler_instance = Mock()
        mock_callback_instance = Mock()
        mock_profiler.return_value = mock_profiler_instance
        mock_callback.return_value = mock_callback_instance
        
        monitor = ResourceMonitor(autostart=False)
        
        monitor.start()
        mock_callback_instance.start.assert_called_once()
        
        monitor.stop()
        mock_profiler_instance.commit.assert_called_once()
        mock_callback_instance.stop.assert_called_once()


class TestGPUAnnotationPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = GPUAnnotationPlugin(name="TestGPUPlugin")
        self.mock_worker = Mock()
        self.mock_worker.name = "gpu-worker"

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2'})
    @patch('builtins.print')
    def test_setup(self, mock_print):
        self.plugin.setup(self.mock_worker)
        
        self.assertEqual(self.plugin.worker, self.mock_worker)
        self.assertEqual(self.plugin.gpu_num, 0)
        
        mock_print.assert_called_once_with(
            "Setting up GPU annotation plugin for worker gpu-worker. GPU: 0"
        )

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,3,4'})
    def test_setup_different_gpu(self):
        self.plugin.setup(self.mock_worker)
        self.assertEqual(self.plugin.gpu_num, 2)

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'})
    @patch('pynvml.nvmlDeviceGetHandleByIndex')
    @patch('nvtx.start_range')
    def test_transition_to_executing(self, mock_start_range, mock_get_handle):
        mock_handle = Mock()
        mock_get_handle.return_value = mock_handle
        mock_start_range.return_value = "nvtx_mark_123"
        
        self.plugin.setup(self.mock_worker)
        
        task_key = "gpu-task-456"
        self.plugin.transition(task_key, "ready", "executing")
        
        mock_get_handle.assert_called_once_with(1)
        mock_start_range.assert_called_once_with(message=task_key, domain="compute")
        
        # Verify mark is stored
        self.assertEqual(self.plugin.marks[task_key], "nvtx_mark_123")

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'})
    @patch('pynvml.nvmlDeviceGetHandleByIndex')
    @patch('nvtx.end_range')
    def test_transition_from_executing(self, mock_end_range, mock_get_handle):
        mock_handle = Mock()
        mock_get_handle.return_value = mock_handle
        
        self.plugin.setup(self.mock_worker)
        
        # Setup existing mark
        task_key = "gpu-task-789"
        self.plugin.marks[task_key] = "nvtx_mark_789"
        
        self.plugin.transition(task_key, "executing", "memory")
        
        mock_get_handle.assert_called_once_with(0)
        mock_end_range.assert_called_once_with("nvtx_mark_789")
        
        # Verify mark is removed
        self.assertNotIn(task_key, self.plugin.marks)

    @patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'})
    @patch('pynvml.nvmlDeviceGetHandleByIndex')
    @patch('nvtx.start_range')
    @patch('nvtx.end_range')
    def test_transition_no_action_for_other_states(self, mock_end_range, mock_start_range, mock_get_handle):
        self.plugin.setup(self.mock_worker)
        
        # Transition that doesn't involve executing
        self.plugin.transition("task", "ready", "memory")
        
        mock_get_handle.assert_not_called()
        mock_start_range.assert_not_called()
        mock_end_range.assert_not_called()


class TestPluginInitialization(unittest.TestCase):
    def test_worker_task_plugin_default_name(self):
        plugin = WorkerTaskPlugin()
        self.assertEqual(plugin.name, "TracePlugin")

    def test_worker_task_plugin_custom_name(self):
        plugin = WorkerTaskPlugin(name="CustomWorkerPlugin")
        self.assertEqual(plugin.name, "CustomWorkerPlugin")

    def test_resource_monitor_default_values(self):
        with patch('socket.gethostname', return_value='test-host'), \
             patch('dasf.profile.plugins.EventProfiler'), \
             patch('dasf.profile.plugins.SystemMonitor'), \
             patch('dasf.profile.plugins.PeriodicCallback'):
            
            monitor = ResourceMonitor(autostart=False)
            self.assertEqual(monitor.time, 100)
            self.assertEqual(monitor.name, "ResourceMonitor")

    def test_gpu_annotation_plugin_default_name(self):
        plugin = GPUAnnotationPlugin()
        self.assertEqual(plugin.name, "GPUAnnotationPlugin")
        self.assertIsNone(plugin.gpu_num)
        self.assertEqual(plugin.marks, {})


if __name__ == '__main__':
    unittest.main()
