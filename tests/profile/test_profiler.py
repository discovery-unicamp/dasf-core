#!/usr/bin/env python3

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import ormsgpack

from dasf.profile.profiler import (
    CompleteEvent,
    DurationBeginEvent,
    DurationEndEvent,
    EventDatabase,
    EventPhases,
    EventProfiler,
    EventTypes,
    FileDatabase,
    InstantEvent,
    InstantEventScope,
    event_classes,
)


class TestEventClasses(unittest.TestCase):
    def test_instant_event_creation(self):
        event = InstantEvent(
            name="test_event",
            timestamp=123.456,
            args={"key": "value"}
        )
        self.assertEqual(event.name, "test_event")
        self.assertEqual(event.timestamp, 123.456)
        self.assertEqual(event.phase, EventPhases.INSTANT)
        self.assertEqual(event.scope, InstantEventScope.GLOBAL)
        self.assertEqual(event.process_id, 0)
        self.assertEqual(event.thread_id, 0)
        self.assertEqual(event.args, {"key": "value"})

    def test_complete_event_creation(self):
        event = CompleteEvent(
            name="compute_task",
            timestamp=100.0,
            duration=5.5,
            process_id=1,
            thread_id=2,
            args={"size": 1024}
        )
        self.assertEqual(event.name, "compute_task")
        self.assertEqual(event.timestamp, 100.0)
        self.assertEqual(event.duration, 5.5)
        self.assertEqual(event.phase, EventPhases.COMPLETE)
        self.assertEqual(event.process_id, 1)
        self.assertEqual(event.thread_id, 2)
        self.assertEqual(event.args, {"size": 1024})

    def test_duration_begin_event_creation(self):
        event = DurationBeginEvent(
            name="start_task",
            timestamp=200.0,
            args={"task_id": "task_123"}
        )
        self.assertEqual(event.name, "start_task")
        self.assertEqual(event.timestamp, 200.0)
        self.assertEqual(event.phase, EventPhases.DURATION_BEGIN)

    def test_duration_end_event_creation(self):
        event = DurationEndEvent(
            name="end_task",
            timestamp=300.0
        )
        self.assertEqual(event.name, "end_task")
        self.assertEqual(event.timestamp, 300.0)
        self.assertEqual(event.phase, EventPhases.DURATION_BEGIN)

    def test_event_classes_mapping(self):
        self.assertEqual(event_classes[EventPhases.COMPLETE], CompleteEvent)
        self.assertEqual(event_classes[EventPhases.INSTANT], InstantEvent)
        self.assertEqual(event_classes[EventPhases.DURATION_BEGIN], DurationBeginEvent)
        self.assertEqual(event_classes[EventPhases.DURATION_END], DurationEndEvent)


class TestEventDatabase(unittest.TestCase):
    def test_event_database_is_abstract(self):
        with self.assertRaises(TypeError):
            EventDatabase()

    def test_event_database_context_manager_interface(self):
        class MockEventDatabase(EventDatabase):
            def record(self, event):
                pass

            def commit(self):
                pass

            def get_traces(self):
                return []

        db = MockEventDatabase()
        with db as opened_db:
            self.assertIs(opened_db, db)


class TestFileDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test_traces.msgpack")

    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)

    def test_file_database_creation(self):
        db = FileDatabase(
            database_file=self.temp_file,
            commit_threshold=100,
            remove_old_output_file=True
        )
        self.assertEqual(db.database_file, Path(self.temp_file))
        self.assertEqual(db.commit_threshold, 100)
        self.assertTrue(db.commit_on_close)

    def test_record_and_commit_events(self):
        db = FileDatabase(
            database_file=self.temp_file,
            commit_threshold=2,
            remove_old_output_file=True
        )

        event1 = InstantEvent("test1", 100.0)
        event2 = CompleteEvent("test2", 200.0, 5.0)

        db.record(event1)
        self.assertEqual(db.queue.qsize(), 1)

        db.record(event2)
        self.assertEqual(db.queue.qsize(), 0)

        self.assertTrue(os.path.exists(self.temp_file))

    def test_get_traces_from_file(self):
        db = FileDatabase(
            database_file=self.temp_file,
            remove_old_output_file=True
        )

        event1 = InstantEvent("test1", 100.0, args={"key": "value1"})
        event2 = CompleteEvent("test2", 200.0, 5.0, args={"key": "value2"})

        db.record(event1)
        db.record(event2)
        db.commit()

        traces = list(db.get_traces())
        self.assertEqual(len(traces), 2)

        self.assertEqual(traces[0].name, "test1")
        self.assertEqual(traces[0].timestamp, 100.0)
        self.assertEqual(traces[0].args, {"key": "value1"})

        self.assertEqual(traces[1].name, "test2")
        self.assertEqual(traces[1].timestamp, 200.0)
        self.assertEqual(traces[1].duration, 5.0)
        self.assertEqual(traces[1].args, {"key": "value2"})

    def test_close_commits_on_close(self):
        db = FileDatabase(
            database_file=self.temp_file,
            commit_threshold=100,
            commit_on_close=True
        )

        event = InstantEvent("test", 100.0)
        db.record(event)
        self.assertEqual(db.queue.qsize(), 1)

        db.close()
        self.assertEqual(db.queue.qsize(), 0)
        self.assertTrue(os.path.exists(self.temp_file))

    def test_remove_old_output_file(self):
        with open(self.temp_file, 'w') as f:
            f.write("old content")

        self.assertTrue(os.path.exists(self.temp_file))

        db = FileDatabase(
            database_file=self.temp_file,
            remove_old_output_file=True
        )

        self.assertFalse(os.path.exists(self.temp_file))

    def test_string_representation(self):
        db = FileDatabase(database_file=self.temp_file)
        expected = f"FileDatabase at {self.temp_file}"
        self.assertEqual(str(db), expected)
        self.assertEqual(repr(db), expected)


class TestEventProfiler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_event_profiler_with_default_database(self):
        with patch('dasf.profile.profiler.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value.hex = "12345678"
            mock_uuid.return_value.__getitem__ = lambda self, key: "12345678"[key]

            profiler = EventProfiler()
            self.assertIsInstance(profiler.database, FileDatabase)
            self.assertTrue(profiler.output_file.startswith("traces-"))

    def test_event_profiler_with_custom_database_file(self):
        test_file = os.path.join(self.temp_dir, "custom_traces.msgpack")
        profiler = EventProfiler(database_file=test_file)

        self.assertEqual(profiler.output_file, test_file)
        self.assertIsInstance(profiler.database, FileDatabase)

    def test_event_profiler_with_custom_database(self):
        mock_database = Mock(spec=EventDatabase)
        profiler = EventProfiler(database=mock_database)

        self.assertIs(profiler.database, mock_database)
        self.assertIsNone(profiler.output_file)

    def test_cannot_specify_both_database_and_file(self):
        mock_database = Mock(spec=EventDatabase)
        with self.assertRaises(ValueError):
            EventProfiler(database_file="test.msgpack", database=mock_database)

    def test_record_complete_event(self):
        mock_database = Mock(spec=EventDatabase)
        profiler = EventProfiler(database=mock_database)

        profiler.record_complete_event(
            "test_task", 100.0, 5.0, 
            process_id=1, thread_id=2, args={"size": 1024}
        )

        mock_database.record.assert_called_once()
        event = mock_database.record.call_args[0][0]
        self.assertIsInstance(event, CompleteEvent)
        self.assertEqual(event.name, "test_task")
        self.assertEqual(event.timestamp, 100.0)
        self.assertEqual(event.duration, 5.0)

    def test_record_instant_event(self):
        mock_database = Mock(spec=EventDatabase)
        profiler = EventProfiler(database=mock_database)

        profiler.record_instant_event(
            "instant_event", 200.0,
            scope=InstantEventScope.PROCESS
        )

        mock_database.record.assert_called_once()
        event = mock_database.record.call_args[0][0]
        self.assertIsInstance(event, InstantEvent)
        self.assertEqual(event.name, "instant_event")
        self.assertEqual(event.timestamp, 200.0)
        self.assertEqual(event.scope, InstantEventScope.PROCESS)

    def test_record_duration_begin_event(self):
        mock_database = Mock(spec=EventDatabase)
        profiler = EventProfiler(database=mock_database)

        profiler.record_duration_begin_event("begin_task", 300.0)

        mock_database.record.assert_called_once()
        event = mock_database.record.call_args[0][0]
        self.assertIsInstance(event, DurationBeginEvent)
        self.assertEqual(event.name, "begin_task")
        self.assertEqual(event.timestamp, 300.0)

    def test_record_duration_end_event(self):
        mock_database = Mock(spec=EventDatabase)
        profiler = EventProfiler(database=mock_database)

        profiler.record_duration_end_event("end_task", 400.0)

        mock_database.record.assert_called_once()
        event = mock_database.record.call_args[0][0]
        self.assertIsInstance(event, DurationEndEvent)
        self.assertEqual(event.name, "end_task")
        self.assertEqual(event.timestamp, 400.0)

    def test_get_traces(self):
        mock_database = Mock(spec=EventDatabase)
        expected_traces = [InstantEvent("test", 100.0)]
        mock_database.get_traces.return_value = expected_traces

        profiler = EventProfiler(database=mock_database)
        traces = profiler.get_traces()

        self.assertEqual(traces, expected_traces)
        mock_database.get_traces.assert_called_once()

    def test_commit(self):
        mock_database = Mock(spec=EventDatabase)
        profiler = EventProfiler(database=mock_database)

        profiler.commit()

        mock_database.commit.assert_called_once()

    def test_string_representation(self):
        mock_database = Mock(spec=EventDatabase)
        mock_database.__str__ = Mock(return_value="MockDatabase")
        
        profiler = EventProfiler(database=mock_database)
        expected = "EventProfiler(database=MockDatabase)"
        
        self.assertEqual(str(profiler), expected)
        self.assertEqual(repr(profiler), expected)
