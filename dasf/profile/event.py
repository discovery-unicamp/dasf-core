import time
import threading
import os
import json
from pathlib import Path
from typing import List
from dasf.profile.database import TraceEvent, TraceDatabase, SingleFileTraceDatabase


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TraceDatabase(metaclass=Singleton):
    db_name: str = "traces.txt"

    def __init__(self, database: TraceDatabase = None):
        self._database = database or SingleFileTraceDatabase(self.db_name)

    @property
    def database(self) -> TraceDatabase:
        return self._database


def get_time_ms():
    return time.time() * 1000


def add_trace_duration_begin(
    name: str,
    process_id: str,
    thread_id: str,
    category: List[str] = None,
    timestamp: float = None,
    thread_timestamp: float = None,
    data: dict = None,
):
    event = TraceEvent(
        name=name,
        category=category,
        phase="B",
        timestamp=get_time_ms(),
        process_id=process_id,
        thread_id=thread_id,
        data=data,
        thread_timestamp=thread_timestamp,
        color_name=None,
    )
    TraceDatabase().database.add_trace_event(event)


def add_trace_duration_end(
    name: str,
    process_id: str,
    thread_id: str,
    category: List[str] = None,
    timestamp: float = None,
    thread_timestamp: float = None,
    data: dict = None,
):
    event = TraceEvent(
        name=name,
        category=category,
        phase="E",
        timestamp=get_time_ms(),
        process_id=process_id,
        thread_id=thread_id,
        data=data,
        thread_timestamp=thread_timestamp,
        color_name=None,
    )
    TraceDatabase().database.add_trace_event(event)


def add_trace_complete(
    name: str,
    process_id: str,
    thread_id: str,
    timestamp: float,
    duration: float,
    thread_timestamp: float = None,
    thread_duration: float = None,
    category: List[str] = None,
    data: dict = None,
):
    if thread_timestamp is not None or thread_duration is not None:
        if thread_timestamp is None or thread_duration is None:
            raise ValueError(
                "initial_thread_timestamp and thread_duration must be set together"
            )
    event = TraceEvent(
        name=name,
        category=category,
        phase="X",
        timestamp=timestamp,
        duration=duration,
        process_id=process_id,
        thread_id=thread_id,
        data=data,
        thread_timestamp=thread_timestamp,
        thread_duration=thread_duration,
        color_name=None,
    )
    TraceDatabase().database.add_trace_event(event)


def get_traces() -> List[TraceEvent]:
    return TraceDatabase().database.get_traces()


def to_chrome_event_format(
    trace_events: List[TraceEvent],
    trace_options: dict = None,
    format_kwargs: dict = None,
) -> str:
    traces = []
    # stack_frames = []
    for trace in trace_events:
        if isinstance(trace, TraceEvent):
            t = {
                "name": trace.name,
                "ph": trace.phase,
                "cat": ",".join(trace.category) if trace.category else "default",
                "ts": trace.timestamp,
                "pid": trace.process_id,
                "tid": trace.thread_id,
            }

            if trace.data is not None:
                t["args"] = trace.data
            if trace.thread_timestamp is not None:
                t["tts"] = trace.thread_timestamp
            if trace.color_name is not None:
                t["cname"] = trace.color_name
            if trace.duration is not None:
                t["dur"] = trace.duration
            if trace.thread_duration is not None:
                t["tdur"] = trace.thread_duration

            traces.append(t)

    traces = {
        "traceEvents": traces,
        # "stackFrames": stack_frames
    }

    if trace_options is not None:
        traces.update(trace_options)

    format_kwargs = format_kwargs or {}
    return json.dumps(traces, **format_kwargs)
