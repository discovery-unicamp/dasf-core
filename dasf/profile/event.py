import time
import threading
import os
import json
from pathlib import Path
from typing import List, Optional
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
    pids = set()
    tids = set()
    # stack_frames = []
    for trace in trace_events:
        if isinstance(trace, TraceEvent):
            pids.add(trace.process_id)
            tids.add((trace.process_id, trace.thread_id))

    pids = list(pids)
    tids = list(tids)

    for trace in trace_events:
        if isinstance(trace, TraceEvent):
            t = {
                "name": trace.name,
                "ph": trace.phase,
                "cat": ",".join(trace.category) if trace.category else "default",
                "ts": trace.timestamp * 1e6,
                "pid": pids.index(trace.process_id),
                "tid": tids.index((trace.process_id, trace.thread_id)),
            }

            if trace.data is not None:
                t["args"] = trace.data
            if trace.thread_timestamp is not None:
                t["tts"] = trace.thread_timestamp
            if trace.color_name is not None:
                t["cname"] = trace.color_name
            if trace.duration is not None:
                t["dur"] = trace.duration * 1e6
            if trace.thread_duration is not None:
                t["tdur"] = trace.thread_duration
            # pids.add(trace.process_id)
            # threads.add((trace.process_id, trace.thread_id))
            traces.append(t)

    # print(f"PIDS: {pids}")
    for pid in pids:
        traces.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pids.index(pid),
                "args": {"name": pid},
            }
        )

    for pid, tid in tids:
        traces.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pids.index(pid),
                "tid": tids.index((pid, tid)),
                "args": {"name": tid},
            }
        )

    traces = {
        "traceEvents": traces,
        # "stackFrames": stack_frames
    }

    if trace_options is not None:
        traces.update(trace_options)

    format_kwargs = format_kwargs or {}
    return json.dumps(traces, **format_kwargs)


class Profile:
    def __init__(
        self,
        trace_file: str = "traces.txt",
        remove_old_trace_file: bool = True,
        processed_filename: Optional[str] = "profile.json",
        process_trace_options: dict = None,
        process_trace_kwargs: dict = None,
    ):
        self.trace_file = Path(trace_file)
        self.processed_filename = Path(processed_filename)
        self.remove_old_trace_file = remove_old_trace_file
        self.process_trace_options = process_trace_options
        self.process_trace_kwargs = process_trace_kwargs

    def __enter__(self):
        if self.remove_old_trace_file:
            self.trace_file.unlink(missing_ok=True)
        db = SingleFileTraceDatabase(self.trace_file)
        TraceDatabase(db)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.processed_filename is not None:
            # print("Processing traces...")
            traces = get_traces()
            with self.processed_filename.open("w") as f:
                f.write(
                    to_chrome_event_format(
                        traces, self.process_trace_options, self.process_trace_kwargs
                    )
                )
            print(f"Chrome trace file written to {self.processed_filename}")
