import json

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List


@dataclass
class TraceEvent:
    # Mandadory options
    name: str  # name: name of the event, (displayed in Trace Viewer).
    phase: str  # ph: the event type (single character).
    timestamp: float  # ts: tracing clock timestamp (microsecond granularity).
    process_id: str  # pid: the process identifier.
    thread_id: str  # tid: the thread identifier.

    # Global options
    category: List[str] = None  # cat: event categoies.
    data: dict = None  # args: dictionary of arguments provided for the event.
    thread_timestamp: float = None  # tts: thread clock timestamp (microsecond granularity).
    color_name: str = None  # cname: color name for the event

    # Duration event fields (X)
    duration: float = None          # dur: tracing clock duration of complete events (microsecond granularity).
    thread_duration: float = None   # tdur: the thread clock duration of complete events (microsecond granularity).


class TraceDatabase:
    def add_trace_event(self, trace: TraceEvent):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError

    def get_traces(self) -> List[TraceEvent]:
        raise NotImplementedError


class SingleFileTraceDatabase(TraceDatabase):
    def __init__(self, path: Path, encoder: callable = json.dumps, decoder: callable = json.loads):
        self._path = Path(path)
        self._encoder = encoder
        self._decoder = decoder

    # Note, this process is not process-safe!
    def add_trace_event(self, trace: TraceEvent) -> int:  # returns the record id
        obj = f"trace: {self._encoder(asdict(trace))}\n"
        with self._path.open("a") as f:
            f.write(obj)

    def get_traces(self) -> List[TraceEvent]:
        traces = []
        if not self._path.exists():
            return traces
        with self._path.open("r") as f:
            for line in f:
                if not line:
                    continue
                if line.startswith("trace: "):
                    trace = TraceEvent(**self._decoder(line[7:]))
                    traces.append(trace)
        return traces
