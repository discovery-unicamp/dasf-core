import atexit
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from typing import Iterable, Union

import ormsgpack
import portalocker


class EventPhases:
    COMPLETE = "X"
    DURATION_BEGIN = "B"
    DURATION_END = "E"
    INSTANT = "I"
    ASYNC_BEGIN = "b"
    ASYNC_INSTANT = "n"
    ASYNC_END = "e"
    FLOW_BEGIN = "s"
    FLOW_STEP = "t"
    FLOW_END = "f"
    COUNTER = "C"
    OBJECT_CREATED = "N"
    OBJECT_SNAPSHOT = "O"
    OBJECT_DESTROYED = "D"
    METADATA = "M"
    MARK = "R"


class InstantEventScope:
    GLOBAL = "g"
    PROCESS = "p"
    THREAD = "t"


@dataclass
class InstantEvent:
    name: str
    timestamp: float
    phase: str = EventPhases.INSTANT
    scope: str = InstantEventScope.GLOBAL
    process_id: int = 0
    thread_id: int = 0
    args: dict = field(default_factory=dict)


@dataclass
class CompleteEvent:
    name: str
    timestamp: float
    duration: float
    phase: str = EventPhases.COMPLETE
    process_id: int = 0
    thread_id: int = 0
    args: dict = field(default_factory=dict)


@dataclass
class DurationBeginEvent:
    name: str
    timestamp: float
    phase: str = EventPhases.DURATION_BEGIN
    process_id: int = 0
    thread_id: int = 0
    args: dict = field(default_factory=dict)


@dataclass
class DurationEndEvent:
    name: str
    timestamp: float
    phase: str = EventPhases.DURATION_BEGIN
    process_id: int = 0
    thread_id: int = 0
    args: dict = field(default_factory=dict)


EventTypes = Union[CompleteEvent, InstantEvent, DurationBeginEvent, DurationEndEvent]
event_classes = {
    EventPhases.COMPLETE: CompleteEvent,
    EventPhases.INSTANT: InstantEvent,
    EventPhases.DURATION_BEGIN: DurationBeginEvent,
    EventPhases.DURATION_END: DurationEndEvent,
}


class EventDatabase(ABC):
    def open(self) -> "EventDatabase":
        return self

    @abstractmethod
    def record(self, event: EventTypes):
        raise NotImplementedError

    @abstractmethod
    def commit(self):
        raise NotImplementedError

    @abstractmethod
    def get_traces(self) -> Iterable[EventTypes]:
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *args, **kwargs):
        self.close()


class FileDatabase(EventDatabase):
    def __init__(
        self,
        database_file: str = "traces.msgpack",
        commit_threshold: int = 5000,
        remove_old_output_file: bool = False,
        commit_on_close: bool = True,
        lock_timeout: int = 30,
        default_byte_size: int = 8,
        flush: bool = True,
    ):
        self.database_file = Path(database_file)
        self.commit_threshold = commit_threshold
        self.commit_on_close = commit_on_close
        self.queue = SimpleQueue()
        self.lock_timeout = lock_timeout
        self.byte_size = default_byte_size
        self.flush = flush
        if remove_old_output_file:
            self.database_file.unlink(missing_ok=True)

        # Register a function to commit the events when the program exits
        atexit.register(self.close)

    def record(self, event: EventTypes):
        self.queue.put(event)
        if self.queue.qsize() >= self.commit_threshold:
            self.commit()

    def commit(self):
        # TODO: implement async commit.
        # Create a exclusive lock file to prevent other processes from
        # writing to the file.
        with portalocker.Lock(self.database_file, mode="ab",
                              timeout=self.lock_timeout) as f:
            # Write each event to file.
            # Always write the size of the event first (8 bytes) then the
            # event data.
            events = []
            while not self.queue.empty():
                event = self.queue.get()
                packed_data = ormsgpack.packb(event)
                size = len(packed_data).to_bytes(self.byte_size, byteorder="big")
                events.append(size)
                events.append(packed_data)
            events = b"".join(events)
            f.write(events)

            if self.flush:
                f.flush()
                os.fsync(f.fileno())

    def get_traces(self) -> Iterable[EventTypes]:
        with self.database_file.open("rb") as f:
            while True:
                chunk = f.read(self.byte_size)
                if chunk == b"":
                    return
                size = int.from_bytes(chunk, byteorder="big")
                data = f.read(size)
                data = ormsgpack.unpackb(data)
                data = event_classes[data["phase"]](**data)
                yield data

    def close(self):
        if self.commit_on_close:
            self.commit()

    def __str__(self) -> str:
        return f"FileDatabase at {self.database_file}"

    def __repr__(self) -> str:
        return f"FileDatabase at {self.database_file}"


# Singleton instance of the database
class EventProfiler:
    traces_file_prefix = "traces-"

    default_database = FileDatabase
    default_database_kwargs = {
        "commit_threshold": 1000,
        "remove_old_output_file": False,
        "commit_on_close": True,
    }

    def __init__(
        self,
        database_file: str = None,
        database_creation_kwargs: dict = None,
        database: EventDatabase = None,
    ):
        self.output_file = None
        if database is not None:
            if database_file is not None:
                raise ValueError(
                    "Cannot specify both output_file and database arguments"
                )
            self.database = database
        else:
            if database_creation_kwargs is None:
                database_creation_kwargs = self.default_database_kwargs
            if database_file is None:
                database_file = (f"{self.traces_file_prefix}"
                                 f"{str(uuid.uuid4())[:8]}.msgpack")
            self.output_file = database_file
            self.database = self.default_database(
                database_file, **database_creation_kwargs
            )

    def _record(self, event: EventTypes):
        self.database.record(event)

    def record_complete_event(
        self,
        name: str, timestamp: float, duration: float, **kwargs
    ):
        self._record(CompleteEvent(name, timestamp, duration, **kwargs))

    def record_instant_event(self, name: str, timestamp: float, **kwargs):
        self._record(InstantEvent(name, timestamp, **kwargs))

    def record_duration_begin_event(self, name: str, timestamp: float, **kwargs):
        self._record(DurationBeginEvent(name, timestamp, **kwargs))

    def record_duration_end_event(self, name: str, timestamp: float, **kwargs):
        self._record(DurationEndEvent(name, timestamp, **kwargs))

    def get_traces(self) -> Iterable[EventTypes]:
        return self.database.get_traces()

    def __str__(self):
        return f"EventProfiler(database={self.database})"

    def __repr__(self) -> str:
        return f"EventProfiler(database={self.database})"

    def commit(self):
        self.database.commit()
