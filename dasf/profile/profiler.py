"""A module for profiling objects and helpers."""
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
    """An enumeration of the different event phases."""
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
    """An enumeration of the different instant event scopes."""
    GLOBAL = "g"
    PROCESS = "p"
    THREAD = "t"


@dataclass
class InstantEvent:
    """
    A dataclass for instant events.

    Parameters
    ----------
    name : str
        The name of the event.
    timestamp : float
        The timestamp of the event.
    phase : str
        The phase of the event.
    scope : str
        The scope of the event.
    process_id : int
        The process ID of the event.
    thread_id : int
        The thread ID of the event.
    args : dict
        The arguments of the event.
    """
    name: str
    timestamp: float
    phase: str = EventPhases.INSTANT
    scope: str = InstantEventScope.GLOBAL
    process_id: int = 0
    thread_id: int = 0
    args: dict = field(default_factory=dict)


@dataclass
class CompleteEvent:
    """
    A dataclass for complete events.

    Parameters
    ----------
    name : str
        The name of the event.
    timestamp : float
        The timestamp of the event.
    duration : float
        The duration of the event.
    phase : str
        The phase of the event.
    process_id : int
        The process ID of the event.
    thread_id : int
        The thread ID of the event.
    args : dict
        The arguments of the event.
    """
    name: str
    timestamp: float
    duration: float
    phase: str = EventPhases.COMPLETE
    process_id: int = 0
    thread_id: int = 0
    args: dict = field(default_factory=dict)


@dataclass
class DurationBeginEvent:
    """
    A dataclass for duration begin events.

    Parameters
    ----------
    name : str
        The name of the event.
    timestamp : float
        The timestamp of the event.
    phase : str
        The phase of the event.
    process_id : int
        The process ID of the event.
    thread_id : int
        The thread ID of the event.
    args : dict
        The arguments of the event.
    """
    name: str
    timestamp: float
    phase: str = EventPhases.DURATION_BEGIN
    process_id: int = 0
    thread_id: int = 0
    args: dict = field(default_factory=dict)


@dataclass
class DurationEndEvent:
    """
    A dataclass for duration end events.

    Parameters
    ----------
    name : str
        The name of the event.
    timestamp : float
        The timestamp of the event.
    phase : str
        The phase of the event.
    process_id : int
        The process ID of the event.
    thread_id : int
        The thread ID of the event.
    args : dict
        The arguments of the event.
    """
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
    """An abstract class for event databases."""
    def open(self) -> "EventDatabase":
        """Open the database."""
        return self

    @abstractmethod
    def record(self, event: EventTypes):
        """
        Record an event.

        Parameters
        ----------
        event : EventTypes
            The event to record.
        """
        raise NotImplementedError

    @abstractmethod
    def commit(self):
        """Commit the events to the database."""
        raise NotImplementedError

    @abstractmethod
    def get_traces(self) -> Iterable[EventTypes]:
        """
        Get the traces from the database.

        Returns
        -------
        Iterable[EventTypes]
            An iterable of the traces.
        """
        raise NotImplementedError

    def close(self):
        """Close the database."""
        pass

    def __enter__(self):
        """Enter the context manager."""
        return self.open()

    def __exit__(self, *args, **kwargs):
        """Exit the context manager."""
        self.close()


class FileDatabase(EventDatabase):
    """
    An event database that stores events in a file.

    Parameters
    ----------
    database_file : str
        The path to the database file.
    commit_threshold : int
        The number of events to queue before committing to the file.
    remove_old_output_file : bool
        Whether to remove the old output file.
    commit_on_close : bool
        Whether to commit the events when the database is closed.
    lock_timeout : int
        The timeout for the file lock.
    default_byte_size : int
        The default byte size for the event size.
    flush : bool
        Whether to flush the file after writing.
    """
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
        """
        Constructor for the FileDatabase class.

        Parameters
        ----------
        database_file : str
            The path to the database file.
        commit_threshold : int
            The number of events to queue before committing to the file.
        remove_old_output_file : bool
            Whether to remove the old output file.
        commit_on_close : bool
            Whether to commit the events when the database is closed.
        lock_timeout : int
            The timeout for the file lock.
        default_byte_size : int
            The default byte size for the event size.
        flush : bool
            Whether to flush the file after writing.
        """
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
        """
        Record an event.

        Parameters
        ----------
        event : EventTypes
            The event to record.
        """
        self.queue.put(event)
        if self.queue.qsize() >= self.commit_threshold:
            self.commit()

    def commit(self):
        """Commit the events to the database."""
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
        """
        Get the traces from the database.

        Returns
        -------
        Iterable[EventTypes]
            An iterable of the traces.
        """
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
        """Close the database."""
        if self.commit_on_close:
            self.commit()

    def __str__(self) -> str:
        """Return the string representation of the database."""
        return f"FileDatabase at {self.database_file}"

    def __repr__(self) -> str:
        """Return the string representation of the database."""
        return f"FileDatabase at {self.database_file}"


# Singleton instance of the database
class EventProfiler:
    """
    A class for profiling events.

    Parameters
    ----------
    database_file : str
        The path to the database file.
    database_creation_kwargs : dict
        The keyword arguments for creating the database.
    database : EventDatabase
        The event database.
    """
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
        """
        Constructor for the EventProfiler class.

        Parameters
        ----------
        database_file : str
            The path to the database file.
        database_creation_kwargs : dict
            The keyword arguments for creating the database.
        database : EventDatabase
            The event database.
        """
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
        """
        Record an event.

        Parameters
        ----------
        event : EventTypes
            The event to record.
        """
        self.database.record(event)

    def record_complete_event(
        self,
        name: str, timestamp: float, duration: float, **kwargs
    ):
        """
        Record a complete event.

        Parameters
        ----------
        name : str
            The name of the event.
        timestamp : float
            The timestamp of the event.
        duration : float
            The duration of the event.
        """
        self._record(CompleteEvent(name, timestamp, duration, **kwargs))

    def record_instant_event(self, name: str, timestamp: float, **kwargs):
        """
        Record an instant event.

        Parameters
        ----------
        name : str
            The name of the event.
        timestamp : float
            The timestamp of the event.
        """
        self._record(InstantEvent(name, timestamp, **kwargs))

    def record_duration_begin_event(self, name: str, timestamp: float, **kwargs):
        """
        Record a duration begin event.

        Parameters
        ----------
        name : str
            The name of the event.
        timestamp : float
            The timestamp of the event.
        """
        self._record(DurationBeginEvent(name, timestamp, **kwargs))

    def record_duration_end_event(self, name: str, timestamp: float, **kwargs):
        """
        Record a duration end event.

        Parameters
        ----------
        name : str
            The name of the event.
        timestamp : float
            The timestamp of the event.
        """
        self._record(DurationEndEvent(name, timestamp, **kwargs))

    def get_traces(self) -> Iterable[EventTypes]:
        """
        Get the traces from the database.

        Returns
        -------
        Iterable[EventTypes]
            An iterable of the traces.
        """
        return self.database.get_traces()

    def __str__(self):
        """Return the string representation of the profiler."""
        return f"EventProfiler(database={self.database})"

    def __repr__(self) -> str:
        """Return the string representation of the profiler."""
        return f"EventProfiler(database={self.database})"

    def commit(self):
        """Commit the events to the database."""
        self.database.commit()
