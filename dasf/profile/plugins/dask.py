import time
import socket
from distributed.diagnostics.plugin import WorkerPlugin, SchedulerPlugin
from dasf.profile.event import add_trace_complete


class TaskTimePlugin(WorkerPlugin):
    def setup(self, worker):
        self.worker = worker
        self.start_times = dict()
        self.hostname = socket.gethostname()

    def transition(self, key, start, finish, *args, **kwargs):
        if finish == "executing":
            self.start_times[key] = round(time.time()*1000)
        elif start == "executing" and key in self.start_times:
            duration = round(time.time()*1000) - self.start_times[key]
            # add_trace_complete(
            #     name=key,
            #     process_id=self.hostname,
            #     thread_id=self.worker.name,
            #     timestamp=self.start_times[key],
            #     duration=duration,
            #     category=["worker", "processing"],
            #     data={"from": start, "to": finish},
            # )

            add_trace_complete(
                name="Dask task",
                process_id=self.hostname,
                thread_id=self.worker.name,
                timestamp=self.start_times[key],
                duration=duration,
                category=["worker", "dask", "processing time"],
                data={"from": start, "to": finish, "key": key},
            )
