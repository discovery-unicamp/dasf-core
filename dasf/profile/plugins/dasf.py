import time
import socket
from distributed.diagnostics.plugin import WorkerPlugin, SchedulerPlugin
from dasf.profile.event import add_trace_complete
from dasf.pipeline import PipelinePlugin


class PipelineTaskTimer(PipelinePlugin):
    def __init__(self):
        self.start_times = dict()
        self.hostname = socket.gethostname()

    def on_task_start(self, func, params, name):
        self.start_times[name] = round(time.time()*1000)
        print(f"Pipeline Task Timer Start: {name}: {self.start_times[name]}")

    def on_task_end(self, func, params, name, ret):
        duration = round(time.time()*1000) - self.start_times[name]
        print(f"Pipeline Task Timer End: {name}: {duration}")
        add_trace_complete(
            name=name,
            process_id="dasf-core",
            thread_id=9999,
            timestamp=self.start_times[name],
            duration=duration,
            category=["dasf", "task time"],
            data={"task": name},
        )
