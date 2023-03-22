import time
import socket
from distributed.diagnostics.plugin import WorkerPlugin, SchedulerPlugin
from dasf.profile.event import add_trace_complete


class TaskTimePlugin(WorkerPlugin):
    def setup(self, worker):
        self.worker = worker
        self.start_times = dict()
        self.transfer_times = dict()
        self.waiting_time = dict()
        self.hostname = socket.gethostname()

    def transition(self, key, start, finish, *args, **kwargs):
        if finish == "executing": # start execting     
            self.start_times[key] = time.time()
            if key in self.waiting_time: 
                duration = time.time() - self.waiting_time[key]
                add_trace_complete(
                    name="Waiting",
                    process_id=self.hostname,
                    thread_id=f"worker-{self.worker.name}",
                    timestamp=self.transfer_times[key],
                    duration=duration,
                    category=["worker", "dask", "waiting"],
                    data={"from": start, "to": finish, "key": key},
                )
            
        elif (start == "executing" or start == "long-running") and key in self.start_times: # end executing
            if key in self.start_times:
                duration = time.time() - self.start_times[key]
                add_trace_complete(
                    name="Processing",
                    process_id=self.hostname,
                    thread_id=f"worker-{self.worker.name}",
                    timestamp=self.start_times[key],
                    duration=duration,
                    category=["worker", "dask", "processing"],
                    data={"from": start, "to": finish, "key": key},
                )
        if finish == "fetch": # start transfer
            self.transfer_times[key] = time.time()
        elif start == "flight" and finish == "memory":
            if key in self.transfer_times:
                duration = time.time() - self.transfer_times[key]
                add_trace_complete(
                    name="Transfering",
                    process_id=self.hostname,
                    thread_id=f"worker-{self.worker.name}",
                    timestamp=self.transfer_times[key],
                    duration=duration,
                    category=["worker", "dask", "transfering"],
                    data={"from": start, "to": finish, "key": key},
                )
            
        if finish == "waiting":
            self.waiting_time[key] = time.time()    
        
