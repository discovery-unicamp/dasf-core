from pathlib import Path
import threading
import time
import json
import psutil

from pyspectator.computer import Computer
from pyspectator.processor import Cpu
from pyspectator.convert import UnitByte
import schedule

from dasf.pipeline import PipelinePlugin

class Format:
    @staticmethod
    def temperature(value):
        formatted_value = ''
        if isinstance(value, (int, float)):
            formatted_value = str(value) + '°C'
        return formatted_value

    @staticmethod
    def byte_value(value):
        formatted_value = ''
        if isinstance(value, (int, float)):
            value, unit = UnitByte.auto_convert(value)
            value = '{:.2f}'.format(value)
            unit = UnitByte.get_name_reduction(unit)
            formatted_value = value + unit
        return formatted_value

    @staticmethod
    def percent(value):
        formatted_value = ''
        if isinstance(value, (int, float)):
            formatted_value = str(value) + '%'
        return formatted_value

def run_continuously(scheduler, interval=1):
    """Continuously run, while executing pending jobs at each
    elapsed time interval.
    @return cease_continuous_run: threading. Event which can
    be set to cease continuous run. Please note that it is
    *intended behavior that run_continuously() does not run
    missed jobs*. For example, if you've registered a job that
    should run every minute and you set a continuous run
    interval of one hour then your job won't be run 60 times
    at each interval but only once.
    """
    cease_continuous_run = threading.Event()

    class ScheduleThread(threading.Thread):
        @classmethod
        def run(cls):
            while not cease_continuous_run.is_set():
                scheduler.run_pending()
                time.sleep(interval)

    continuous_thread = ScheduleThread()
    continuous_thread.start()
    return cease_continuous_run


class ResourceMonitor(PipelinePlugin):
    def __init__(self, path: str = None, monitor_interval=0.1, verbose: bool = False):
        self.path = path
        self.interval = monitor_interval
        self.events = []
        self.scheduler = schedule.Scheduler()
        self.stop_timer = None
        self.verbose = verbose

    @staticmethod
    def get_info(event_list, verbose: bool = False):
        computer = Computer()
        data = {
            "timestamp": time.time(),
            "cpu.load": Format.percent(computer.processor.load),
            "cpu.percent": Format.percent(psutil.cpu_percent(percpu=False)),
            "cpus.percent": {
                i: Format.percent(percent)
                for i, percent in enumerate(psutil.cpu_percent(percpu=True))
            },
            # "cpu.temperature": Format.temperature(computer.processor.temperature),
            "bytes.sent": Format.byte_value(computer.network_interface.bytes_sent),
            "bytes.received": Format.byte_value(computer.network_interface.bytes_recv),
            "virtual_memory.available": Format.byte_value(computer.virtual_memory.available),
            "virtual_memory.total": Format.byte_value(computer.virtual_memory.total),
            "virtual_memory.used": Format.percent(computer.virtual_memory.used_percent)
        }
        event_list.append(data)
        if verbose:
            print(data)

    def on_pipeline_start(self, fn_keys):
        self.scheduler.every(self.interval).seconds.do(ResourceMonitor.get_info, event_list=self.events, verbose=self.verbose)
        self.stop_timer = run_continuously(scheduler=self.scheduler, interval=self.interval)

    def on_pipeline_end(self):
        if self.stop_timer is not None:
            self.stop_timer.set()

            if self.path is not None:
                with Path(self.path).open("w") as f:
                    json.dump(self.events, f, sort_keys=True, indent=4)
