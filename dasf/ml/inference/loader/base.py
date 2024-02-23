from dask.distributed import Worker

from dasf.utils.funcs import get_dask_running_client
from dasf.utils.decorators import task_handler


class BaseLoader:
    def __init__(self):
        self.model_instances = {}

    def inference(self, model, data):
        raise NotImplementedError("Inference must be implemented")

    def load_model(self):
        raise NotImplementedError("Load Model must be implemented")

    def load_model_distributed(self, **kwargs):
        # try:
        Worker.model = self.load_model(**kwargs)
        return "UP"

    # except:
    # return "DOWN"

    def _lazy_load(self, **kwargs):
        client = get_dask_running_client()
        self.model_instances = {}
        if client:
            worker_addresses = list(client.scheduler_info()["workers"].keys())
            self.model_instances = client.run(
                self.load_model_distributed, **kwargs, workers=worker_addresses
            )

    def _load(self, **kwargs):
        self.model_instances = {"local": self.load_model(**kwargs)}

    def _lazy_load_cpu(self, **kwargs):
        if not (hasattr(self, "device") and self.device):
            self.device = "cpu"
        self._lazy_load(**kwargs)

    def _lazy_load_gpu(self, **kwargs):
        if not (hasattr(self, "device") and self.device):
            self.device = "gpu"
        self._lazy_load(**kwargs)

    def _load_cpu(self, **kwargs):
        if not (hasattr(self, "device") and self.device):
            self.device = "cpu"
        self._load(**kwargs)

    def _load_gpu(self, **kwargs):
        if not (hasattr(self, "device") and self.device):
            self.device = "gpu"
        self._load(**kwargs)

    @task_handler
    def load(self, **kwargs):
        ...

    def predict(self, data):
        if not self.model_instances:
            raise RuntimeError(
                "Models have not been loaded. load method must be executed beforehand."
            )
        if "local" in self.model_instances:
            model = self.model_instances["local"]
        else:
            model = Worker.model
        data = self.preprocessing(data)
        output = self.inference(model, data)
        return self.postprocessing(output)

    def preprocessing(self, data):
        return data

    def postprocessing(self, data):
        return data
