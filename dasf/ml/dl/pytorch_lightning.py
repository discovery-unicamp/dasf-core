#!/usr/bin/env python3

import uuid

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from dask_pytorch_ddp.results import DaskResultsHandler

from dasf.utils import utils
from dasf.pipeline import Operator
from dasf.pipeline.types import TaskExecutorType
from dasf.ml.dl.clusters import DaskClusterEnvironment
from dasf.utils.utils import get_gpu_count
from dasf.utils.utils import get_dask_gpu_count
from dasf.utils.utils import get_worker_info
from dasf.utils.utils import get_dask_running_client


class TorchDataLoader(pl.LightningDataModule):
    def __init__(self, train, val=None, test=None, batch_size=64):
        super().__init__()

        self._train = train
        self._val = val
        self._test = test

        self._batch_size = batch_size

    def prepare_data(self):
        if self._train and hasattr(self._train, 'download') and self._train._download:
            self._train.download()

        if self._val and hasattr(self._val, 'download') and self._val._download:
            self._val.download()

        if self._test and hasattr(self._test, 'download') and self._test._download:
            self._test.download()

    def train_dataloader(self):
        return DataLoader(self._train.load())

    def val_dataloader(self):
        return DataLoader(self._val.load())

    def test_dataloader(self):
        return DataLoader(self._test.load())


def run_dask_clustered(func, client=None, **kwargs):
    if client is None:
        client = get_dask_running_client()

    all_workers = get_worker_info(client)

    for worker in all_workers:
        futures = client.submit(func, **kwargs, workers=[worker["worker"]])

    utils.sync_future_loop(futures)


def fit(model, X, y, max_iter, accel, strategy, devices, ngpus, batch_size,
        plugins=None):

    # Variable world_size is based on the number of Dask workers
    if plugins is not None and isinstance(plugins, list):
        nodes = 1
        for plugin in plugins:
            if isinstance(plugin, DaskClusterEnvironment):
                nodes = plugin.world_size()
                break
    else:
        nodes = 1

    # Use it for heterogeneous workers.
    if ngpus < 0:
        ngpus = -1

    dataloader = TorchDataLoader(train=X, val=y)

    trainer = pl.Trainer(max_epochs=max_iter, accelerator=accel,
                         strategy=strategy, gpus=ngpus, plugins=plugins,
                         devices=devices, num_nodes=nodes)

    trainer.fit(model, dataloader)


class NeuralNetClassifier:
    def __init__(self, model, max_iter=100):
        self._model = model

        self._accel = None
        self._strategy = None
        self._max_iter = max_iter
        self._devices = 0
        self._ngpus = 0

        self.__trainer = False
        self.__handler = DaskResultsHandler(uuid.uuid4().hex)

    def _lazy_fit_generic(self, X, y, accel, ngpus):
        self._accel = accel
        self._strategy = "ddp"
        self._ngpus = self._ndevices = ngpus

        plugins = [DaskClusterEnvironment()]

        run_dask_clustered(fit, model=self._model, X=X, y=y,
                           max_iter=self._max_iter, accel=self._accel,
                           strategy=self._strategy, ndevices=self._ndevices,
                           ngpus=self._ngpus, plugins=plugins)

    def _lazy_fit_gpu(self, X, y=None):
        self._lazy_fit_generic(X=X, y=y, accel="gpu",
                               ngpus=len(get_dask_gpu_count()))

    def _lazy_fit_cpu(self, X, y=None):
        self._lazy_fit_generic(X=X, y=y, accel="cpu",
                               ngpus=len(get_dask_gpu_count()))

    def __fit_generic(self, X, y, accel, ngpus):
        self._accel = accel
        self._strategy = "dp"
        self._ngpus = self._ndevices = ngpus

        dataloader = TorchDataLoader(train=X, val=y)

        self.__trainer = pl.Trainer(max_epochs=self._max_iter,
                                    accelerator=accel,
                                    gpus=ngpus)

        self.__trainer.fit(self._model, dataloader)

    def _fit_gpu(self, X, y=None):
        self.__fit_generic(X, y, "gpu", len(get_gpu_count()))

    def _fit_cpu(self, X, y=None):
        self.__fit_generic(X, y, "cpu", 0)


class Trainer(Operator):
    def __init__(self, name="PyTorch Lightning Pipeline", num_epochs=100, batch_size=16):
        super().__init__(name)

        self.accel = None
        self.strategy = None
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.devices = 0
        self.ngpus = 0

        self.batch_size_auto = None
        if self.batch_size < 0:
            self.batch_size_auto = 'binsearch'

    def setup(self, executor):
        self.dtype = executor.dtype

        if hasattr(executor, "client"):
            self.client = executor.client

        if self.dtype == TaskExecutorType.single_cpu:
            self.accel = "cpu"
            self.strategy = "dp"
            self.ngpus = 0
            self.devices = 1
        elif self.dtype == TaskExecutorType.single_gpu:
            self.accel = "gpu"
            self.strategy = "dp"
            self.ngpus = executor.ngpus
            self.devices = self.ngpus
        elif self.dtype == TaskExecutorType.multi_cpu:
            self.accel = "cpu"
            self.strategy = "ddp"
            self.ngpus = 0
            self.devices = 1
        elif self.dtype == TaskExecutorType.multi_gpu:
            self.accel = "gpu"
            self.strategy = "ddp"
            self.ngpus = executor.ngpus
            self.devices = self.ngpus

    def __run_clustered(self, model, train, val, batch_size, auto_scale_batch_size=None):
        all_workers = utils.get_worker_info(self.client)

        for worker in all_workers:
            sct = self.client.scatter(train, broadcast=True)
            scv = self.client.scatter(val, broadcast=True)
            futures = self.client.submit(fit, model, self.epochs, self.accel, self.strategy,
                                         self.devices, sct, scv, worker, batch_size,
                                         auto_scale_batch_size, self.ngpus, workers=[worker["worker"]])

        utils.sync_future_loop(futures)

    def run(self, model, train, val=None):
        print(len(train), len(val))
        if utils.is_executor_single(self.dtype):
            trainer = pl.Trainer(max_epochs=self.epochs, accelerator=self.accel, gpus=self.ngpus, auto_scale_batch_size=self.batch_size_auto)
            trainer.fit(model.model, DataLoader(train, batch_size=self.batch_size), DataLoader(val, batch_size=self.batch_size))
        else:
            self.__run_clustered(model.model, train, val, batch_size=self.batch_size, auto_scale_batch_size=self.batch_size_auto)
