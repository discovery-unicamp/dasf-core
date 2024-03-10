#!/usr/bin/env python3

import uuid

import pytorch_lightning as pl
from dask_pytorch_ddp.results import DaskResultsHandler
from torch.utils.data import DataLoader

from dasf.ml.dl.clusters import DaskClusterEnvironment
from dasf.transforms.base import Fit
from dasf.utils.funcs import (
    get_dask_gpu_count,
    get_dask_running_client,
    get_gpu_count,
    get_worker_info,
    sync_future_loop,
)


class TorchDataLoader(pl.LightningDataModule):
    def __init__(self, train, val=None, test=None, batch_size=64):
        super().__init__()

        self._train = train
        self._val = val
        self._test = test

        self._batch_size = batch_size

    def prepare_data(self):
        if self._train is not None and hasattr(self._train, "download"):
            self._train.download()

        if self._val is not None and hasattr(self._val, "download"):
            self._val.download()

        if self._test is not None and hasattr(self._test, "download"):
            self._test.download()

    def setup(self, stage=None):
        if self._train is not None and hasattr(self._train, "load"):
            self._train.load()

        if self._val is not None and hasattr(self._val, "load"):
            self._val.load()

        if self._test is not None and hasattr(self._test, "load"):
            self._test.load()

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self._batch_size)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=self._batch_size)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self._batch_size)


def run_dask_clustered(func, client=None, **kwargs):
    if client is None:
        client = get_dask_running_client()

    all_workers = get_worker_info(client)

    for worker in all_workers:
        # Including worker metadata into kwargs
        kwargs['meta'] = worker

        futures = client.submit(func, **kwargs, workers=[worker["worker"]])

    sync_future_loop(futures)


def fit(model, X, y, max_iter, accel, strategy, devices, ngpus, batch_size=32,
        plugins=None, meta=None):

    if meta is None:
        plugin = DaskClusterEnvironment(metadata=meta)

        nodes = plugin.world_size()

        if plugins is None:
            plugins = list()

        plugins.append(plugin)
    else:
        nodes = 1

    # Use it for heterogeneous workers.
    if ngpus < 0:
        ngpus = -1

    dataloader = TorchDataLoader(train=X, val=y, batch_size=batch_size)

    trainer = pl.Trainer(
        max_epochs=max_iter,
        accelerator=accel,
        strategy=strategy,
        gpus=ngpus,
        plugins=plugins,
        devices=devices,
        num_nodes=nodes,
    )

    trainer.fit(model, datamodule=dataloader)


class NeuralNetClassifier(Fit):
    def __init__(self, model, max_iter=100, batch_size=32):
        self._model = model

        self._accel = None
        self._strategy = None
        self._max_iter = max_iter
        self._devices = 0
        self._ngpus = 0
        self._batch_size = batch_size

        self.__trainer = False
        self.__handler = DaskResultsHandler(uuid.uuid4().hex)

    def _lazy_fit_generic(self, X, y, accel, ngpus):
        self._accel = accel
        self._strategy = "ddp"
        self._ngpus = self._ndevices = ngpus

        plugins = [DaskClusterEnvironment()]

        run_dask_clustered(
            fit,
            model=self._model,
            X=X,
            y=y,
            max_iter=self._max_iter,
            accel=self._accel,
            strategy=self._strategy,
            devices=self._ndevices,
            ngpus=self._ngpus,
            batch_size=self._batch_size,
            plugins=plugins,
        )

    def _lazy_fit_gpu(self, X, y=None):
        self._lazy_fit_generic(X=X, y=y, accel="gpu", ngpus=get_dask_gpu_count())

    def _lazy_fit_cpu(self, X, y=None):
        self._lazy_fit_generic(X=X, y=y, accel="cpu", ngpus=get_dask_gpu_count())

    def __fit_generic(self, X, y, accel, ngpus):
        self._accel = accel
        self._strategy = "dp"
        self._ngpus = self._ndevices = ngpus

        dataloader = TorchDataLoader(train=X, val=y, batch_size=self._batch_size)

        self.__trainer = pl.Trainer(
            max_epochs=self._max_iter, accelerator=accel, devices=ngpus
        )

        self.__trainer.fit(self._model, datamodule=dataloader)

    def _fit_gpu(self, X, y=None):
        self.__fit_generic(X, y, "gpu", get_gpu_count())

    def _fit_cpu(self, X, y=None):
        self.__fit_generic(X, y, "cpu", 0)
