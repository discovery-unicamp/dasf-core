#!/usr/bin/env python3

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from dasf.utils import utils
from dasf.pipeline import Operator
from dasf.pipeline.types import TaskExecutorType
from dasf.ml.dl.clusters import DaskClusterEnvironment


def trainer(
    model,
    epochs,
    accel,
    strategy,
    devices,
    train,
    val,
    metadata,
    batch_size,
    auto_scale_batch_size,
    ngpus,
):
    print(metadata, devices, auto_scale_batch_size, strategy, accel, batch_size)

    # Variable world_size is based on the number of Dask workers
    nodes = metadata["world_size"]

    val_loader = None
    if val:
        val_loader = DataLoader(val, batch_size=batch_size)

    if ngpus > 0:
        ngpus = -1

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accel,
        strategy=strategy,
        gpus=ngpus,
        plugins=[DaskClusterEnvironment(metadata)],
        devices=devices,
        num_nodes=nodes,
        auto_scale_batch_size=auto_scale_batch_size,
    )
    trainer.fit(model, DataLoader(train, batch_size=batch_size), val_loader)


class Trainer(Operator):
    def __init__(
        self, name="PyTorch Lightning Pipeline", num_epochs=100, batch_size=16
    ):
        super().__init__(name)

        self.accel = None
        self.strategy = None
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.devices = 0
        self.ngpus = 0

        self.batch_size_auto = None
        if self.batch_size < 0:
            self.batch_size_auto = "binsearch"

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

    def __run_clustered(
        self, model, train, val, batch_size, auto_scale_batch_size=None
    ):
        all_workers = utils.get_worker_info(self.client)

        for worker in all_workers:
            sct = self.client.scatter(train, broadcast=True)
            scv = self.client.scatter(val, broadcast=True)
            futures = self.client.submit(
                trainer,
                model,
                self.epochs,
                self.accel,
                self.strategy,
                self.devices,
                sct,
                scv,
                worker,
                batch_size,
                auto_scale_batch_size,
                self.ngpus,
                workers=[worker["worker"]],
            )

        utils.sync_future_loop(futures)

    def run(self, model, train, val=None):
        print(len(train), len(val))
        if utils.is_executor_single(self.dtype):
            trainer = pl.Trainer(
                max_epochs=self.epochs,
                accelerator=self.accel,
                gpus=self.ngpus,
                auto_scale_batch_size=self.batch_size_auto,
            )
            trainer.fit(
                model.model,
                DataLoader(train, batch_size=self.batch_size),
                DataLoader(val, batch_size=self.batch_size),
            )
        else:
            self.__run_clustered(
                model.model,
                train,
                val,
                batch_size=self.batch_size,
                auto_scale_batch_size=self.batch_size_auto,
            )
