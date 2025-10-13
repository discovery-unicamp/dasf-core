#!/usr/bin/env python3
"""This module contains wrappers and utilities for PyTorch Lightning."""

import pytorch_lightning as pl
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
    """A PyTorch Lightning data module for handling datasets.

    This class wraps datasets for training, validation, and testing,
    and handles data preparation, setup, and dataloader creation.

    Parameters
    ----------
    train : torch.utils.data.Dataset
        The training dataset.
    val : torch.utils.data.Dataset, optional
        The validation dataset (default is None).
    test : torch.utils.data.Dataset, optional
        The test dataset (default is None).
    batch_size : int, optional
        The batch size for the dataloaders (default is 64).

    """
    def __init__(self, train, val=None, test=None, batch_size=64):
        """Initialize the `TorchDataLoader`.
        """
        super().__init__()

        self._train = train
        self._val = val
        self._test = test

        self._batch_size = batch_size

    def prepare_data(self):
        """Download data if needed.

        This method downloads the data for the training, validation, and
        test datasets if a `download` method is available.
        """
        if self._train is not None and hasattr(self._train, "download"):
            self._train.download()

        if self._val is not None and hasattr(self._val, "download"):
            self._val.download()

        if self._test is not None and hasattr(self._test, "download"):
            self._test.download()

    def setup(self, stage=None):
        """Load data for the appropriate stage.

        This method loads the data for the training, validation, and
        test datasets if a `load` method is available.

        Parameters
        ----------
        stage : str, optional
            The stage to setup ('fit', 'validate', 'test', or 'predict').
            (default is None).
        """
        if self._train is not None and hasattr(self._train, "load"):
            self._train.load()

        if self._val is not None and hasattr(self._val, "load"):
            self._val.load()

        if self._test is not None and hasattr(self._test, "load"):
            self._test.load()

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(self._train, batch_size=self._batch_size)

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(self._val, batch_size=self._batch_size)

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(self._test, batch_size=self._batch_size)


def run_dask_clustered(func, client=None, **kwargs):
    """Run a function on all workers in a Dask cluster.

    Parameters
    ----------
    func : callable
        The function to run on each worker.
    client : dask.distributed.Client, optional
        The Dask client to use (default is None).
    **kwargs
        Keyword arguments to pass to the function.
    """
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
    """Fit a PyTorch Lightning model.

    Parameters
    ----------
    model : pytorch_lightning.LightningModule
        The model to fit.
    X : array-like
        The training data.
    y : array-like
        The training labels.
    max_iter : int
        The maximum number of epochs.
    accel : str
        The accelerator to use ('cpu', 'gpu', 'tpu', 'ipu', 'auto').
    strategy : str
        The strategy to use for distributed training.
    devices : int
        The number of devices to use.
    ngpus : int
        The number of GPUs to use.
    batch_size : int, optional
        The batch size (default is 32).
    plugins : list, optional
        A list of plugins to use (default is None).
    meta : dict, optional
        Worker metadata (default is None).
    """

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
    """A wrapper for a PyTorch Lightning model that implements the `Fit`
    interface.

    Parameters
    ----------
    model : pytorch_lightning.LightningModule
        The model to wrap.
    max_iter : int, optional
        The maximum number of epochs (default is 100).
    batch_size : int, optional
        The batch size (default is 32).
    """
    def __init__(self, model, max_iter=100, batch_size=32):
        """Initialize the `NeuralNetClassifier`.
        """
        self._model = model

        self._accel = None
        self._strategy = None
        self._max_iter = max_iter
        self._devices = 0
        self._ngpus = 0
        self._batch_size = batch_size

        self.__trainer = False

    def _lazy_fit_generic(self, X, y, accel, ngpus):
        """Lazily fit the model on a Dask cluster.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like
            The training labels.
        accel : str
            The accelerator to use.
        ngpus : int
            The number of GPUs to use.
        """
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
        """Lazily fit the model on a Dask cluster using GPUs.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like, optional
            The training labels (default is None).
        """
        self._lazy_fit_generic(X=X, y=y, accel="gpu", ngpus=get_dask_gpu_count())

    def _lazy_fit_cpu(self, X, y=None):
        """Lazily fit the model on a Dask cluster using CPUs.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like, optional
            The training labels (default is None).
        """
        self._lazy_fit_generic(X=X, y=y, accel="cpu", ngpus=get_dask_gpu_count())

    def __fit_generic(self, X, y, accel, ngpus):
        """Fit the model.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like
            The training labels.
        accel : str
            The accelerator to use.
        ngpus : int
            The number of GPUs to use.
        """
        self._accel = accel
        self._strategy = "dp"
        self._ngpus = self._ndevices = ngpus

        dataloader = TorchDataLoader(train=X, val=y, batch_size=self._batch_size)

        self.__trainer = pl.Trainer(
            max_epochs=self._max_iter, accelerator=accel, devices=ngpus
        )

        self.__trainer.fit(self._model, datamodule=dataloader)

    def _fit_gpu(self, X, y=None):
        """Fit the model using GPUs.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like, optional
            The training labels (default is None).
        """
        self.__fit_generic(X, y, "gpu", get_gpu_count())

    def _fit_cpu(self, X, y=None):
        """Fit the model using CPUs.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like, optional
            The training labels (default is None).
        """
        self.__fit_generic(X, y, "cpu", 0)
