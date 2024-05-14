import os

from dasf.utils.decorators import task_handler

import lightning as L
import numpy as np
from dasf.utils.decorators import task_handler
import lightning as L
from torch.utils.data import DataLoader
from dasf.utils.types import is_gpu_array, is_dask_array
from typing import Any, Tuple, Union


class LazyDatasetComputer:
    """This class encapsulates a map-style dataset that returns a Dask or GPU
    array. The __getitem__ method will compute the dask array before returning
    it. Thus, we can wrap this class into a DataLoader to make it compatible
    with PyTorch Lightning training loop.
    """

    def __init__(self, dataset: Any, unsqueeze_dim: int = None):
        """Maps a dataset to a LazyDatasetComputer object.

        Parameters
        ----------
        dataset : Any
            A Dasf map-style like dataset. The __getitem__ method should return
            either a tuple or a single object, in CPU/GPU or Dask array format.
        unsqueeze_dim : int, optional
            The dimension to be unsqueezed in the output, by default None
        """
        self.dataset = dataset
        self.unsqueeze_dim = unsqueeze_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Compute the dask array and return it.

        Parameters
        ----------
        index : int
            The index of the dataset to be returned.

        Returns
        -------
        _type_
            np.ndarray or tuple of np.ndarray
        """
        res = self.dataset[index]
        if isinstance(res, tuple):
            if is_dask_array(res[0]):
                res = tuple(map(lambda x: x.compute(), res))
            if is_gpu_array(res[0]):
                res = tuple(map(lambda x: x.get(), res))

            if self.unsqueeze_dim is not None:
                res = tuple(
                    map(
                        lambda x: np.expand_dims(x, axis=self.unsqueeze_dim),
                        res,
                    )
                )

        else:
            if is_dask_array(res):
                res = res.compute()
            if is_gpu_array(res):
                res = res.get()
            if self.unsqueeze_dim is not None:
                res = np.expand_dims(res, axis=self.unsqueeze_dim)

        return res


class LightningTrainer:
    def __init__(
        self,
        model: L.LightningModule,
        use_gpu: bool = False,
        batch_size: int = 1,
        max_epochs: int = 1,
        limit_train_batches: int = None,
        limit_val_batches: int = None,
        devices: int = "auto",
        num_nodes: int = 1,
        shuffle: bool = True,
        strategy: str = "ddp",
        unsqueeze_dim: int = None,
    ):
        """
        Initialize the LightningFit class.

        Parameters
        ----------
        model : LightningModule
            The LightningModule instance representing the model to be trained.
        use_gpu : bool, optional
            Flag indicating whether to use GPU for training, by default False.
        batch_size : int, optional
            The batch size for training, by default 1.
        max_epochs : int, optional
            The maximum number of epochs for training, by default 1.
        limit_train_batches : int, optional
            The number of batches to consider for training, by default None.
        limit_val_batches : int, optional
            The number of batches to consider for validation, by default None.
        devices : int, optional
            The number of devices to use for training, by default "auto".
        num_nodes : int, optional
            The number of nodes to use for distributed training, by default 1.
        shuffle : bool, optional
            Flag indicating whether to shuffle the data during training, by default True.
        strategy : str, optional
            The strategy to use for distributed training, by default "ddp".
        unsqueeze_dim : int, optional
            The dimension to unsqueeze the input data, by default None.
        """
        self.model = model
        self.accelerator = "gpu" if use_gpu else "cpu"
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.devices = devices
        self.num_nodes = num_nodes
        self.shuffle = shuffle
        self.strategy = strategy
        self.unsqueeze_dim = unsqueeze_dim

    @task_handler
    def fit(self, train_data: Any, val_data: Any = None): 
        """Perform the training of the model using torch Lightning.

        Parameters
        ----------
        train_data : Any
            A dasf map-style like dataset containing the training data.
        val_data : Any, optional
            A dasf map-style like dataset containing the validation data.
        """
        ...

    def _fit(self, train_data, val_data=None):
        train_data = LazyDatasetComputer(
            train_data, unsqueeze_dim=self.unsqueeze_dim
        )
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=self.shuffle
        )

        val_loader = None
        if val_data is not None:
            val_data = LazyDatasetComputer(val_data)
            val_loader = DataLoader(
                val_data, batch_size=self.batch_size, shuffle=True
            )

        # As workers deals with spawn method, the pipeline will be re-executed
        # until here in each worker. This may leads to multiple tasks
        self.trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            num_nodes=self.num_nodes,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            strategy=self.strategy,
        )
        self.trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        return self.model

    def _lazy_fit_cpu(self, train_data, val_data=None):
        self._fit(train_data, val_data)

    def _lazy_fit_gpu(self, train_data, val_data=None):
        self._fit(train_data, val_data)

    def _fit_cpu(self, train_data, val_data=None):
        self._fit(train_data, val_data)

    def _fit_gpu(self, train_data, val_data=None):
        self._fit(train_data, val_data)
