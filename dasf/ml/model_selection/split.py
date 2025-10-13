#!/usr/bin/env python3
"""This module contains a wrapper for the `train_test_split` function."""

from dask_ml.model_selection import train_test_split as train_test_split_mcpu
from sklearn.model_selection import train_test_split as train_test_split_cpu

from dasf.transforms import TargeteredTransform, Transform

try:
    import GPUtil
    if len(GPUtil.getGPUs()) == 0:  # check if GPU are available in current env
        raise ImportError("There is no GPU available here")

    from cuml.model_selection import train_test_split as train_test_split_gpu
except ImportError:
    pass


class train_test_split(TargeteredTransform, Transform):
    """A wrapper for the `train_test_split` function from scikit-learn,
    dask-ml, and cuml.

    Parameters
    ----------
    output : str, optional
        The output to return ('train' or 'test') (default is 'train').
    test_size : float, optional
        The proportion of the dataset to include in the test split
        (default is None).
    train_size : float, optional
        The proportion of the dataset to include in the train split
        (default is None).
    random_state : int, optional
        The seed used by the random number generator (default is None).
    shuffle : bool, optional
        Whether or not to shuffle the data before splitting
        (default is None).
    blockwise : bool, optional
        Whether to split the data blockwise (default is True).
    convert_mixed_types : bool, optional
        Whether to convert mixed-type columns to a single type
        (default is False).
    **kwargs
        Keyword arguments to pass to the `TargeteredTransform`.
    """
    def __init__(
        self,
        output="train",
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=None,
        blockwise=True,
        convert_mixed_types=False,
        **kwargs
    ):
        """Initialize the `train_test_split` transform."""
        TargeteredTransform.__init__(self, **kwargs)

        self.output = output
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle

        # Exclusive for Dask operations
        self.blockwise = blockwise

        self.convert_mixed_types = convert_mixed_types

    def _lazy_transform_cpu(self, X):
        """Lazily split the data using dask-ml.

        Parameters
        ----------
        X : tuple
            A tuple containing the data and labels.

        Returns
        -------
        tuple
            A tuple containing the training or test data and labels.
        """
        X, y = X
        X_train, X_test, y_train, y_test = train_test_split_mcpu(
            X,
            y,
            train_size=self.train_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
            blockwise=self.blockwise,
        )
        if self.output == "train":
            return X_train, y_train
        elif self.output == "test":
            return X_test, y_test

    def _lazy_transform_gpu(self, X):
        """Lazily split the data using dask-ml.

        Parameters
        ----------
        X : tuple
            A tuple containing the data and labels.

        Raises
        ------
        NotImplementedError
            This function is not implemented for Dask and CuML.
        """
        raise NotImplementedError(
            "Function train_test_split() is not implemented for Dask and CuML"
        )

    def _transform_cpu(self, X):
        """Split the data using scikit-learn.

        Parameters
        ----------
        X : tuple
            A tuple containing the data and labels.

        Returns
        -------
        tuple
            A tuple containing the training or test data and labels.
        """
        X, y = X
        X_train, X_test, y_train, y_test = train_test_split_cpu(
            X,
            y,
            train_size=self.train_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        if self.output == "train":
            return X_train, y_train
        elif self.output == "test":
            return X_test, y_test

    def _transform_gpu(self, X):
        """Split the data using cuml.

        Parameters
        ----------
        X : tuple
            A tuple containing the data and labels.

        Returns
        -------
        tuple
            A tuple containing the training or test data and labels.
        """
        X, y = X
        X_train, X_test, y_train, y_test = train_test_split_gpu(
            X,
            y,
            train_size=self.train_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        if self.output == "train":
            return X_train, y_train
        elif self.output == "test":
            return X_test, y_test
