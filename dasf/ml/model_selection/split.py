#!/usr/bin/env python3

from dask_ml.model_selection import train_test_split as train_test_split_mcpu
from sklearn.model_selection import train_test_split as train_test_split_cpu

from dasf.transforms import TargeteredTransform, Transform

try:
    import GPUtil
    assert len(GPUtil.getGPUs()) != 0 # check if GPU are available in current env

    from cuml.model_selection import train_test_split as train_test_split_gpu
except ImportError:
    pass


class train_test_split(TargeteredTransform, Transform):
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
        raise NotImplementedError(
            "Function train_test_split() is not implemented for Dask and CuML"
        )

    def _transform_cpu(self, X):
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
