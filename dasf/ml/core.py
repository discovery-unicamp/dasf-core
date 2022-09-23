#!/usr/bin/env python3

import os
import pickle

from pathlib import Path


class MLGeneric:
    def __init__(self, name, checkpoint=False, **kwargs):
        # Machine Learning Algorithm
        self._cached_dir = os.path.abspath(str(Path.home()) + "/.cache/dasf/ml/")
        os.makedirs(self._cached_dir, exist_ok=True)

        self._tmp = os.path.abspath(self._cached_dir + "/" + name.lower())

        self.__checkpoint = checkpoint

    def dump(self, model):
        if self.get_checkpoint():
            with open(self._tmp, "wb") as fh:
                pickle.dump(model, fh)

    def load(self, model):
        if self.get_checkpoint() and os.path.exists(self._tmp):
            with open(self._tmp, "rb") as fh:
                return pickle.load(fh)
        return model
