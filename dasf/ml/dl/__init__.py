#!/usr/bin/env python3

""" Init module for Deep Learning algorithms. """

from dasf.ml.dl.lightning_fit import LightningTrainer
from dasf.ml.dl.pytorch_lightning import NeuralNetClassifier

__all__ = ["NeuralNetClassifier", "LightningTrainer"]
