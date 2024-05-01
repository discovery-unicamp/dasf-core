#!/usr/bin/env python3

""" Init module for SVM ML algorithms. """

from dasf.ml.svm.svm import SVC  # noqa
from dasf.ml.svm.svm import SVR  # noqa
from dasf.ml.svm.svm import LinearSVC  # noqa
from dasf.ml.svm.svm import LinearSVR  # noqa

__all__ = ["SVC", "SVR", "LinearSVC", "LinearSVR"]
