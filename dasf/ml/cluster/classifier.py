#!/usr/bin/env python3

from dasf.transforms.base import (
    Fit,
    FitPredict,
    FitTransform,
    GetParams,
    Predict,
    SetParams,
    TargeteredTransform,
    Transform,
)


class ClusterClassifier(
    Fit, FitPredict, FitTransform, Predict,
    GetParams, SetParams, TargeteredTransform
):
    def __init__(self, **kwargs):
        TargeteredTransform.__init__(self, **kwargs)
