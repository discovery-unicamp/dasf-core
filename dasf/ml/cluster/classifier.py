#!/usr/bin/env python3

from dasf.transforms.transforms import _Fit
from dasf.transforms.transforms import _FitPredict
from dasf.transforms.transforms import _FitTransform
from dasf.transforms.transforms import _Predict
from dasf.transforms.transforms import _Transform
from dasf.transforms.transforms import _GetParams
from dasf.transforms.transforms import _SetParams


class ClusterClassifier(
    _Fit, _FitPredict, _FitTransform, _Predict,
    _Transform, _GetParams, _SetParams
):
    pass
