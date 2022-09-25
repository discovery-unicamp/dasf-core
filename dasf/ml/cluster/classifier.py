#!/usr/bin/env python3

from dasf.transforms import Fit
from dasf.transforms import FitPredict
from dasf.transforms import FitTransform
from dasf.transforms import Predict
from dasf.transforms import Transform
from dasf.transforms import GetParams
from dasf.transforms import SetParams


class ClusterClassifier(
    Fit, FitPredict, FitTransform, Predict,
    Transform, GetParams, SetParams
):
    pass
