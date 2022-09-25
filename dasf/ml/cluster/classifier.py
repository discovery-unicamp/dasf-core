#!/usr/bin/env python3

from dasf.transforms.base import Fit
from dasf.transforms.base import FitPredict
from dasf.transforms.base import FitTransform
from dasf.transforms.base import Predict
from dasf.transforms.base import Transform
from dasf.transforms.base import GetParams
from dasf.transforms.base import SetParams


class ClusterClassifier(
    Fit, FitPredict, FitTransform, Predict,
    Transform, GetParams, SetParams
):
    pass
