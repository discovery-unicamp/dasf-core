#!/usr/bin/env python3

from dasf.transforms.transforms import _FitPredict
from dasf.transforms.transforms import _FitTransform
from dasf.transforms.transforms import _Predict
from dasf.transforms.transforms import _Transform
from dasf.transforms.transforms import _GetParams
from dasf.transforms.transforms import _SetParams


# We don't need to extend _FitLazy, _FitLocal because they are already
# extended by both _TransformLazy, _TransformLocal.
class ClusterClassifier(_FitPredict, _FitTransform,
                        _Predict, _Transform,
                        _GetParams, _SetParams):
    pass
