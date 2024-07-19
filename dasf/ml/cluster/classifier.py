""" Generic Clustering Classifier algorithm method. """
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
    """
    Generic Clustering Classifier based on regular method for this type of
    classifier

    Most of the Clustering method has a fit, a fit_predict, a fit_transform,
    a predict, a get_params, and a set_params methods.

    """
    def __init__(self, **kwargs):
        """ A generic constructor method. """
        TargeteredTransform.__init__(self, **kwargs)
