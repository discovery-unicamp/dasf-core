#!/usr/bin/env python3
""" Generic Mixture Classifier algorithm method. """

from dasf.transforms.base import Fit, FitPredict, FitTransform, GetParams, SetParams


class MixtureClassifier(
    Fit, FitPredict, FitTransform,
    GetParams, SetParams
):
    """
    Generic Mixture Classifier based on regular method for this type of
    classifier

    Most of the Mixture methods has a fit, a fit_predict, a fit_transform,
    a get_params, and a set_params methods.

    """
    ...
