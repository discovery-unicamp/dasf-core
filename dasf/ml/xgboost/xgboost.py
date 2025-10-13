#!/usr/bin/env python3
"""This module contains a wrapper for the XGBoost Regressor."""

import GPUtil
import xgboost as xgb

from dasf.transforms import Fit, FitPredict, Predict
from dasf.utils.funcs import is_gpu_supported


class XGBRegressor(Fit, FitPredict, Predict):
    """A wrapper for the XGBoost Regressor that implements the `Fit`,
    `FitPredict`, and `Predict` interfaces.

    Parameters
    ----------
    max_depth : int, optional
        Maximum tree depth for base learners (default is None).
    max_leaves : int, optional
        Maximum number of leaves; 0 indicates no limit (default is None).
    max_bin : int, optional
        If using histogram-based algorithm, maximum number of bins per feature
        (default is None).
    grow_policy : str, optional
        Tree growing policy. 'depthwise' follows splitting rules closer to
        lightgbm, 'lossguide' chooses splits with the highest loss change
        (default is None).
    learning_rate : float, optional
        Boosting learning rate (xgb's 'eta') (default is None).
    n_estimators : int, optional
        Number of boosting rounds (default is 100).
    verbosity : int, optional
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug)
        (default is None).
    objective : str, optional
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (default is None).
    booster : str, optional
        Specify which booster to use: gbtree, gblinear or dart
        (default is None).
    tree_method : str, optional
        Specify which tree method to use
        (default is None).
    n_jobs : int, optional
        Number of parallel threads used to run xgboost (default is None).
    gamma : float, optional
        Minimum loss reduction required to make a further partition on a leaf
        node of the tree (default is None).
    min_child_weight : float, optional
        Minimum sum of instance weight needed in a child (default is None).
    max_delta_step : float, optional
        Maximum delta step we allow each tree's weight estimation to be
        (default is None).
    subsample : float, optional
        Subsample ratio of the training instance (default is None).
    sampling_method : str, optional
        Sampling method. 'uniform' creates new trees using all training data,
        'gradient_based' uses gradient-based sampling
        (default is None).
    colsample_bytree : float, optional
        Subsample ratio of columns when constructing each tree
        (default is None).
    colsample_bylevel : float, optional
        Subsample ratio of columns for each level (default is None).
    colsample_bynode : float, optional
        Subsample ratio of columns for each node (split) (default is None).
    reg_alpha : float, optional
        L1 regularization term on weights (xgb's alpha) (default is None).
    reg_lambda : float, optional
        L2 regularization term on weights (xgb's lambda) (default is None).
    scale_pos_weight : float, optional
        Balancing of positive and negative weights (default is None).
    base_score : float, optional
        The initial prediction score of all instances, global bias
        (default is None).
    random_state : int, optional
        Random number seed (default is None).
    num_parallel_tree : int, optional
        Used for boosting random forest (default is None).
    monotone_constraints : str, optional
        Constraint of variable monotonicity. See tutorial for more information
        (default is None).
    interaction_constraints : str, optional
        Constraints for interaction representing permitted interactions
        (default is None).
    importance_type : str, optional
        The feature importance type for the feature_importances_ property:
        'gain', 'weight', 'cover', 'total_gain' or 'total_cover'
        (default is None).
    gpu_id : int, optional
        Device ordinal (default is None).
    validate_parameters : bool, optional
        Give warnings for unknown parameter (default is None).
    predictor : str, optional
        The type of predictor algorithm to use. Provides the same results but
        allows the use of GPU or CPU (default is None).
    enable_categorical : bool, optional
        Experimental support for categorical features (default is False).
    max_cat_to_onehot : int, optional
        If categorical features are enabled, this specifies the maximum number
        of categories to use one-hot encoding for (default is None).
    eval_metric : str, optional
        Metric used for evaluation (default is None).
    early_stopping_rounds : int, optional
        Activates early stopping. Validation metric needs to improve at least
        once in every early_stopping_rounds round(s) to continue training
        (default is None).
    callbacks : list, optional
        List of callback functions that are applied at end of each iteration
        (default is None).
    **kwargs
        Other keyword arguments.
    """
    def __init__(
        self,
        max_depth=None,
        max_leaves=None,
        max_bin=None,
        grow_policy=None,
        learning_rate=None,
        n_estimators=100,
        verbosity=None,
        objective=None,
        booster=None,
        tree_method=None,
        n_jobs=None,
        gamma=None,
        min_child_weight=None,
        max_delta_step=None,
        subsample=None,
        sampling_method=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        reg_alpha=None,
        reg_lambda=None,
        scale_pos_weight=None,
        base_score=None,
        random_state=None,
        num_parallel_tree=None,
        monotone_constraints=None,
        interaction_constraints=None,
        importance_type=None,
        gpu_id=None,
        validate_parameters=None,
        predictor=None,
        enable_categorical=False,
        max_cat_to_onehot=None,
        eval_metric=None,
        early_stopping_rounds=None,
        callbacks=None,
        **kwargs
    ):

        self.__xgb_cpu = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaves=max_leaves,
            max_bin=max_bin,
            grow_policy=grow_policy,
            learning_rate=learning_rate,
            verbosity=verbosity,
            objective=objective,
            booster=booster,
            tree_method=tree_method,
            n_jobs=n_jobs,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            sampling_method=sampling_method,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            random_state=random_state,
        )

        self.__xgb_mcpu = xgb.dask.DaskXGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaves=max_leaves,
            max_bin=max_bin,
            grow_policy=grow_policy,
            learning_rate=learning_rate,
            verbosity=verbosity,
            objective=objective,
            booster=booster,
            tree_method=tree_method,
            n_jobs=n_jobs,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            sampling_method=sampling_method,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            random_state=random_state,
        )

        if is_gpu_supported():
            if gpu_id is None:
                gpus = GPUtil.getGPUs()
                if len(gpus) > 0:
                    gpu_id = gpus[0].id

            self.__xgb_gpu = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_leaves=max_leaves,
                max_bin=max_bin,
                grow_policy=grow_policy,
                learning_rate=learning_rate,
                verbosity=verbosity,
                objective=objective,
                booster=booster,
                tree_method='gpu_hist',
                n_jobs=n_jobs,
                gamma=gamma,
                min_child_weight=min_child_weight,
                max_delta_step=max_delta_step,
                subsample=subsample,
                sampling_method=sampling_method,
                colsample_bytree=colsample_bytree,
                colsample_bylevel=colsample_bylevel,
                colsample_bynode=colsample_bynode,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                scale_pos_weight=scale_pos_weight,
                base_score=base_score,
                random_state=random_state,
                gpu_id=gpu_id
            )

            self.__xgb_mgpu = xgb.dask.DaskXGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_leaves=max_leaves,
                max_bin=max_bin,
                grow_policy=grow_policy,
                learning_rate=learning_rate,
                verbosity=verbosity,
                objective=objective,
                booster=booster,
                tree_method='gpu_hist',
                n_jobs=n_jobs,
                gamma=gamma,
                min_child_weight=min_child_weight,
                max_delta_step=max_delta_step,
                subsample=subsample,
                sampling_method=sampling_method,
                colsample_bytree=colsample_bytree,
                colsample_bylevel=colsample_bylevel,
                colsample_bynode=colsample_bynode,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                scale_pos_weight=scale_pos_weight,
                base_score=base_score,
                random_state=random_state,
            )

    def _lazy_fit_cpu(self, X, y=None, sample_weight=None, *args, **kwargs):
        """Lazily fit the model using dask-xgboost.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like, optional
            The training labels (default is None).
        sample_weight : array-like, optional
            The sample weights (default is None).
        *args
            Positional arguments to pass to the `fit` method.
        **kwargs
            Keyword arguments to pass to the `fit` method.

        Returns
        -------
        dask_xgboost.XGBRegressor
            The fitted model.
        """
        return self.__xgb_mcpu.fit(X=X, y=y, *args, **kwargs)

    def _lazy_fit_gpu(self, X, y=None, sample_weight=None, *args, **kwargs):
        """Lazily fit the model using dask-xgboost.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like, optional
            The training labels (default is None).
        sample_weight : array-like, optional
            The sample weights (default is None).
        *args
            Positional arguments to pass to the `fit` method.
        **kwargs
            Keyword arguments to pass to the `fit` method.

        Returns
        -------
        dask_xgboost.XGBRegressor
            The fitted model.
        """
        return self.__xgb_mgpu.fit(X=X, y=y, *args, **kwargs)

    def _fit_cpu(self, X, y=None, sample_weight=None):
        """Fit the model using xgboost.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like, optional
            The training labels (default is None).
        sample_weight : array-like, optional
            The sample weights (default is None).

        Returns
        -------
        xgboost.XGBRegressor
            The fitted model.
        """
        return self.__xgb_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None, sample_weight=None, *args, **kwargs):
        """Fit the model using xgboost.

        Parameters
        ----------
        X : array-like
            The training data.
        y : array-like, optional
            The training labels (default is None).
        sample_weight : array-like, optional
            The sample weights (default is None).
        *args
            Positional arguments to pass to the `fit` method.
        **kwargs
            Keyword arguments to pass to the `fit` method.

        Returns
        -------
        xgboost.XGBRegressor
            The fitted model.
        """
        return self.__xgb_gpu.fit(X=X, y=y, *args, **kwargs)

    def _lazy_predict_cpu(self, X, sample_weight=None, **kwargs):
        """Lazily predict using dask-xgboost.

        Parameters
        ----------
        X : array-like
            The data to predict.
        sample_weight : array-like, optional
            The sample weights (default is None).
        **kwargs
            Keyword arguments to pass to the `predict` method.

        Returns
        -------
        dask.array.Array
            The predicted values.
        """
        return self.__xgb_mcpu.predict(X=X, **kwargs)

    def _lazy_predict_gpu(self, X, sample_weight=None, **kwargs):
        """Lazily predict using dask-xgboost.

        Parameters
        ----------
        X : array-like
            The data to predict.
        sample_weight : array-like, optional
            The sample weights (default is None).
        **kwargs
            Keyword arguments to pass to the `predict` method.

        Returns
        -------
        dask.array.Array
            The predicted values.
        """
        return self.__xgb_mgpu.predict(X=X, **kwargs)

    def _predict_cpu(self, X, sample_weight=None, **kwargs):
        """Predict using xgboost.

        Parameters
        ----------
        X : array-like
            The data to predict.
        sample_weight : array-like, optional
            The sample weights (default is None).
        **kwargs
            Keyword arguments to pass to the `predict` method.

        Returns
        -------
        numpy.ndarray
            The predicted values.
        """
        return self.__xgb_cpu.predict(X=X, **kwargs)

    def _predict_gpu(self, X, sample_weight=None, **kwargs):
        """Predict using xgboost.

        Parameters
        ----------
        X : array-like
            The data to predict.
        sample_weight : array-like, optional
            The sample weights (default is None).
        **kwargs
            Keyword arguments to pass to the `predict` method.

        Returns
        -------
        numpy.ndarray
            The predicted values.
        """
        return self.__xgb_gpu.predict(X=X, **kwargs)
