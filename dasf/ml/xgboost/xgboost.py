#!/usr/bin/env python3

import GPUtil

import xgboost as xgb

from dasf.transforms import Fit
from dasf.transforms import Predict
from dasf.transforms import FitPredict

from dasf.utils.funcs import is_gpu_supported


class XGBRegressor(Fit, FitPredict, Predict):
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
        return self.__xgb_mcpu.fit(X=X, y=y, *args, **kwargs)

    def _lazy_fit_gpu(self, X, y=None, sample_weight=None, *args, **kwargs):
        return self.__xgb_mgpu.fit(X=X, y=y, *args, **kwargs)

    def _fit_cpu(self, X, y=None, sample_weight=None):
        return self.__xgb_cpu.fit(X=X, y=y)

    def _fit_gpu(self, X, y=None, sample_weight=None, *args, **kwargs):
        return self.__xgb_gpu.fit(X=X, y=y, *args, **kwargs)

    def _lazy_predict_cpu(self, X, sample_weight=None, **kwargs):
        return self.__xgb_mcpu.predict(X=X, **kwargs)

    def _lazy_predict_gpu(self, X, sample_weight=None, **kwargs):
        return self.__xgb_mgpu.predict(X=X, **kwargs)

    def _predict_cpu(self, X, sample_weight=None, **kwargs):
        return self.__xgb_cpu.predict(X=X, **kwargs)

    def _predict_gpu(self, X, sample_weight=None, **kwargs):
        return self.__xgb_gpu.predict(X=X, **kwargs)
