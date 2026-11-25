"""
tuning_service.py

Searches for good XGBoost settings with Optuna, using training data only.

    1. Hold out a few subjects as a validation set. Tuning never sees them.
    2. Optuna tries settings, scoring each by grouped cross-validation on
       the remaining subjects (a subject is never in train and test at once).
    3. The best settings are checked once on the held-out subjects.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from sleep_staging.constants import Constants
from sleep_staging.model_service import ModelService


class TuningService(object):

    @staticmethod
    def split_validation_subjects(df, n_validation, seed):
        generator = np.random.default_rng(seed)
        subjects = sorted(df["subject"].unique())
        validation = set(generator.choice(subjects, size=n_validation, replace=False))

        is_validation = df["subject"].isin(validation)
        tuning_df = df[~is_validation].reset_index(drop=True)
        validation_df = df[is_validation].reset_index(drop=True)
        return tuning_df, validation_df, validation

    @staticmethod
    def suggest_params(trial):
        """The search space. Each line is one setting Optuna may change."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": Constants.RANDOM_SEED,
        }

    @staticmethod
    def cross_validate(params, tuning_df, features, label,
                       n_folds=Constants.TUNING_INNER_FOLDS, seed=Constants.RANDOM_SEED):
        """Mean macro-F1 over folds grouped by subject, stratified by stage."""
        x = tuning_df[features].values
        y = tuning_df[label].values
        groups = tuning_df["subject"].values

        folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for train_index, test_index in folds.split(x, y, groups):
            model = ModelService.build(params)
            model.fit(x[train_index], y[train_index])
            predicted = model.predict(x[test_index])
            scores.append(f1_score(y[test_index], predicted, average="macro"))
        return np.mean(scores)

    @staticmethod
    def make_objective(tuning_df, features, label):
        def objective(trial):
            params = TuningService.suggest_params(trial)
            return TuningService.cross_validate(params, tuning_df, features, label)
        return objective

    @staticmethod
    def validate(params, tuning_df, validation_df, features, label):
        """Train on all tuning subjects, score the held-out subjects once."""
        model = ModelService.build(params)
        model.fit(tuning_df[features], tuning_df[label])

        predicted = model.predict(validation_df[features])
        overall = f1_score(validation_df[label], predicted, average="macro")

        rows = []
        for subject in sorted(validation_df["subject"].unique()):
            is_subject = validation_df["subject"] == subject
            subject_predicted = model.predict(validation_df.loc[is_subject, features])
            score = f1_score(validation_df.loc[is_subject, label], subject_predicted,
                             average="macro")
            rows.append({"subject": subject, "macro_f1": score})
        return overall, pd.DataFrame(rows)

    @staticmethod
    def get_history(study):
        rows = []
        for trial in study.trials:
            if trial.value is None:
                continue
            row = {"trial": trial.number, "value": trial.value}
            row.update(trial.params)
            rows.append(row)
        return pd.DataFrame(rows)
