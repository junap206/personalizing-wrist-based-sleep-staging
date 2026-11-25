"""
model_service.py

Builds the XGBoost model, trains it, and scores its predictions.
Every experiment uses these functions, so every experiment uses the
same model settings.
"""

import json
import os

from sklearn.metrics import cohen_kappa_score, f1_score
from xgboost import XGBClassifier

from sleep_staging.constants import Constants


class ModelService(object):

    @staticmethod
    def load_params(params_file, n_estimators=None, seed=Constants.RANDOM_SEED):
        """Read tuned hyperparameters (from tune_xgboost.py)."""
        if os.path.exists(params_file):
            with open(params_file) as file:
                params = json.load(file)
        else:
            print("  WARNING: " + params_file + " not found, using XGBoost defaults")
            params = {"random_state": seed}

        if n_estimators:
            params["n_estimators"] = n_estimators

        # Removed in recent XGBoost versions; passing it raises an error.
        params.pop("use_label_encoder", None)
        return params

    @staticmethod
    def build(params):
        return XGBClassifier(**params, eval_metric="mlogloss", verbosity=0)

    @staticmethod
    def train_and_score(train, test, features, label, params):
        """Fit on train, predict test, return a dict of scores."""
        model = ModelService.build(params)
        model.fit(train[features], train[label])

        predicted = model.predict(test[features])
        true = test[label].values
        return ModelService.score(true, predicted)

    @staticmethod
    def score(true, predicted):
        stage_codes = sorted(Constants.STAGE_NAMES)
        per_class_f1 = f1_score(true, predicted, average=None,
                                labels=stage_codes, zero_division=0)

        scores = {
            "macro_f1": f1_score(true, predicted, average="macro"),
            "kappa": cohen_kappa_score(true, predicted),
            "accuracy": (predicted == true).mean(),
        }
        for code, value in zip(stage_codes, per_class_f1):
            scores["f1_" + Constants.STAGE_NAMES[code]] = value
        return scores