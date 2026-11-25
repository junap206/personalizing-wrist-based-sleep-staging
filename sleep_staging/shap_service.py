"""
shap_service.py

SHAP feature importance. SHAP splits every prediction into a contribution
from each feature. Averaging the size of those contributions over many
epochs gives a ranking of which features the model relies on.

The model is trained on most subjects and explained on a few held-out
subjects it never saw, so the ranking reflects generalisation
"""

import numpy as np
import pandas as pd

from sleep_staging.constants import Constants
from sleep_staging.feature_set_service import FeatureSetService
from sleep_staging.model_service import ModelService


class ShapService(object):

    @staticmethod
    def split_subjects(df, seed=Constants.RANDOM_SEED):
        """Hold out about 10% of subjects (at least 3) for explaining."""
        subjects = sorted(df["subject"].unique())
        n_held_out = max(Constants.SHAP_MIN_HELD_OUT_SUBJECTS, len(subjects) // 10)
        generator = np.random.default_rng(seed)
        held_out = set(generator.choice(subjects, size=n_held_out, replace=False))

        is_held_out = df["subject"].isin(held_out)
        return df[~is_held_out], df[is_held_out], held_out

    @staticmethod
    def train(train, features, label, params):
        model = ModelService.build(params)
        model.fit(train[features], train[label])
        return model

    @staticmethod
    def sample_epochs(test, label, n_sample, seed=Constants.RANDOM_SEED):
        """
        Stratified sample: each stage keeps its share of epochs, so the
        rarer stages (Wake, REM) are still represented.
        """
        n_sample = min(n_sample, len(test))
        chosen = []
        for _, stage_rows in test.groupby(label):
            share = int(round(n_sample * len(stage_rows) / len(test)))
            n_take = min(max(1, share), len(stage_rows))
            chosen.extend(stage_rows.sample(n_take, random_state=seed).index)
        return test.loc[chosen]

    @staticmethod
    def explain(model, x):
        """SHAP values as a list with one (epochs x features) array per class."""
        import shap
        values = shap.TreeExplainer(model).shap_values(x)

        # Depending on the shap version this is a list or a 3-D array.
        if isinstance(values, list):
            return values
        values = np.asarray(values)
        if values.ndim == 3:
            return [values[:, :, k] for k in range(values.shape[2])]
        return [values]

    @staticmethod
    def importance_table(features, per_class):
        """One row per feature, most important first."""
        rows = []
        for index, feature in enumerate(features):
            row = {"feature": feature}
            class_importance = []
            for code, class_values in enumerate(per_class):
                importance = np.abs(class_values[:, index]).mean()
                class_importance.append(importance)
                row["shap_" + Constants.STAGE_NAMES.get(code, str(code))] = float(importance)

            source = FeatureSetService.get_signal_source(feature)
            row["mean_abs_shap"] = float(np.mean(class_importance))
            row["group"] = Constants.SIGNAL_LABELS[source]
            row["source"] = source
            row["is_context"] = FeatureSetService.is_context(feature)
            row["is_personal"] = FeatureSetService.is_personal(feature)
            rows.append(row)

        table = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
        total = table["mean_abs_shap"].sum()
        table["pct_of_total"] = table["mean_abs_shap"] / total * 100
        table["cumulative_pct"] = table["pct_of_total"].cumsum()
        return table.reset_index(drop=True)

    @staticmethod
    def count_features_for_share(table, percent):
        """How many top features it takes to reach `percent` of importance."""
        return int((table["cumulative_pct"] < percent).sum()) + 1
