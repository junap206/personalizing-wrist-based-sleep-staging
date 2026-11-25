"""
ablation_service.py

Ablation: train the same model on different subsets of the features
(heart rate only, motion only, etc) and compare how well each one does.
This shows which signal is doing the work.
"""

import time

import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from sleep_staging.constants import Constants
from sleep_staging.feature_set_service import FeatureSetService
from sleep_staging.loso_service import LosoService


class AblationService(object):

    @staticmethod
    def run(df, all_features, label, subjects, params, set_names):
        """
        Leave-one-subject-out for every feature set.
        Returns per-subject results (all sets stacked) and per-class F1.
        """
        result_tables = []
        per_class_f1 = {}

        for set_name in set_names:
            features = FeatureSetService.select(all_features, set_name)
            start = time.time()
            per_subject, predictions = LosoService.run(
                df, features, label, subjects, params, verbose=False)

            per_subject["set"] = set_name
            per_subject["n_features"] = len(features)
            result_tables.append(per_subject)
            per_class_f1[set_name] = AblationService.get_per_class_f1(predictions)

            minutes = (time.time() - start) / 60
            print(f"  {set_name:<12}{len(features):>4} features   "
                  f"macro-F1 {per_subject['macro_f1'].mean():.3f}   "
                  f"kappa {per_subject['kappa'].mean():.3f}   ({minutes:.1f} min)")

        return pd.concat(result_tables, ignore_index=True), per_class_f1

    @staticmethod
    def get_per_class_f1(predictions):
        codes = sorted(Constants.STAGE_NAMES)
        return f1_score(predictions["true"], predictions["pred"],
                        average=None, labels=codes)

    @staticmethod
    def summarise(results):
        """One row per feature set, weakest first."""
        summary = results.groupby("set").agg(
            n_features=("n_features", "first"),
            macro_f1=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            kappa=("kappa", "mean"),
            accuracy=("accuracy", "mean"),
        )
        summary = summary.reset_index().sort_values("macro_f1")
        return summary.reset_index(drop=True)

    @staticmethod
    def compare_to_reference(results, reference="all"):
        """
        Paired t-test of every set against the reference set, within subject.
        p_corrected applies a Bonferroni correction for the number of sets
        compared.
        """
        reference_scores = AblationService.get_scores(results, reference)
        other_sets = [name for name in results["set"].unique() if name != reference]

        rows = []
        for set_name in other_sets:
            scores = AblationService.get_scores(results, set_name)
            common = reference_scores.index.intersection(scores.index)
            if len(common) < Constants.MINIMUM_PAIRED_SUBJECTS:
                continue
            difference = scores[common] - reference_scores[common]
            _, p_value = stats.ttest_rel(scores[common], reference_scores[common])
            rows.append({
                "set": set_name,
                "reference": reference,
                "n": len(common),
                "mean_difference": difference.mean(),
                "p": p_value,
                "p_corrected": min(1.0, p_value * len(other_sets)),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def get_scores(results, set_name):
        rows = results[results["set"] == set_name]
        return rows.set_index("subject")["macro_f1"]
