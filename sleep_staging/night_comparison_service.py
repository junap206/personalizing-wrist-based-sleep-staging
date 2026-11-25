"""
night_comparison_service.py

Good night vs bad night. For each subject, take their best-scored and
worst-scored night, then compare features between the two groups.

Every subject contributes one night to each group, so "some people are just
easier" cannot explain a difference: the same people are in both groups.
"""

import numpy as np
import pandas as pd

from sleep_staging.constants import Constants
from sleep_staging.feature_set_service import FeatureSetService
from sleep_staging.sleep_architecture_service import SleepArchitectureService
from sleep_staging.statistics_service import StatisticsService

SCORE_COLUMNS = ["macro_f1", "kappa", "accuracy"]


class NightComparisonService(object):

    @staticmethod
    def summarise_nights(features, night_scores):
        """One row per night: mean of every feature, sleep architecture, scores."""
        feature_columns = FeatureSetService.get_model_features(features.columns)
        means = features.groupby(["subject", "night"])[feature_columns].mean()

        rows = []
        for (subject, night), epochs in features.groupby(["subject", "night"]):
            stages = SleepArchitectureService.get_stages(epochs)
            row = {"subject": subject, "night": night}
            row.update(SleepArchitectureService.describe(stages))
            rows.append(row)
        architecture = pd.DataFrame(rows).set_index(["subject", "night"])

        scores = night_scores.set_index(["subject", "night"])[SCORE_COLUMNS]
        nights = means.join(architecture).join(scores)
        return nights.dropna(subset=["macro_f1"])

    @staticmethod
    def pick_best_and_worst(nights, min_nights=Constants.MIN_NIGHTS_FOR_PAIRING):
        """List of (subject, best night index, worst night index)."""
        pairs = []
        for subject, subject_nights in nights.groupby(level="subject"):
            if len(subject_nights) < min_nights:
                continue
            best = subject_nights["macro_f1"].idxmax()
            worst = subject_nights["macro_f1"].idxmin()
            if best != worst:
                pairs.append((subject, best, worst))
        return pairs

    @staticmethod
    def get_groups(nights, pairs):
        best_rows = [best for _, best, _ in pairs]
        worst_rows = [worst for _, _, worst in pairs]
        return nights.loc[best_rows], nights.loc[worst_rows]

    @staticmethod
    def compare(nights, pairs, columns):
        """
        Paired test of every column, best night vs worst night, with the
        Benjamini-Hochberg correction across all columns tested.
        Sorted by effect size, largest first.
        """
        good, bad = NightComparisonService.get_groups(nights, pairs)

        rows = []
        for column in columns:
            result = StatisticsService.paired_comparison(
                good[column].values, bad[column].values)
            if result is None:
                continue
            usable = np.isfinite(good[column].values) & np.isfinite(bad[column].values)
            good_mean = good[column].values[usable].mean()
            bad_mean = bad[column].values[usable].mean()
            rows.append({
                "feature": column,
                "good_mean": good_mean,
                "bad_mean": bad_mean,
                "diff": result["difference"],
                "pct_change": (result["difference"] / abs(bad_mean) * 100
                               if bad_mean != 0 else np.nan),
                "cohens_d": result["cohens_d"],
                "t_p": result["t_p"],
                "wilcoxon_p": result["wilcoxon_p"],
                "n": result["n"],
            })

        table = pd.DataFrame(rows)
        if table.empty:
            return table
        table = StatisticsService.benjamini_hochberg(table, "t_p")
        table = table.sort_values("cohens_d", key=abs, ascending=False)
        return table.reset_index(drop=True)
