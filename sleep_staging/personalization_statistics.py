"""
personalization_statistics.py

Turns raw results into per-subject scores and statistical tests.

The SUBJECT is the unit of analysis: scores are averaged over donors first,
so each subject contributes one number per arm and K.
"""

import numpy as np
import pandas as pd
from scipy import stats

from sleep_staging.constants import Constants


class PersonalizationStatistics(object):

    @staticmethod
    def average_over_donors(results):
        """One row per (arm, K, subject)."""
        grouped = results.groupby(["arm", "k", "subject"])
        per_subject = grouped.agg(
            macro_f1=("macro_f1", "mean"),
            kappa=("kappa", "mean"),
            f1_REM=("f1_REM", "mean"),
            accuracy=("accuracy", "mean"),
        )
        return per_subject.reset_index()

    @staticmethod
    def summarise(per_subject):
        """One row per (arm, K): mean and SD across subjects."""
        grouped = per_subject.groupby(["arm", "k"])
        summary = grouped.agg(
            n_subjects=("subject", "nunique"),
            macro_f1=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            kappa=("kappa", "mean"),
            kappa_sd=("kappa", "std"),
            f1_REM=("f1_REM", "mean"),
            f1_REM_sd=("f1_REM", "std"),
            accuracy=("accuracy", "mean"),
        )
        return summary.reset_index()

    @staticmethod
    def get_arm_scores(per_subject, arm, k):
        """macro-F1 per subject for one arm at one K, indexed by subject."""
        rows = per_subject[(per_subject["arm"] == arm) & (per_subject["k"] == k)]
        return rows.set_index("subject")["macro_f1"]

    @staticmethod
    def get_gaps(per_subject, k):
        """Per subject at one K: personal macro-F1 minus stranger macro-F1."""
        personal = PersonalizationStatistics.get_arm_scores(per_subject, "personal", k)
        stranger = PersonalizationStatistics.get_arm_scores(per_subject, "stranger", k)
        common = personal.index.intersection(stranger.index)
        return personal[common] - stranger[common]

    @staticmethod
    def change_within_arm(per_subject, arm, k_from, k_to):
        """
        Paired test of one arm's macro-F1 at k_to vs k_from (same subjects).
        Not controlled for data volume: use compare_arms for that.
        """
        start = PersonalizationStatistics.get_arm_scores(per_subject, arm, k_from)
        end = PersonalizationStatistics.get_arm_scores(per_subject, arm, k_to)
        common = start.index.intersection(end.index)
        return PersonalizationStatistics.paired_test(end[common].values, start[common].values)

    @staticmethod
    def add_corrected_p(tests):
        """Bonferroni correction across the K values tested."""
        tests = tests.copy()
        tests["t_p_corrected"] = (tests["t_p"] * len(tests)).clip(upper=1.0)
        return tests

    @staticmethod
    def compare_arms(per_subject, k_values):
        """Paired test of personal vs stranger at every K > 0."""
        rows = []
        for k in k_values:
            if k == 0:
                continue

            personal = PersonalizationStatistics.get_arm_scores(per_subject, "personal", k)
            stranger = PersonalizationStatistics.get_arm_scores(per_subject, "stranger", k)
            common = personal.index.intersection(stranger.index)
            if len(common) < Constants.MINIMUM_PAIRED_SUBJECTS:
                continue

            row = PersonalizationStatistics.paired_test(
                personal[common].values, stranger[common].values)
            row["k"] = k
            rows.append(row)

        columns = ["k", "n", "personal", "stranger", "gap", "ci_lo", "ci_hi",
                   "cohens_d", "t_p", "wilcoxon_p", "n_improved"]
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def paired_test(personal, stranger):
        gap = personal - stranger
        n = len(gap)

        _, t_p = stats.ttest_rel(personal, stranger)
        try:
            _, wilcoxon_p = stats.wilcoxon(personal, stranger)
        except ValueError:
            wilcoxon_p = np.nan

        cohens_d = gap.mean() / (gap.std(ddof=1) + 1e-9)
        ci_lo, ci_hi = stats.t.interval(Constants.CONFIDENCE_LEVEL, n - 1,
                                        loc=gap.mean(), scale=stats.sem(gap))
        return {
            "n": n,
            "personal": personal.mean(),
            "stranger": stranger.mean(),
            "gap": gap.mean(),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "cohens_d": cohens_d,
            "t_p": t_p,
            "wilcoxon_p": wilcoxon_p,
            "n_improved": int((gap > 0).sum()),
        }

    @staticmethod
    def test_dose_response(per_subject):
        """
        Mixed model: does the personal arm improve faster with K nights than the
        stranger arm?  Returns (slope difference per night, p-value).
        """
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            return np.nan, np.nan

        data = per_subject[per_subject["k"] > 0].copy()
        data["personal"] = (data["arm"] == "personal").astype(int)
        try:
            model = smf.mixedlm("macro_f1 ~ k * personal", data,
                                groups=data["subject"]).fit(reml=False)
        except Exception:
            return np.nan, np.nan
        return model.params.get("k:personal", np.nan), model.pvalues.get("k:personal", np.nan)
