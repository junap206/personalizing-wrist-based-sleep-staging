"""
heterogeneity_service.py

Does personalization help the subjects the population model scores worst?

Each subject appears once in each arm, so the clean test is paired:
gap = personal - stranger (at K), regressed on the baseline score. The
baseline term cancels inside the gap.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from sleep_staging.constants import Constants


class HeterogeneityService(object):

    @staticmethod
    def build_change_table(per_subject, k):
        """One row per (subject, arm): K=0 score, score at K, and the change."""
        is_baseline = (per_subject["arm"] == "personal") & (per_subject["k"] == 0)
        baseline = per_subject[is_baseline].set_index("subject")["macro_f1"]

        rows = []
        for arm in Constants.ARMS:
            at_k = per_subject[(per_subject["arm"] == arm) & (per_subject["k"] == k)]
            at_k = at_k.set_index("subject")["macro_f1"]
            for subject in baseline.index.intersection(at_k.index):
                rows.append({
                    "subject": subject,
                    "arm": arm,
                    "baseline": baseline[subject],
                    "at_k": at_k[subject],
                    "delta": at_k[subject] - baseline[subject],
                })
        return pd.DataFrame(rows)

    @staticmethod
    def get_gaps(changes):
        """Per subject: personal minus stranger at K, plus the K=0 score."""
        personal = changes[changes["arm"] == "personal"].set_index("subject")
        stranger = changes[changes["arm"] == "stranger"].set_index("subject")
        common = personal.index.intersection(stranger.index)
        return pd.DataFrame({
            "baseline": personal.loc[common, "baseline"],
            "gap": personal.loc[common, "at_k"] - stranger.loc[common, "at_k"],
        })

    @staticmethod
    def arm_slope(changes, arm):
        """Change from K=0 regressed on the K=0 score, for one arm."""
        rows = changes[changes["arm"] == arm]
        fit = stats.linregress(rows["baseline"], rows["delta"])
        return {"slope": fit.slope, "intercept": fit.intercept,
                "r": fit.rvalue, "p": fit.pvalue, "n": len(rows)}

    @staticmethod
    def interaction_test(changes):
        """
        delta ~ baseline * arm, fitted to both arms together. The
        baseline:arm term asks whether the slopes differ. This treats the
        two rows of a subject as independent; paired_slope_test does not.
        """
        data = changes.copy()
        data["arm"] = (data["arm"] == "personal").astype(int)
        model = smf.ols("delta ~ baseline * arm", data=data).fit()
        return model.params.get("baseline:arm", np.nan), model.pvalues.get("baseline:arm", np.nan)

    @staticmethod
    def paired_slope_test(changes):
        """
        Gap (personal - stranger) regressed on baseline, one row per subject.
        Same slope as the interaction term, with a paired p-value.
        """
        gaps = HeterogeneityService.get_gaps(changes)
        fit = stats.linregress(gaps["baseline"], gaps["gap"])
        return fit.slope, fit.pvalue

    @staticmethod
    def split_at_median(changes):
        """Gap in the harder half vs the easier half of subjects (Welch t-test)."""
        gaps = HeterogeneityService.get_gaps(changes)
        median = gaps["baseline"].median()
        hard = gaps.loc[gaps["baseline"] <= median, "gap"]
        easy = gaps.loc[gaps["baseline"] > median, "gap"]
        _, p_value = stats.ttest_ind(hard, easy, equal_var=False)
        return hard, easy, p_value

    @staticmethod
    def test_moderators(changes, night_scores=None, profile=None):
        """
        Exploratory: correlate each subject's gap with subject properties.
        Not corrected for multiple testing.
        """
        gaps = HeterogeneityService.get_gaps(changes)
        properties = pd.DataFrame(index=gaps.index)
        if night_scores is not None:
            by_subject = night_scores.groupby("subject")
            properties["consistency_sd"] = by_subject["macro_f1"].std()
            properties["n_nights"] = by_subject["night"].nunique()
        if profile is not None:
            for column in Constants.MODERATOR_PROPERTIES:
                if column in profile:
                    properties[column] = profile[column]
        properties = properties.join(gaps["gap"]).dropna(subset=["gap"])

        rows = []
        for column in properties.columns:
            if column == "gap":
                continue
            usable = properties[[column, "gap"]].dropna()
            if len(usable) < Constants.MIN_SUBJECTS_FOR_CORRELATION or usable[column].std() == 0:
                continue
            r, p_value = stats.pearsonr(usable[column], usable["gap"])
            rows.append({"property": column, "r": r, "p": p_value, "n": len(usable)})

        if not rows:
            return None
        return pd.DataFrame(rows).sort_values("p")
