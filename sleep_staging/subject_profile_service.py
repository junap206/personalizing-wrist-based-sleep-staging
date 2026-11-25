"""
subject_profile_service.py

One row per subject describing their typical sleep and how much it varies,
and which of those properties go with being hard for the model to score.
"""

import numpy as np
import pandas as pd
from scipy import stats

from sleep_staging.constants import Constants
from sleep_staging.sleep_architecture_service import SleepArchitectureService
from sleep_staging.statistics_service import StatisticsService

NOT_PREDICTORS = ("macro_f1", "kappa", "accuracy", "n_epochs")


class SubjectProfileService(object):

    @staticmethod
    def describe_night(epochs):
        stages = SleepArchitectureService.get_stages(epochs)
        architecture = SleepArchitectureService.describe(stages)
        return {
            "transitions_per_hour": architecture["transitions_per_hour"],
            "n_transitions": architecture["n_transitions"],
            "hours": architecture["hours"],
            "frac_wake": architecture["frac_wake"],
            "frac_rem": architecture["frac_rem"],
            "hr_mean": epochs["hr_mean"].mean(),
            "hr_std": epochs["hr_std"].mean(),
            "mag_std": epochs["mag_std"].mean(),
            "move_frac": epochs["move_frac"].mean(),
            "stillness": epochs["epochs_since_move"].mean(),
        }

    @staticmethod
    def build(features, night_scores, baseline):
        """Average each subject's nights; add variability and baseline scores."""
        rows = []
        for subject, subject_epochs in features.groupby("subject"):
            nights = []
            for _, night_epochs in subject_epochs.groupby("night"):
                nights.append(SubjectProfileService.describe_night(night_epochs))
            nights = pd.DataFrame(nights)

            row = {"subject": subject, "n_nights": len(nights)}
            for column in nights.columns:
                row[column] = nights[column].mean()
            # How much this person varies against themselves.
            row["frag_sd"] = nights["transitions_per_hour"].std()
            row["hr_sd_across_nights"] = nights["hr_mean"].std()
            rows.append(row)

        profile = pd.DataFrame(rows).set_index("subject")
        score_sd = night_scores.groupby("subject")["macro_f1"].std().rename("score_sd")
        profile = profile.join(score_sd)

        baseline_scores = baseline.set_index("subject")[["macro_f1", "kappa", "accuracy"]]
        profile = profile.join(baseline_scores)
        return profile.dropna(subset=["macro_f1"])

    @staticmethod
    def correlate_with_difficulty(profile, target="macro_f1"):
        """
        Pearson and Spearman correlation of every subject property with the
        subject's baseline score, Benjamini-Hochberg corrected.
        Sorted by the size of the Pearson correlation.
        """
        rows = []
        for column in profile.columns:
            if column in NOT_PREDICTORS or column == target:
                continue
            x = profile[column].values
            y = profile[target].values
            usable = np.isfinite(x) & np.isfinite(y)
            if usable.sum() < Constants.MIN_SUBJECTS_FOR_CORRELATION or np.std(x[usable]) == 0:
                continue

            r, p_value = stats.pearsonr(x[usable], y[usable])
            rho, rho_p = stats.spearmanr(x[usable], y[usable])
            rows.append({"feature": column, "pearson_r": r, "p": p_value,
                         "spearman_rho": rho, "spearman_p": rho_p, "n": int(usable.sum())})

        table = StatisticsService.benjamini_hochberg(pd.DataFrame(rows), "p")
        order = table["pearson_r"].abs().sort_values(ascending=False).index
        return table.reindex(order).reset_index(drop=True)
