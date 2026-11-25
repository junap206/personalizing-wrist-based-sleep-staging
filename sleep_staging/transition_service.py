"""
transition_service.py

Are the model's errors concentrated where the true sleep stage changes?

For every epoch we find how many epochs away the nearest change in the TRUE
stage is (0 = this epoch is part of a change), then compare error rates by
that distance. Error rates are computed per subject first and then
averaged, so subjects with more epochs do not dominate.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from sleep_staging.constants import Constants

NO_TRANSITION = 10 ** 9


class TransitionService(object):

    # ------------------------------------------------------------------
    # Per-epoch annotation
    # ------------------------------------------------------------------
    @staticmethod
    def distance_to_transition(true):
        """Epochs to the nearest change in the true stage (0 = at a change)."""
        n = len(true)
        is_change = np.zeros(n, dtype=bool)
        is_change[1:] |= true[1:] != true[:-1]    # differs from the previous epoch
        is_change[:-1] |= true[:-1] != true[1:]   # differs from the next epoch

        change_positions = np.where(is_change)[0]
        if len(change_positions) == 0:
            return np.full(n, NO_TRANSITION)

        positions = np.arange(n)
        left = np.searchsorted(change_positions, positions, side="right") - 1
        right = np.clip(left + 1, 0, len(change_positions) - 1)
        left = np.clip(left, 0, len(change_positions) - 1)
        to_left = np.abs(positions - change_positions[left])
        to_right = np.abs(positions - change_positions[right])
        return np.minimum(to_left, to_right)

    @staticmethod
    def transition_type(true):
        """
        Name the change each transition epoch belongs to, e.g. 'NREM->REM'.
        Both epochs on either side of a change get the label.
        """
        names = Constants.STAGE_NAMES
        labels = [None] * len(true)
        for i in range(1, len(true)):
            if true[i] == true[i - 1]:
                continue
            label = names[true[i - 1]] + "->" + names[true[i]]
            labels[i] = label
            if labels[i - 1] is None:
                labels[i - 1] = label
        return labels

    @staticmethod
    def get_bin(distance):
        for lowest, highest, name in Constants.TRANSITION_DISTANCE_BINS:
            if lowest <= distance <= highest:
                return name
        return Constants.TRANSITION_DISTANCE_BINS[-1][2]

    @staticmethod
    def annotate(predictions):
        """Add distance, bin, transition type, and error to every epoch."""
        nights = []
        for _, night in predictions.groupby(["subject", "night"]):
            night = night.sort_values("epoch").copy()
            true = night["true"].values
            night["dist"] = TransitionService.distance_to_transition(true)
            night["ttype"] = TransitionService.transition_type(true)
            nights.append(night)

        epochs = pd.concat(nights, ignore_index=True)
        epochs["error"] = (epochs["true"] != epochs["pred"]).astype(int)
        epochs["bin"] = epochs["dist"].apply(TransitionService.get_bin)
        return epochs

    # ------------------------------------------------------------------
    # Error rates
    # ------------------------------------------------------------------
    @staticmethod
    def error_by_distance(epochs):
        """Returns (summary per bin, per-subject error rate per bin)."""
        order = [name for _, _, name in Constants.TRANSITION_DISTANCE_BINS]
        per_subject = epochs.groupby(["subject", "bin"])["error"].mean()
        per_subject = per_subject.unstack("bin").reindex(columns=order)

        epochs_per_bin = epochs.groupby("bin").size()
        summary = pd.DataFrame({
            "error_rate": per_subject.mean(),
            "sd_across_subjects": per_subject.std(),
            "n_subjects": per_subject.notna().sum(),
            "n_epochs": epochs_per_bin.reindex(order),
            "share_of_epochs": (epochs_per_bin / len(epochs)).reindex(order),
        })
        return summary, per_subject

    @staticmethod
    def compare_transition_to_stable(per_subject):
        """Paired t-test: error at transitions vs in stable sleep, per subject."""
        at_transition = per_subject["at transition"]
        stable = per_subject[">10 (stable)"]
        both = at_transition.notna() & stable.notna()
        _, p_value = stats.ttest_rel(at_transition[both], stable[both])
        return p_value, int(both.sum())

    @staticmethod
    def error_by_type(epochs):
        """Error rate at transition epochs, by type of transition."""
        at_change = epochs[epochs["dist"] == 0].dropna(subset=["ttype"])
        per_subject = at_change.groupby(["subject", "ttype"])["error"].mean()
        per_subject = per_subject.unstack("ttype")

        table = pd.DataFrame({
            "error_rate": per_subject.mean(),
            "sd": per_subject.std(),
            "n_subjects": per_subject.notna().sum(),
            "n_epochs": at_change.groupby("ttype").size(),
        })
        return table.sort_values("error_rate", ascending=False)

    @staticmethod
    def error_concentration(epochs):
        """Share of all epochs vs share of all errors that sit near a change."""
        near = epochs["dist"] <= Constants.NEAR_TRANSITION_EPOCHS
        return {
            "share_epochs_near": near.mean(),
            "share_errors_near": epochs.loc[near, "error"].sum() / epochs["error"].sum(),
        }

    # ------------------------------------------------------------------
    # Night level
    # ------------------------------------------------------------------
    @staticmethod
    def score_nights(epochs):
        """One row per night: macro-F1, how fragmented it was, error near/far."""
        near_limit = Constants.NEAR_TRANSITION_EPOCHS
        far_limit = Constants.FAR_FROM_TRANSITION_EPOCHS

        rows = []
        for (subject, night), night_epochs in epochs.groupby(["subject", "night"]):
            if night_epochs["true"].nunique() < 2:
                continue
            true_sequence = night_epochs.sort_values("epoch")["true"].values
            near = night_epochs["dist"] <= near_limit
            far = night_epochs["dist"] > far_limit
            rows.append({
                "subject": subject,
                "night": night,
                "macro_f1": f1_score(night_epochs["true"], night_epochs["pred"],
                                     average="macro"),
                "frac_near_transition": near.mean(),
                "n_transitions": int((np.diff(true_sequence) != 0).sum()),
                "error_near": night_epochs.loc[near, "error"].mean(),
                "error_far": night_epochs.loc[far, "error"].mean(),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def within_subject_correlation(nights):
        """
        Remove each subject's own mean from both variables, then correlate.
        This asks: on a subject's more fragmented nights, is F1 lower?

        Returns r, p, and p_adjusted. p_adjusted uses n - n_subjects - 1
        degrees of freedom, because one mean was estimated per subject.
        """
        for column in ("macro_f1", "frac_near_transition"):
            subject_mean = nights.groupby("subject")[column].transform("mean")
            nights[column + "_c"] = nights[column] - subject_mean

        usable = nights.dropna(subset=["macro_f1_c", "frac_near_transition_c"])
        r, p_value = stats.pearsonr(usable["frac_near_transition_c"], usable["macro_f1_c"])

        degrees = len(usable) - usable["subject"].nunique() - 1
        t_value = r * np.sqrt(degrees / (1 - r ** 2))
        p_adjusted = 2 * stats.t.sf(abs(t_value), degrees)
        return nights, r, p_value, p_adjusted
