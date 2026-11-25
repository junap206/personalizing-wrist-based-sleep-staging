"""
label_service.py

Questions about the labels themselves.

    Automatic vs expert: the dataset has the Dreem headband's automatic
    labels and the expert-corrected ones. How often, and where, do they
    disagree?

    Unknown epochs: epochs the headband could not score (code 5) are dropped
    everywhere. If they sit at stage transitions, dropping them removes
    hard cases and flatters every result.
"""

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import cohen_kappa_score

from sleep_staging.constants import Constants
from sleep_staging.raw_data_service import RawDataService

UNKNOWN = 5
NO_CHANGE = 10 ** 6


class LabelService(object):

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_both(root):
        """One row per epoch with both label sets, before anything is dropped."""
        nights = []
        for folder, subject, night in RawDataService.find_nights(root):
            mat = loadmat(os.path.join(folder, "labels.mat"))
            if "expert_label" not in mat or "dreem_label" not in mat:
                continue
            expert = np.asarray(mat["expert_label"]).ravel().astype(int)
            automatic = np.asarray(mat["dreem_label"]).ravel().astype(int)
            n = min(len(expert), len(automatic))
            nights.append(pd.DataFrame({
                "subject": subject, "night": night, "epoch": np.arange(n),
                "expert5": expert[:n], "auto5": automatic[:n],
            }))

        labels = pd.concat(nights, ignore_index=True)
        to_three = Constants.FIVE_TO_THREE_CLASS
        labels["expert3"] = labels["expert5"].map(to_three).fillna(-1).astype(int)
        labels["auto3"] = labels["auto5"].map(to_three).fillna(-1).astype(int)
        return LabelService.add_distances(labels)

    @staticmethod
    def distance_to_change(stages):
        """
        Epochs to the nearest change in the expert stage. Unscored epochs
        (-1) are ignored: a change next to one does not count.
        """
        n = len(stages)
        scored = stages >= 0
        is_change = np.zeros(n, dtype=bool)
        is_change[1:] |= (stages[1:] != stages[:-1]) & scored[1:] & scored[:-1]
        is_change[:-1] |= (stages[:-1] != stages[1:]) & scored[:-1] & scored[1:]

        change_positions = np.where(is_change)[0]
        if len(change_positions) == 0:
            return np.full(n, NO_CHANGE)
        positions = np.arange(n)
        left = np.searchsorted(change_positions, positions, side="right") - 1
        left = np.clip(left, 0, len(change_positions) - 1)
        right = np.clip(left + 1, 0, len(change_positions) - 1)
        to_left = np.abs(positions - change_positions[left])
        to_right = np.abs(positions - change_positions[right])
        return np.minimum(to_left, to_right)

    @staticmethod
    def add_distances(labels):
        nights = []
        for _, night in labels.groupby(["subject", "night"]):
            night = night.sort_values("epoch").copy()
            night["dist_expert"] = LabelService.distance_to_change(night["expert3"].values)
            nights.append(night)
        return pd.concat(nights, ignore_index=True)

    # ------------------------------------------------------------------
    # Automatic vs expert
    # ------------------------------------------------------------------
    @staticmethod
    def agreement(labels):
        """Agreement and kappa between the two label sets, scored epochs only."""
        both = LabelService.scored_by_both(labels)
        return {
            "n_epochs": len(both),
            "agree5": (both["expert5"] == both["auto5"]).mean(),
            "agree3": (both["expert3"] == both["auto3"]).mean(),
            "kappa5": cohen_kappa_score(both["expert5"], both["auto5"]),
            "kappa3": cohen_kappa_score(both["expert3"], both["auto3"]),
        }

    @staticmethod
    def scored_by_both(labels):
        return labels[(labels["expert3"] >= 0) & (labels["auto3"] >= 0)].copy()

    @staticmethod
    def top_corrections(labels, n=8):
        """Most common (automatic -> expert) changes."""
        both = LabelService.scored_by_both(labels)
        changed = both[both["expert5"] != both["auto5"]]
        counts = changed.groupby(["auto5", "expert5"]).size()
        return counts.sort_values(ascending=False).head(n), len(changed)

    @staticmethod
    def corrections_near_transitions(labels):
        """Do expert corrections cluster where the stage changes?"""
        both = LabelService.scored_by_both(labels)
        both["disagree"] = (both["expert3"] != both["auto3"]).astype(int)
        near = both["dist_expert"] <= Constants.NEAR_TRANSITION_EPOCHS
        far = both["dist_expert"] > Constants.FAR_FROM_TRANSITION_EPOCHS
        return {
            "rate_near": both.loc[near, "disagree"].mean(),
            "rate_far": both.loc[far, "disagree"].mean(),
            "share_epochs_near": near.mean(),
            "share_corrections_near": both.loc[near, "disagree"].sum() / both["disagree"].sum(),
        }

    # ------------------------------------------------------------------
    # Unknown epochs
    # ------------------------------------------------------------------
    @staticmethod
    def neighbour_patterns(labels):
        """
        For every epoch with a scored epoch on both sides, record whether the
        two neighbours are the same stage or different stages. Returns the
        share of 'different' for Unknown epochs and, as the chance level,
        for ordinary scored epochs.
        """
        unknown_patterns = []
        scored_patterns = []
        for _, night in labels.groupby(["subject", "night"]):
            stages = night.sort_values("epoch")["expert3"].values
            raw = night.sort_values("epoch")["expert5"].values
            for i in range(1, len(stages) - 1):
                before = stages[i - 1]
                after = stages[i + 1]
                if before < 0 or after < 0:
                    continue
                differs = before != after
                if raw[i] == UNKNOWN:
                    unknown_patterns.append(differs)
                elif stages[i] >= 0:
                    scored_patterns.append(differs)
        return {
            "n_unknown_bounded": len(unknown_patterns),
            "unknown_between_different": np.mean(unknown_patterns) if unknown_patterns else np.nan,
            "chance_between_different": np.mean(scored_patterns) if scored_patterns else np.nan,
        }

    @staticmethod
    def unknown_summary(labels):
        is_unknown = labels["expert5"] == UNKNOWN
        near = labels["dist_expert"] <= Constants.NEAR_TRANSITION_EPOCHS
        is_scored = ~is_unknown & (labels["expert3"] >= 0)
        per_subject = labels.groupby("subject")["expert5"].apply(
            lambda stages: (stages == UNKNOWN).mean() * 100)
        return {
            "n_unknown": int(is_unknown.sum()),
            "share_unknown": is_unknown.mean(),
            "unknown_near": near[is_unknown].mean() if is_unknown.any() else np.nan,
            "scored_near": near[is_scored].mean(),
            "median_subject_pct": per_subject.median(),
            "max_subject_pct": per_subject.max(),
            "max_subject": per_subject.idxmax(),
        }
