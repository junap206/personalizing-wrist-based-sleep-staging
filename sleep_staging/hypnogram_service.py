"""
hypnogram_service.py

Data for hypnogram figures: each night's true stages, the model's
predicted stages, wrist motion, and a few summary numbers.

Predictions are placed by their EPOCH number, not their row position.
The predictions file has the Unknown epochs removed, so counting rows
would shift every later prediction one epoch to the left per Unknown epoch.
"""

import numpy as np

from sleep_staging.constants import Constants
from sleep_staging.raw_data_service import RawDataService

# Plot rows, deepest sleep at the bottom (the conventional hypnogram layout).
STAGE_ROW = {3: 0, 2: 1, 1: 2, 4: 3, 0: 4}      # five-class code -> row
STAGE_ROW_LABELS = ["N3", "N2", "N1", "REM", "Wake"]
PREDICTED_ROW = {0: 4, 1: 1, 2: 3}               # three-class: Wake, NREM (drawn at N2), REM


class HypnogramService(object):

    @staticmethod
    def nights_by_subject(root, subjects):
        """{subject: [(night, folder), ...]} with nights in numeric order (1, 2, ..., 10)."""
        grouped = {}
        for folder, subject, night in RawDataService.find_nights(root, subjects):
            grouped.setdefault(subject, []).append((night, folder))
        for subject in grouped:
            grouped[subject].sort(key=lambda item: (len(item[0]), item[0]))
        return grouped

    @staticmethod
    def load_night(folder):
        """True five-class labels, mean jerk per epoch, and recording start."""
        motion, _, labels, recording_start = RawDataService.load_night(folder)
        epochs = RawDataService.get_epoch_index(motion["t"].values, recording_start)
        motion = motion.assign(epoch=epochs)
        inside = (motion["epoch"] >= 0) & (motion["epoch"] < len(labels))
        motion = motion[inside].copy()

        motion["mag"] = np.sqrt(motion["x"] ** 2 + motion["y"] ** 2 + motion["z"] ** 2)
        motion["jerk"] = motion["mag"].diff().abs()
        jerk = motion.groupby("epoch")["jerk"].mean().reindex(np.arange(len(labels)))
        return labels, jerk.values, recording_start

    @staticmethod
    def describe(labels):
        """Night length, fragmentation, and stage fractions (Unknown left out)."""
        scored = labels[np.isin(labels, list(STAGE_ROW))]
        if len(scored) == 0:
            return {}
        transitions = int((np.diff(scored) != 0).sum())
        hours = len(scored) * Constants.EPOCH_SECONDS / 3600
        return {
            "hours": hours,
            "transitions": transitions,
            "trans_per_hour": transitions / hours,
            "frac_wake": (scored == 0).mean(),
            "frac_n3": (scored == 3).mean(),
            "frac_rem": (scored == 4).mean(),
        }

    @staticmethod
    def epoch_hours(n_epochs):
        return np.arange(n_epochs) * Constants.EPOCH_SECONDS / 3600

    @staticmethod
    def true_rows(labels):
        """Plot row for every epoch; Unknown epochs become gaps (NaN)."""
        rows = np.full(len(labels), np.nan)
        for epoch, stage in enumerate(labels):
            if stage in STAGE_ROW:
                rows[epoch] = STAGE_ROW[stage]
        return HypnogramService.epoch_hours(len(labels)), rows

    @staticmethod
    def predicted_rows(night_predictions, n_epochs):
        """Plot row for every epoch, placed by epoch number; missing epochs are gaps."""
        rows = np.full(n_epochs, np.nan)
        for epoch, stage in zip(night_predictions["epoch"], night_predictions["pred"]):
            if 0 <= epoch < n_epochs:
                rows[epoch] = PREDICTED_ROW.get(int(stage), np.nan)
        return HypnogramService.epoch_hours(n_epochs), rows

    @staticmethod
    def get_night_predictions(predictions, subject, night):
        is_night = ((predictions["subject"] == subject)
                    & (predictions["night"].astype(str) == str(night)))
        return predictions[is_night].sort_values("epoch")

    @staticmethod
    def accuracy(night_predictions):
        """Share of epochs where the predicted stage equals the true three-class stage."""
        if len(night_predictions) == 0:
            return np.nan
        return (night_predictions["true"] == night_predictions["pred"]).mean()

    @staticmethod
    def choose_subjects(subjects, night_scores, n):
        """The n/2 most and n/2 least consistent subjects (needs 3+ scored nights)."""
        if night_scores is None:
            return subjects[:n]
        spread = night_scores.groupby("subject")["macro_f1"].agg(["std", "count"])
        spread = spread[spread["count"] >= 3].sort_values("std")
        spread = spread[spread.index.isin(subjects)]
        if len(spread) < n:
            return list(spread.index)
        half = n // 2
        return list(spread.index[:half]) + list(spread.index[-(n - half):])
