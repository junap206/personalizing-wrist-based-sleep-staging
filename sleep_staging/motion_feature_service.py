"""
motion_feature_service.py

Per-epoch movement features from the wrist accelerometer.

Raw x, y, z are NOT used directly: they encode wrist ORIENTATION, and when
someone rolls over their baseline shifts for the rest of the night. The
magnitude sqrt(x^2 + y^2 + z^2) and its rate of change do not depend on
orientation.
"""

import numpy as np
import pandas as pd

from sleep_staging.constants import Constants
from sleep_staging.raw_data_service import RawDataService

AXES = ("x", "y", "z")


class MotionFeatureService(object):

    @staticmethod
    def build(motion, n_epochs, recording_start):
        samples = MotionFeatureService.prepare_samples(motion, n_epochs, recording_start)
        features = MotionFeatureService.get_epoch_summaries(samples, n_epochs)
        features = MotionFeatureService.add_stillness(features, n_epochs)
        features = MotionFeatureService.add_rolling_movement(features)
        return features

    @staticmethod
    def prepare_samples(motion, n_epochs, recording_start):
        """Tag every sample with its epoch, magnitude, jerk, and movement flag."""
        df = motion.copy()
        df["epoch"] = RawDataService.get_epoch_index(df["t"].values, recording_start)
        df = df[(df["epoch"] >= 0) & (df["epoch"] < n_epochs)].copy()

        df["mag"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
        df["jerk"] = df["mag"].diff().abs().fillna(0)
        df["is_move"] = (df["jerk"] > Constants.MOVE_THRESHOLD).astype(int)
        return df

    @staticmethod
    def get_epoch_summaries(samples, n_epochs):
        by_epoch = samples.groupby("epoch")

        features = pd.DataFrame(index=np.arange(n_epochs))
        features["mag_mean"] = by_epoch["mag"].mean()
        features["mag_std"] = by_epoch["mag"].std()          # near zero in deep sleep
        features["jerk_mean"] = by_epoch["jerk"].mean()
        features["jerk_std"] = by_epoch["jerk"].std()
        features["jerk_max"] = by_epoch["jerk"].max()
        features["jerk_p95"] = by_epoch["jerk"].quantile(0.95)
        features["move_count"] = by_epoch["is_move"].sum()
        features["move_frac"] = by_epoch["is_move"].mean()
        features["accel_n_samples"] = by_epoch.size().reindex(features.index).fillna(0)
        features["orient_shift"] = MotionFeatureService.get_orientation_shift(by_epoch)
        return features.fillna(0)

    @staticmethod
    def get_orientation_shift(by_epoch):
        """How far the wrist rotated between the start and end of the epoch."""
        total = 0
        for axis in AXES:
            first = by_epoch[axis].first()
            last = by_epoch[axis].last()
            total = total + (last - first).abs()
        return total

    @staticmethod
    def add_stillness(features, n_epochs):
        """Epochs since the last movement, and until the next one."""
        had_move = (features["move_count"] > 0).values

        since = np.zeros(n_epochs)
        counter = 0
        for i in range(n_epochs):
            counter = 0 if had_move[i] else counter + 1
            since[i] = counter

        until = np.zeros(n_epochs)
        counter = 0
        for i in range(n_epochs - 1, -1, -1):
            counter = 0 if had_move[i] else counter + 1
            until[i] = counter

        features["epochs_since_move"] = since
        features["epochs_until_move"] = until
        return features

    @staticmethod
    def add_rolling_movement(features):
        """Fraction of moving samples over the surrounding 5 and 15 minutes."""
        move_frac = features["move_frac"]
        features["move_frac_5min"] = move_frac.rolling(
            Constants.MOVE_SHORT_WINDOW_EPOCHS, center=True, min_periods=1).mean()
        features["move_frac_15min"] = move_frac.rolling(
            Constants.MOVE_LONG_WINDOW_EPOCHS, center=True, min_periods=1).mean()
        return features
