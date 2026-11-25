"""
heart_rate_feature_service.py

Per-epoch heart-rate features.

Why normalize: across subjects heart rate differs by ~15 bpm, but between
sleep stages within one subject it differs by only ~3 bpm. Raw bpm mostly
tells the model WHO the person is, not WHAT STAGE they are in. Subtracting
the night's own median removes the person and leaves the stage signal.
"""

import numpy as np
import pandas as pd

from sleep_staging.constants import Constants
from sleep_staging.raw_data_service import RawDataService

SUMMARY_COLUMNS = ("hr_mean", "hr_std", "hr_min", "hr_max", "hr_range")


class HeartRateFeatureService(object):

    @staticmethod
    def build(heart_rate, n_epochs, recording_start):
        features = HeartRateFeatureService.get_epoch_summaries(
            heart_rate, n_epochs, recording_start)
        features = HeartRateFeatureService.add_normalised(features)
        features = HeartRateFeatureService.add_trends(features)
        return features

    @staticmethod
    def get_epoch_summaries(heart_rate, n_epochs, recording_start):
        """Mean, spread, and range of heart rate inside each epoch."""
        df = heart_rate.copy()
        df["epoch"] = RawDataService.get_epoch_index(df["t"].values, recording_start)
        df = df[(df["epoch"] >= 0) & (df["epoch"] < n_epochs)]
        by_epoch = df.groupby("epoch")["bpm"]

        features = pd.DataFrame(index=np.arange(n_epochs))
        features["hr_mean"] = by_epoch.mean()
        features["hr_std"] = by_epoch.std()
        features["hr_min"] = by_epoch.min()
        features["hr_max"] = by_epoch.max()
        features["hr_range"] = features["hr_max"] - features["hr_min"]
        features["hr_n_samples"] = by_epoch.count().reindex(features.index).fillna(0)

        # Epochs with no reading: carry the last value forward, then back.
        for column in SUMMARY_COLUMNS:
            features[column] = features[column].ffill().bfill()
        return features

    @staticmethod
    def add_normalised(features):
        """Heart rate relative to this night's own median (personal features)."""
        night_median = features["hr_mean"].median()
        night_std = features["hr_mean"].std()
        features["hr_relative"] = features["hr_mean"] - night_median
        features["hr_zscore"] = (features["hr_mean"] - night_median) / (night_std + 1e-6)
        return features

    @staticmethod
    def add_trends(features):
        """Slow drift across the night, and short-term change."""
        baseline = features["hr_mean"].rolling(
            Constants.HR_BASELINE_EPOCHS, center=True, min_periods=1).mean()
        features["hr_baseline_30min"] = baseline
        features["hr_vs_baseline"] = features["hr_mean"] - baseline

        features["hr_delta"] = features["hr_mean"].diff().fillna(0)
        features["hr_delta_abs"] = features["hr_delta"].abs()
        return features
