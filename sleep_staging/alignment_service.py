"""
alignment_service.py

Checks that one night's watch signals line up with its sleep-stage labels.
1. Tags each sample with its epoch and stage

2. Measures basic data quality 
- sample_rate - gap between samples and the implied rate
- share_inside_window - what fraction of samples fall inside labeled period
- epochs_without_samples - how many epochs have no samples at all

3. Runs the two physiological tests
- motion_by_stage - average sample-to-sample change in acceleration magnitude in Wake vs N3
- heart_rate_by_stage - average bpm in Wake vs N3

4. Agregegates to one value per epoch

"""

import numpy as np

from sleep_staging.raw_data_service import RawDataService

WAKE = 0
N3 = 3


class AlignmentService(object):

    @staticmethod
    def attach_stage(signal, labels, recording_start):
        """Add each sample's epoch and stage (-1 outside the labelled window)."""
        epochs = RawDataService.get_epoch_index(signal["t"].values, recording_start)
        tagged = signal.copy()
        tagged["epoch"] = epochs
        inside = (epochs >= 0) & (epochs < len(labels))
        tagged["stage"] = np.where(inside, labels[np.clip(epochs, 0, len(labels) - 1)], -1)
        return tagged

    @staticmethod
    def sampling_rate(signal):
        """Median gap between samples (seconds) and the rate it implies (Hz)."""
        gaps = np.diff(signal["t"].values)
        gaps = gaps[(gaps > 0) & (gaps < 60)]
        if len(gaps) == 0:
            return None
        median_gap = float(np.median(gaps))
        return median_gap, 1 / median_gap

    @staticmethod
    def share_inside_window(signal, recording_start, n_epochs, epoch_seconds):
        end = recording_start + n_epochs * epoch_seconds
        return ((signal["t"] >= recording_start) & (signal["t"] < end)).mean()

    @staticmethod
    def epochs_without_samples(heart_rate, labels, recording_start):
        tagged = AlignmentService.attach_stage(heart_rate, labels, recording_start)
        covered = tagged.loc[tagged["stage"] >= 0, "epoch"].nunique()
        return len(labels) - covered

    @staticmethod
    def motion_by_stage(motion, labels, recording_start):
        """Mean sample-to-sample change in acceleration magnitude, Wake vs N3."""
        tagged = AlignmentService.attach_stage(motion, labels, recording_start)
        tagged["mag"] = np.sqrt(tagged["x"] ** 2 + tagged["y"] ** 2 + tagged["z"] ** 2)
        tagged["jerk"] = tagged["mag"].diff().abs()
        return (tagged.loc[tagged["stage"] == WAKE, "jerk"].mean(),
                tagged.loc[tagged["stage"] == N3, "jerk"].mean())

    @staticmethod
    def heart_rate_by_stage(heart_rate, labels, recording_start):
        tagged = AlignmentService.attach_stage(heart_rate, labels, recording_start)
        return (tagged.loc[tagged["stage"] == WAKE, "bpm"].mean(),
                tagged.loc[tagged["stage"] == N3, "bpm"].mean())

    @staticmethod
    def check(awake_value, deep_value):
        """'ok' if the awake value is higher, 'problem' if not, None if untestable."""
        if np.isnan(awake_value) or np.isnan(deep_value):
            return None
        return "ok" if awake_value > deep_value else "problem"

    @staticmethod
    def per_epoch_mean(signal, column, recording_start, n_epochs):
        epochs = RawDataService.get_epoch_index(signal["t"].values, recording_start)
        means = signal.assign(epoch=epochs).groupby("epoch")[column].mean()
        return means.reindex(np.arange(n_epochs)).values
