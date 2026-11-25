"""
walch_feature_service.py

The four features of Walch et al. (2019), reimplemented from their
sleep_classifiers repository so they can be evaluated in our pipeline.

    walch_activity  band-passed accelerometer magnitude, Gaussian-smoothed
                    over a ~10 minute window
    walch_hr        heart-rate VARIABILITY: standard deviation of the
                    difference-of-Gaussians filtered heart rate in the window
    walch_cosine    circadian sleep-drive cosine
    walch_time      hours since the recording started

Both signals are resampled to 1 Hz before windowing, as in their code.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from sleep_staging.constants import Constants
from sleep_staging.raw_data_service import RawDataService


class WalchFeatureService(object):

    # ------------------------------------------------------------------
    # Filters copied from their utils.py
    # ------------------------------------------------------------------
    @staticmethod
    def smooth_gauss(values, n_points):
        """Gaussian-weighted average of a window (their smooth_gauss)."""
        weights = np.ones(n_points)
        center = int(n_points / 2.0)
        for i in range(n_points):
            weights[i] = np.exp(-0.5 * (((i - center) / Constants.WALCH_GAUSS_SIGMA) ** 2))
        weights = weights / np.sum(weights)
        return float(np.sum(weights * values[:n_points]))

    @staticmethod
    def convolve_with_dog(values, n_points):
        """Difference-of-Gaussians filter (their convolve_with_dog)."""
        values = values - np.mean(values)

        kernel = np.ones(n_points) / n_points
        center = int(n_points / 2.0)
        for i in range(n_points):
            narrow = np.exp(-0.5 * (((i - center) / Constants.WALCH_DOG_SIGMA_NARROW) ** 2))
            wide = np.exp(-0.5 * (((i - center) / Constants.WALCH_DOG_SIGMA_WIDE) ** 2))
            kernel[i] = narrow - Constants.WALCH_DOG_SCALAR * wide

        # Pad both ends by mirroring, so the output keeps its length.
        half = int(n_points / 2)
        values = np.insert(values, 0, np.flip(values[0:half]))
        values = np.insert(values, len(values) - 1, np.flip(values[-half:]))
        return np.convolve(values, kernel, mode="valid")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_one_hz_times(recording_start, n_epochs):
        end = recording_start + n_epochs * Constants.EPOCH_SECONDS
        return np.arange(recording_start, end, 1.0)

    @staticmethod
    def get_window(one_hz_times, epoch_start):
        """Indices of the ~10 minute window centred on one epoch."""
        center = epoch_start + Constants.EPOCH_SECONDS / 2
        start = center - Constants.WALCH_WINDOW_SECONDS
        end = center + Constants.WALCH_WINDOW_SECONDS
        inside = (one_hz_times > start) & (one_hz_times < end)
        return np.where(inside)[0]

    @staticmethod
    def get_epoch_start(recording_start, epoch):
        return recording_start + epoch * Constants.EPOCH_SECONDS

    # ------------------------------------------------------------------
    # The four features
    # ------------------------------------------------------------------
    @staticmethod
    def activity_counts(motion, recording_start, n_epochs):
        magnitude = np.sqrt(motion["x"] ** 2 + motion["y"] ** 2 + motion["z"] ** 2)
        activity = np.abs(WalchFeatureService.bandpass(magnitude.values))

        one_hz_times = WalchFeatureService.get_one_hz_times(recording_start, n_epochs)
        one_hz_activity = np.interp(one_hz_times, motion["t"].values, activity)

        counts = np.zeros(n_epochs)
        for epoch in range(n_epochs):
            epoch_start = WalchFeatureService.get_epoch_start(recording_start, epoch)
            window = WalchFeatureService.get_window(one_hz_times, epoch_start)
            if len(window) > 0:
                values = one_hz_activity[window]
                counts[epoch] = WalchFeatureService.smooth_gauss(values, len(values))
        return counts

    @staticmethod
    def bandpass(magnitude):
        """Keep human-movement frequencies (0.25-3 Hz)."""
        nyquist = Constants.ACCELEROMETER_HZ / 2
        if nyquist <= Constants.WALCH_BANDPASS_HIGH_HZ:
            return magnitude
        try:
            low = Constants.WALCH_BANDPASS_LOW_HZ / nyquist
            high = Constants.WALCH_BANDPASS_HIGH_HZ / nyquist
            b, a = butter(4, [low, high], btype="band")
            return filtfilt(b, a, magnitude)
        except Exception:
            return magnitude

    @staticmethod
    def heart_rate_variability(heart_rate, recording_start, n_epochs):
        one_hz_times = WalchFeatureService.get_one_hz_times(recording_start, n_epochs)
        one_hz_bpm = np.interp(one_hz_times, heart_rate["t"].values,
                               heart_rate["bpm"].values)
        filtered = WalchFeatureService.filter_heart_rate(one_hz_bpm, len(one_hz_times))

        variability = np.zeros(n_epochs)
        for epoch in range(n_epochs):
            epoch_start = WalchFeatureService.get_epoch_start(recording_start, epoch)
            window = WalchFeatureService.get_window(one_hz_times, epoch_start)
            if len(window) > 0:
                variability[epoch] = np.std(filtered[window])
        return variability

    @staticmethod
    def filter_heart_rate(one_hz_bpm, n_samples):
        """Difference-of-Gaussians filter, then scale by the 90th percentile."""
        kernel_length = min(Constants.WALCH_WINDOW_SECONDS, len(one_hz_bpm))
        if kernel_length > 10:
            filtered = WalchFeatureService.convolve_with_dog(one_hz_bpm, kernel_length)
            filtered = WalchFeatureService.match_length(filtered, n_samples)
        else:
            filtered = one_hz_bpm - np.mean(one_hz_bpm)

        scale = np.percentile(np.abs(filtered), 90)
        if scale > 0:
            filtered = filtered / scale
        return filtered

    @staticmethod
    def match_length(values, n_samples):
        if len(values) > n_samples:
            return values[:n_samples]
        if len(values) < n_samples:
            return np.pad(values, (0, n_samples - len(values)), mode="edge")
        return values

    @staticmethod
    def circadian_cosine(recording_start, n_epochs):
        """Sleep-drive cosine evaluated at the centre of each epoch."""
        epoch_centers = (recording_start
                         + np.arange(n_epochs) * Constants.EPOCH_SECONDS
                         + Constants.EPOCH_SECONDS / 2)
        hour_of_day = RawDataService.get_local_hour(recording_start, epoch_centers)
        phase = (hour_of_day - Constants.SLEEP_DRIVE_PEAK_HOUR) * 2 * np.pi / 24.0
        return np.cos(phase)

    @staticmethod
    def elapsed_hours(n_epochs):
        return np.arange(n_epochs) * Constants.EPOCH_SECONDS / 3600.0
