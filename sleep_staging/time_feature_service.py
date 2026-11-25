"""
time_feature_service.py

Where the epoch sits in the night, and in the 24-hour biological day.

minutes_elapsed and frac_through_night say how far into THIS RECORDING we are.
The clock features say where we are in the day: two people who go to bed at
22:00 and 02:00 are both "60 minutes in" at different points of their
circadian cycle.
"""

import numpy as np
import pandas as pd

from sleep_staging.constants import Constants
from sleep_staging.raw_data_service import RawDataService


class TimeFeatureService(object):

    @staticmethod
    def build(n_epochs, recording_start):
        epoch_numbers = np.arange(n_epochs)

        features = pd.DataFrame(index=epoch_numbers)
        features["minutes_elapsed"] = epoch_numbers * Constants.EPOCH_SECONDS / 60.0
        # Needs the end of the recording, so it is dropped in the real-time variant.
        features["frac_through_night"] = epoch_numbers / max(n_epochs - 1, 1)

        epoch_starts = recording_start + epoch_numbers * Constants.EPOCH_SECONDS
        hour_of_day = RawDataService.get_local_hour(recording_start, epoch_starts)
        features["hour_of_day"] = hour_of_day
        features["clock_cosine"] = TimeFeatureService.sleep_drive_cosine(hour_of_day)
        # The sine resolves the cosine's ambiguity: 03:00 and 07:00 share a
        # cosine value but not a sine value.
        features["clock_sine"] = TimeFeatureService.sleep_drive_sine(hour_of_day)

        features["hours_elapsed"] = epoch_numbers * Constants.EPOCH_SECONDS / 3600.0
        return features

    @staticmethod
    def get_phase(hour_of_day):
        return (hour_of_day - Constants.SLEEP_DRIVE_PEAK_HOUR) * 2 * np.pi / 24.0

    @staticmethod
    def sleep_drive_cosine(hour_of_day):
        """Peaks (+1) at SLEEP_DRIVE_PEAK_HOUR, lowest 12 hours later."""
        return np.cos(TimeFeatureService.get_phase(hour_of_day))

    @staticmethod
    def sleep_drive_sine(hour_of_day):
        return np.sin(TimeFeatureService.get_phase(hour_of_day))
