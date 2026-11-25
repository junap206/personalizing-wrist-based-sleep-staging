"""
raw_data_service.py 

1. Finds nights - walks through folders, keeps ones with all three files
subject ID and night name from the folder path.
2. Cleans the motion and heart rate CSV's:
- detects whether header
- rename columns to t, x, y, z, and t, bpm
- converts everything to numeric, drops NaN rows
- sorts by time
- renumbers rows
3. Extracts the labels
4. Converts the recording the start time to Unix seconds 

Each night folder contains:
    motion.csv   time, x, y, z acceleration  (~50 Hz)
    hr.csv       time, beats per minute      (~0.2 Hz)
    labels.mat   recStart (local time string) and one label per 30 s epoch
"""

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytz
from scipy.io import loadmat

from sleep_staging.constants import Constants

TIMEZONE = pytz.timezone(Constants.LOCAL_TIMEZONE)
DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%d-%b-%Y %H:%M:%S")


class RawDataService(object):

    # ------------------------------------------------------------------
    # Finding nights
    # ------------------------------------------------------------------
    @staticmethod
    def find_nights(root, subjects=None):
        """
        Every folder holding all three night files is one night.
        Returns a sorted list of (folder, subject, night).
        """
        found = []
        for folder, _, file_names in os.walk(root):
            if not set(Constants.NIGHT_FILES) <= set(file_names):
                continue

            night = os.path.basename(folder)
            subject = os.path.basename(os.path.dirname(folder))
            if subjects and subject not in subjects:
                continue
            found.append((folder, subject, night))

        return sorted(found, key=lambda row: (row[1], row[2]))

    # ------------------------------------------------------------------
    # Loading one night
    # ------------------------------------------------------------------
    @staticmethod
    def load_night(folder, which_label="expert"):
        """Return motion, heart rate, labels, and recording start (Unix s)."""
        motion = RawDataService.read_signal(
            os.path.join(folder, "motion.csv"), ["x", "y", "z"])
        heart_rate = RawDataService.read_signal(
            os.path.join(folder, "hr.csv"), ["bpm"])

        mat = loadmat(os.path.join(folder, "labels.mat"))
        recording_start = RawDataService.parse_recording_start(mat["recStart"])
        labels = RawDataService.get_labels(mat, which_label)
        return motion, heart_rate, labels, recording_start

    @staticmethod
    def get_label_key(mat, which_label):
        """Expert-reviewed labels by default; automatic Dreem labels on request."""
        key = "expert_label" if which_label == "expert" else "dreem_label"
        if key not in mat:
            key = "dreem_label" if "dreem_label" in mat else "expert_label"
        return key

    @staticmethod
    def get_labels(mat, which_label):
        key = RawDataService.get_label_key(mat, which_label)
        return np.asarray(mat[key]).ravel().astype(int)

    @staticmethod
    def read_signal(path, value_columns):
        """Read a signal CSV whether or not it has a header row."""
        first_row = pd.read_csv(path, nrows=1, header=None)
        has_header = not RawDataService.is_number(first_row.iloc[0, 0])

        df = pd.read_csv(path, header=0 if has_header else None)
        df = df.iloc[:, :1 + len(value_columns)]
        df.columns = ["t"] + value_columns
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df.dropna().sort_values("t").reset_index(drop=True)

    @staticmethod
    def is_number(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    @staticmethod
    def parse_recording_start(raw_value):
        """
        recStart is a local-time string such as '2022-03-05 22:45:04'.
        Convert it to Unix seconds so it lines up with the watch timestamps.
        """
        value = RawDataService.unwrap(raw_value)
        if isinstance(value, float):
            return value

        text = str(value).strip()
        local_time = None
        for date_format in DATE_FORMATS:
            try:
                local_time = dt.datetime.strptime(text, date_format)
                break
            except ValueError:
                continue
        if local_time is None:
            raise ValueError("Could not parse recStart: " + repr(text))

        # Times inside a daylight-saving change are ambiguous or do not exist.
        try:
            return TIMEZONE.localize(local_time, is_dst=None).timestamp()
        except pytz.exceptions.AmbiguousTimeError:
            return TIMEZONE.localize(local_time, is_dst=True).timestamp()
        except pytz.exceptions.NonExistentTimeError:
            return TIMEZONE.localize(local_time, is_dst=False).timestamp()

    @staticmethod
    def unwrap(value):
        """MATLAB files nest values inside arrays; dig out the single value."""
        for _ in range(10):
            array = np.asarray(value)
            if array.dtype.kind in "SU":
                return str(array.ravel()[0])
            if array.dtype.kind in "iufc":
                return float(array.ravel()[0])
            if array.dtype == object and array.size:
                value = array.ravel()[0]
                continue
            return str(array.ravel()[0]) if array.size else ""
        return str(value)

    @staticmethod
    def get_epoch_index(timestamps, recording_start):
        """Which 30-second epoch each timestamp falls in."""
        seconds = np.asarray(timestamps) - recording_start
        return np.floor(seconds / Constants.EPOCH_SECONDS).astype(int)

    @staticmethod
    def get_local_hour(recording_start, timestamps):
        """Local clock hour (0-24) of each Unix timestamp in the night."""
        start_local = dt.datetime.fromtimestamp(recording_start, TIMEZONE)
        midnight = TIMEZONE.localize(
            dt.datetime(start_local.year, start_local.month, start_local.day)
        ).timestamp()
        seconds_since_midnight = timestamps - midnight
        return (seconds_since_midnight / 3600.0) % 24.0
