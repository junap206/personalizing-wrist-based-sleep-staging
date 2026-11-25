"""
sleep_architecture_service.py

Describes the shape of one night from its sequence of true stages:
how much of each stage, and how fragmented it was.
"""

import numpy as np

from sleep_staging.constants import Constants

WAKE = 0
NREM = 1
REM = 2


class SleepArchitectureService(object):

    @staticmethod
    def count_transitions(stages):
        """Number of times the stage changes from one epoch to the next."""
        return int((np.diff(stages) != 0).sum())

    @staticmethod
    def longest_run(stages):
        """Length of the longest unbroken stretch in a single stage."""
        change_points = np.where(np.diff(stages) != 0)[0] + 1
        runs = np.split(stages, change_points)
        return max(len(list(run)) for run in runs)

    @staticmethod
    def get_hours(n_epochs):
        return n_epochs * Constants.EPOCH_SECONDS / 3600

    @staticmethod
    def describe(stages):
        """Stage fractions and fragmentation for one night (stages in epoch order)."""
        n_epochs = len(stages)
        hours = SleepArchitectureService.get_hours(n_epochs)
        transitions = SleepArchitectureService.count_transitions(stages)
        return {
            "n_epochs": n_epochs,
            "hours": hours,
            "frac_wake": (stages == WAKE).mean(),
            "frac_nrem": (stages == NREM).mean(),
            "frac_rem": (stages == REM).mean(),
            "n_transitions": transitions,
            "transitions_per_hour": transitions / hours,
            "longest_run": SleepArchitectureService.longest_run(stages),
        }

    @staticmethod
    def get_stages(night_epochs):
        """True three-class stages of one night, in epoch order."""
        return night_epochs.sort_values("epoch")[Constants.THREE_CLASS_LABEL].values
