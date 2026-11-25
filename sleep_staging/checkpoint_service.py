"""
checkpoint_service.py

Saves results after every subject so a long run can be stopped and resumed.
Refuses to resume if the settings changed, so results are never mixed.
"""

import json
import os

import pandas as pd


class CheckpointService(object):

    @staticmethod
    def get_partial_path(prefix):
        return prefix + "_partial.csv"

    @staticmethod
    def get_config_path(prefix):
        return prefix + "_config.json"

    @staticmethod
    def save_config(prefix, config):
        """
        Write the run settings. If a partial run exists with different
        settings, stop instead of mixing the two.
        """
        partial_path = CheckpointService.get_partial_path(prefix)
        config_path = CheckpointService.get_config_path(prefix)

        if os.path.exists(partial_path) and os.path.exists(config_path):
            with open(config_path) as file:
                previous = json.load(file)
            if not CheckpointService.same_config(previous, config):
                raise SystemExit("\n  STOP: " + partial_path
                                 + " was made with DIFFERENT settings.\n"
                                 + "  Delete it to start over, or use a new --tag.\n")

        with open(config_path, "w") as file:
            json.dump(config, file, indent=2, default=str)

    @staticmethod
    def same_config(first, second):
        first_text = json.dumps(first, sort_keys=True, default=str)
        second_text = json.dumps(second, sort_keys=True, default=str)
        return first_text == second_text

    @staticmethod
    def finished_subjects(prefix):
        partial_path = CheckpointService.get_partial_path(prefix)
        if not os.path.exists(partial_path):
            return set()
        finished = pd.read_csv(partial_path)["subject"].unique()
        return set(finished)

    @staticmethod
    def append(prefix, rows):
        """Add one subject's rows to the partial file."""
        partial_path = CheckpointService.get_partial_path(prefix)
        write_header = not os.path.exists(partial_path)
        pd.DataFrame(rows).to_csv(partial_path, mode="a", index=False,
                                  header=write_header)

    @staticmethod
    def load(prefix):
        return pd.read_csv(CheckpointService.get_partial_path(prefix))