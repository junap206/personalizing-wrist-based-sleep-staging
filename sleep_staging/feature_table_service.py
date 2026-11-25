"""
feature_table_service.py

Runs a feature builder over every night and stacks the results into one
table: one row per 30-second epoch.
"""

import sys
import time

import pandas as pd

from sleep_staging.constants import Constants
from sleep_staging.feature_set_service import FeatureSetService
from sleep_staging.raw_data_service import RawDataService

LINE = "=" * 60


class FeatureTableService(object):

    @staticmethod
    def build(root, subjects, build_night):
        """
        build_night(folder, subject, night) must return one night's table.
        Nights that fail are reported and skipped.
        """
        nights = RawDataService.find_nights(root, subjects)
        if not nights:
            sys.exit("No nights found under " + repr(root) + ". Check the path.")

        n_subjects = len(set(subject for _, subject, _ in nights))
        print(f"Found {len(nights)} nights from {n_subjects} subjects.\n")

        tables = []
        failures = []
        start = time.time()
        for number, (folder, subject, night) in enumerate(nights, 1):
            progress = f"[{number:3d}/{len(nights)}] {subject}/{night}:"
            try:
                table = build_night(folder, subject, night)
                tables.append(table)
                print(f"{progress} {len(table):4d} epochs")
            except Exception as error:
                failures.append((subject, night, str(error)))
                print(f"{progress} FAILED - {error}")

        if not tables:
            sys.exit("Nothing extracted.")

        all_nights = pd.concat(tables, ignore_index=True)
        seconds = time.time() - start
        return all_nights, len(nights), failures, seconds

    @staticmethod
    def add_labels(table, subject, night, labels):
        """Put identifiers first and labels last; drop Unknown epochs."""
        table.insert(0, "epoch", range(len(labels)))
        table.insert(0, "night", night)
        table.insert(0, "subject", subject)
        table["stage5"] = labels
        table["stage3"] = [Constants.FIVE_TO_THREE_CLASS.get(int(s), -1) for s in labels]
        return table[table["stage3"] >= 0].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    @staticmethod
    def print_summary(table, out_path, n_nights, seconds):
        features = FeatureSetService.get_model_features(table.columns)
        print("\n" + LINE)
        print("Wrote " + out_path)
        print(f"  {len(table):,} epochs  x  {len(features)} features")
        print(f"  {table['subject'].nunique()} subjects, {n_nights} nights")
        print(f"  took {seconds:.0f} s")

    @staticmethod
    def print_class_balance(table):
        print("\n3-CLASS BALANCE")
        counts = table["stage3"].value_counts().sort_index()
        for code, count in counts.items():
            name = Constants.STAGE_NAMES[code]
            print(f"  {name:<5}: {count:7,}  ({count / len(table) * 100:5.1f}%)")

    @staticmethod
    def print_means_by_stage(table, columns, width):
        print("\nSANITY CHECK  (mean by stage)")
        for column in columns:
            if column not in table:
                continue
            means = table.groupby("stage3")[column].mean()
            parts = []
            for code, value in means.items():
                parts.append(f"{Constants.STAGE_NAMES[code]} {value:8.4f}")
            print(f"  {column:<{width}} " + " | ".join(parts))

    @staticmethod
    def print_failures(failures):
        if not failures:
            return
        print(f"\n{len(failures)} nights failed:")
        for subject, night, error in failures[:10]:
            print(f"  {subject}/{night}: {error}")
