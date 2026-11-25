"""
extract_features.py

Turns raw nights into one table the model can learn from.
Every 30-second epoch becomes one row: 68 feature columns plus the true
sleep stage.

Examples:
    python scripts/features/extract_features.py data --out features_v2.csv
    python scripts/features/extract_features.py data --subjects Bidslab00 Bidslab16
    python scripts/features/extract_features.py data --label auto --out features_auto.csv

Output columns: subject, night, epoch, <68 features>, stage5, stage3
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants
from sleep_staging.context_feature_service import ContextFeatureService
from sleep_staging.feature_table_service import FeatureTableService
from sleep_staging.heart_rate_feature_service import HeartRateFeatureService
from sleep_staging.motion_feature_service import MotionFeatureService
from sleep_staging.raw_data_service import RawDataService
from sleep_staging.time_feature_service import TimeFeatureService

SANITY_COLUMNS = ("hr_zscore", "mag_std", "move_frac", "epochs_since_move")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Builds the feature table.")
    parser.add_argument("root", help="folder holding the subject folders (usually 'data')")
    parser.add_argument("--out", default=os.path.join(Constants.FEATURES_DIR, "features.csv"))
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="limit to these subjects, e.g. --subjects Bidslab00")
    parser.add_argument("--label", choices=["expert", "auto"], default="expert",
                        help="expert-reviewed or automatic Dreem labels")
    return parser.parse_args()


def build_night(folder, subject, night, which_label):
    motion, heart_rate, labels, recording_start = RawDataService.load_night(
        folder, which_label)
    n_epochs = len(labels)

    heart_rate_features = HeartRateFeatureService.build(
        heart_rate, n_epochs, recording_start)
    motion_features = MotionFeatureService.build(motion, n_epochs, recording_start)
    time_features = TimeFeatureService.build(n_epochs, recording_start)

    table = pd.concat([heart_rate_features, motion_features, time_features], axis=1)
    table = ContextFeatureService.add(table)
    return FeatureTableService.add_labels(table, subject, night, labels)


def main():
    args = parse_arguments()

    def build_one(folder, subject, night):
        return build_night(folder, subject, night, args.label)

    table, n_nights, failures, seconds = FeatureTableService.build(
        args.root, args.subjects, build_one)
    table.to_csv(args.out, index=False)

    FeatureTableService.print_summary(table, args.out, n_nights, seconds)
    FeatureTableService.print_class_balance(table)
    FeatureTableService.print_means_by_stage(table, SANITY_COLUMNS, width=18)
    print("\n  Expect: Wake has the highest movement and the highest hr_zscore.")
    print("  If not, something is wrong - stop and check before modelling.")

    n_missing = table.isna().sum().sum()
    print(f"\nMissing values: {n_missing}" + ("  <- should be 0" if n_missing else "  (good)"))
    FeatureTableService.print_failures(failures)
    print()


if __name__ == "__main__":
    main()
