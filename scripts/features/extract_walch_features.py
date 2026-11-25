"""
extract_walch_features.py

Builds the four Walch et al. (2019) features for every epoch, so their
feature set can be compared with ours in the same pipeline.

Examples:
    python scripts/features/extract_walch_features.py data --out walch_features.csv
    python scripts/features/extract_walch_features.py data --subjects Bidslab00

Output columns: subject, night, epoch, walch_activity, walch_hr,
                walch_cosine, walch_time, stage5, stage3
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants
from sleep_staging.feature_set_service import FeatureSetService
from sleep_staging.feature_table_service import FeatureTableService
from sleep_staging.raw_data_service import RawDataService
from sleep_staging.walch_feature_service import WalchFeatureService


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract Walch-style features.")
    parser.add_argument("root", help="folder holding the subject folders (usually 'data')")
    parser.add_argument("--out", default=Constants.WALCH_FEATURES_FILE)
    parser.add_argument("--subjects", nargs="*", default=None)
    return parser.parse_args()


def build_night(folder, subject, night):
    motion, heart_rate, labels, recording_start = RawDataService.load_night(folder)
    n_epochs = len(labels)

    table = pd.DataFrame({
        "walch_activity": WalchFeatureService.activity_counts(
            motion, recording_start, n_epochs),
        "walch_hr": WalchFeatureService.heart_rate_variability(
            heart_rate, recording_start, n_epochs),
        "walch_cosine": WalchFeatureService.circadian_cosine(recording_start, n_epochs),
        "walch_time": WalchFeatureService.elapsed_hours(n_epochs),
    })
    return FeatureTableService.add_labels(table, subject, night, labels)


def print_feature_ranges(table):
    print("\nFEATURE RANGES")
    for column in FeatureSetService.get_model_features(table.columns):
        values = table[column]
        print(f"  {column:<20} {values.min():>10.4f}  to  {values.max():.4f}"
              f"   mean {values.mean():.4f}")


def main():
    args = parse_arguments()

    table, n_nights, failures, seconds = FeatureTableService.build(
        args.root, args.subjects, build_night)
    table.to_csv(args.out, index=False)

    FeatureTableService.print_summary(table, args.out, n_nights, seconds)
    print_feature_ranges(table)
    FeatureTableService.print_class_balance(table)
    walch_columns = FeatureSetService.get_model_features(table.columns)
    FeatureTableService.print_means_by_stage(table, walch_columns, width=20)
    print("\n  walch_hr is heart-rate VARIABILITY: highest in Wake, lowest in NREM.")
    print("  walch_activity should be highest in Wake.")
    FeatureTableService.print_failures(failures)
    print()


if __name__ == "__main__":
    main()