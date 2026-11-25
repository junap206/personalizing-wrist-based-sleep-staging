"""
train_baseline.py

The population model: trained on other people, never on the person it
scores. Evaluated with leave-one-subject-out across all subjects.
Every personalization result is measured against this baseline.

Examples:
    python scripts/training/train_baseline.py                          # features_v2.csv, 3-class
    python scripts/training/train_baseline.py --quick                  # 6 subjects, a fast check
    python scripts/training/train_baseline.py --classes 5 --out-prefix five_class
    python scripts/training/train_baseline.py walch_features.csv --out-prefix walch_baseline

Outputs:
    <prefix>_per_subject.csv   one row per held-out subject
    <prefix>_predictions.csv   every epoch with its prediction
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants
from sleep_staging.dataset_service import DatasetService
from sleep_staging.loso_service import LosoService
from sleep_staging.model_service import ModelService
from sleep_staging.performance_report import PerformanceReport


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("features", nargs="?", default=Constants.FEATURES_FILE)
    parser.add_argument("--classes", type=int, choices=[3, 5], default=3)
    parser.add_argument("--params", default=Constants.PARAMS_FILE)
    parser.add_argument("--quick", action="store_true",
                        help="only 6 subjects, for a fast check")
    parser.add_argument("--out-prefix", default=os.path.join(Constants.BASELINE_DIR, "baseline"))
    return parser.parse_args()


def main():
    args = parse_arguments()

    df, features, label = DatasetService.load(args.features, args.classes)
    params = ModelService.load_params(args.params)
    subjects = DatasetService.get_subjects(df)
    if args.quick:
        subjects = subjects[:Constants.QUICK_BASELINE_SUBJECTS]

    print(f"{len(df):,} epochs | {len(features)} features | "
          f"{df['subject'].nunique()} subjects")
    print(f"Holding out {len(subjects)} subject(s), one at a time.\n")

    start = time.time()
    per_subject, predictions = LosoService.run(df, features, label, subjects, params)
    print(f"\nTotal time: {(time.time() - start) / 60:.1f} min")

    per_subject.to_csv(args.out_prefix + "_per_subject.csv", index=False)
    predictions.to_csv(args.out_prefix + "_predictions.csv", index=False)

    stage_names = DatasetService.get_stage_names(args.classes)
    title = f"BASELINE RESULT  ({args.classes}-class, {len(per_subject)} subjects)"
    PerformanceReport.print_overall(per_subject, title)
    PerformanceReport.print_spread(per_subject)
    PerformanceReport.print_hardest(per_subject)
    PerformanceReport.print_per_class(predictions, stage_names)
    PerformanceReport.print_confusion(predictions, stage_names)
    PerformanceReport.print_published_reference()
    print(f"Wrote {args.out_prefix}_per_subject.csv and {args.out_prefix}_predictions.csv\n")


if __name__ == "__main__":
    main()
