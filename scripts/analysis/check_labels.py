"""
check_labels.py

Three checks on the labels, for the Methods and Limitations sections.

    A. How often do the automatic and expert labels disagree, and where?
    B. Do the dropped Unknown epochs sit at stage transitions?
    C. Does a model trained on the automatic labels score as well against
       the expert labels as one trained on the expert labels?

Examples:
    python scripts/analysis/check_labels.py
    python scripts/analysis/check_labels.py --skip-train     # parts A and B only (fast)

Needs: data/ folder, features_v2.csv
Outputs: labels_both.csv, labels_expert_trained.csv, labels_auto_trained.csv
"""

import argparse
import os
import sys
import time

from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants
from sleep_staging.dataset_service import DatasetService
from sleep_staging.label_service import LabelService
from sleep_staging.loso_service import LosoService
from sleep_staging.model_service import ModelService

LINE = "=" * 72
FIVE_CLASS_NAMES = dict(Constants.FIVE_CLASS_STAGE_NAMES)
FIVE_CLASS_NAMES[5] = "Unknown"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=Constants.DATA_DIR)
    parser.add_argument("--features", default=Constants.FEATURES_FILE)
    parser.add_argument("--params", default=Constants.PARAMS_FILE)
    parser.add_argument("--skip-train", action="store_true", help="parts A and B only")
    return parser.parse_args()


def print_heading(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


# ----------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------
def report_agreement(labels):
    print_heading("PART A -- AUTOMATIC vs EXPERT LABELS")
    agreement = LabelService.agreement(labels)
    print(f"\n  {agreement['n_epochs']:,} epochs scored by both.")
    print(f"  5-class agreement {agreement['agree5'] * 100:5.1f}%   "
          f"kappa {agreement['kappa5']:.3f}")
    print(f"  3-class agreement {agreement['agree3'] * 100:5.1f}%   "
          f"kappa {agreement['kappa3']:.3f}")
    print(f"  The expert changed {(1 - agreement['agree5']) * 100:.1f}% of epochs.")

    corrections, n_changed = LabelService.top_corrections(labels)
    print("\n  Most common corrections (automatic -> expert):")
    for (automatic, expert), count in corrections.items():
        print(f"    {FIVE_CLASS_NAMES[automatic]:>5} -> {FIVE_CLASS_NAMES[expert]:<5} "
              f"{count:>7,}  ({count / n_changed * 100:4.1f}% of corrections)")

    near = LabelService.corrections_near_transitions(labels)
    print(f"\n  Disagreement within 1 epoch of a transition : {near['rate_near'] * 100:5.1f}%")
    print(f"  Disagreement more than 5 epochs away        : {near['rate_far'] * 100:5.1f}%")
    print(f"  {near['share_corrections_near'] * 100:.0f}% of corrections are within 1 epoch "
          f"of a transition (those epochs are {near['share_epochs_near'] * 100:.0f}% of data).")


# ----------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------
def report_unknown(labels):
    print_heading("PART B -- WHERE ARE THE UNKNOWN EPOCHS?")
    summary = LabelService.unknown_summary(labels)
    print(f"\n  {summary['n_unknown']:,} Unknown epochs "
          f"({summary['share_unknown'] * 100:.2f}% of all).")
    if summary["n_unknown"] == 0:
        return

    print("\n  Within 1 epoch of a stage change:")
    print(f"    Unknown epochs : {summary['unknown_near'] * 100:5.1f}%")
    print(f"    scored epochs  : {summary['scored_near'] * 100:5.1f}%")

    patterns = LabelService.neighbour_patterns(labels)
    print("\n  Epochs with a scored neighbour on both sides: how often are the two")
    print("  neighbours DIFFERENT stages (i.e. the epoch sits on a boundary)?")
    print(f"    Unknown epochs                : "
          f"{patterns['unknown_between_different'] * 100:5.1f}%  "
          f"(n={patterns['n_unknown_bounded']})")
    print(f"    ordinary scored epochs (chance): "
          f"{patterns['chance_between_different'] * 100:5.1f}%")
    print(f"\n  Per-subject Unknown rate: median {summary['median_subject_pct']:.2f}%, "
          f"max {summary['max_subject_pct']:.2f}% ({summary['max_subject']})")


# ----------------------------------------------------------------------
# Part C
# ----------------------------------------------------------------------
def report_training(features_file, labels, params):
    print_heading("PART C -- TRAIN ON AUTOMATIC LABELS, SCORE ON EXPERT LABELS")
    df, features, _ = DatasetService.load(features_file)
    df["night"] = df["night"].astype(str)
    labels = labels.copy()
    labels["night"] = labels["night"].astype(str)
    df = df.merge(labels[["subject", "night", "epoch", "auto3", "expert3"]],
                  on=["subject", "night", "epoch"], how="inner")
    subjects = DatasetService.get_subjects(df)
    print(f"\n  {len(df):,} epochs joined. Running leave-one-subject-out twice...")

    start = time.time()
    expert_trained, _ = LosoService.run(df, features, "expert3", subjects, params,
                                        verbose=False, eval_label="expert3")
    automatic_trained, _ = LosoService.run(df, features, "auto3", subjects, params,
                                           verbose=False, eval_label="expert3")
    print(f"  done ({(time.time() - start) / 60:.1f} min)")

    print(f"\n  {'trained on':<14}{'macro-F1':>10}{'kappa':>8}{'accuracy':>10}")
    print("  " + "-" * 42)
    for name, result in (("expert", expert_trained), ("automatic", automatic_trained)):
        print(f"  {name:<14}{result['macro_f1'].mean():>10.3f}"
              f"{result['kappa'].mean():>8.3f}{result['accuracy'].mean():>10.3f}")

    expert_scores = expert_trained.set_index("subject")["macro_f1"]
    automatic_scores = automatic_trained.set_index("subject")["macro_f1"]
    common = expert_scores.index.intersection(automatic_scores.index)
    difference = expert_scores[common] - automatic_scores[common]
    _, p_value = stats.ttest_rel(expert_scores[common], automatic_scores[common])
    margin = 1.96 * difference.sem()
    print(f"\n  Expert-trained minus automatic-trained: {difference.mean():+.3f} "
          f"(95% CI {difference.mean() - margin:+.3f} to {difference.mean() + margin:+.3f}), "
          f"paired p = {p_value:.4f}")

    expert_trained.to_csv(os.path.join(Constants.ANALYSIS_DIR, "labels_expert_trained.csv"), index=False)
    automatic_trained.to_csv(os.path.join(Constants.ANALYSIS_DIR, "labels_auto_trained.csv"), index=False)


def main():
    args = parse_arguments()
    print("Loading both label sets from the raw files...")
    labels = LabelService.load_both(args.data)
    labels.to_csv(os.path.join(Constants.ANALYSIS_DIR, "labels_both.csv"), index=False)
    print(f"  {len(labels):,} epochs, {labels['subject'].nunique()} subjects")

    report_agreement(labels)
    report_unknown(labels)
    if not args.skip_train:
        report_training(args.features, labels, ModelService.load_params(args.params))
    print(LINE + "\n")


if __name__ == "__main__":
    main()
