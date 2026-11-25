"""
compare_models.py

Compares models that were scored on the same subjects with
leave-one-subject-out (the *_per_subject.csv files from train_baseline.py).
For each score, every model is compared with the first one by a paired
t-test across subjects. No training.

Examples:
    python scripts/training/train_baseline.py walch_features.csv --out-prefix walch_baseline
    python scripts/training/compare_models.py baseline_per_subject.csv walch_baseline_per_subject.csv \
        --labels "Our model" "Walch features"

Output: model_comparison.csv
"""

import argparse
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402

SCORES = [("macro_f1", "macro-F1"), ("kappa", "kappa"), ("accuracy", "accuracy")]
LINE = "=" * 78


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="per-subject files; the first is the reference")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out", default=os.path.join(Constants.BASELINE_DIR, "model_comparison.csv"))
    return parser.parse_args()


def load(files, labels):
    """One table per model, indexed by subject, keeping only shared subjects."""
    tables = [pd.read_csv(path).set_index("subject") for path in files]
    common = tables[0].index
    for table in tables[1:]:
        common = common.intersection(table.index)
    return {label: table.loc[common] for label, table in zip(labels, tables)}, len(common)


def compare(models):
    labels = list(models)
    reference = models[labels[0]]
    rows = []
    for label in labels:
        for column, name in SCORES:
            scores = models[label][column]
            row = {"model": label, "score": name, "mean": scores.mean(), "sd": scores.std()}
            if label != labels[0]:
                difference = scores - reference[column]
                _, p_value = stats.ttest_rel(scores, reference[column])
                low, high = stats.t.interval(0.95, len(difference) - 1, loc=difference.mean(),
                                             scale=stats.sem(difference))
                row.update({"difference": difference.mean(), "ci_low": low,
                            "ci_high": high, "p": p_value,
                            "n_better": int((difference > 0).sum())})
            rows.append(row)
    return pd.DataFrame(rows)


def print_report(table, n_subjects, reference):
    print("\n" + LINE)
    print(f"MODEL COMPARISON  ({n_subjects} subjects, leave-one-subject-out)")
    print(LINE)
    for _, name in SCORES:
        rows = table[table["score"] == name]
        print(f"\n  {name}")
        print(f"  {'model':<22}{'mean':>8}{'sd':>8}{'vs ' + reference:>22}{'p':>11}{'better':>9}")
        for _, row in rows.iterrows():
            line = f"  {row['model']:<22}{row['mean']:>8.3f}{row['sd']:>8.3f}"
            if pd.notna(row.get("p")):
                interval = f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
                line += (f"   {row['difference']:+.3f} {interval}"
                         f"{row['p']:>11.2e}{int(row['n_better']):>5}/{n_subjects}")
            print(line)
    print("\n  p: paired t-test across subjects against the first model.")
    print("  better: subjects where this model scored higher than the first model.")
    print(LINE + "\n")


def main():
    args = parse_arguments()
    labels = args.labels or args.files
    if len(labels) != len(args.files):
        raise SystemExit("Give one --labels entry per file.")
    models, n_subjects = load(args.files, labels)
    table = compare(models)
    table.to_csv(args.out, index=False)
    print_report(table, n_subjects, labels[0])
    print(f"Wrote {args.out}\n")


if __name__ == "__main__":
    main()