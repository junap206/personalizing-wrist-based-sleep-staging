"""
run_2x2.py

Do personal-reference features help the model use a subject's own nights?

    rows    : feature set (all features vs. no personal-reference features)
    columns : K (personal nights in training)

For each subject and feature set:
    training benefit = personal-arm macro-F1 at K=3 minus at K=start
    controlled gap   = personal minus stranger at K=3 (same amount of data)

Each is compared between the two feature sets with a paired t-test across
subjects. A larger benefit with personal-reference features is evidence
that they help the model use subject-specific data.

start defaults to K=1: at K=0 the personal-reference heart-rate features
fall back to the population median, so they are not yet personal.

Examples:
    python scripts/experiments/run_2x2.py
    python scripts/experiments/run_2x2.py --from-k 0

Reads the per-subject files from run_all_personalization.bat (same donors
and model settings for both feature sets).
Outputs: two_by_two.csv, two_by_two.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.figure_service import FigureService  # noqa: E402
from sleep_staging.personalization_statistics import PersonalizationStatistics  # noqa: E402

SETS = [("With personal-reference", os.path.join(Constants.PERSONALIZATION_DIR, "personalization_v2_final_all_per_subject.csv"), "#27ae60"),
        ("Without personal-reference", os.path.join(Constants.PERSONALIZATION_DIR, "personalization_v2_final_nonpersonal_per_subject.csv"),
         "#3498db")]
END_K = 3
LINE = "=" * 70


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-personal", default=SETS[0][1])
    parser.add_argument("--without-personal", default=SETS[1][1])
    parser.add_argument("--from-k", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--out", default=os.path.join(Constants.FIGURES_DIR, "two_by_two.png"))
    return parser.parse_args()


def get_scores(per_subject, arm, k):
    return PersonalizationStatistics.get_arm_scores(per_subject, arm, k)


def subject_measures(per_subject, from_k):
    """Per subject: scores at each K, training benefit, and controlled gap."""
    measures = pd.DataFrame({
        f"K{from_k}": get_scores(per_subject, "personal", from_k),
        f"K{END_K}": get_scores(per_subject, "personal", END_K),
        "stranger": get_scores(per_subject, "stranger", END_K),
    }).dropna()
    measures["benefit"] = measures[f"K{END_K}"] - measures[f"K{from_k}"]
    measures["gap"] = measures[f"K{END_K}"] - measures["stranger"]
    return measures


def test(values):
    """Is the mean of per-subject values different from zero?"""
    return PersonalizationStatistics.paired_test(values, np.zeros(len(values)))


def print_report(with_personal, without_personal, from_k):
    print("\n" + LINE)
    print(f"THE 2x2  (personal arm, mean macro-F1, {len(with_personal)} subjects)")
    print(LINE)
    print(f"\n  {'':30}{f'K={from_k}':>10}{f'K={END_K}':>10}{'benefit':>11}")
    for label, measures in (("With personal-reference", with_personal),
                            ("Without personal-reference", without_personal)):
        print(f"  {label:<30}{measures[f'K{from_k}'].mean():>10.4f}"
              f"{measures[f'K{END_K}'].mean():>10.4f}{measures['benefit'].mean():>+11.4f}")

    for column, title in (("benefit", f"TRAINING BENEFIT (K={from_k} -> K={END_K})"),
                          ("gap", f"CONTROLLED GAP (personal - stranger at K={END_K})")):
        print("\n" + LINE)
        print(title)
        print(LINE)
        for label, measures in (("with personal-reference", with_personal),
                                ("without personal-reference", without_personal)):
            result = test(measures[column].values)
            print(f"  {label:<28} {result['gap']:+.4f}   p = {result['t_p']:.4f}")
        difference = PersonalizationStatistics.paired_test(
            with_personal[column].values, without_personal[column].values)
        print(f"  difference (with - without) {difference['gap']:+.4f}   "
              f"95% CI [{difference['ci_lo']:+.4f}, {difference['ci_hi']:+.4f}]   "
              f"paired p = {difference['t_p']:.4f}")
    print(LINE + "\n")


def plot(with_personal, without_personal, from_k, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    colors = [SETS[0][2], SETS[1][2]]

    # Absolute scores, all four cells
    groups = [with_personal[f"K{from_k}"].values, with_personal[f"K{END_K}"].values,
              without_personal[f"K{from_k}"].values, without_personal[f"K{END_K}"].values]
    labels = [f"with\nK={from_k}", f"with\nK={END_K}", f"without\nK={from_k}",
              f"without\nK={END_K}"]
    FigureService.boxplot_with_points(axes[0], groups, labels, [colors[0]] * 2 + [colors[1]] * 2)
    axes[0].set_ylabel("macro-F1 (personal arm)")
    axes[0].set_title("Absolute scores\nwith / without personal-reference features",
                      fontsize=10)

    # Training benefit per subject, each tested against zero
    benefits = [with_personal["benefit"].values, without_personal["benefit"].values]
    FigureService.boxplot_with_points(axes[1], benefits, ["with", "without"], colors)
    axes[1].axhline(0, color="#e74c3c", lw=1, ls="--")
    FigureService.label_p_values(axes[1], benefits, [test(b)["t_p"] for b in benefits])
    difference = PersonalizationStatistics.paired_test(benefits[0], benefits[1])
    axes[1].set_ylabel(f"benefit: K={END_K} minus K={from_k}")
    axes[1].set_title(f"Training benefit (each vs 0)\nwith vs without: paired p = "
                      f"{difference['t_p']:.3g}", fontsize=10)

    # Same benefit, subject by subject
    axes[2].scatter(benefits[1], benefits[0], s=25, alpha=0.7, color="#2c3e50")
    limit = np.abs(np.concatenate(benefits)).max() * 1.1
    axes[2].plot([-limit, limit], [-limit, limit], color="#e74c3c", ls="--", lw=1)
    axes[2].set_xlabel("benefit WITHOUT personal-reference features")
    axes[2].set_ylabel("benefit WITH personal-reference features")
    axes[2].set_title("Each dot is one subject; above the line = more benefit with them.\n"
                      f"p = {difference['t_p']:.3g} (paired t-test, not a correlation)",
                      fontsize=10)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote " + path)


def main():
    args = parse_arguments()
    with_results = pd.read_csv(args.with_personal)
    without_results = pd.read_csv(args.without_personal)
    with_personal = subject_measures(with_results, args.from_k)
    without_personal = subject_measures(without_results, args.from_k)

    common = with_personal.index.intersection(without_personal.index)
    with_personal = with_personal.loc[common]
    without_personal = without_personal.loc[common]

    table = with_personal.add_prefix("with_").join(without_personal.add_prefix("without_"))
    table.to_csv(os.path.join(Constants.PERSONALIZATION_DIR, "two_by_two.csv"))
    print_report(with_personal, without_personal, args.from_k)
    plot(with_personal, without_personal, args.from_k, args.out)


if __name__ == "__main__":
    main()
