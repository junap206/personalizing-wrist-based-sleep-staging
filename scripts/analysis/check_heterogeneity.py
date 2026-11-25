"""
check_heterogeneity.py

Does personalization help the subjects the population model scores worst,
or is that regression to the mean? Compares how improvement depends on the
K=0 score in the personal arm vs the stranger arm.

Examples:
    python scripts/analysis/check_heterogeneity.py
    python scripts/analysis/check_heterogeneity.py --k 2
    python scripts/analysis/check_heterogeneity.py --results personalization_v2_final_realtime_per_subject.csv

Needs: the personalization per-subject file (default: all features), and
optionally night_scores.csv and subject_profile.csv for the moderator check.
Outputs: heterogeneity_table.csv, heterogeneity.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; 
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.heterogeneity_service import HeterogeneityService  # noqa: E402
from sleep_staging.personalization_statistics import PersonalizationStatistics  # noqa: E402

LINE = "=" * 72
ARM_COLORS = {"personal": "#27ae60", "stranger": "#c0392b"}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=Constants.PERSONALIZATION_PER_SUBJECT_FILE)
    parser.add_argument("--k", type=int, default=Constants.MAX_PERSONAL_NIGHTS)
    parser.add_argument("--scores", default=Constants.NIGHT_SCORES_FILE)
    parser.add_argument("--profile", default=Constants.SUBJECT_PROFILE_FILE)
    return parser.parse_args()


def load_per_subject(path):
    """Accept a per-subject file, or a raw results file (averaged over donors here)."""
    results = pd.read_csv(path)
    if "donor" in results.columns:
        print("  (one row per donor found: averaging over donors first)")
        results = PersonalizationStatistics.average_over_donors(results)
    return results


def load_optional(path, index=None):
    if not os.path.exists(path):
        print(f"  ({path} not found: skipping it in the moderator check)")
        return None
    table = pd.read_csv(path)
    return table.set_index(index) if index else table


def print_heading(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def print_report(changes, k, slopes, interaction, paired, halves, moderators):
    print_heading(f"DOES PERSONALIZATION HELP HARD SUBJECTS MORE?  (K={k})")
    print("\n  Change from K=0, regressed on the K=0 score, per arm:\n")
    print(f"  {'arm':<10}{'n':>4}{'slope':>9}{'r':>8}{'p':>10}")
    print("  " + "-" * 42)
    for arm in Constants.ARMS:
        fit = slopes[arm]
        print(f"  {arm:<10}{fit['n']:>4}{fit['slope']:>9.3f}{fit['r']:>8.3f}{fit['p']:>10.4f}")
    print("\n  A negative slope means low-baseline subjects gained more.")
    print("  Regression to the mean produces that in BOTH arms.")

    slope_difference, paired_p = paired
    coefficient, interaction_p = interaction
    print_heading("THE TEST THAT MATTERS: does the slope differ between arms?")
    print(f"\n  slope difference (personal - stranger) = {slope_difference:+.3f}")
    print(f"  paired test (gap regressed on baseline, one row per subject): p = {paired_p:.4f}")
    print(f"  unpaired interaction term (baseline x arm) = {coefficient:+.3f}, "
          f"p = {interaction_p:.4f}")
    if paired_p < Constants.SIGNIFICANCE_LEVEL:
        direction = "HARD" if slope_difference < 0 else "EASY"
        print("\n  -> The personal arm's benefit depends on baseline beyond regression")
        print(f"     to the mean: it helps {direction} subjects more.")
    else:
        print("\n  -> No reliable difference: the baseline dependence is the same in")
        print("     both arms, i.e. regression to the mean. The effect is uniform.")

    hard, easy, welch_p = halves
    print_heading("SAME QUESTION, SIMPLER FORM")
    print("\n  Personal-minus-stranger gap, subjects split at the median K=0 score:\n")
    print(f"  harder half: gap {hard.mean():+.3f}  (n={len(hard)})")
    print(f"  easier half: gap {easy.mean():+.3f}  (n={len(easy)})")
    print(f"  difference : {hard.mean() - easy.mean():+.3f}   Welch p = {welch_p:.4f}")

    if moderators is not None:
        print_heading("OTHER CANDIDATE MODERATORS  (exploratory, uncorrected)")
        print(f"\n  {'property':<24}{'r':>8}{'p':>10}")
        print("  " + "-" * 42)
        for _, row in moderators.iterrows():
            print(f"  {row['property']:<24}{row['r']:>8.3f}{row['p']:>10.4f}")
    print(LINE + "\n")


def plot_heterogeneity(changes, k, slopes, paired, path=os.path.join(Constants.FIGURES_DIR, "heterogeneity.png")):
    fig, axis = plt.subplots(figsize=(8, 5.5))
    for arm in Constants.ARMS:
        rows = changes[changes["arm"] == arm]
        fit = slopes[arm]
        axis.scatter(rows["baseline"], rows["delta"], s=28, alpha=0.7,
                     color=ARM_COLORS[arm], label=f"{arm} (n={len(rows)})")
        x = np.linspace(rows["baseline"].min(), rows["baseline"].max(), 50)
        axis.plot(x, fit["intercept"] + fit["slope"] * x, color=ARM_COLORS[arm], lw=2,
                  label=f"{arm}: slope {fit['slope']:+.2f}")

    axis.axhline(0, color="0.5", lw=0.8)
    axis.set_xlabel("macro-F1 at K=0 (population model)")
    axis.set_ylabel(f"change in macro-F1 at K={k}")
    axis.set_title("Same slope in both arms = regression to the mean.\n"
                   f"Slope difference {paired[0]:+.2f}, paired p = {paired[1]:.3g}",
                   fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote " + path)


def main():
    args = parse_arguments()
    print("Loading " + args.results)
    per_subject = load_per_subject(args.results)
    available = sorted(per_subject["k"].unique())
    if args.k not in available:
        raise SystemExit(f"K={args.k} is not in the results; available: {available}")

    changes = HeterogeneityService.build_change_table(per_subject, args.k)
    changes.to_csv(os.path.join(Constants.ANALYSIS_DIR, "heterogeneity_table.csv"), index=False)

    slopes = {arm: HeterogeneityService.arm_slope(changes, arm) for arm in Constants.ARMS}
    interaction = HeterogeneityService.interaction_test(changes)
    paired = HeterogeneityService.paired_slope_test(changes)
    halves = HeterogeneityService.split_at_median(changes)
    moderators = HeterogeneityService.test_moderators(
        changes, load_optional(args.scores), load_optional(args.profile, index="subject"))

    print_report(changes, args.k, slopes, interaction, paired, halves, moderators)
    plot_heterogeneity(changes, args.k, slopes, paired)
    print("Wrote heterogeneity_table.csv\n")


if __name__ == "__main__":
    main()
