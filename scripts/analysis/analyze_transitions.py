"""
analyze_transitions.py

Are the model's errors concentrated near sleep-stage transitions?

    1. Does the error rate rise as an epoch gets closer to a stage change?
    2. Which kinds of change (Wake->NREM, NREM->REM, ...) are hardest?
    3. Within one subject, do nights with more changes score worse?

Needs three-class predictions (Wake / NREM / REM) from train_baseline.py.

Example:
    python scripts/analysis/analyze_transitions.py --plot

Outputs:
    transition_analysis.csv   error rate per distance bin
    transition_by_type.csv    error rate per type of transition
    transition_nights.csv     one row per night
    transitions.png           (with --plot)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.figure_service import FigureService  # noqa: E402
from sleep_staging.night_score_service import NightScoreService  # noqa: E402
from sleep_staging.transition_service import TransitionService  # noqa: E402

LINE = "=" * 72


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=Constants.PREDICTIONS_FILE,
                        help="three-class predictions from train_baseline.py")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", default=os.path.join(Constants.ANALYSIS_DIR, "transition_analysis.csv"))
    return parser.parse_args()


def print_heading(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def print_report(summary, transition_test, by_type, concentration, correlation):
    print_heading("ERROR RATE BY DISTANCE TO NEAREST TRUE TRANSITION")
    print("\n  Per-subject error rates, averaged across subjects.\n")
    print(f"  {'distance':<16}{'error':>8}{'sd':>8}{'n subj':>8}{'% of epochs':>13}")
    print("  " + "-" * 55)
    for name, row in summary.iterrows():
        print(f"  {name:<16}{row['error_rate']:>8.3f}{row['sd_across_subjects']:>8.3f}"
              f"{int(row['n_subjects']):>8}{row['share_of_epochs'] * 100:>12.1f}%")

    at_change = summary.loc["at transition", "error_rate"]
    stable = summary.loc[">10 (stable)", "error_rate"]
    p_value, n_subjects = transition_test
    print(f"\n  At transitions vs stable stretches: {at_change:.3f} vs {stable:.3f}  "
          f"({at_change / stable:.1f}x)")
    print(f"  Paired t-test across {n_subjects} subjects: p = {p_value:.2e}")

    print_heading("WHERE DO THE ERRORS LIVE?")
    share_epochs = concentration["share_epochs_near"]
    share_errors = concentration["share_errors_near"]
    print(f"\n  Epochs within 1 of a transition : {share_epochs * 100:5.1f}% of all epochs")
    print(f"  Errors within 1 of a transition : {share_errors * 100:5.1f}% of all errors")
    print(f"  Errors are {share_errors / share_epochs:.2f}x over-represented near transitions.")

    print_heading("BY TRANSITION TYPE  (error rate at the transition epochs)")
    print(f"\n  {'transition':<16}{'error':>8}{'sd':>8}{'n subj':>8}{'n epochs':>10}")
    print("  " + "-" * 52)
    for name, row in by_type.iterrows():
        print(f"  {name:<16}{row['error_rate']:>8.3f}{row['sd']:>8.3f}"
              f"{int(row['n_subjects']):>8}{int(row['n_epochs']):>10}")

    nights, r, p_value, p_adjusted = correlation
    print_heading("DOES FRAGMENTATION EXPLAIN WHICH NIGHTS ARE HARD?")
    print("\n  Within-subject correlation between a night's transition-adjacent")
    print("  fraction and its macro-F1 (each subject's own mean removed):")
    print(f"    r = {r:+.3f}   p = {p_value:.2e}   ({len(nights)} nights)")
    print(f"    p with degrees of freedom adjusted for subject means = {p_adjusted:.2e}")
    print(LINE + "\n")


def plot_transitions(summary, per_subject, transition_test, by_type, correlation,
                     path=os.path.join(Constants.FIGURES_DIR, "transitions.png")):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Panel 1: per-subject error rate by distance to a transition
    bins = list(summary.index)
    groups = [per_subject[name].dropna().values for name in bins]
    colors = ["#c0392b"] + ["#2980b9"] * (len(bins) - 1)
    FigureService.boxplot_with_points(axes[0], groups, bins, colors)
    p_values = [transition_test[0]] + [None] * (len(bins) - 1)
    FigureService.label_p_values(axes[0], groups, p_values)
    axes[0].set_ylabel("error rate per each subject")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Error vs distance to the nearest transition\n"
                      "p: paired t-test, at transition vs stable (>10)", fontsize=10)

    # Panel 2: error rate by type of transition
    top = by_type.head(8).iloc[::-1]
    bars = axes[1].barh(top.index, top["error_rate"], color="#c0392b")
    for bar, value in zip(bars, top["error_rate"]):
        axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{value:.2f}", va="center", fontsize=8)
    axes[1].set_xlim(0, 1.1)
    axes[1].set_xlabel("error rate at transition epochs")
    axes[1].set_title("By transition type", fontsize=10)
    axes[1].grid(alpha=0.25, axis="x")

    # Panel 3: within-subject fragmentation vs performance
    nights, r, _, p_adjusted = correlation
    axes[2].scatter(nights["frac_near_transition_c"], nights["macro_f1_c"],
                    s=16, alpha=0.6, color="#27ae60")
    axes[2].axhline(0, color="0.6", lw=0.7)
    axes[2].axvline(0, color="0.6", lw=0.7)
    axes[2].text(0.05, 0.95, f"r = {r:.3f}\np = {p_adjusted:.1e}\nn = {len(nights)} nights",
                 transform=axes[2].transAxes, va="top", fontsize=10,
                 bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8})
    axes[2].set_xlabel("transition-adjacent fraction (subject-centred)")
    axes[2].set_ylabel("night macro-F1 (subject-centred)")
    axes[2].set_title("Within subject: fragmentation vs performance", fontsize=10)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote " + path)


def main():
    args = parse_arguments()
    predictions = NightScoreService.load_predictions(args.pred)
    print(f"Loaded {len(predictions):,} scored epochs from "
          f"{predictions['subject'].nunique()} subjects ({args.pred}).")

    epochs = TransitionService.annotate(predictions)
    summary, per_subject = TransitionService.error_by_distance(epochs)
    transition_test = TransitionService.compare_transition_to_stable(per_subject)
    by_type = TransitionService.error_by_type(epochs)
    concentration = TransitionService.error_concentration(epochs)
    nights = TransitionService.score_nights(epochs)
    correlation = TransitionService.within_subject_correlation(nights)

    summary.to_csv(args.out)
    by_type.to_csv(os.path.join(Constants.ANALYSIS_DIR, "transition_by_type.csv"))
    correlation[0].to_csv(os.path.join(Constants.ANALYSIS_DIR, "transition_nights.csv"), index=False)

    print_report(summary, transition_test, by_type, concentration, correlation)
    if args.plot:
        plot_transitions(summary, per_subject, transition_test, by_type, correlation)
    print(f"Wrote {args.out}, transition_by_type.csv, transition_nights.csv\n")


if __name__ == "__main__":
    main()
