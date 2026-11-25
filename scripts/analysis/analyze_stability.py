"""
analyze_stability.py

Is a subject's difficulty a stable trait, or night-to-night noise?
If it is stable, there is something personal for the model to learn.

Reads the baseline predictions, scores every night, and reports the ICC
split-half reliability, and the most and least consistent subjects.

Example
    python scripts/analysis/analyze_stability.py
    python scripts/analysis/analyze_stability.py --plot

Outputs:
    night_scores.csv   one row per subject-night (used by later analyses)
    stability.png      every night of every subject (with --plot)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.night_score_service import NightScoreService  # noqa: E402
from sleep_staging.stability_service import StabilityService  # noqa: E402

LINE = "=" * 66


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=Constants.PREDICTIONS_FILE)
    parser.add_argument("--out", default=Constants.NIGHT_SCORES_FILE)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def print_heading(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def describe_icc(icc):
    if icc >= 0.75:
        return "excellent - difficulty is strongly personal"
    if icc >= 0.5:
        return "moderate to good - a real personal component"
    if icc >= 0.25:
        return "fair - some personal signal, plenty of noise"
    return "poor - little stable personal signal"


def print_report(night_scores):
    print_heading("NIGHT-LEVEL SCORES")
    f1 = night_scores["macro_f1"]
    print(f"  {len(night_scores)} nights from {night_scores['subject'].nunique()} subjects")
    print(f"  macro-F1: mean {f1.mean():.3f}, range {f1.min():.3f} to {f1.max():.3f}")

    icc = StabilityService.intraclass_correlation(night_scores)
    if icc:
        print_heading("INTRACLASS CORRELATION")
        print(f"  ICC = {icc['icc']:.3f}   ({icc['n_subjects']} subjects, "
              f"{icc['n_nights']} nights, {icc['mean_nights']:.1f} nights each)")
        print(f"  between-subject sd {icc['between_sd']:.3f} | "
              f"within-subject sd {icc['within_sd']:.3f}")
        print("  Interpretation: " + describe_icc(icc["icc"]))

    split = StabilityService.split_half(night_scores)
    if split:
        print_heading("SPLIT-HALF RELIABILITY")
        print(f"  r = {split['mean_r']:.3f}   95% interval [{split['lo']:.3f}, "
              f"{split['hi']:.3f}]   ({split['n_subjects']} subjects with 4+ nights)")
        print("  Half of a subject's nights predicting their other half.")

    headroom = StabilityService.headroom(night_scores)
    print_heading("ROUGH HEADROOM")
    print(f"  Current mean per subject              : {headroom['current_mean']:.3f}")
    print(f"  If every subject hit their best night : {headroom['if_best_night']:.3f}")
    print(f"  Difference                            : {headroom['gain']:.3f}")
    print("  (An optimistic ceiling, not a prediction.)")

    consistency = StabilityService.consistency_by_subject(night_scores)
    print_heading("MOST CONSISTENT SUBJECTS (lowest night-to-night sd)")
    print_subjects(consistency.nsmallest(5, "std"))
    print("\nLEAST CONSISTENT SUBJECTS (highest night-to-night sd)")
    print_subjects(consistency.nlargest(5, "std"))

    print_heading("WHAT MAKES A NIGHT HARD?")
    for column, label in (("rem_frac", "REM fraction"), ("wake_frac", "Wake fraction"),
                          ("n_epochs", "night length")):
        correlation = night_scores["macro_f1"].corr(night_scores[column])
        print(f"  correlation with {label:<15}: {correlation:+.3f}")
    print(LINE + "\n")


def print_subjects(rows):
    for subject, row in rows.iterrows():
        print(f"  {subject:<12} mean {row['mean']:.3f}   sd {row['std']:.3f}   "
              f"({int(row['count'])} nights, {row['min']:.3f}-{row['max']:.3f})")


def plot_stability(night_scores, path=os.path.join(Constants.FIGURES_DIR, "stability.png")):
    consistency = StabilityService.consistency_by_subject(night_scores)
    order = list(consistency.sort_values("mean").index)

    fig, axis = plt.subplots(figsize=(12, 6))
    for position, subject in enumerate(order):
        scores = night_scores.loc[night_scores["subject"] == subject, "macro_f1"].values
        axis.scatter([position] * len(scores), scores, s=22, alpha=0.6, color="#1f77b4")
        axis.scatter([position], [scores.mean()], s=70, marker="_", color="crimson", zorder=3)

    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order, rotation=90, fontsize=7)
    axis.set_ylabel("macro-F1 per night")
    axis.set_title("Each dot is one night. Red dash is the subject mean.\n"
                   "Tight columns = consistent subjects. Spread out = noisy.", loc="left")
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote " + path)


def main():
    args = parse_arguments()
    predictions = NightScoreService.load_predictions(args.pred)
    print(f"Loaded {len(predictions):,} scored epochs from {args.pred}.")

    night_scores = NightScoreService.score_nights(predictions)
    night_scores.to_csv(args.out, index=False)
    print_report(night_scores)
    if args.plot:
        plot_stability(night_scores)
    print(f"Wrote {args.out}\n")


if __name__ == "__main__":
    main()
