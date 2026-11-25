"""
plot_hypnograms.py

One page per subject: every night stacked, the true hypnogram with the
model's prediction drawn over it (black), and wrist motion beside it.

The prediction line is drawn whenever a predictions file is found.
NREM predictions are drawn on the N2 row, since the model is three-class.

Examples:
    python scripts/plotting/plot_hypnograms.py                      # 6 subjects: 3 steadiest, 3 most variable
    python scripts/plotting/plot_hypnograms.py --subjects Bidslab36 Bidslab42
    python scripts/plotting/plot_hypnograms.py --all-nights-one-axis

Outputs (in hypnograms/):
    hypnogram_<subject>.png      the main per-subject page
    overlay_<subject>.png        all nights on one axis (--all-nights-one-axis)
    subject_comparison.png       fragmentation and stage fractions per subject
"""

import argparse
import datetime as dt
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.figure_service import FigureService  # noqa: E402
from sleep_staging.hypnogram_service import HypnogramService, STAGE_ROW_LABELS  # noqa: E402
from sleep_staging.raw_data_service import TIMEZONE  # noqa: E402

TRUE_COLOR = "#d62728"
PREDICTED_COLOR = "#1a1a1a"
PREDICTED_ALPHA = 0.55


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=Constants.DATA_DIR)
    parser.add_argument("--scores", default=Constants.NIGHT_SCORES_FILE)
    parser.add_argument("--pred", default=Constants.PREDICTIONS_FILE)
    parser.add_argument("--no-pred", action="store_true", help="leave out the prediction line")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="every subject")
    parser.add_argument("--n", type=int, default=6, help="subjects to pick automatically")
    parser.add_argument("--all-nights-one-axis", action="store_true",
                        help="also draw each subject's nights on a single axis")
    parser.add_argument("--outdir", default=os.path.join(Constants.FIGURES_DIR, "hypnograms"))
    return parser.parse_args()


def load_optional(path, description):
    if not os.path.exists(path):
        print(f"  ({path} not found: no {description})")
        return None
    print(f"  using {path} for {description}")
    return pd.read_csv(path)


def get_night_f1(night_scores, subject, night):
    if night_scores is None:
        return ""
    row = night_scores[(night_scores["subject"] == subject)
                       & (night_scores["night"].astype(str) == str(night))]
    return f"   F1 {row['macro_f1'].iloc[0]:.3f}" if len(row) else ""


def draw_night(axes_row, subject, night, folder, night_scores, predictions):
    """One row of a subject page: hypnogram (+ prediction) and motion."""
    labels, jerk, recording_start = HypnogramService.load_night(folder)
    stats = HypnogramService.describe(labels)
    hypnogram_axis, motion_axis = axes_row

    hours, rows = HypnogramService.true_rows(labels)
    hypnogram_axis.step(hours, rows, where="post", lw=1.0, color=TRUE_COLOR, zorder=3)

    accuracy_text = ""
    if predictions is not None:
        night_predictions = HypnogramService.get_night_predictions(predictions, subject, night)
        if len(night_predictions):
            hours, rows = HypnogramService.predicted_rows(night_predictions, len(labels))
            hypnogram_axis.step(hours, rows, where="post", lw=1.8, color=PREDICTED_COLOR,
                                alpha=PREDICTED_ALPHA, zorder=2)
            accuracy_text = f"   acc {HypnogramService.accuracy(night_predictions):.0%}"

    hypnogram_axis.set_yticks(range(len(STAGE_ROW_LABELS)))
    hypnogram_axis.set_yticklabels(STAGE_ROW_LABELS, fontsize=8)
    hypnogram_axis.set_ylim(-0.5, 4.5)
    hypnogram_axis.set_ylabel(f"night {night}", fontsize=9)
    hypnogram_axis.grid(alpha=0.2)
    start = dt.datetime.fromtimestamp(recording_start, TIMEZONE)
    hypnogram_axis.text(
        0.01, 0.95,
        f"start {start:%H:%M}   {stats['transitions']} transitions   "
        f"{stats['trans_per_hour']:.1f}/hr{get_night_f1(night_scores, subject, night)}"
        f"{accuracy_text}",
        transform=hypnogram_axis.transAxes, va="top", fontsize=8, color="0.35")

    motion_axis.plot(HypnogramService.epoch_hours(len(jerk)), jerk, lw=0.6, color=TRUE_COLOR)
    motion_axis.set_ylabel("motion", fontsize=8)
    motion_axis.grid(alpha=0.2)
    motion_axis.tick_params(labelsize=7)
    return stats


def plot_subject(subject, nights, night_scores, predictions, outdir):
    fig, axes = plt.subplots(len(nights), 2, figsize=(15, 2.1 * len(nights) + 1.6),
                             gridspec_kw={"width_ratios": [3, 1]}, squeeze=False)
    rows = []
    for axes_row, (night, folder) in zip(axes, nights):
        stats = draw_night(axes_row, subject, night, folder, night_scores, predictions)
        rows.append({"night": night, **stats})
    summary = pd.DataFrame(rows)

    longest = summary["hours"].max()
    for row_number, axes_row in enumerate(axes):
        for axis in axes_row:
            axis.set_xlim(0, longest)
            if row_number < len(nights) - 1:
                axis.set_xticklabels([])
    axes[-1][0].set_xlabel("hours since recording start")
    axes[-1][1].set_xlabel("hours")

    if predictions is not None:
        legend = [Line2D([0], [0], color=TRUE_COLOR, lw=1.0, label="true stage (EEG)"),
                  Line2D([0], [0], color=PREDICTED_COLOR, lw=1.8, alpha=PREDICTED_ALPHA,
                         label="model prediction")]
        axes[0][0].legend(handles=legend, fontsize=8, loc="upper right", framealpha=0.8)

    fig.suptitle(f"{subject}\n{len(nights)} nights  |  transitions/hr "
                 f"{summary['trans_per_hour'].min():.1f}-{summary['trans_per_hour'].max():.1f}"
                 f"  |  N3 {summary['frac_n3'].min() * 100:.0f}-"
                 f"{summary['frac_n3'].max() * 100:.0f}%  |  REM "
                 f"{summary['frac_rem'].min() * 100:.0f}-{summary['frac_rem'].max() * 100:.0f}%",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(outdir, f"hypnogram_{subject}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, summary


def plot_all_nights_one_axis(subject, nights, outdir):
    fig, axis = plt.subplots(figsize=(14, 4.5))
    colors = plt.cm.tab10.colors
    for number, (night, folder) in enumerate(nights):
        labels, _, _ = HypnogramService.load_night(folder)
        hours, rows = HypnogramService.true_rows(labels)
        offset = (number - len(nights) / 2) * 0.06   # keep the lines from hiding each other
        axis.step(hours, rows + offset, where="post", lw=1.0, alpha=0.75,
                  color=colors[number % len(colors)], label=f"night {night}")
    axis.set_yticks(range(len(STAGE_ROW_LABELS)))
    axis.set_yticklabels(STAGE_ROW_LABELS)
    axis.set_xlabel("hours since recording start")
    axis.set_title(f"{subject}: all nights overlaid", fontsize=11)
    axis.legend(fontsize=8, ncol=len(nights))
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"overlay_{subject}.png"), dpi=200)
    plt.close(fig)


def plot_comparison(summaries, outdir):
    subjects = sorted(summaries)
    panels = [("trans_per_hour", "transitions per hour"),
              ("frac_n3", "fraction N3"), ("frac_rem", "fraction REM")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (column, title) in zip(axes, panels):
        groups = [summaries[subject][column].values for subject in subjects]
        FigureService.boxplot_with_points(axis, groups, subjects, ["#c0392b"] * len(groups))
        axis.set_title(title, fontsize=10)
        axis.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("Each dot is one night. A tall box means that subject varies a lot.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, "subject_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    args = parse_arguments()
    night_scores = load_optional(args.scores, "night F1 labels and subject choice")
    predictions = None if args.no_pred else load_optional(args.pred, "the prediction line")

    nights_by_subject = HypnogramService.nights_by_subject(args.data, args.subjects)
    available = sorted(nights_by_subject)
    if args.subjects or args.all:
        subjects = available
    else:
        subjects = HypnogramService.choose_subjects(available, night_scores, args.n)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nPlotting {len(subjects)} subjects: {', '.join(subjects)}\n")
    summaries = {}
    for subject in subjects:
        nights = nights_by_subject[subject]
        path, summary = plot_subject(subject, nights, night_scores, predictions, args.outdir)
        summaries[subject] = summary
        spread = summary["trans_per_hour"]
        print(f"  {subject}: {len(nights)} nights | transitions/hr "
              f"{spread.min():.1f}-{spread.max():.1f} -> {os.path.basename(path)}")
        if args.all_nights_one_axis:
            plot_all_nights_one_axis(subject, nights, args.outdir)

    if len(summaries) > 1:
        path = plot_comparison(summaries, args.outdir)
        print("\n  comparison -> " + os.path.basename(path))
    print(f"\nWrote {len(summaries)} pages to {args.outdir}/\n")


if __name__ == "__main__":
    main()
