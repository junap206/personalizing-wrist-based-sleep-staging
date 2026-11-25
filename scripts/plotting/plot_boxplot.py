"""
plot_boxplot.py

The main figure: one box per K, each dot one subject's gap
(personal macro-F1 minus stranger macro-F1). A box above the dashed zero
line means a subject's own nights helped more than a stranger's.

Each results file becomes one panel, side by side on a shared axis.

Examples:
    python scripts/plotting/plot_boxplot.py
    python scripts/plotting/plot_boxplot.py --results personalization_v2_final_all_per_subject.csv \
                           --labels "All features"

Outputs: boxplot_personalization.png, boxplot_tests.csv
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

DEFAULT_RESULTS = [os.path.join(Constants.PERSONALIZATION_DIR, "personalization_v2_final_all_per_subject.csv"),
                   os.path.join(Constants.PERSONALIZATION_DIR, "personalization_v2_final_realtime_per_subject.csv")]
DEFAULT_LABELS = ["All features (offline)", "Real-time features"]
BOX_COLOR = "#27ae60"
LINE = "=" * 72


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=DEFAULT_RESULTS)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--out", default=os.path.join(Constants.FIGURES_DIR, "personalization", "boxplot_personalization.png"))
    return parser.parse_args()


def analyse(path):
    """Gaps per K for the boxes, and the paired tests (Bonferroni-corrected)."""
    per_subject = pd.read_csv(path)
    k_values = sorted(k for k in per_subject["k"].unique() if k > 0)
    gaps = [PersonalizationStatistics.get_gaps(per_subject, k).values for k in k_values]
    tests = PersonalizationStatistics.compare_arms(per_subject, k_values)
    tests = PersonalizationStatistics.add_corrected_p(tests)
    return k_values, gaps, tests


def print_tests(label, tests):
    print("\n" + LINE)
    print(f"PERSONAL - STRANGER GAP: {label}")
    print(LINE)
    print(f"  {'K':>3}{'n':>5}{'mean gap':>11}{'95% CI':>22}{'d':>7}"
          f"{'p':>10}{'p (Bonf.)':>11}{'improved':>10}")
    for _, row in tests.iterrows():
        print(f"  {int(row['k']):>3}{int(row['n']):>5}{row['gap']:>+11.4f}"
              f"   [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]{row['cohens_d']:>7.2f}"
              f"{row['t_p']:>10.4f}{row['t_p_corrected']:>11.4f}"
              f"{int(row['n_improved']):>6}/{int(row['n'])}")


def draw_panel(axis, label, k_values, gaps, tests):
    labels = [f"K={k}" for k in k_values]
    FigureService.boxplot_with_points(axis, gaps, labels, [BOX_COLOR] * len(gaps))
    axis.axhline(0, color="#e74c3c", lw=1.2, ls="--", zorder=1)
    FigureService.label_p_values(axis, gaps, list(tests["t_p_corrected"]))

    bottom = axis.get_ylim()[0]
    for position, (_, row) in enumerate(tests.iterrows(), 1):
        axis.text(position, bottom, f"{int(row['n_improved'])}/{int(row['n'])} improved",
                  ha="center", va="bottom", fontsize=8, color="#7f8c8d")
    axis.set_title(label, fontsize=11, fontweight="bold")
    axis.set_xlabel("K = the subject's own nights added to training")


def main():
    args = parse_arguments()
    if len(args.labels) != len(args.results):
        raise SystemExit("Give one --labels entry per --results file.")

    panels = []
    all_tests = []
    for path, label in zip(args.results, args.labels):
        k_values, gaps, tests = analyse(path)
        print_tests(label, tests)
        tests.insert(0, "panel", label)
        all_tests.append(tests)
        panels.append((label, k_values, gaps, tests))
    pd.concat(all_tests).to_csv(os.path.join(Constants.PERSONALIZATION_DIR, "boxplot_tests.csv"), index=False)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.5),
                             sharey=True, squeeze=False)
    for axis, (label, k_values, gaps, tests) in zip(axes[0], panels):
        draw_panel(axis, label, k_values, gaps, tests)
    axes[0][0].set_ylabel("Δ macro-F1 (personal − stranger)")

    top = max(axis.get_ylim()[1] for axis in axes[0])
    bottom = min(np.min([np.min(g) for g in panel[2]]) for panel in panels)
    for axis in axes[0]:
        axis.set_ylim(bottom - (top - bottom) * 0.08, top)
    fig.suptitle("Each dot is one subject. Dashed line = no benefit.\n"
                 "p: paired t-test, Bonferroni-corrected for the number of K values",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.out} and boxplot_tests.csv\n")


if __name__ == "__main__":
    main()
