"""
compare_nights.py

What makes a night hard to score? For each subject, compare their best and
worst night (by baseline macro-F1) on every feature and on sleep
architecture. Paired within subject; Benjamini-Hochberg corrected.

Examples:
    python scripts/analysis/compare_nights.py --plot
    python scripts/analysis/compare_nights.py --no-context      # leave out _prev/_next copies

Needs: features_v2.csv, night_scores.csv (from analyze_stability.py)
Outputs: good_vs_bad.csv, good_vs_bad.png (with --plot)
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
from sleep_staging.feature_set_service import FeatureSetService  # noqa: E402
from sleep_staging.night_comparison_service import NightComparisonService, SCORE_COLUMNS  # noqa: E402

LINE = "=" * 72
CATEGORIES = {
    "Sleep composition": ["frac_wake", "frac_nrem", "frac_rem"],
    "Fragmentation": ["n_transitions", "transitions_per_hour", "longest_run"],
    "Night length": ["n_epochs", "hours"],
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=Constants.FEATURES_FILE)
    parser.add_argument("--scores", default=Constants.NIGHT_SCORES_FILE)
    parser.add_argument("--out", default=Constants.GOOD_VS_BAD_FILE)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-context", action="store_true",
                        help="leave out the _prev/_next context copies")
    return parser.parse_args()


def choose_columns(nights, no_context):
    columns = [column for column in nights.columns if column not in SCORE_COLUMNS]
    if no_context:
        columns = [column for column in columns if not FeatureSetService.is_context(column)]
    return columns


def print_heading(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def print_report(nights, pairs, table):
    good, bad = NightComparisonService.get_groups(nights, pairs)
    print_heading("GOOD NIGHT vs BAD NIGHT")
    print(f"  {len(pairs)} subjects contributed one best and one worst night each.")
    print(f"\n  Good nights: macro-F1 {good['macro_f1'].mean():.3f}  "
          f"(range {good['macro_f1'].min():.3f}-{good['macro_f1'].max():.3f})")
    print(f"  Bad nights : macro-F1 {bad['macro_f1'].mean():.3f}  "
          f"(range {bad['macro_f1'].min():.3f}-{bad['macro_f1'].max():.3f})")
    print(f"  Gap        : {good['macro_f1'].mean() - bad['macro_f1'].mean():.3f}")
    if table.empty:
        print("\n  Nothing to compare.")
        return

    significant = table[table["significant"]]
    print_heading(f"FEATURES THAT DIFFER  ({len(significant)} of {len(table)} "
                  "survive Benjamini-Hochberg correction)")
    shown = significant.head(15) if len(significant) else table.head(10)
    if significant.empty:
        print("  None survive correction. The strongest raw differences:")
    print(f"\n  {'feature':<26}{'good':>10}{'bad':>10}{'change':>9}{'d':>7}{'p':>10}")
    print("  " + "-" * 70)
    for _, row in shown.iterrows():
        print_row(row)
    print("\n  d = Cohen's d for paired data: 0.2 small, 0.5 medium, 0.8 large.")
    print("  * survives Benjamini-Hochberg correction (FDR 5%).")

    print_heading("BY CATEGORY  (top 3 by effect size)")
    for name, columns in get_categories(table).items():
        rows = table[table["feature"].isin(columns)]
        if rows.empty:
            continue
        print(f"\n  {name}")
        top = rows.reindex(rows["cohens_d"].abs().sort_values(ascending=False).index).head(3)
        for _, row in top.iterrows():
            marker = "*" if row["significant"] else " "
            print(f"   {marker}{row['feature']:<26} d={row['cohens_d']:+.2f}  p={row['t_p']:.4f}")
    print(LINE + "\n")


def print_row(row):
    marker = "*" if row["significant"] else " "
    change = f"{row['pct_change']:+.0f}%" if np.isfinite(row["pct_change"]) else "  n/a"
    print(f" {marker}{row['feature']:<26}{row['good_mean']:>10.3f}{row['bad_mean']:>10.3f}"
          f"{change:>9}{row['cohens_d']:>7.2f}{row['t_p']:>10.4f}")


def get_categories(table):
    """Fixed categories plus own-epoch heart-rate and motion features."""
    categories = dict(CATEGORIES)
    own_epoch = [f for f in table["feature"] if not FeatureSetService.is_context(f)]
    categories["Heart rate"] = [f for f in own_epoch if f.startswith("hr_")]
    categories["Motion"] = [f for f in own_epoch
                            if f.startswith(("mag_", "jerk_", "move_", "orient_", "epochs_"))]
    return categories


def plot_pairs(nights, pairs, table, path=os.path.join(Constants.FIGURES_DIR, "good_vs_bad.png")):
    """The six largest effects. Each grey line joins one subject's two nights."""
    good, bad = NightComparisonService.get_groups(nights, pairs)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for axis, (_, row) in zip(axes.ravel(), table.head(6).iterrows()):
        column = row["feature"]
        worst_values = bad[column].values
        best_values = good[column].values
        for worst, best in zip(worst_values, best_values):
            axis.plot([0, 1], [worst, best], color="0.7", lw=0.8, zorder=1)
        axis.scatter([0] * len(worst_values), worst_values, s=18, color="#c0392b", zorder=2)
        axis.scatter([1] * len(best_values), best_values, s=18, color="#27ae60", zorder=2)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["worst\nnight", "best\nnight"])
        marker = " *" if row["significant"] else ""
        axis.set_title(f"{column}\nd={row['cohens_d']:+.2f}, p={row['t_p']:.3g}{marker}",
                       fontsize=10)
        axis.grid(alpha=0.25, axis="y")

    fig.suptitle("Best vs worst night per subject. Each grey line is one subject.\n"
                 "* survives Benjamini-Hochberg correction", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote " + path)


def main():
    args = parse_arguments()
    print("Loading...")
    features = pd.read_csv(args.features)
    night_scores = pd.read_csv(args.scores)
    print(f"  {len(features):,} epochs, {len(night_scores)} scored nights")

    nights = NightComparisonService.summarise_nights(features, night_scores)
    pairs = NightComparisonService.pick_best_and_worst(nights)
    print(f"  {len(nights)} nights, {len(pairs)} subjects with a distinct best and worst night")

    columns = choose_columns(nights, args.no_context)
    table = NightComparisonService.compare(nights, pairs, columns)
    table.to_csv(args.out, index=False)

    print_report(nights, pairs, table)
    if args.plot and not table.empty:
        plot_pairs(nights, pairs, table)
    print(f"Wrote {args.out}\n")


if __name__ == "__main__":
    main()
