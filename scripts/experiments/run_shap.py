"""
run_shap.py

Which features does the tuned model rely on? Ranks all features by SHAP
importance, by signal source, and separately for Wake, NREM, and REM.

Examples:
    python scripts/experiments/run_shap.py
    python scripts/experiments/run_shap.py --n-sample 15000     # slower, tighter estimates

Outputs:
    shap_importance.csv   every feature ranked
    shap_summary.png      beeswarm plot (NREM class)
    shap_bar.png          top 25 features, coloured by signal source
    shap_by_class.png     which features drive Wake vs NREM vs REM
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.dataset_service import DatasetService  # noqa: E402
from sleep_staging.model_service import ModelService  # noqa: E402
from sleep_staging.shap_service import ShapService  # noqa: E402

LINE = "=" * 72
STAGES = list(Constants.STAGE_NAMES.values())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=Constants.FEATURES_FILE)
    parser.add_argument("--params", default=Constants.PARAMS_FILE)
    parser.add_argument("--n-sample", type=int, default=Constants.SHAP_SAMPLE_EPOCHS)
    parser.add_argument("--out", default=os.path.join(Constants.ANALYSIS_DIR, "shap_importance.csv"))
    return parser.parse_args()


# ----------------------------------------------------------------------
# Terminal report
# ----------------------------------------------------------------------
def print_top_features(table, n=20):
    print("\n" + LINE)
    print(f"TOP {n} FEATURES")
    print(LINE)
    print(f"  {'#':<4}{'feature':<28}{'importance':>12}{'% total':>9}{'cum %':>8}")
    print("  " + "-" * 60)
    for rank, row in table.head(n).iterrows():
        print(f"  {rank + 1:<4}{row['feature']:<28}{row['mean_abs_shap']:>12.4f}"
              f"{row['pct_of_total']:>9.1f}{row['cumulative_pct']:>8.1f}")


def print_concentration(table):
    print("\n" + LINE)
    print("HOW CONCENTRATED IS THE IMPORTANCE?")
    print(LINE)
    for n in (1, 3, 5, 10, 20, 30):
        if n <= len(table):
            share = table["cumulative_pct"].iloc[n - 1]
            print(f"  top {n:>2} features carry {share:>5.1f}% of total importance")
    n80 = ShapService.count_features_for_share(table, 80)
    n95 = ShapService.count_features_for_share(table, 95)
    print(f"\n  {n80} features account for 80% of importance.")
    print(f"  {n95} features account for 95%.")


def print_share_by(table, column, names):
    shares = table.groupby(column).agg(n=("feature", "size"), pct=("pct_of_total", "sum"))
    for key, row in shares.sort_values("pct", ascending=False).iterrows():
        print(f"  {names[key]:<22}{int(row['n']):>4} features{row['pct']:>8.1f}% of importance")


def print_groups(table):
    print("\n" + LINE)
    print("BY SIGNAL SOURCE")
    print(LINE)
    print_share_by(table, "group", {name: name for name in table["group"].unique()})

    print("\n" + LINE)
    print("PERSONAL-REFERENCE vs NON-PERSONAL")
    print(LINE)
    print_share_by(table, "is_personal", {True: "personal-reference", False: "non-personal"})

    print("\n" + LINE)
    print("CONTEXT COLUMNS (_prev / _next)")
    print(LINE)
    print_share_by(table, "is_context", {True: "context copies", False: "own-epoch features"})


def print_by_class(table):
    print("\n" + LINE)
    print("WHAT DRIVES EACH CLASS  (top 5)")
    print(LINE)
    for stage in STAGES:
        column = "shap_" + stage
        print(f"\n  {stage}")
        for _, row in table.nlargest(5, column).iterrows():
            print(f"    {row['feature']:<28}{row[column]:.4f}")
    print(LINE + "\n")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_beeswarm(per_class, x, path=os.path.join(Constants.FIGURES_DIR, "shap_summary.png")):
    import shap
    nrem = 1  # class code for NREM
    plt.figure(figsize=(10, 8))
    shap.summary_plot(per_class[nrem], x, max_display=20, show=False)
    plt.title("SHAP summary: NREM class", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("wrote " + path)


def plot_top_features(table, path=os.path.join(Constants.FIGURES_DIR, "shap_bar.png")):
    top = table.head(25).iloc[::-1]
    colors = [Constants.SIGNAL_COLORS[source] for source in top["source"]]

    fig, axis = plt.subplots(figsize=(9, 8))
    axis.barh(top["feature"], top["mean_abs_shap"], color=colors)
    axis.set_xlabel("mean |SHAP value|")
    axis.set_title("Top 25 features, coloured by signal source", fontsize=11)

    handles = []
    for source, color in Constants.SIGNAL_COLORS.items():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
    axis.legend(handles, Constants.SIGNAL_LABELS.values(), fontsize=8, loc="lower right")
    axis.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote " + path)


def plot_by_class(table, path=os.path.join(Constants.FIGURES_DIR, "shap_by_class.png")):
    top = table.head(12).iloc[::-1]
    bar_height = 0.8 / len(STAGES)
    rows = np.arange(len(top))

    fig, axis = plt.subplots(figsize=(9, 6))
    for offset, stage in enumerate(STAGES):
        axis.barh(rows + offset * bar_height, top["shap_" + stage],
                  height=bar_height, label=stage)
    axis.set_yticks(rows + bar_height)
    axis.set_yticklabels(top["feature"])
    axis.set_xlabel("mean |SHAP value|")
    axis.set_title("Which features drive which class", fontsize=11)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote " + path)


def main():
    args = parse_arguments()

    print("Loading...")
    df, features, label = DatasetService.load(args.features)
    params = ModelService.load_params(args.params)
    print(f"  {len(df):,} epochs, {len(features)} features, "
          f"{df['subject'].nunique()} subjects\n")

    train, test, held_out = ShapService.split_subjects(df)
    print(f"  training on {len(train):,} epochs from "
          f"{df['subject'].nunique() - len(held_out)} subjects")
    print(f"  explaining epochs from {len(held_out)} held-out subjects")

    start = time.time()
    model = ShapService.train(train, features, label, params)
    print(f"  fit in {time.time() - start:.0f}s")

    sample = ShapService.sample_epochs(test, label, args.n_sample)
    print(f"  computing SHAP on {len(sample):,} sampled epochs...")
    start = time.time()
    per_class = ShapService.explain(model, sample[features])
    print(f"  done in {time.time() - start:.0f}s")

    table = ShapService.importance_table(features, per_class)
    table.to_csv(args.out, index=False)

    print_top_features(table)
    print_concentration(table)
    print_groups(table)
    print_by_class(table)
    plot_beeswarm(per_class, sample[features])
    plot_top_features(table)
    plot_by_class(table)
    print(f"\nWrote {args.out}\n")


if __name__ == "__main__":
    main()
