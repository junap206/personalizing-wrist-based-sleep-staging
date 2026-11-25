"""
plot_ablation_by_k.py

Which kinds of features benefit from a subject's own nights?

Left:  macro-F1 of the personal arm at K = 0..3 for each feature set.
Right: a table per feature set:
         gap at K=1 and K=3   personal - stranger (controls for data volume),
                              Bonferroni-corrected over K = 1, 2, 3
         K=0 -> K=3           change within the personal arm (paired t-test)

Reads the per-subject files written by run_all_personalization.bat.
No training.

Outputs: ablation_by_k.png, ablation_by_k_table.csv
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.figure_service import FigureService  # noqa: E402
from sleep_staging.personalization_statistics import PersonalizationStatistics  # noqa: E402

# (label in the figure, run tag from run_all_personalization.bat, colour)
FEATURE_SETS = [
    ("All features", "all", "#2c3e50"),
    ("Personal-reference", "personal", "#e74c3c"),
    ("Non-personal", "nonpersonal", "#3498db"),
    ("HR", "hr", "#c0392b"),
    ("Motion", "motion", "#2980b9"),
    ("HR + motion", "hrmotion", "#8e44ad"),
    ("Walch (4 features)", "walch", "#f39c12"),
]
SHOWN_K = (1, 3)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default=os.path.join(Constants.PERSONALIZATION_DIR,
                                             "personalization_v2_final_{tag}_per_subject.csv"),
                        help="file name pattern; {tag} is replaced by each run tag")
    parser.add_argument("--out", default=os.path.join(Constants.FIGURES_DIR, "ablation_by_k.png"))
    return parser.parse_args()


def load_results(pattern):
    loaded = []
    for label, tag, color in FEATURE_SETS:
        path = pattern.format(tag=tag)
        if os.path.exists(path):
            loaded.append((label, color, pd.read_csv(path)))
        else:
            print("  MISSING: " + path)
    return loaded


def format_cell(value, p_value):
    return f"{value:+.4f} {FigureService.significance_stars(p_value)}\n(p={p_value:.2g})"


def build_table(loaded):
    rows = []
    for label, _, per_subject in loaded:
        k_values = sorted(k for k in per_subject["k"].unique() if k > 0)
        tests = PersonalizationStatistics.compare_arms(per_subject, k_values)
        tests = PersonalizationStatistics.add_corrected_p(tests).set_index("k")
        change = PersonalizationStatistics.change_within_arm(
            per_subject, "personal", 0, max(k_values))

        row = {"feature set": label}
        for k in SHOWN_K:
            row[f"gap K={k}"] = tests.loc[k, "gap"]
            row[f"p K={k} (Bonf.)"] = tests.loc[k, "t_p_corrected"]
        row["personal K0->K3"] = change["gap"]
        row["p K0->K3"] = change["t_p"]
        rows.append(row)
    return pd.DataFrame(rows)


def draw_curves(axis, loaded):
    for label, color, per_subject in loaded:
        personal = per_subject[per_subject["arm"] == "personal"]
        by_k = personal.groupby("k")["macro_f1"].agg(["mean", "sem"])
        axis.errorbar(by_k.index, by_k["mean"], yerr=by_k["sem"], marker="o", lw=1.8,
                      capsize=4, color=color, label=label)
    axis.set_xlabel("K = the subject's own nights added to training")
    axis.set_ylabel("macro-F1, personal arm (mean ± SEM)")
    axis.set_xticks([0, 1, 2, 3])
    axis.set_title("Performance as personal nights are added", fontsize=11)
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.25)


def draw_table(axis, table):
    axis.axis("off")
    headers = ["Feature set", "Gap K=1", "Gap K=3", "Personal\nK=0 → K=3"]
    cells = []
    for _, row in table.iterrows():
        cells.append([row["feature set"],
                      format_cell(row["gap K=1"], row["p K=1 (Bonf.)"]),
                      format_cell(row["gap K=3"], row["p K=3 (Bonf.)"]),
                      format_cell(row["personal K0->K3"], row["p K0->K3"])])
    drawn = axis.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(8.5)
    drawn.scale(1.0, 2.6)
    for (row_number, _), cell in drawn.get_celld().items():
        if row_number == 0:
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(weight="bold")
    axis.set_title("Gap = personal − stranger (Bonferroni over K=1,2,3).\n"
                   "Right column: change within the personal arm (paired t-test).",
                   fontsize=9)


def main():
    args = parse_arguments()
    loaded = load_results(args.pattern)
    if not loaded:
        raise SystemExit("No results found. Run run_all_personalization.bat first.")

    table = build_table(loaded)
    table.to_csv(os.path.join(Constants.ABLATION_DIR, "ablation_by_k_table.csv"), index=False)
    print(table.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1, 1.15]})
    draw_curves(axes[0], loaded)
    draw_table(axes[1], table)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.out} and ablation_by_k_table.csv")


if __name__ == "__main__":
    main()
