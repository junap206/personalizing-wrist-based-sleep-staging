"""
run_ablation.py

Which signal does the work? Trains the tuned model on subsets of the
features and compares them with leave-one-subject-out.

Default sets: HR, motion, time, and their combinations, plus all features.
This is the comparison in Walch et al. (2019), Table 6.

Examples:
    python scripts/experiments/run_ablation.py
    python scripts/experiments/run_ablation.py --quick
    python scripts/experiments/run_ablation.py --sets personal nonpersonal all --out-prefix ablation_personal

Outputs (prefix default 'ablation'):
    <prefix>_results.csv   one row per (feature set, subject)
    <prefix>_summary.csv   one row per feature set
    <prefix>_tests.csv     paired test of each set against all features
    <prefix>.png           boxplot, one dot per subject, p-values on top
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.ablation_service import AblationService  # noqa: E402
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.dataset_service import DatasetService  # noqa: E402
from sleep_staging.figure_service import FigureService  # noqa: E402
from sleep_staging.model_service import ModelService  # noqa: E402

DEFAULT_SETS = ("hr", "motion", "time", "hr_motion", "hr_time", "motion_time", "all")
LINE = "=" * 74


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=Constants.FEATURES_FILE)
    parser.add_argument("--params", default=Constants.PARAMS_FILE)
    parser.add_argument("--sets", nargs="*", default=list(DEFAULT_SETS),
                        choices=Constants.FEATURE_SETS)
    parser.add_argument("--quick", action="store_true", help="8 subjects only")
    parser.add_argument("--out-prefix", default=os.path.join(Constants.ABLATION_DIR, "ablation"))
    return parser.parse_args()


# ----------------------------------------------------------------------
# Terminal report
# ----------------------------------------------------------------------
def print_summary(summary):
    print("\n" + LINE)
    print("ABLATION RESULTS")
    print(LINE)
    print(f"\n  {'feature set':<14}{'n feat':>8}{'macro-F1':>11}{'sd':>8}"
          f"{'kappa':>9}{'accuracy':>11}")
    print("  " + "-" * 61)
    for _, row in summary.iterrows():
        print(f"  {row['set']:<14}{int(row['n_features']):>8}{row['macro_f1']:>11.3f}"
              f"{row['macro_f1_sd']:>8.3f}{row['kappa']:>9.3f}{row['accuracy']:>11.3f}")


def print_signal_values(summary):
    """How much each signal adds on top of the others."""
    scores = summary.set_index("set")["macro_f1"]
    comparisons = [
        ("motion", "hr_motion", "adding heart rate to motion"),
        ("hr", "hr_motion", "adding motion to heart rate"),
        ("hr_motion", "all", "adding time to heart rate + motion"),
        ("time", "all", "adding the watch signals to time alone"),
    ]
    print("\n" + LINE)
    print("WHAT EACH SIGNAL IS WORTH")
    print(LINE)
    for without, with_signal, description in comparisons:
        if without in scores and with_signal in scores:
            gain = scores[with_signal] - scores[without]
            print(f"  {description:<44}{gain:+.3f}")


def print_tests(tests):
    print("\n" + LINE)
    print("PAIRED TEST AGAINST ALL FEATURES  (within subject)")
    print(LINE)
    print(f"\n  {'feature set':<14}{'difference':>12}{'p':>12}{'Bonferroni p':>15}")
    print("  " + "-" * 53)
    for _, row in tests.iterrows():
        print(f"  {row['set']:<14}{row['mean_difference']:>+12.3f}"
              f"{row['p']:>12.2e}{row['p_corrected']:>15.2e}")


def print_per_class(summary, per_class_f1):
    names = list(Constants.STAGE_NAMES.values())
    print("\n" + LINE)
    print("PER-CLASS F1 BY FEATURE SET  (all epochs pooled)")
    print(LINE)
    print(f"\n  {'feature set':<14}" + "".join(f"{name:>10}" for name in names))
    print("  " + "-" * 44)
    for set_name in summary["set"]:
        values = per_class_f1[set_name]
        print(f"  {set_name:<14}" + "".join(f"{value:>10.3f}" for value in values))
    print(LINE + "\n")


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def get_set_color(set_name):
    if set_name in Constants.SIGNAL_COLORS:
        return Constants.SIGNAL_COLORS[set_name]
    return "#34495e"


def plot_ablation(results, summary, tests, path):
    order = list(summary["set"])
    groups = [results.loc[results["set"] == name, "macro_f1"].values for name in order]
    labels = [Constants.FEATURE_SET_LABELS[name] for name in order]
    colors = [get_set_color(name) for name in order]

    corrected = dict(zip(tests["set"], tests["p_corrected"]))
    p_values = [corrected.get(name) for name in order]

    fig, axis = plt.subplots(figsize=(1.3 * len(order) + 2, 5.5))
    FigureService.boxplot_with_points(axis, groups, labels, colors)
    FigureService.label_p_values(axis, groups, p_values)
    axis.set_ylabel("macro-F1 per subject")
    axis.set_title("Feature ablation (leave-one-subject-out). Each dot is one subject.\n"
                   "p: paired t-test vs all features, Bonferroni-corrected",
                   fontsize=10)
    axis.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote " + path)


def main():
    args = parse_arguments()

    print("Loading...")
    df, all_features, label = DatasetService.load(args.features)
    params = ModelService.load_params(args.params)
    subjects = DatasetService.get_subjects(df)
    if args.quick:
        subjects = subjects[:Constants.QUICK_TEST_SUBJECTS]
    print(f"  {len(df):,} epochs, {len(all_features)} features, {len(subjects)} subjects")
    print(f"  feature sets: {', '.join(args.sets)}\n")

    start = time.time()
    results, per_class_f1 = AblationService.run(
        df, all_features, label, subjects, params, args.sets)
    print(f"\n  total {(time.time() - start) / 60:.1f} min")

    summary = AblationService.summarise(results)
    tests = AblationService.compare_to_reference(results, reference="all")

    results.to_csv(args.out_prefix + "_results.csv", index=False)
    summary.to_csv(args.out_prefix + "_summary.csv", index=False)
    tests.to_csv(args.out_prefix + "_tests.csv", index=False)

    print_summary(summary)
    print_signal_values(summary)
    print_tests(tests)
    print_per_class(summary, per_class_f1)
    plot_ablation(results, summary, tests, os.path.join(Constants.FIGURES_DIR, os.path.basename(args.out_prefix) + ".png"))
    print(f"Wrote {args.out_prefix}_results.csv, _summary.csv, _tests.csv\n")


if __name__ == "__main__":
    main()
