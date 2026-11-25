"""
check_night_length.py

The personal and stranger arms add the same NUMBER of nights, but nights
differ in length. This checks whether the personal arm simply received more
epochs, and whether that could explain the personal-vs-stranger gap.

No training: it counts epochs in the features file and reads the results of
a finished first-K personalization run.

    python scripts/analysis/check_night_length.py
    python scripts/analysis/check_night_length.py --results personalization_v2_final_realtime_results.csv

Only valid for runs with --order first (the default); random-order runs
pick nights at random and the picks are not saved.
"""

import argparse
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402

LINE = "=" * 74


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=Constants.FEATURES_FILE)
    parser.add_argument("--results", default=os.path.join(Constants.PERSONALIZATION_DIR, "personalization_v2_final_all_results.csv"))
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def epochs_in_first_nights(features):
    """For each subject: epochs added when the first K nights are used, K = 0..7."""
    counts = features.groupby(["subject", "night"]).size()
    added = {}
    for subject, per_night in counts.groupby(level="subject"):
        lengths = per_night.sort_index(level="night").to_numpy()
        added[subject] = [int(lengths[:k].sum()) for k in range(len(lengths) + 1)]
    return added


def per_subject_table(results, added, k):
    """One row per target: epochs added and macro-F1 in each arm, donors averaged."""
    rows = results[results["k"] == k].copy()
    rows["added_epochs"] = [
        added[donor if arm == "stranger" else subject][k]
        for arm, subject, donor in zip(rows["arm"], rows["subject"], rows["donor"])
    ]
    table = rows.groupby(["subject", "arm"])[["added_epochs", "macro_f1"]].mean().unstack("arm")
    out = pd.DataFrame({
        "personal_epochs": table[("added_epochs", "personal")],
        "stranger_epochs": table[("added_epochs", "stranger")],
        "gap": table[("macro_f1", "personal")] - table[("macro_f1", "stranger")],
    })
    out["epoch_difference"] = out["personal_epochs"] - out["stranger_epochs"]
    return out, rows


def within_subject_stranger(rows):
    """Within each target, did a donor with longer nights give a better stranger score?"""
    stranger = rows[rows["arm"] == "stranger"].copy()
    for column in ("added_epochs", "macro_f1"):
        stranger[column] = stranger[column] - stranger.groupby("subject")[column].transform("mean")
    return stats.pearsonr(stranger["added_epochs"], stranger["macro_f1"])


def main():
    args = parse_arguments()
    features = pd.read_csv(args.features, usecols=["subject", "night"])
    results = pd.read_csv(args.results)
    added = epochs_in_first_nights(features)
    table, rows = per_subject_table(results, added, args.k)

    difference = table["epoch_difference"]
    _, p_difference = stats.ttest_rel(table["personal_epochs"], table["stranger_epochs"])
    r_gap, p_gap = stats.pearsonr(difference, table["gap"])
    fit = stats.linregress(difference / 1000, table["gap"])
    t_crit = stats.t.ppf(0.975, len(table) - 2)
    r_donor, p_donor = within_subject_stranger(rows)

    print("\n" + LINE)
    print(f"NIGHT LENGTH CHECK  (K = {args.k}, {len(table)} targets, {args.results})")
    print(LINE)
    print(f"  Epochs added, personal arm : {table['personal_epochs'].mean():7.0f} "
          f"(SD {table['personal_epochs'].std():.0f})")
    print(f"  Epochs added, stranger arm : {table['stranger_epochs'].mean():7.0f} "
          f"(SD {table['stranger_epochs'].std():.0f}, averaged over 3 donors)")
    print(f"  Personal minus stranger    : {difference.mean():+7.0f} epochs "
          f"(paired p = {p_difference:.3f}); "
          f"{(difference > 0).sum()}/{len(table)} targets got more epochs from their own nights")
    print()
    print("  Does the gap track the difference in epochs?")
    print(f"    correlation across targets       r = {r_gap:+.3f}, p = {p_gap:.3f}")
    print(f"    slope                            {fit.slope:+.4f} macro-F1 per 1,000 extra epochs")
    print(f"    gap at equal epochs (intercept)  {fit.intercept:+.4f} "
          f"[{fit.intercept - t_crit * fit.intercept_stderr:+.4f}, "
          f"{fit.intercept + t_crit * fit.intercept_stderr:+.4f}]")
    print()
    print("  Within a target, did donors with more epochs give a better stranger score?")
    print(f"    r = {r_donor:+.3f}, p = {p_donor:.3f}  (deviations from each target's mean)")
    print(LINE)
    print("  Reading it: if the intercept stays near the observed gap and its interval")
    print("  excludes 0, night length does not explain the personal benefit.")
    print(LINE + "\n")
    table.to_csv(os.path.join(Constants.ANALYSIS_DIR, "night_length_check.csv"))
    print("Wrote night_length_check.csv\n")


if __name__ == "__main__":
    main()