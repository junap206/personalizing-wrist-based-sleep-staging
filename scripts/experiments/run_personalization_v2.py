"""
run_personalization_v2.py

Does a person's own prior nights improve sleep staging more than the same
amount of data from a stranger?

For every target subject, K = 0..3:
    personal arm = all other subjects + K of the target's own nights
    stranger arm = all other subjects + K nights from a donor (a stranger)
Both arms are tested on the target's remaining nights.

The same 3 donors are used for a subject in every run, so results from
different feature sets are directly comparable.

Examples:
    python scripts/experiments/run_personalization_v2.py --quick --tag test
    python scripts/experiments/run_personalization_v2.py --tag final_all
    python scripts/experiments/run_personalization_v2.py --realtime --tag final_realtime
    python scripts/experiments/run_personalization_v2.py --feature-set personal --tag final_personal
    python scripts/experiments/run_personalization_v2.py --features data/features/walch_features.csv --tag final_walch

Outputs (prefix = personalization_v2_<tag>):
    _results.csv      one row per (arm, K, subject, donor)
    _per_subject.csv  one row per (arm, K, subject), donors averaged
    _summary.csv      one row per (arm, K)
    _tests.csv        paired personal-vs-stranger test at each K
    _config.json      the exact settings used
    .png              personalization curves

if run time stops then just restart it should save everything """

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.checkpoint_service import CheckpointService
from sleep_staging.constants import Constants
from sleep_staging.dataset_service import DatasetService
from sleep_staging.feature_set_service import FeatureSetService
from sleep_staging.model_service import ModelService
from sleep_staging.personalization_report import PersonalizationReport
from sleep_staging.personalization_service import PersonalizationService
from sleep_staging.personalization_statistics import PersonalizationStatistics


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=Constants.FEATURES_FILE)
    parser.add_argument("--feature-set", default="all", choices=Constants.FEATURE_SETS)
    parser.add_argument("--realtime", action="store_true",
                        help="drop features that need future data")
    parser.add_argument("--order", default="first", choices=["first", "random"],
                        help="which of the target's nights go into training")
    parser.add_argument("--donors", type=int, default=Constants.DONORS_PER_SUBJECT)
    parser.add_argument("--k-max", type=int, default=Constants.MAX_PERSONAL_NIGHTS)
    parser.add_argument("--classes", type=int, default=3, choices=[3, 5])
    parser.add_argument("--params", default=Constants.PARAMS_FILE)
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="override the tree count in the params file")
    parser.add_argument("--seed", type=int, default=Constants.RANDOM_SEED)
    parser.add_argument("--quick", action="store_true",
                        help="8 subjects and 2 donors, for testing")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def get_output_prefix(tag):
    if tag:
        return os.path.join(Constants.PERSONALIZATION_DIR, "personalization_v2_" + tag)
    return os.path.join(Constants.PERSONALIZATION_DIR, "personalization_v2")


def choose_features(all_features, set_name, realtime):
    features = FeatureSetService.select(all_features, set_name)
    dropped = []
    if realtime:
        features, dropped = FeatureSetService.keep_causal(features)
    return features, dropped


def choose_subjects(df, k_max, quick):
    """
    Targets need more than k_max nights (k_max to train + 1 to test).
    Donors need at least k_max nights.
    The same targets are used at every K.
    """
    nights_per_subject = df.groupby("subject")["night"].nunique()
    targets = sorted(nights_per_subject[nights_per_subject > k_max].index)
    donor_pool = sorted(nights_per_subject[nights_per_subject >= k_max].index)
    if quick:
        targets = targets[:Constants.QUICK_TEST_SUBJECTS]
    return targets, donor_pool


def print_setup(df, features, dropped, targets, n_donors, k_values, args, params):
    print(f"  {len(df):,} epochs, {len(features)} features")
    print(f"  {len(targets)} subjects with > {args.k_max} nights (used at every K)")
    print(f"  {n_donors} donors per subject, order = {args.order}, "
          f"feature set = {args.feature_set}")
    print(f"  XGBoost: {params.get('n_estimators', 'default')} trees, "
          f"lr {params.get('learning_rate', 'default')}")
    if dropped:
        print(f"  real-time: dropped {dropped}")
    n_fits = len(targets) * n_donors * len(k_values) * 2
    print(f"  ~{n_fits} model fits\n")


def run_all_subjects(df, features, label, targets, donor_pool, n_donors,
                     k_values, args, params, prefix):
    finished = CheckpointService.finished_subjects(prefix)
    if finished:
        print(f"  RESUMING: {len(finished)} subjects already finished\n")

    for number, subject in enumerate(targets, 1):
        if subject in finished:
            continue

        start = time.time()
        generator = PersonalizationService.get_subject_generator(args.seed, subject)
        donors = PersonalizationService.choose_donors(
            subject, donor_pool, n_donors, generator)
        rows = PersonalizationService.run_subject(
            df, features, label, subject, donors, k_values,
            args.order, generator, params)
        CheckpointService.append(prefix, rows)

        minutes = (time.time() - start) / 60
        print(f"  [{number:>2}/{len(targets)}] {subject:<12} {minutes:.1f} min  (saved)")


def save_outputs(prefix, results, per_subject, summary, tests):
    results.to_csv(prefix + "_results.csv", index=False)
    per_subject.to_csv(prefix + "_per_subject.csv", index=False)
    summary.to_csv(prefix + "_summary.csv", index=False)
    tests.to_csv(prefix + "_tests.csv", index=False)
    print("Wrote " + prefix + "_results.csv, _per_subject.csv, _summary.csv, _tests.csv")


def main():
    args = parse_arguments()
    prefix = get_output_prefix(args.tag)

    # ---- 1. Data and features ----
    print("Loading...")
    df, all_features, label = DatasetService.load(args.features, args.classes)
    features, dropped = choose_features(all_features, args.feature_set, args.realtime)

    # ---- 2. Model settings (identical for every experiment) ----
    params = ModelService.load_params(args.params, args.n_estimators, args.seed)

    # ---- 3. Subjects, donors, K values ----
    targets, donor_pool = choose_subjects(df, args.k_max, args.quick)
    n_donors = Constants.QUICK_TEST_DONORS if args.quick else args.donors
    k_values = list(range(0, args.k_max + 1))
    print_setup(df, features, dropped, targets, n_donors, k_values, args, params)

    # ---- 4. Record settings, then run (resumable) ----
    config = {
        "feature_set": args.feature_set,
        "n_features": len(features),
        "realtime": args.realtime,
        "order": args.order,
        "donors": n_donors,
        "k_max": args.k_max,
        "seed": args.seed,
        "subjects": len(targets),
        "features_file": args.features,
        "xgb_params": params,
    }
    CheckpointService.save_config(prefix, config)

    start = time.time()
    run_all_subjects(df, features, label, targets, donor_pool, n_donors,
                     k_values, args, params, prefix)
    print(f"\n  total {(time.time() - start) / 60:.1f} min")

    # ---- 5. Statistics ----
    results = CheckpointService.load(prefix)
    per_subject = PersonalizationStatistics.average_over_donors(results)
    summary = PersonalizationStatistics.summarise(per_subject)
    tests = PersonalizationStatistics.compare_arms(per_subject, k_values)
    dose_response = PersonalizationStatistics.test_dose_response(per_subject)

    # ---- 6. Save, report, plot ----
    save_outputs(prefix, results, per_subject, summary, tests)
    PersonalizationReport.print_report(summary, tests, dose_response, dropped, args.order)
    PersonalizationReport.plot_curves(summary, tests, os.path.join(
        Constants.FIGURES_DIR, "personalization", os.path.basename(prefix) + ".png"))


if __name__ == "__main__":
    main()
