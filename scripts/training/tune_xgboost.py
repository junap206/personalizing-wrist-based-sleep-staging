"""
tune_xgboost.py

Finds good XGBoost settings with Optuna, using training data only, and
saves them to xgb_best_params.json. Every training script loads that file,
so every experiment uses the same tuned model.

Examples:
    python scripts/training/tune_xgboost.py                     # 80 trials
    python scripts/training/tune_xgboost.py --n-trials 150      # more thorough

Outputs:
    xgb_best_params.json   the tuned settings
    tuning_history.csv     every trial's settings and score
    tuning.png             search history and which settings mattered
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.dataset_service import DatasetService  # noqa: E402
from sleep_staging.tuning_service import TuningService  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
LINE = "=" * 66


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=Constants.FEATURES_FILE)
    parser.add_argument("--classes", type=int, choices=[3, 5], default=3)
    parser.add_argument("--n-trials", type=int, default=Constants.TUNING_TRIALS)
    parser.add_argument("--n-val", type=int, default=Constants.TUNING_VALIDATION_SUBJECTS,
                        help="subjects held out for validation")
    parser.add_argument("--seed", type=int, default=Constants.RANDOM_SEED)
    parser.add_argument("--out", default=Constants.PARAMS_FILE)
    return parser.parse_args()


def run_search(tuning_df, features, label, n_trials, seed):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = TuningService.make_objective(tuning_df, features, label)

    start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  optimization took {(time.time() - start) / 60:.1f} min")
    return study


def print_report(study, best_params, validation_f1, validation_scores,
                 tuning_df, validation_subjects, features, label):
    print("\n" + LINE)
    print("TUNING RESULTS")
    print(LINE)
    print(f"\n  {len(study.trials)} trials completed")
    print(f"  best inner CV macro-F1: {study.best_value:.4f}")
    print(f"  validation macro-F1 (held-out {len(validation_subjects)} subjects): "
          f"{validation_f1:.4f}")
    print(f"\n  tuning subjects: {tuning_df['subject'].nunique()}")
    print(f"  validation subjects: {sorted(validation_subjects)}")

    print("\n  BEST PARAMETERS:")
    for name, value in sorted(best_params.items()):
        shown = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"    {name:<22} {shown}")

    print("\n  Validation per subject:")
    for _, row in validation_scores.iterrows():
        print(f"    {row['subject']:<14} {row['macro_f1']:.3f}")

    default_score = TuningService.cross_validate(
        Constants.DEFAULT_XGB_PARAMS, tuning_df, features, label)
    print("\n" + LINE)
    print("COMPARISON TO DEFAULTS")
    print(LINE)
    print(f"  default parameters inner CV: {default_score:.4f}")
    print(f"  tuned parameters inner CV:   {study.best_value:.4f}")
    print(f"  improvement:                 {study.best_value - default_score:+.4f}")
    print(LINE + "\n")


def plot_search(study, path=os.path.join(Constants.FIGURES_DIR, "tuning.png")):
    values = [trial.value for trial in study.trials if trial.value is not None]
    best_so_far = np.maximum.accumulate(values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(range(len(values)), values, s=12, alpha=0.6, color="#2980b9")
    axes[0].plot(range(len(best_so_far)), best_so_far, color="#c0392b", lw=1.5,
                 label="best so far")
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel("macro-F1 (inner CV)")
    axes[0].set_title("Optimization history", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    importances = optuna.importance.get_param_importances(study)
    names = list(importances.keys())[:8]
    values = [importances[name] for name in names]
    axes[1].barh(names[::-1], values[::-1], color="#27ae60")
    axes[1].set_xlabel("importance")
    axes[1].set_title("Which hyperparameters matter", fontsize=11)
    axes[1].grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote " + path)


def main():
    args = parse_arguments()

    print("Loading...")
    df, features, label = DatasetService.load(args.features, args.classes)
    print(f"  {len(df):,} epochs, {len(features)} features, "
          f"{df['subject'].nunique()} subjects")

    tuning_df, validation_df, validation_subjects = TuningService.split_validation_subjects(
        df, args.n_val, args.seed)
    print(f"  tuning on {tuning_df['subject'].nunique()} subjects, "
          f"validating on {len(validation_subjects)}")
    print(f"  running {args.n_trials} Optuna trials...\n")

    study = run_search(tuning_df, features, label, args.n_trials, args.seed)
    best_params = study.best_params
    best_params["random_state"] = Constants.RANDOM_SEED

    validation_f1, validation_scores = TuningService.validate(
        best_params, tuning_df, validation_df, features, label)

    with open(args.out, "w") as file:
        json.dump(best_params, file, indent=2)
    TuningService.get_history(study).to_csv(os.path.join(Constants.TUNING_DIR, "tuning_history.csv"), index=False)

    print_report(study, best_params, validation_f1, validation_scores,
                 tuning_df, validation_subjects, features, label)
    plot_search(study)
    print(f"Wrote {args.out}, tuning_history.csv, tuning.png\n")


if __name__ == "__main__":
    main()
