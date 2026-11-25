# Sleep Staging Personalization

## Target Question: Does a wrist-based sleep staging model improve if it has seen a few of a person's prior nights?

This project investigates whether subject-specific training data improves automated sleep staging from Apple Watch signals, using a volume-matched stranger control to
isolate the effect of personal data from generic data augmentation.

## Dataset
- 47 healthy adults, 253 nights (3–7 each)
- Apple Watch: heart rate (~0.2 Hz) + 3-axis acceleration (~50 Hz)
- Dreem 2 EEG headband: expert-reviewed AASM sleep stages (30-second epochs)
- Source: [PhysioNet Sleep-Accel](https://physionet.org/content/sleep-accel/)

## Model
- **XGBoost** with Optuna-tuned hyperparameters
- **68 features:** heart rate (14), motion (15), time (6), context copies (36)
- **Evaluation:** leave-one-subject-out, macro-F1 primary, Cohen's kappa secondary
- **Three-class:** Wake / NREM / REM

## Repository layout

```
sleep_staging/          the core library: model, features, statistics (imported by every script)
  constants.py          every shared setting and every file/folder path
  *_service.py          one service per job (features, LOSO training, personalization, SHAP, ...)
scripts/                runnable entry points, grouped by pipeline stage
  features/             raw nights -> feature tables
  training/             hyperparameter tuning, LOSO baseline, model comparison
  experiments/          personalization (main experiment), ablation, 2x2, SHAP
  analysis/             follow-up analyses and sanity checks (stability, transitions, labels, ...)
  plotting/             figure generation, including make_paper_figures.py
configs/                tuned XGBoost hyperparameters (xgb_best_params.json)
results/                output tables (CSVs are git-ignored; run configs are kept)
  baseline/ personalization/ ablation/ analysis/ tuning/
figures/                working figures from each script
paper_figures/          final publication figures (from scripts/plotting/make_paper_figures.py)
legacy/                 earlier versions of experiments, kept for reference only (not maintained)
data/                   raw PhysioNet nights + data/features/ tables (git-ignored, see below)
```

Every path lives in `sleep_staging/constants.py`, so scripts can be run from any folder and
always read from / write to the folders above.

## Reproducing the results

```
pip install numpy pandas scipy scikit-learn xgboost optuna shap statsmodels matplotlib pytz
```

1. Download the dataset from PhysioNet into `data/` (one folder per subject, one subfolder per night).
2. Extract features:
   `python scripts/features/extract_features.py data --out data/features/features_v2.csv`
   `python scripts/features/extract_walch_features.py data`
3. Tune and train the baseline:
   `python scripts/training/tune_xgboost.py`
   `python scripts/training/train_baseline.py`
4. Run the personalization experiments (all nine runs, resumable):
   `scripts\experiments\run_all_personalization.bat`
5. Run the analyses in `scripts/analysis/` (e.g. `analyze_stability.py`, `check_heterogeneity.py`).
6. Redraw every paper figure from the saved results:
   `python scripts/plotting/make_paper_figures.py`

Each script's docstring describes its inputs, outputs and options (`--help` lists them).
