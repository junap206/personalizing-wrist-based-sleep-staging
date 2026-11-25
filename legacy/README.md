# Legacy scripts

Earlier, self-contained versions of experiments, kept so past results can be traced.
They are superseded by the scripts in `scripts/` and are not maintained: they still expect
to be run from a folder holding `features_v2.csv` and `xgb_best_params.json`.

| Script | Replaced by |
|---|---|
| `run_personalization.py` | `scripts/experiments/run_personalization_v2.py` |
| `run_feature_split.py`, `run_nonpersonal_only.py` | `run_personalization_v2.py --feature-set ...` and `scripts/experiments/run_ablation.py` |
| `run_ablation_by_k.py` | `scripts/plotting/plot_ablation_by_k.py` (from the `run_all_personalization.bat` runs) |
| `check_heterogemeity.py` | `scripts/analysis/check_heterogeneity.py` |
