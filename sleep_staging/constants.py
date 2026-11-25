# constants.py
#Every value shared across the project lives here.
#To change a setting for all experiments, change in this file only.

import os

import os



class Constants(object):

    
    # FOLDERS
    # Absolute paths, so every script works from any working directory.
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")                # raw nights (not in git)
    FEATURES_DIR = os.path.join(DATA_DIR, "features")        # extracted feature tables
    CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline")
    PERSONALIZATION_DIR = os.path.join(RESULTS_DIR, "personalization")
    ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation")
    ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
    TUNING_DIR = os.path.join(RESULTS_DIR, "tuning")
    FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
    PAPER_FIGURES_DIR = os.path.join(ROOT_DIR, "paper_figures")

    # FILES
    FEATURES_FILE = os.path.join(FEATURES_DIR, "features_v2.csv")
    WALCH_FEATURES_FILE = os.path.join(FEATURES_DIR, "walch_features.csv")
    PARAMS_FILE = os.path.join(CONFIGS_DIR, "xgb_best_params.json")

    # Written by train_baseline.py (tuned XGBoost, leave-one-subject-out).
    # Every prediction-based analysis reads this one file.
    PREDICTIONS_FILE = os.path.join(BASELINE_DIR, "baseline_predictions.csv")
    NIGHT_SCORES_FILE = os.path.join(ANALYSIS_DIR, "night_scores.csv")
    BASELINE_PER_SUBJECT_FILE = os.path.join(BASELINE_DIR, "baseline_per_subject.csv")
    SUBJECT_PROFILE_FILE = os.path.join(ANALYSIS_DIR, "subject_profile.csv")
    SUBJECT_CORRELATIONS_FILE = os.path.join(ANALYSIS_DIR, "subject_correlations.csv")
    GOOD_VS_BAD_FILE = os.path.join(ANALYSIS_DIR, "good_vs_bad.csv")

    # Main personalization result (clean control, all features), one row
    # per (arm, K, subject) with donors already averaged.
    PERSONALIZATION_PER_SUBJECT_FILE = os.path.join(
        PERSONALIZATION_DIR, "personalization_v2_final_all_per_subject.csv")

    
    # Raw data
    # A folder is one night if it contains all three of these files.
    NIGHT_FILES = ("motion.csv", "hr.csv", "labels.mat")

    EPOCH_SECONDS = 30
    LOCAL_TIMEZONE = "US/Eastern"     # recStart is stored as local time

    # Five AASM stages -> three classes. Wake 0, N1/N2/N3 -> NREM 1, REM 2.
    # Stage 5 (Unknown) is not in the map and is dropped.
    FIVE_TO_THREE_CLASS = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}

    
    # Our feature extraction (extract_features.py)
    # A sample counts as real movement if the magnitude jumps by more than
    # this between consecutive ~20 ms samples. Deep sleep sits far below it;
    # rolling over sits far above it.
    MOVE_THRESHOLD = 0.02

    HR_BASELINE_EPOCHS = 60           # 30-minute rolling heart-rate baseline
    MOVE_SHORT_WINDOW_EPOCHS = 10     # 5-minute rolling movement fraction
    MOVE_LONG_WINDOW_EPOCHS = 30      # 15-minute rolling movement fraction

    # Sleep drive peaks around 05:00 local time.
    SLEEP_DRIVE_PEAK_HOUR = 5.0

    # Features copied from neighbouring epochs, and how far away.
    CONTEXT_FEATURES = ("hr_zscore", "hr_vs_baseline", "hr_std",
                        "mag_std", "jerk_mean", "move_frac")
    CONTEXT_LAGS = (1, 2, 4)


    # Walch et al. (2019) features (extract_walch_features.py)
    # Values copied from their sleep_classifiers repository.
    WALCH_WINDOW_SECONDS = 10 * 30 - 15   # 285 s each side of the epoch
    WALCH_GAUSS_SIGMA = 50                # smooth_gauss in their utils.py
    WALCH_DOG_SIGMA_NARROW = 120          # convolve_with_dog in their utils.py
    WALCH_DOG_SIGMA_WIDE = 600
    WALCH_DOG_SCALAR = 0.75
    WALCH_BANDPASS_LOW_HZ = 0.25          # human movement frequencies
    WALCH_BANDPASS_HIGH_HZ = 3.0
    ACCELEROMETER_HZ = 50

    
    # Columns
    # Columns that identify an epoch or hold its label. Never model inputs.
    META_COLUMNS = ("subject", "night", "epoch", "stage3", "stage5")

    THREE_CLASS_LABEL = "stage3"
    FIVE_CLASS_LABEL = "stage5"

    STAGE_NAMES = {0: "Wake", 1: "NREM", 2: "REM"}
    FIVE_CLASS_STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    
    # Feature groups
    # Personal-reference features: computed relative to one subject's own
    # baseline (their median heart rate, their own movement rhythm).
    PERSONAL_BASE_FEATURES = (
        "hr_zscore",
        "hr_relative",
        "hr_vs_baseline",
        "epochs_since_move",
        "epochs_until_move",
    )

    # Context copies of a feature, taken from neighbouring epochs.
    CONTEXT_SUFFIXES = ("_prev1", "_prev2", "_prev4", "_next1", "_next2", "_next4")

    HEART_RATE_PREFIX = "hr_"
    MOTION_PREFIXES = (
        "mag_",
        "jerk_",
        "move_",
        "orient_",
        "accel_",
        "epochs_since",
        "epochs_until",
    )

    # Features that need the end of the recording or future epochs.
    # They cannot be computed while the person is still asleep.
    NON_CAUSAL_FEATURES = ("frac_through_night", "epochs_until_move")
    FUTURE_CONTEXT_MARKER = "_next"

    FEATURE_SETS = ("all", "personal", "nonpersonal", "hr", "motion", "time",
                    "hr_motion", "hr_time", "motion_time")

    # Names used in figures and tables.
    FEATURE_SET_LABELS = {
        "all": "All features",
        "personal": "Personal-reference",
        "nonpersonal": "Non-personal",
        "hr": "HR",
        "motion": "Motion",
        "time": "Time",
        "hr_motion": "HR + motion",
        "hr_time": "HR + time",
        "motion_time": "Motion + time",
    }
    SIGNAL_COLORS = {"hr": "#c0392b", "motion": "#2980b9", "time": "#27ae60"}
    SIGNAL_LABELS = {"hr": "heart rate", "motion": "motion", "time": "time / circadian"}

    
    # Per-subject heart-rate normalisation
    # hr_zscore and hr_relative are rebuilt from this column using only the
    # nights the model is allowed to see (prevents test-night leakage).
    HEART_RATE_SOURCE = "hr_mean"
    NORMALISED_HEART_RATE_FEATURES = ("hr_relative", "hr_zscore")

    
    # Personalization experiment
    RANDOM_SEED = 0
    MAX_PERSONAL_NIGHTS = 3       # K runs from 0 to this value
    DONORS_PER_SUBJECT = 3        # strangers averaged in the control arm
    QUICK_TEST_SUBJECTS = 8
    QUICK_TEST_DONORS = 2
    MINIMUM_PAIRED_SUBJECTS = 5   # fewer than this and a test is skipped

    ARMS = ("personal", "stranger")

    
    # Baseline (train_baseline.py)
    QUICK_BASELINE_SUBJECTS = 6

    # Walch et al. (2019), Table 6: best three-class result on their data
    # (31 subjects, one night each, thresholds tuned on ground truth).
    WALCH_PUBLISHED_ACCURACY = 0.723
    WALCH_PUBLISHED_KAPPA = 0.302

    # Hyperparameter tuning (tune_xgboost.py)
    TUNING_TRIALS = 80
    TUNING_VALIDATION_SUBJECTS = 8   # held out, never seen while tuning
    TUNING_INNER_FOLDS = 5           # grouped by subject
    DEFAULT_XGB_PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 0}

    
    # SHAP feature importance (run_shap.py)
    SHAP_SAMPLE_EPOCHS = 6000       # SHAP on every epoch would take hours
    SHAP_MIN_HELD_OUT_SUBJECTS = 3  # explained subjects are never trained on

    
    # Night-to-night stability (analyze_stability.py)
    MIN_NIGHTS_FOR_ICC = 3
    MIN_NIGHTS_FOR_SPLIT_HALF = 4
    SPLIT_HALF_REPEATS = 2000

    # Errors near stage transitions (analyze_transitions.py)
    # (lowest distance, highest distance, label), in epochs from the
    # nearest change in the true stage.
    TRANSITION_DISTANCE_BINS = (
        (0, 0, "at transition"),
        (1, 1, "1 epoch"),
        (2, 2, "2 epochs"),
        (3, 5, "3-5"),
        (6, 10, "6-10"),
        (11, 10 ** 9, ">10 (stable)"),
    )
    NEAR_TRANSITION_EPOCHS = 1     # "near" = at most this far from a change
    FAR_FROM_TRANSITION_EPOCHS = 5  # "far" = more than this

    
    # Good night vs bad night (compare_nights.py)
    MIN_NIGHTS_FOR_PAIRING = 2

    
    # Subject profiles and clustering (analyze_subjects.py)
    CLUSTER_FEATURES = ("transitions_per_hour", "hr_mean", "frac_rem",
                        "frac_wake", "stillness", "hours")
    CLUSTER_SIZES = (2, 3, 4, 5, 6)        # numbers of clusters to try
    CLUSTER_STABILITY_REPEATS = 200
    MIN_SUBJECTS_FOR_CORRELATION = 10

    
    # Who benefits from personalization? (check_heterogeneity.py)
    MODERATOR_PROPERTIES = ("transitions_per_hour", "hr_mean", "frac_rem",
                            "frac_wake", "stillness", "frag_sd")

    # Statistics
    SIGNIFICANCE_LEVEL = 0.05
    FALSE_DISCOVERY_RATE = 0.05   # Benjamini-Hochberg, for many features at once
    CONFIDENCE_LEVEL = 0.95