"""
night_score_service.py

Loads a predictions file (one row per epoch: subject, night, epoch, true,
pred) and scores every night separately.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

REQUIRED_COLUMNS = {"subject", "night", "epoch", "true", "pred"}
WAKE = 0
REM = 2


class NightScoreService(object):

    @staticmethod
    def load_predictions(path):
        predictions = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(predictions.columns)
        if missing:
            raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
        return predictions

    @staticmethod
    def score_nights(predictions):
        """One row per subject-night. Nights with a single stage are skipped over. """
        rows = []
        for (subject, night), epochs in predictions.groupby(["subject", "night"]):
            if epochs["true"].nunique() < 2:
                continue
            rows.append(NightScoreService.score_night(subject, night, epochs))
        return pd.DataFrame(rows)

    @staticmethod
    def score_night(subject, night, epochs):
        true = epochs["true"]
        predicted = epochs["pred"]
        return {
            "subject": subject,
            "night": night,
            "macro_f1": f1_score(true, predicted, average="macro"),
            "kappa": cohen_kappa_score(true, predicted),
            "accuracy": (true == predicted).mean(),
            "n_epochs": len(epochs),
            "rem_frac": (true == REM).mean(),
            "wake_frac": (true == WAKE).mean(),
        }
