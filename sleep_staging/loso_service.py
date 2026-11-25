"""
loso_service.py


Each subject in turn is removed completely. The
model trains on everyone else and is scored on the held-out subject. The
model has therefore never seen the person it scores, which is how a new
user of a wearable would experience it.
"""

import time

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from sleep_staging.model_service import ModelService


class LosoService(object):

    @staticmethod
    def run(df, features, label, subjects, params, verbose=True, eval_label=None):
        """
        Train on `label`. Score on `eval_label` if given (e.g. train on the
        automatic labels, score against the expert labels); epochs missing
        either label (-1) are left out of that side.

        Returns:
            per_subject   one row per held-out subject
            predictions   one row per epoch: subject, night, epoch, true, pred
        """
        subject_rows = []
        prediction_tables = []

        for number, held_out in enumerate(subjects, 1):
            start = time.time()
            is_held_out = df["subject"] == held_out
            train = df[~is_held_out]
            test = df[is_held_out]
            score_label = label
            if eval_label is not None:
                train = train[train[label] >= 0]
                test = test[test[eval_label] >= 0]
                score_label = eval_label

            model = ModelService.build(params)
            model.fit(train[features], train[label])
            predicted = model.predict(test[features])
            true = test[score_label].values

            row = LosoService.score_subject(held_out, test, true, predicted)
            subject_rows.append(row)
            prediction_tables.append(
                LosoService.make_prediction_table(held_out, test, true, predicted))

            if verbose:
                seconds = time.time() - start
                print(f"[{number:3d}/{len(subjects)}] {held_out:<12} "
                      f"macro-F1 {row['macro_f1']:.3f}   kappa {row['kappa']:.3f}   "
                      f"acc {row['accuracy']:.3f}   ({seconds:.0f}s)")

        per_subject = pd.DataFrame(subject_rows)
        predictions = pd.concat(prediction_tables, ignore_index=True)
        return per_subject, predictions

    @staticmethod
    def score_subject(subject, test, true, predicted):
        return {
            "subject": subject,
            "macro_f1": f1_score(true, predicted, average="macro"),
            "kappa": cohen_kappa_score(true, predicted),
            "accuracy": (predicted == true).mean(),
            "n_epochs": len(test),
            "n_nights": test["night"].nunique(),
        }

    @staticmethod
    def make_prediction_table(subject, test, true, predicted):
        """Every epoch with its prediction, for later error analysis."""
        return pd.DataFrame({
            "subject": subject,
            "night": test["night"].values,
            "epoch": test["epoch"].values,
            "true": true,
            "pred": predicted,
        })
