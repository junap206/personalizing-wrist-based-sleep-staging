"""
personalization_service.py

The core of the personalization experiment.

For one target subject and one donor:
    base training set = every subject except the target AND the donor

    personal arm = base + K of the target's own nights
    stranger arm = base + K of the donor's nights   (same amount of new data)

    Both arms are tested on the target's remaining nights.
    The gap between the arms is the benefit of the data being the target's own.
"""

import zlib

import numpy as np
import pandas as pd

from sleep_staging.constants import Constants
from sleep_staging.model_service import ModelService


class PersonalizationService(object):

    # ------------------------------------------------------------------
    # Randomness: one generator per subject
    # ------------------------------------------------------------------
    @staticmethod
    def get_subject_generator(seed, subject):
        """
        Seeded by subject name, so a subject gets the same donors and the
        same random night picks in every run, every feature set, and after
        every resume.
        """
        subject_code = zlib.crc32(str(subject).encode())
        return np.random.default_rng([seed, subject_code])

    @staticmethod
    def choose_donors(subject, donor_pool, n_donors, generator):
        candidates = []
        for donor in donor_pool:
            if donor != subject:
                candidates.append(donor)

        n_chosen = min(n_donors, len(candidates))
        chosen = generator.choice(candidates, size=n_chosen, replace=False)
        return list(chosen)

    # ------------------------------------------------------------------
    # Nights
    # ------------------------------------------------------------------
    @staticmethod
    def get_nights(df, subject):
        is_subject = df["subject"] == subject
        return sorted(df.loc[is_subject, "night"].unique())

    @staticmethod
    def pick_nights(nights, k, order, generator):
        """Choose which K nights go into training."""
        if k == 0:
            return []
        if order == "first":
            return list(nights[:k])
        return list(generator.choice(nights, size=k, replace=False))

    # ------------------------------------------------------------------
    # Leak-free heart-rate normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def rebuild_normalisation(df, visible_nights, subject):
        """
        Recompute hr_relative and hr_zscore for one subject using ONLY the
        nights the model is allowed to see.

        visible_nights empty (K=0): use the population median, so the model
        gets no information about this subject.
        """
        out = df.copy()
        source = Constants.HEART_RATE_SOURCE

        # Feature files without raw heart rate (e.g. Walch) need no rebuild.
        if source not in out.columns:
            return out

        is_subject = out["subject"] == subject

        if len(visible_nights) == 0:
            reference = out.loc[~is_subject, source].median()
            spread = out.loc[~is_subject, source].std()
        else:
            is_visible = is_subject & out["night"].isin(visible_nights)
            reference = out.loc[is_visible, source].median()
            spread = out.loc[is_visible, source].std()

        if not np.isfinite(spread) or spread == 0:
            spread = out[source].std()

        heart_rate = out.loc[is_subject, source]
        if "hr_relative" in out:
            out.loc[is_subject, "hr_relative"] = heart_rate - reference
        if "hr_zscore" in out:
            out.loc[is_subject, "hr_zscore"] = (heart_rate - reference) / (spread + 1e-6)
        return out

    # ------------------------------------------------------------------
    # One arm, one subject
    # ------------------------------------------------------------------
    @staticmethod
    def score_arm(data, base_rows, added_subject, added_nights, test_rows,
                  features, label, params):
        """Train on base + the added nights, score on the test nights."""
        is_added = (data["subject"] == added_subject) & data["night"].isin(added_nights)
        train = pd.concat([data[base_rows], data[is_added]], ignore_index=True)
        test = data[test_rows]
        return ModelService.train_and_score(train, test, features, label, params)

    @staticmethod
    def run_subject(df, features, label, subject, donors, k_values, order,
                    generator, params):
        """Every donor x every K x both arms, for one target subject."""
        subject_nights = PersonalizationService.get_nights(df, subject)
        rows = []

        for donor in donors:
            donor_nights = PersonalizationService.get_nights(df, donor)
            base_rows = ~df["subject"].isin([subject, donor])

            for k in k_values:
                # Need K nights to train on plus at least one night to test on.
                if len(subject_nights) <= k or len(donor_nights) < k:
                    continue

                train_nights = PersonalizationService.pick_nights(
                    np.array(subject_nights), k, order, generator)
                test_nights = [n for n in subject_nights if n not in train_nights]
                test_rows = (df["subject"] == subject) & df["night"].isin(test_nights)

                # Personal arm: target's own K nights, normalised from them.
                data = PersonalizationService.rebuild_normalisation(
                    df, train_nights, subject)
                scores = PersonalizationService.score_arm(
                    data, base_rows, subject, train_nights, test_rows,
                    features, label, params)
                rows.append(PersonalizationService.make_row(
                    "personal", k, subject, donor, len(test_nights), scores))

                # Stranger arm: donor's K nights; target has seen nothing of
                # this subject, so it gets population normalisation.
                donor_train_nights = PersonalizationService.pick_nights(
                    np.array(donor_nights), k, order, generator)
                data = PersonalizationService.rebuild_normalisation(
                    df, [], subject)
                scores = PersonalizationService.score_arm(
                    data, base_rows, donor, donor_train_nights, test_rows,
                    features, label, params)
                rows.append(PersonalizationService.make_row(
                    "stranger", k, subject, donor, len(test_nights), scores))

        return rows

    @staticmethod
    def make_row(arm, k, subject, donor, n_test_nights, scores):
        row = {
            "arm": arm,
            "k": k,
            "subject": subject,
            "donor": donor,
            "n_test_nights": n_test_nights,
        }
        row.update(scores)
        return row