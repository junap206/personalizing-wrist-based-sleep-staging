"""
stability_service.py

Is a subject's difficulty a stable trait of that person, or night-to-night
noise? Three measures, each from per-night scores:

    ICC         share of the variance that comes from who the person is
    split-half  does the average of half a subject's nights predict the
                average of their other half?
    headroom    how much higher the mean would be if every subject scored
                like their own best night (an optimistic ceiling)
"""

import numpy as np

from sleep_staging.constants import Constants


class StabilityService(object):

    @staticmethod
    def keep_subjects_with(night_scores, min_nights):
        counts = night_scores.groupby("subject").size()
        keep = counts[counts >= min_nights].index
        return night_scores[night_scores["subject"].isin(keep)]

    @staticmethod
    def intraclass_correlation(night_scores, column="macro_f1",
                               min_nights=Constants.MIN_NIGHTS_FOR_ICC):
        """
        One-way ICC from between- and within-subject mean squares.
        ICC = (MS_between - MS_within) / (MS_between + (k - 1) MS_within),
        with k the mean number of nights per subject.
        """
        data = StabilityService.keep_subjects_with(night_scores, min_nights)
        if data["subject"].nunique() < 3:
            return None

        grand_mean = data[column].mean()
        by_subject = data.groupby("subject")[column]
        subject_means = by_subject.mean()
        nights_per_subject = by_subject.size()
        k = nights_per_subject.mean()

        ss_between = (nights_per_subject * (subject_means - grand_mean) ** 2).sum()
        ms_between = ss_between / (len(subject_means) - 1)

        squared_deviations = []
        for subject, nights in data.groupby("subject"):
            squared_deviations.append(((nights[column] - subject_means[subject]) ** 2).sum())
        ss_within = sum(squared_deviations)
        ms_within = ss_within / (len(data) - len(subject_means))

        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        return {
            "icc": icc,
            "between_sd": np.sqrt(max(ms_between - ms_within, 0) / k),
            "within_sd": np.sqrt(ms_within),
            "n_subjects": len(subject_means),
            "n_nights": len(data),
            "mean_nights": k,
        }

    @staticmethod
    def split_half(night_scores, column="macro_f1",
                   repeats=Constants.SPLIT_HALF_REPEATS, seed=Constants.RANDOM_SEED):
        """
        Repeatedly split each subject's nights at random into two halves,
        then correlate first-half means with second-half means across
        subjects. Returns the mean correlation and its 95% interval.
        """
        counts = night_scores.groupby("subject").size()
        subjects = counts[counts >= Constants.MIN_NIGHTS_FOR_SPLIT_HALF].index
        if len(subjects) < 5:
            return None

        generator = np.random.default_rng(seed)
        correlations = []
        for _ in range(repeats):
            first_halves = []
            second_halves = []
            for subject in subjects:
                values = night_scores.loc[night_scores["subject"] == subject, column].values
                order = generator.permutation(len(values))
                half = len(values) // 2
                first_halves.append(values[order[:half]].mean())
                second_halves.append(values[order[half:2 * half]].mean())
            if np.std(first_halves) > 0 and np.std(second_halves) > 0:
                correlations.append(np.corrcoef(first_halves, second_halves)[0, 1])

        correlations = np.array(correlations)
        return {
            "mean_r": correlations.mean(),
            "lo": np.percentile(correlations, 2.5),
            "hi": np.percentile(correlations, 97.5),
            "n_subjects": len(subjects),
        }

    @staticmethod
    def headroom(night_scores):
        per_subject = night_scores.groupby("subject")["macro_f1"].agg(["mean", "max"])
        current = per_subject["mean"].mean()
        best = per_subject["max"].mean()
        return {"current_mean": current, "if_best_night": best, "gain": best - current}

    @staticmethod
    def consistency_by_subject(night_scores, min_nights=Constants.MIN_NIGHTS_FOR_ICC):
        """Mean, spread, and range of each subject's nightly scores."""
        per_subject = night_scores.groupby("subject")["macro_f1"].agg(
            ["mean", "std", "min", "max", "count"])
        return per_subject[per_subject["count"] >= min_nights]
