"""
statistics_service.py

Statistical tests shared by several analyses.
"""

import numpy as np
from scipy import stats

from sleep_staging.constants import Constants


class StatisticsService(object):

    @staticmethod
    def benjamini_hochberg(table, p_column, rate=Constants.FALSE_DISCOVERY_RATE):
        """
        Mark which tests survive the Benjamini-Hochberg correction.

        When many features are tested at once, some look significant by
        chance. Sort the p-values; the i-th smallest of m must be at most
        (i / m) * rate. Every test up to the last one that passes is
        significant.

        Returns the table sorted by p, with 'bh_threshold' and 'significant'.
        """
        table = table.sort_values(p_column).reset_index(drop=True)
        n_tests = len(table)
        table["bh_threshold"] = (table.index + 1) / n_tests * rate
        passes = table[p_column] <= table["bh_threshold"]
        table["significant"] = passes
        if passes.any():
            last_passing = table.index[passes].max()
            table["significant"] = table.index <= last_passing
        return table

    @staticmethod
    def paired_comparison(first, second):
        """
        Paired t-test and Wilcoxon test of first vs second (same subjects),
        plus Cohen's d for paired data. Returns None if it cannot be tested.
        """
        usable = np.isfinite(first) & np.isfinite(second)
        first = first[usable]
        second = second[usable]
        if len(first) < Constants.MINIMUM_PAIRED_SUBJECTS or np.allclose(first, second):
            return None

        difference = first - second
        spread = difference.std(ddof=1)
        cohens_d = difference.mean() / spread if spread > 0 else 0.0

        _, t_p = stats.ttest_rel(first, second)
        try:
            _, wilcoxon_p = stats.wilcoxon(first, second)
        except ValueError:
            wilcoxon_p = np.nan

        return {
            "difference": difference.mean(),
            "cohens_d": cohens_d,
            "t_p": t_p,
            "wilcoxon_p": wilcoxon_p,
            "n": len(first),
        }
