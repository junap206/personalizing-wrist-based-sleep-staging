"""
feature_set_service.py

Decides which feature columns belong to which feature set.

Classification rule used in the paper:
    A feature is PERSONAL if computing it requires subtracting or referencing
    a statistic specific to one subject (their median heart rate, their own
    movement rhythm). Every other feature is NON-PERSONAL.
"""

from sleep_staging.constants import Constants


class FeatureSetService(object):

    @staticmethod
    def get_model_features(columns):
        """All columns except identifiers and labels."""
        features = []
        for column in columns:
            if column not in Constants.META_COLUMNS:
                features.append(column)
        return features

    @staticmethod
    def get_base_name(feature):
        """'hr_zscore_prev2' -> 'hr_zscore'."""
        base_name = feature
        for suffix in Constants.CONTEXT_SUFFIXES:
            base_name = base_name.replace(suffix, "")
        return base_name

    @staticmethod
    def is_personal(feature):
        base_name = FeatureSetService.get_base_name(feature)
        return base_name in Constants.PERSONAL_BASE_FEATURES

    @staticmethod
    def get_signal_source(feature):
        """Return 'hr', 'motion', or 'time'."""
        base_name = FeatureSetService.get_base_name(feature)
        if base_name.startswith(Constants.HEART_RATE_PREFIX):
            return "hr"
        if base_name.startswith(Constants.MOTION_PREFIXES):
            return "motion"
        return "time"

    @staticmethod
    def select(features, set_name):
        """Return the subset of features named by set_name."""
        if set_name == "all":
            return features

        selected = []
        for feature in features:
            if FeatureSetService.belongs_to(feature, set_name):
                selected.append(feature)
        return selected

    @staticmethod
    def belongs_to(feature, set_name):
        source = FeatureSetService.get_signal_source(feature)
        personal = FeatureSetService.is_personal(feature)

        if set_name == "personal":
            return personal
        if set_name == "nonpersonal":
            return not personal
        if set_name == "hr":
            return source == "hr"
        if set_name == "motion":
            return source == "motion"
        if set_name == "time":
            return source == "time"
        if set_name == "hr_motion":
            return source in ("hr", "motion")
        if set_name == "hr_time":
            return source in ("hr", "time")
        if set_name == "motion_time":
            return source in ("motion", "time")
        raise ValueError("Unknown feature set: " + str(set_name))

    @staticmethod
    def is_context(feature):
        """True for copies taken from neighbouring epochs (_prev / _next)."""
        return FeatureSetService.get_base_name(feature) != feature

    @staticmethod
    def is_causal(feature):
        """True if the feature can be computed while the person is asleep."""
        if feature in Constants.NON_CAUSAL_FEATURES:
            return False
        if Constants.FUTURE_CONTEXT_MARKER in feature:
            return False
        return True

    @staticmethod
    def keep_causal(features):
        """Real-time variant: drop features that look into the future."""
        kept = []
        dropped = []
        for feature in features:
            if FeatureSetService.is_causal(feature):
                kept.append(feature)
            else:
                dropped.append(feature)
        return kept, dropped
