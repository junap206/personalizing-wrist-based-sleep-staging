"""
dataset_service.py

Loads the feature table that extract_features.py wrote, and chooses the
label column for the three- or five-class problem
"""

import pandas as pd

from sleep_staging.constants import Constants
from sleep_staging.feature_set_service import FeatureSetService


class DatasetService(object):

    @staticmethod
    def get_label_column(classes):
        if classes == 3:
            return Constants.THREE_CLASS_LABEL
        return Constants.FIVE_CLASS_LABEL

    @staticmethod
    def get_stage_names(classes):
        if classes == 3:
            return Constants.STAGE_NAMES
        return Constants.FIVE_CLASS_STAGE_NAMES

    @staticmethod
    def load(features_file, classes=3):
        """Return the table, its feature columns, and the label column."""
        label = DatasetService.get_label_column(classes)
        df = pd.read_csv(features_file)
        df = df[df[label] >= 0].reset_index(drop=True)
        features = FeatureSetService.get_model_features(df.columns)
        return df, features, label

    @staticmethod
    def get_subjects(df):
        return sorted(df["subject"].unique())
