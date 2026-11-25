"""
context_feature_service.py

Sleep stages come in long runs, so neighbouring epochs carry a lot of
information. For a few key features, copy the value from 1, 2, and 4 epochs
before (_prev) and after (_next).

The _next copies look into the future, so the real-time variant drops them.
"""

import pandas as pd

from sleep_staging.constants import Constants


class ContextFeatureService(object):

    @staticmethod
    def add(features):
        copies = {}
        for column in Constants.CONTEXT_FEATURES:
            if column not in features.columns:
                continue
            for lag in Constants.CONTEXT_LAGS:
                copies[column + "_prev" + str(lag)] = features[column].shift(lag)
                copies[column + "_next" + str(lag)] = features[column].shift(-lag)

        context = pd.DataFrame(copies, index=features.index)
        # The first and last epochs of the night have no neighbour: fill
        # them from the nearest epoch that does.
        context = context.bfill().ffill()
        return pd.concat([features, context], axis=1)
