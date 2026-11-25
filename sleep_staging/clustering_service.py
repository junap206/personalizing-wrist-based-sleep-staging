"""
clustering_service.py

Do subjects fall into a few distinct sleep types

k-means always returns clusters, even from random data. So the structure
is checked against a null: the same procedure on data whose columns have
been shuffled independently, which destroys any real grouping..
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sleep_staging.constants import Constants


class ClusteringService(object):

    @staticmethod
    def standardise(profile):
        """Cluster features on a common scale. Returns (subject index, matrix)."""
        data = profile[list(Constants.CLUSTER_FEATURES)].dropna()
        return data.index, StandardScaler().fit_transform(data)

    @staticmethod
    def cluster(matrix, k, n_init=20, seed=Constants.RANDOM_SEED):
        return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(matrix)

    @staticmethod
    def silhouette_by_k(matrix):
        """Silhouette score (higher = cleaner separation) for each k."""
        scores = {}
        for k in Constants.CLUSTER_SIZES:
            labels = ClusteringService.cluster(matrix, k)
            scores[k] = silhouette_score(matrix, labels)
        return scores

    @staticmethod
    def assign(profile, subjects, matrix, k):
        labelled = profile.copy()
        labelled["cluster"] = np.nan
        labelled.loc[subjects, "cluster"] = ClusteringService.cluster(matrix, k)
        return labelled

    @staticmethod
    def compare_to_shuffled(matrix, k, repeats=Constants.CLUSTER_STABILITY_REPEATS,
                            seed=Constants.RANDOM_SEED):
        """
        Mean silhouette on random halves of the real data, and on the same
        halves with every column shuffled independently. If the real data
        does not beat the shuffled data, the clusters are not real.
        """
        generator = np.random.default_rng(seed)
        real_scores = []
        shuffled_scores = []
        n_subjects = len(matrix)

        for _ in range(repeats):
            half = matrix[generator.permutation(n_subjects)[:n_subjects // 2]]
            if len(np.unique(half, axis=0)) <= k:
                continue
            labels = ClusteringService.cluster(half, k, n_init=10, seed=0)
            real_scores.append(silhouette_score(half, labels))

            columns = [generator.permutation(half[:, j]) for j in range(half.shape[1])]
            shuffled = np.column_stack(columns)
            labels = ClusteringService.cluster(shuffled, k, n_init=10, seed=0)
            shuffled_scores.append(silhouette_score(shuffled, labels))

        return np.mean(real_scores), np.mean(shuffled_scores)
