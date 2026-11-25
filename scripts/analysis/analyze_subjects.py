"""
analyze_subjects.py

Are some PEOPLE hard to score, and do people fall into distinct sleep types?

    1. Which subject properties go with a low baseline score?
    2. Do subjects form clusters (checked against shuffled data)?
    3. Do any clusters differ in how well the model scores them?

Examples:
    python scripts/analysis/analyze_subjects.py --plot
    python scripts/analysis/analyze_subjects.py --k 4

Needs: features_v2.csv, baseline_per_subject.csv, night_scores.csv
Outputs: subject_profile.csv, subject_correlations.csv, subject_clusters.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.clustering_service import ClusteringService  # noqa: E402
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.figure_service import FigureService  # noqa: E402
from sleep_staging.subject_profile_service import SubjectProfileService  # noqa: E402

LINE = "=" * 72
CLUSTER_COLORS = ("#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#f39c12", "#16a085")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=Constants.FEATURES_FILE)
    parser.add_argument("--baseline", default=Constants.BASELINE_PER_SUBJECT_FILE)
    parser.add_argument("--scores", default=Constants.NIGHT_SCORES_FILE)
    parser.add_argument("--k", type=int, default=None, help="force a number of clusters")
    parser.add_argument("--out", default=Constants.SUBJECT_PROFILE_FILE)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def print_heading(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def print_correlations(correlations, n_subjects):
    print_heading("PART 1 -- WHAT PREDICTS A HARD SUBJECT?")
    print(f"  {n_subjects} subjects, each property against baseline macro-F1.\n")
    print(f"  {'property':<26}{'r':>8}{'rho':>8}{'p':>10}")
    print("  " + "-" * 52)
    for _, row in correlations.head(10).iterrows():
        marker = "*" if row["significant"] else " "
        print(f" {marker}{row['feature']:<26}{row['pearson_r']:>8.3f}"
              f"{row['spearman_rho']:>8.3f}{row['p']:>10.4f}")
    print("\n  * survives Benjamini-Hochberg correction (FDR 5%).")


def print_clusters(silhouettes, k, shuffled_check):
    print_heading("PART 2 -- DO SUBJECTS FORM TYPES?")
    print("  Silhouette by number of clusters (higher = cleaner separation):")
    for size, score in silhouettes.items():
        marker = "   <- used" if size == k else ""
        print(f"    k={size}: {score:.3f}{marker}")
    real, shuffled = shuffled_check
    print(f"\n  Check at k={k}:")
    print(f"    silhouette on random halves of the real data : {real:.3f}")
    print(f"    same with every column shuffled (no structure): {shuffled:.3f}")
    if real > shuffled + 0.05:
        print("  -> Real data clusters better than shuffled data.")
    else:
        print("  -> No better than shuffled data: the clusters are not real structure.")


def get_cluster_groups(profile):
    clustered = profile.dropna(subset=["cluster"])
    labels = []
    groups = []
    for cluster, subjects in clustered.groupby("cluster"):
        labels.append(f"cluster {int(cluster)}\n(n={len(subjects)})")
        groups.append(subjects["macro_f1"].values)
    return labels, groups


def test_clusters(groups):
    """One-way ANOVA of baseline macro-F1 across clusters."""
    testable = [values for values in groups if len(values) > 1]
    if len(testable) < 2:
        return None
    return stats.f_oneway(*testable)


def print_cluster_difficulty(profile, k, anova):
    print_heading(f"PART 3 -- DO THE {k} CLUSTERS DIFFER IN DIFFICULTY?")
    clustered = profile.dropna(subset=["cluster"])
    summary = clustered.groupby("cluster").agg(
        n=("macro_f1", "size"),
        macro_f1=("macro_f1", "mean"),
        frag=("transitions_per_hour", "mean"),
        hr=("hr_mean", "mean"),
        rem=("frac_rem", "mean"),
        wake=("frac_wake", "mean"),
        still=("stillness", "mean"),
    ).round(3)
    print()
    print(summary.to_string())
    if anova is not None:
        print(f"\n  One-way ANOVA on macro-F1 across clusters: "
              f"F={anova.statistic:.2f}, p={anova.pvalue:.4f}")
    print(LINE + "\n")


def plot_subjects(profile, correlations, k, anova, path=os.path.join(Constants.FIGURES_DIR, "subject_clusters.png")):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    strongest = correlations.iloc[0]
    axes[0].scatter(profile[strongest["feature"]], profile["macro_f1"], s=30, color="#2c3e50")
    axes[0].set_xlabel(strongest["feature"])
    axes[0].set_ylabel("baseline macro-F1")
    axes[0].set_title(f"Strongest subject-level correlate\n"
                      f"r={strongest['pearson_r']:+.3f}, p={strongest['p']:.3g}", fontsize=10)
    axes[0].grid(alpha=0.25)

    clustered = profile.dropna(subset=["cluster"])
    for cluster, subjects in clustered.groupby("cluster"):
        axes[1].scatter(subjects["transitions_per_hour"], subjects["hr_mean"], s=35,
                        color=CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)],
                        label=f"cluster {int(cluster)} (n={len(subjects)})")
    axes[1].set_xlabel("transitions per hour")
    axes[1].set_ylabel("mean heart rate")
    axes[1].set_title(f"k-means, {k} clusters", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    labels, groups = get_cluster_groups(profile)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(groups))]
    FigureService.boxplot_with_points(axes[2], groups, labels, colors)
    axes[2].set_ylabel("baseline macro-F1")
    anova_text = f"ANOVA p={anova.pvalue:.3g}" if anova is not None else ""
    axes[2].set_title("Difficulty by cluster\n" + anova_text, fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote " + path)


def main():
    args = parse_arguments()
    print("Loading...")
    features = pd.read_csv(args.features)
    night_scores = pd.read_csv(args.scores)
    baseline = pd.read_csv(args.baseline)

    profile = SubjectProfileService.build(features, night_scores, baseline)
    print(f"  {len(profile)} subjects profiled")
    correlations = SubjectProfileService.correlate_with_difficulty(profile)

    subjects, matrix = ClusteringService.standardise(profile)
    silhouettes = ClusteringService.silhouette_by_k(matrix)
    k = args.k if args.k else max(silhouettes, key=silhouettes.get)
    profile = ClusteringService.assign(profile, subjects, matrix, k)
    shuffled_check = ClusteringService.compare_to_shuffled(matrix, k)
    anova = test_clusters(get_cluster_groups(profile)[1])

    profile.to_csv(args.out)
    correlations.to_csv(Constants.SUBJECT_CORRELATIONS_FILE, index=False)

    print_correlations(correlations, len(profile))
    print_clusters(silhouettes, k, shuffled_check)
    print_cluster_difficulty(profile, k, anova)
    if args.plot:
        plot_subjects(profile, correlations, k, anova)
    print(f"Wrote {args.out} and subject_correlations.csv\n")


if __name__ == "__main__":
    main()
