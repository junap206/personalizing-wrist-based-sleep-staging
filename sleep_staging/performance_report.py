"""
performance_report.py

Prints how well a model did: overall scores, spread across subjects,
per-class F1, and the confusion matrix.
"""

from sklearn.metrics import confusion_matrix, f1_score

from sleep_staging.constants import Constants

LINE = "=" * 64


class PerformanceReport(object):

    @staticmethod
    def print_overall(per_subject, title):
        print("\n" + LINE)
        print(title)
        print(LINE)
        f1 = per_subject["macro_f1"]
        kappa = per_subject["kappa"]
        print(f"\n  macro-F1   {f1.mean():.3f}  +/- {f1.std():.3f}")
        print(f"  kappa      {kappa.mean():.3f}  +/- {kappa.std():.3f}")
        print(f"  accuracy   {per_subject['accuracy'].mean():.3f}")

    @staticmethod
    def print_spread(per_subject):
        """How different subjects are from each other."""
        f1 = per_subject["macro_f1"]
        worst = per_subject.loc[f1.idxmin(), "subject"]
        best = per_subject.loc[f1.idxmax(), "subject"]

        print("\nSPREAD ACROSS SUBJECTS")
        print(f"  worst    {f1.min():.3f}   ({worst})")
        print(f"  25th     {f1.quantile(.25):.3f}")
        print(f"  median   {f1.median():.3f}")
        print(f"  75th     {f1.quantile(.75):.3f}")
        print(f"  best     {f1.max():.3f}   ({best})")
        print(f"  range    {f1.max() - f1.min():.3f}")

    @staticmethod
    def print_hardest(per_subject, n=5):
        print(f"\nHARDEST {n} SUBJECTS")
        for _, row in per_subject.nsmallest(n, "macro_f1").iterrows():
            print(f"  {row['subject']:<12} {row['macro_f1']:.3f}   "
                  f"({int(row['n_nights'])} nights)")

    @staticmethod
    def print_per_class(predictions, stage_names):
        """F1 for each stage, pooling every epoch from every subject."""
        codes = sorted(stage_names)
        per_class = f1_score(predictions["true"], predictions["pred"],
                             average=None, labels=codes)
        print("\nPER-CLASS F1  (all epochs pooled)")
        for code, value in zip(codes, per_class):
            print(f"  {stage_names[code]:<5} {value:.3f}")

    @staticmethod
    def print_confusion(predictions, stage_names):
        """Rows = true stage, columns = predicted stage, % of each row."""
        codes = sorted(stage_names)
        counts = confusion_matrix(predictions["true"], predictions["pred"], labels=codes)
        percent = counts / counts.sum(axis=1, keepdims=True) * 100

        print("\nCONFUSION MATRIX  (rows = truth, columns = predicted, % of row)")
        header = "        "
        for code in codes:
            header += f"{stage_names[code]:>8}"
        print(header)
        for code, row in zip(codes, percent):
            line = f"  {stage_names[code]:<6}"
            for value in row:
                line += f"{value:7.1f}%"
            print(line)

    @staticmethod
    def print_published_reference():
        print("\n" + LINE)
        print("Reference: Walch et al. (2019), best three-class result on their")
        print(f"data: accuracy {Constants.WALCH_PUBLISHED_ACCURACY:.3f}, "
              f"kappa {Constants.WALCH_PUBLISHED_KAPPA:.3f}. Different cohort,")
        print("and their thresholds were tuned on ground truth.")
        print(LINE + "\n")
