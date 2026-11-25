"""
personalization_report.py

Prints the results table and draws the personalization curves.
"""

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sleep_staging.constants import Constants  # noqa: E402

LINE = "=" * 74
ARM_COLORS = {"personal": "#27ae60", "stranger": "#c0392b"}
GREY = "#7f8c8d"


class PersonalizationReport(object):

    @staticmethod
    def significance_stars(p_value, n_tests):
        """Stars after Bonferroni correction for n_tests comparisons."""
        corrected = p_value * n_tests
        if corrected < 0.001:
            return "***"
        if corrected < 0.01:
            return "**"
        if corrected < Constants.SIGNIFICANCE_LEVEL:
            return "*"
        return "n.s."

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------
    @staticmethod
    def print_report(summary, tests, dose_response, dropped, order):
        n_tests = max(1, len(tests))
        PersonalizationReport.print_curve(summary, dropped, order)
        PersonalizationReport.print_tests(tests, n_tests)
        PersonalizationReport.print_dose_response(dose_response)
        PersonalizationReport.print_verdict(tests, n_tests)

    @staticmethod
    def print_curve(summary, dropped, order):
        print("\n" + LINE)
        print("PERSONALIZATION -- CLEAN CONTROL")
        print(LINE)
        print("\n  Base = all subjects except target AND donor. Both arms add K novel")
        print("  nights: the target's (personal) or the donor's (stranger).")
        print("  Night order: " + order + ". Same subjects at every K.")
        if dropped:
            print("  REAL-TIME variant: dropped " + str(len(dropped))
                  + " non-causal features.")

        print(f"\n  {'arm':<11}{'K':>3}{'n':>5}{'macro-F1':>11}{'kappa':>9}{'REM F1':>9}")
        print("  " + "-" * 50)
        for arm in Constants.ARMS:
            arm_rows = summary[summary["arm"] == arm].sort_values("k")
            for _, row in arm_rows.iterrows():
                print(f"  {arm:<11}{int(row['k']):>3}{int(row['n_subjects']):>5}"
                      f"{row['macro_f1']:>11.3f}{row['kappa']:>9.3f}"
                      f"{row['f1_REM']:>9.3f}")
            print()

    @staticmethod
    def print_tests(tests, n_tests):
        threshold = Constants.SIGNIFICANCE_LEVEL / n_tests

        print(LINE)
        print("PERSONAL vs STRANGER  (paired within subject, donors averaged)")
        print(LINE)
        print(f"\n  Bonferroni threshold for {n_tests} K values: p < {threshold:.4f}\n")
        print(f"  {'K':>3}{'n':>5}{'gap':>9}{'95% CI':>18}{'d':>7}{'t p':>10}"
              f"{'Wilcoxon p':>12}{'improved':>10}")
        print("  " + "-" * 74)

        for _, row in tests.iterrows():
            marker = "*" if row["t_p"] < threshold else " "
            print(f" {marker}{int(row['k']):>3}{int(row['n']):>5}{row['gap']:>+9.3f}"
                  f"   [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]"
                  f"{row['cohens_d']:>7.2f}{row['t_p']:>10.4f}"
                  f"{row['wilcoxon_p']:>12.4f}"
                  f"{int(row['n_improved']):>6}/{int(row['n'])}")
        print("\n  * survives Bonferroni.")

    @staticmethod
    def print_dose_response(dose_response):
        slope, p_value = dose_response
        print("\n  Mixed model (macro_f1 ~ K x arm, random intercept per subject):")
        print(f"    K x personal = {slope:+.4f} per night,  p = {p_value:.4f}")

    @staticmethod
    def print_verdict(tests, n_tests):
        print("\n" + LINE)
        print("VERDICT")
        print(LINE)
        if len(tests) == 0:
            print("  No tests could be run.")
            print(LINE + "\n")
            return

        last = tests[tests["k"] == tests["k"].max()].iloc[0]
        k = int(last["k"])
        threshold = Constants.SIGNIFICANCE_LEVEL / n_tests

        if last["t_p"] < threshold and last["gap"] > 0:
            print(f"  At K={k} the personal arm beats a clean, volume-matched")
            print(f"  control by {last['gap']:+.3f} (p={last['t_p']:.4f}).")
        elif last["t_p"] < Constants.SIGNIFICANCE_LEVEL and last["gap"] > 0:
            print(f"  Positive gap at K={k} (p={last['t_p']:.4f}): significant at 0.05")
            print("  but not after Bonferroni correction.")
        else:
            print(f"  No reliable gap at K={k}.")
        print(LINE + "\n")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    @staticmethod
    def plot_curves(summary, tests, path):
        panels = [("macro_f1", "macro-F1"), ("kappa", "kappa"), ("f1_REM", "REM F1")]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

        for axis, (column, title) in zip(axes, panels):
            PersonalizationReport.plot_panel(axis, summary, column, title)

        PersonalizationReport.annotate_p_values(axes[0], tests)

        n_tests = max(1, len(tests))
        threshold = Constants.SIGNIFICANCE_LEVEL / n_tests
        fig.suptitle("Clean control: donor removed from base. Error bars = SEM.\n"
                     f"Bonferroni threshold: p < {threshold:.4f}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote " + path)

    @staticmethod
    def plot_panel(axis, summary, column, title):
        for arm in Constants.ARMS:
            arm_rows = summary[summary["arm"] == arm].sort_values("k")
            sem = arm_rows[column + "_sd"] / np.sqrt(arm_rows["n_subjects"])
            axis.errorbar(arm_rows["k"], arm_rows[column], yerr=sem,
                          marker="o", capsize=4, lw=1.8,
                          color=ARM_COLORS[arm], label=arm)
        axis.set_xlabel("K = novel nights added")
        axis.set_title(title, fontsize=11)
        axis.set_xticks(sorted(summary["k"].unique()))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)

    @staticmethod
    def annotate_p_values(axis, tests):
        n_tests = max(1, len(tests))
        for _, row in tests.iterrows():
            stars = PersonalizationReport.significance_stars(row["t_p"], n_tests)
            color = ARM_COLORS["personal"] if stars != "n.s." else GREY

            axis.annotate(f"p={row['t_p']:.3f}\n{stars}",
                          (row["k"], row["personal"]),
                          textcoords="offset points", xytext=(0, 12),
                          ha="center", fontsize=8, fontweight="bold", color=color)

            midpoint = (row["personal"] + row["stranger"]) / 2
            axis.annotate(f"Δ={row['gap']:+.3f}",
                          (row["k"], midpoint),
                          textcoords="offset points", xytext=(18, 0),
                          ha="left", fontsize=7, color="#34495e")