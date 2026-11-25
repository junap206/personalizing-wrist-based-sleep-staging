"""
make_paper_figures.py

Redraws every figure that carries a statistical test, from the result files
your runs already saved. Nothing is retrained: the script only reads CSVs
(and the saved predictions / feature table for three figures) and redraws.

What changes compared with the original figures:
    - every test that belongs on a figure is on it, written as
      "p = 0.012 *" or "p < 0.001 ***"
    - p-values sit in a clear band above the data, never on top of it
    - Bonferroni-corrected p-values are used wherever several K values or
      feature sets are tested, and each figure says which test it shows
    - one consistent, colour-blind-safe style for the whole paper

The original PNGs are left untouched. New figures go to paper_figures/
with the same file names, so they can replace the old ones in Overleaf.

Examples:
    python scripts/plotting/make_paper_figures.py
    python scripts/plotting/make_paper_figures.py --only curves boxplot
    python scripts/plotting/make_paper_figures.py --outdir figures_v2

Figures (use these names with --only):
    curves         personalization_v2_final_<tag>.png, one per run
    boxplot        boxplot_personalization.png
    ablation_by_k  ablation_by_k.png
    two_by_two     two_by_two.png
    heterogeneity  heterogeneity.png
    ablation       ablation.png and ablation_personal.png
    transitions    transitions.png
    stability      stability.png
    good_vs_bad    good_vs_bad.png
    clusters       subject_clusters.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files, no window; must come before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root, so `sleep_staging` imports
from sleep_staging.ablation_service import AblationService  # noqa: E402
from sleep_staging.constants import Constants  # noqa: E402
from sleep_staging.heterogeneity_service import HeterogeneityService  # noqa: E402
from sleep_staging.personalization_statistics import PersonalizationStatistics  # noqa: E402

# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------
PER_SUBJECT_PATTERN = os.path.join(Constants.PERSONALIZATION_DIR,
                                   "personalization_v2_final_{tag}_per_subject.csv")

# (run tag, name used on the figures)
RUNS = [
    ("all", "All features"),
    ("realtime", "Real-time features"),
    ("random", "Random night order"),
    ("nonpersonal", "Non-personal features"),
    ("personal", "Personal-reference features"),
    ("hr", "Heart rate only"),
    ("motion", "Motion only"),
    ("hrmotion", "Heart rate + motion"),
    ("walch", "Walch et al. features"),
]

# Panels of the main box plot: (run tag, panel title)
BOXPLOT_PANELS = [
    ("all", "All features (offline)"),
    ("realtime", "Real-time features"),
    ("random", "Random night order"),
]

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
PERSONAL = "#2a78d6"   # blue
STRANGER = "#eb6834"   # orange
ARM_COLORS = {"personal": PERSONAL, "stranger": STRANGER}
ARM_NAMES = {"personal": "Own nights (personal)", "stranger": "Stranger's nights"}
BEST_NIGHT = PERSONAL
WORST_NIGHT = STRANGER

# Colours for the feature-set curves (validated colour-blind-safe set)
SET_COLORS = {
    "all": INK,
    "nonpersonal": "#2a78d6",
    "personal": "#eb6834",
    "hr": "#e87ba4",
    "motion": "#1baf7a",
    "hrmotion": "#6250d6",
    "walch": "#eda100",
}

# Two neutral greys for "with" / "without" personal-reference features
WITH_COLOR = "#3d3d3d"
WITHOUT_COLOR = "#a3a3a3"

DPI = 300


def apply_style():
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ----------------------------------------------------------------------
# p-value text
# ----------------------------------------------------------------------
def format_p(p_value):
    if p_value is None or not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < Constants.SIGNIFICANCE_LEVEL:
        return "*"
    return "n.s."


def p_text(p_value):
    """'p = 0.012 *', 'p < 0.001 ***' or 'p = 0.079 n.s.'"""
    return f"{format_p(p_value)} {stars(p_value)}"


def p_color(p_value):
    return INK if p_value < Constants.SIGNIFICANCE_LEVEL else MUTED


# ----------------------------------------------------------------------
# Layout helpers
# ----------------------------------------------------------------------
def make_room(axis, low, high, top=0.30, bottom=0.06):
    """
    Set the y-limits so the data fills the middle of the panel, leaving a
    band of `top` (fraction of the panel height) free for p-values.
    """
    span = high - low if high > low else 1.0
    total = span / (1.0 - top - bottom)
    axis.set_ylim(low - bottom * total, high + top * total)


def band_text(axis, x, text, color=INK, y=0.98, size=9.5):
    """Text placed at data x, but at a fixed height in the panel's top band."""
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    axis.text(x, y, text, transform=transform, ha="center", va="top",
              fontsize=size, color=color, linespacing=1.3)


def bracket(axis, x1, x2, text, color=INK, y=0.80):
    """A comparison bracket between two boxes, drawn inside the top band."""
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    axis.plot([x1, x1, x2, x2], [y - 0.025, y, y, y - 0.025], color=MUTED,
              lw=1.0, transform=transform, clip_on=False)
    axis.text((x1 + x2) / 2, y + 0.012, text, transform=transform, ha="center",
              va="bottom", fontsize=9.5, color=color)


def corner_note(axis, text, loc="upper left"):
    """A small boxed note inside a panel (for r, ICC, slopes...)."""
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    axis.text(x, y, text, transform=axis.transAxes, ha=ha, va=va, fontsize=9.5,
              color=INK, linespacing=1.35,
              bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                    "edgecolor": GRID})


def footnote(fig, text):
    fig.text(0.0, -0.01, text, ha="left", va="top", fontsize=8.5, color=MUTED,
             wrap=True)


def draw_boxes(axis, groups, labels, colors):
    """One box per group with every subject drawn as a dot."""
    positions = np.arange(1, len(groups) + 1)
    boxes = axis.boxplot(groups, positions=positions, widths=0.55, patch_artist=True,
                         showfliers=False,
                         medianprops={"color": INK, "linewidth": 1.6},
                         whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.22)
        box.set_edgecolor(color)

    jitter = np.random.default_rng(Constants.RANDOM_SEED)
    for position, values, color in zip(positions, groups, colors):
        offsets = jitter.uniform(-0.12, 0.12, len(values))
        axis.scatter(position + offsets, values, s=16, alpha=0.65, color=color,
                     edgecolor="white", linewidth=0.4, zorder=3)

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlim(0.4, len(groups) + 0.6)


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote " + path)


def read_csv(path, **kwargs):
    if not os.path.exists(path):
        print(f"  skipped: {path} not found")
        return None
    return pd.read_csv(path, **kwargs)


# ----------------------------------------------------------------------
# Statistics shared by the personalization figures
# ----------------------------------------------------------------------
def arm_scores(per_subject, arm, k, metric):
    rows = per_subject[(per_subject["arm"] == arm) & (per_subject["k"] == k)]
    return rows.set_index("subject")[metric]


def gap_tests(per_subject, metric):
    """
    Personal vs stranger at every K > 0 for one metric: mean gap and paired
    t-test, Bonferroni-corrected over the K values tested.
    """
    rows = []
    for k in sorted(per_subject["k"].unique()):
        if k == 0:
            continue
        personal = arm_scores(per_subject, "personal", k, metric)
        stranger = arm_scores(per_subject, "stranger", k, metric)
        common = personal.index.intersection(stranger.index)
        _, p_value = stats.ttest_rel(personal[common], stranger[common])
        rows.append({"k": k, "n": len(common),
                     "gap": (personal[common] - stranger[common]).mean(),
                     "p": p_value})
    tests = pd.DataFrame(rows)
    tests["p_corrected"] = (tests["p"] * len(tests)).clip(upper=1.0)
    return tests


# ----------------------------------------------------------------------
# Figure: personalization curves (one per run)
# ----------------------------------------------------------------------
CURVE_METRICS = [("macro_f1", "Macro-F1"), ("kappa", "Cohen's κ"), ("f1_REM", "REM F1")]


def draw_curve_panel(axis, per_subject, metric, title):
    lows, highs = [], []
    for arm, offset in (("personal", -0.05), ("stranger", 0.05)):
        rows = per_subject[per_subject["arm"] == arm]
        by_k = rows.groupby("k")[metric].agg(["mean", "std", "count"])
        sem = by_k["std"] / np.sqrt(by_k["count"])
        axis.errorbar(by_k.index + offset, by_k["mean"], yerr=sem, color=ARM_COLORS[arm],
                      marker="o", markersize=6, linewidth=2, capsize=3,
                      markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        lows.append((by_k["mean"] - sem).min())
        highs.append((by_k["mean"] + sem).max())

    make_room(axis, min(lows), max(highs), top=0.30)
    for _, row in gap_tests(per_subject, metric).iterrows():
        band_text(axis, row["k"], f"Δ {row['gap']:+.3f}\n{p_text(row['p_corrected'])}",
                  color=p_color(row["p_corrected"]))

    k_values = sorted(per_subject["k"].unique())
    axis.set_xticks(k_values)
    axis.set_xlim(min(k_values) - 0.35, max(k_values) + 0.35)
    axis.set_xlabel("Nights added to training (K)")
    axis.set_ylabel(title)
    axis.set_title(title)


def figure_curves(outdir):
    for tag, label in RUNS:
        per_subject = read_csv(PER_SUBJECT_PATTERN.format(tag=tag))
        if per_subject is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), layout="constrained")
        for axis, (metric, title) in zip(axes, CURVE_METRICS):
            draw_curve_panel(axis, per_subject, metric, title)

        handles = [Line2D([0], [0], color=ARM_COLORS[arm], marker="o", lw=2,
                          label=ARM_NAMES[arm]) for arm in Constants.ARMS]
        fig.legend(handles=handles, loc="outside upper center", ncol=2)
        n_subjects = per_subject["subject"].nunique()
        n_tests = per_subject.loc[per_subject["k"] > 0, "k"].nunique()
        footnote(fig, f"{label}. {n_subjects} subjects; points are mean ± SEM. "
                      "Δ = personal − stranger. p: paired t-test across subjects, "
                      f"Bonferroni-corrected for {n_tests} values of K.")
        save(fig, outdir, f"personalization_v2_final_{tag}.png")


# ----------------------------------------------------------------------
# Figure: main box plot of personal − stranger gaps
# ----------------------------------------------------------------------
def figure_boxplot(outdir):
    panels = []
    for tag, title in BOXPLOT_PANELS:
        path = PER_SUBJECT_PATTERN.format(tag=tag)
        if not os.path.exists(path):
            continue
        per_subject = pd.read_csv(path)
        k_values = sorted(k for k in per_subject["k"].unique() if k > 0)
        gaps = [PersonalizationStatistics.get_gaps(per_subject, k).values for k in k_values]
        tests = PersonalizationStatistics.compare_arms(per_subject, k_values)
        tests = PersonalizationStatistics.add_corrected_p(tests)
        panels.append((title, k_values, gaps, tests))
    if not panels:
        print("  skipped: no personalization results found")
        return

    every_gap = np.concatenate([np.concatenate(gaps) for _, _, gaps, _ in panels])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.8),
                             sharey=True, squeeze=False, layout="constrained")
    for axis, (title, k_values, gaps, tests) in zip(axes[0], panels):
        draw_boxes(axis, gaps, [f"K = {k}" for k in k_values], [PERSONAL] * len(gaps))
        axis.axhline(0, color=STRANGER, lw=1.2, ls="--", zorder=1)
        make_room(axis, every_gap.min(), every_gap.max(), top=0.22, bottom=0.12)
        for position, (_, row) in enumerate(tests.iterrows(), 1):
            band_text(axis, position, p_text(row["t_p_corrected"]),
                      color=p_color(row["t_p_corrected"]))
            band_text(axis, position, f"{int(row['n_improved'])}/{int(row['n'])} improved",
                      color=MUTED, y=0.075, size=8.5)
        axis.set_title(title)
        axis.set_xlabel("Nights added to training (K)")
    axes[0][0].set_ylabel("Δ macro-F1 (personal − stranger)")
    footnote(fig, "Each dot is one subject. Above the dashed line, the subject's own "
                  "nights helped more than a stranger's. p: paired t-test, "
                  "Bonferroni-corrected for the number of K values.")
    save(fig, outdir, "boxplot_personalization.png")


# ----------------------------------------------------------------------
# Figure: which feature sets personalize (curves + table)
# ----------------------------------------------------------------------
ABLATION_BY_K_SETS = [
    ("all", "All features"),
    ("personal", "Personal-reference"),
    ("nonpersonal", "Non-personal"),
    ("hr", "Heart rate"),
    ("motion", "Motion"),
    ("hrmotion", "Heart rate + motion"),
    ("walch", "Walch (4 features)"),
]


def ablation_by_k_row(label, per_subject):
    k_values = sorted(k for k in per_subject["k"].unique() if k > 0)
    tests = PersonalizationStatistics.compare_arms(per_subject, k_values)
    tests = PersonalizationStatistics.add_corrected_p(tests).set_index("k")
    change = PersonalizationStatistics.change_within_arm(
        per_subject, "personal", 0, max(k_values))
    return {"label": label,
            "gap_1": tests.loc[1, "gap"], "p_1": tests.loc[1, "t_p_corrected"],
            "gap_3": tests.loc[3, "gap"], "p_3": tests.loc[3, "t_p_corrected"],
            "change": change["gap"], "p_change": change["t_p"]}


def draw_set_curves(axis, loaded):
    for tag, label, per_subject in loaded:
        personal = per_subject[per_subject["arm"] == "personal"]
        by_k = personal.groupby("k")["macro_f1"].agg(["mean", "sem"])
        axis.errorbar(by_k.index, by_k["mean"], yerr=by_k["sem"], marker="o",
                      markersize=5, lw=2 if tag == "all" else 1.6, capsize=3,
                      color=SET_COLORS[tag], label=label)
    axis.set_xticks([0, 1, 2, 3])
    axis.set_xlabel("Nights added to training (K)")
    axis.set_ylabel("Macro-F1, personal arm (mean ± SEM)")
    axis.set_title("Performance as own nights are added")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)


def draw_set_table(axis, rows):
    axis.axis("off")
    headers = ["Feature set", "Gap, K = 1", "Gap, K = 3", "Own nights,\nK = 0 → 3"]
    cells, p_grid = [], []
    for row in rows:
        cells.append([row["label"],
                      f"{row['gap_1']:+.4f}\n{p_text(row['p_1'])}",
                      f"{row['gap_3']:+.4f}\n{p_text(row['p_3'])}",
                      f"{row['change']:+.4f}\n{p_text(row['p_change'])}"])
        p_grid.append([None, row["p_1"], row["p_3"], row["p_change"]])

    table = axis.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center",
                       colWidths=[0.30, 0.23, 0.23, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.7)
    for (row_number, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row_number == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold", color=INK)
            continue
        p_value = p_grid[row_number - 1][column]
        if p_value is None:
            cell.set_text_props(color=INK, ha="left")
            cell.PAD = 0.04
        elif p_value < Constants.SIGNIFICANCE_LEVEL:
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=MUTED)
    axis.set_title("Personal − stranger gap, and the change within the personal arm")


def figure_ablation_by_k(outdir):
    loaded = []
    for tag, label in ABLATION_BY_K_SETS:
        per_subject = read_csv(PER_SUBJECT_PATTERN.format(tag=tag))
        if per_subject is not None:
            loaded.append((tag, label, per_subject))
    if not loaded:
        return

    rows = [ablation_by_k_row(label, per_subject) for _, label, per_subject in loaded]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.25]})
    draw_set_curves(axes[0], loaded)
    draw_set_table(axes[1], rows)
    footnote(fig, "Gap = personal − stranger macro-F1: paired t-test, Bonferroni-corrected "
                  "for K = 1, 2, 3. Last column: change within the personal arm from K = 0 "
                  "to K = 3, paired t-test (not volume-controlled). Bold = p < 0.05.")
    save(fig, outdir, "ablation_by_k.png")


# ----------------------------------------------------------------------
# Figure: 2 x 2 (with vs without personal-reference features)
# ----------------------------------------------------------------------
TWO_BY_TWO_FROM_K = 1
TWO_BY_TWO_END_K = 3


def subject_measures(per_subject):
    start, end = TWO_BY_TWO_FROM_K, TWO_BY_TWO_END_K
    measures = pd.DataFrame({
        "start": arm_scores(per_subject, "personal", start, "macro_f1"),
        "end": arm_scores(per_subject, "personal", end, "macro_f1"),
        "stranger": arm_scores(per_subject, "stranger", end, "macro_f1"),
    }).dropna()
    measures["benefit"] = measures["end"] - measures["start"]
    measures["gap"] = measures["end"] - measures["stranger"]
    return measures


def draw_with_without(axis, with_values, without_values, title, ylabel):
    groups = [with_values, without_values]
    draw_boxes(axis, groups, ["With", "Without"], [WITH_COLOR, WITHOUT_COLOR])
    axis.axhline(0, color=STRANGER, lw=1.2, ls="--", zorder=1)
    every = np.concatenate(groups)
    make_room(axis, every.min(), every.max(), top=0.36, bottom=0.06)

    for position, values in enumerate(groups, 1):
        _, p_value = stats.ttest_1samp(values, 0.0)
        band_text(axis, position, f"vs 0: {p_text(p_value)}", color=p_color(p_value),
                  y=0.80, size=9)
    difference = PersonalizationStatistics.paired_test(with_values, without_values)
    bracket(axis, 1, 2, f"with vs without: {p_text(difference['t_p'])}",
            color=p_color(difference["t_p"]), y=0.90)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Personal-reference features")
    return difference


def figure_two_by_two(outdir):
    with_results = read_csv(PER_SUBJECT_PATTERN.format(tag="all"))
    without_results = read_csv(PER_SUBJECT_PATTERN.format(tag="nonpersonal"))
    if with_results is None or without_results is None:
        return
    with_set = subject_measures(with_results)
    without_set = subject_measures(without_results)
    common = with_set.index.intersection(without_set.index)
    with_set, without_set = with_set.loc[common], without_set.loc[common]

    start, end = TWO_BY_TWO_FROM_K, TWO_BY_TWO_END_K
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), layout="constrained")
    draw_with_without(axes[0], with_set["benefit"].values, without_set["benefit"].values,
                      f"Training benefit (K = {start} → {end})",
                      f"Personal-arm macro-F1, K = {end} minus K = {start}")
    gap_test = draw_with_without(axes[1], with_set["gap"].values, without_set["gap"].values,
                                 f"Controlled gap at K = {end}",
                                 "Δ macro-F1 (personal − stranger)")

    axis = axes[2]
    axis.scatter(without_set["gap"], with_set["gap"], s=26, alpha=0.75, color=WITH_COLOR,
                 edgecolor="white", linewidth=0.5, zorder=3)
    limit = np.abs(np.concatenate([with_set["gap"], without_set["gap"]])).max() * 1.12
    axis.plot([-limit, limit], [-limit, limit], color=STRANGER, ls="--", lw=1.1)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal")
    axis.grid(True, axis="both")
    axis.set_xlabel("Gap without personal-reference features")
    axis.set_ylabel("Gap with personal-reference features")
    axis.set_title(f"Controlled gap at K = {end}, per subject")
    corner_note(axis, f"Above the line = larger gap with them\n"
                      f"paired {p_text(gap_test['t_p'])}")
    footnote(fig, f"{len(common)} subjects. Each dot is one subject. Dashed line: zero "
                  "(left, middle) or equal gaps (right). p: one-sample t-test against 0, "
                  "and paired t-test between feature sets.")
    save(fig, outdir, "two_by_two.png")


# ----------------------------------------------------------------------
# Figure: heterogeneity (regression to the mean)
# ----------------------------------------------------------------------
def figure_heterogeneity(outdir):
    per_subject = read_csv(Constants.PERSONALIZATION_PER_SUBJECT_FILE)
    if per_subject is None:
        return
    k = Constants.MAX_PERSONAL_NIGHTS
    changes = HeterogeneityService.build_change_table(per_subject, k)
    slope_difference, paired_p = HeterogeneityService.paired_slope_test(changes)

    fig, axis = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    for arm in Constants.ARMS:
        rows = changes[changes["arm"] == arm]
        fit = HeterogeneityService.arm_slope(changes, arm)
        axis.scatter(rows["baseline"], rows["delta"], s=28, alpha=0.7,
                     color=ARM_COLORS[arm], edgecolor="white", linewidth=0.5, zorder=3)
        x = np.linspace(rows["baseline"].min(), rows["baseline"].max(), 50)
        axis.plot(x, fit["intercept"] + fit["slope"] * x, color=ARM_COLORS[arm], lw=2,
                  label=f"{ARM_NAMES[arm]}: slope {fit['slope']:+.2f} ({format_p(fit['p'])})")

    axis.axhline(0, color=MUTED, lw=0.9)
    axis.grid(True, axis="both")
    axis.set_xlabel("Macro-F1 at K = 0 (population model)")
    axis.set_ylabel(f"Change in macro-F1, K = 0 → {k}")
    axis.set_title("Do the hardest subjects gain the most?")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    corner_note(axis, f"Do the slopes differ?\nΔ slope {slope_difference:+.3f}, "
                      f"paired {p_text(paired_p)}", loc="upper right")
    footnote(fig, f"{changes['subject'].nunique()} subjects, one dot per subject per arm. "
                  "A negative slope in both arms is regression to the mean; only a "
                  "difference between the slopes would mean personal nights help weaker "
                  "subjects more. Paired test: gap regressed on the K = 0 score.")
    save(fig, outdir, "heterogeneity.png")


# ----------------------------------------------------------------------
# Figure: feature ablation (leave-one-subject-out)
# ----------------------------------------------------------------------
ABLATION_FILES = [(os.path.join(Constants.ABLATION_DIR, "ablation_results.csv"), "ablation.png"),
                  (os.path.join(Constants.ABLATION_DIR, "ablation_personal_results.csv"),
                   "ablation_personal.png")]


def draw_ablation(results, outdir, name):
    tests = AblationService.compare_to_reference(results, reference="all")
    corrected = dict(zip(tests["set"], tests["p_corrected"]))
    means = results.groupby("set")["macro_f1"].mean().sort_values()
    order = [s for s in means.index if s != "all"] + ["all"]
    n_features = results.groupby("set")["n_features"].first()

    groups = [results.loc[results["set"] == s, "macro_f1"].values for s in order]
    labels = [f"{Constants.FEATURE_SET_LABELS.get(s, s)}\n({n_features[s]} feat.)"
              for s in order]
    colors = [INK if s == "all" else PERSONAL for s in order]

    fig, axis = plt.subplots(figsize=(1.35 * len(order) + 2.2, 5.0), layout="constrained")
    draw_boxes(axis, groups, labels, colors)
    every = np.concatenate(groups)
    make_room(axis, every.min(), every.max(), top=0.18, bottom=0.05)
    for position, set_name in enumerate(order, 1):
        if set_name == "all":
            band_text(axis, position, "reference", color=MUTED)
        else:
            band_text(axis, position, p_text(corrected[set_name]),
                      color=p_color(corrected[set_name]))
    axis.set_ylabel("Macro-F1 per subject")
    axis.set_title("Which signals the model needs")
    footnote(fig, f"{results['subject'].nunique()} subjects, leave-one-subject-out; each "
                  "dot is one subject. p: paired t-test against all features, "
                  f"Bonferroni-corrected for {len(tests)} comparisons.")
    save(fig, outdir, name)


def figure_ablation(outdir):
    for results_path, name in ABLATION_FILES:
        results = read_csv(results_path)
        if results is not None:
            draw_ablation(results, outdir, name)


# ----------------------------------------------------------------------
# Figure: errors at stage transitions
# ----------------------------------------------------------------------
def figure_transitions(outdir):
    from sleep_staging.night_score_service import NightScoreService
    from sleep_staging.transition_service import TransitionService

    if not os.path.exists(Constants.PREDICTIONS_FILE):
        print(f"  skipped: {Constants.PREDICTIONS_FILE} not found")
        return
    print("  reading saved predictions (about a minute)...")
    predictions = NightScoreService.load_predictions(Constants.PREDICTIONS_FILE)
    epochs = TransitionService.annotate(predictions)
    summary, per_subject = TransitionService.error_by_distance(epochs)
    transition_p, n_subjects = TransitionService.compare_transition_to_stable(per_subject)
    by_type = TransitionService.error_by_type(epochs)
    nights = TransitionService.score_nights(epochs)
    nights, r, _, p_adjusted = TransitionService.within_subject_correlation(nights)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # Panel 1: error by distance to the nearest true transition
    axis = axes[0]
    bins = list(summary.index)
    groups = [per_subject[name].dropna().values for name in bins]
    colors = [STRANGER] + [PERSONAL] * (len(bins) - 1)
    draw_boxes(axis, groups, [name.replace(" (stable)", "\n(stable)") for name in bins],
               colors)
    every = np.concatenate(groups)
    make_room(axis, every.min(), every.max(), top=0.18, bottom=0.05)
    ratio = summary["error_rate"].iloc[0] / summary["error_rate"].iloc[-1]
    bracket(axis, 1, len(bins), f"{ratio:.1f}× higher at transitions: {p_text(transition_p)}",
            color=p_color(transition_p), y=0.90)
    axis.set_ylabel("Error rate per subject")
    axis.set_xlabel("Distance to the nearest true stage change (epochs)")
    axis.set_title("Error vs distance to a transition")

    # Panel 2: error by type of transition
    axis = axes[1]
    shown = by_type.head(8).iloc[::-1]
    axis.barh(shown.index, shown["error_rate"], color=STRANGER, alpha=0.85, height=0.6)
    for y, (_, row) in enumerate(shown.iterrows()):
        axis.text(row["error_rate"] + 0.015, y,
                  f"{row['error_rate']:.2f}  (n = {int(row['n_epochs']):,})",
                  va="center", fontsize=9, color=INK)
    axis.set_xlim(0, 1.25)
    axis.grid(True, axis="x")
    axis.grid(False, axis="y")
    axis.set_xlabel("Error rate at transition epochs")
    axis.set_title("By type of transition")

    # Panel 3: within-subject fragmentation vs night score
    axis = axes[2]
    axis.scatter(nights["frac_near_transition_c"], nights["macro_f1_c"], s=18, alpha=0.6,
                 color=PERSONAL, edgecolor="white", linewidth=0.4, zorder=3)
    fit = stats.linregress(nights["frac_near_transition_c"], nights["macro_f1_c"])
    x = np.linspace(nights["frac_near_transition_c"].min(),
                    nights["frac_near_transition_c"].max(), 50)
    axis.plot(x, fit.intercept + fit.slope * x, color=INK, lw=1.6)
    axis.axhline(0, color=MUTED, lw=0.8)
    axis.axvline(0, color=MUTED, lw=0.8)
    axis.grid(True, axis="both")
    corner_note(axis, f"r = {r:+.3f}\n{p_text(p_adjusted)}\n{len(nights)} nights",
                loc="upper right")
    axis.set_xlabel("Share of epochs near a transition\n(minus the subject's mean)")
    axis.set_ylabel("Night macro-F1 (minus the subject's mean)")
    axis.set_title("More fragmented nights score lower")

    footnote(fig, f"Left: {n_subjects} subjects, paired t-test of error at transitions vs "
                  "stable stretches (>10 epochs away). Middle: n = number of transition "
                  "epochs. Right: within-subject correlation; p uses degrees of freedom "
                  "adjusted for the subject means removed.")
    save(fig, outdir, "transitions.png")


# ----------------------------------------------------------------------
# Figure: night-to-night stability
# ----------------------------------------------------------------------
def figure_stability(outdir):
    from sleep_staging.stability_service import StabilityService

    night_scores = read_csv(Constants.NIGHT_SCORES_FILE)
    if night_scores is None:
        return
    consistency = StabilityService.consistency_by_subject(night_scores)
    order = list(consistency.sort_values("mean").index)
    icc = StabilityService.intraclass_correlation(night_scores)
    split = StabilityService.split_half(night_scores)

    fig, axis = plt.subplots(figsize=(13, 5.0), layout="constrained")
    for position, subject in enumerate(order):
        scores = night_scores.loc[night_scores["subject"] == subject, "macro_f1"].values
        axis.scatter([position] * len(scores), scores, s=20, alpha=0.6, color=PERSONAL,
                     edgecolor="white", linewidth=0.4, zorder=3)
        axis.plot([position - 0.35, position + 0.35], [scores.mean()] * 2, color=INK,
                  lw=2, zorder=4)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(order, rotation=90, fontsize=7.5)
    axis.set_xlim(-0.8, len(order) - 0.2)
    axis.set_ylabel("Macro-F1 per night")
    axis.set_title("Night-to-night consistency of each subject")

    lines = []
    if icc:
        lines.append(f"ICC = {icc['icc']:.3f}  (between-subject SD {icc['between_sd']:.3f}, "
                     f"within-subject SD {icc['within_sd']:.3f})")
    if split:
        lines.append(f"Split-half r = {split['mean_r']:.3f}  "
                     f"[95% interval {split['lo']:.3f} to {split['hi']:.3f}]")
    if lines:
        corner_note(axis, "\n".join(lines), loc="lower right")
    footnote(fig, "Each dot is one night; the black dash is the subject's mean. Subjects "
                  "are ordered by mean score. ICC: share of the variance in night scores "
                  "that lies between subjects.")
    save(fig, outdir, "stability.png")


# ----------------------------------------------------------------------
# Figure: best vs worst night
# ----------------------------------------------------------------------
GOOD_VS_BAD_PANELS = 6


def figure_good_vs_bad(outdir):
    from sleep_staging.night_comparison_service import NightComparisonService

    if not (os.path.exists(Constants.FEATURES_FILE)
            and os.path.exists(Constants.NIGHT_SCORES_FILE)):
        print("  skipped: needs the feature table and night_scores.csv")
        return
    print("  reading the feature table (about a minute)...")
    features = pd.read_csv(Constants.FEATURES_FILE)
    night_scores = pd.read_csv(Constants.NIGHT_SCORES_FILE)
    nights = NightComparisonService.summarise_nights(features, night_scores)
    pairs = NightComparisonService.pick_best_and_worst(nights)

    table = read_csv(Constants.GOOD_VS_BAD_FILE)
    if table is None:
        return
    good, bad = NightComparisonService.get_groups(nights, pairs)
    shown = table.head(GOOD_VS_BAD_PANELS)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), layout="constrained")
    for axis, (_, row) in zip(axes.ravel(), shown.iterrows()):
        column = row["feature"]
        worst_values = bad[column].values
        best_values = good[column].values
        for worst, best in zip(worst_values, best_values):
            axis.plot([0, 1], [worst, best], color="#cfcfcf", lw=0.8, zorder=1)
        axis.scatter([0] * len(worst_values), worst_values, s=18, color=WORST_NIGHT,
                     zorder=3, edgecolor="white", linewidth=0.4)
        axis.scatter([1] * len(best_values), best_values, s=18, color=BEST_NIGHT,
                     zorder=3, edgecolor="white", linewidth=0.4)
        every = np.concatenate([worst_values, best_values])
        every = every[np.isfinite(every)]
        make_room(axis, every.min(), every.max(), top=0.22, bottom=0.05)

        verdict = "survives FDR" if row["significant"] else "n.s. after FDR"
        transform = axis.transAxes
        axis.text(0.5, 0.98, f"d = {row['cohens_d']:+.2f}   {format_p(row['t_p'])}\n{verdict}",
                  transform=transform, ha="center", va="top", fontsize=9.5,
                  color=INK if row["significant"] else MUTED)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Worst night", "Best night"])
        axis.set_xlim(-0.4, 1.4)
        axis.set_title(column)

    footnote(fig, f"{len(pairs)} subjects; each grey line joins one subject's worst and best "
                  "night (by baseline macro-F1). The six features with the largest effect "
                  "are shown. d: Cohen's d for paired data. p: paired t-test; FDR: "
                  f"Benjamini–Hochberg correction across all {len(table)} features tested.")
    save(fig, outdir, "good_vs_bad.png")


# ----------------------------------------------------------------------
# Figure: subject profiles and clusters
# ----------------------------------------------------------------------
CLUSTER_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e87ba4")


def figure_clusters(outdir):
    profile = read_csv(Constants.SUBJECT_PROFILE_FILE, index_col=0)
    correlations = read_csv(Constants.SUBJECT_CORRELATIONS_FILE)
    if profile is None or correlations is None:
        return
    clustered = profile.dropna(subset=["cluster"])
    clusters = sorted(clustered["cluster"].unique())
    groups = [clustered.loc[clustered["cluster"] == c, "macro_f1"].values for c in clusters]
    testable = [values for values in groups if len(values) > 1]
    anova_p = stats.f_oneway(*testable).pvalue if len(testable) >= 2 else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained")

    # Panel 1: strongest subject-level correlate of difficulty
    axis = axes[0]
    strongest = correlations.iloc[0]
    x = profile[strongest["feature"]]
    axis.scatter(x, profile["macro_f1"], s=30, color=PERSONAL, edgecolor="white",
                 linewidth=0.5, zorder=3)
    usable = profile[[strongest["feature"], "macro_f1"]].dropna()
    fit = stats.linregress(usable.iloc[:, 0], usable.iloc[:, 1])
    line_x = np.linspace(usable.iloc[:, 0].min(), usable.iloc[:, 0].max(), 50)
    axis.plot(line_x, fit.intercept + fit.slope * line_x, color=INK, lw=1.6)
    axis.grid(True, axis="both")
    verdict = "survives FDR" if strongest["significant"] else "n.s. after FDR"
    corner_note(axis, f"r = {strongest['pearson_r']:+.3f}\n{format_p(strongest['p'])}"
                      f"\n{verdict}", loc="upper right")
    axis.set_xlabel(strongest["feature"])
    axis.set_ylabel("Baseline macro-F1")
    axis.set_title("Strongest correlate of difficulty")

    # Panel 2: the clusters
    axis = axes[1]
    for index, cluster in enumerate(clusters):
        rows = clustered[clustered["cluster"] == cluster]
        axis.scatter(rows["transitions_per_hour"], rows["hr_mean"], s=34,
                     color=CLUSTER_COLORS[index % len(CLUSTER_COLORS)], edgecolor="white",
                     linewidth=0.5, label=f"Cluster {int(cluster)} (n = {len(rows)})",
                     zorder=3)
    axis.grid(True, axis="both")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    axis.set_xlabel("Transitions per hour")
    axis.set_ylabel("Mean heart rate (bpm)")
    axis.set_title(f"k-means, {len(clusters)} clusters")

    # Panel 3: difficulty by cluster
    axis = axes[2]
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(clusters))]
    draw_boxes(axis, groups, [f"Cluster {int(c)}" for c in clusters], colors)
    every = np.concatenate(groups)
    make_room(axis, every.min(), every.max(), top=0.18, bottom=0.05)
    if np.isfinite(anova_p):
        band_text(axis, (len(clusters) + 1) / 2, f"ANOVA across clusters: {p_text(anova_p)}",
                  color=p_color(anova_p))
    axis.set_ylabel("Baseline macro-F1")
    axis.set_title("Difficulty by cluster")

    footnote(fig, f"{len(profile)} subjects. Left: Pearson correlation with the baseline "
                  "score; FDR: Benjamini–Hochberg correction across all properties tested. "
                  "Right: one-way ANOVA of baseline macro-F1 across clusters.")
    save(fig, outdir, "subject_clusters.png")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
FIGURES = {
    "curves": figure_curves,
    "boxplot": figure_boxplot,
    "ablation_by_k": figure_ablation_by_k,
    "two_by_two": figure_two_by_two,
    "heterogeneity": figure_heterogeneity,
    "ablation": figure_ablation,
    "transitions": figure_transitions,
    "stability": figure_stability,
    "good_vs_bad": figure_good_vs_bad,
    "clusters": figure_clusters,
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=Constants.PAPER_FIGURES_DIR)
    parser.add_argument("--only", nargs="*", choices=list(FIGURES), default=list(FIGURES))
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.outdir, exist_ok=True)
    apply_style()
    for name in args.only:
        print(name)
        FIGURES[name](args.outdir)
    print(f"\nDone. New figures are in {args.outdir}/")


if __name__ == "__main__":
    main()