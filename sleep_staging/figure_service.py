"""
figure_service.py

Shared plotting helpers, so every figure in the paper looks the same.
"""

import numpy as np

from sleep_staging.constants import Constants

GREY = "#7f8c8d"
DARK = "#2c3e50"


class FigureService(object):

    @staticmethod
    def significance_stars(p_value):
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < Constants.SIGNIFICANCE_LEVEL:
            return "*"
        return "n.s."

    @staticmethod
    def boxplot_with_points(axis, groups, labels, colors):
        """One box per group, with every subject drawn as a dot."""
        positions = np.arange(1, len(groups) + 1)
        boxes = axis.boxplot(groups, positions=positions, widths=0.55,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": DARK, "linewidth": 1.5})
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.35)
            box.set_edgecolor(color)

        jitter = np.random.default_rng(Constants.RANDOM_SEED)
        for position, values, color in zip(positions, groups, colors):
            offsets = jitter.normal(0, 0.06, len(values))
            axis.scatter(position + offsets, values, s=12, alpha=0.6,
                         color=color, zorder=3)

        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.grid(alpha=0.25, axis="y")

    @staticmethod
    def label_p_values(axis, groups, p_values):
        """
        Write 'p=... stars' above each box. p_values[i] is None for a box
        with no test (e.g. the reference). The y-limit is raised so the
        labels always stay inside the plot.
        """
        top = max(np.max(values) for values in groups)
        bottom = min(np.min(values) for values in groups)
        step = (top - bottom) * 0.06

        for position, (values, p_value) in enumerate(zip(groups, p_values), 1):
            if p_value is None:
                continue
            stars = FigureService.significance_stars(p_value)
            color = DARK if stars != "n.s." else GREY
            axis.text(position, np.max(values) + step, f"p={p_value:.3g}\n{stars}",
                      ha="center", va="bottom", fontsize=8, color=color)

        axis.set_ylim(bottom - step, top + step * 5)
