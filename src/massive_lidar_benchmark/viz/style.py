"""Shared Matplotlib styling."""

from __future__ import annotations

import matplotlib as mpl

METHOD_COLORS = {
    "ekf": "#2C7BB6",
    "mcl": "#D95F02",
    "gt": "#111111",
    "particle": "#1B9E77",
    "scan": "#6E6E6E",
}


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
