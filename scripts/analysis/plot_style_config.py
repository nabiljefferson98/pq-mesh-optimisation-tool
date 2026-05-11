"""
plot_style_config.py
---------------------
Shared Matplotlib style settings imported by all dissertation plot scripts.

What this file does (in plain English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instead of setting font sizes, colours, and line widths separately in
every plotting script, we define them all here in one place. Every
plotting script imports this module and calls `apply_dissertation_style()`
at the top. This ensures all figures in Chapter 4 of the dissertation
look visually consistent when placed side by side.

This file is NOT intended to be run directly. It is a shared helper
imported by the other plot scripts.

Usage
~~~~~
    from scripts.analysis.plot_style_config import apply_dissertation_style, COLOURS

    apply_dissertation_style()
    fig, ax = plt.subplots()
    ax.plot(x, y, color=COLOURS['planarity'])
"""

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Colour palette
# Consistent colours used across all plots so the same concept always
# maps to the same colour (e.g. planarity is always the teal/blue line).
# ---------------------------------------------------------------------------

COLOURS = {
    "planarity": "#2E86AB",  # Steel blue  -- planarity data
    "fairness": "#A23B72",  # Dark rose   -- fairness data
    "closeness": "#F18F01",  # Amber       -- closeness / time data
    "before": "#E84855",  # Coral red   -- 'before optimisation' bars
    "after": "#3BB273",  # Mint green  -- 'after optimisation' bars
    "fit_line": "#C62828",  # Deep red    -- regression / fit lines
    "neutral": "#555555",  # Dark grey   -- secondary lines
    "spot": "#2E86AB",
    "blub": "#A23B72",
    "oloid": "#3BB273",
    "bob": "#F18F01",
}
"""
Colour assignments for all series used in dissertation plots.
Keys correspond to energy terms, before/after states, and mesh names.
"""

# ---------------------------------------------------------------------------
# Marker styles -- one distinct marker per mesh for black-and-white printing
# ---------------------------------------------------------------------------

MARKERS = {
    "spot": "o",
    "blub": "s",
    "oloid": "^",
    "bob": "D",
}

# ---------------------------------------------------------------------------
# Shared style function
# ---------------------------------------------------------------------------


def apply_dissertation_style():
    """
    Apply consistent Matplotlib style settings for dissertation figures.

    Call this function once at the top of any plotting script, before
    creating any figures. It sets:
      - Font family (serif, matching LaTeX body text)
      - Font sizes for titles, axis labels, tick labels, and legends
      - Figure DPI (300, suitable for print)
      - Clean grid and tight-layout defaults
      - Line width and marker size defaults

    No parameters. No return value. Modifies Matplotlib's global rcParams.

    Example
    -------
        from scripts.analysis.plot_style_config import apply_dissertation_style
        apply_dissertation_style()
        fig, ax = plt.subplots()
    """
    plt.rcParams.update(
        {
            # -- Fonts
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # -- Lines and markers
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            # -- Figure
            "figure.dpi": 300,
            "figure.autolayout": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # -- Grid
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )
