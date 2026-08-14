"""Shared matplotlib styling and plot helpers for the analysis notebooks.

The categorical palette is assigned by series identity in a fixed order and has
been validated for colour-vision deficiency separation (worst adjacent CVD
Delta-E 9.1, worst adjacent normal-vision Delta-E 19.6 on a light surface).
Series are always identified by a legend or a direct label, never by colour
alone.
"""

from __future__ import annotations

import numpy as np

# Fixed categorical order.  Never cycle or reorder: colour follows the entity.
PALETTE = ["steelblue", "orange", "green", "red", "orchid", "brown"]
BLUE, ORANGE, GREEN, RED, MAGENTA, BROWN = PALETTE

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

# Stable role assignments used across the three notebooks.
ROLE_COLORS = {
    "far": BLUE,
    "near": ORANGE,
    "combined": BROWN,
    "external": INK_SECONDARY,
    "nufit": INK,
    "dayabay": MAGENTA,
    "juno": BLUE,
    "anchor": RED,
    "dip": ORANGE,
}

# 1, 2, 3 sigma for two degrees of freedom
DELTA_CHI2_2DOF = (2.30, 6.18, 11.83)
DELTA_CHI2_1DOF = (1.0, 4.0, 9.0)


def use_style() -> None:
    """Apply the shared rcParams.  Call once at the top of a notebook."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "axes.spines.top": True,
            "axes.spines.right": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 2.0,
            "lines.markersize": 10,
            "legend.frameon": False,
            # "legend.framealpha": 0.92,
            "legend.edgecolor": GRID,
            "legend.facecolor": SURFACE,
            "legend.fontsize": 9,
            "mathtext.fontset": "cm",
            "text.usetex": True,
            "font.family": "serif",                  # Options: 'serif', 'sans-serif', 'monospace'
            "font.serif": ["Computer Modern Roman"], # Maps to standard LaTeX font
            # amssymb is needed for \lesssim and friends in axis labels/annotations.
            "text.latex.preamble": r"\usepackage{amsmath,amssymb}",

        }
    )


# ---------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------
def confidence_contours(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    chi2_grid: np.ndarray,
    *,
    color: str = BLUE,
    levels=DELTA_CHI2_2DOF,
    label: str | None = None,
    fill_alpha: float = 0.18,
    linestyles=("-", "--", ":"),
    linewidth: float = 1.8,
    fill: bool = True,
):
    """Draw Delta chi^2 confidence contours from a 2D grid.

    ``chi2_grid`` has shape ``(len(y), len(x))``; it is shifted to its own
    minimum before contouring.
    """

    delta = np.asarray(chi2_grid, dtype=float)
    delta = delta - np.nanmin(delta)
    levels = list(levels)

    if fill:
        ax.contourf(
            x, y, delta, levels=[0.0, levels[0]], colors=[color], alpha=fill_alpha
        )
    for level, style in zip(levels, linestyles):
        ax.contour(
            x,
            y,
            delta,
            levels=[level],
            colors=[color],
            linestyles=[style],
            linewidths=linewidth,
        )
    if label is not None:
        ax.plot([], [], color=color, lw=linewidth, label=label)
    return ax


def mark_best_fit(ax, x: float, y: float, *, color: str = INK, marker: str = "*", size: float = 11):
    ax.plot(
        x,
        y,
        marker=marker,
        color=color,
        markersize=size,
        markeredgecolor=SURFACE,
        markeredgewidth=0.8,
        linestyle="none",
        zorder=6,
    )
    return ax


def error_ellipse(
    ax,
    center: tuple[float, float],
    cov: np.ndarray,
    *,
    n_sigma: float = 1.0,
    color: str = INK,
    label: str | None = None,
    **kwargs,
):
    """Gaussian confidence ellipse from a 2x2 covariance matrix."""

    from matplotlib.patches import Ellipse

    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_sigma * np.sqrt(np.maximum(vals, 0.0))
    patch = Ellipse(
        center,
        width,
        height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        label=label,
        **kwargs,
    )
    ax.add_patch(patch)
    return patch


def load_ellipse_set(path):
    """Load ellipse parameters written by ``digitized/fit_outer_ellipse.py``.

    Returns ``(ellipses, meta)`` where ``ellipses`` maps the integer sigma level
    to a dict with ``center``, ``semi_axes``, ``angle_deg``,
    ``contour_covariance`` (the covariance whose Delta chi^2 = 1 contour is that
    curve) and ``implied_1sigma_covariance``.
    """

    import json

    with open(path) as handle:
        payload = json.load(handle)

    ellipses = {}
    for level, entry in payload["ellipses"].items():
        ellipses[int(level)] = {
            "center": np.array(entry["center"], dtype=float),
            "semi_axes": np.array(entry["semi_axes"], dtype=float),
            "angle_deg": float(entry["angle_deg"]),
            "contour_covariance": np.array(entry["contour_covariance"], dtype=float),
            "implied_1sigma_covariance": np.array(
                entry["implied_1sigma_covariance"], dtype=float
            ),
        }
    return ellipses, payload


def swap_ellipse_axes(entry: dict) -> dict:
    """Re-express one ellipse under the coordinate swap (x, y) -> (y, x).

    The swap is a reflection, so the centre swaps, the semi-axes keep their
    identity (the major axis stays the major axis) and the orientation becomes
    ``90 - angle``.  Rotating by -90 instead, or swapping the semi-axes, gives a
    different ellipse whenever the two axes have different scales.
    """

    swapped = {
        "center": entry["center"][::-1].copy(),
        "semi_axes": entry["semi_axes"].copy(),
        "angle_deg": 90.0 - entry["angle_deg"],
    }
    for key in ("contour_covariance", "implied_1sigma_covariance"):
        if key in entry:
            cov = np.asarray(entry[key], dtype=float)
            swapped[key] = cov[::-1, ::-1].copy()
    return swapped


def plot_ellipse_params(ax, center, semi_axes, angle_deg, *, color=INK, label=None, **kwargs):
    """Draw an ellipse given centre, semi-axes and orientation."""

    from matplotlib.patches import Ellipse

    patch = Ellipse(
        tuple(center),
        2.0 * semi_axes[0],
        2.0 * semi_axes[1],
        angle=angle_deg,
        facecolor="none",
        edgecolor=color,
        label=label,
        **kwargs,
    )
    ax.add_patch(patch)
    return patch


def annotate_right(ax, x, y, text, *, color=INK_SECONDARY, dx=0.01, **kwargs):
    """Direct label placed just past the end of a curve."""

    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset fontsize",
        color=color,
        va="center",
        fontsize=9,
        **kwargs,
    )


def hline_reference(ax, value: float, label: str, *, color: str = INK_SECONDARY, **kwargs):
    ax.axhline(value, color=color, lw=1.2, ls="--", zorder=1, **kwargs)
    ax.annotate(
        label,
        xy=(0.99, value),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=color,
    )
    return ax


def table(rows, headers, *, floatfmt="{:.4g}") -> str:
    """Small monospace table for notebook text output."""

    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    body = [[fmt(v) for v in row] for row in rows]
    widths = [
        max(len(str(headers[i])), *(len(r[i]) for r in body)) if body else len(str(headers[i]))
        for i in range(len(headers))
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "-" * len(line)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in body]
    return "\n".join(out)
