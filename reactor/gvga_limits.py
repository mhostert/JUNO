"""Existing constraints in the (g_V, g_A) neutrino-electron NC-coupling plane.

Convention (ours, PDG, CHARM II, TEXONO, LSND): g_V = -1/2 + 2 sin^2 theta_W
(SM -0.0395 incl. radiative corrections; -0.0554 at tree level with
sin^2 theta_W = 0.2223), g_A = -1/2.  Electron-flavour experiments see the CC
piece too, so their cross sections depend on (g_V + 1, g_A + 1) and constrain
BANDS along the resulting degeneracies, not ellipses:

  nu_e e   (LSND, LAMPF): sigma/sigma_0 = (g_V+g_A+2)^2 + (g_V-g_A)^2/3
  nubar_e e (TEXONO)    : sigma/sigma_0 = (g_V-g_A)^2   + (g_V+g_A+2)^2/3

Muon-flavour experiments (CHARM II, BNL E734) are pure NC and give ellipses.
Neutrino tridents (CCFR, CHARM II) constrain the nu_mu-MUON couplings, not
nu-e, and are deliberately not drawn on this plane.

Numbers (all 1 sigma unless stated):
  CHARM II   g_V = -0.035 +- 0.017, g_A = -0.503 +- 0.017      [Vilain:1994qy]
  BNL E734   g_V = -0.107 +- 0.045, g_A = -0.514 +- 0.036      [Ahrens:1990fp]
  PDG 2024 WA g_V = -0.040 +- 0.015, g_A = -0.507 +- 0.014, rho = -0.05
  LSND       sigma/sigma_0 = 2.34 +- 0.35 (nu_e)               [LSND:2001akn]
  TEXONO     xi = sigma/sigma_SM = 1.08 +- 0.26 (nubar_e), SM at s2w = 0.2387
                                                               [TEXONO:2009knm]
"""
from __future__ import annotations

import numpy as np

SW2_TEXONO = 0.2387


def nue_band(gv, ga):
    """nu_e e cross-section factor (LSND/LAMPF), relative to sigma_0."""
    return (gv + ga + 2.0) ** 2 + (gv - ga) ** 2 / 3.0


def nubar_e_band(gv, ga):
    """nubar_e e cross-section factor (TEXONO), relative to sigma_0."""
    return (gv - ga) ** 2 + (gv + ga + 2.0) ** 2 / 3.0


def draw_gvga_limits(ax, pl, zoom: bool = False, n_sigma: float = 1.0):
    """Overlay the existing (g_V, g_A) constraints on ``ax``.

    ``zoom=True`` draws only the ellipses (for a plot window of ~+-0.05 around
    the SM point); ``zoom=False`` also fills the nu_e / nubar_e bands.
    Returns nothing; adds labelled artists for the legend.
    """
    from matplotlib.patches import Ellipse

    def ell(cx, cy, sx, sy, color, label, ls="-", rho=0.0):
        cov = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        w, h = 2 * n_sigma * np.sqrt(vals[1]), 2 * n_sigma * np.sqrt(vals[0])
        ax.add_patch(Ellipse((cx, cy), w, h, angle=ang, fill=False, edgecolor=color,
                             lw=1.6, ls=ls, label=label))

    ell(-0.035, -0.503, 0.017, 0.017, pl.INK_SECONDARY, r"CHARM II ($\nu_\mu e$), $1\sigma$")
    ell(-0.040, -0.507, 0.015, 0.014, pl.INK, r"PDG 2024 world average, $1\sigma$",
        ls="--", rho=-0.05)
    if zoom:
        return
    ell(-0.107, -0.514, 0.045, 0.036, pl.MAGENTA, r"BNL E734 ($\nu_\mu e$), $1\sigma$")

    xl, yl = ax.get_xlim(), ax.get_ylim()
    gv, ga = np.meshgrid(np.linspace(xl[0], xl[1], 400), np.linspace(yl[0], yl[1], 400))
    # LSND nu_e band: 2.34 +- 0.35 (1 sigma)
    lo, hi = 2.34 - n_sigma * 0.35, 2.34 + n_sigma * 0.35
    ax.contourf(gv, ga, nue_band(gv, ga), levels=[lo, hi], colors=[pl.GREEN], alpha=0.25)
    ax.plot([], [], color=pl.GREEN, lw=6, alpha=0.4, label=r"LSND ($\nu_e e$), $1\sigma$")
    # TEXONO nubar_e band: xi = 1.08 +- 0.26 of the SM value at s2w = 0.2387
    gv_t = 2 * SW2_TEXONO - 0.5
    ref = nubar_e_band(gv_t, -0.5)
    lo, hi = (1.08 - n_sigma * 0.26) * ref, (1.08 + n_sigma * 0.26) * ref
    ax.contourf(gv, ga, nubar_e_band(gv, ga), levels=[lo, hi], colors=[pl.ORANGE], alpha=0.25)
    ax.plot([], [], color=pl.ORANGE, lw=6, alpha=0.4, label=r"TEXONO ($\bar\nu_e e$), $1\sigma$")
