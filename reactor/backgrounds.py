"""Background models for JUNO prompt-energy spectra.

Shapes are simple analytic parameterisations normalised to unit integral over
the analysis window; the absolute rates are supplied by the caller.  Nominal
rates for 20 kt follow the JUNO Yellow Book (arXiv:2104.02565) Table 5-1.
"""

from __future__ import annotations

import numpy as np

# Nominal background rates in events / day for the 20 kt fiducial volume.
JUNO_BACKGROUND_RATES = {
    "geoneutrino": 1.2,
    "accidental": 0.8,
    "li9he8": 0.8,
    "fast_neutron": 0.1,
    "alpha_n": 0.05,
    "world_reactor": 1.0,
}

# Rate uncertainties (fractional) quoted alongside them.
JUNO_BACKGROUND_RATE_ERRORS = {
    "geoneutrino": 0.30,
    "accidental": 0.01,
    "li9he8": 0.20,
    "fast_neutron": 1.00,
    "alpha_n": 0.50,
    "world_reactor": 0.02,
}


def _beta_like(e: np.ndarray, endpoint: float, weight: float = 1.0) -> np.ndarray:
    x = np.clip(e / endpoint, 0.0, 1.0)
    shape = weight * x**2 * (1.0 - x) ** 2
    return np.where(e >= endpoint, 0.0, shape)


def geoneutrino_shape(prompt_centers_mev: np.ndarray) -> np.ndarray:
    """U/Th geoneutrino prompt spectrum (endpoints 2.49 and 1.47 MeV prompt)."""

    e = np.asarray(prompt_centers_mev, dtype=float)
    # Chondritic Th/U mass ratio 3.9 gives roughly a 0.27 event ratio.
    return _beta_like(e, 2.49, 1.0) + _beta_like(e, 1.47, 0.45)


def accidental_shape(prompt_centers_mev: np.ndarray) -> np.ndarray:
    """Steeply falling accidental-coincidence prompt spectrum."""

    e = np.asarray(prompt_centers_mev, dtype=float)
    return np.exp(-e / 0.55)


def li9he8_shape(prompt_centers_mev: np.ndarray) -> np.ndarray:
    """Cosmogenic 9Li / 8He beta-n spectrum, broad up to ~11 MeV."""

    e = np.asarray(prompt_centers_mev, dtype=float)
    return _beta_like(e, 11.0, 1.0) + 0.3 * _beta_like(e, 8.0, 1.0)


def fast_neutron_shape(prompt_centers_mev: np.ndarray) -> np.ndarray:
    """Fast-neutron recoils: nearly flat over the analysis window."""

    e = np.asarray(prompt_centers_mev, dtype=float)
    return np.ones_like(e)


def alpha_n_shape(prompt_centers_mev: np.ndarray) -> np.ndarray:
    """13C(alpha,n)16O: low-energy rise plus the 6.13 MeV de-excitation bump."""

    e = np.asarray(prompt_centers_mev, dtype=float)
    low = np.exp(-e / 1.2)
    bump = 0.15 * np.exp(-0.5 * ((e - 4.5) / 0.7) ** 2)
    return low + bump


def world_reactor_shape(prompt_centers_mev: np.ndarray) -> np.ndarray:
    """Distant world reactors: same prompt shape as the signal, fully averaged."""

    from .flux import juno_average_fractions, mixed_spectrum_per_fission
    from .cross_sections import load_ibd_cross_section
    from .constants import PROMPT_ENERGY_OFFSET_MEV

    e = np.asarray(prompt_centers_mev, dtype=float)
    e_nu = e + PROMPT_ENERGY_OFFSET_MEV
    shape = mixed_spectrum_per_fission(e_nu, juno_average_fractions()) * load_ibd_cross_section()(e_nu)
    # Fully averaged oscillation: constant survival factor, so shape is unchanged.
    return np.maximum(shape, 0.0)


SHAPES = {
    "geoneutrino": geoneutrino_shape,
    "accidental": accidental_shape,
    "li9he8": li9he8_shape,
    "fast_neutron": fast_neutron_shape,
    "alpha_n": alpha_n_shape,
    "world_reactor": world_reactor_shape,
}


def background_counts(
    name: str,
    prompt_edges_mev: np.ndarray,
    exposure_days: float,
    rate_per_day: float,
) -> np.ndarray:
    """Expected counts per reconstructed bin for one background component.

    ``rate_per_day`` is the total rate inside the analysis window, so the shape
    is renormalised over the supplied binning.
    """

    if name not in SHAPES:
        raise KeyError(f"Unknown background {name!r}. Known: {sorted(SHAPES)}")
    edges = np.asarray(prompt_edges_mev, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    shape = SHAPES[name](centers) * widths
    total = shape.sum()
    if total <= 0.0:
        return np.zeros_like(centers)
    return shape / total * rate_per_day * exposure_days


def total_background_counts(
    rates: dict[str, float],
    prompt_edges_mev: np.ndarray,
    exposure_days: float,
) -> np.ndarray:
    """Sum of all requested background components."""

    edges = np.asarray(prompt_edges_mev, dtype=float)
    total = np.zeros(edges.size - 1)
    for name, rate in rates.items():
        total += background_counts(name, edges, exposure_days, rate)
    return total


def geoneutrino_counts(
    prompt_edges_mev: np.ndarray,
    exposure_days: float,
    rate_per_day: float = 1.2,
) -> np.ndarray:
    """Backwards-compatible helper for the geoneutrino component alone."""

    return background_counts("geoneutrino", prompt_edges_mev, exposure_days, rate_per_day)
