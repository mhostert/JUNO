"""Neutrino interactions on the deuterium in the JUNO scintillator.

Natural hydrogen carries 156 ppm of deuterium, so JUNO's 1.44e33 free protons
come with 2.2e29 deuterons -- about 750 kg of D, 2.3% of the 13C count.

Deuterium is the target one would most like to have: the neutral-current
breakup nu + d -> nu + n + p is a pure Gamow-Teller transition on the simplest
of all nuclei, with a matrix element known from pionless effective field theory
to about 1% rather than from a shell model.  Two channels are open, and the gap
between their thresholds decides everything:

    nubar_e + d -> nubar_e + n + p     E_th = B_d                    = 2.224 MeV
    nubar_e + d -> e+ + n + n          E_th = B_d + (m_n - m_p) + m_e = 4.028 MeV

with B_d = m_p + m_n - M_d the deuteron binding energy.  The neutral current
pays nothing beyond the binding energy, since the outgoing neutrino is massless
and the final state is just an unbound np pair; the charged current must in
addition turn a proton into a neutron and materialise a positron.

Cross-section normalisation
---------------------------
As for 13C, the *shapes* are derived and the two overall *scales* are external.

Both channels are allowed (Gamow-Teller) transitions, so near threshold the
energy dependence is phase space: an outgoing lepton against a two-body
continuum of relative energy eps,

    NC:  sigma ∝ int_0^Q  (Q - eps)^2 sqrt(eps) d(eps) ∝ Q^(7/2),  Q = E - B_d
    CC:  sigma ∝ int_0^Q' p_e E_e sqrt(eps) d(eps),  E_e = Q' - eps + m_e

(the sqrt(eps) is the non-relativistic np / nn density of states).  This omits
the 1S0 final-state interaction, which enhances small eps in *both* channels and
therefore largely cancels in the ratio and in the reactor folding; the shapes
here are used only to weight the reactor spectrum and to compare the HALEU core
with the 235U-dominated cores at which the scales were fixed.

The scales are the per-fission cross sections for a 235U reactor spectrum,

    sigma_CC = 1.06e-44 cm^2 / fission,   sigma_NC = 5.6e-45 cm^2 / fission

known to roughly 10%: theory from pionless EFT (Kubodera-Nozawa and successors)
and measurement from the Bugey nubar-d experiment, which agree at that level.
Every rate below is derived from those two numbers and the geometry, so the
uncertainty on any quoted rate is the ~10% on the scale, not more.

Signatures
----------
The two channels part company completely, and it is the signature rather than
the rate that decides.  The charged current gives a prompt positron followed by
*two* neutron captures sharing a vertex -- a triple coincidence, essentially
background free.  The neutral current gives a lone 2.22 MeV capture gamma with
no prompt at all (the proton recoil is sub-MeV and heavily quenched), buried in
the singles continuum.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    DELTA_NP,
    JUNO2025_TARGET_PROTONS,
    M_E,
    SECONDS_PER_DAY,
)
from .detector import integration_weights

#: Deuterium/hydrogen atom ratio in natural water and hydrocarbons (VSMOW).
D_H_ABUNDANCE = 1.5576e-4

#: Deuterons in the JUNO fiducial target.
N_D = D_H_ABUNDANCE * JUNO2025_TARGET_PROTONS
#: ... and their mass in kg.
M_D_KG = N_D * 2.0141 / 6.02214076e23 / 1.0e3

#: Deuteron binding energy [MeV]: m_p + m_n - M_d.
B_D = 2.2246
#: Thresholds [MeV].
E_TH_NC = B_D
E_TH_CC = B_D + DELTA_NP + M_E

#: Per-fission cross sections for a 235U-dominated reactor spectrum
#: [cm^2 / fission], and their fractional uncertainty.
SIGMA_CC_PER_FISSION = 1.06e-44
SIGMA_NC_PER_FISSION = 5.6e-45
SIGMA_PER_FISSION_REL_ERR = 0.10


def shape_nc(e_mev):
    """Relative nubar + d -> nubar + n + p cross section (allowed, phase space)."""

    q = np.asarray(e_mev, dtype=float) - E_TH_NC
    return np.where(q > 0.0, np.maximum(q, 0.0) ** 3.5, 0.0)


def shape_cc(e_mev, n_eps: int = 200):
    """Relative nubar_e + d -> e+ + n + n cross section (allowed, phase space)."""

    e = np.atleast_1d(np.asarray(e_mev, dtype=float))
    q = e - E_TH_CC
    out = np.zeros_like(e)
    good = q > 0.0
    if np.any(good):
        # eps = x * Q, x in (0, 1): trapezoid on a fixed grid, vectorised over E
        x = np.linspace(0.0, 1.0, n_eps)
        qq = q[good][:, None]
        eps = x[None, :] * qq
        e_tot = (qq - eps) + M_E
        p = np.sqrt(np.maximum(e_tot**2 - M_E**2, 0.0))
        out[good] = np.trapezoid(p * e_tot * np.sqrt(eps), eps, axis=1)
    return out if np.ndim(e_mev) else float(out[0])


def flux_fraction_above(flux_density, e_grid, threshold: float) -> float:
    """Fraction of the modelled antineutrino flux lying above ``threshold``."""

    w = integration_weights(e_grid)
    tot = float(np.sum(w * flux_density))
    above = float(np.sum(w * np.where(e_grid > threshold, flux_density, 0.0)))
    return above / tot if tot > 0 else 0.0


def _spectrum_correction(flux_density, flux_235, e_grid, shape) -> float:
    """Ratio of the shape-weighted HALEU flux to the reference 235U flux.

    Both fluxes are normalised to unit total, so this is purely the effect of
    the core's isotopic mix on the threshold-weighted rate.
    """

    w = integration_weights(e_grid)
    s = shape(e_grid)
    a = float(np.sum(w * flux_density * s)) / float(np.sum(w * flux_density))
    b = float(np.sum(w * flux_235 * s)) / float(np.sum(w * flux_235))
    return a / b if b > 0 else 1.0


def rates_per_day(geom, flux_density, flux_235, e_grid, efficiency: float = 1.0,
                  n_d: float = N_D) -> dict:
    """CC and NC event rates per day.

    ``geom`` is the fission rate divided by 4 pi L^2 [fissions / cm^2 / s], the
    same quantity the EvES and IBD rates are built from, so the per-fission
    cross sections apply directly.  ``flux_density`` and ``flux_235`` are the
    core's and a pure-235U antineutrino spectrum on ``e_grid``, used only for
    the small isotopic correction.
    """

    out = {}
    for name, sigma, shape in (("CC", SIGMA_CC_PER_FISSION, shape_cc),
                               ("NC", SIGMA_NC_PER_FISSION, shape_nc)):
        k = _spectrum_correction(flux_density, flux_235, e_grid, shape)
        out[name] = geom * n_d * sigma * k * efficiency * SECONDS_PER_DAY
    return out


def accidental_double_neutron_per_day(ibd_rate_per_day: float,
                                      fiducial_radius_m: float = 16.5,
                                      window_s: float = 1.0e-3,
                                      distance_m: float = 1.0) -> float:
    """Rate of IBD pairs faking the CC triple coincidence.

    An ordinary IBD picks up a second neutron capture from another IBD inside
    the coincidence window and the vertex cut.  Both this and the signal scale
    linearly with reactor power, so the signal-to-background is a property of
    the cut rather than of the exposure.
    """

    r = ibd_rate_per_day / SECONDS_PER_DAY
    f_vol = (distance_m / fiducial_radius_m) ** 3
    return r * (r * window_s * f_vol) * SECONDS_PER_DAY


def doped_rates_per_day(rates: dict, d_over_h: float) -> dict:
    """Rates for a scintillator doped to a deuterium/hydrogen ratio ``d_over_h``."""

    k = d_over_h / D_H_ABUNDANCE
    return {name: k * value for name, value in rates.items()}


def d2o_cell_rates_per_day(rates: dict, mass_tonnes: float = 1.0,
                           distance_m: float = 10.0,
                           reference_distance_m: float = 50.0) -> dict:
    """Rates for a dedicated D2O cell of ``mass_tonnes`` at ``distance_m``.

    ``rates`` are the in-situ JUNO rates at ``reference_distance_m``.
    """

    n_d_cell = mass_tonnes * 1.0e6 / 20.0276 * 6.02214076e23 * 2.0
    k = (n_d_cell / N_D) * (reference_distance_m / distance_m) ** 2
    return {name: k * value for name, value in rates.items()}
