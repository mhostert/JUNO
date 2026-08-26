"""Neutrino interactions on 13C in the JUNO scintillator.

Natural carbon is 1.1% 13C by atom, so a 20 kt carbon-rich scintillator contains
~210 t of it -- 9.7e30 nuclei.  JUNO exploits this for a model-independent 8B solar
measurement [PoS(ICRC2025)1041]; here the same two channels appear in the
single-hit spectrum, and one of them is a reactor signal in its own right.

**Neutral current**, nu_x + 13C -> nu_x + 13C*(3/2-, 3.685 MeV).  Flavour blind,
identical for neutrinos and *antineutrinos*, so the near reactor excites it too.
The level sits below the 4.946 MeV neutron separation energy, so it is particle
bound and de-excites by a single 3.685 MeV gamma: a **monoenergetic line**,
73 keV wide at JUNO's resolution, sitting above the 2.615 MeV external-gamma
wall in the cleanest part of the E$\\nu$ES window.

**Charged current**, nu_e + 13C -> e- + 13N(g.s.), threshold 2.2 MeV.  This one
is nu_e only: the antineutrino partner nubar_e + 13C -> e+ + 13B has a threshold
of 14.5 MeV, far above the reactor spectrum, so the reactor contributes
*nothing* here and the CC channel is purely a solar background.  The 13N decays
back by beta+ with a 863 s lifetime, giving a delayed coincidence that tags it.

Cross-section normalisation
---------------------------
Both channels are allowed transitions, so near threshold the energy dependence
is pure phase space:

    NC:  sigma ∝ (E_nu - 3.685)^2      (outgoing massless neutrino)
    CC:  sigma ∝ p_e E_e,  E_e = E_nu - Q + m_e   (E_e is the *total* energy,
                                                   Q = 2.22 MeV the threshold)

The overall scales are fixed by JUNO's own quoted 8B yields -- 3032 NC and 3929
CC events in 10 years -- folded over the 8B spectrum (NC with the full flux, CC
with the LMA-suppressed nu_e component).  This is an *anchored extrapolation*:
the 8B spectrum weights 3.7-15 MeV while the reactor weights 3.7-9 MeV, so the
phase-space law is being carried a modest way down in energy, and the resulting
reactor rate should be read as good to a factor of order unity rather than a
percent-level prediction.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    AVOGADRO,
    JUNO_CARBON_MASS_FRACTION,
    MOLAR_MASS_C,
    SECONDS_PER_DAY,
)
from .detector import integration_weights
from .solar import SOLAR_FLUXES, _load_solar_shape, solar_pee
from .theta13 import DEFAULT_TRUTH

M_E = 0.51099895

#: 13C*(3/2-) excitation energy [MeV].  S_n(13C) = 4.946 MeV, so it gamma-decays.
EX_NC = 3.685
#: nu_e + 13C -> e- + 13N(g.s.) threshold [MeV]: the atomic mass difference
#: 13N - 13C, since the electron created and the one shed by the Z change cancel.
Q_CC = 2.2205
#: 13N beta+ lifetime [s] -- the delayed tag on the CC channel.
TAU_N13 = 863.0

#: JUNO 8B yields over 10 years in 20 kt [PoS(ICRC2025)1041 Tab. 1].
JUNO_NC_10YR = 3032.0
JUNO_CC_10YR = 3929.0
_TEN_YEARS = 10.0 * 365.25 * SECONDS_PER_DAY

#: 13C content: 20 kt LS, 87.99% carbon by mass, 1.1% 13C *by atom* (the natural
#: isotopic abundance), so the carbon mole count uses the natural mean molar
#: mass 12.011 and the abundance is applied to it.  9.71e30 nuclei, 210 t.
LS_MASS_G = 2.0e10
CARBON_MASS_FRACTION = JUNO_CARBON_MASS_FRACTION
C13_ABUNDANCE = 0.011
N_13C = LS_MASS_G * CARBON_MASS_FRACTION / MOLAR_MASS_C * AVOGADRO * C13_ABUNDANCE
M_13C_TONNES = N_13C * 13.003 / AVOGADRO / 1.0e6


def _b8_flux(e_mev):
    tab = _load_solar_shape("b8")
    return SOLAR_FLUXES["b8"] * np.interp(e_mev, tab[:, 0], tab[:, 1],
                                          left=0.0, right=0.0)


def _shape_nc(e_mev):
    e = np.asarray(e_mev, dtype=float)
    return np.where(e > EX_NC, (e - EX_NC) ** 2, 0.0)


def _shape_cc(e_mev):
    e = np.asarray(e_mev, dtype=float)
    e_tot = e - Q_CC + M_E          # total electron energy
    p = np.sqrt(np.maximum(e_tot**2 - M_E**2, 0.0))
    return np.where(e > Q_CC, p * e_tot, 0.0)


def _anchor():
    """Fix the two normalisations against JUNO's quoted 8B yields."""

    e = np.linspace(0.1, 20.0, 6000)
    w = integration_weights(e)
    phi = _b8_flux(e)
    c_nc = JUNO_NC_10YR / _TEN_YEARS / N_13C / np.sum(w * phi * _shape_nc(e))
    pee = solar_pee(e, DEFAULT_TRUTH)
    c_cc = JUNO_CC_10YR / _TEN_YEARS / N_13C / np.sum(w * phi * pee * _shape_cc(e))
    return c_nc, c_cc


C_NC, C_CC = _anchor()


def sigma_nc(e_mev):
    """nu_x + 13C -> nu_x + 13C*(3.685) cross section [cm^2], any flavour."""

    return C_NC * _shape_nc(e_mev)


def sigma_cc(e_mev):
    """nu_e + 13C -> e- + 13N(g.s.) cross section [cm^2].  nu_e only."""

    return C_CC * _shape_cc(e_mev)


def reactor_nc_rate(flux_density, e_grid, efficiency: float = 1.0,
                    n_13c: float = N_13C) -> float:
    """NC excitation rate [per second] for a reactor flux at the detector.

    ``flux_density`` is dPhi/dE [/cm^2/s/MeV] on ``e_grid``.  Antineutrinos
    excite the state exactly as neutrinos do.
    """

    w = integration_weights(e_grid)
    return float(n_13c * efficiency * np.sum(w * flux_density * sigma_nc(e_grid)))


def solar_nc_rate(efficiency: float = 1.0, n_13c: float = N_13C) -> float:
    """Solar 8B NC excitation rate [per second]."""

    return JUNO_NC_10YR / _TEN_YEARS * efficiency * n_13c / N_13C


def solar_cc_rate(efficiency: float = 1.0, n_13c: float = N_13C,
                  tag_efficiency: float = 0.5) -> float:
    """Solar 8B CC rate [per second] surviving the delayed 13N tag."""

    return (JUNO_CC_10YR / _TEN_YEARS * efficiency * n_13c / N_13C
            * (1.0 - tag_efficiency))


def solar_cc_visible_shape(t_edges):
    """Normalised prompt-electron spectrum of the solar CC channel, per bin.

    The electron carries T_e = E_nu - Q_CC, so the 8B spectrum maps directly
    onto visible energy.
    """

    e = np.linspace(Q_CC + 1e-3, 20.0, 4000)
    w = integration_weights(e)
    dens = _b8_flux(e) * solar_pee(e, DEFAULT_TRUTH) * _shape_cc(e) * w
    t = e - Q_CC
    hist, _ = np.histogram(t, bins=t_edges, weights=dens)
    tot = dens.sum()
    return hist / tot if tot > 0 else hist
