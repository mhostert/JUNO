"""Reactor antineutrino survival probabilities.

Three levels of approximation are available:

* :func:`survival_probability_2nu` -- the two-flavour form used for the
  short-baseline intuition, ``1 - sin^2(2 th13) sin^2(Delta_ee)``.
* :func:`survival_probability_ee` -- the exact three-flavour vacuum
  probability written in terms of ``Delta m^2_ee`` (Minakata et al.).
* :func:`survival_probability_matter` -- exact numerical three-flavour
  probability in constant-density matter, used for the mass-ordering study.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    DEFAULT_OSCILLATION_PARAMS,
    EARTH_CRUST_DENSITY_G_CM3,
    ELECTRON_FRACTION_YE,
    KM_EV2_PER_GEV,
    OscillationParameters,
)


def _phases(e_nu_mev, baseline_km, params):
    e_gev = np.maximum(np.asarray(e_nu_mev, dtype=float) / 1000.0, 1.0e-12)
    delta_ee = KM_EV2_PER_GEV * abs(params.dm2_ee) * baseline_km / e_gev
    delta_21 = KM_EV2_PER_GEV * params.dm2_21 * baseline_km / e_gev
    return delta_ee, delta_21


def survival_probability_2nu(
    e_nu_mev: np.ndarray | float,
    baseline_km: float,
    params: OscillationParameters = DEFAULT_OSCILLATION_PARAMS,
) -> np.ndarray:
    """Two-flavour short-baseline limit, 1 - sin^2(2 th13) sin^2(Delta_ee)."""

    delta_ee, _ = _phases(e_nu_mev, baseline_km, params)
    return 1.0 - params.sin2_2theta13 * np.sin(delta_ee) ** 2


def survival_probability_ee(
    e_nu_mev: np.ndarray | float,
    baseline_km: float,
    params: OscillationParameters = DEFAULT_OSCILLATION_PARAMS,
) -> np.ndarray:
    """Exact three-flavour vacuum P_ee in the Delta m^2_ee parameterisation.

    Follows Minakata, Nunokawa, Parke & Zukanovich Funchal (hep-ph/0701151):

        P = 1 - (1/2) sin^2(2 th13) [1 - sqrt(1 - sin^2(2 th12) sin^2 D21)
                                         cos(2|D_ee| +- Phi)]
              - cos^4(th13) sin^2(2 th12) sin^2(D21)

    with Phi = arctan(cos(2 th12) tan D21) - D21 cos(2 th12), and the sign of
    Phi given by the mass ordering (+ for normal, - for inverted).  That sign
    is the entire vacuum mass-ordering signal: it is invisible at short
    baselines, where Phi -> 0, and is what JUNO resolves at 52.5 km.
    """

    delta_ee, delta_21 = _phases(e_nu_mev, baseline_km, params)

    s12 = params.sin2_theta12
    c13 = 1.0 - params.sin2_theta13
    cos_2theta12 = 1.0 - 2.0 * s12
    sin2_2theta12 = params.sin2_2theta12

    # arctan(cos(2 th12) tan D21) must be taken on the branch that is
    # continuous in D21, not the principal one: tan has period pi, so add
    # pi * round(D21/pi).  Without this Phi jumps by 2 pi at D21 = pi, which
    # is outside JUNO's window (D21 < 2.8 there) but inside it for L/E
    # beyond ~35 km/MeV.  With the correct branch Phi -> 2 pi sin^2(th12)
    # at D21 = pi, the plateau of Fig. 13 of arXiv:2107.12410.
    # The reduction to the principal half-period must be applied to the
    # argument of the tangent, not just added afterwards: tan(D21) and
    # tan(D21 - k pi) have opposite signs in floating point when D21 lands
    # exactly on a half-period boundary, which would put Phi off by pi there.
    k = np.floor(delta_21 / np.pi + 0.5)
    reduced = delta_21 - k * np.pi  # in [-pi/2, pi/2)
    phi = np.arctan(cos_2theta12 * np.tan(reduced)) + k * np.pi
    phi -= delta_21 * cos_2theta12
    phi = np.sign(params.ordering) * phi

    solar_amp = np.sqrt(np.maximum(0.0, 1.0 - sin2_2theta12 * np.sin(delta_21) ** 2))

    p_ee = 1.0
    p_ee = p_ee - 0.5 * params.sin2_2theta13 * (
        1.0 - solar_amp * np.cos(2.0 * np.abs(delta_ee) + phi)
    )
    p_ee = p_ee - c13**2 * sin2_2theta12 * np.sin(delta_21) ** 2
    return np.clip(p_ee, 0.0, 1.0)


def survival_probability_3nu_masses(
    e_nu_mev: np.ndarray | float,
    baseline_km: float,
    params: OscillationParameters = DEFAULT_OSCILLATION_PARAMS,
) -> np.ndarray:
    """Textbook vacuum P_ee written directly with Delta m^2_21, 31, 32.

    Independent implementation used as a consistency check on
    :func:`survival_probability_ee`.
    """

    e_gev = np.maximum(np.asarray(e_nu_mev, dtype=float) / 1000.0, 1.0e-12)
    s13 = params.sin2_theta13
    c13 = 1.0 - s13
    s12 = params.sin2_theta12
    c12 = 1.0 - s12

    # |U_e1|^2, |U_e2|^2, |U_e3|^2
    ue1, ue2, ue3 = c12 * c13, s12 * c13, s13

    d21 = KM_EV2_PER_GEV * params.dm2_21 * baseline_km / e_gev
    d31 = KM_EV2_PER_GEV * params.dm2_31 * baseline_km / e_gev
    d32 = KM_EV2_PER_GEV * params.dm2_32 * baseline_km / e_gev

    p_ee = 1.0
    p_ee -= 4.0 * ue1 * ue2 * np.sin(d21) ** 2
    p_ee -= 4.0 * ue1 * ue3 * np.sin(d31) ** 2
    p_ee -= 4.0 * ue2 * ue3 * np.sin(d32) ** 2
    return np.clip(p_ee, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Constant-density matter (exact, numerical)
# ---------------------------------------------------------------------------
def matter_potential_ev(density_g_cm3: float = EARTH_CRUST_DENSITY_G_CM3) -> float:
    """Charged-current matter potential V = sqrt(2) G_F n_e, in eV.

    V[eV] = 7.63e-14 * Y_e * rho[g/cm^3].
    """

    return 7.6324e-14 * ELECTRON_FRACTION_YE * density_g_cm3


def survival_probability_matter(
    e_nu_mev: np.ndarray | float,
    baseline_km: float,
    params: OscillationParameters = DEFAULT_OSCILLATION_PARAMS,
    density_g_cm3: float = EARTH_CRUST_DENSITY_G_CM3,
) -> np.ndarray:
    """Exact three-flavour P(nubar_e -> nubar_e) in constant-density matter.

    The 3x3 Hamiltonian is diagonalised numerically at every energy.  The
    mass ordering enters through the sign of ``params.dm2_31``.  For
    antineutrinos the matter potential flips sign.
    """

    e = np.atleast_1d(np.asarray(e_nu_mev, dtype=float))
    s13 = np.sqrt(params.sin2_theta13)
    c13 = np.sqrt(1.0 - params.sin2_theta13)
    s12 = np.sqrt(params.sin2_theta12)
    c12 = np.sqrt(1.0 - params.sin2_theta12)
    # P_ee is independent of theta23 and delta_CP, but the diagonalisation
    # needs a genuine unitary matrix, so fix theta23 = pi/4 and delta = 0.
    s23 = c23 = np.sqrt(0.5)
    u = np.array(
        [
            [c12 * c13, s12 * c13, s13],
            [-s12 * c23 - c12 * s23 * s13, c12 * c23 - s12 * s23 * s13, s23 * c13],
            [s12 * s23 - c12 * c23 * s13, -c12 * s23 - s12 * c23 * s13, c23 * c13],
        ]
    )

    m2 = np.diag([0.0, params.dm2_21, params.dm2_31])  # eV^2
    h_vac = u @ m2 @ u.T  # eV^2, in flavour basis

    # H = (1/2E) [ M^2 + diag(A, 0, 0) ], A = +2 E V for nu, -2 E V for nubar.
    v_ev = matter_potential_ev(density_g_cm3)
    a = -2.0 * (e * 1.0e6) * v_ev  # eV^2, minus sign for antineutrinos

    h = np.broadcast_to(h_vac, (e.size, 3, 3)).copy()
    h[:, 0, 0] += a

    evals, evecs = np.linalg.eigh(h)  # eV^2 ; evecs columns are mass states

    # Phase differences: Delta_ij = 1.267 * (m_i^2 - m_j^2)[eV^2] * L[km] / E[GeV]
    e_gev = np.maximum(e / 1000.0, 1.0e-12)
    phase = KM_EV2_PER_GEV * baseline_km / e_gev  # multiplies eV^2 differences

    ue = evecs[:, 0, :] ** 2  # |U^m_ei|^2, shape (n_e, 3)
    p_ee = np.ones(e.size)
    for i in range(3):
        for j in range(i + 1, 3):
            dphi = phase * (evals[:, i] - evals[:, j])
            p_ee -= 4.0 * ue[:, i] * ue[:, j] * np.sin(dphi) ** 2
    return np.clip(p_ee, 0.0, 1.0) if np.ndim(e_nu_mev) else float(p_ee[0])


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def first_minimum_loe(params: OscillationParameters = DEFAULT_OSCILLATION_PARAMS) -> float:
    """L/E in km/MeV of the first atmospheric disappearance maximum."""

    return np.pi / (2.0 * KM_EV2_PER_GEV * abs(params.dm2_ee)) / 1000.0


def dm2ee_from_dm231(
    dm2_31: float,
    sin2_theta12: float = DEFAULT_OSCILLATION_PARAMS.sin2_theta12,
    dm2_21: float = DEFAULT_OSCILLATION_PARAMS.dm2_21,
) -> float:
    """Convert a signed Delta m^2_31 to |Delta m^2_ee|."""

    return abs(dm2_31 - sin2_theta12 * dm2_21)
