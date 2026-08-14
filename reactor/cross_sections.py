"""Inverse beta decay cross section.

Two implementations are provided so that they can be cross-checked against
each other in the validation notebook:

* :class:`IBDCrossSection` -- interpolation of the tabulated cross section
  shipped in ``reactor/data``.
* :func:`vogel_beacom` -- the analytic naive / first-order expressions of
  Vogel & Beacom, Phys. Rev. D60 (1999) 053003.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from .constants import (
    DELTA_NP,
    F_PLUS_G_SQUARED,
    IBD_THRESHOLD,
    M_E,
    M_P,
    NEUTRON_LIFETIME_S,
    NEUTRON_PHASE_SPACE_F,
)

DATA_FILE = "data/TCS_CC_anue_p_1026_SV.txt"


# ---------------------------------------------------------------------------
# Analytic (Vogel & Beacom 1999)
# ---------------------------------------------------------------------------
def _sigma0_cm2() -> float:
    """Overall normalisation sigma_0 in cm^2 / MeV^2.

    sigma_0 = 2 pi^2 / (m_e^5 f tau_n)  in natural units, converted to cm^2.
    Equivalently sigma_0 (1 + 3 g^2) = 0.0952e-42 cm^2 / MeV^2.
    """

    # hbar^2 c^2 = 3.8937930e-22 MeV^2 cm^2 ; hbar = 6.582119569e-22 MeV s
    hbar_c2 = 3.8937930e-22  # MeV^2 cm^2
    hbar = 6.582119569e-22  # MeV s
    return (
        2.0
        * np.pi**2
        * hbar_c2
        * hbar
        / (M_E**5 * NEUTRON_PHASE_SPACE_F * NEUTRON_LIFETIME_S * F_PLUS_G_SQUARED)
    )


SIGMA0_CM2_PER_MEV2 = _sigma0_cm2()


def vogel_beacom(
    e_nu_mev: np.ndarray | float,
    order: int = 1,
) -> np.ndarray:
    """IBD cross section in cm^2.

    ``order=0`` gives the naive (zeroth-order) expression
    sigma = sigma_0 (1 + 3 g^2) E_e p_e, and ``order=1`` adds the
    O(1/M) recoil and weak-magnetism correction of Vogel & Beacom Eq. (13).
    """

    e = np.asarray(e_nu_mev, dtype=float)
    prefactor = SIGMA0_CM2_PER_MEV2 * F_PLUS_G_SQUARED

    # Zeroth order kinematics
    e_e0 = e - DELTA_NP
    p_e0 = np.sqrt(np.maximum(e_e0**2 - M_E**2, 0.0))
    sigma = prefactor * e_e0 * p_e0

    if order >= 1:
        # First-order correction, Vogel & Beacom Eq. (11)-(13).
        # <E_e^(1)> = E_e^(0) [1 - E_nu/M (1 - v cos_theta_avg)] - y^2/M
        y2 = (DELTA_NP**2 - M_E**2) / 2.0
        v0 = np.divide(p_e0, e_e0, out=np.zeros_like(e_e0), where=e_e0 > 0)
        # Angular average of cos(theta) weighted by the differential rate
        cos_avg = -0.034 * v0 + 2.4e-3 * e / M_P
        e_e1 = e_e0 * (1.0 - e / M_P * (1.0 - v0 * cos_avg)) - y2 / M_P
        p_e1 = np.sqrt(np.maximum(e_e1**2 - M_E**2, 0.0))
        sigma = prefactor * e_e1 * p_e1 * (1.0 - 7.22e-3 * np.log(np.maximum(e, 1e-6) / M_E))

    return np.where(e > IBD_THRESHOLD, sigma, 0.0)


def positron_energy(e_nu_mev: np.ndarray | float, order: int = 1) -> np.ndarray:
    """Mean positron total energy for a given neutrino energy, MeV."""

    e = np.asarray(e_nu_mev, dtype=float)
    e_e0 = e - DELTA_NP
    if order == 0:
        return e_e0
    y2 = (DELTA_NP**2 - M_E**2) / 2.0
    p_e0 = np.sqrt(np.maximum(e_e0**2 - M_E**2, 0.0))
    v0 = np.divide(p_e0, e_e0, out=np.zeros_like(e_e0), where=e_e0 > 0)
    cos_avg = -0.034 * v0 + 2.4e-3 * e / M_P
    return e_e0 * (1.0 - e / M_P * (1.0 - v0 * cos_avg)) - y2 / M_P


# ---------------------------------------------------------------------------
# Tabulated
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IBDCrossSection:
    """Interpolated IBD cross section, cm^2, as a function of E_nu in MeV."""

    energy_mev: np.ndarray
    sigma_cm2: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path, table_scale_cm2: float = 1.0e-38) -> "IBDCrossSection":
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Expected a two-column table, got shape {data.shape}.")
        return cls(energy_mev=data[:, 0], sigma_cm2=data[:, 1] * table_scale_cm2)

    @classmethod
    def from_package_data(cls, table_scale_cm2: float = 1.0e-38) -> "IBDCrossSection":
        with resources.as_file(resources.files("reactor").joinpath(DATA_FILE)) as path:
            return cls.from_file(path, table_scale_cm2=table_scale_cm2)

    def __call__(self, e_nu_mev: np.ndarray | float) -> np.ndarray:
        e = np.asarray(e_nu_mev, dtype=float)
        sigma = np.interp(e, self.energy_mev, self.sigma_cm2, left=0.0, right=self.sigma_cm2[-1])
        return np.where(e > IBD_THRESHOLD, sigma, 0.0)


@lru_cache(maxsize=4)
def load_ibd_cross_section(table_scale_cm2: float = 1.0e-38) -> IBDCrossSection:
    """Load (and cache) the packaged IBD table."""

    return IBDCrossSection.from_package_data(table_scale_cm2=table_scale_cm2)


class AnalyticIBD:
    """Callable wrapper around :func:`vogel_beacom` with the same interface."""

    def __init__(self, order: int = 1) -> None:
        self.order = order

    def __call__(self, e_nu_mev: np.ndarray | float) -> np.ndarray:
        return vogel_beacom(e_nu_mev, order=self.order)
