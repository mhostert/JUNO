"""Physical constants, detector parameters, and reference oscillation inputs.

All energies are in MeV, baselines in km, mass-squared splittings in eV^2,
unless the symbol name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
CM_PER_KM = 1.0e5
SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.25
MEV_TO_J = 1.602176634e-13
AVOGADRO = 6.02214076e23

# Oscillation phase convention: Delta = 1.267 * dm2[eV^2] * L[km] / E[GeV]
#   = (dm2 L)/(4E) in natural units.  The exact coefficient is
#   1e9 / (4 * hbar*c[eV*m]) with hbar*c = 1.973269804e-7 eV m.
OSC_COEFF = 1.0e9 / (4.0 * 1.973269804e-7) * 1.0e-9  # -> 1.2669 for eV^2 km / GeV
KM_EV2_PER_GEV = 1.266932679815

# ---------------------------------------------------------------------------
# Particle masses / IBD kinematics
# ---------------------------------------------------------------------------
M_E = 0.51099895  # MeV
M_P = 938.27208816  # MeV
M_N = 939.56542052  # MeV
DELTA_NP = M_N - M_P  # 1.29333 MeV
IBD_THRESHOLD = ((M_N + M_E) ** 2 - M_P**2) / (2.0 * M_P)  # 1.806 MeV

# E_prompt = E_e+ + m_e = (E_nu - Delta) + m_e  (positron KE + 2 annihilation gammas)
PROMPT_ENERGY_OFFSET_MEV = DELTA_NP - M_E  # 0.78233 MeV

# Weak interaction constants for the IBD cross section (Vogel & Beacom 1999)
NEUTRON_LIFETIME_S = 878.4  # s (PDG 2024)
F_PLUS_G_SQUARED = 1.0 + 3.0 * 1.2756**2  # f^2 + 3 g^2 with g_A = 1.2756
# Phase-space factor for free-neutron decay (Vogel & Beacom Eq. 9)
NEUTRON_PHASE_SPACE_F = 1.7152

# ---------------------------------------------------------------------------
# Reactor fission physics
# ---------------------------------------------------------------------------
ISOTOPES = ("U235", "U238", "Pu239", "Pu241")

# Thermal energy release per fission, Ma et al. (2013) / Kopeikin, in MeV.
FISSION_ENERGIES_MEV = {
    "U235": 202.36,
    "U238": 205.99,
    "Pu239": 211.12,
    "Pu241": 214.26,
}
FISSION_ENERGY_ERRORS_MEV = {
    "U235": 0.26,
    "U238": 0.52,
    "Pu239": 0.34,
    "Pu241": 0.33,
}

# Reference IBD yields per fission (Huber-Mueller), 1e-43 cm^2 / fission.
# Used only as an external cross-check in the validation notebook.
HM_IBD_YIELD_REF_1E43 = {
    "U235": 6.69,
    "U238": 10.10,
    "Pu239": 4.40,
    "Pu241": 6.03,
}

# ---------------------------------------------------------------------------
# JUNO detector
# ---------------------------------------------------------------------------
JUNO_MASS_KT = 20.0
# 20 kt of LAB-based LS with a hydrogen mass fraction of 12.0%:
#   N_p = 20e9 g * 0.120 * N_A / 1.00794 g/mol
JUNO_HYDROGEN_MASS_FRACTION = 0.1201
JUNO_TARGET_PROTONS = JUNO_MASS_KT * 1.0e9 * JUNO_HYDROGEN_MASS_FRACTION * AVOGADRO / 1.00794

# IBD selection efficiency quoted in the JUNO Yellow Book (arXiv:2104.02565).
JUNO_IBD_EFFICIENCY = 0.822

# Energy resolution, sigma_E/E = sqrt( (a/sqrt(E))^2 + b^2 + (c/E)^2 ), E in MeV.
# JUNO Yellow Book / first-data values.  a is the photostatistics term.
JUNO_RESOLUTION_ABC = (0.0261, 0.0082, 0.0123)
# The simplified model used in the movable-reactor draft: sigma_E = 0.03 sqrt(E).
DRAFT_RESOLUTION_A = 0.03

# Effective matter density along the JUNO baseline (upper continental crust).
EARTH_CRUST_DENSITY_G_CM3 = 2.6
ELECTRON_FRACTION_YE = 0.5

# ---------------------------------------------------------------------------
# Oscillation parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OscillationParameters:
    """Oscillation parameters for reactor antineutrino disappearance.

    The atmospheric splitting is carried as ``dm2_ee`` (the effective
    "reactor" splitting).  ``ordering`` is +1 for normal and -1 for inverted;
    it only matters for the exact three-flavour / matter treatment.
    """

    sin2_theta13: float = 0.02215
    dm2_ee: float = 2.494e-3
    sin2_theta12: float = 0.307
    dm2_21: float = 7.41e-5
    ordering: int = 1

    # --- derived quantities -------------------------------------------------
    @property
    def sin2_2theta13(self) -> float:
        return 4.0 * self.sin2_theta13 * (1.0 - self.sin2_theta13)

    @property
    def sin2_2theta12(self) -> float:
        return 4.0 * self.sin2_theta12 * (1.0 - self.sin2_theta12)

    @property
    def theta13_deg(self) -> float:
        return float(np.degrees(np.arcsin(np.sqrt(self.sin2_theta13))))

    @property
    def theta12_deg(self) -> float:
        return float(np.degrees(np.arcsin(np.sqrt(self.sin2_theta12))))

    @property
    def dm2_31(self) -> float:
        """Signed Delta m^2_31 implied by dm2_ee.

        dm2_ee = cos^2(th12) dm2_31 + sin^2(th12) dm2_32
               = dm2_31 - sin^2(th12) dm2_21     (identically, both orderings)
        """
        return self.ordering * abs(self.dm2_ee) + self.sin2_theta12 * self.dm2_21

    @property
    def dm2_32(self) -> float:
        return self.dm2_31 - self.dm2_21

    def replace(self, **kwargs) -> "OscillationParameters":
        return replace(self, **kwargs)


def dm2ee_from_dm231(dm2_31: float, sin2_theta12: float, dm2_21: float) -> float:
    """|Delta m^2_ee| from a signed Delta m^2_31."""

    return abs(dm2_31 - sin2_theta12 * dm2_21)


# NuFit 6.1 (2024) with SK atmospheric data, normal ordering.
NUFIT61_NO = OscillationParameters(
    sin2_theta13=0.02215,
    dm2_ee=2.494e-3,
    sin2_theta12=0.307,
    dm2_21=7.41e-5,
    ordering=1,
)
# 1 sigma symmetrised uncertainties quoted alongside the NuFit 6.1 central values.
NUFIT61_NO_ERRORS = {
    "sin2_theta13": 0.00060,
    "dm2_ee": 0.024e-3,
    "sin2_theta12": 0.012,
    "dm2_21": 0.21e-5,
}

NUFIT61_IO = OscillationParameters(
    sin2_theta13=0.02231,
    dm2_ee=2.465e-3,
    sin2_theta12=0.307,
    dm2_21=7.41e-5,
    ordering=-1,
)

# Daya Bay final 3158-day result (arXiv:2211.14988):
#   sin^2(2 theta13) = 0.0851 +- 0.0024,  dm2_ee = (2.466 +- 0.060) e-3 eV^2
DAYABAY_SIN2_2THETA13 = 0.0851
DAYABAY_SIN2_2THETA13_ERR = 0.0024
DAYABAY_DM2EE = 2.466e-3
DAYABAY_DM2EE_ERR = 0.060e-3


def sin2_theta_from_sin2_2theta(sin2_2theta: float) -> float:
    """Small-angle branch of sin^2(theta) given sin^2(2 theta)."""

    return 0.5 * (1.0 - np.sqrt(1.0 - sin2_2theta))


DAYABAY_SIN2_THETA13 = float(sin2_theta_from_sin2_2theta(DAYABAY_SIN2_2THETA13))
DAYABAY_SIN2_THETA13_ERR = float(
    DAYABAY_SIN2_2THETA13_ERR / (2.0 * np.sqrt(1.0 - DAYABAY_SIN2_2THETA13))
)

# ---------------------------------------------------------------------------
# JUNO first oscillation measurement, arXiv:2511.14593 (59.1 days)
# ---------------------------------------------------------------------------
JUNO2025_SIN2_THETA12 = 0.3092
JUNO2025_SIN2_THETA12_ERR = 0.0087
JUNO2025_DM2_21 = 7.50e-5
JUNO2025_DM2_21_ERR = 0.12e-5
JUNO2025_LIVETIME_DAYS = 59.1  # 69 calendar days, 26 Aug - 2 Nov 2025
JUNO2025_CANDIDATES = 2379

# Selection efficiencies, their Table 1.
JUNO2025_EFFICIENCIES = {
    "fiducial volume": 0.806,
    "PMT flasher rejection": 0.999,
    "muon veto": 0.936,
    "multiplicity": 0.974,
    "prompt-delayed coincidence": 0.951,
}
JUNO2025_EFFICIENCY_TOTAL = 0.699
JUNO2025_EFFICIENCY_TOTAL_ERR = 0.016  # relative

# Free target protons, from the LS volume, density and hydrogen fraction.
JUNO2025_TARGET_PROTONS = 1.442e33
JUNO2025_TARGET_PROTONS_ERR = 0.014e33

# Signal rates in counts per day, their Table 1.
JUNO2025_SIGNAL_CPD = 33.5  # observed, not efficiency corrected
JUNO2025_SIGNAL_CPD_ERR = 1.7
JUNO2025_SIGNAL_CPD_CORRECTED = 47.9  # efficiency corrected
JUNO2025_SIGNAL_CPD_CORRECTED_ERR = 2.6
JUNO2025_NONOSC_CPD = 150.9  # efficiency corrected, no oscillation
JUNO2025_NONOSC_CPD_ERR = 2.7

# Background rates in counts per day: (pre-fit, pre-fit error, best fit).
JUNO2025_BACKGROUNDS_CPD = {
    "9Li/8He": (4.3, 1.4, 3.9),
    "geoneutrino": (1.2, 0.5, 1.4),
    "world reactors": (0.88, 0.09, 0.88),
    "214Bi-214Po": (0.18, 0.10, 0.20),
    "13C(a,n)16O": (0.04, 0.02, 0.04),
    "fast neutrons": (0.02, 0.02, 0.02),
    "double neutrons": (0.05, 0.05, 0.07),
    "atmospheric neutrinos": (0.08, 0.04, 0.07),
    "accidentals": (0.049, 0.003, 0.049),
}

# Reactor-flux and detector rate uncertainties, their Table 2.
JUNO2025_RATE_SYSTEMATICS = {
    "target protons": 0.010,
    "reference spectrum": 0.012,
    "thermal power": 0.005,
    "fission fraction": 0.006,
    "spent nuclear fuel": 0.003,
    "non-equilibrium": 0.002,
    "different fission fraction": 0.001,
}

# Response systematics quoted in the text.
JUNO2025_ENERGY_SCALE_ERR = 0.005  # overall energy scale
JUNO2025_NONLINEARITY_ERR = 0.010  # positron non-linearity
JUNO2025_MATTER_DENSITY = 2.55  # g/cm^3, along the JUNO baseline
JUNO2025_MATTER_DENSITY_ERR = 0.25

# The prompt-energy relation quoted in the paper.
JUNO2025_PROMPT_OFFSET_MEV = 0.784

DEFAULT_OSCILLATION_PARAMS = NUFIT61_NO

# ---------------------------------------------------------------------------
# Confidence-level thresholds
# ---------------------------------------------------------------------------
DELTA_CHI2_1DOF = {1: 1.00, 2: 4.00, 3: 9.00}
DELTA_CHI2_2DOF = {1: 2.30, 2: 6.18, 3: 11.83}
