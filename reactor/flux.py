"""Reactor antineutrino flux, fission fractions, and reactor-core definitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from .constants import (
    CM_PER_KM,
    FISSION_ENERGIES_MEV,
    ISOTOPES,
    MEV_TO_J,
)

# ---------------------------------------------------------------------------
# Huber (U235, Pu239, Pu241) + Mueller (U238) polynomial parameterisations.
# dN/dE = exp( sum_k a_k E^(k-1) ),  antineutrinos / fission / MeV, E in MeV.
# Huber 2011 (arXiv:1106.0687) Table 3; Mueller et al. 2011 (arXiv:1101.2663).
# ---------------------------------------------------------------------------
HM = {
    "U235": np.array([4.367, -4.577, 2.100, -5.294e-1, 6.186e-2, -2.777e-3]),
    "U238": np.array([4.833e-1, 1.927e-1, -1.283e-1, -6.762e-3, 2.233e-3, -1.536e-4]),
    "Pu239": np.array([4.757, -5.392, 2.563, -6.596e-1, 7.820e-2, -3.536e-3]),
    "Pu241": np.array([2.990, -2.882, 1.278, -3.343e-1, 3.905e-2, -1.754e-3]),
}

# Validity range of the polynomial fits.  Outside it the exponential blows up
# or undershoots, so we clamp.
HM_EMIN, HM_EMAX = 1.8, 8.5

# Off-equilibrium correction (Mueller et al. Table VII, 450-day irradiation),
# multiplicative factor 1 + delta(E)/100, tabulated at 2, 3, 4, 5, 6, 7, 8 MeV.
OFF_EQUILIBRIUM_PERCENT = {
    "U235": np.array([4.4, 2.0, 1.1, 0.6, 0.0, 0.0, 0.0]),
    "U238": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "Pu239": np.array([1.3, 0.7, 0.4, 0.2, 0.0, 0.0, 0.0]),
    "Pu241": np.array([1.6, 0.9, 0.6, 0.4, 0.0, 0.0, 0.0]),
}
OFF_EQUILIBRIUM_ENERGIES = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

FractionModel = Callable[[float], Mapping[str, float]]


# ---------------------------------------------------------------------------
# Fission fractions
# ---------------------------------------------------------------------------
def normalize_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    """Return isotope fractions ordered by the four Huber-Mueller parents."""

    values = {iso: float(fractions.get(iso, 0.0)) for iso in ISOTOPES}
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("Fission fractions must sum to a positive value.")
    return {iso: val / total for iso, val in values.items()}


def constant_fractions(fractions: Mapping[str, float]) -> FractionModel:
    """Build a burnup-independent fission-fraction model."""

    normalized = normalize_fractions(fractions)
    return lambda burnup: normalized


def juno_average_fractions(_: float = 0.0) -> dict[str, float]:
    """Cycle-averaged PWR fission fractions used throughout the JUNO studies."""

    return {"U235": 0.58, "U238": 0.07, "Pu239": 0.30, "Pu241": 0.05}


def dayabay_average_fractions(_: float = 0.0) -> dict[str, float]:
    """Daya Bay cycle-averaged fission fractions (arXiv:1607.05378)."""

    return {"U235": 0.561, "U238": 0.076, "Pu239": 0.307, "Pu241": 0.056}


def pwr_cycle_fractions(burnup: float) -> dict[str, float]:
    """Simple LEU PWR burnup model centred on the JUNO cycle average."""

    b = float(np.clip(burnup, 0.0, 1.0))
    fractions = {
        "U238": 0.07,
        "Pu239": 0.25 + 0.10 * b,
        "Pu241": 0.04 + 0.02 * b,
    }
    fractions["U235"] = 1.0 - sum(fractions.values())
    return normalize_fractions(fractions)


def haleu_fractions(burnup: float = 0.0, evolve: bool = True) -> dict[str, float]:
    """Compact HALEU microreactor core: 98% U235, 2% U238 when fresh.

    With ``evolve=True`` (the default) the U235 fission fraction is converted
    into Pu239/Pu241 following the *Daya Bay* fuel-evolution trajectory: the
    ingrowth of the plutonium fission fractions across their measured 20
    burnup groups, applied on top of the fresh HALEU composition, with the
    U238 fraction held constant at 0.02.  ``burnup`` is the position in that
    cycle, 0 to 1.

    This deliberately follows the Daya Bay (LEU) conversion rate, as a
    conservative model: a core with only 2% U238 breeds plutonium more slowly,
    so the true spectral drift should be smaller than modelled here.
    """

    b = float(np.clip(burnup, 0.0, 1.0))
    f238 = 0.02
    if not evolve or b == 0.0:
        return {"U235": 1.0 - f238, "U238": f238, "Pu239": 0.0, "Pu241": 0.0}

    from .dayabay_data import load_flux_evolution

    ev = load_flux_evolution()
    beta = np.linspace(0.0, 1.0, len(ev["group"]))
    f239 = float(np.interp(b, beta, ev["f239"] - ev["f239"][0]))
    f241 = float(np.interp(b, beta, ev["f241"] - ev["f241"][0]))
    f235 = 1.0 - f238 - f239 - f241
    return normalize_fractions({"U235": f235, "U238": f238, "Pu239": f239, "Pu241": f241})


# ---------------------------------------------------------------------------
# Per-fission antineutrino spectra
# ---------------------------------------------------------------------------
def spectrum_per_fission(
    e_nu_mev: np.ndarray | float,
    isotope: str,
    *,
    clamp: bool = True,
) -> np.ndarray:
    """Huber-Mueller antineutrino spectrum, nu / fission / MeV."""

    if isotope not in HM:
        raise KeyError(f"Unknown isotope {isotope!r}. Expected one of {ISOTOPES}.")
    e = np.atleast_1d(np.asarray(e_nu_mev, dtype=float))
    e_eval = np.clip(e, HM_EMIN, HM_EMAX) if clamp else e

    coeffs = HM[isotope]
    poly = np.zeros_like(e_eval)
    power = np.ones_like(e_eval)
    for coeff in coeffs:
        poly += coeff * power
        power *= e_eval
    out = np.exp(poly)

    if clamp:
        # Kill the unphysical tail above the fit range rather than freezing it.
        out = np.where(e > HM_EMAX, out * np.exp(-(e - HM_EMAX) / 0.2), out)
        out = np.where(e < HM_EMIN, 0.0, out)
    return out if np.ndim(e_nu_mev) else out


def off_equilibrium_correction(e_nu_mev: np.ndarray, isotope: str) -> np.ndarray:
    """Multiplicative long-lived-fission-product correction (Mueller Table VII)."""

    e = np.asarray(e_nu_mev, dtype=float)
    percent = np.interp(
        e,
        OFF_EQUILIBRIUM_ENERGIES,
        OFF_EQUILIBRIUM_PERCENT[isotope],
        left=OFF_EQUILIBRIUM_PERCENT[isotope][0],
        right=0.0,
    )
    return 1.0 + percent / 100.0


def mixed_spectrum_per_fission(
    e_nu_mev: np.ndarray | float,
    fractions: Mapping[str, float],
    *,
    off_equilibrium: bool = False,
) -> np.ndarray:
    """Fission-fraction-weighted antineutrino spectrum, nu / fission / MeV."""

    fractions = normalize_fractions(fractions)
    e = np.asarray(e_nu_mev, dtype=float)
    total = np.zeros_like(e, dtype=float)
    for isotope, fraction in fractions.items():
        if fraction == 0.0:
            continue
        contrib = fraction * spectrum_per_fission(e, isotope)
        if off_equilibrium:
            contrib = contrib * off_equilibrium_correction(e, isotope)
        total = total + contrib
    return total


def average_energy_per_fission(fractions: Mapping[str, float]) -> float:
    """Mean thermal energy release per fission, MeV."""

    fractions = normalize_fractions(fractions)
    return sum(fractions[iso] * FISSION_ENERGIES_MEV[iso] for iso in ISOTOPES)


def fission_rate_per_second(power_gwth: float, fractions: Mapping[str, float]) -> float:
    """Fissions per second for a thermal power (GW) and isotope mix."""

    mean_energy_j = average_energy_per_fission(fractions) * MEV_TO_J
    return power_gwth * 1.0e9 / mean_energy_j


# ---------------------------------------------------------------------------
# Reactor cores
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReactorCore:
    """A single reactor core: thermal power, baseline, and fuel model."""

    name: str
    power_gwth: float
    baseline_km: float
    fraction_model: FractionModel = juno_average_fractions
    duty_cycle: float = 1.0
    off_equilibrium: bool = False

    def fractions(self, burnup: float = 0.0) -> dict[str, float]:
        return normalize_fractions(self.fraction_model(burnup))

    def fission_rate(self, burnup: float = 0.0) -> float:
        """Fissions per second, including the duty cycle."""

        return self.duty_cycle * fission_rate_per_second(self.power_gwth, self.fractions(burnup))

    def spectrum_per_fission(self, e_nu_mev: np.ndarray, burnup: float = 0.0) -> np.ndarray:
        return mixed_spectrum_per_fission(
            e_nu_mev, self.fractions(burnup), off_equilibrium=self.off_equilibrium
        )

    def flux_at_detector(self, e_nu_mev: np.ndarray, burnup: float = 0.0) -> np.ndarray:
        """Unoscillated flux at the detector, nu / cm^2 / s / MeV."""

        distance_cm = self.baseline_km * CM_PER_KM
        geometric = 4.0 * np.pi * distance_cm**2
        return self.fission_rate(burnup) * self.spectrum_per_fission(e_nu_mev, burnup) / geometric

    def at_baseline(self, baseline_km: float) -> "ReactorCore":
        return ReactorCore(
            name=self.name,
            power_gwth=self.power_gwth,
            baseline_km=float(baseline_km),
            fraction_model=self.fraction_model,
            duty_cycle=self.duty_cycle,
            off_equilibrium=self.off_equilibrium,
        )


# JUNO reference reactor complex: Yangjiang (6 x 2.9 GW) + Taishan (2 x 4.6 GW),
# baselines from the JUNO Yellow Book (arXiv:2104.02565) Table 2-1.
JUNO_CORE_TABLE = (
    ("YJ-C1", 2.9, 52.74),
    ("YJ-C2", 2.9, 52.82),
    ("YJ-C3", 2.9, 52.41),
    ("YJ-C4", 2.9, 52.49),
    ("YJ-C5", 2.9, 52.11),
    ("YJ-C6", 2.9, 52.19),
    ("TS-C1", 4.6, 52.77),
    ("TS-C2", 4.6, 52.64),
)

#: Distant complexes that are *not* at the oscillation maximum but still carry a
#: few percent of the signal.  They matter far more than their rate suggests.
#:
#: Averaged over the neutrino energies that feed the solar dip (2.9-3.5 MeV),
#: the solar phase Delta_21 and the resulting survival probability are
#:
#:     Yangjiang/Taishan   52.5 km   0.50 pi   <P_ee> = 0.145   (solar minimum)
#:     Daya Bay           215.0 km   2.05 pi   <P_ee> = 0.849   (solar maximum)
#:     Taipingling        265.0 km   2.52 pi   <P_ee> = 0.280
#:     Fangchenggang      411.7 km   3.92 pi   <P_ee> = 0.653
#:
#: Daya Bay lands almost exactly on a solar maximum for those energies, so its
#: flux arrives nearly unoscillated and *fills in* the dip that theta12 is
#: measured from.  Dropping these cores biases sin^2(theta12) low by about one
#: standard deviation and costs ~7 units of chi^2; see notebook 3, section 9.
#:
#: Only Daya Bay is carried as *signal*, which is JUNO's own definition: their
#: signal sum runs over the nine reactors of Tab. 2 of JUNO:2022mxj -- "eight
#: reactors at a distance of about 53 km and a single effective reactor from the
#: Daya Bay complex at 215 km" (quoted in NuFit, arXiv:2601.09791v2, Sec. 2).
#: Everything further away is in the release's ``world reactors`` background,
#: which we take as given.
#:
#: The other two entries are at zero power for different reasons:
#:
#: * Taipingling (Huizhou) was still starting up during the 2025 dataset.
#:   Restore its design 17.4 GW for future projections.
#: * Fangchenggang belongs to the world-reactors background under JUNO's
#:   definition.  NuFit's v2 note added moves it into the signal instead.  Doing
#:   that *consistently* -- adding the core and removing its 25.7 predicted
#:   events from the 51.9-event background, with the normalisation anchor raised
#:   to match -- changes sin^2(theta12) by less than 0.001 and the 2D pull
#:   against the official map by 0.04, i.e. it is a null effect either way
#:   (notebook 3, section 9.4).  Adding it to the signal while leaving it in the
#:   background double-counts it and shifts sin^2(theta12) by +0.002.
JUNO_DISTANT_CORE_TABLE = (
    ("DYB", 17.4, 215.0),      # Daya Bay + Ling Ao complex
    ("TPL", 0.0, 265.0),       # Taipingling / Huizhou, not yet contributing in 2025
    ("FCG", 0.0, 411.7),       # Fangchenggang; already in the world-reactors background
)

# Nominal load factor: reactors are off for refuelling roughly 1 month / year.
JUNO_DUTY_CYCLE = 11.0 / 12.0


def default_juno_cores(
    use_cycle: bool = False,
    duty_cycle: float = JUNO_DUTY_CYCLE,
    off_equilibrium: bool = False,
    include_distant: bool = True,
) -> list[ReactorCore]:
    """The reactor cores seen by JUNO.

    The eight Yangjiang + Taishan cores at the oscillation maximum, plus -- when
    ``include_distant`` -- the Daya Bay complex at 215 km, whose contribution is
    small in rate but not in effect: see :data:`JUNO_DISTANT_CORE_TABLE`.
    Cores with zero power are dropped.
    """

    fraction_model = pwr_cycle_fractions if use_cycle else juno_average_fractions
    table = list(JUNO_CORE_TABLE)
    if include_distant:
        table += [row for row in JUNO_DISTANT_CORE_TABLE if row[1] > 0.0]
    return [
        ReactorCore(name, power, baseline, fraction_model, duty_cycle, off_equilibrium)
        for name, power, baseline in table
    ]


def equivalent_single_core(cores: list[ReactorCore]) -> ReactorCore:
    """Single core with the same total power and the flux-weighted baseline.

    The effective baseline is defined by matching sum_i P_i / L_i^2.
    """

    total_power = sum(core.power_gwth for core in cores)
    weight = sum(core.power_gwth / core.baseline_km**2 for core in cores)
    l_eff = np.sqrt(total_power / weight)
    ref = cores[0]
    return ReactorCore(
        name="JUNO-equivalent",
        power_gwth=total_power,
        baseline_km=float(l_eff),
        fraction_model=ref.fraction_model,
        duty_cycle=ref.duty_cycle,
        off_equilibrium=ref.off_equilibrium,
    )


def movable_reactor(
    baseline_km: float,
    power_mwth: float = 100.0,
    evolve: bool = False,
    duty_cycle: float = 1.0,
    off_equilibrium: bool = False,
) -> ReactorCore:
    """The movable compact near source."""

    return ReactorCore(
        name=f"MR-{baseline_km:.4g}km",
        power_gwth=power_mwth / 1000.0,
        baseline_km=float(baseline_km),
        fraction_model=lambda burnup: haleu_fractions(burnup, evolve=evolve),
        duty_cycle=duty_cycle,
        off_equilibrium=off_equilibrium,
    )


def taishan_tao_core() -> ReactorCore:
    """The single Taishan core watched by JUNO-TAO at 44 m."""

    return ReactorCore("TS-C1(TAO)", 4.6, 0.044, juno_average_fractions, JUNO_DUTY_CYCLE)
