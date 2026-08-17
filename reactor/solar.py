"""Solar neutrino fluxes and survival probability (shared by several studies).

Continuum shapes from ``reactor/data/solar`` (Bahcall et al.), B16-GS98 SSM
normalisations, the pep line treated analytically, and the day-time adiabatic
LMA survival probability.
"""

from __future__ import annotations

import numpy as np

#: Solar fluxes at Earth, /cm^2/s (B16-GS98 SSM); shapes from reactor/data/solar.
SOLAR_FLUXES = {"b8": 5.46e6, "hep": 7.98e3, "n13": 2.78e8, "o15": 2.05e8,
                "f17": 5.29e6}
PEP_FLUX, PEP_E = 1.44e8, 1.442     # line, treated analytically

#: Release Tab. 1 background priors + our choices for the two additions.
IBD_BACKGROUND_PRIORS = {
    "far reactors": 0.02,       # nine cores + power data
    "geoneutrino": 0.42,
    "9Li/8He": 0.33,
    "world reactors": 0.10,
    "214Bi-214Po": 0.56,
    "other": 1.00,
}
SOLAR_PRIOR = 0.03              # B8-dominated in the window (SNO NC 2% + P_ee)


def _load_solar_shape(name: str) -> np.ndarray:
    from importlib import resources
    from pathlib import Path

    with resources.as_file(resources.files("reactor").joinpath(
            f"data/solar/fluxes/{name}_spectrum.txt")) as path:
        text = Path(path).read_text()
    rows = []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return np.asarray(rows)


def solar_flux_density(e_mev: np.ndarray) -> np.ndarray:
    """Summed solar nu_e flux density /cm^2/s/MeV (continua only; pep separate)."""

    out = np.zeros_like(e_mev)
    for name, flux in SOLAR_FLUXES.items():
        tab = _load_solar_shape(name)
        out += flux * np.interp(e_mev, tab[:, 0], tab[:, 1], left=0.0, right=0.0)
    return out


def solar_pee(e_mev: np.ndarray, params) -> np.ndarray:
    """Day-time adiabatic LMA survival probability at production <n_e> ~ 100 N_A."""

    n_e = 100.0 * 6.022e23                       # /cm^3
    v = 7.6324e-14 * n_e / 6.022e23 * 1.0       # sqrt(2) G_F n_e in eV per (n_e/N_A)
    v = 7.6324e-14 * (n_e / 6.022e23)           # eV
    beta = 2.0 * (e_mev * 1e6) * v * (1 - params.sin2_theta13) / params.dm2_21
    c2 = 1.0 - 2.0 * params.sin2_theta12
    c2m = (c2 - beta) / np.sqrt((c2 - beta) ** 2 + (1 - c2**2))
    c13_4 = (1 - params.sin2_theta13) ** 2
    return c13_4 * 0.5 * (1 + c2m * c2) + params.sin2_theta13 ** 2
