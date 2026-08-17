"""Neutron (and gamma) shielding estimate for the parked reactor at JUNO.

The question: how much shielding does a 10 MW core at ~50 m need so that its
neutrons (and direct gammas) do not spoil JUNO's single-hit physics?

Source term
-----------
Fission rate R_f = P / E_fission with nu_bar = 2.43 neutrons/fission (U235),
emitted with the Watt spectrum chi(E) ~ exp(-E/a) sinh(sqrt(bE)),
a = 0.988 MeV, b = 2.249 /MeV (<E> ~ 2.0 MeV).  A fraction ``leakage``
(default 25%, deliberately high for a compact high-leakage HALEU core)
escapes the core/reflector and must be stopped by the deployable package.

Transmission
------------
Classic *removal cross-section* theory: for a shield followed by hydrogenous
material (always true here -- the final layer is JUNO's own water pool), fast
flux attenuates as exp(-sum_i Sigma_R,i d_i).  Fission-spectrum-averaged
removal cross sections per element (Blizard/Lamarsh):
sigma_R ~ H 1.05 b, C 0.81 b, O 0.92 b, Fe 1.98 b, Pb 3.53 b.  The
deep-penetration spectrum hardens; for hydrogen-dominated materials we scale
Sigma_R(E) with the n-p total cross section, normalised at 2 MeV, and
integrate over the Watt spectrum.  Scattering build-up and duct streaming are
NOT modelled; a safety factor (default x5) stands in for build-up, and the
notebook states loudly that an engineering design needs Monte Carlo.

Geometry
--------
Point source at L from the detector centre; the fraction pi R_LS^2 / (4 pi
L^2) heads toward the liquid scintillator; every line to the LS crosses at
least (R_pool - R_LS) ~ 4 m of pool water.  Every neutron entering the LS is
counted as >= 1 single-hit event (its 2.2 MeV np-capture gamma is inside every
analysis window) -- conservative.

Direct core gammas are treated with the same layered-slab machinery using
mass-attenuation anchors (NIST) for water/steel/lead and the FRJ-1 core
spectrum of :mod:`reactor.alps` -- they are the harder constraint, and the
reason the package needs a high-Z layer, not just hydrogen.
"""

from __future__ import annotations

import numpy as np

from .alps import reactor_photon_flux

NU_BAR = 2.43
E_PER_FISSION_J = 200.0 * 1.602e-13
WATT_A, WATT_B = 0.988, 2.249      # MeV, 1/MeV

#: number densities [atoms/cm^3] per material
_COMPOSITIONS = {
    "water":    {"H": 6.69e22, "O": 3.34e22},
    "poly":     {"H": 8.16e22, "C": 4.08e22},
    "steel":    {"Fe": 8.49e22},
    "lead":     {"Pb": 3.30e22},
    "concrete": {"H": 1.37e22, "O": 4.56e22, "Si": 1.58e22, "Ca": 0.11e23},
}

#: fission-spectrum-averaged removal cross sections [barn] (Blizard/Lamarsh)
_SIGMA_R = {"H": 1.05, "C": 0.81, "O": 0.92, "Si": 1.23, "Ca": 1.35,
            "Fe": 1.98, "Pb": 3.53}

#: n-p total cross section [barn] vs E [MeV] -- sets the spectral hardening
#: of hydrogen-dominated shields
_SIG_NP = np.array([[0.5, 6.1], [1.0, 4.25], [2.0, 2.9], [3.0, 2.25],
                    [4.0, 1.85], [5.0, 1.6], [6.0, 1.4], [8.0, 1.15],
                    [10.0, 0.94], [14.0, 0.69]])

#: gamma mass attenuation mu/rho [cm^2/g] vs E [MeV] (NIST anchors), and rho
_GAMMA_MU = {
    "water": (1.00, np.array([[0.5, 0.0966], [1.0, 0.0707], [2.0, 0.0494],
                              [3.0, 0.0396], [4.0, 0.0340], [5.0, 0.0303],
                              [6.0, 0.0277], [8.0, 0.0243], [10.0, 0.0222]])),
    "steel": (7.87, np.array([[0.5, 0.0840], [1.0, 0.0600], [2.0, 0.0425],
                              [3.0, 0.0362], [4.0, 0.0331], [6.0, 0.0304],
                              [8.0, 0.0295], [10.0, 0.0294]])),
    "lead":  (11.35, np.array([[0.5, 0.1613], [1.0, 0.0710], [2.0, 0.0457],
                               [3.0, 0.0421], [4.0, 0.0420], [6.0, 0.0436],
                               [8.0, 0.0459], [10.0, 0.0489]])),
    "concrete": (2.30, np.array([[0.5, 0.0870], [1.0, 0.0637], [2.0, 0.0445],
                                 [3.0, 0.0363], [4.0, 0.0317], [6.0, 0.0273],
                                 [8.0, 0.0250], [10.0, 0.0237]])),
    "poly": (0.95, np.array([[0.5, 0.1063], [1.0, 0.0777], [2.0, 0.0541],
                             [3.0, 0.0431], [4.0, 0.0367], [6.0, 0.0294],
                             [8.0, 0.0254], [10.0, 0.0229]])),
}


def watt_spectrum(e_mev):
    """Normalised U235 thermal-fission Watt spectrum [1/MeV]."""

    e = np.asarray(e_mev, dtype=float)
    chi = np.exp(-e / WATT_A) * np.sinh(np.sqrt(WATT_B * e))
    grid = np.linspace(1e-3, 20.0, 4000)
    norm = np.trapezoid(np.exp(-grid / WATT_A)
                        * np.sinh(np.sqrt(WATT_B * grid)), grid)
    return chi / norm


def sigma_removal(material: str, e_mev) -> np.ndarray:
    """Macroscopic removal cross section [1/cm] with spectral hardening.

    Fission-average per element, scaled with the n-p cross section
    (normalised at 2 MeV) for the hydrogen-bearing part -- the high-energy
    tail penetrates deeper, and this captures that at estimate level.
    """

    e = np.atleast_1d(np.asarray(e_mev, dtype=float))
    scale = np.interp(e, _SIG_NP[:, 0], _SIG_NP[:, 1]) / 2.9
    sig = np.zeros_like(e)
    for elem, n in _COMPOSITIONS[material].items():
        s = n * _SIGMA_R[elem] * 1e-24
        sig += s * (scale if elem == "H" else
                    np.clip(scale, 0.7, 1.3))     # heavy elements flatter
    return sig


def gamma_mu(material: str, e_mev) -> np.ndarray:
    """Gamma attenuation coefficient [1/cm], log-log NIST interpolation."""

    rho, tab = _GAMMA_MU[material]
    e = np.clip(np.atleast_1d(np.asarray(e_mev, dtype=float)),
                tab[0, 0], tab[-1, 0])
    return rho * np.exp(np.interp(np.log(e), np.log(tab[:, 0]),
                                  np.log(tab[:, 1])))


def transmission(e_mev, layers, kind: str = "neutron") -> np.ndarray:
    """exp(-sum_i Sigma_i(E) d_i) through [(material, thickness_cm), ...]."""

    e = np.atleast_1d(np.asarray(e_mev, dtype=float))
    tau = np.zeros_like(e)
    fn = sigma_removal if kind == "neutron" else gamma_mu
    for mat, d_cm in layers:
        tau += fn(mat, e) * d_cm
    return np.exp(-np.clip(tau, 0.0, 200.0))


class ShieldingEstimate:
    """Neutron and gamma rates into the JUNO LS from the parked core."""

    def __init__(
        self,
        power_mw: float = 10.0,
        baseline_m: float = 50.0,
        leakage: float = 0.25,
        ls_radius_m: float = 17.7,
        water_path_m: float = 4.05,       # pool wall to acrylic, side-on
        buildup_factor: float = 5.0,      # stands in for scattering build-up
        e_grid: np.ndarray | None = None,
    ):
        self.power_mw = power_mw
        self.baseline_m = baseline_m
        self.fission_rate = power_mw * 1e6 / E_PER_FISSION_J
        self.neutron_source = self.fission_rate * NU_BAR * leakage
        self.geom = (np.pi * ls_radius_m**2) / (4.0 * np.pi * baseline_m**2)
        self.water_path_cm = water_path_m * 100.0
        self.buildup = buildup_factor
        self.e = np.linspace(0.1, 14.0, 600) if e_grid is None else e_grid
        self._watt = watt_spectrum(self.e)

    # -- neutrons ------------------------------------------------------------
    def neutron_rate_per_day(self, package_layers=()) -> float:
        """Neutrons/day entering the LS (each >= 1 single-hit event)."""

        layers = list(package_layers) + [("water", self.water_path_cm)]
        t = transmission(self.e, layers, "neutron")
        frac = np.trapezoid(self._watt * t, self.e)
        return (self.neutron_source * self.geom * frac
                * self.buildup * 86400.0)

    def required_water_equivalent_cm(self, target_per_day: float,
                                     material: str = "water") -> float:
        """Package thickness of ``material`` needed to reach the target."""

        lo, hi = 0.0, 1000.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.neutron_rate_per_day([(material, mid)]) > target_per_day:
                lo = mid
            else:
                hi = mid
        return hi

    # -- direct core gammas --------------------------------------------------
    def gamma_rate_per_day(self, package_layers=(),
                           e_min_mev: float = 1.0) -> float:
        """Direct core gammas/day above threshold entering the LS."""

        e = self.e[self.e >= e_min_mev]
        phi = reactor_photon_flux(e, self.power_mw)      # /MeV/s, whole core
        layers = list(package_layers) + [("water", self.water_path_cm)]
        t = transmission(e, layers, "gamma")
        return (np.trapezoid(phi * t, e) * self.geom
                * self.buildup * 86400.0)

    def required_gamma_layer_cm(self, target_per_day: float,
                                material: str = "lead",
                                extra_layers=()) -> float:
        lo, hi = 0.0, 500.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            layers = list(extra_layers) + [(material, mid)]
            if self.gamma_rate_per_day(layers) > target_per_day:
                lo = mid
            else:
                hi = mid
        return hi
