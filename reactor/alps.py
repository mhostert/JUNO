"""Axion-like particles from the near reactor, detected in JUNO.

The reactor core is an intense photon source (~5e18 gamma/s at 10 MW).  Photons
scattering in the fuel convert to ALPs through Primakoff (g_agamma) or
Compton-like (g_aee) processes; the ALPs stream through shielding and are
detected in JUNO via the inverse processes or via decay in flight,
a -> gamma gamma / e+ e-.  With a 16 kt fiducial target and a ~33 m decay
path, JUNO is a scattering *and* decay experiment at once -- the configuration
that targets the "cosmological triangle" of the ALP-photon parameter space at
MeV masses.

Implementation follows the latest calculation, Aristizabal Sierra et al.,
arXiv:2511.01812 (JUNO-TAO/CLOUD), with the underlying formulas from
arXiv:2010.15712 and the original proposal of Dent et al., arXiv:1912.05733:

* reactor photon flux: the FRJ-1 power law dPhi/dE = 5.8e17 (P/MW)
  exp(-1.1 E/MeV) /MeV/s  [2010.15712 Eq. (3.1)];
* Primakoff conversion in the fuel (U, Z=92) with the *exact* differential
  cross section of Aloni et al. [2010.15712 Eq. (3.8)], atomic screening form
  factor [Eqs. (3.9)-(3.10)], and the branching normalisation
  sigma_P/sigma_SM with sigma_SM from XCOM anchors;
* Compton-like production gamma e -> a e [Eq. (3.19)], convolved over the
  photon flux [2511.01812 Eq. (14)];
* detection: inverse Primakoff on C and H (x2 the production cross section),
  inverse Compton [2511.01812 Eq. (22)], and decays with survival/decay
  probabilities [Eqs. (3), (10)].  Axio-electric absorption is neglected:
  sigma_A ~ Z^5 and carbon at MeV energies gives ~1e-4 of the Ge rates of the
  reference analyses -- utterly subdominant to inverse Compton here.
* statistics: single-bin counting chi^2 = Ns^2/(Ns + Nb + (0.1 Ns)^2)
  [2511.01812 Eq. (54)], 90% CL at Delta chi^2 = 4.61 (2 dof).

Validation battery (notebook 7): the exact Primakoff cross section against an
independent implementation of the Dent et al. forward form; Compton against an
independent change-of-variables integration; decay widths and lab decay
lengths against analytic benchmarks including the corrected Eq. (53) decay
-length bound of 2511.01812; the SM photon cross section against its XCOM
anchors; flux totals.
"""

from __future__ import annotations

import numpy as np

ALPHA = 1.0 / 137.035999
ME = 0.51099895e-3            # GeV
HBARC_GEV_M = 1.9733e-16      # GeV.m
GEV2_TO_CM2 = 3.8938e-28      # (1/GeV^2) -> cm^2

#: core / target nuclei (masses in GeV)
M_U, Z_U = 221.7, 92
M_C, Z_C = 11.188, 6
M_H, Z_H = 0.9383, 1

# ---------------------------------------------------------------------------
# Reactor photon flux and SM photon cross section in uranium
# ---------------------------------------------------------------------------
def reactor_photon_flux(e_mev, power_mw: float):
    """FRJ-1 power law, photons / MeV / s from the whole core."""

    e = np.asarray(e_mev, dtype=float)
    return 5.8e17 * power_mw * np.exp(-1.1 * e)


#: XCOM total attenuation anchors for uranium, (E [MeV], mu/rho [cm^2/g]).
#: Total-with-coherent values; the MeV region (which dominates here) is flat
#: to ~10% and enters the limits only as g ~ sigma_SM^(1/4).
_XCOM_U = np.array([
    [0.10, 1.954], [0.15, 1.20], [0.20, 0.65], [0.30, 0.31],
    [0.50, 0.155], [0.80, 0.093], [1.00, 0.0776], [1.50, 0.0570],
    [2.00, 0.0484], [3.00, 0.0428], [4.00, 0.0416], [5.00, 0.0424],
    [6.00, 0.0437], [8.00, 0.0471], [10.0, 0.0508], [14.0, 0.0566],
])


def sigma_sm_uranium(e_mev):
    """Total SM photon cross section per U atom [cm^2], log-log interpolated."""

    e = np.clip(np.asarray(e_mev, dtype=float), 0.1, 14.0)
    mu = np.exp(np.interp(np.log(e), np.log(_XCOM_U[:, 0]),
                          np.log(_XCOM_U[:, 1])))
    return mu * 238.03 / 6.022e23


# ---------------------------------------------------------------------------
# Primakoff conversion, exact form (Aloni et al. / 2010.15712 Eq. 3.8)
# ---------------------------------------------------------------------------
def _screening_f2(t_abs, Z):
    """Atomic screening form factor, Moliere/Thomas-Fermi [Eqs. 3.9-3.10]."""

    a0 = 184.15 / np.sqrt(2.718) / (Z ** (1.0 / 3.0) * ME)   # 1/GeV
    x = a0**2 * t_abs
    return (x / (1.0 + x)) ** 2


def sigma_primakoff(e_gamma_mev, ma_mev: float, g_gevinv: float,
                    Z: int = Z_U, M: float = M_U, n_t: int = 160):
    """Total Primakoff production cross section per atom [cm^2].

    Exact 2->2 form, numerically integrated over t on a log grid (the
    integrand is sharply forward-peaked).  Detection through inverse
    Primakoff is 2x this (spin counting).
    """

    E = np.atleast_1d(np.asarray(e_gamma_mev, dtype=float)) * 1e-3
    ma = ma_mev * 1e-3
    s = M**2 + 2.0 * M * E
    p1 = (s - M**2) / (2.0 * np.sqrt(s))
    k1sq = (s + ma**2 - M**2) ** 2 / (4.0 * s) - ma**2
    valid = k1sq > 0.0
    k1 = np.sqrt(np.where(valid, k1sq, 1.0))
    t_hi = ma**4 / (4.0 * s) - (p1 - k1) ** 2          # closer to zero
    t_lo = ma**4 / (4.0 * s) - (p1 + k1) ** 2
    t_hi = np.minimum(t_hi, -1e-24)

    u = np.linspace(np.log(-t_hi), np.log(-t_lo), n_t, axis=-1)  # (nE, n_t)
    t = -np.exp(u)
    s2 = s[:, None]
    G = (ma**2 * t * (M**2 + s2) - ma**4 * M**2
         - t * ((M**2 - s2) ** 2 + s2 * t))
    dsdt = (2.0 * ALPHA * Z**2 * _screening_f2(-t, Z) * g_gevinv**2
            * M**4 * G / (t**2 * (M**2 - s2) ** 2 * (t - 4.0 * M**2) ** 2))
    sig = np.trapezoid(dsdt * (-t), u, axis=-1)
    out = np.where(valid, np.maximum(sig, 0.0), 0.0) * GEV2_TO_CM2
    return out if np.ndim(e_gamma_mev) else float(out[0])


def sigma_primakoff_dent(e_gamma_mev, ma_mev: float, g_gevinv: float,
                         Z: int = Z_U, n_th: int = 4000):
    """Independent cross-check: the Dent et al. (1912.05733 Eq. 2) forward
    form, dsigma/dcos = (1/4) g^2 alpha Z^2 F^2 |p|^4 sin^2/t^2 with the
    recoilless kinematics E_a ~ E_gamma, t = m_a^2 - 2E(E_a - |p| cos).
    Valid for m_a << E; used only for validation."""

    E = np.atleast_1d(np.asarray(e_gamma_mev, dtype=float)) * 1e-3
    ma = ma_mev * 1e-3
    out = np.zeros_like(E)
    for i, e in enumerate(E):
        if e <= ma:
            continue
        pa = np.sqrt(e**2 - ma**2)
        c = np.linspace(-1.0, 1.0, n_th)
        t = ma**2 - 2.0 * e * (e - pa * c)
        t = np.minimum(t, -1e-30)
        ds = (0.25 * g_gevinv**2 * ALPHA * Z**2 * _screening_f2(-t, Z)
              * pa**4 * (1.0 - c**2) / t**2)
        out[i] = np.trapezoid(ds, c)
    return out * GEV2_TO_CM2


# ---------------------------------------------------------------------------
# Compton-like production and inverse Compton detection (g_aee)
# ---------------------------------------------------------------------------
def _compton_xlims(s, ma):
    root = np.sqrt(np.maximum((s - ME**2 + ma**2) ** 2 - 4.0 * s * ma**2, 0.0))
    xmin = ((s - ME**2) * (s - ME**2 + ma**2) - (s - ME**2) * root) \
        / (2.0 * s * (s - ME**2))
    xmax = ((s - ME**2) * (s - ME**2 + ma**2) + (s - ME**2) * root) \
        / (2.0 * s * (s - ME**2))
    return xmin, xmax


def _dsigma_compton_dx(x, s, ma, g):
    """dsigma/dx per electron [1/GeV^2], gamma e -> a e [2010.15712 Eq. 3.19]."""

    pref = g**2 * ALPHA * x / (4.0 * (s - ME**2) * (1.0 - x))
    brk = (x - 2.0 * ma**2 * s / (s - ME**2) ** 2
           + 2.0 * ma**2 / (s - ME**2) ** 2
           * (ME**2 / (1.0 - x) + ma**2 / x))
    return np.maximum(pref * brk, 0.0)


def sigma_compton(e_gamma_mev, ma_mev: float, g: float, n_x: int = 300):
    """Total Compton-like production cross section per electron [cm^2]."""

    E = np.atleast_1d(np.asarray(e_gamma_mev, dtype=float)) * 1e-3
    ma = ma_mev * 1e-3
    s = ME**2 + 2.0 * E * ME
    valid = s > (ma + ME) ** 2
    xmin, xmax = _compton_xlims(s, ma)
    eps = 1e-9
    x = (xmin[:, None] * (1 - np.linspace(eps, 1 - eps, n_x))
         + xmax[:, None] * np.linspace(eps, 1 - eps, n_x))
    ds = _dsigma_compton_dx(x, s[:, None], ma, g)
    sig = np.trapezoid(ds, x, axis=-1)
    out = np.where(valid, sig, 0.0) * GEV2_TO_CM2
    return out if np.ndim(e_gamma_mev) else float(out[0])


def sigma_inverse_compton(e_a_mev, ma_mev: float, g: float, n: int = 300):
    """Inverse Compton a e -> gamma e, total per electron [cm^2]
    [2511.01812 Eq. (22), integrated over the final photon energy]."""

    Ea = np.atleast_1d(np.asarray(e_a_mev, dtype=float)) * 1e-3
    ma = ma_mev * 1e-3
    out = np.zeros_like(Ea)
    for i, ea in enumerate(Ea):
        if ea <= ma:
            continue
        pa = np.sqrt(ea**2 - ma**2)
        y = 2.0 * ME * ea + ma**2
        eg_lo = y / (2.0 * (ME + ea + pa))
        eg_hi = y / (2.0 * (ME + ea - pa))
        eg = np.linspace(eg_lo * (1 + 1e-9), eg_hi * (1 - 1e-9), n)
        c = (ME + ea - y / (2.0 * eg)) / pa
        s2t = np.maximum(1.0 - c**2, 0.0)
        ds = (ALPHA * g**2 / (32.0 * np.pi) * y / (ME**2 * pa**2 * eg)
              * (1.0 + 4.0 * ME**2 * eg**2 / y**2 - 4.0 * ME * eg / y
                 - 4.0 * ma**2 * ME * pa**2 * eg * s2t / y**3))
        out[i] = np.trapezoid(np.maximum(ds, 0.0), eg)
    return out * GEV2_TO_CM2 if np.ndim(e_a_mev) else float(out[0] * GEV2_TO_CM2)


# ---------------------------------------------------------------------------
# Decays
# ---------------------------------------------------------------------------
def gamma_agg(ma_mev: float, g_gevinv: float) -> float:
    """Gamma(a -> gamma gamma) [GeV]."""

    return g_gevinv**2 * (ma_mev * 1e-3) ** 3 / (64.0 * np.pi)


def gamma_aee(ma_mev: float, g: float) -> float:
    """Gamma(a -> e+ e-) [GeV]; zero below threshold."""

    ma = ma_mev * 1e-3
    if ma <= 2.0 * ME:
        return 0.0
    return g**2 * ma / (8.0 * np.pi) * np.sqrt(1.0 - 4.0 * ME**2 / ma**2)


def decay_length_m(e_a_mev, ma_mev: float, gamma_gev: float):
    """Lab decay length [m]: (p_a/m_a) * hbar c / Gamma."""

    Ea = np.asarray(e_a_mev, dtype=float) * 1e-3
    ma = ma_mev * 1e-3
    if gamma_gev <= 0.0:
        return np.full_like(Ea, np.inf)
    pa = np.sqrt(np.maximum(Ea**2 - ma**2, 1e-30))
    return (pa / ma) * HBARC_GEV_M / gamma_gev


# ---------------------------------------------------------------------------
# The JUNO search
# ---------------------------------------------------------------------------
class ALPSearchJUNO:
    """Event yields and sensitivity for the parked reactor + JUNO."""

    def __init__(
        self,
        power_mw: float = 10.0,
        baseline_m: float = 50.0,
        years: float = 3.0,
        duty: float = 0.9,
        det_radius_m: float = 16.5,
        lab_mass_kt: float = 16.2,       # fiducial LAB mass (0.806 x 20 kt)
        window_mev: tuple = (3.0, 10.0),
        background_events: float | None = None,
        sigma_sys: float = 0.10,
        n_e: int = 121,
    ):
        self.power_mw = power_mw
        self.baseline_m = baseline_m
        self.livetime_s = years * 365.25 * 86400.0 * duty
        self.det_radius_m = det_radius_m
        self.window_mev = window_mev
        self.sigma_sys = sigma_sys
        self.background_events = background_events

        m_g = lab_mass_kt * 1e9
        f_c, f_h = 0.8799, 0.1201        # LAB ~ CnH2n mass fractions
        NA = 6.022e23
        self.n_carbon = m_g * f_c / 12.011 * NA
        self.n_hydrogen = m_g * f_h / 1.008 * NA
        self.n_electrons = m_g * (f_c * 6 / 12.011 + f_h / 1.008) * NA

        # geometry [2511.01812 Sec. II]: flux dilution to the detector centre,
        # survival to the front face, decay over the mean chord of the sphere
        self.fourpiL2_cm2 = 4.0 * np.pi * (baseline_m * 100.0) ** 2
        self.area_cm2 = np.pi * (det_radius_m * 100.0) ** 2
        self.l_front_m = baseline_m - det_radius_m
        self.chord_m = 4.0 * det_radius_m / 3.0

        self.e_a = np.linspace(window_mev[0], window_mev[1], n_e)
        self._phi = reactor_photon_flux(self.e_a, power_mw)
        self._sm = sigma_sm_uranium(self.e_a)

    # -- g_agamma channel ---------------------------------------------------
    def _flux0_gagg(self, ma):
        """Emitted ALP flux /MeV/s at g = 1 GeV^-1 (delta approx, Eq. 3.7)."""

        return self._phi * sigma_primakoff(self.e_a, ma, 1.0) / self._sm

    def events_gagg(self, ma_mev: float, g: float) -> dict:
        flux = self._flux0_gagg(ma_mev) * g**2
        ell = decay_length_m(self.e_a, ma_mev, gamma_agg(ma_mev, g))
        surv_c = np.exp(-self.baseline_m / ell)
        surv_f = np.exp(-self.l_front_m / ell)

        sig_det = 2.0 * (self.n_carbon * sigma_primakoff(self.e_a, ma_mev, g,
                                                         Z=Z_C, M=M_C)
                         + self.n_hydrogen * sigma_primakoff(self.e_a, ma_mev,
                                                             g, Z=Z_H, M=M_H))
        scatter = np.trapezoid(flux * surv_c * sig_det, self.e_a) \
            / self.fourpiL2_cm2 * self.livetime_s
        p_dec = 1.0 - np.exp(-self.chord_m / ell)
        decay = np.trapezoid(flux * surv_f * p_dec, self.e_a) \
            * self.area_cm2 / self.fourpiL2_cm2 * self.livetime_s
        return {"scatter": float(scatter), "decay": float(decay),
                "total": float(scatter + decay)}

    # -- g_aee channel ------------------------------------------------------
    def _flux0_gaee(self, ma, n_eg: int = 200):
        """Emitted ALP flux /MeV/s at g = 1 (Compton conv., 2511 Eq. 14)."""

        eg = np.linspace(max(0.15, ma * 1.05), 14.0, n_eg)
        phi_g = reactor_photon_flux(eg, self.power_mw)
        sm = sigma_sm_uranium(eg)
        Eg = eg[None, :] * 1e-3
        Ea = self.e_a[:, None] * 1e-3
        ma_g = ma * 1e-3
        s = ME**2 + 2.0 * Eg * ME
        x = 1.0 - Ea / Eg + ma_g**2 / (2.0 * Eg * ME)
        xmin, xmax = _compton_xlims(s, ma_g)
        ok = (s > (ma_g + ME) ** 2) & (x > xmin) & (x < xmax) \
            & (x > 0) & (x < 1)
        ds_dx = _dsigma_compton_dx(np.clip(x, 1e-9, 1 - 1e-9), s, ma_g, 1.0)
        # dsigma/dEa = dsigma/dx / E_gamma  (per electron; x92 electrons/U,
        # in GeV^-2 / GeV -> x 1e-3 for /MeV)
        ds_dEa = np.where(ok, ds_dx / Eg, 0.0) * GEV2_TO_CM2 * 1e-3
        return np.trapezoid(phi_g[None, :] * Z_U * ds_dEa / sm[None, :],
                            eg, axis=1)

    def events_gaee(self, ma_mev: float, g: float) -> dict:
        flux = self._flux0_gaee(ma_mev) * g**2
        ell = decay_length_m(self.e_a, ma_mev, gamma_aee(ma_mev, g))
        surv_c = np.exp(-self.baseline_m / ell)
        surv_f = np.exp(-self.l_front_m / ell)

        sig_det = self.n_electrons * sigma_inverse_compton(self.e_a, ma_mev, g)
        scatter = np.trapezoid(flux * surv_c * sig_det, self.e_a) \
            / self.fourpiL2_cm2 * self.livetime_s
        p_dec = 1.0 - np.exp(-self.chord_m / ell)
        decay = np.trapezoid(flux * surv_f * p_dec, self.e_a) \
            * self.area_cm2 / self.fourpiL2_cm2 * self.livetime_s
        return {"scatter": float(scatter), "decay": float(decay),
                "total": float(scatter + decay)}

    # -- statistics ---------------------------------------------------------
    def estimate_background(self) -> float:
        """Reactor-on single-hit background in the window over the exposure.

        Built from the machinery of notebooks 5-6: untagged IBD positrons,
        reactor EvES, and solar EvES, all restricted to the analysis window.
        Cosmogenics (e.g. 12B) are not modelled -- stated caveat.
        """

        from .detector import integration_weights
        from .near_sm import FixedNearReactor
        from .sterile import solar_flux_density, solar_pee
        from .near_sm import GA_SM, GV_SM, eves_dsigma_dT

        lo, hi = self.window_mev
        base = FixedNearReactor(power_mwth=self.power_mw,
                                baseline_m=self.baseline_m,
                                recoil_edges=np.arange(lo, hi + 1e-9, 0.25))
        eves = float(base.eves_spectrum_T().sum()) * self.livetime_s
        singles = float(base.ibd_singles().sum()) * self.livetime_s
        # solar EvES in the window (whole fiducial volume, calendar time)
        es = np.linspace(1.0, 18.0, 500)
        ws = integration_weights(es)
        phi = solar_flux_density(es)
        pee = solar_pee(es, base.truth)
        t_grid = base.t_grid
        k_e = eves_dsigma_dT(es[None, :], t_grid[:, None], "e", False,
                             GV_SM, GA_SM)
        k_m = eves_dsigma_dT(es[None, :], t_grid[:, None], "mu", False,
                             GV_SM, GA_SM)
        spec = k_e @ (phi * pee * ws) + k_m @ (phi * (1 - pee) * ws)
        w_t = integration_weights(t_grid)
        in_win = (t_grid >= lo) & (t_grid <= hi)
        # fiducial electrons x selection efficiency; solar runs on calendar time
        solar = float((spec * w_t)[in_win].sum()) * self.n_electrons * 0.8 \
            * (self.livetime_s / 0.9)
        total = eves + singles + solar
        self._bkg_parts = {"reactor EvES": eves, "IBD singles": singles,
                           "solar EvES": solar}
        return total

    def chi2(self, n_signal: float, n_background: float | None = None) -> float:
        nb = (self.background_events if n_background is None
              else n_background) or 0.0
        ns = n_signal
        return ns**2 / (ns + nb + (self.sigma_sys * ns) ** 2)

    def chi2_grid(self, coupling: str, ma_grid, g_grid,
                  background: float | None = None) -> np.ndarray:
        """chi^2 over the (ma, g) plane for 'gagg' or 'gaee'."""

        out = np.empty((len(ma_grid), len(g_grid)))
        for i, ma in enumerate(ma_grid):
            flux0 = (self._flux0_gagg(ma) if coupling == "gagg"
                     else self._flux0_gaee(ma))
            det0 = (2.0 * (self.n_carbon
                           * sigma_primakoff(self.e_a, ma, 1.0, Z=Z_C, M=M_C)
                           + self.n_hydrogen
                           * sigma_primakoff(self.e_a, ma, 1.0, Z=Z_H, M=M_H))
                    if coupling == "gagg"
                    else self.n_electrons
                    * sigma_inverse_compton(self.e_a, ma, 1.0))
            gam0 = (gamma_agg(ma, 1.0) if coupling == "gagg"
                    else gamma_aee(ma, 1.0))
            ell0 = decay_length_m(self.e_a, ma, gam0) if gam0 > 0 else None
            for j, g in enumerate(g_grid):
                if ell0 is not None:
                    ell = ell0 / g**2
                    surv_c = np.exp(-self.baseline_m / ell)
                    surv_f = np.exp(-self.l_front_m / ell)
                    p_dec = 1.0 - np.exp(-self.chord_m / ell)
                else:
                    surv_c = surv_f = 1.0
                    p_dec = 0.0
                flux = flux0 * g**2
                ns = (np.trapezoid(flux * surv_c * det0 * g**2, self.e_a)
                      / self.fourpiL2_cm2
                      + np.trapezoid(flux * surv_f * p_dec, self.e_a)
                      * self.area_cm2 / self.fourpiL2_cm2) * self.livetime_s
                out[i, j] = self.chi2(ns, background)
        return out
