"""Joint near+far theta13 / dm2_ee sensitivity with the standard-method ingredients.

Far side
--------
The nine-reactor JUNO signal (Yangjiang + Taishan + the Daya Bay effective
core) with the measured Daya Bay ``Total`` yield spectrum, matter effects at
2.55 g/cm^3, the released positron non-linearity, the a=3.3%/b=1% resolution,
and the JUNO 2025 background components scaled by livetime -- i.e. the
ingredients of ``reactor.nufit.standard_juno_fit``, projected to an arbitrary
exposure.  (The bin-per-bin rescaling of the standard fit has no projection
analogue: it pins the prediction to the *measured* un-oscillated spectrum,
which does not exist for future data.  Its role here is played by the measured
flux plus its released covariance.)

Near side
---------
A movable HALEU microreactor: 98% U235 / 2% U238 fresh, plutonium ingrowth
following the Daya Bay fuel-evolution trajectory (``reactor.flux
.haleu_fractions``).  Its per-fission IBD yield uses the **measured** Daya Bay
U235 and Pu239 spectra; U238 and Pu241 (percent-level contributions) come from
Huber-Mueller x Vogel-Beacom.

Systematics, as joint Gaussian modes over the stacked (far + stops) bins
------------------------------------------------------------------------
* far reactor rate 1.8% (their Tab. 2)          -- far only
* near thermal power (default 2%)               -- near only, common to stops
* selection efficiency 1.6%                     -- SHARED far/near
* energy scale 0.5%, bias 0.5%, resolution 5%   -- SHARED far/near
* background components at the release priors   -- SHARED far/near
* fuel evolution: the ingrown Pu fraction, default +-30%   -- near only
* U238 contribution, default +-15%              -- near only
* the full 75x75 Daya Bay covariance over [U235, Pu239, Total]: each eigenmode
  distorts the far spectrum (through Total) and the near spectrum (through its
  U235 and Pu239 components) **coherently** -- the far and near fluxes are
  correlated because they are anchored to the same measurement.

``correlated_detector=False`` splits the shared modes into independent far and
near copies, to quantify what the correlation buys.

Statistics: Asimov.  C = diag(prediction) + sum_k v_k v_k^T with unit-Gaussian
modes v_k; Fisher errors and Delta chi^2 = d^T C^-1 d follow analytically, so
no nuisance minimisation is ever needed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .constants import (
    DAYABAY_DM2EE,
    DAYABAY_SIN2_THETA13,
    JUNO2025_DM2_21,
    JUNO2025_EFFICIENCY_TOTAL,
    JUNO2025_SIN2_THETA12,
    JUNO2025_TARGET_PROTONS,
    SECONDS_PER_DAY,
    CM_PER_KM,
    OscillationParameters,
)
from .cross_sections import vogel_beacom
from .dayabay_data import load_unfolded
from .detector import (
    DetectorResponse,
    EnergyResolution,
    TabulatedNonLinearity,
    gaussian_bin_response,
    integration_weights,
)
from .flux import (
    default_juno_cores,
    fission_rate_per_second,
    haleu_fractions,
    spectrum_per_fission,
)
from .juno_data import load_spectrum
from .oscillations import survival_probability_ee, survival_probability_matter

#: Truth point: JUNO 2025 solar parameters, Daya Bay atmospheric ones.
DEFAULT_TRUTH = OscillationParameters(
    sin2_theta12=JUNO2025_SIN2_THETA12,
    dm2_21=JUNO2025_DM2_21,
    sin2_theta13=DAYABAY_SIN2_THETA13,
    dm2_ee=DAYABAY_DM2EE,
)

#: Release background priors (their Tab. 1), same as the standard fit.
BACKGROUND_PRIORS = {
    "9Li/8He": 0.33,
    "geoneutrino": 0.42,
    "world reactors": 0.10,
    "214Bi-214Po": 0.56,
    "other": 1.00,
}


@dataclass(frozen=True)
class Stop:
    """One stop of the movable-reactor schedule."""

    baseline_km: float
    days: float
    burnup: float  # position in the Daya Bay-shaped fuel cycle, 0..1


def schedule(baselines_km, days, cycle_days: float = 548.0, start_burnup: float = 0.0):
    """Build stops with cumulative burnup along the programme.

    ``cycle_days`` is the length of the fuel-evolution cycle; the default is a
    Daya Bay-like 18 months, so "burnup" advances at the Daya Bay rate.  Each
    stop is assigned the burnup at its midpoint.
    """

    baselines_km = np.atleast_1d(np.asarray(baselines_km, dtype=float))
    days = np.atleast_1d(np.asarray(days, dtype=float))
    if days.size == 1:
        days = np.full(baselines_km.size, days[0])
    out, elapsed = [], 0.0
    for length, t in zip(baselines_km, days):
        beta = start_burnup + (elapsed + 0.5 * t) / cycle_days
        out.append(Stop(float(length), float(t), float(min(beta, 1.0))))
        elapsed += t
    return tuple(out)


class MicroreactorYield:
    """Per-fission IBD yield of the HALEU source, cm^2 / fission / MeV.

    U235 and Pu239 from the measured Daya Bay unfolded spectra; U238 and Pu241
    from Huber-Mueller x Vogel-Beacom.
    """

    def __init__(self, e_nu_mev: np.ndarray):
        spectra, _, _ = load_unfolded()
        xsec = vogel_beacom(e_nu_mev, order=1)
        self.e = e_nu_mev
        self.c235 = spectra["U235"](e_nu_mev)
        self.c239 = spectra["Pu239"](e_nu_mev)
        self.c238 = spectrum_per_fission(e_nu_mev, "U238") * xsec
        self.c241 = spectrum_per_fission(e_nu_mev, "Pu241") * xsec

    def components(self, burnup: float) -> dict[str, np.ndarray]:
        f = haleu_fractions(burnup, evolve=True)
        return {
            "U235": f["U235"] * self.c235,
            "U238": f["U238"] * self.c238,
            "Pu239": f["Pu239"] * self.c239,
            "Pu241": f["Pu241"] * self.c241,
        }

    def __call__(self, burnup: float) -> np.ndarray:
        return sum(self.components(burnup).values())

    def ingrowth_derivative(self, burnup: float) -> np.ndarray:
        """d(yield)/d(scale) for Pu ingrowth scaled by (1 + scale).

        Scaling the ingrown Pu fractions moves fission share from U235 into
        Pu239/Pu241, at fixed U238.
        """

        f = haleu_fractions(burnup, evolve=True)
        return (f["Pu239"] * (self.c239 - self.c235)
                + f["Pu241"] * (self.c241 - self.c235))


class NearFarTheta13:
    """Asimov Fisher / Delta chi^2 for the joint near + far dataset."""

    def __init__(
        self,
        far_days: float = 6 * 365.25,
        stops: tuple = (),
        power_mwth: float = 100.0,
        truth: OscillationParameters = DEFAULT_TRUTH,
        include_far: bool = True,
        # -- priors ---------------------------------------------------------
        sigma_far_rate: float = 0.018,
        sigma_efficiency: float = 0.016,
        sigma_power: float = 0.02,
        sigma_scale: float = 0.005,
        sigma_bias: float = 0.005,
        sigma_res: float = 0.05,
        sigma_evolution: float = 0.30,
        sigma_u238: float = 0.15,
        use_flux_covariance: bool = True,
        correlated_detector: bool = True,
        # -- response / binning --------------------------------------------
        resolution_a: float = 0.033,
        resolution_b: float = 0.010,
        far_edges: np.ndarray | None = None,
        near_edges: np.ndarray | None = None,
        e_nu_grid: np.ndarray | None = None,
        matter_density: float = 2.55,
    ):
        self.truth = truth
        self.stops = tuple(stops)
        self.include_far = include_far
        self.matter_density = matter_density
        enu = np.linspace(1.806, 13.0, 2600) if e_nu_grid is None else e_nu_grid
        self.e_nu_grid = enu
        w = integration_weights(enu)

        self.far_edges = (np.arange(0.94, 9.0 + 1e-9, 0.02)
                          if far_edges is None else far_edges)
        self.near_edges = (np.linspace(1.02, 8.22, 201)
                           if near_edges is None else near_edges)

        # -- response and its derivatives (shared detector) -----------------
        nl = TabulatedNonLinearity.from_release("positron")
        res = EnergyResolution(a=resolution_a, b=resolution_b, c=0.0)
        e_dep = DetectorResponse(use_ibd_recoil=True).deposited_energy(enu)

        def response(edges, xi_scl=0.0, xi_bias=0.0, xi_res=0.0):
            e_vis = e_dep * ((1.0 + xi_scl) * nl.factor(e_dep) + xi_bias)
            sigma = (1.0 + xi_res) * res.sigma(e_vis)
            return gaussian_bin_response(e_vis, edges, sigma)

        h = 1.0e-3
        self._R = {}
        self._dR = {}
        for tag, edges in (("far", self.far_edges), ("near", self.near_edges)):
            self._R[tag] = response(edges)
            self._dR[tag] = [
                (response(edges, **{k: h}) - response(edges, **{k: -h})) / (2 * h)
                for k in ("xi_scl", "xi_bias", "xi_res")
            ]

        # -- far flux (Daya Bay Total yield, nine cores) ---------------------
        spectra, cov75, _ = load_unfolded()
        self._dyb_centers = spectra["Total"].centers
        exposure = lambda days: (JUNO2025_TARGET_PROTONS * JUNO2025_EFFICIENCY_TOTAL
                                 * days * SECONDS_PER_DAY)
        self._far_density = []
        self._far_baselines = []
        if include_far:
            total_yield = spectra["Total"](enu)
            for c in default_juno_cores(include_distant=True):
                dens = (c.fission_rate() * c.duty_cycle * total_yield
                        / (4.0 * np.pi * (c.baseline_km * CM_PER_KM) ** 2)
                        * w * exposure(far_days))
                self._far_density.append(dens)
                self._far_baselines.append(c.baseline_km)

        # -- near flux (HALEU microreactor) ----------------------------------
        self._yield = MicroreactorYield(enu)
        self._near_density = []      # per stop: prefactor x yield components
        self._near_prefactor = []
        for st in self.stops:
            fr = haleu_fractions(st.burnup, evolve=True)
            rate = fission_rate_per_second(power_mwth / 1000.0, fr)
            pref = (rate / (4.0 * np.pi * (st.baseline_km * CM_PER_KM) ** 2)
                    * w * exposure(st.days))
            self._near_prefactor.append(pref)
            self._near_density.append(pref * self._yield(st.burnup))

        # -- backgrounds (release shapes, scaled by livetime) ----------------
        rel = load_spectrum()
        self._bkg_names = list(rel.backgrounds)
        self._bkg_prior = np.array([BACKGROUND_PRIORS[k] for k in self._bkg_names])

        def bkg_on(edges, days):
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            out = []
            for name in self._bkg_names:
                dens = rel.backgrounds[name] / rel.widths / rel.livetime_days
                out.append(np.interp(centers, rel.centers, dens,
                                     left=0.0, right=0.0) * widths * days)
            return np.array(out)         # (n_bkg, n_bins)

        self._bkg_far = (bkg_on(self.far_edges, far_days) if include_far
                         else np.zeros((len(self._bkg_names),
                                        len(self.far_edges) - 1)))
        self._bkg_near = [bkg_on(self.near_edges, st.days) for st in self.stops]

        # -- Asimov prediction at truth --------------------------------------
        self._n_far = (len(self.far_edges) - 1) if include_far else 0
        self._n_near = len(self.near_edges) - 1
        far_sig, near_sig = self._signal_pieces(truth)
        self._far_sig0, self._near_sig0 = far_sig, near_sig
        self.signal0 = self._stack(far_sig, near_sig)
        self.background0 = self._stack(
            self._bkg_far.sum(axis=0) if include_far else np.zeros(0),
            [b.sum(axis=0) for b in self._bkg_near])
        self.asimov = self.signal0 + self.background0

        # -- systematic modes -------------------------------------------------
        modes = []
        zeros_far = np.zeros(self._n_far)
        zeros_near = [np.zeros(self._n_near) for _ in self.stops]

        if include_far:
            modes.append(("far rate 1.8%",
                          self._stack(sigma_far_rate * far_sig, zeros_near)))
        if self.stops:
            modes.append(("near power",
                          self._stack(zeros_far,
                                      [sigma_power * s for s in near_sig])))

        def shared(name, far_part, near_part):
            if correlated_detector:
                modes.append((name, self._stack(far_part, near_part)))
            else:
                if include_far and np.any(far_part):
                    modes.append((name + " (far)",
                                  self._stack(far_part, zeros_near)))
                if self.stops and any(np.any(p) for p in near_part):
                    modes.append((name + " (near)",
                                  self._stack(zeros_far, near_part)))

        shared("efficiency 1.6%", sigma_efficiency * far_sig,
               [sigma_efficiency * s for s in near_sig])

        # energy scale / bias / resolution act on the signal through dR
        far_flux0 = self._far_flux_vector(truth)
        near_flux0 = self._near_flux_vectors(truth)
        for k, (label, width) in enumerate(
                [("energy scale 0.5%", sigma_scale),
                 ("energy bias 0.5%", sigma_bias),
                 ("resolution 5%", sigma_res)]):
            fp = (width * (self._dR["far"][k] @ far_flux0) if include_far
                  else zeros_far)
            nparts = [width * (self._dR["near"][k] @ f) for f in near_flux0]
            shared(label, fp, nparts)

        # backgrounds: one pull per component, shared far/near
        for i, name in enumerate(self._bkg_names):
            fp = self._bkg_prior[i] * self._bkg_far[i] if include_far else zeros_far
            nparts = [self._bkg_prior[i] * b[i] for b in self._bkg_near]
            shared(f"bkg {name}", fp, nparts)

        # near-only spectral systematics
        if self.stops:
            u238 = [self._R["near"] @ (pref
                    * self._yield.components(st.burnup)["U238"]
                    * self._pee_near(truth, st.baseline_km))
                    for pref, st in zip(self._near_prefactor, self.stops)]
            modes.append(("U238 15%",
                          self._stack(zeros_far, [sigma_u238 * u for u in u238])))
            evo = [self._R["near"] @ (pref
                   * self._yield.ingrowth_derivative(st.burnup)
                   * self._pee_near(truth, st.baseline_km))
                   for pref, st in zip(self._near_prefactor, self.stops)]
            modes.append(("fuel evolution 30%",
                          self._stack(zeros_far, [sigma_evolution * e for e in evo])))

        # the joint Daya Bay flux covariance: [U235, Pu239, Total]
        if use_flux_covariance:
            vals = np.concatenate([spectra["U235"].values,
                                   spectra["Pu239"].values,
                                   spectra["Total"].values])
            rel_cov = np.asarray(cov75) * 1e-86 / np.outer(vals, vals)
            evals, evecs = np.linalg.eigh(rel_cov)
            keep = evals > 1e-10 * evals.max()
            psi = (evecs[:, keep] * np.sqrt(evals[keep])).T   # (n_modes, 75)
            cen = self._dyb_centers
            for k, mode in enumerate(psi):
                rel_u = np.interp(enu, cen, mode[:25])
                rel_p = np.interp(enu, cen, mode[25:50])
                rel_t = np.interp(enu, cen, mode[50:])
                fp = zeros_far
                if include_far:
                    fp = self._R["far"] @ (far_flux0 * rel_t)
                nparts = []
                for pref, st in zip(self._near_prefactor, self.stops):
                    comp = self._yield.components(st.burnup)
                    dist = comp["U235"] * rel_u + comp["Pu239"] * rel_p
                    nparts.append(self._R["near"] @ (
                        pref * dist * self._pee_near(truth, st.baseline_km)))
                modes.append((f"DYB flux mode {k}", self._stack(fp, nparts)))

        self.mode_names = [m[0] for m in modes]
        self._modes = np.array([m[1] for m in modes]) if modes else \
            np.zeros((0, self.asimov.size))

        cov = np.diag(np.maximum(self.asimov, 1e-9))
        if len(self._modes):
            cov = cov + self._modes.T @ self._modes
        self._cho = cho_factor(cov)

    # -- prediction ----------------------------------------------------------
    def _pee_near(self, params, baseline_km):
        return survival_probability_ee(self.e_nu_grid, baseline_km, params)

    def _far_flux_vector(self, params):
        total = np.zeros_like(self.e_nu_grid)
        for dens, L in zip(self._far_density, self._far_baselines):
            total += dens * survival_probability_matter(
                self.e_nu_grid, L, params, self.matter_density)
        return total

    def _near_flux_vectors(self, params):
        return [dens * self._pee_near(params, st.baseline_km)
                for dens, st in zip(self._near_density, self.stops)]

    def _signal_pieces(self, params):
        far = (self._R["far"] @ self._far_flux_vector(params)
               if self.include_far else np.zeros(0))
        near = [self._R["near"] @ f for f in self._near_flux_vectors(params)]
        return far, near

    def _stack(self, far, near_list):
        parts = ([far] if self.include_far else []) + list(near_list)
        return np.concatenate(parts) if parts else np.zeros(0)

    def predict(self, params: OscillationParameters) -> np.ndarray:
        far, near = self._signal_pieces(params)
        return self._stack(far, near)

    # -- statistics ----------------------------------------------------------
    def chi2(self, params: OscillationParameters) -> float:
        d = self.predict(params) - self.signal0
        return float(d @ cho_solve(self._cho, d))

    def fisher_errors(self, param_names=("sin2_theta13", "dm2_ee",
                                         "sin2_theta12", "dm2_21")) -> dict:
        steps = {"sin2_theta13": 4e-4, "dm2_ee": 2e-5,
                 "sin2_theta12": 3e-3, "dm2_21": 8e-7}
        derivs = []
        for name in param_names:
            h = steps[name]
            up = replace(self.truth, **{name: getattr(self.truth, name) + h})
            dn = replace(self.truth, **{name: getattr(self.truth, name) - h})
            derivs.append((self.predict(up) - self.predict(dn)) / (2 * h))
        d = np.array(derivs)
        sol = np.array([cho_solve(self._cho, row) for row in d])
        fisher = d @ sol.T
        cov = np.linalg.inv(fisher)
        out = {n: float(np.sqrt(cov[i, i])) for i, n in enumerate(param_names)}
        out["covariance"] = cov
        out["params"] = list(param_names)
        return out

    def rate_summary(self) -> dict:
        far = float(self._far_sig0.sum()) if self.include_far else 0.0
        return {
            "far signal": far,
            "near signal per stop": [float(s.sum()) for s in self._near_sig0],
            "background total": float(self.background0.sum()),
            "modes": len(self.mode_names),
        }
