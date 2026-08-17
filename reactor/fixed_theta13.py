"""theta13 from a FIXED near reactor: exploiting L/E inside the JUNO volume.

The mobile-reactor programme (notebook 2) uses a movable source to sample the
theta13 oscillation at several baselines, with an anchor stop cancelling
flux/normalisation systematics.  This module asks the alternative question: a
*fixed* reactor at 50 m - 1.5 km, and JUNO's own size doing the sampling.

The physics
-----------
For a point source at distance D from the centre of a fiducial sphere of
radius R (~16.5 m), vertices span L in [D-R, D+R].  The theta13 disappearance

    P_ee = 1 - sin^2(2 theta13) sin^2(1.267 dm2_ee L / E) - (solar, negligible)

has an oscillation length ~ 1.6 km x E[MeV] / (dm2_ee/2.5e-3), so across the
detector the *phase* differs between the near and far walls.  Binning events
in reconstructed vertex baseline L and energy E, the fit uses the *L/E shape*
of the disappearance, which no flux/normalisation systematic can mimic --
those are all coherent across L.  The rate goes as 1/D^2 but the phase lever
arm grows as D: the notebook scans D = 50 m .. 1.5 km to find where they
balance.

Delta m^2_ee is not free here: JUNO's own GW reactors at 52.5 km determine
it (via the fine atmospheric wiggles, independently of the near source), and
the module accepts a Gaussian prior sigma(dm2_ee) computed with
:class:`reactor.theta13.NearFarTheta13` for a stated far exposure.

Monte Carlo
-----------
``VertexMonteCarlo`` samples IBD vertices uniformly in the fiducial sphere,
draws the true energy from the reactor spectrum, applies the vertex smearing
sigma_vtx = sigma_1 / sqrt(E_vis) and returns the (L_reco, E_reco) event
list, so geometric acceptance, projection of the smearing onto L, and the
finite core are all done by sampling rather than by formula.  The analytic
baseline distribution A(L)/L^2 is used only to *validate* the MC.

Statistics
----------
Asimov Fisher analysis on the binned (L, E) prediction with the joint-mode
covariance C = diag(mu) + V^T V (Woodbury), the same machinery as
:mod:`reactor.sterile`; systematics: free flux normalisation (10%), the
measured U235 25-mode shape covariance, U238, fuel evolution, energy
scale/bias/resolution, and a Gaussian-correlated response non-uniformity
field over L.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .constants import (
    JUNO2025_TARGET_PROTONS,
    OscillationParameters,
    SECONDS_PER_DAY,
)
from .cross_sections import vogel_beacom
from .detector import (
    DetectorResponse,
    EnergyResolution,
    TabulatedNonLinearity,
    gaussian_bin_response,
    integration_weights,
)
from .near_sm import FixedNearReactor
from .theta13 import DEFAULT_TRUTH

DAY = SECONDS_PER_DAY


# ---------------------------------------------------------------------------
# Monte Carlo of vertices in the fiducial volume
# ---------------------------------------------------------------------------
class VertexMonteCarlo:
    """Sample IBD vertices in the fiducial sphere illuminated by a point core."""

    def __init__(self, distance_m: float, detector_radius_m: float = 16.5,
                 core_radius_m: float = 0.5, sigma_vertex_m: float = 0.10,
                 seed: int = 12345):
        self.D = distance_m
        self.R = detector_radius_m
        self.a_core = core_radius_m
        self.sigma_1 = sigma_vertex_m
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int, e_true_sampler):
        """Return dict with true/reco baselines [m] and energies [MeV]."""

        rng = self.rng
        # uniform in the sphere; the 1/L^2 flux weight is applied as an event
        # weight so the sample stays unbiased in position for validation
        u = rng.random(n)
        r = self.R * u ** (1.0 / 3.0)
        cos_t = 2.0 * rng.random(n) - 1.0
        phi = 2.0 * np.pi * rng.random(n)
        sin_t = np.sqrt(1.0 - cos_t**2)
        x = r * sin_t * np.cos(phi)
        y = r * sin_t * np.sin(phi)
        z = r * cos_t
        # source at (0, 0, -D); finite core: uniform ball of radius a
        if self.a_core > 0:
            uc = rng.random(n)
            rc = self.a_core * uc ** (1.0 / 3.0)
            ct = 2.0 * rng.random(n) - 1.0
            ph = 2.0 * np.pi * rng.random(n)
            st = np.sqrt(1.0 - ct**2)
            sx, sy, sz = rc * st * np.cos(ph), rc * st * np.sin(ph), -self.D + rc * ct
        else:
            sx = sy = np.zeros(n); sz = np.full(n, -self.D)
        l_true = np.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2)
        e_true = e_true_sampler(n)
        w = 1.0 / l_true**2                    # flux weight
        # vertex smearing (isotropic Gaussian, sigma_1/sqrt(E_vis))
        e_vis = np.maximum(e_true - 0.78, 0.5)
        sig = self.sigma_1 / np.sqrt(e_vis)
        xr = x + rng.normal(0, sig); yr = y + rng.normal(0, sig)
        zr = z + rng.normal(0, sig)
        l_reco = np.sqrt(xr**2 + yr**2 + (zr + self.D) ** 2)
        return {"l_true": l_true, "l_reco": l_reco, "e_true": e_true,
                "weight": w, "r": r}

    @staticmethod
    def analytic_dNdL(L, D, R):
        """Un-normalised analytic baseline density A(L)/L^2 (point core)."""

        cap = np.clip(1.0 - (L**2 + D**2 - R**2) / (2.0 * L * D), 0.0, 2.0)
        return 2.0 * np.pi * L**2 * cap / L**2


# ---------------------------------------------------------------------------
# The fixed-reactor theta13 analysis
# ---------------------------------------------------------------------------
class FixedReactorTheta13:
    """Asimov Fisher analysis of theta13 from (L_reco, E_reco) binned IBD."""

    def __init__(
        self,
        distance_m: float,
        power_mwth: float = 10.0,
        years: float = 3.0,
        duty_cycle: float = 0.9,
        detector_radius_m: float = 16.5,
        core_radius_m: float = 0.5,
        sigma_vertex_m: float = 0.10,
        n_l_bins: int = 12,
        e_edges: np.ndarray | None = None,
        truth: OscillationParameters = DEFAULT_TRUTH,
        sigma_dm2ee_prior: float | None = None,   # absolute, eV^2
        # systematics
        sigma_norm: float = 0.10,
        sigma_uniformity: float = 0.005,
        uniformity_corr_m: float = 3.0,
        sigma_scale: float = 0.005,
        sigma_bias: float = 0.005,
        sigma_res: float = 0.05,
        sigma_u238: float = 0.15,
        sigma_evolution: float = 0.30,
        use_flux_covariance: bool = True,
        l_binned: bool = True,
        n_mc: int = 400_000,
        burnup: float = 0.5,
        seed: int = 12345,
    ):
        self.D, self.R = distance_m, detector_radius_m
        self.truth = truth
        self.years = years
        self.l_binned = l_binned
        base = FixedNearReactor(power_mwth=power_mwth, baseline_m=distance_m,
                                burnup=burnup, use_flux_covariance=use_flux_covariance,
                                include_backgrounds=False)
        self._base = base
        enu = base.e_nu_grid
        self.e_nu_grid = enu
        w_e = integration_weights(enu)
        self.e_edges = (np.arange(1.0, 9.0 + 1e-9, 0.2)
                        if e_edges is None else e_edges)
        n_e = len(self.e_edges) - 1

        # -- energy response (positron) + pull derivatives ---------------------
        res = EnergyResolution(a=0.033, b=0.010, c=0.0)
        nl_p = TabulatedNonLinearity.from_release("positron")
        e_dep = DetectorResponse(use_ibd_recoil=True).deposited_energy(enu)

        def resp(sc=0.0, bi=0.0, re_=0.0):
            ev = e_dep * ((1.0 + sc) * nl_p.factor(e_dep) + bi)
            return gaussian_bin_response(ev, self.e_edges, (1.0 + re_) * res.sigma(ev))

        self._R = resp()
        hh = 1e-3
        self._dR = [(resp(**{k: hh}) - resp(**{k: -hh})) / (2 * hh)
                    for k in ("sc", "bi", "re_")]

        # -- Monte Carlo of vertices -> per-L-bin baseline distributions -------
        self.mc = VertexMonteCarlo(distance_m, detector_radius_m,
                                   core_radius_m, sigma_vertex_m, seed)
        yld = base._yield_ibd * w_e
        cdf = np.cumsum(yld) / yld.sum()

        def e_sampler(n):
            return np.interp(self.mc.rng.random(n), cdf, enu)

        ev = self.mc.sample(n_mc, e_sampler)
        self._mc_events = ev
        lo, hi = max(distance_m - detector_radius_m, 1e-3), distance_m + detector_radius_m
        n_l = n_l_bins if l_binned else 1
        self.l_edges = np.linspace(lo, hi, n_l + 1)
        self.n_l = n_l
        # per L_reco bin: distribution of TRUE baseline (weighted), which is what
        # the oscillation depends on -- this is where resolution & core enter
        self._l_true_nodes, self._l_true_w = [], []
        for i in range(n_l):
            sel = (ev["l_reco"] >= self.l_edges[i]) & (ev["l_reco"] < self.l_edges[i + 1])
            if not np.any(sel):
                self._l_true_nodes.append(np.array([0.5 * (self.l_edges[i] + self.l_edges[i+1])]))
                self._l_true_w.append(np.array([0.0])); continue
            hist, edges = np.histogram(ev["l_true"][sel], bins=25,
                                       weights=ev["weight"][sel])
            cen = 0.5 * (edges[:-1] + edges[1:])
            self._l_true_nodes.append(cen)
            self._l_true_w.append(hist / max(hist.sum(), 1e-300))
        # event fraction per L bin (relative), then absolute normalisation
        frac = np.array([ev["weight"][(ev["l_reco"] >= self.l_edges[i])
                                      & (ev["l_reco"] < self.l_edges[i + 1])].sum()
                         for i in range(n_l)])
        frac /= frac.sum()
        self.l_fraction = frac
        # total events: rate through the sphere = base rate at D scaled by the
        # exact geometric average of 1/L^2 over the volume (MC gives it)
        mean_inv_l2 = float(np.mean(ev["weight"]))
        rate_scale = mean_inv_l2 * distance_m**2      # relative to point at D
        t_live = years * 365.25 * DAY * duty_cycle
        # spectrum density (per grid point) at the reference distance D
        self._dens_ref = (base._geom * JUNO2025_TARGET_PROTONS * base.efficiency
                          * yld * t_live * rate_scale)
        self.total_events = float(self._dens_ref.sum())

        # -- prior on dm2_ee (from JUNO's far reactors) --------------------------
        self.sigma_dm2ee_prior = sigma_dm2ee_prior

        # -- systematics ---------------------------------------------------------
        self.sigma_norm = sigma_norm
        self.sigma_uniformity = sigma_uniformity
        self.uniformity_corr_m = uniformity_corr_m
        self.sigma_energy = (sigma_scale, sigma_bias, sigma_res)
        self.sigma_u238 = sigma_u238
        self.sigma_evolution = sigma_evolution
        self._build()

    # -- oscillation --------------------------------------------------------------
    def _pee_bin(self, i: int, params: OscillationParameters) -> np.ndarray:
        """Survival probability per E_nu, averaged over the true-L distribution
        of L_reco bin i (theta13 term with dm2_ee, plus the solar term)."""

        e = self.e_nu_grid
        s2_13 = params.sin2_theta13
        s22_13 = 4.0 * s2_13 * (1.0 - s2_13)
        s22_12 = 4.0 * params.sin2_theta12 * (1.0 - params.sin2_theta12)
        out = np.zeros_like(e)
        for L, w in zip(self._l_true_nodes[i], self._l_true_w[i]):
            if w == 0.0:
                continue
            d_ee = 1.267 * params.dm2_ee * L / e
            d_21 = 1.267 * params.dm2_21 * L / e
            out += w * (1.0 - s22_13 * np.sin(d_ee) ** 2
                        - (1.0 - s2_13) ** 2 * s22_12 * np.sin(d_21) ** 2)
        return out

    def predict(self, params: OscillationParameters) -> np.ndarray:
        parts = []
        for i in range(self.n_l):
            dens = self._dens_ref * self.l_fraction[i] * self._pee_bin(i, params)
            parts.append(self._R @ dens)
        return np.concatenate(parts)

    # -- covariance ---------------------------------------------------------------
    def _build(self):
        mu = self.predict(self.truth)
        self.asimov = mu
        n_e = len(self.e_edges) - 1
        base = self._base
        modes = [self.sigma_norm * mu]
        # per-L-bin unoscillated-ish structures for the shape modes
        pieces = []
        for i in range(self.n_l):
            pieces.append(self._dens_ref * self.l_fraction[i]
                          * self._pee_bin(i, self.truth))

        def flux_mode(rel):
            return np.concatenate([self._R @ (p * rel) for p in pieces])

        modes.append(self.sigma_u238 * flux_mode(base._share_238_yield))
        modes.append(self.sigma_evolution * flux_mode(base._evo_rel_yield))
        for m in base._u235_modes:
            modes.append(flux_mode(m * base._share_235_yield))
        for k, w in enumerate(self.sigma_energy):
            modes.append(w * np.concatenate([self._dR[k] @ p for p in pieces]))
        # uniformity field over L
        if self.n_l > 1:
            cen = 0.5 * (self.l_edges[:-1] + self.l_edges[1:])
            corr = np.exp(-0.5 * (cen[:, None] - cen[None, :]) ** 2
                          / self.uniformity_corr_m**2)
            ev, evec = np.linalg.eigh(corr)
            keep = ev > 1e-8 * ev.max()
            fm = (evec[:, keep] * np.sqrt(ev[keep])).T
        else:
            fm = np.ones((1, 1))
        for f in fm:
            modes.append(self.sigma_uniformity * np.concatenate(
                [f[i] * (self._R @ pieces[i]) for i in range(self.n_l)]))
        V = np.array(modes)
        d_inv = 1.0 / np.maximum(mu, 1e-9)
        vd = V * d_inv[None, :]
        self._V, self._vd, self._d_inv = V, vd, d_inv
        self._m_cho = cho_factor(np.eye(len(modes)) + vd @ V.T)
        self.n_modes = len(modes)

    def _cinv(self, w):
        x = self._d_inv * w
        return x - self._vd.T @ cho_solve(self._m_cho, self._V @ x)

    # -- Fisher -------------------------------------------------------------------
    def fisher(self, params=("sin2_theta13", "dm2_ee")):
        from dataclasses import replace
        steps = {"sin2_theta13": 3e-4, "dm2_ee": 2e-5}
        d = []
        for name in params:
            h = steps[name]
            up = replace(self.truth, **{name: getattr(self.truth, name) + h})
            dn = replace(self.truth, **{name: getattr(self.truth, name) - h})
            d.append((self.predict(up) - self.predict(dn)) / (2 * h))
        d = np.array(d)
        F = d @ np.array([self._cinv(r) for r in d]).T
        if "dm2_ee" in params and self.sigma_dm2ee_prior:
            j = list(params).index("dm2_ee")
            F[j, j] += 1.0 / self.sigma_dm2ee_prior**2
        return np.linalg.inv(F)

    def sigma_theta13_deg(self) -> float:
        from .optimize import sin2theta13_error_to_deg
        cov = self.fisher()
        return sin2theta13_error_to_deg(float(np.sqrt(cov[0, 0])), self.truth)


# ---------------------------------------------------------------------------
# Two fixed reactors of identical design: near flux monitor + far physics unit
# ---------------------------------------------------------------------------
class TwoFixedReactorsTheta13:
    """Joint fit of two identical fixed reactors at D_near and D_far.

    Shared systematics (flux shape covariance, U238, fuel evolution, energy
    response) are coherent between the two sources; the near unit measures
    them with enormous statistics, so the far unit's L/E structure survives.
    Powers can differ; the flux normalisations are shared only through the
    reactor-power ratio prior ``sigma_power_ratio``.
    """

    def __init__(self, d_near_m: float, d_far_m: float, power_near_mw: float = 10.0,
                 power_far_mw: float = 10.0, sigma_power_ratio: float = 0.02,
                 sigma_dm2ee_prior: float | None = None, years: float = 3.0,
                 seed: int = 12345, **kw):
        self.near = FixedReactorTheta13(d_near_m, power_mwth=power_near_mw, years=years,
                                        seed=seed, **kw)
        self.far = FixedReactorTheta13(d_far_m, power_mwth=power_far_mw, years=years,
                                       seed=seed + 1, **kw)
        self.sigma_dm2ee_prior = sigma_dm2ee_prior
        self.sigma_power_ratio = sigma_power_ratio
        self.truth = self.near.truth
        self._build()

    def predict(self, params):
        return np.concatenate([self.near.predict(params), self.far.predict(params)])

    def _build(self):
        n, f = self.near, self.far
        mu = np.concatenate([n.asimov, f.asimov])
        zn, zf = np.zeros(n.asimov.size), np.zeros(f.asimov.size)
        # modes: [shared flux norm][power ratio][shared shape modes][own uniformity]
        modes = [np.concatenate([n.sigma_norm * n.asimov, f.sigma_norm * f.asimov]),
                 np.concatenate([zn, self.sigma_power_ratio * f.asimov])]
        # shape/energy modes are the same physical nuisances in both: recombine
        # per-mode vectors (both objects were built with identical priors)
        Vn, Vf = n._V[1:], f._V[1:]        # drop each object's own norm mode
        # the last n_l-ish rows of each are uniformity fields (uncorrelated
        # between the two units only insofar as they share the detector: same
        # detector -> SAME field, so keep them shared too)
        for a, b in zip(Vn, Vf):
            modes.append(np.concatenate([a, b]))
        V = np.array(modes)
        d_inv = 1.0 / np.maximum(mu, 1e-9)
        vd = V * d_inv[None, :]
        self._V, self._vd, self._d_inv = V, vd, d_inv
        self._m_cho = cho_factor(np.eye(len(modes)) + vd @ V.T)

    def _cinv(self, w):
        x = self._d_inv * w
        return x - self._vd.T @ cho_solve(self._m_cho, self._V @ x)

    def fisher(self, params=("sin2_theta13", "dm2_ee")):
        from dataclasses import replace
        steps = {"sin2_theta13": 3e-4, "dm2_ee": 2e-5}
        d = []
        for name in params:
            h = steps[name]
            up = replace(self.truth, **{name: getattr(self.truth, name) + h})
            dn = replace(self.truth, **{name: getattr(self.truth, name) - h})
            d.append((self.predict(up) - self.predict(dn)) / (2 * h))
        d = np.array(d)
        F = d @ np.array([self._cinv(r) for r in d]).T
        if "dm2_ee" in params and self.sigma_dm2ee_prior:
            j = list(params).index("dm2_ee")
            F[j, j] += 1.0 / self.sigma_dm2ee_prior**2
        return np.linalg.inv(F)

    def sigma_theta13_deg(self) -> float:
        from .optimize import sin2theta13_error_to_deg
        return sin2theta13_error_to_deg(float(np.sqrt(self.fisher()[0, 0])), self.truth)
