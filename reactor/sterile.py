"""Sterile-neutrino oscillations across the finite JUNO detector.

A reactor parked 50-100 m from the detector centre illuminates a target whose
fiducial radius (~16.5 m) is comparable to the standoff: the baseline varies by
a factor ~2 across the volume.  For Delta m^2_41 ~ 0.1-10 eV^2 the oscillation
length at reactor energies is metres to tens of metres, so the disappearance
pattern *wiggles across the detector* -- observable by binning events in the
reconstructed vertex baseline L, where no flux or cross-section systematic can
follow (the geometry is exact, and the source spectrum carries no
L-dependence).

Geometry
--------
For a point at distance D from the centre of a fiducial sphere of radius R,
the target volume between baselines L and L+dL is A(L) dL with

    A(L) = 2 pi L^2 [1 - (L^2 + D^2 - R^2) / (2 L D)],   |D-R| <= L <= D+R.

Baseline smearing: the high-dm2 wall
------------------------------------
Two effects smear the true oscillation baseline at fixed reconstructed L:

* the **finite core** -- a uniform ball of radius ``a`` has line-of-sight
  variance a^2/5;
* the **vertex resolution** -- sigma_vtx(E) = sigma_1 / sqrt(E[MeV])
  (JUNO-like, ~10 cm at 1 MeV).

For Gaussian-like smearing the oscillating term damps analytically,

    <sin^2(kL)> = 1/2 [1 - cos(2 k L) exp(-2 k^2 sigma_L^2)],
    sigma_L^2 = sigma_1^2 / E + a^2 / 5,      k = 1.267 dm2 / E [1/m],

which is what terminates the reach at high dm2 (bin-width averaging is handled
exactly by sub-nodes).  Counts are conserved by the smearing, so the null
spectra are unaffected.

Oscillation and statistics
--------------------------
The **full 3+1 vacuum** electron-antineutrino survival probability is used:

    P_ee = 1 - 4 sum_{i<j} |U_ei|^2 |U_ej|^2 sin^2(Delta_ij),

with |U_e4|^2 = sin^2 theta14 and |U_e1,2,3|^2 = cos^2 theta14 x (3-nu values).
Splitting off the sterile part,

    P_ee = P3(cos^4 th14 x standard) - sin^2(2 th14) sum_i f_i sin^2(Delta_i4),
    f_i = |U_ei|^2_{3nu}   (sum f_i = 1),   Delta_i4 = 1.267 (m4^2 - m_i^2) L / E,

so at baselines where Delta_31 or Delta_21 are not negligible (D ~ 1 km, or
dm2_41 <~ 1e-2 eV^2 where the sterile and atmospheric frequencies mix) the
three "sterile" phases Delta_14, Delta_24, Delta_34 are all carried, and so is
the theta13 disappearance itself (the null hypothesis is the 3-nu spectrum,
not the un-oscillated one).  The prediction remains linear in sin^2(2 th14) up
to O(sin^4 th14) corrections in the standard term (checked to be < 1e-3 of the
sterile term for s22 <= 0.1), so against the 3-nu Asimov

    Delta chi^2(dm2, s22) = s22^2 Q(dm2),  Q = W^T C^-1 W,

still holds and the exclusion curve is analytic.  C = diag(mu) + V^T V is
handled by the Woodbury identity, so fine binnings cost nothing.  Identical
for IBD (CC) and EvES (CC+NC): an oscillated nu_e is sterile.

Why the analysis is shape-only (and why it must be)
---------------------------------------------------
The Daya Bay U235 spectrum was *measured* at ~500 m: a sterile with
dm2_41 ~ 0.1-3 eV^2 would have depleted it by ~sin^2(2 th14)/2 in normalisation
(and left a <~1% E-tilt).  Using its normalisation as a prior would therefore
inject the very signal being searched for.  So the DEFAULT analysis leaves the
flux normalisation completely free (sigma_norm = 1e3) AND the reactor E-shape
completely free (``free_shape=True``: one unconstrained mode per E and per T
bin, coherent across L).  Only the L-direction wiggle structure at fixed E can
then carry sterile information -- and it carries essentially all of it: the
limits move by < 0.5% across 0.03-3 eV^2 relative to the Daya-Bay-prior
analysis (notebook 6).  The Daya Bay spectrum is used only as a *shape template*
for the event rates, never as a constraint.

Systematics
-----------
* flux normalisation -- FREE (see above); ``sigma_norm`` may be tightened for
  studies, and the notebook shows the limits are invariant to it;
* reactor E-shape -- FREE by default (``free_shape``); when off, the 25-mode
  measured Daya Bay U235 shape covariance and the U238 component (via
  :class:`~reactor.near_sm.FixedNearReactor`) act as E-direction priors,
  coherent in L;
* **fuel evolution**: the run-averaged HALEU burnup (default mid-cycle, Daya
  Bay trajectory) with a +-30% ingrowth uncertainty -- again E-direction only;
* detector response non-uniformity: a Gaussian-correlated field over L
  (default 0.5% amplitude, 1.5 m correlation length) -- the one systematic in
  the wiggle direction, modelled with a physical correlation length so that
  finer binning does not manufacture artificial high-frequency freedom;
* the IBD/EvES channel ratio (0.5%).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .constants import JUNO2025_TARGET_PROTONS, SECONDS_PER_DAY
from .cross_sections import vogel_beacom
from .detector import (
    DetectorResponse,
    EnergyResolution,
    TabulatedNonLinearity,
    gaussian_bin_response,
    integration_weights,
)
from .flux import haleu_fractions
from .oscillations import survival_probability_ee
from .near_sm import (
    ELECTRONS_PER_FREE_PROTON,
    FixedNearReactor,
    GA_SM,
    GV_SM,
    eves_dsigma_dT,
)

DAY = SECONDS_PER_DAY

from .solar import (  # noqa: F401  (re-exported for compatibility)
    PEP_E, PEP_FLUX, SOLAR_FLUXES, SOLAR_PRIOR, IBD_BACKGROUND_PRIORS,
    solar_flux_density, solar_pee, _load_solar_shape,
)


class SterileNearReactor:
    """Joint IBD + EvES sterile search with vertex-baseline binning."""

    def __init__(
        self,
        distance_m: float = 50.0,
        power_mwth: float = 10.0,
        mw_yr: float = 27.0,
        detector_radius_m: float = 16.5,
        l_bin_m: float = 0.5,
        n_sub: int = 5,
        include_eves: bool = True,
        l_binned: bool = True,
        # source and reconstruction
        core_radius_m: float = 0.5,
        sigma_vertex_m: float = 0.10,
        burnup: float = 0.5,
        # systematics.  DEFAULT = SHAPE-ONLY: free flux normalisation AND free
        # reactor E-shape, so nothing about the Daya Bay measurement (which
        # would itself carry a sterile depletion) enters the sensitivity.
        sigma_norm: float = 1.0e3,
        sigma_uniformity: float = 0.005,
        uniformity_corr_m: float = 1.5,
        sigma_channel_ratio: float = 0.005,
        sigma_u238: float = 0.15,
        sigma_evolution: float = 0.30,
        use_flux_covariance: bool = True,
        untagged_fraction: float = 0.01,
        extra_backgrounds: bool = True,
        ordering: int = +1,          # +1 normal, -1 inverted (for Delta_34)
        free_shape: bool = True,     # E-shape fully free (one mode per E bin, coherent in L)
        n_e_grid: int = 1200,
        e_edges: np.ndarray | None = None,
        t_edges: np.ndarray | None = None,
    ):
        self.distance_m = distance_m
        self.mw_yr = mw_yr
        self.detector_radius_m = detector_radius_m
        self.include_eves = include_eves
        self.core_radius_m = core_radius_m
        self.sigma_vertex_m = sigma_vertex_m
        self.ordering = ordering
        self.free_shape = free_shape

        base = FixedNearReactor(
            power_mwth=power_mwth, baseline_m=distance_m, burnup=burnup,
            untagged_fraction=untagged_fraction,
            use_flux_covariance=use_flux_covariance, n_e_grid=n_e_grid,
        )
        self._base = base
        enu = base.e_nu_grid
        self.e_nu_grid = enu
        w_e = integration_weights(enu)

        # 0.05 MeV matches the energy resolution (~77 keV at 4 MeV); 0.1 MeV
        # loses up to ~7% in reach at high dm2 (convergence shown in notebook 6)
        self.e_edges = (np.arange(1.0, 9.0 + 1e-9, 0.05)
                        if e_edges is None else e_edges)
        self.t_edges = (np.arange(1.0, 6.5 + 1e-9, 0.25)
                        if t_edges is None else t_edges)

        # -- responses ---------------------------------------------------------
        res = EnergyResolution(a=0.033, b=0.010, c=0.0)
        nl_p = TabulatedNonLinearity.from_release("positron")
        e_dep = DetectorResponse(use_ibd_recoil=True).deposited_energy(enu)
        e_vis = nl_p.visible_energy(e_dep)
        self._R_ibd = gaussian_bin_response(e_vis, self.e_edges, res.sigma(e_vis))
        R_ibd_t = gaussian_bin_response(e_vis, self.t_edges, res.sigma(e_vis))

        t_grid = base.t_grid
        w_t = integration_weights(t_grid)
        nl_e = TabulatedNonLinearity.from_release("electron")
        t_vis = nl_e.visible_energy(t_grid)
        R_t = gaussian_bin_response(t_vis, self.t_edges, res.sigma(t_vis))
        # collapse the EvES kernel once: (n_tbins x nE), so every later EvES
        # spectrum is a single small matmul
        K = eves_dsigma_dT(enu[None, :], t_grid[:, None], "e", True, GV_SM, GA_SM)
        self._R_t_w = R_t * w_t[None, :]
        self._A_eves = self._R_t_w @ K
        self._t_grid = t_grid

        # -- geometry ----------------------------------------------------------
        D, R = distance_m, detector_radius_m
        lo, hi = abs(D - R), D + R
        n_l = max(1, int(np.ceil((hi - lo) / l_bin_m))) if l_binned else 1
        self.l_edges = np.linspace(lo, hi, n_l + 1)
        self.n_l = n_l

        v_fid_cm3 = 4.0 / 3.0 * np.pi * R**3 * 1e6
        n_p_cm3 = base.efficiency * JUNO2025_TARGET_PROTONS / v_fid_cm3
        n_e_cm3 = n_p_cm3 * ELECTRONS_PER_FREE_PROTON
        t_live = mw_yr * base.seconds_per_mw_yr

        def cap(L):
            return np.clip(1.0 - (L**2 + D**2 - R**2) / (2.0 * L * D), 0.0, 2.0)

        self._l_nodes, self._c_ibd, self._c_eves = [], [], []
        self._vol_frac = np.zeros(n_l)
        for i in range(n_l):
            nodes = np.linspace(self.l_edges[i], self.l_edges[i + 1], n_sub + 1)
            nodes = 0.5 * (nodes[:-1] + nodes[1:])
            dl_cm = (self.l_edges[i + 1] - self.l_edges[i]) / n_sub * 100.0
            area_cm2 = 2.0 * np.pi * (nodes * 100.0) ** 2 * cap(nodes)
            coeff = (base.fission_rate / (4.0 * np.pi * (nodes * 100.0) ** 2)
                     * area_cm2 * dl_cm * t_live)
            self._l_nodes.append(nodes)
            self._c_ibd.append(coeff * n_p_cm3)
            self._c_eves.append(coeff * n_e_cm3)
            self._vol_frac[i] = float((area_cm2 * dl_cm).sum()) / v_fid_cm3

        # -- null spectra ------------------------------------------------------
        yld = base._yield_ibd * w_e
        flux_w = base._flux * w_e
        self._ibd_yld = yld
        self._eves_flux = flux_w

        # 3-nu survival probability, averaged over each L bin's sub-nodes:
        # the null hypothesis is the standard oscillated spectrum
        self._truth = base.truth
        self._p3_bin = []
        for i in range(n_l):
            p3 = np.zeros_like(enu)
            for L, c in zip(self._l_nodes[i], self._c_ibd[i]):
                p3 += c * survival_probability_ee(enu, L / 1000.0, base.truth)
            self._p3_bin.append(p3 / self._c_ibd[i].sum())

        self._ibd_null, self._eves_null, self.ibd_singles = [], [], []
        for i in range(n_l):
            g = float(self._c_ibd[i].sum())
            ge = float(self._c_eves[i].sum())
            p3 = self._p3_bin[i]
            self._ibd_null.append(self._R_ibd @ (yld * g * p3))
            if include_eves:
                self._eves_null.append(self._A_eves @ (flux_w * ge * p3))
                self.ibd_singles.append(untagged_fraction * (R_ibd_t @ (yld * g * p3)))
        self.null = self._stack(self._ibd_null,
                                [n + s for n, s in zip(self._eves_null,
                                                       self.ibd_singles)]
                                if include_eves else None)

        # -- additional backgrounds -------------------------------------------
        # IBD: the far-reactor signal (release Npred_signal, oscillations
        # included) plus every release background component, distributed over
        # L in proportion to *volume* -- unlike the near signal's A(L)/L^2.
        from .juno_data import load_spectrum

        rel = load_spectrum()
        self.days_calendar = t_live / DAY / base.duty_cycle
        e_cen = 0.5 * (self.e_edges[:-1] + self.e_edges[1:])
        e_wid = np.diff(self.e_edges)
        zero_rel = np.zeros_like(rel.pred_signal)
        if not extra_backgrounds:
            rel_scale = 0.0
        else:
            rel_scale = 1.0
        comps = {"far reactors": rel_scale * rel.pred_signal + zero_rel,
                 "geoneutrino": rel_scale * rel.backgrounds["geoneutrino"],
                 "9Li/8He": rel_scale * rel.backgrounds["9Li/8He"],
                 "world reactors": rel_scale * rel.backgrounds["world reactors"],
                 "214Bi-214Po": rel_scale * rel.backgrounds["214Bi-214Po"],
                 "other": rel_scale * rel.backgrounds["other"]}
        self.ibd_backgrounds = {}
        for name, counts in comps.items():
            dens = counts / rel.widths / rel.livetime_days        # /MeV/day
            shape = (np.interp(e_cen, rel.centers, dens, left=0.0, right=0.0)
                     * e_wid * self.days_calendar)
            self.ibd_backgrounds[name] = [shape * self._vol_frac[i]
                                          for i in range(n_l)]

        # EvES: solar neutrinos (continua from reactor/data/solar + pep line),
        # nu_e/nu_mu-weighted by the adiabatic LMA survival probability.
        self.solar_eves = None
        if include_eves and not extra_backgrounds:
            self.solar_eves = [np.zeros(len(self.t_edges) - 1)
                               for _ in range(n_l)]
        elif include_eves:
            par = base.truth
            es = np.linspace(1.0, 18.0, 900)
            ws = integration_weights(es)
            phi = solar_flux_density(es)
            pee_s = solar_pee(es, par)
            k_e = eves_dsigma_dT(es[None, :], self._t_grid[:, None],
                                 "e", False, GV_SM, GA_SM)
            k_m = eves_dsigma_dT(es[None, :], self._t_grid[:, None],
                                 "mu", False, GV_SM, GA_SM)
            spec_t = k_e @ (phi * pee_s * ws) + k_m @ (phi * (1 - pee_s) * ws)
            p0 = float(solar_pee(np.array([PEP_E]), par)[0])
            spec_t = spec_t + PEP_FLUX * (
                p0 * eves_dsigma_dT(PEP_E, self._t_grid, "e", False, GV_SM, GA_SM)
                + (1 - p0) * eves_dsigma_dT(PEP_E, self._t_grid, "mu", False,
                                            GV_SM, GA_SM))
            n_e_total = n_e_cm3 * v_fid_cm3
            binned = (self._R_t_w @ spec_t) * n_e_total * self.days_calendar * DAY
            self.solar_eves = [binned * self._vol_frac[i] for i in range(n_l)]

        bkg_ib_tot = [sum(self.ibd_backgrounds[nm][i]
                          for nm in self.ibd_backgrounds) for i in range(n_l)]
        bkg_ev_tot = (list(self.solar_eves) if include_eves else None)
        self.background_extra = self._stack(bkg_ib_tot, bkg_ev_tot)
        self.asimov = self.null + self.background_extra
        self._bkg_ib_tot = bkg_ib_tot

        # -- systematic modes --------------------------------------------------
        modes = [sigma_norm * self.null]
        n_ib = len(self.e_edges) - 1
        n_ev = len(self.t_edges) - 1

        # background normalisation priors, coherent across L
        for name, per_l in self.ibd_backgrounds.items():
            ev0 = [np.zeros(n_ev)] * n_l if include_eves else None
            modes.append(IBD_BACKGROUND_PRIORS[name] * self._stack(per_l, ev0))
        if include_eves:
            modes.append(SOLAR_PRIOR * self._stack([np.zeros(n_ib)] * n_l,
                                                   self.solar_eves))
        if include_eves:
            modes.append(sigma_channel_ratio
                         * self._stack([np.zeros(n_ib)] * n_l, self._eves_null))

        # E-direction distortions, coherent across L: flux shape, U238, evolution
        sigma_ibd = vogel_beacom(enu, order=1)
        f = base.fractions
        # d(spectrum)/d(Pu-ingrowth scale), both channels
        evo_flux = (f["Pu239"] * (base._flux_239 - base._flux_235)
                    + f["Pu241"] * (base._flux_241 - base._flux_235))
        evo_yield = evo_flux * sigma_ibd
        rel_pairs = [(base._share_238_yield, base._share_238_flux, sigma_u238),
                     (evo_yield / np.maximum(base._yield_ibd, 1e-99),
                      evo_flux / np.maximum(base._flux, 1e-99), sigma_evolution)]
        rel_pairs += [(m * base._share_235_yield, m * base._share_235_flux, 1.0)
                      for m in base._u235_modes]
        for rel_y, rel_f, width in rel_pairs:
            ib = [self._R_ibd @ (yld * rel_y * float(self._c_ibd[i].sum()))
                  for i in range(n_l)]
            ev = ([self._A_eves @ (flux_w * rel_f * float(self._c_eves[i].sum()))
                   for i in range(n_l)] if include_eves else None)
            modes.append(width * self._stack(ib, ev))

        # optional: the reactor E-shape completely free -- one 100% mode per
        # reconstructed-E bin (IBD) and per T bin (EvES), coherent across all
        # L bins.  Then ONLY the L-direction structure at fixed E carries any
        # sterile information: the analysis is independent of the source
        # spectrum shape as well as its normalisation.
        if free_shape:
            for k in range(n_ib):
                ib = [np.where(np.arange(n_ib) == k, self._ibd_null[i], 0.0)
                      for i in range(n_l)]
                ev = ([np.zeros(n_ev)] * n_l) if include_eves else None
                modes.append(self._stack(ib, ev))
            if include_eves:
                for k in range(n_ev):
                    ib = [np.zeros(n_ib)] * n_l
                    ev = [np.where(np.arange(n_ev) == k, self._eves_null[i], 0.0)
                          for i in range(n_l)]
                    modes.append(self._stack(ib, ev))

        # detector non-uniformity: Gaussian-correlated field over L
        centers = 0.5 * (self.l_edges[:-1] + self.l_edges[1:])
        if n_l > 1:
            corr = np.exp(-0.5 * (centers[:, None] - centers[None, :]) ** 2
                          / max(uniformity_corr_m, 1e-3) ** 2)
            evals, evecs = np.linalg.eigh(corr)
            keep = evals > 1e-8 * evals.max()
            field_modes = (evecs[:, keep] * np.sqrt(evals[keep])).T
        else:
            field_modes = np.ones((1, 1))
        for fm in field_modes:
            ib = [fm[i] * (self._ibd_null[i] + bkg_ib_tot[i]) for i in range(n_l)]
            ev = ([fm[i] * (self._eves_null[i] + self.ibd_singles[i]
                            + self.solar_eves[i])
                   for i in range(n_l)] if include_eves else None)
            modes.append(sigma_uniformity * self._stack(ib, ev))

        # -- Woodbury pieces ---------------------------------------------------
        self._V = np.array(modes)
        self._d_inv = 1.0 / np.maximum(self.asimov, 1e-9)
        vd = self._V * self._d_inv[None, :]
        m_small = np.eye(len(modes)) + vd @ self._V.T
        self._m_cho = cho_factor(m_small)
        self._vd = vd
        self.n_modes = len(modes)

    # -- helpers ----------------------------------------------------------------
    def _stack(self, ibd_list, eves_list):
        parts = list(ibd_list)
        if self.include_eves and eves_list is not None:
            parts += list(eves_list)
        return np.concatenate(parts)

    def _cinv(self, w: np.ndarray) -> np.ndarray:
        """C^-1 w via the Woodbury identity."""

        x = self._d_inv * w
        y = self._V @ x
        return x - self._vd.T @ cho_solve(self._m_cho, y)

    def _damping(self, dm2: float) -> np.ndarray:
        """exp(-2 k^2 sigma_L^2) per neutrino energy."""

        k = 1.267 * dm2 / self.e_nu_grid
        sigma2 = self.sigma_vertex_m**2 / self.e_nu_grid \
            + self.core_radius_m**2 / 5.0
        return np.exp(-2.0 * k**2 * sigma2)

    # -- oscillation ------------------------------------------------------------
    def _sterile_phases(self, dm2_41: float):
        """(dm2_i4, f_i) for i = 1, 2, 3: the three sterile splittings and the
        3-nu electron-flavour weights |U_ei|^2 that multiply sin^2(Delta_i4)."""

        p = self._truth
        s12, s13 = p.sin2_theta12, p.sin2_theta13
        f1 = (1.0 - s12) * (1.0 - s13)
        f2 = s12 * (1.0 - s13)
        f3 = s13
        # m4^2 - m1^2 = dm2_41 ; m4^2 - m2^2 = dm2_41 - dm2_21 ;
        # m4^2 - m3^2 = dm2_41 - dm2_31.  From the definition
        # dm2_ee = c12^2 dm2_31 + s12^2 dm2_32  =>  dm2_31 = dm2_ee + s12^2 dm2_21
        # (normal ordering; inverted: dm2_31 = -dm2_ee + s12^2 dm2_21 -- checked
        # against the module's exact 3-nu P_ee to 1e-5).
        dm2_31 = self.ordering * p.dm2_ee + s12 * p.dm2_21
        return [(dm2_41, f1), (dm2_41 - p.dm2_21, f2), (dm2_41 - dm2_31, f3)]

    def wiggle_template(self, dm2: float) -> np.ndarray:
        """d(prediction)/d(sin^2 2theta14): the sterile deficit at s22 = 1,
        full 3+1 -- all three Delta_i4 phases, weighted by |U_ei|^2."""

        e = self.e_nu_grid
        ib, ev = [], []
        terms = self._sterile_phases(dm2)
        for i in range(self.n_l):
            s2_i = np.zeros_like(e)
            s2_e = np.zeros_like(e)
            ci, ce = self._c_ibd[i], self._c_eves[i]
            for j, L in enumerate(self._l_nodes[i]):
                osc = np.zeros_like(e)
                for dm2_i4, f_i in terms:
                    damp = self._damping(abs(dm2_i4))
                    k2 = 2.0 * 1.267 * dm2_i4 / e
                    osc += f_i * 0.5 * (1.0 - np.cos(k2 * L) * damp)
                s2_i += ci[j] * osc
                s2_e += ce[j] * osc
            ib.append(self._R_ibd @ (self._ibd_yld * s2_i))
            if self.include_eves:
                ev.append(self._A_eves @ (self._eves_flux * s2_e))
        return self._stack(ib, ev if self.include_eves else None)

    def survival_4nu(self, e_mev, L_m, dm2_41: float, s22_14: float) -> np.ndarray:
        """Exact 3+1 vacuum P_ee (no smearing) -- for validation and plots."""

        e = np.asarray(e_mev, dtype=float)
        p = self._truth
        s14 = 0.5 * (1.0 - np.sqrt(1.0 - s22_14))      # sin^2 theta14
        p3 = survival_probability_ee(e, L_m / 1000.0, p)
        # exact: with |U_e4|^2 = s14 and |U_ei|^2 = (1 - s14) f_i,
        #   P = 1 - (1-s14)^2 (1 - P3) - 4 s14 (1-s14) sum_i f_i sin^2 D_i4
        ster = np.zeros_like(e)
        for dm2_i4, f_i in self._sterile_phases(dm2_41):
            ster += f_i * np.sin(1.267 * dm2_i4 * L_m / e) ** 2
        return 1.0 - (1.0 - s14) ** 2 * (1.0 - p3) - 4.0 * s14 * (1.0 - s14) * ster

    # -- statistics -------------------------------------------------------------
    def q_value(self, dm2: float) -> float:
        w = self.wiggle_template(dm2)
        return float(w @ self._cinv(w))

    def limit(self, dm2, delta_chi2: float = 5.99) -> float:
        q = self.q_value(dm2)
        return float(np.sqrt(delta_chi2 / max(q, 1e-30))) if q > 0 else np.inf

    def limit_curve(self, dm2_grid, delta_chi2: float = 5.99) -> np.ndarray:
        return np.array([self.limit(d, delta_chi2) for d in np.atleast_1d(dm2_grid)])

    # -- diagnostics ------------------------------------------------------------
    def rates_per_day(self) -> dict:
        t = self.mw_yr * self._base.seconds_per_mw_yr
        out = {"IBD / day": sum(x.sum() for x in self._ibd_null) / t * DAY,
               "L bins": self.n_l,
               "L range [m]": (float(self.l_edges[0]), float(self.l_edges[-1])),
               "modes": self.n_modes}
        if self.include_eves:
            out["EvES / day"] = sum(x.sum() for x in self._eves_null) / t * DAY
            out["solar EvES / day"] = (sum(x.sum() for x in self.solar_eves)
                                       / self.days_calendar)
        out["IBD backgrounds / day"] = {
            name: sum(x.sum() for x in per) / self.days_calendar
            for name, per in self.ibd_backgrounds.items()}
        return out

    def ratio_vs_L(self, dm2: float, s22: float, e_lo: float, e_hi: float):
        """Oscillated/unoscillated IBD ratio per L bin, in an E_rec slice."""

        e_cen = 0.5 * (self.e_edges[:-1] + self.e_edges[1:])
        sel = (e_cen >= e_lo) & (e_cen < e_hi)
        centers = 0.5 * (self.l_edges[:-1] + self.l_edges[1:])
        w = self.wiggle_template(dm2)
        n_ib = len(self.e_edges) - 1
        out = []
        for i in range(self.n_l):
            null_i = self._ibd_null[i][sel].sum()
            wig_i = w[i * n_ib:(i + 1) * n_ib][sel].sum()
            out.append(1.0 - s22 * wig_i / max(null_i, 1e-30))
        return centers, np.array(out)
