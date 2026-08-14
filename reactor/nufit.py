"""Reproduction of NuFit's JUNO analysis (Esteban et al., arXiv:2601.09791v2).

Implements their prescription as literally as the paper allows, so it can be
compared against :class:`~reactor.juno_fit.JUNO2025Model` ingredient by
ingredient.

Signal (their Eq. 2.1)
----------------------
Nine reactors -- the eight Yangjiang/Taishan cores plus one effective Daya Bay
core at 215 km, from Tab. 2 of the JUNO design report -- weighted by P/4pi L^2.
The IBD cross section is Vogel-Beacom by default (their Ref. [18]); they checked
Strumia-Vissani gives "very similar results", and both are available here.

Reactor flux (their Appendix A)
-------------------------------
The Daya Bay unfolded ``Total`` spectrum, at the *Daya Bay average* fission
fractions 0.564 : 0.076 : 0.304 : 0.056 with no isotope correction.  The 25
bin-averaged values Phi0_i are turned into a continuous shape

    phi0(E) sigma(E) = phi_huber(E) sigma(E) * sum_n y0_n delta_n(E)

where delta_n are cardinal cubic interpolation polynomials and the coefficients
solve  sum_n M_in y0_n = Phi0_i  with  M_in = <phi_huber sigma delta_n>_i  --
i.e. the interpolant reproduces the measured *bin averages* exactly, which
centre-value interpolation does not.  The 25x25 covariance (from the released
75x75, keeping the Total block) is eigendecomposed, Psi = O D, and each column
is pushed through the same M^-1 to give 25 unit-Gaussian flux pulls.

Bin-per-bin rescaling
---------------------
Their cnf 1 is defined *after* rescaling the prediction bin by bin so that the
un-oscillated spectrum matches JUNO's own (their Fig. 1, right panel).  JUNO's
un-oscillated prediction is recovered from the release as
(data - background) / Pee_meas.  This forces flux x response to agree with
JUNO's in every bin; the flux model then enters only through the within-bin
oscillation weighting.

Systematics (their Eqs. 2.5-2.7 and Tab. 1)
-------------------------------------------
    xi_norm   1.8% (Tab. 2 rates; the note-added config uses 2.4%)
    xi_bg     33 / 42 / 10 / 56 / 100 %  (LiHe, Geo, world-reac, BiPo, other)
    xi_LiHe,2 9Li/8He shape, 20% at 1 MeV, linear in energy
    xi_scl    0.5%   energy scale;  xi_bias 0.5% (5% in cnf 4)
    xi_res    5%     resolution    (40% in cnf 6)
    25 flux pulls, unit width
    r_BG      rescales every background EXCEPT geoneutrinos (1.15 in cnf 2)

Resolution sigma(E) = E sqrt(a^2/E + b^2) with a = 3.3%, b = 1%; the positron
non-linearity curve from the release.  Neutron recoil enters through the mean
positron energy at O(1/M); their top-hat refinement (Capozzi et al.) would add
(width^2/12) ~ 44 keV^2 to a resolution variance of ~4500 keV^2 at 3 MeV, a
sub-percent change in sigma, and is not implemented.

Efficiency: at fixed oscillation parameters the prediction is affine in all
3 + 25 linear pulls, so a basis (signal + one column per pull) is built once
per grid point and each likelihood call is a 28-term dot product in bin space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    JUNO2025_DM2_21,
    JUNO2025_EFFICIENCY_TOTAL,
    JUNO2025_NONOSC_CPD,
    JUNO2025_SIN2_THETA12,
    JUNO2025_TARGET_PROTONS,
    SECONDS_PER_DAY,
    OscillationParameters,
)
from .cross_sections import load_ibd_cross_section, vogel_beacom
from .detector import (
    DetectorResponse,
    EnergyResolution,
    TabulatedNonLinearity,
    gaussian_bin_response,
    integration_weights,
)
from .flux import default_juno_cores, mixed_spectrum_per_fission
from .juno_data import JUNOSpectrum, load_spectrum, load_survival_probability
from .oscillations import survival_probability_ee, survival_probability_matter
from .statistics import chi2_cnp, chi2_poisson

#: Their background normalisation priors (Sec. 2); ``other`` is 100%, against
#: the 50% our own model assumes.
NUFIT_BACKGROUND_PRIORS = {
    "9Li/8He": 0.33,
    "geoneutrino": 0.42,
    "world reactors": 0.10,
    "214Bi-214Po": 0.56,
    "other": 1.00,
}

#: Backgrounds that ``r_BG`` rescales -- everything except the geoneutrinos.
RESCALED_BACKGROUNDS = ("9Li/8He", "world reactors", "214Bi-214Po", "other")

#: The fission fractions the Daya Bay measurement corresponds to; NuFit use
#: these for all reactors, with no isotope correction (their Appendix A).
DAYABAY_AVG_FRACTIONS = {"U235": 0.564, "U238": 0.076, "Pu239": 0.304, "Pu241": 0.056}


@dataclass(frozen=True)
class NuFitConfig:
    """One column of their Tab. 1."""

    name: str = "cnf 2"
    r_bg: float = 1.15
    r_nl: float = 1.0
    sigma_bias: float = 0.005
    r_res: float = 1.0
    sigma_res: float = 0.05
    likelihood: str = "poisson"

    sigma_norm: float = 0.018
    """1.8% for every Tab. 1 configuration; 2.4% only in the note-added one."""
    sigma_scale: float = 0.005
    lihe_shape: float = 0.20
    """Width of the 9Li/8He shape pull at 1 MeV; the distortion grows as xi * E."""

    use_lihe_shape: bool = True
    use_energy_pulls: bool = True
    """Ablation switches.  These fix the pull at zero rather than shrinking its
    prior: a near-zero width makes the penalty gradient ~1e12 and L-BFGS-B
    terminates at the start point, freezing every *other* nuisance too."""


#: Their Tab. 1.  cnf 2 is their default.  "cnf upd" is the note-added
#: configuration: sigma_norm raised to 2.4% by the selection efficiency, and no
#: background rescaling.
CONFIGURATIONS = {
    "cnf 1": NuFitConfig("cnf 1", r_bg=1.00, likelihood="cnp"),
    "cnf 2": NuFitConfig("cnf 2", r_bg=1.15, likelihood="poisson"),
    "cnf 3": NuFitConfig("cnf 3", r_bg=1.15, r_nl=1.024, likelihood="poisson"),
    "cnf 4": NuFitConfig("cnf 4", r_bg=1.15, sigma_bias=0.05, likelihood="poisson"),
    "cnf 5": NuFitConfig("cnf 5", r_bg=1.00, r_res=1.3, likelihood="poisson"),
    "cnf 6": NuFitConfig("cnf 6", r_bg=1.15, sigma_res=0.40, likelihood="poisson"),
    "cnf upd": NuFitConfig("cnf upd", r_bg=1.00, sigma_norm=0.024, likelihood="poisson"),
    # The repository's standard: CNP (JUNO's own statistic) with the full
    # documented rate budget -- Tab. 2 rates (1.8%) + selection efficiency
    # (1.6%) in quadrature, the note-added correction -- and no rescaling.
    "standard": NuFitConfig("standard", r_bg=1.00, sigma_norm=0.024, likelihood="cnp"),
}


def standard_juno_fit(**overrides) -> "NuFitJUNOModel":
    """The repository's standard fit to the JUNO 2025 release.

    The NuFit prescription of notebook 4 -- nine-reactor signal, Appendix-A
    flux with its 25 pulls, bin-per-bin rescaling to JUNO's own un-oscillated
    spectrum, Vogel-Beacom, the full nuisance set, no background rescaling --
    with the CNP statistic JUNO themselves use and the *full documented* rate
    budget: sigma_norm = 2.4% = 1.8% (their Tab. 2 rate systematics) + 1.6%
    (selection efficiency, their Tab. 1) in quadrature, the correction NuFit's
    note added makes to their own Tab. 1 configurations.  Choosing the 1.8% of
    their cnf 1 instead would understate a documented uncertainty.

    Keyword arguments override any :class:`NuFitJUNOModel` field.
    """

    kwargs = {"config": CONFIGURATIONS["standard"]}
    kwargs.update(overrides)
    return NuFitJUNOModel(**kwargs)


@dataclass
class NuFitJUNOModel:
    """JUNO 2025 spectrum fit in NuFit's parameterisation."""

    config: NuFitConfig = field(default_factory=lambda: CONFIGURATIONS["cnf 2"])
    spectrum: JUNOSpectrum = field(default_factory=load_spectrum)
    e_nu_grid: np.ndarray = field(default_factory=lambda: np.linspace(1.806, 13.0, 2600))
    sin2_theta13: float = 0.021748
    dm2_ee: float = 2.466e-3
    matter_density: float = 2.55
    use_matter: bool = True
    include_distant_cores: bool = True

    fangchenggang: str = "background"
    """Where the Fangchenggang plant (12.1 GW_th at 411.7 km) lives.

    ``'background'`` (default) leaves it inside the released ``world reactors``
    background, which is JUNO's own bookkeeping -- its ~26 predicted events fit
    within that 51.9-event component (notebook 3, section 9.4).  ``'signal'``
    adds it to the reactor signal as NuFit's note added does, *and* removes its
    predicted oscillated events from the world-reactors background so it is not
    counted twice.
    """

    flux_model: str = "nufit"
    """``'nufit'`` -- their Appendix A construction (default);
    ``'dayabay'`` -- our centre-interpolated yield; ``'huber-mueller'``."""

    use_flux_pulls: bool = True
    bin_scaling: bool = True
    """Rescale the reactor prediction bin per bin to JUNO's un-oscillated
    spectrum, which is how their cnf 1 is defined."""

    cross_section: str = "vogel-beacom"
    """``'vogel-beacom'`` (their Ref. [18]) or ``'strumia-vissani'``."""

    resolution_a: float = 0.033
    resolution_b: float = 0.010

    def __post_init__(self) -> None:
        s = self.spectrum
        enu = self.e_nu_grid
        weights = integration_weights(enu)
        if self.cross_section == "vogel-beacom":
            xsec_fn = lambda e: vogel_beacom(e, order=1)
        else:
            xsec_fn = load_ibd_cross_section()
        xsec = xsec_fn(enu)

        exposure = (
            JUNO2025_TARGET_PROTONS
            * JUNO2025_EFFICIENCY_TOTAL
            * s.livetime_days
            * SECONDS_PER_DAY
        )
        self._cores = list(default_juno_cores(
            duty_cycle=1.0, include_distant=self.include_distant_cores
        ))
        self._fcg_index = None
        if self.fangchenggang == "signal":
            from .flux import ReactorCore, juno_average_fractions

            self._cores.append(
                ReactorCore("FCG", 12.1, 411.7, juno_average_fractions, 1.0, False)
            )
            self._fcg_index = len(self._cores) - 1
        elif self.fangchenggang != "background":
            raise KeyError(f"fangchenggang must be 'background' or 'signal', "
                           f"got {self.fangchenggang!r}")
        self.n_flux_modes = 0
        self._flux_modes = np.zeros((0, enu.size))

        from .constants import CM_PER_KM

        if self.flux_model == "nufit":
            S0, modes_rel = self._appendix_a_flux(enu, xsec_fn)
            self._density = [
                c.fission_rate() * S0 / (4.0 * np.pi * (c.baseline_km * CM_PER_KM) ** 2)
                * weights * exposure
                for c in self._cores
            ]
            if self.use_flux_pulls:
                self._flux_modes = modes_rel
                self.n_flux_modes = int(modes_rel.shape[0])
        elif self.flux_model == "dayabay":
            from .dayabay_data import dayabay_yield_model

            yield_values = dayabay_yield_model(fractions=None)(enu)
            self._density = [
                c.fission_rate() * yield_values
                / (4.0 * np.pi * (c.baseline_km * CM_PER_KM) ** 2)
                * weights * exposure
                for c in self._cores
            ]
            if self.use_flux_pulls:
                self._build_interp_flux_modes()
        elif self.flux_model == "huber-mueller":
            self._density = [
                c.flux_at_detector(enu) * xsec * weights * exposure for c in self._cores
            ]
        else:
            raise KeyError(f"unknown flux_model {self.flux_model!r}")
        self._baselines = [c.baseline_km for c in self._cores]

        # -- response and its derivatives with respect to the energy pulls ---
        self._nl = TabulatedNonLinearity.from_release("positron")
        self._resolution = EnergyResolution(a=self.resolution_a, b=self.resolution_b, c=0.0)
        self._e_dep = DetectorResponse(use_ibd_recoil=True).deposited_energy(enu)

        h = 1.0e-3
        base = self._response(0.0, 0.0, 0.0)
        self._R0 = base
        self._dR = np.stack(
            [
                (self._response(h, 0.0, 0.0) - self._response(-h, 0.0, 0.0)) / (2 * h),
                (self._response(0.0, h, 0.0) - self._response(0.0, -h, 0.0)) / (2 * h),
                (self._response(0.0, 0.0, h) - self._response(0.0, 0.0, -h)) / (2 * h),
            ]
        )

        unoscillated = float((base @ np.sum(self._density, axis=0)).sum())
        target = JUNO2025_NONOSC_CPD * JUNO2025_EFFICIENCY_TOTAL * s.livetime_days
        self.normalisation = target / unoscillated

        # -- bin-per-bin rescaling to JUNO's own un-oscillated spectrum ------
        ours_unosc = self.normalisation * (base @ np.sum(self._density, axis=0))
        self._bin_scale = np.ones(len(s.n_obs))
        if self.bin_scaling:
            _, _, pee, _ = load_survival_probability()
            ok = np.abs(pee) > 1e-6
            theirs = np.where(
                ok, (s.n_obs - s.background_total) / np.where(ok, pee, 1.0), np.nan
            )
            good = np.isfinite(theirs) & (theirs > 0) & (ours_unosc > 1e-6)
            self._bin_scale = np.where(
                good, theirs / np.maximum(ours_unosc, 1e-30), 1.0
            )

        self._bkg_names = list(s.backgrounds)
        self._bkg = np.array([s.backgrounds[k] for k in self._bkg_names])
        self._bkg_prior = np.array([NUFIT_BACKGROUND_PRIORS[k] for k in self._bkg_names])
        self._rescaled = np.array(
            [1.0 if k in RESCALED_BACKGROUNDS else 0.0 for k in self._bkg_names]
        )
        self._lihe = self._bkg_names.index("9Li/8He")

        if self._fcg_index is not None:
            # Remove Fangchenggang's predicted oscillated events from the
            # world-reactors background: it is now in the signal instead.
            par = self.parameters(JUNO2025_SIN2_THETA12, JUNO2025_DM2_21)
            if self.use_matter:
                p_fcg = survival_probability_matter(
                    enu, 411.7, par, self.matter_density
                )
            else:
                p_fcg = survival_probability_ee(enu, 411.7, par)
            fcg_pred = self.normalisation * (
                self._R0 @ (self._density[self._fcg_index] * p_fcg)
            )
            self._bkg = self._bkg.copy()
            i_wr = self._bkg_names.index("world reactors")
            self._bkg[i_wr] = np.maximum(self._bkg[i_wr] - fcg_pred, 0.0)

    # -- their Appendix A ----------------------------------------------------
    def _appendix_a_flux(self, enu, xsec_fn):
        """Continuous flux and 25 pull modes from the Daya Bay release."""

        from scipy.interpolate import CubicSpline

        from .dayabay_data import load_unfolded

        spectra, cov75, _ = load_unfolded()
        tot = spectra["Total"]
        edges, vals = tot.edges, tot.values
        n = len(vals)
        cov = np.asarray(cov75)[2 * n :, 2 * n :] * 1e-86  # released as (1e-43)^2

        nodes = tot.centers
        basis = [CubicSpline(nodes, np.eye(n)[k]) for k in range(n)]

        # M_in = <phi_huber sigma delta_n>_i on a dedicated 1 keV grid
        fine = np.linspace(edges[0], edges[-1], 7701)
        step = fine[1] - fine[0]
        ph_fine = mixed_spectrum_per_fission(fine, DAYABAY_AVG_FRACTIONS) * xsec_fn(fine)
        d_fine = np.array([b(fine) for b in basis])
        idx = np.clip(np.searchsorted(edges, fine, side="right") - 1, 0, n - 1)
        widths = np.diff(edges)
        M = np.zeros((n, n))
        for i in range(n):
            m = idx == i
            M[i] = (ph_fine[m] * d_fine[:, m]).sum(axis=1) * step / widths[i]

        y0 = np.linalg.solve(M, vals)
        evals, evecs = np.linalg.eigh(cov)
        psi = evecs * np.sqrt(np.maximum(evals, 0.0))       # columns are modes
        y_modes = np.linalg.solve(M, psi)                   # (n, n_modes)

        self.appendix_a_residual = float(np.max(np.abs(M @ y0 - vals) / vals))

        # evaluate on the integration grid; hold the modulation constant
        # outside the measured range (only the >9.5 MeV tail is affected)
        e_eval = np.clip(enu, edges[0], edges[-1])
        d_enu = np.array([b(e_eval) for b in basis])        # (n, n_enu)
        g0 = y0 @ d_enu
        gk = y_modes.T @ d_enu                              # (n_modes, n_enu)
        ph = mixed_spectrum_per_fission(enu, DAYABAY_AVG_FRACTIONS) * xsec_fn(enu)
        S0 = np.where(enu >= edges[0], ph * np.maximum(g0, 0.0), 0.0)
        safe = np.abs(g0) > 1e-12
        modes_rel = np.divide(gk, np.where(safe, g0, 1.0),
                              out=np.zeros_like(gk), where=safe)
        return S0, modes_rel

    def _build_interp_flux_modes(self) -> None:
        """Flux pulls for the centre-interpolated model (finite differences)."""

        from .dayabay_data import dayabay_yield_model, load_unfolded

        spectra, cov75, _ = load_unfolded()
        values = spectra["Total"].values
        n = len(values)
        cov = np.asarray(cov75)[2 * n :, 2 * n :] * 1e-86
        rel = cov / np.outer(values, values)

        base = dayabay_yield_model(fractions=None)(self.e_nu_grid)
        safe = np.where(base > 0, base, 1.0)
        step = 0.02
        jac = np.zeros((n, self.e_nu_grid.size))
        for k in range(n):
            scales = np.ones(n)
            scales[k] = 1.0 + step
            perturbed = dayabay_yield_model(fractions=None, bin_scales=scales)(self.e_nu_grid)
            jac[k] = (perturbed - base) / (safe * step)

        evals, evecs = np.linalg.eigh(rel)
        keep = evals > 1e-12 * max(evals.max(), 1e-300)
        modes = (evecs[:, keep] * np.sqrt(evals[keep])).T
        self._flux_modes = modes @ jac
        self.n_flux_modes = int(self._flux_modes.shape[0])

    # -- response ------------------------------------------------------------
    def _response(self, xi_scl: float, xi_bias: float, xi_res: float) -> np.ndarray:
        """Their Eq. (2.6): Et_pr = E_pr r_nl [(1+xi_scl) F_nl(E_pr) + xi_bias]."""

        cfg = self.config
        e_vis = self._e_dep * cfg.r_nl * (
            (1.0 + xi_scl) * self._nl.factor(self._e_dep) + xi_bias
        )
        sigma = (1.0 + xi_res) * cfg.r_res * self._resolution.sigma(e_vis)
        return gaussian_bin_response(e_vis, self.spectrum.edges, sigma)

    def parameters(self, sin2_theta12: float, dm2_21: float) -> OscillationParameters:
        return OscillationParameters(
            sin2_theta12=float(sin2_theta12),
            dm2_21=float(dm2_21),
            sin2_theta13=self.sin2_theta13,
            dm2_ee=self.dm2_ee,
        )

    def _flux_vector(self, sin2_theta12, dm2_21, flux_pulls=None) -> np.ndarray:
        params = self.parameters(sin2_theta12, dm2_21)
        total = np.zeros_like(self.e_nu_grid)
        for baseline, density in zip(self._baselines, self._density):
            if self.use_matter:
                p = survival_probability_matter(
                    self.e_nu_grid, baseline, params, self.matter_density
                )
            else:
                p = survival_probability_ee(self.e_nu_grid, baseline, params)
            total += density * p
        if flux_pulls is not None and self.n_flux_modes:
            total = total * (1.0 + np.asarray(flux_pulls, dtype=float) @ self._flux_modes)
        return total

    def signal(self, sin2_theta12, dm2_21, energy_pulls=(0.0, 0.0, 0.0),
               flux_pulls=None) -> np.ndarray:
        flux = self._flux_vector(sin2_theta12, dm2_21, flux_pulls)
        matrix = self._R0 + np.tensordot(np.asarray(energy_pulls, dtype=float),
                                         self._dR, axes=1)
        return self._bin_scale * self.normalisation * (matrix @ flux)

    def unoscillated(self) -> np.ndarray:
        return self._bin_scale * self.normalisation * (
            self._R0 @ np.sum(self._density, axis=0)
        )

    # -- backgrounds ---------------------------------------------------------
    def background(self, bkg_pulls=None, lihe_shape: float = 0.0) -> np.ndarray:
        """Their Eq. (2.5), including r_BG and the 9Li/8He linear shape pull."""

        n = len(self._bkg_names)
        pulls = np.zeros(n) if bkg_pulls is None else np.asarray(bkg_pulls, dtype=float)
        scale = (1.0 + pulls) * np.where(self._rescaled > 0, self.config.r_bg, 1.0)
        out = scale[:, None] * self._bkg
        if lihe_shape:
            out[self._lihe] = out[self._lihe] * (1.0 + lihe_shape * self.spectrum.centers)
        return out.sum(axis=0)

    # -- likelihood ----------------------------------------------------------
    def _stat(self, pred: np.ndarray) -> float:
        if self.config.likelihood == "poisson":
            return chi2_poisson(self.spectrum.n_obs, pred)
        return chi2_cnp(self.spectrum.n_obs, pred)

    def chi2(self, sin2_theta12, dm2_21, pulls=None) -> float:
        """``pulls`` = [norm, bkg x n, lihe_shape, scale, bias, res, flux x k]."""

        cfg = self.config
        n = len(self._bkg_names)
        n_f = self.n_flux_modes
        p = np.zeros(1 + n + 4 + n_f) if pulls is None else np.asarray(pulls, dtype=float)
        norm, bkg, shape = p[0], p[1 : 1 + n], p[1 + n]
        energy, fp = p[2 + n : 5 + n], p[5 + n :]

        pred = (1.0 + norm) * self.signal(sin2_theta12, dm2_21, energy,
                                          fp if n_f else None)
        pred = pred + self.background(bkg, shape)

        value = self._stat(pred)
        value += (norm / cfg.sigma_norm) ** 2
        value += float(np.sum((bkg / self._bkg_prior) ** 2))
        value += (shape / cfg.lihe_shape) ** 2
        value += (energy[0] / cfg.sigma_scale) ** 2
        value += (energy[1] / cfg.sigma_bias) ** 2
        value += (energy[2] / cfg.sigma_res) ** 2
        value += float(np.sum(fp ** 2))
        return value

    def _basis(self, sin2_theta12, dm2_21):
        """Signal and its derivative with respect to every linear pull.

        Both the energy nuisances (through the response matrix) and the flux
        pulls (through the spectrum) enter linearly, so at fixed oscillation
        parameters the signal is affine in the 3 + 25 pulls.  Neglected terms
        are products of two pulls, parts in 10^4 at their prior widths.
        """

        flux = self._flux_vector(sin2_theta12, dm2_21)
        k = self._bin_scale * self.normalisation
        base = k * (self._R0 @ flux)
        cols = [k * (dR @ flux) for dR in self._dR]
        if self.n_flux_modes:
            cols += list(k * (self._R0 @ (flux * self._flux_modes).T).T)
        return base, np.asarray(cols)

    def _profile(self, sin2_theta12, dm2_21, x0=None):
        from scipy.optimize import minimize

        n = len(self._bkg_names)
        n_f = self.n_flux_modes
        start = np.zeros(1 + n + 4 + n_f) if x0 is None else np.asarray(x0, dtype=float)
        cfg = self.config
        base, cols = self._basis(sin2_theta12, dm2_21)

        inv_bkg = 1.0 / self._bkg_prior
        widths = np.array([cfg.sigma_scale, cfg.sigma_bias, cfg.sigma_res])

        def objective(x):
            norm, bkg, shape = x[0], x[1 : 1 + n], x[1 + n]
            lin = x[2 + n :]                       # 3 energy + n_f flux
            pred = (1.0 + norm) * (base + lin @ cols)
            pred = pred + self.background(bkg, shape)
            value = self._stat(pred)
            value += (norm / cfg.sigma_norm) ** 2
            value += float(np.sum((bkg * inv_bkg) ** 2))
            value += (shape / cfg.lihe_shape) ** 2
            value += float(np.sum((lin[:3] / widths) ** 2))
            if n_f:
                value += float(np.sum(lin[3:] ** 2))
            return value

        free = (0.0, 0.0)
        bounds = [(None, None)] * (1 + n)
        bounds += [(None, None) if cfg.use_lihe_shape else free]
        bounds += [(None, None) if cfg.use_energy_pulls else free] * 3
        bounds += [(None, None)] * n_f
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        return float(result.fun), result.x

    def profiled_chi2(self, sin2_theta12, dm2_21, x0=None):
        return self._profile(sin2_theta12, dm2_21, x0)

    def fit(self, x0=(0.309, 7.50e-5)) -> dict:
        from scipy.optimize import minimize

        n = len(self._bkg_names)
        warm = {"x": None}

        def objective(z):
            value, pulls = self._profile(z[0], z[1] * 1e-5, warm["x"])
            warm["x"] = pulls
            return value

        result = minimize(objective, np.array([x0[0], x0[1] * 1e5]), method="Nelder-Mead",
                          options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-6})
        chi2, pulls = self._profile(result.x[0], result.x[1] * 1e-5, warm["x"])
        return {
            "sin2_theta12": float(result.x[0]),
            "dm2_21": float(result.x[1] * 1e-5),
            "chi2": float(chi2),
            "norm_pull": float(pulls[0]),
            "bkg_pulls": dict(zip(self._bkg_names, pulls[1 : 1 + n])),
            "lihe_shape_pull": float(pulls[1 + n]),
            "energy_pulls": dict(zip(("scale", "bias", "resolution"), pulls[2 + n : 5 + n])),
            "flux_pulls_rms": (float(np.sqrt(np.mean(pulls[5 + n :] ** 2)))
                               if self.n_flux_modes else 0.0),
            "n_bins": len(self.spectrum.n_obs),
            "n_free": 3,
            "config": self.config.name,
        }

    def scan(self, s12_grid, dm2_grid) -> np.ndarray:
        out = np.empty((len(dm2_grid), len(s12_grid)))
        warm = None
        for j, dm2 in enumerate(dm2_grid):
            for i, s12 in enumerate(s12_grid):
                out[j, i], warm = self._profile(s12, dm2, warm)
        return out
