"""Fit of the JUNO 2025 first-measurement spectrum.

Wraps everything needed to predict the released 66-bin prompt-energy spectrum
and fit the solar parameters to it, following the collaboration's analysis as
closely as the public information allows.

What is taken from the release or the paper (arXiv:2511.14593), and what is
assumed, is documented in :class:`JUNO2025Model`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    DAYABAY_DM2EE,
    DAYABAY_SIN2_THETA13,
    JUNO2025_BACKGROUNDS_CPD,
    JUNO2025_EFFICIENCY_TOTAL,
    JUNO2025_EFFICIENCY_TOTAL_ERR,
    JUNO2025_LIVETIME_DAYS,
    JUNO2025_MATTER_DENSITY,
    JUNO2025_NONOSC_CPD,
    JUNO2025_RATE_SYSTEMATICS,
    JUNO2025_TARGET_PROTONS,
    SECONDS_PER_DAY,
    OscillationParameters,
)
from .cross_sections import load_ibd_cross_section
from .detector import DetectorResponse, gaussian_bin_response, integration_weights, juno_2025_response
from .flux import default_juno_cores
from .juno_data import JUNOSpectrum, load_spectrum
from .oscillations import survival_probability_ee, survival_probability_matter
from .statistics import chi2_cnp, cnp_variance

#: Relative pre-fit constraints on each background, from their Table 1.
BACKGROUND_PRIORS = {
    "9Li/8He": 1.4 / 4.3,
    "geoneutrino": 0.5 / 1.2,
    "world reactors": 0.09 / 0.88,
    "214Bi-214Po": 0.10 / 0.18,
    "other": 0.5,
}


def total_rate_systematic() -> float:
    """Rate uncertainties of their Table 2 plus the efficiency, in quadrature."""

    terms = list(JUNO2025_RATE_SYSTEMATICS.values()) + [JUNO2025_EFFICIENCY_TOTAL_ERR]
    return float(np.sqrt(np.sum(np.square(terms))))


@dataclass
class JUNO2025Model:
    """Prediction of the released spectrum as a function of the solar parameters.

    Taken from the release / paper
    ------------------------------
    * the 66-bin measured spectrum and every background component (Fig. 3a);
    * the energy resolution fitted to the eight calibration peaks;
    * the positron non-linearity curve, applied rather than corrected for,
      because the release is binned in visible energy (see notebook 0);
    * live time 59.1 d, total efficiency 0.699, N_p = 1.442e33;
    * matter density 2.55 g/cm^3;
    * Daya Bay values for the atmospheric parameters, which JUNO constrains
      externally rather than fitting.

    Assumed here
    ------------
    * Huber-Mueller reactor spectra with cycle-averaged fission fractions,
      where JUNO uses a Daya-Bay-anchored, largely model-independent spectrum;
    * the overall normalisation is anchored to JUNO's quoted non-oscillated
      rate of 150.9 cpd rather than to nominal reactor power, because YJ1/YJ4
      outages and a typhoon reduced the delivered power during the run;
    * a diagonal CNP statistical covariance with pull terms for the
      normalisation and each background, rather than JUNO's full covariance
      matrix including the Daya Bay spectral constraint.
    """

    spectrum: JUNOSpectrum = field(default_factory=load_spectrum)
    response: DetectorResponse = field(default_factory=juno_2025_response)
    e_nu_grid: np.ndarray = field(default_factory=lambda: np.linspace(1.806, 13.0, 2600))
    matter_density: float = JUNO2025_MATTER_DENSITY
    sin2_theta13: float = DAYABAY_SIN2_THETA13
    dm2_ee: float = DAYABAY_DM2EE
    anchor_normalisation: bool = True
    use_matter: bool = True
    flux_bin_scales: np.ndarray | None = None
    flux_model: str = "huber-mueller"
    """Reactor spectrum: ``'huber-mueller'`` or ``'dayabay'``.

    ``'dayabay'`` uses the unfolded IBD-yield spectrum of the Daya Bay 2025
    release, re-weighted to JUNO's fuel composition.  It carries both features a
    summation model misses -- the ~7% flux deficit and the 5 MeV bump -- and is
    the spectrum JUNO's own analysis is anchored to.
    """
    rate_prior: float | None = None
    """Gaussian width of the prior on the overall normalisation.

    ``None`` uses the Table 2 rate systematics combined with the efficiency
    (2.4%).  This is the single most important choice for the *shape* of the
    confidence region: it controls how much of theta12 is determined by the
    event rate rather than by the spectral shape, and therefore the
    theta12--dm2_21 correlation.  See the diagnostic in notebook 1.
    """

    include_distant_cores: bool = True
    """Include the Daya Bay complex at 215 km in the signal.

    Its survival probability is near its energy-averaged value rather than at
    the solar minimum, so it fills in the solar dip.  Leaving it out biases
    sin^2(theta12) low by about one standard deviation; see notebook 3.
    """

    ordering: int = 1
    """Mass ordering, +1 (normal) or -1 (inverted).

    Enters only through the sign of Phi in the vacuum probability, and through
    the sign of dm2_31 in the matter one.  It is the entire vacuum mass-ordering
    signal; see notebook 3.
    """

    def __post_init__(self) -> None:
        s = self.spectrum
        enu = self.e_nu_grid
        weights = integration_weights(enu)
        xsec = load_ibd_cross_section()(enu)

        exposure = (
            JUNO2025_TARGET_PROTONS
            * JUNO2025_EFFICIENCY_TOTAL
            * s.livetime_days
            * SECONDS_PER_DAY
        )
        self._cores = default_juno_cores(
            duty_cycle=1.0, include_distant=self.include_distant_cores
        )
        if self.flux_model == "huber-mueller":
            self._density = [
                c.flux_at_detector(enu) * xsec * weights * exposure for c in self._cores
            ]
        elif self.flux_model == "dayabay":
            from .dayabay_data import dayabay_yield_model
            from .constants import CM_PER_KM

            # The Daya Bay product is the IBD yield per fission, so it replaces
            # the spectrum and the cross section together.
            yield_model = dayabay_yield_model(fractions=self._cores[0].fractions(),
                                              bin_scales=self.flux_bin_scales)
            yield_values = yield_model(enu)
            self._density = [
                c.fission_rate()
                * yield_values
                / (4.0 * np.pi * (c.baseline_km * CM_PER_KM) ** 2)
                * weights
                * exposure
                for c in self._cores
            ]
        else:
            raise KeyError(
                f"flux_model must be 'huber-mueller' or 'dayabay', got {self.flux_model!r}"
            )
        self._baselines = [c.baseline_km for c in self._cores]

        e_vis = self.response.visible_energy(enu)
        self._response_matrix = gaussian_bin_response(
            e_vis, s.edges, self.response.resolution.sigma(e_vis)
        )

        # Anchor to JUNO's quoted non-oscillated rate.
        unoscillated = float((self._response_matrix @ np.sum(self._density, axis=0)).sum())
        target = JUNO2025_NONOSC_CPD * JUNO2025_EFFICIENCY_TOTAL * s.livetime_days
        self.unoscillated_counts = unoscillated
        self.normalisation = target / unoscillated if self.anchor_normalisation else 1.0

        self._bkg_names = list(s.backgrounds)
        self._bkg = np.array([s.backgrounds[k] for k in self._bkg_names])
        self._bkg_prior = np.array([BACKGROUND_PRIORS[k] for k in self._bkg_names])
        self.rate_systematic = (
            total_rate_systematic() if self.rate_prior is None else float(self.rate_prior)
        )

    # -- prediction ---------------------------------------------------------
    def parameters(self, sin2_theta12: float, dm2_21: float) -> OscillationParameters:
        return OscillationParameters(
            sin2_theta12=float(sin2_theta12),
            dm2_21=float(dm2_21),
            sin2_theta13=self.sin2_theta13,
            dm2_ee=self.dm2_ee,
            ordering=self.ordering,
        )

    def signal(self, sin2_theta12: float, dm2_21: float) -> np.ndarray:
        """Predicted signal counts per released bin, at nominal normalisation."""

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
        return self.normalisation * (self._response_matrix @ total)

    def unoscillated(self) -> np.ndarray:
        return self.normalisation * (self._response_matrix @ np.sum(self._density, axis=0))

    def background(self, scales=None) -> np.ndarray:
        if scales is None:
            return self._bkg.sum(axis=0)
        return (np.asarray(scales, dtype=float)[:, None] * self._bkg).sum(axis=0)

    def predict(self, sin2_theta12, dm2_21, norm: float = 1.0, bkg_scales=None) -> np.ndarray:
        return norm * self.signal(sin2_theta12, dm2_21) + self.background(bkg_scales)

    # -- likelihood ---------------------------------------------------------
    def chi2(self, sin2_theta12, dm2_21, norm=1.0, bkg_scales=None, priors=True) -> float:
        pred = self.predict(sin2_theta12, dm2_21, norm, bkg_scales)
        value = chi2_cnp(self.spectrum.n_obs, pred)
        if priors:
            value += ((norm - 1.0) / self.rate_systematic) ** 2
            if bkg_scales is not None:
                value += float(
                    np.sum(((np.asarray(bkg_scales) - 1.0) / self._bkg_prior) ** 2)
                )
        return value

    def profiled_chi2(self, sin2_theta12, dm2_21, x0=None) -> tuple[float, np.ndarray]:
        """Minimise over the normalisation and the background scales.

        Uses a quasi-Newton step warm-started from the previous grid point,
        which is what makes a full surface scan affordable: Nelder-Mead needs
        several hundred evaluations per point, L-BFGS-B a few dozen.
        """

        from scipy.optimize import minimize

        n_b = len(self._bkg_names)
        start = np.ones(1 + n_b) if x0 is None else np.asarray(x0, dtype=float)

        # The signal shape is fixed once the oscillation parameters are, so
        # cache it and vary only the linear coefficients.
        signal = self.signal(sin2_theta12, dm2_21)

        def objective(x):
            pred = x[0] * signal + (np.asarray(x[1:])[:, None] * self._bkg).sum(axis=0)
            value = chi2_cnp(self.spectrum.n_obs, pred)
            value += ((x[0] - 1.0) / self.rate_systematic) ** 2
            value += float(np.sum(((x[1:] - 1.0) / self._bkg_prior) ** 2))
            return value

        result = minimize(objective, start, method="L-BFGS-B",
                          bounds=[(0.5, 1.5)] + [(0.0, 4.0)] * n_b,
                          options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-9})
        return float(result.fun), result.x

    def fit(self, x0=(0.309, 7.50e-5)) -> dict:
        """Fit the solar parameters together with all nuisance parameters."""

        from scipy.optimize import minimize

        n_b = len(self._bkg_names)
        start = np.array([x0[0], x0[1] * 1e5, 1.0, *np.ones(n_b)])

        def objective(x):
            return self.chi2(x[0], x[1] * 1e-5, norm=x[2], bkg_scales=x[3:])

        result = minimize(objective, start, method="Nelder-Mead",
                          options={"maxiter": 40000, "maxfev": 40000,
                                   "xatol": 1e-7, "fatol": 1e-7})
        return {
            "sin2_theta12": float(result.x[0]),
            "dm2_21": float(result.x[1] * 1e-5),
            "norm": float(result.x[2]),
            "bkg_scales": dict(zip(self._bkg_names, result.x[3:])),
            "chi2": float(result.fun),
            "n_bins": len(self.spectrum.n_obs),
            "n_free": 3 + n_b,
            "success": bool(result.success),
        }

    def scan(self, s12_grid, dm2_grid, profile: bool = False) -> np.ndarray:
        """Chi-square surface, shape ``(len(dm2_grid), len(s12_grid))``."""

        out = np.empty((len(dm2_grid), len(s12_grid)))
        best = None
        for i, dm2 in enumerate(dm2_grid):
            for j, s12 in enumerate(s12_grid):
                if profile:
                    out[i, j], best = self.profiled_chi2(s12, dm2, x0=best)
                else:
                    out[i, j] = self.chi2(s12, dm2)
        return out

    # -- diagnostics --------------------------------------------------------
    def flux_covariance(self, sin2_theta12, dm2_21, rel_step: float = 0.02) -> np.ndarray:
        """Covariance on the predicted spectrum from the Daya Bay flux uncertainty.

        Propagates the released 25-bin covariance of the unfolded total spectrum
        through to the JUNO binning by finite differences.  This is the
        model-independent replacement for an assumed spectral-shape systematic:
        it is the actual measurement uncertainty on the reactor spectrum.
        """

        if self.flux_model != "dayabay":
            raise ValueError("flux covariance requires flux_model='dayabay'")
        from .dayabay_data import load_unfolded

        spectra, cov75, _ = load_unfolded()
        values = spectra["Total"].values
        n = len(values)
        cov_total = np.asarray(cov75)[2 * n :, 2 * n :] * 1e-86  # released in (1e-43)^2

        reference = self.signal(sin2_theta12, dm2_21)
        jac = np.zeros((reference.size, n))
        for k in range(n):
            scales = np.ones(n)
            scales[k] = 1.0 + rel_step
            perturbed = JUNO2025Model(
                spectrum=self.spectrum, response=self.response, e_nu_grid=self.e_nu_grid,
                matter_density=self.matter_density, sin2_theta13=self.sin2_theta13,
                dm2_ee=self.dm2_ee, anchor_normalisation=False, use_matter=self.use_matter,
                flux_model="dayabay", flux_bin_scales=scales,
            )
            # undo the reference model's anchoring so the two are on one scale
            delta = perturbed.signal(sin2_theta12, dm2_21) * self.normalisation - reference
            jac[:, k] = delta / (rel_step * values[k])
        return jac @ cov_total @ jac.T

    def measured_survival(self) -> tuple[np.ndarray, np.ndarray]:
        """Background-subtracted data divided by the un-oscillated prediction."""

        s = self.spectrum
        signal_obs = s.n_obs - self.background()
        unosc = self.unoscillated()
        good = unosc > 1.0e-6
        ratio = np.where(good, signal_obs / np.where(good, unosc, 1.0), np.nan)
        error = np.where(good, s.n_obs_err / np.where(good, unosc, 1.0), np.nan)
        return ratio, error

    def pulls(self, sin2_theta12, dm2_21, norm=1.0, bkg_scales=None) -> np.ndarray:
        pred = self.predict(sin2_theta12, dm2_21, norm, bkg_scales)
        return (self.spectrum.n_obs - pred) / np.sqrt(
            cnp_variance(self.spectrum.n_obs, pred)
        )
