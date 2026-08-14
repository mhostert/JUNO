"""Covariance construction, chi-square, Fisher information, and scans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .constants import OscillationParameters


# ---------------------------------------------------------------------------
# Systematics budget
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Systematics:
    """Systematic uncertainty budget for one correlation group.

    ``sigma_flux``, ``sigma_xsec`` and ``sigma_eff`` are fully correlated
    (rank-1) normalisation terms.  ``sigma_shape`` is an energy-correlated
    spectral-shape term with correlation length ``shape_lambda_mev``, and is
    correlated across every sample in the group.  ``sigma_eres`` and
    ``sigma_escale`` are rank-1 terms built from the response derivatives.
    """

    sigma_flux: float = 0.05
    sigma_xsec: float = 0.03
    sigma_eff: float = 0.02
    sigma_shape: float = 0.06
    shape_lambda_mev: float = 0.7
    shape_kernel: str = "exponential"
    sigma_eres: float = 0.035
    sigma_escale: float = 0.0
    sigma_bkg: float = 0.0
    uncorrelated_bin_to_bin: float = 0.0

    # JUNO Conceptual Design Report budget, for comparison.
    @classmethod
    def juno_cdr(cls) -> "Systematics":
        return cls(
            sigma_flux=0.02,
            sigma_xsec=0.001,
            sigma_eff=0.01,
            sigma_shape=0.01,
            shape_lambda_mev=0.7,
            sigma_eres=0.035,
        )

    @classmethod
    def statistics_only(cls) -> "Systematics":
        return cls(
            sigma_flux=0.0,
            sigma_xsec=0.0,
            sigma_eff=0.0,
            sigma_shape=0.0,
            sigma_eres=0.0,
            sigma_escale=0.0,
            sigma_bkg=0.0,
        )

    def without(self, *terms: str) -> "Systematics":
        """Copy with the named terms zeroed, e.g. ``.without('shape', 'norm')``."""

        kwargs: dict[str, float] = {}
        for term in terms:
            if term == "norm":
                kwargs.update(sigma_flux=0.0, sigma_xsec=0.0, sigma_eff=0.0)
            elif term == "shape":
                kwargs.update(sigma_shape=0.0)
            elif term == "eres":
                kwargs.update(sigma_eres=0.0)
            elif term == "escale":
                kwargs.update(sigma_escale=0.0)
            elif term == "bkg":
                kwargs.update(sigma_bkg=0.0)
            else:
                raise KeyError(f"Unknown systematic term {term!r}")
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
def _exponential_correlation(
    axis: np.ndarray, correlation_length: float, kernel: str = "exponential"
) -> np.ndarray:
    """Correlation matrix on an energy axis.

    ``exponential`` uses exp(-|dE|/lambda) (Ornstein-Uhlenbeck), whose sample
    paths carry power at all frequencies, so a correlated shape error of this
    kind can mimic a fairly wiggly spectral distortion.  ``gaussian`` uses
    exp(-dE^2 / 2 lambda^2), which is band-limited and therefore much less able
    to imitate an oscillation pattern.  The distinction matters: the draft's
    text says "Gaussian correlation length" while its appendix writes the
    exponential form.
    """

    axis = np.asarray(axis, dtype=float)
    if np.isinf(correlation_length):
        return np.ones((axis.size, axis.size))
    if correlation_length <= 0.0:
        return np.eye(axis.size)

    delta = axis[:, None] - axis[None, :]
    if kernel == "exponential":
        return np.exp(-np.abs(delta) / correlation_length)
    if kernel == "gaussian":
        return np.exp(-0.5 * (delta / correlation_length) ** 2)
    raise KeyError(f"Unknown shape kernel {kernel!r}; use 'exponential' or 'gaussian'.")


def build_covariance(
    counts: np.ndarray,
    energy_centers_mev: np.ndarray,
    *,
    groups: Sequence[str] | None = None,
    systematics: Systematics | dict[str, Systematics] = Systematics(),
    eres_derivative: np.ndarray | None = None,
    escale_derivative: np.ndarray | None = None,
    background_counts: np.ndarray | None = None,
    stat_floor: float = 1.0e-9,
) -> np.ndarray:
    """Assemble C = C_stat + C_norm + C_shape + C_eres + C_escale + C_bkg.

    Parameters
    ----------
    counts:
        ``(n_samples, n_energy)`` predicted counts (signal + background).
    groups:
        Correlation-group tag per sample.  Correlated systematics act only
        within a group; different groups share no nuisance parameters.
    systematics:
        Either one budget applied to every group, or a mapping from group name
        to budget.
    """

    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("counts must have shape (n_samples, n_energy).")
    n_pos, n_energy = counts.shape
    if len(energy_centers_mev) != n_energy:
        raise ValueError("energy_centers_mev length must match the number of energy bins.")

    if groups is None:
        groups = ["all"] * n_pos
    groups = list(groups)
    if len(groups) != n_pos:
        raise ValueError("groups length must match the number of samples.")

    if isinstance(systematics, Systematics):
        budgets = {g: systematics for g in set(groups)}
    else:
        budgets = dict(systematics)

    flat = counts.reshape(-1)
    n_total = flat.size

    cov = np.diag(np.maximum(flat, stat_floor))
    e = np.asarray(energy_centers_mev, dtype=float)

    for group in sorted(set(groups)):
        budget = budgets.get(group)
        if budget is None:
            continue
        idx = np.concatenate(
            [
                np.arange(i * n_energy, (i + 1) * n_energy)
                for i, g in enumerate(groups)
                if g == group
            ]
        )
        sub = flat[idx]

        # Rank-1 normalisation terms
        for sigma in (budget.sigma_flux, budget.sigma_xsec, budget.sigma_eff):
            if sigma > 0.0:
                cov[np.ix_(idx, idx)] += np.outer(sigma * sub, sigma * sub)

        # Energy-correlated shape term, correlated across samples in the group
        if budget.sigma_shape > 0.0:
            n_g = idx.size // n_energy
            corr = np.tile(
                _exponential_correlation(e, budget.shape_lambda_mev, budget.shape_kernel),
                (n_g, n_g),
            )
            cov[np.ix_(idx, idx)] += budget.sigma_shape**2 * np.outer(sub, sub) * corr

        # Uncorrelated bin-to-bin term
        if budget.uncorrelated_bin_to_bin > 0.0:
            cov[idx, idx] += (budget.uncorrelated_bin_to_bin * sub) ** 2

    # Rank-1 response terms (shared across everything: one detector)
    for deriv, key in ((eres_derivative, "sigma_eres"), (escale_derivative, "sigma_escale")):
        if deriv is None:
            continue
        d = np.asarray(deriv, dtype=float).reshape(-1)
        if d.size != n_total:
            raise ValueError("Response derivative has the wrong length.")
        for group in sorted(set(groups)):
            budget = budgets.get(group)
            if budget is None or getattr(budget, key) <= 0.0:
                continue
            idx = np.concatenate(
                [
                    np.arange(i * n_energy, (i + 1) * n_energy)
                    for i, g in enumerate(groups)
                    if g == group
                ]
            )
            sigma = getattr(budget, key)
            cov[np.ix_(idx, idx)] += sigma**2 * np.outer(d[idx], d[idx])

    # Background rate uncertainty (rank-1 on the background component)
    if background_counts is not None:
        bkg = np.asarray(background_counts, dtype=float).reshape(-1)
        for group in sorted(set(groups)):
            budget = budgets.get(group)
            if budget is None or budget.sigma_bkg <= 0.0:
                continue
            idx = np.concatenate(
                [
                    np.arange(i * n_energy, (i + 1) * n_energy)
                    for i, g in enumerate(groups)
                    if g == group
                ]
            )
            cov[np.ix_(idx, idx)] += budget.sigma_bkg**2 * np.outer(bkg[idx], bkg[idx])

    return cov


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------
class _Solver:
    """Cached Cholesky (or pseudo-inverse) solve for a fixed covariance."""

    def __init__(self, covariance: np.ndarray) -> None:
        self.covariance = np.asarray(covariance, dtype=float)
        try:
            self._factor = cho_factor(self.covariance, lower=True, check_finite=False)
            self._pinv = None
        except Exception:
            self._factor = None
            self._pinv = np.linalg.pinv(self.covariance, rcond=1.0e-12)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self._factor is not None:
            return cho_solve(self._factor, rhs, check_finite=False)
        return self._pinv @ rhs


def chi2(residual: np.ndarray, covariance: np.ndarray) -> float:
    """r^T C^-1 r."""

    r = np.asarray(residual, dtype=float).reshape(-1)
    return float(r @ _Solver(covariance).solve(r))


def cnp_variance(n_obs: np.ndarray, n_pred: np.ndarray) -> np.ndarray:
    """Combined Neyman-Pearson variance, 3 / (1/N_obs + 2/N_pred).

    This is the statistical variance JUNO uses (Ji et al., NIM A961, 163677),
    which removes the opposite-sign biases of the pure Neyman (``N_obs``) and
    Pearson (``N_pred``) forms.  Empty bins fall back to the Pearson form.
    """

    obs = np.asarray(n_obs, dtype=float)
    pred = np.maximum(np.asarray(n_pred, dtype=float), 1.0e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        cnp = 3.0 / (1.0 / np.where(obs > 0, obs, np.nan) + 2.0 / pred)
    return np.where(obs > 0, cnp, pred / 2.0)


def chi2_cnp(n_obs: np.ndarray, n_pred: np.ndarray) -> float:
    """Binned CNP chi-square with no correlations."""

    obs = np.asarray(n_obs, dtype=float)
    pred = np.asarray(n_pred, dtype=float)
    return float(np.sum((obs - pred) ** 2 / cnp_variance(obs, pred)))


def chi2_poisson(n_obs: np.ndarray, n_pred: np.ndarray) -> float:
    """Poisson (Baker-Cousins) chi-square, 2 sum [P - O + O ln(O/P)].

    The form NuFit use as their default (arXiv:2601.09791 Eq. 5).  Unlike CNP it
    stays well behaved in nearly empty bins, which matters at the ends of the
    JUNO spectrum where the release has a handful of counts per bin.
    """

    obs = np.asarray(n_obs, dtype=float)
    pred = np.maximum(np.asarray(n_pred, dtype=float), 1.0e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(obs > 0, obs * np.log(obs / pred), 0.0)
    return float(2.0 * np.sum(pred - obs + log_term))


def fisher_uncertainty(derivative: np.ndarray, covariance: np.ndarray) -> float:
    """One-parameter 1 sigma uncertainty from a single Asimov derivative."""

    d = np.asarray(derivative, dtype=float).reshape(-1)
    information = float(d @ _Solver(covariance).solve(d))
    return float("inf") if information <= 0.0 else 1.0 / np.sqrt(information)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaussianPrior:
    """Gaussian external constraint on one oscillation parameter."""

    name: str
    central: float
    sigma: float

    def chi2(self, params: OscillationParameters) -> float:
        if self.sigma <= 0.0 or not np.isfinite(self.sigma):
            return 0.0
        return ((getattr(params, self.name) - self.central) / self.sigma) ** 2


def dayabay_priors() -> list[GaussianPrior]:
    """Daya Bay constraints on the atmospheric sector."""

    from .constants import (
        DAYABAY_DM2EE,
        DAYABAY_DM2EE_ERR,
        DAYABAY_SIN2_THETA13,
        DAYABAY_SIN2_THETA13_ERR,
    )

    return [
        GaussianPrior("sin2_theta13", DAYABAY_SIN2_THETA13, DAYABAY_SIN2_THETA13_ERR),
        GaussianPrior("dm2_ee", DAYABAY_DM2EE, DAYABAY_DM2EE_ERR),
    ]


def nufit_priors(*names: str) -> list[GaussianPrior]:
    """NuFit 6.1 normal-ordering constraints on the requested parameters."""

    from .constants import NUFIT61_NO, NUFIT61_NO_ERRORS

    return [
        GaussianPrior(name, getattr(NUFIT61_NO, name), NUFIT61_NO_ERRORS[name])
        for name in names
    ]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@dataclass
class Analysis:
    """Asimov chi-square analysis on top of a :class:`~reactor.experiment.Predictor`.

    The covariance is built once at the Asimov truth and held fixed, which is
    the standard way of avoiding the bias that a counts-dependent covariance
    introduces into a scan.
    """

    predictor: object
    truth: OscillationParameters
    systematics: Systematics | dict[str, Systematics] = field(default_factory=Systematics)
    priors: Sequence[GaussianPrior] = ()
    matter: bool = False
    include_eres: bool = True
    include_escale: bool = False

    def __post_init__(self) -> None:
        pred = self.predictor
        self.asimov = pred.counts(self.truth, matter=self.matter)

        eres_deriv = escale_deriv = None
        if self.include_eres or self.include_escale:
            e_nu = pred.e_nu
            edges = pred.reco_edges
            if self.include_eres:
                d_resp = pred.response.matrix_derivative_scale(e_nu, edges)
                eres_deriv = self._propagate(d_resp)
            if self.include_escale:
                d_resp = pred.response.matrix_derivative_energy_scale(e_nu, edges)
                escale_deriv = self._propagate(d_resp)

        self.covariance = build_covariance(
            self.asimov,
            pred.prompt_centers,
            groups=pred.sample_groups,
            systematics=self.systematics,
            eres_derivative=eres_deriv,
            escale_derivative=escale_deriv,
            background_counts=pred.background_counts(),
        )
        self._solver = _Solver(self.covariance)

    # -- internals ----------------------------------------------------------
    def _propagate(self, d_response: np.ndarray) -> np.ndarray:
        """Push a response-matrix derivative through to counts."""

        pred = self.predictor
        from .oscillations import survival_probability_ee, survival_probability_matter

        prob_fn = survival_probability_matter if self.matter else survival_probability_ee
        out = np.empty((pred.n_samples, pred.n_energy))
        cache: dict[float, np.ndarray] = {}
        for idx, entries in enumerate(pred._kernels):
            total = np.zeros_like(pred.e_nu)
            for baseline, density in entries:
                p = cache.get(baseline)
                if p is None:
                    p = prob_fn(pred.e_nu, baseline, self.truth)
                    cache[baseline] = p
                total += density * p
            out[idx] = d_response @ total
        return out

    # -- chi-square ---------------------------------------------------------
    def chi2(self, params: OscillationParameters) -> float:
        pred = self.predictor.flat_counts(params, matter=self.matter)
        r = self.asimov.reshape(-1) - pred
        value = float(r @ self._solver.solve(r))
        for prior in self.priors:
            value += prior.chi2(params)
        return value

    def chi2_at(self, **kwargs) -> float:
        return self.chi2(self.truth.replace(**kwargs))

    # -- Fisher information -------------------------------------------------
    def fisher_matrix(
        self, parameter_names: Sequence[str], *, rel_step: float = 1.0e-3
    ) -> np.ndarray:
        jac = self.predictor.jacobian(
            self.truth, parameter_names, rel_step=rel_step, matter=self.matter
        )
        solved = np.array([self._solver.solve(row) for row in jac])
        fisher = jac @ solved.T
        for i, name in enumerate(parameter_names):
            for prior in self.priors:
                if prior.name == name and prior.sigma > 0:
                    fisher[i, i] += 1.0 / prior.sigma**2
        return fisher

    def fisher_errors(
        self, parameter_names: Sequence[str], *, rel_step: float = 1.0e-3
    ) -> dict[str, float]:
        """Profiled 1 sigma errors from the Fisher matrix."""

        fisher = self.fisher_matrix(parameter_names, rel_step=rel_step)
        cov = np.linalg.inv(fisher)
        return {name: float(np.sqrt(cov[i, i])) for i, name in enumerate(parameter_names)}

    # -- minimisation / profiling ------------------------------------------
    def minimize(
        self,
        free: Sequence[str],
        *,
        fixed: dict[str, float] | None = None,
        start: OscillationParameters | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Minimise chi-square over ``free``, holding ``fixed`` at given values."""

        base = (start or self.truth).replace(**(fixed or {}))
        x0 = np.array([getattr(base, name) for name in free], dtype=float)
        scale = np.where(np.abs(x0) > 0, np.abs(x0), 1.0)

        def objective(x):
            values = {name: float(x[i] * scale[i]) for i, name in enumerate(free)}
            return self.chi2(base.replace(**values))

        result = minimize(
            objective,
            x0 / scale,
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 5000},
        )
        best = {name: float(result.x[i] * scale[i]) for i, name in enumerate(free)}
        return float(result.fun), best

    def profile(
        self,
        scan_name: str,
        scan_values: np.ndarray,
        free: Sequence[str],
    ) -> np.ndarray:
        """1D profile chi-square over ``scan_name``, minimising over ``free``."""

        out = np.empty(len(scan_values))
        for i, value in enumerate(scan_values):
            if not free:
                out[i] = self.chi2(self.truth.replace(**{scan_name: float(value)}))
            else:
                out[i], _ = self.minimize(free, fixed={scan_name: float(value)})
        return out

    def grid(
        self,
        name_x: str,
        values_x: np.ndarray,
        name_y: str,
        values_y: np.ndarray,
        *,
        free: Sequence[str] = (),
    ) -> np.ndarray:
        """2D chi-square grid, shape ``(len(values_y), len(values_x))``."""

        out = np.empty((len(values_y), len(values_x)))
        for i, vy in enumerate(values_y):
            for j, vx in enumerate(values_x):
                fixed = {name_x: float(vx), name_y: float(vy)}
                if free:
                    out[i, j], _ = self.minimize(free, fixed=fixed)
                else:
                    out[i, j] = self.chi2(self.truth.replace(**fixed))
        return out


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def covariance_from_chi2_surface(
    x: np.ndarray, y: np.ndarray, chi2_grid: np.ndarray, delta_max: float = 9.0
) -> np.ndarray:
    """2x2 covariance from a quadratic fit near the minimum of a chi-square surface.

    ``chi2_grid`` has shape ``(len(y), len(x))``.  Only points within
    ``delta_max`` of the minimum are used, so the result describes the core of
    the region rather than its tails.  Returns the covariance in the units of
    ``x`` and ``y``, from which the correlation -- the *tilt* of the contour --
    follows directly.
    """

    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    d = np.asarray(chi2_grid, float) - np.nanmin(chi2_grid)
    mask = np.isfinite(d) & (d < delta_max)
    design = np.stack(
        [np.ones(mask.sum()), X[mask], Y[mask], X[mask] ** 2, X[mask] * Y[mask], Y[mask] ** 2],
        axis=1,
    )
    coeff, *_ = np.linalg.lstsq(design, d[mask], rcond=None)
    hessian = 2.0 * np.array([[coeff[3], coeff[4] / 2.0], [coeff[4] / 2.0, coeff[5]]])
    return np.linalg.inv(hessian) * 2.0


def correlation_from_covariance(cov: np.ndarray) -> float:
    cov = np.asarray(cov, dtype=float)
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def profiled_error_from_grid(
    chi2_grid: np.ndarray, values: np.ndarray, axis: int
) -> float:
    """1 sigma error from the minimum-projected profile of a 2D grid."""

    profile = chi2_grid.min(axis=axis)
    profile = profile - profile.min()
    return _crossing_width(values, profile, 1.0)


def _crossing_width(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Half-width of the region y < level, by linear interpolation."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i_min = int(np.argmin(y))
    below = y <= level
    if not below.any():
        return float("nan")

    def edge(indices, direction):
        for i in indices:
            if y[i] > level:
                j = i - direction
                if 0 <= j < x.size and y[j] != y[i]:
                    t = (level - y[j]) / (y[i] - y[j])
                    return x[j] + t * (x[i] - x[j])
                return x[i]
        return x[indices[-1]] if len(indices) else x[i_min]

    lo = edge(range(i_min, -1, -1), -1)
    hi = edge(range(i_min, x.size), +1)
    return 0.5 * (hi - lo)


def mass_ordering_delta_chi2(
    predictor,
    truth: OscillationParameters,
    *,
    systematics: Systematics | dict[str, Systematics] = Systematics(),
    priors: Sequence[GaussianPrior] = (),
    free: Sequence[str] = ("dm2_ee", "sin2_theta12", "dm2_21", "sin2_theta13"),
    matter: bool = True,
) -> float:
    """Asimov Delta chi^2 between the wrong and the true mass ordering.

    Asimov data are generated with ``truth`` (which carries the true
    ordering); the fit is then repeated with the ordering flipped, minimising
    over ``free``.  The returned value is chi2_min(wrong) - chi2_min(true).
    """

    analysis = Analysis(
        predictor, truth, systematics=systematics, priors=priors, matter=matter
    )
    chi2_true, _ = analysis.minimize(free)
    wrong = truth.replace(ordering=-truth.ordering)
    chi2_wrong, _ = analysis.minimize(free, start=wrong, fixed={"ordering": wrong.ordering})
    return float(chi2_wrong - chi2_true)


def scan_oscillation_grid(
    predict: Callable[[OscillationParameters], np.ndarray],
    observed_counts: np.ndarray,
    covariance: np.ndarray,
    sin2_theta13_grid: np.ndarray,
    dm2ee_grid: np.ndarray,
    *,
    fixed_sin2_theta12: float,
    fixed_dm2_21: float,
) -> np.ndarray:
    """Scan chi-square over the (sin^2 theta13, dm2_ee) plane."""

    observed = np.asarray(observed_counts, dtype=float).reshape(-1)
    solver = _Solver(covariance)
    values = np.empty((len(dm2ee_grid), len(sin2_theta13_grid)))
    for i, dm2ee in enumerate(dm2ee_grid):
        for j, s13 in enumerate(sin2_theta13_grid):
            params = OscillationParameters(
                sin2_theta13=float(s13),
                dm2_ee=float(dm2ee),
                sin2_theta12=fixed_sin2_theta12,
                dm2_21=fixed_dm2_21,
            )
            r = observed - np.asarray(predict(params), dtype=float).reshape(-1)
            values[i, j] = float(r @ solver.solve(r))
    return values
