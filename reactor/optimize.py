"""Baseline-schedule optimisation for the movable near reactor.

The figure of merit is the Fisher-information bound on ``sigma(theta13)``.
Stops are added greedily (each new position and dwell chosen to maximally
reduce ``sigma(theta13)`` without revisiting earlier choices), and the dwell
allocation is then refined with L-BFGS-B at fixed positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize

from .constants import DEFAULT_OSCILLATION_PARAMS, OscillationParameters
from .cross_sections import load_ibd_cross_section
from .detector import DetectorResponse
from .experiment import Predictor, default_e_nu_grid, default_reco_edges, near_program
from .statistics import Analysis, GaussianPrior, Systematics


def sin2theta13_error_to_deg(sigma_sin2: float, params: OscillationParameters) -> float:
    """Convert sigma(sin^2 theta13) into sigma(theta13) in degrees."""

    sin_2theta = 2.0 * np.sqrt(params.sin2_theta13 * (1.0 - params.sin2_theta13))
    return float(np.degrees(sigma_sin2 / sin_2theta))


def theta13_deg_error_to_sin2(sigma_deg: float, params: OscillationParameters) -> float:
    sin_2theta = 2.0 * np.sqrt(params.sin2_theta13 * (1.0 - params.sin2_theta13))
    return float(np.radians(sigma_deg) * sin_2theta)


@dataclass
class ProgramSpec:
    """Everything needed to evaluate a movable-reactor schedule."""

    truth: OscillationParameters = DEFAULT_OSCILLATION_PARAMS
    power_mwth: float = 100.0
    systematics: Systematics = field(default_factory=Systematics)
    priors: tuple[GaussianPrior, ...] = ()
    include_far: bool = True
    reco_edges: np.ndarray = field(default_factory=default_reco_edges)
    e_nu_grid: np.ndarray = field(default_factory=default_e_nu_grid)
    response: DetectorResponse = field(default_factory=DetectorResponse)
    parameters: tuple[str, ...] = ("sin2_theta13", "dm2_ee")
    extra_samples: tuple = ()
    background_rates: dict | None = None

    def __post_init__(self) -> None:
        self._xsec = load_ibd_cross_section()

    # -- core evaluation ----------------------------------------------------
    def predictor(self, baselines_km, exposures_days) -> Predictor:
        samples = near_program(
            baselines_km,
            exposures_days,
            power_mwth=self.power_mwth,
            include_far=self.include_far,
            background_rates=self.background_rates,
        )
        samples = list(samples) + list(self.extra_samples)
        return Predictor(
            samples,
            reco_edges_mev=self.reco_edges,
            e_nu_grid_mev=self.e_nu_grid,
            response=self.response,
            cross_section=self._xsec,
        )

    def analysis(self, baselines_km, exposures_days) -> Analysis:
        return Analysis(
            self.predictor(baselines_km, exposures_days),
            self.truth,
            systematics=self.systematics,
            priors=self.priors,
        )

    def errors(self, baselines_km, exposures_days) -> dict[str, float]:
        return self.analysis(baselines_km, exposures_days).fisher_errors(self.parameters)

    def sigma_theta13_deg(self, baselines_km, exposures_days) -> float:
        err = self.errors(baselines_km, exposures_days)
        return sin2theta13_error_to_deg(err["sin2_theta13"], self.truth)


# ---------------------------------------------------------------------------
# Greedy stop placement
# ---------------------------------------------------------------------------
@dataclass
class Schedule:
    baselines_km: np.ndarray
    fractions: np.ndarray
    total_days: float
    sigma_theta13_deg: float

    @property
    def exposures_days(self) -> np.ndarray:
        return self.fractions * self.total_days

    def __repr__(self) -> str:
        stops = ", ".join(
            f"{l:.3f} km ({f:.0%})" for l, f in zip(self.baselines_km, self.fractions)
        )
        return f"Schedule[{stops} | {self.total_days:.0f} d | sigma(th13)={self.sigma_theta13_deg:.4f} deg]"


def greedy_schedule(
    spec: ProgramSpec,
    total_days: float,
    n_stops: int,
    *,
    baseline_grid: np.ndarray | None = None,
    fraction_grid: np.ndarray | None = None,
    verbose: bool = False,
) -> Schedule:
    """Add stops one at a time, each chosen to minimise sigma(theta13).

    At step ``k`` the already-chosen positions are frozen and only the new
    position and the split of exposure between old and new are optimised.
    """

    if baseline_grid is None:
        baseline_grid = np.concatenate(
            [np.linspace(0.2, 0.6, 9), np.linspace(0.7, 3.0, 24)]
        )
    if fraction_grid is None:
        fraction_grid = np.linspace(0.05, 0.95, 19)

    baselines: list[float] = []
    fractions = np.array([], dtype=float)
    best_sigma = np.inf

    for step in range(n_stops):
        best = None
        for candidate in baseline_grid:
            if any(abs(candidate - b) < 1.0e-6 for b in baselines):
                continue
            trial_baselines = baselines + [float(candidate)]
            if step == 0:
                trials = [np.array([1.0])]
            else:
                trials = [
                    np.concatenate([fractions * (1.0 - f), [f]]) for f in fraction_grid
                ]
            for trial_fractions in trials:
                sigma = spec.sigma_theta13_deg(
                    trial_baselines, trial_fractions * total_days
                )
                if best is None or sigma < best[0]:
                    best = (sigma, trial_baselines, trial_fractions)

        if best is None:
            break
        sigma, trial_baselines, trial_fractions = best
        if step > 0 and sigma >= best_sigma * (1.0 - 1.0e-4):
            if verbose:
                print(f"  stop {step + 1}: no improvement ({sigma:.5f} vs {best_sigma:.5f})")
            break
        baselines, fractions, best_sigma = trial_baselines, trial_fractions, sigma
        if verbose:
            print(f"  stop {step + 1}: L={baselines[-1]:.3f} km, sigma={sigma:.5f} deg")

    return Schedule(np.array(baselines), np.asarray(fractions), total_days, best_sigma)


def refine_fractions(
    spec: ProgramSpec,
    baselines_km,
    total_days: float,
    *,
    initial_fractions=None,
    min_fraction: float = 1.0e-3,
) -> Schedule:
    """L-BFGS-B refinement of the dwell fractions at fixed positions."""

    baselines_km = np.asarray(baselines_km, dtype=float)
    n = baselines_km.size
    if n == 1:
        sigma = spec.sigma_theta13_deg(baselines_km, np.array([total_days]))
        return Schedule(baselines_km, np.array([1.0]), total_days, sigma)

    if initial_fractions is None:
        initial_fractions = np.full(n, 1.0 / n)
    initial_fractions = np.asarray(initial_fractions, dtype=float)

    # Softmax parameterisation keeps the fractions positive and normalised.
    x0 = np.log(np.maximum(initial_fractions, min_fraction))
    x0 -= x0[0]

    def to_fractions(x):
        w = np.exp(x - x.max())
        return w / w.sum()

    def objective(x):
        fractions = to_fractions(x)
        if np.any(fractions * total_days < 1.0e-3):
            return 1.0e6
        return spec.sigma_theta13_deg(baselines_km, fractions * total_days)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-12, "eps": 1e-4},
    )
    fractions = to_fractions(result.x)
    return Schedule(baselines_km, fractions, total_days, float(result.fun))


def optimize_schedule(
    spec: ProgramSpec,
    total_days: float,
    n_stops: int = 2,
    *,
    baseline_grid: np.ndarray | None = None,
    refine_positions: bool = True,
    verbose: bool = False,
) -> Schedule:
    """Greedy placement followed by dwell (and optionally position) refinement."""

    schedule = greedy_schedule(
        spec, total_days, n_stops, baseline_grid=baseline_grid, verbose=verbose
    )
    schedule = refine_fractions(
        spec, schedule.baselines_km, total_days, initial_fractions=schedule.fractions
    )
    if refine_positions and schedule.baselines_km.size > 1:
        schedule = refine_positions_and_fractions(spec, schedule, verbose=verbose)
    return schedule


def refine_positions_and_fractions(
    spec: ProgramSpec,
    schedule: Schedule,
    *,
    min_baseline_km: float | None = None,
    max_baseline_km: float = 3.0,
    verbose: bool = False,
) -> Schedule:
    """Joint Nelder-Mead polish of positions and dwell fractions."""

    baselines = np.asarray(schedule.baselines_km, dtype=float)
    n = baselines.size
    if min_baseline_km is None:
        min_baseline_km = float(baselines.min())

    x0 = np.concatenate([baselines, np.log(np.maximum(schedule.fractions, 1e-3))])

    def unpack(x):
        b = np.clip(x[:n], min_baseline_km, max_baseline_km)
        w = np.exp(x[n:] - x[n:].max())
        return b, w / w.sum()

    def objective(x):
        b, f = unpack(x)
        if np.any(f * schedule.total_days < 1.0e-2):
            return 1.0e6
        return spec.sigma_theta13_deg(b, f * schedule.total_days)

    result = minimize(
        objective, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000}
    )
    b, f = unpack(result.x)
    order = np.argsort(b)
    if verbose:
        print(f"  refined: {np.round(b[order], 4)} km, fractions {np.round(f[order], 3)}")
    return Schedule(b[order], f[order], schedule.total_days, float(result.fun))


# ---------------------------------------------------------------------------
# Exposure scaling
# ---------------------------------------------------------------------------
def sigma_vs_exposure(
    spec: ProgramSpec,
    baselines_km,
    fractions,
    total_days_grid: np.ndarray,
) -> np.ndarray:
    """sigma(theta13) in degrees as a function of total exposure."""

    baselines_km = np.asarray(baselines_km, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    return np.array(
        [spec.sigma_theta13_deg(baselines_km, fractions * t) for t in total_days_grid]
    )


def infinite_exposure_floor(
    spec: ProgramSpec,
    baselines_km,
    fractions,
    *,
    reference_days: float = 1.0e6,
) -> float:
    """Systematics-limited floor on sigma(theta13), degrees.

    Evaluated at a very large exposure, where the statistical term is
    negligible compared with the correlated systematic blocks.
    """

    baselines_km = np.asarray(baselines_km, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    return spec.sigma_theta13_deg(baselines_km, fractions * reference_days)


def time_to_reach(
    spec: ProgramSpec,
    baselines_km,
    fractions,
    target_deg: float = 0.11,
    *,
    t_min: float = 1.0,
    t_max: float = 4000.0,
) -> float:
    """Total exposure in days needed to reach ``target_deg``.

    Returns ``inf`` when the systematics floor lies above the target.
    """

    baselines_km = np.asarray(baselines_km, dtype=float)
    fractions = np.asarray(fractions, dtype=float)

    def f(t):
        return spec.sigma_theta13_deg(baselines_km, fractions * t) - target_deg

    if f(t_max) > 0.0:
        floor = infinite_exposure_floor(spec, baselines_km, fractions)
        if floor > target_deg:
            return float("inf")
        return float("nan")
    if f(t_min) < 0.0:
        return float(t_min)
    return float(brentq(f, t_min, t_max, xtol=0.5))


def best_single_stop(
    spec: ProgramSpec,
    total_days: float,
    *,
    baseline_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Best single baseline and its sigma(theta13) for a given exposure."""

    if baseline_grid is None:
        baseline_grid = np.linspace(0.2, 3.0, 57)
    sigmas = np.array(
        [spec.sigma_theta13_deg([b], [total_days]) for b in baseline_grid]
    )
    i = int(np.argmin(sigmas))
    return float(baseline_grid[i]), float(sigmas[i])
