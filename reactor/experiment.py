"""Detector configuration, dataset definition, and event-rate prediction.

The analysis is organised around :class:`Sample` objects.  A sample is one
block of data taking: a set of reactor cores seen for a given live time.  The
JUNO far reactors are one sample; each stop of the movable near reactor is
another.  A :class:`Predictor` stacks samples into a single flattened
observable vector and evaluates it fast by pre-computing everything that does
not depend on the oscillation parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .constants import (
    DEFAULT_OSCILLATION_PARAMS,
    JUNO_IBD_EFFICIENCY,
    JUNO_MASS_KT,
    JUNO_TARGET_PROTONS,
    SECONDS_PER_DAY,
    OscillationParameters,
)
from .cross_sections import IBDCrossSection, load_ibd_cross_section
from .detector import DetectorResponse, integration_weights
from .flux import ReactorCore, default_juno_cores, movable_reactor
from .oscillations import survival_probability_ee, survival_probability_matter


@dataclass(frozen=True)
class Detector:
    """JUNO target model."""

    mass_kt: float = JUNO_MASS_KT
    target_protons: float = JUNO_TARGET_PROTONS
    efficiency: float = JUNO_IBD_EFFICIENCY

    def scaled_to_mass(self, mass_kt: float) -> "Detector":
        factor = mass_kt / self.mass_kt
        return replace(self, mass_kt=mass_kt, target_protons=self.target_protons * factor)


@dataclass(frozen=True)
class Sample:
    """One block of data taking.

    Parameters
    ----------
    name:
        Human-readable label.
    cores:
        Reactor cores illuminating the detector during this block.
    exposure_days:
        Live time in days.
    group:
        Systematics grouping tag.  Samples that share a group share their
        correlated flux / shape nuisance parameters; samples in different
        groups are treated as fully decoupled.
    background_rates:
        Mapping of background name to events per day (see
        :mod:`reactor.backgrounds`).
    burnup:
        Fractional position in the fuel cycle, passed to the fission-fraction
        models.
    """

    name: str
    cores: tuple[ReactorCore, ...]
    exposure_days: float
    group: str = "far"
    background_rates: tuple[tuple[str, float], ...] = ()
    burnup: float = 0.0
    detector: Detector = field(default_factory=Detector)

    def with_exposure(self, exposure_days: float) -> "Sample":
        return replace(self, exposure_days=float(exposure_days))


# ---------------------------------------------------------------------------
# Sample factories
# ---------------------------------------------------------------------------
def juno_far_sample(
    exposure_days: float,
    *,
    name: str = "JUNO-far",
    duty_cycle: float | None = None,
    detector: Detector | None = None,
    background_rates: dict[str, float] | None = None,
    use_cycle: bool = False,
) -> Sample:
    """The reactor cores JUNO sees.

    The eight Yangjiang + Taishan cores at ~52.5 km plus the distant complexes
    of :data:`~reactor.flux.JUNO_DISTANT_CORE_TABLE`, which carry a few percent
    of the rate but arrive far less oscillated and so fill in the solar dip.
    """

    kwargs = {} if duty_cycle is None else {"duty_cycle": duty_cycle}
    cores = tuple(default_juno_cores(use_cycle=use_cycle, **kwargs))
    return Sample(
        name=name,
        cores=cores,
        exposure_days=float(exposure_days),
        group="far",
        background_rates=tuple((background_rates or {}).items()),
        detector=detector or Detector(),
    )


def near_stop_sample(
    baseline_km: float,
    exposure_days: float,
    *,
    power_mwth: float = 100.0,
    include_far: bool = True,
    far_duty_cycle: float | None = None,
    name: str | None = None,
    evolve: bool = False,
    burnup: float = 0.0,
    detector: Detector | None = None,
    background_rates: dict[str, float] | None = None,
) -> Sample:
    """One stop of the movable near reactor, optionally with the far cores on."""

    cores: list[ReactorCore] = [
        movable_reactor(baseline_km, power_mwth=power_mwth, evolve=evolve)
    ]
    if include_far:
        kwargs = {} if far_duty_cycle is None else {"duty_cycle": far_duty_cycle}
        cores.extend(default_juno_cores(**kwargs))
    return Sample(
        name=name or f"near-{baseline_km:.4g}km",
        cores=tuple(cores),
        exposure_days=float(exposure_days),
        group="near",
        background_rates=tuple((background_rates or {}).items()),
        burnup=burnup,
        detector=detector or Detector(),
    )


def near_program(
    baselines_km,
    exposure_days,
    *,
    power_mwth: float = 100.0,
    include_far: bool = True,
    **kwargs,
) -> list[Sample]:
    """A multi-stop movable-reactor schedule."""

    baselines_km = np.atleast_1d(np.asarray(baselines_km, dtype=float))
    exposure_days = np.atleast_1d(np.asarray(exposure_days, dtype=float))
    if exposure_days.size == 1:
        exposure_days = np.full(baselines_km.size, exposure_days[0])
    if baselines_km.size != exposure_days.size:
        raise ValueError("baselines and exposures must have the same length")
    return [
        near_stop_sample(
            float(l), float(t), power_mwth=power_mwth, include_far=include_far, **kwargs
        )
        for l, t in zip(baselines_km, exposure_days)
    ]


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def default_reco_edges(n_bins: int = 200, e_lo: float = 1.02, e_hi: float = 8.22) -> np.ndarray:
    """Uniform reconstructed-prompt-energy edges.

    The default window corresponds to E_nu in [1.8, 9.0] MeV, matching the
    binning used in the movable-reactor analysis.
    """

    return np.linspace(e_lo, e_hi, n_bins + 1)


def juno_reco_edges(bin_width_mev: float = 0.02, e_lo: float = 0.94, e_hi: float = 9.0):
    """Fine binning of the kind JUNO uses for its own spectral analyses."""

    return np.arange(e_lo, e_hi + 0.5 * bin_width_mev, bin_width_mev)


def default_e_nu_grid(n: int = 1200, e_lo: float = 1.806, e_hi: float = 10.0) -> np.ndarray:
    return np.linspace(e_lo, e_hi, n)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class Predictor:
    """Stacked prediction of prompt-energy spectra over a list of samples.

    All parameter-independent factors (flux, cross section, exposure, detector
    response) are pre-computed at construction, so evaluating the prediction
    for new oscillation parameters costs one survival-probability evaluation
    per core plus one matrix-vector product per sample.
    """

    def __init__(
        self,
        samples,
        *,
        reco_edges_mev: np.ndarray | None = None,
        e_nu_grid_mev: np.ndarray | None = None,
        response: DetectorResponse | None = None,
        cross_section: IBDCrossSection | None = None,
        include_backgrounds: bool = True,
    ) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("At least one sample is required.")

        self.reco_edges = (
            default_reco_edges() if reco_edges_mev is None else np.asarray(reco_edges_mev, float)
        )
        self.e_nu = default_e_nu_grid() if e_nu_grid_mev is None else np.asarray(e_nu_grid_mev, float)
        self.response = response or DetectorResponse()
        self.cross_section = cross_section or load_ibd_cross_section()
        self.include_backgrounds = include_backgrounds

        self.n_energy = self.reco_edges.size - 1
        self.n_samples = len(self.samples)

        # (n_bins, n_e_nu) detector response, shared by all samples
        self._resp = self.response.matrix(self.e_nu, self.reco_edges)
        self._weights = integration_weights(self.e_nu)
        self._xsec = self.cross_section(self.e_nu)

        # Per sample: list of (baseline_km, density) with density already
        # containing flux * xsec * N_p * eff * T * weights.
        self._kernels: list[list[tuple[float, np.ndarray]]] = []
        self._backgrounds: list[np.ndarray] = []
        for sample in self.samples:
            entries = []
            for core in sample.cores:
                flux = core.flux_at_detector(self.e_nu, burnup=sample.burnup)
                density = (
                    flux
                    * self._xsec
                    * sample.detector.target_protons
                    * sample.detector.efficiency
                    * sample.exposure_days
                    * SECONDS_PER_DAY
                    * self._weights
                )
                entries.append((core.baseline_km, density))
            self._kernels.append(entries)
            self._backgrounds.append(self._background_counts(sample))

    # -- geometry -----------------------------------------------------------
    @property
    def prompt_centers(self) -> np.ndarray:
        return 0.5 * (self.reco_edges[:-1] + self.reco_edges[1:])

    @property
    def energy_centers_tiled(self) -> np.ndarray:
        return np.tile(self.prompt_centers, self.n_samples)

    @property
    def sample_groups(self) -> list[str]:
        return [s.group for s in self.samples]

    @property
    def exposures_days(self) -> np.ndarray:
        return np.array([s.exposure_days for s in self.samples], dtype=float)

    def _background_counts(self, sample: Sample) -> np.ndarray:
        if not sample.background_rates or not self.include_backgrounds:
            return np.zeros(self.n_energy)
        from .backgrounds import background_counts

        total = np.zeros(self.n_energy)
        for name, rate in sample.background_rates:
            total += background_counts(name, self.reco_edges, sample.exposure_days, rate)
        return total

    # -- prediction ---------------------------------------------------------
    def counts(
        self,
        params: OscillationParameters = DEFAULT_OSCILLATION_PARAMS,
        *,
        oscillate: bool = True,
        include_backgrounds: bool | None = None,
        matter: bool = False,
        signal_scale: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """Predicted counts, shape ``(n_samples, n_energy_bins)``."""

        include_bkg = self.include_backgrounds if include_backgrounds is None else include_backgrounds
        prob_fn = survival_probability_matter if matter else survival_probability_ee

        scale = np.atleast_1d(np.asarray(signal_scale, dtype=float))
        if scale.size == 1:
            scale = np.full(self.n_samples, scale[0])

        cache: dict[float, np.ndarray] = {}
        out = np.empty((self.n_samples, self.n_energy))
        for idx, entries in enumerate(self._kernels):
            total = np.zeros_like(self.e_nu)
            for baseline, density in entries:
                if oscillate:
                    p = cache.get(baseline)
                    if p is None:
                        p = prob_fn(self.e_nu, baseline, params)
                        cache[baseline] = p
                    total += density * p
                else:
                    total += density
            out[idx] = scale[idx] * (self._resp @ total)
            if include_bkg:
                out[idx] = out[idx] + self._backgrounds[idx]
        return out

    def counts_with_probability(self, prob_fn, *, include_backgrounds: bool | None = None):
        """Predicted counts using an arbitrary ``prob_fn(e_nu_mev, baseline_km)``.

        Useful for isolating individual pieces of the oscillation probability,
        e.g. replacing the fast atmospheric term by its average.
        """

        include_bkg = (
            self.include_backgrounds if include_backgrounds is None else include_backgrounds
        )
        cache: dict[float, np.ndarray] = {}
        out = np.empty((self.n_samples, self.n_energy))
        for idx, entries in enumerate(self._kernels):
            total = np.zeros_like(self.e_nu)
            for baseline, density in entries:
                p = cache.get(baseline)
                if p is None:
                    p = np.asarray(prob_fn(self.e_nu, baseline), dtype=float)
                    cache[baseline] = p
                total += density * p
            out[idx] = self._resp @ total
            if include_bkg:
                out[idx] = out[idx] + self._backgrounds[idx]
        return out

    def flat_counts(self, *args, **kwargs) -> np.ndarray:
        return self.counts(*args, **kwargs).reshape(-1)

    def background_counts(self) -> np.ndarray:
        return np.asarray(self._backgrounds)

    def rate_per_day(
        self,
        params: OscillationParameters = DEFAULT_OSCILLATION_PARAMS,
        **kwargs,
    ) -> np.ndarray:
        """Total signal events per day for each sample."""

        counts = self.counts(params, include_backgrounds=False, **kwargs)
        return counts.sum(axis=1) / self.exposures_days

    # -- derivatives --------------------------------------------------------
    def jacobian(
        self,
        params: OscillationParameters,
        parameter_names,
        *,
        rel_step: float = 1.0e-3,
        **kwargs,
    ) -> np.ndarray:
        """Central-difference d(counts)/d(parameter), shape (n_par, n_bins_total)."""

        rows = []
        for name in parameter_names:
            value = getattr(params, name)
            step = abs(value) * rel_step if value != 0.0 else rel_step
            up = self.flat_counts(params.replace(**{name: value + step}), **kwargs)
            dn = self.flat_counts(params.replace(**{name: value - step}), **kwargs)
            rows.append((up - dn) / (2.0 * step))
        return np.asarray(rows)

    # -- reconfiguration ----------------------------------------------------
    def with_exposures(self, exposure_days) -> "Predictor":
        """A new predictor with the same samples at different live times."""

        exposure_days = np.atleast_1d(np.asarray(exposure_days, dtype=float))
        if exposure_days.size == 1:
            exposure_days = np.full(self.n_samples, exposure_days[0])
        samples = [s.with_exposure(t) for s, t in zip(self.samples, exposure_days)]
        return Predictor(
            samples,
            reco_edges_mev=self.reco_edges,
            e_nu_grid_mev=self.e_nu,
            response=self.response,
            cross_section=self.cross_section,
            include_backgrounds=self.include_backgrounds,
        )

    def scaled_exposure(self, factor: float) -> "Predictor":
        return self.with_exposures(self.exposures_days * factor)
