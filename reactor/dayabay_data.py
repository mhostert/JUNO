"""Loaders for the Daya Bay 2025 comprehensive flux and spectrum release.

Files live in ``reactor/data/DayaBay_release_2025/anc`` and accompany

    Daya Bay Collaboration, "Comprehensive Measurement of the Reactor
    Antineutrino Spectrum and Flux at Daya Bay", Phys. Rev. Lett. 134, 201802
    (2025).

The quantity measured is the **IBD yield per fission**,
``sigma_f(E) = S(E) * sigma_IBD(E)`` in units of
``1e-43 cm^2 / fission / MeV`` -- flux and cross section already combined.  It
therefore replaces the Huber-Mueller spectrum *and* the cross-section table in
one step, and it carries the two features a summation model does not reproduce:
the ~6% flux deficit and the 5 MeV bump.

Three products are provided:

* ``unfolded`` -- spectra in true antineutrino energy (1.8-9.5 MeV), the ones
  to use as an input model, with covariance and the SVD smearing matrix;
* ``prompt`` -- the same in reconstructed prompt energy (0.7-8 MeV), for
  comparison against a detector-level prediction;
* ``flux`` -- 20 fuel-evolution groups giving effective fission fractions and
  the aggregated IBD yield, which fixes the fuel composition the spectra
  correspond to.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

RELEASE = "data/DayaBay_release_2025/anc"
ISOTOPE_TAGS = ("U235", "Pu239", "PuCombo", "Total")


def data_path(name: str) -> Path:
    with resources.as_file(resources.files("reactor").joinpath(f"{RELEASE}/{name}")) as path:
        return Path(path)


def _threshold_shape(energy_mev):
    """Huber-Mueller x sigma_IBD, used only to shape the threshold half-bin."""

    from .cross_sections import load_ibd_cross_section
    from .flux import juno_average_fractions, mixed_spectrum_per_fission

    e = np.atleast_1d(np.asarray(energy_mev, dtype=float))
    out = mixed_spectrum_per_fission(e, juno_average_fractions()) * load_ibd_cross_section()(e)
    return out if np.ndim(energy_mev) else float(out[0])


@dataclass(frozen=True)
class YieldSpectrum:
    """Binned IBD yield per fission, in cm^2 / fission / MeV."""

    edges: np.ndarray
    values: np.ndarray
    errors: np.ndarray

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def integral(self) -> float:
        """Total IBD yield per fission, cm^2 / fission."""

        return float(np.sum(self.values * self.widths))

    def __call__(
        self,
        energy_mev: np.ndarray | float,
        interpolate: bool = True,
        taper: bool = True,
    ) -> np.ndarray:
        """Yield density at the given energies, zero outside the measured range.

        Linear interpolation runs between bin centres.  Below the first centre
        the release gives no shape information, but the yield must vanish at the
        IBD threshold 1.806 MeV: with ``taper=True`` the Huber-Mueller shape,
        renormalised to match at the first centre, is used there.  Holding the
        first bin flat instead over-predicts the threshold half-bin by ~80%,
        which lands squarely in JUNO's first prompt-energy bin.
        """

        e = np.asarray(energy_mev, dtype=float)
        if interpolate:
            out = np.interp(e, self.centers, self.values,
                            left=self.values[0], right=self.values[-1])
        else:
            idx = np.clip(np.searchsorted(self.edges, e, side="right") - 1,
                          0, len(self.values) - 1)
            out = self.values[idx]

        if taper:
            low = e < self.centers[0]
            if np.any(low):
                ref = float(np.atleast_1d(_threshold_shape(self.centers[0]))[0])
                shape = np.atleast_1d(_threshold_shape(e)) / max(ref, 1e-300)
                out = np.where(low, self.values[0] * shape.reshape(np.shape(out)), out)
        return np.where((e >= self.edges[0]) & (e <= self.edges[-1]), out, 0.0)


def _parse_blocks(name: str) -> tuple[dict[str, YieldSpectrum], list[np.ndarray]]:
    """Split one release file into its named spectra and its trailing matrices."""

    lines = data_path(name).read_text().splitlines()
    spectra: dict[str, list[list[float]]] = {}
    matrices: list[list[list[float]]] = []
    current: str | None = None
    matrix: list[list[float]] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            tag = stripped.lstrip("#").strip()
            if tag in ISOTOPE_TAGS:
                current = tag
                spectra[current] = []
            elif "matrix" in tag.lower():
                current = None
                if matrix:
                    matrices.append(matrix)
                    matrix = []
            continue
        if not stripped:
            continue
        parts = stripped.split()
        try:
            row = [float(v) for v in parts]
        except ValueError:
            continue
        if current is not None and len(row) == 4:
            spectra[current].append(row)
        elif len(row) > 4:
            matrix.append(row)
    if matrix:
        matrices.append(matrix)

    out = {}
    for tag, rows in spectra.items():
        arr = np.asarray(rows, dtype=float)
        edges = np.concatenate([arr[:, 0], arr[-1:, 1]])
        # released in 1e-43 cm^2/fission/MeV
        out[tag] = YieldSpectrum(edges, arr[:, 2] * 1e-43, arr[:, 3] * 1e-43)
    return out, [np.asarray(m, dtype=float) for m in matrices]


def load_unfolded(pu_combo: bool = False):
    """Public wrapper; normalises the argument so that ``load_unfolded()`` and
    ``load_unfolded(False)`` share one cache entry rather than parsing twice and
    returning two independent sets of objects."""

    return _load_unfolded(bool(pu_combo))


@lru_cache(maxsize=2)
def _load_unfolded(pu_combo: bool):
    """Unfolded spectra in true antineutrino energy.

    Returns ``(spectra, covariance, smearing)``.  The covariance is the 75x75
    matrix over the three stacked spectra; ``smearing`` is the additional SVD
    smearing matrix, which must be applied to a *model* before comparing it with
    these points -- but not when using them as an input spectrum, which is what
    :func:`dayabay_yield_model` does.
    """

    name = (
        "DYB_unfolded_spectra_tot_U235_PuCombo.txt"
        if pu_combo
        else "DYB_unfolded_spectra_tot_U235_Pu239.txt"
    )
    spectra, matrices = _parse_blocks(name)
    cov = matrices[0] if matrices else None
    smear = matrices[1] if len(matrices) > 1 else None
    return spectra, cov, smear


@lru_cache(maxsize=1)
def load_prompt():
    """Spectra in reconstructed prompt energy, with their covariances."""

    spectra, matrices = _parse_blocks("DYB_prompt_spectra.txt")
    return spectra, matrices


@lru_cache(maxsize=1)
def load_flux_evolution() -> dict[str, np.ndarray]:
    """The 20 fuel-evolution groups: fission fractions and aggregated IBD yield."""

    rows = []
    for line in data_path("DYB_flux.txt").read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 6 and not line.strip().startswith("#"):
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    arr = np.asarray(rows, dtype=float)
    return {
        "group": arr[:, 0],
        "f235": arr[:, 1],
        "f238": arr[:, 2],
        "f239": arr[:, 3],
        "f241": arr[:, 4],
        "sigma_f": arr[:, 5] * 1e-43,
    }


def dayabay_mean_fractions() -> dict[str, float]:
    """Fission fractions the released total spectrum corresponds to."""

    f = load_flux_evolution()
    return {
        "U235": float(np.mean(f["f235"])),
        "U238": float(np.mean(f["f238"])),
        "Pu239": float(np.mean(f["f239"])),
        "Pu241": float(np.mean(f["f241"])),
    }


# ---------------------------------------------------------------------------
# Yield model usable as a drop-in reactor spectrum
# ---------------------------------------------------------------------------
@dataclass
class DayaBayYield:
    """IBD yield per fission as a function of neutrino energy.

    Calling the object returns ``sigma_f(E)`` in cm^2 / fission / MeV, i.e. the
    reactor spectrum with the IBD cross section already folded in.

    ``fractions`` optionally re-weights the measured total to a different fuel
    composition, using the measured U235 and Pu239 spectra for those two
    isotopes and Huber-Mueller for U238 and Pu241, which the fuel-evolution fit
    cannot separate.  JUNO's cycle-average composition differs from Daya Bay's
    by at most 1.6% in any fraction, so this is a small correction.

    Above the measured range the Huber-Mueller shape is used, normalised to
    match at the last measured bin; the reactor rate there is negligible.
    """

    pu_combo: bool = False
    fractions: dict[str, float] | None = None
    extrapolate: bool = True
    bin_scales: np.ndarray | None = None
    """Per-bin multiplicative factors on the measured total spectrum.

    Used to propagate the released covariance: perturbing one bin and
    re-evaluating gives the derivative of any prediction with respect to that
    bin's measured content.
    """

    def __post_init__(self) -> None:
        spectra, _, _ = load_unfolded(self.pu_combo)
        self.total = spectra["Total"]
        if self.bin_scales is not None:
            scales = np.asarray(self.bin_scales, dtype=float)
            if scales.shape != self.total.values.shape:
                raise ValueError(
                    f"bin_scales must have {self.total.values.shape[0]} entries"
                )
            self.total = YieldSpectrum(self.total.edges,
                                       self.total.values * scales,
                                       self.total.errors * scales)
        self.u235 = spectra["U235"]
        self.pu = spectra["PuCombo" if self.pu_combo else "Pu239"]
        self.reference_fractions = dayabay_mean_fractions()

    def _isotope_correction(self, e: np.ndarray) -> np.ndarray:
        if self.fractions is None:
            return np.zeros_like(e)

        from .cross_sections import load_ibd_cross_section
        from .flux import normalize_fractions, spectrum_per_fission

        target = normalize_fractions(self.fractions)
        ref = self.reference_fractions
        xsec = load_ibd_cross_section()(e)

        correction = np.zeros_like(e)
        correction += (target["U235"] - ref["U235"]) * self.u235(e)
        pu_key = "Pu239"
        correction += (target[pu_key] - ref[pu_key]) * self.pu(e)
        for iso in ("U238", "Pu241"):
            correction += (target[iso] - ref[iso]) * spectrum_per_fission(e, iso) * xsec
        return correction

    def __call__(self, energy_mev: np.ndarray | float) -> np.ndarray:
        e = np.atleast_1d(np.asarray(energy_mev, dtype=float))
        out = self.total(e) + self._isotope_correction(e)

        if self.extrapolate:
            hi = self.total.edges[-1]
            above = e > hi
            if np.any(above):
                from .cross_sections import load_ibd_cross_section
                from .flux import mixed_spectrum_per_fission

                mix = self.fractions or self.reference_fractions
                xsec = load_ibd_cross_section()
                model = lambda x: mixed_spectrum_per_fission(x, mix) * xsec(x)
                edge = self.total.values[-1]
                scale = edge / max(float(model(np.array([self.total.centers[-1]]))[0]), 1e-300)
                out = np.where(above, model(e) * scale, out)
        return out if np.ndim(energy_mev) else float(out[0])


def dayabay_yield_model(fractions: dict[str, float] | None = None, **kwargs) -> DayaBayYield:
    """Convenience constructor for :class:`DayaBayYield`."""

    return DayaBayYield(fractions=fractions, **kwargs)
