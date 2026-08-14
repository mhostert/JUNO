"""Loaders for the JUNO 2025 first-measurement data release.

Files live in ``reactor/data/JUNO_data_release_2025`` and accompany

    JUNO Collaboration, "First measurement of reactor neutrino oscillations at
    JUNO", arXiv:2511.14593.

Everything the release provides is exposed here in analysis-ready form:

* :func:`load_spectrum` -- the 66-bin measured prompt-energy spectrum with the
  collaboration's best-fit signal and each background component, converted from
  the file's ``events / 0.1 MeV`` convention to counts per bin.
* :func:`load_chi2_map` -- the official 100x100 Delta chi^2 surface in
  (sin^2 theta12, Delta m^2_21).  This supersedes any digitized contour.
* :func:`load_energy_resolution` -- the eight calibration peaks used to
  characterise sigma_E/E.
* :func:`load_nonlinearity` -- the fitted positron / electron / gamma energy
  non-linearity curves with their uncertainty bands.
* :func:`load_scintillator_nl`, :func:`load_b12`, :func:`load_c11` -- the
  calibration inputs behind the non-linearity model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

RELEASE = "data/JUNO_data_release_2025"

#: Background components as named in the release.
BACKGROUND_COLUMNS = {
    "9Li/8He": "Npred_9Li8He",
    "geoneutrino": "Npred_Geonu",
    "world reactors": "Npred_WorldRea",
    "214Bi-214Po": "Npred_Bipo",
    "other": "Npred_OtherBkg",
}


def data_path(name: str) -> Path:
    with resources.as_file(resources.files("reactor").joinpath(f"{RELEASE}/{name}")) as path:
        return Path(path)


def _read_csv(name: str, comment: str = "#") -> dict[str, np.ndarray]:
    """Minimal CSV reader returning a dict of columns (no pandas dependency)."""

    path = data_path(name)
    with open(path, encoding="utf-8-sig") as handle:
        lines = [ln for ln in handle if ln.strip() and not ln.lstrip().startswith(comment)]
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if all(p == "" for p in parts):
            continue
        rows.append(parts)

    out: dict[str, np.ndarray] = {}
    for i, name_i in enumerate(header):
        if not name_i:
            continue
        col = [r[i] if i < len(r) else "" for r in rows]
        try:
            out[name_i] = np.array([float(v) if v else np.nan for v in col])
        except ValueError:
            out[name_i] = np.array(col, dtype=object)
    return out


# ---------------------------------------------------------------------------
# Measured spectrum (Fig. 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JUNOSpectrum:
    """The 59.1-day prompt-energy spectrum, in counts per bin.

    The release stores every column as ``events / 0.1 MeV``; the bin widths vary
    from 0.1 MeV in the core of the spectrum to 2.6 MeV in the last bin, so all
    columns here have already been multiplied by ``width / 0.1 MeV``.
    """

    edges: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    n_obs: np.ndarray
    n_obs_err: np.ndarray
    pred_best_fit: np.ndarray
    pred_signal: np.ndarray
    backgrounds: dict[str, np.ndarray]
    livetime_days: float

    @property
    def background_total(self) -> np.ndarray:
        return np.sum(list(self.backgrounds.values()), axis=0)

    @property
    def n_total(self) -> float:
        return float(self.n_obs.sum())

    def rate_per_day(self, counts: np.ndarray) -> float:
        return float(np.sum(counts) / self.livetime_days)

    def rebin_matrix(self, fine_edges: np.ndarray) -> np.ndarray:
        """Matrix mapping a fine histogram onto these bins.

        Returns an ``(n_coarse, n_fine)`` matrix of overlap fractions, so that
        ``M @ fine_counts`` gives the counts in the release binning.
        """

        fine_edges = np.asarray(fine_edges, dtype=float)
        lo_f, hi_f = fine_edges[:-1], fine_edges[1:]
        lo_c, hi_c = self.edges[:-1, None], self.edges[1:, None]
        overlap = np.clip(np.minimum(hi_c, hi_f) - np.maximum(lo_c, lo_f), 0.0, None)
        return overlap / np.maximum(hi_f - lo_f, 1e-12)


@lru_cache(maxsize=1)
def load_spectrum(livetime_days: float = 59.1) -> JUNOSpectrum:
    """Load the measured prompt-energy spectrum of Fig. 3a."""

    c = _read_csv("fig3_panel_a_data.csv")
    centers, widths = c["Ec"], c["Ewid"]
    scale = widths / 0.1  # release convention: events / 0.1 MeV

    edges = np.concatenate([[centers[0] - widths[0] / 2.0], centers + widths / 2.0])
    backgrounds = {name: c[col] * scale for name, col in BACKGROUND_COLUMNS.items()}

    return JUNOSpectrum(
        edges=edges,
        centers=centers,
        widths=widths,
        n_obs=c["Nobs"] * scale,
        n_obs_err=c["Nobs_err"] * scale,
        pred_best_fit=c["Npred_best_fit"] * scale,
        pred_signal=c["Npred_signal"] * scale,
        backgrounds=backgrounds,
        livetime_days=livetime_days,
    )


@lru_cache(maxsize=1)
def load_survival_probability() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fig. 3c: measured oscillated / non-oscillated ratio and its error."""

    c = _read_csv("fig3_panel_c_data.csv")
    return c["Ec"], c["Ewid"], c["Pee_meas"], c["Pee_err_meas"]


@lru_cache(maxsize=1)
def load_residuals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fig. 3b: per-bin residuals between data and the full model, in sigma."""

    c = _read_csv("fig3_panel_b_data.csv")
    key = next(k for k in c if k not in ("Ec", "Ewid"))
    return c["Ec"], c["Ewid"], c[key]


# ---------------------------------------------------------------------------
# Official Delta chi^2 surface (Fig. 4)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_chi2_map() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fig. 4: the official Delta chi^2 surface.

    Returns ``(sin2_theta12, dm2_21_1e5, chi2)`` with ``chi2`` of shape
    ``(len(dm2_21), len(sin2_theta12))`` so it can be contoured directly
    against the two grid vectors.
    """

    c = _read_csv("fig4_JUNO_DeltaChi2_59.1days.csv")
    dm2 = c["Delta m^2_21 [x 10^{-5} eV^2]"]
    s12 = c["sin^2(theta_12)"]
    chi2 = c["Delta chi^2"]

    dm2_grid = np.unique(dm2)
    s12_grid = np.unique(s12)
    grid = np.full((dm2_grid.size, s12_grid.size), np.nan)
    i = np.searchsorted(dm2_grid, dm2)
    j = np.searchsorted(s12_grid, s12)
    grid[i, j] = chi2
    return s12_grid, dm2_grid, grid


def chi2_map_best_fit() -> tuple[float, float, float]:
    """Best-fit point and minimum of the official surface."""

    s12, dm2, grid = load_chi2_map()
    k = np.unravel_index(np.nanargmin(grid), grid.shape)
    return float(s12[k[1]]), float(dm2[k[0]]), float(grid[k])


# ---------------------------------------------------------------------------
# Detector response
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_energy_resolution() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Measured resolution at the calibration peaks.

    Returns ``(E_rec_MeV, sigma_over_E, sigma_over_E_error, labels)`` with the
    resolutions as fractions rather than percent.
    """

    c = _read_csv("calibration_source_energy_resolution.csv")
    labels = [str(v) for v in c["peak_type"]]
    keep = [i for i, v in enumerate(labels) if v and v != "nan"]
    return (
        c["E_rec (MeV)"][keep],
        c["resolution (%)"][keep] / 100.0,
        c["resolution error (%)"][keep] / 100.0,
        [labels[i] for i in keep],
    )


@lru_cache(maxsize=8)
def load_nonlinearity(kind: str = "positron"):
    """Fitted energy non-linearity curve E_vis/E_dep with its uncertainty band.

    ``kind`` is ``'positron'`` (the one that matters for IBD), ``'electron'``
    or ``'gamma'``.  Returns ``(E_true, factor, err_low, err_high)``; the two
    error columns are signed as in the release (low negative, high positive).
    """

    if kind not in ("positron", "electron", "gamma"):
        raise KeyError(f"kind must be positron/electron/gamma, got {kind!r}")
    c = _read_csv(f"edfig2_{kind}_FullNL.csv")
    return c["True E (MeV)"], c["Full NL"], c["Error Low"], c["Error High"]


@lru_cache(maxsize=1)
def load_scintillator_nl():
    """Scintillator-only non-linearity from the gamma sources: data and best fit."""

    d = _read_csv("edfig2_gamma_ScintNL_data.csv")
    f = _read_csv("edfig2_gamma_ScintNL_bestfit.csv")
    return d, f


@lru_cache(maxsize=1)
def load_b12():
    """Cosmogenic 12B spectrum used to constrain the non-linearity."""

    return _read_csv("edfig2_B12_data.csv"), _read_csv("edfig2_bestfit_B12withN12.csv")


@lru_cache(maxsize=1)
def load_c11():
    """Cosmogenic 11C positron spectrum used to constrain the non-linearity."""

    return _read_csv("edfig2_C11_data.csv"), _read_csv("edfig2_C11_bestfit.csv")
