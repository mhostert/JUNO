"""Detector energy response: non-linearity, resolution, and response matrices.

The prompt-energy chain implemented here is

    E_nu  ->  E_e+ (mean positron total energy, IBD kinematics)
          ->  E_dep = E_e+ + m_e            (positron KE + 2 annihilation gammas)
          ->  E_vis = f_NL(E_dep) * E_dep   (liquid-scintillator non-linearity)
          ->  E_rec  (Gaussian smearing with sigma(E_vis))

``f_NL`` is built from a Birks-quenching + Cherenkov model of the liquid
scintillator light yield, which reproduces the characteristic shape of the
published Daya Bay / JUNO LSNL curves.  A purely empirical parameterisation is
also available for quick studies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import erf

from .constants import (
    DRAFT_RESOLUTION_A,
    JUNO_RESOLUTION_ABC,
    M_E,
    PROMPT_ENERGY_OFFSET_MEV,
)

# ---------------------------------------------------------------------------
# Energy resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyResolution:
    """sigma_E / E = sqrt( (a/sqrt(E))^2 + b^2 + (c/E)^2 ), E = E_vis in MeV.

    ``a`` is the photostatistics term, ``b`` the residual (calibration /
    spatial non-uniformity) term, and ``c`` the dark-noise term.
    """

    a: float = JUNO_RESOLUTION_ABC[0]
    b: float = JUNO_RESOLUTION_ABC[1]
    c: float = JUNO_RESOLUTION_ABC[2]

    @classmethod
    def sqrt_only(cls, a: float = DRAFT_RESOLUTION_A) -> "EnergyResolution":
        """The simplified sigma_E = a sqrt(E) model used in the draft."""

        return cls(a=a, b=0.0, c=0.0)

    def sigma(self, e_vis_mev: np.ndarray | float) -> np.ndarray:
        e = np.maximum(np.asarray(e_vis_mev, dtype=float), 1.0e-6)
        return np.sqrt((self.a**2) * e + (self.b * e) ** 2 + self.c**2)

    def relative(self, e_vis_mev: np.ndarray | float) -> np.ndarray:
        e = np.maximum(np.asarray(e_vis_mev, dtype=float), 1.0e-6)
        return self.sigma(e) / e

    def effective_a(self, e_vis_mev: np.ndarray | float = 1.0) -> np.ndarray:
        """Equivalent single-term a such that a_eff/sqrt(E) = sigma/E."""

        e = np.asarray(e_vis_mev, dtype=float)
        return self.relative(e) * np.sqrt(e)

    def scaled(self, factor: float) -> "EnergyResolution":
        return replace(self, a=self.a * factor, b=self.b * factor, c=self.c * factor)

    @classmethod
    def fit_to_points(
        cls,
        energies_mev: np.ndarray,
        rel_resolution: np.ndarray,
        errors=None,
        relative_floor: float = 0.02,
    ) -> "EnergyResolution":
        """Fit a, b, c to measured sigma_E/E points.

        The quoted statistical errors on JUNO's calibration peaks are at the
        1e-4 level, far below the source-to-source spread the three-parameter
        form can accommodate, so a ``relative_floor`` (default 2% of the
        resolution) is added in quadrature.  Without it the fit is driven by
        one or two peaks and reports a meaningless chi-square.
        """

        from scipy.optimize import least_squares

        e = np.asarray(energies_mev, dtype=float)
        r = np.asarray(rel_resolution, dtype=float)
        sigma = np.hypot(
            np.zeros_like(r) if errors is None else np.asarray(errors, dtype=float),
            relative_floor * r,
        )

        def residual(p):
            return (np.sqrt(p[0] ** 2 / e + p[1] ** 2 + p[2] ** 2 / e**2) - r) / sigma

        sol = least_squares(residual, [0.03, 0.008, 0.010],
                            bounds=([0.0, 0.0, 0.0], [0.2, 0.1, 0.1]))
        return cls(a=float(sol.x[0]), b=float(sol.x[1]), c=float(sol.x[2]))

    @classmethod
    def from_juno_calibration(cls) -> "EnergyResolution":
        """Fit to the eight calibration peaks of the JUNO 2025 release."""

        from .juno_data import load_energy_resolution

        e, r, err, _ = load_energy_resolution()
        return cls.fit_to_points(e, r, err)


# ---------------------------------------------------------------------------
# Liquid-scintillator non-linearity
# ---------------------------------------------------------------------------
# LAB-based scintillator bulk properties
LS_DENSITY_G_CM3 = 0.859
LS_Z_OVER_A = 0.5385  # C18H30-like
LS_MEAN_EXCITATION_EV = 64.7
LS_REFRACTIVE_INDEX = 1.50


def electron_stopping_power(t_mev: np.ndarray) -> np.ndarray:
    """Electron collision stopping power in MeV cm^2 / g (Berger-Seltzer)."""

    t = np.maximum(np.asarray(t_mev, dtype=float), 1.0e-4)
    tau = t / M_E
    gamma = tau + 1.0
    beta2 = 1.0 - 1.0 / gamma**2
    i_over_mec2 = LS_MEAN_EXCITATION_EV * 1.0e-6 / M_E

    # F^-(tau) for electrons
    f_minus = 1.0 - beta2 + (tau**2 / 8.0 - (2.0 * tau + 1.0) * np.log(2.0)) / gamma**2

    coeff = 0.1535 * LS_Z_OVER_A / beta2  # 2 pi r_e^2 m_e c^2 N_A = 0.1535 MeV cm^2/g
    log_term = np.log(tau**2 * (tau + 2.0) / (2.0 * i_over_mec2**2))
    return coeff * (log_term + f_minus)


def cherenkov_yield_density(t_mev: np.ndarray) -> np.ndarray:
    """Relative Cherenkov photon yield per unit path length, arbitrary units."""

    t = np.maximum(np.asarray(t_mev, dtype=float), 1.0e-6)
    gamma = t / M_E + 1.0
    beta2 = 1.0 - 1.0 / gamma**2
    n2 = LS_REFRACTIVE_INDEX**2
    sin2_theta_c = 1.0 - 1.0 / np.maximum(beta2 * n2, 1.0e-12)
    return np.where(beta2 * n2 > 1.0, np.maximum(sin2_theta_c, 0.0), 0.0)


@dataclass(frozen=True)
class NonLinearity:
    """Liquid-scintillator energy non-linearity, E_vis / E_dep.

    The light yield of an electron of kinetic energy ``T`` is modelled as

        L(T) = S(T) + fc * C(T)
        S(T) = int_0^T dT' / (1 + kB dE/dx(T'))          (Birks quenching)
        C(T) = int_0^T dT' sin^2(theta_C)(T') / (dE/dx)  (Cherenkov, per track length)

    A prompt IBD event of deposited energy ``E_dep`` consists of a positron of
    kinetic energy ``E_dep - 2 m_e`` plus two 511 keV annihilation gammas,
    which deposit through several Compton electrons and are therefore quenched
    much more strongly than a single energetic track.  That mismatch, together
    with the sub-threshold suppression of the Cherenkov component, produces the
    characteristic low-energy deficit of the published LSNL curves.

    Parameters
    ----------
    kb:
        Birks constant in cm / MeV.
    fc:
        Weight of the Cherenkov component relative to the scintillation
        component (in the internal units of ``C``, i.e. cm vs MeV).
    gamma_effective_mev:
        Mean Compton-electron energy used to model the 511 keV gammas.
    norm_energy_mev:
        Deposited energy at which the curve is normalised to unity, mimicking
        the experimental calibration convention.
    empirical:
        If given, use ``f(E) = p0 (1 + p1 exp(-p2 E^p3))`` instead of the
        physics model.
    """

    kb: float = 0.0065
    fc: float = 0.55
    gamma_effective_mev: float = 0.24
    norm_energy_mev: float = 12.0
    empirical: tuple[float, float, float, float] | None = None

    # -- physics model ------------------------------------------------------
    def _light_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative light L(T) on a fixed kinetic-energy grid (cached)."""

        key = (self.kb, self.fc)
        cache = _LIGHT_TABLE_CACHE.get(key)
        if cache is not None:
            return cache

        t = np.concatenate([np.linspace(0.0, 0.2, 400)[:-1], np.linspace(0.2, 15.0, 1500)])
        t_mid = np.maximum(t, 1.0e-4)
        dedx = electron_stopping_power(t_mid) * LS_DENSITY_G_CM3  # MeV / cm

        scint_integrand = 1.0 / (1.0 + self.kb * dedx)
        cher_integrand = cherenkov_yield_density(t_mid) / dedx  # cm / MeV

        light = np.concatenate(
            [[0.0], np.cumsum(0.5 * (scint_integrand[1:] + scint_integrand[:-1]) * np.diff(t))]
        ) + self.fc * np.concatenate(
            [[0.0], np.cumsum(0.5 * (cher_integrand[1:] + cher_integrand[:-1]) * np.diff(t))]
        )
        _LIGHT_TABLE_CACHE[key] = (t, light)
        return t, light

    def electron_light(self, t_mev: np.ndarray | float) -> np.ndarray:
        """Total light emitted by an electron of kinetic energy T (arb. units)."""

        t_grid, light = self._light_table()
        return np.interp(np.asarray(t_mev, dtype=float), t_grid, light)

    def _light_for_prompt(self, e_dep_mev: np.ndarray) -> np.ndarray:
        e_dep = np.asarray(e_dep_mev, dtype=float)
        t_positron = np.maximum(e_dep - 2.0 * M_E, 0.0)

        n_compton = 2.0 * M_E / self.gamma_effective_mev
        light_gamma = n_compton * float(self.electron_light(self.gamma_effective_mev))
        return self.electron_light(t_positron) + light_gamma

    # -- public API ---------------------------------------------------------
    def _raw(self, e_dep_mev: np.ndarray) -> np.ndarray:
        if self.empirical is not None:
            p0, p1, p2, p3 = self.empirical
            return p0 * (1.0 + p1 * np.exp(-p2 * np.maximum(e_dep_mev, 1e-6) ** p3))
        return self._light_for_prompt(e_dep_mev) / np.maximum(e_dep_mev, 1.0e-9)

    def factor(self, e_dep_mev: np.ndarray | float) -> np.ndarray:
        """E_vis / E_dep as a function of the deposited prompt energy."""

        e_dep = np.atleast_1d(np.asarray(e_dep_mev, dtype=float))
        out = self._raw(e_dep) / float(self._raw(np.array([self.norm_energy_mev]))[0])
        return out if np.ndim(e_dep_mev) else float(out[0])

    def visible_energy(self, e_dep_mev: np.ndarray | float) -> np.ndarray:
        return np.asarray(e_dep_mev, dtype=float) * self.factor(e_dep_mev)

    # -- tuning -------------------------------------------------------------
    @classmethod
    def fit_to_curve(
        cls,
        e_dep_mev: np.ndarray,
        target_factor: np.ndarray,
        *,
        norm_energy_mev: float = 12.0,
        p0: tuple[float, float, float] = (0.0065, 0.30, 0.30),
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (1.0e-4, 0.0, 0.05),
            (0.05, 3.0, 0.51),
        ),
    ) -> "NonLinearity":
        """Least-squares fit of (kB, fc, E_gamma_eff) to a target LSNL curve."""

        from scipy.optimize import least_squares

        e_dep = np.asarray(e_dep_mev, dtype=float)
        target = np.asarray(target_factor, dtype=float)

        def residual(theta):
            model = cls(
                kb=theta[0],
                fc=theta[1],
                gamma_effective_mev=theta[2],
                norm_energy_mev=norm_energy_mev,
            )
            return model.factor(e_dep) - target

        sol = least_squares(
            residual, x0=np.array(p0), bounds=bounds, xtol=1e-14, ftol=1e-14
        )
        return cls(
            kb=float(sol.x[0]),
            fc=float(sol.x[1]),
            gamma_effective_mev=float(sol.x[2]),
            norm_energy_mev=norm_energy_mev,
        )


_LIGHT_TABLE_CACHE: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

IDENTITY_NL = NonLinearity(empirical=(1.0, 0.0, 1.0, 1.0))

# Representative Daya Bay / JUNO positron LSNL curve (E_vis/E_dep vs prompt
# deposited energy), normalised to unity at high energy.  These are indicative
# values read off the published curves, accurate to roughly half a percent;
# they are used only to tune and validate the physics model above.
def juno_nonlinearity(shape_only: bool = True) -> "NonLinearity":
    """Birks + Cherenkov model fitted to JUNO's released positron NL curve.

    The released curve is normalised so that ``E_vis/E_true`` crosses unity near
    3.6 MeV, whereas this model normalises to unity at ``norm_energy_mev``.  With
    ``shape_only=True`` (the default) the released curve is renormalised to the
    same convention before fitting, so the comparison is of shape alone -- which
    is the physically meaningful part, since the absolute scale is set by
    calibration rather than by the light-yield model.

    Note the fitted ``kb`` is an *effective* quenching parameter: the released
    curve is the full positron non-linearity, including the instrumental
    component from PMT charge reconstruction, which this model has no term for.
    """

    global _JUNO_NL
    key = bool(shape_only)
    if _JUNO_NL.get(key) is None:
        from .juno_data import load_nonlinearity

        e, f, _, _ = load_nonlinearity("positron")
        norm = 12.0
        target = f / np.interp(norm, e, f) if shape_only else f
        sel = (e >= 1.022) & (e <= 11.0)
        _JUNO_NL[key] = NonLinearity.fit_to_curve(
            e[sel][::8], target[sel][::8], norm_energy_mev=norm,
            bounds=((1.0e-4, 0.0, 0.02), (0.2, 5.0, 0.51)),
        )
    return _JUNO_NL[key]


_JUNO_NL: dict = {}


@dataclass(frozen=True)
class TabulatedNonLinearity:
    """Measured energy non-linearity, interpolated from a released curve.

    This is the one to use for anything quantitative about JUNO: it is the
    collaboration's fitted ``E_vis / E_true`` including both scintillator and
    instrumental non-linearity, whereas :class:`NonLinearity` is a two-parameter
    physics model of the scintillator part alone.

    ``shift`` moves the curve within its published uncertainty band, in units
    of that band (``+1`` = upper edge), which is how the residual non-linearity
    systematic is propagated.
    """

    energy_mev: np.ndarray
    factor_table: np.ndarray
    err_low: np.ndarray
    err_high: np.ndarray
    shift: float = 0.0
    renormalise_at: float | None = None

    @classmethod
    def from_release(cls, kind: str = "positron", **kwargs) -> "TabulatedNonLinearity":
        from .juno_data import load_nonlinearity

        e, f, lo, hi = load_nonlinearity(kind)
        return cls(energy_mev=e, factor_table=f, err_low=lo, err_high=hi, **kwargs)

    def factor(self, e_dep_mev: np.ndarray | float) -> np.ndarray:
        e = np.asarray(e_dep_mev, dtype=float)
        table = self.factor_table
        if self.shift > 0:
            table = table + self.shift * np.abs(self.err_high)
        elif self.shift < 0:
            table = table - abs(self.shift) * np.abs(self.err_low)
        out = np.interp(e, self.energy_mev, table)
        if self.renormalise_at is not None:
            out = out / np.interp(self.renormalise_at, self.energy_mev, table)
        return out

    def visible_energy(self, e_dep_mev: np.ndarray | float) -> np.ndarray:
        return np.asarray(e_dep_mev, dtype=float) * self.factor(e_dep_mev)

    def with_shift(self, shift: float) -> "TabulatedNonLinearity":
        return replace(self, shift=shift)

    def band(self, e_dep_mev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper edges of the published uncertainty band."""

        return self.with_shift(-1.0).factor(e_dep_mev), self.with_shift(+1.0).factor(e_dep_mev)


# ---------------------------------------------------------------------------
# Non-linearity uncertainty basis
# ---------------------------------------------------------------------------
def nl_uncertainty_basis(
    e_dep_mev: np.ndarray,
    amplitude: float = 0.01,
    n_curves: int = 4,
) -> np.ndarray:
    """Pull-style basis of residual non-linearity distortions.

    Returns an ``(n_curves, n_energy)`` array of fractional distortions of the
    visible-energy scale, each normalised to ``amplitude`` at 1 MeV and
    decreasing with energy, mimicking the residual LSNL curves that JUNO and
    Daya Bay propagate as a systematic.
    """

    e = np.asarray(e_dep_mev, dtype=float)
    x = np.log(np.maximum(e, 1.0e-3))
    x = (x - x.min()) / max(x.max() - x.min(), 1.0e-9)
    basis = np.stack([np.cos(np.pi * k * x) for k in range(n_curves)])
    # Normalise each curve to `amplitude` in the max-norm.
    basis = basis / np.max(np.abs(basis), axis=1, keepdims=True) * amplitude
    return basis


# ---------------------------------------------------------------------------
# Response matrices
# ---------------------------------------------------------------------------
def integration_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal integration weights for a monotonically increasing grid."""

    x = np.asarray(x, dtype=float)
    weights = np.empty_like(x)
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    return weights


def gaussian_bin_response(
    true_energy_mev: np.ndarray,
    reco_edges_mev: np.ndarray,
    sigma_mev: np.ndarray,
) -> np.ndarray:
    """Probability that an event of true energy E lands in each reco bin.

    Returns an ``(n_bins, n_true)`` matrix.
    """

    true_e = np.asarray(true_energy_mev, dtype=float)
    edges = np.asarray(reco_edges_mev, dtype=float)
    sigma = np.maximum(np.asarray(sigma_mev, dtype=float), 1.0e-6)
    root2 = np.sqrt(2.0)
    arg_hi = (edges[1:, None] - true_e[None, :]) / (root2 * sigma[None, :])
    arg_lo = (edges[:-1, None] - true_e[None, :]) / (root2 * sigma[None, :])
    return 0.5 * (erf(arg_hi) - erf(arg_lo))


@dataclass(frozen=True)
class DetectorResponse:
    """Full E_nu -> reconstructed prompt-energy response."""

    resolution: EnergyResolution = EnergyResolution()
    nonlinearity: "NonLinearity | TabulatedNonLinearity | None" = None
    use_ibd_recoil: bool = True

    def deposited_energy(self, e_nu_mev: np.ndarray) -> np.ndarray:
        """Mean deposited prompt energy for a given neutrino energy."""

        e_nu = np.asarray(e_nu_mev, dtype=float)
        if self.use_ibd_recoil:
            from .cross_sections import positron_energy

            return positron_energy(e_nu, order=1) + M_E
        return e_nu - PROMPT_ENERGY_OFFSET_MEV

    def visible_energy(self, e_nu_mev: np.ndarray) -> np.ndarray:
        e_dep = self.deposited_energy(e_nu_mev)
        if self.nonlinearity is None:
            return e_dep
        return self.nonlinearity.visible_energy(e_dep)

    def _cache_key(self, e_nu_mev, reco_edges_mev, tag, rel_step=0.0):
        # Grids are long-lived, reused objects in this codebase, so identity is
        # a safe and cheap cache key.
        return (
            id(self),
            id(e_nu_mev),
            e_nu_mev.shape,
            id(reco_edges_mev),
            reco_edges_mev.shape,
            tag,
            rel_step,
        )

    def matrix(self, e_nu_mev: np.ndarray, reco_edges_mev: np.ndarray) -> np.ndarray:
        """(n_reco_bins, n_e_nu) response matrix."""

        key = self._cache_key(e_nu_mev, reco_edges_mev, "matrix")
        cached = _RESPONSE_CACHE.get(key)
        if cached is not None:
            return cached

        e_vis = self.visible_energy(e_nu_mev)
        sigma = self.resolution.sigma(e_vis)
        out = gaussian_bin_response(e_vis, reco_edges_mev, sigma)
        _cache_store(key, out, self, e_nu_mev, reco_edges_mev)
        return out

    def matrix_derivative_scale(
        self, e_nu_mev: np.ndarray, reco_edges_mev: np.ndarray, rel_step: float = 1.0e-3
    ) -> np.ndarray:
        """d(response)/d(fractional resolution scale), by central difference."""

        key = self._cache_key(e_nu_mev, reco_edges_mev, "dscale", rel_step)
        cached = _RESPONSE_CACHE.get(key)
        if cached is not None:
            return cached

        up = replace(self, resolution=self.resolution.scaled(1.0 + rel_step))
        dn = replace(self, resolution=self.resolution.scaled(1.0 - rel_step))
        out = (
            up.matrix(e_nu_mev, reco_edges_mev) - dn.matrix(e_nu_mev, reco_edges_mev)
        ) / (2.0 * rel_step)
        _cache_store(key, out, self, e_nu_mev, reco_edges_mev)
        return out

    def matrix_derivative_energy_scale(
        self, e_nu_mev: np.ndarray, reco_edges_mev: np.ndarray, rel_step: float = 1.0e-3
    ) -> np.ndarray:
        """d(response)/d(fractional absolute energy scale)."""

        key = self._cache_key(e_nu_mev, reco_edges_mev, "descale", rel_step)
        cached = _RESPONSE_CACHE.get(key)
        if cached is not None:
            return cached

        e_vis = self.visible_energy(e_nu_mev)
        sigma = self.resolution.sigma(e_vis)
        up = gaussian_bin_response(e_vis * (1.0 + rel_step), reco_edges_mev, sigma)
        dn = gaussian_bin_response(e_vis * (1.0 - rel_step), reco_edges_mev, sigma)
        out = (up - dn) / (2.0 * rel_step)
        _cache_store(key, out, self, e_nu_mev, reco_edges_mev)
        return out


_RESPONSE_CACHE: dict = {}
_RESPONSE_CACHE_PINS: list = []
_RESPONSE_CACHE_MAX = 64


def _cache_store(key, value, *pins) -> None:
    """Store a response matrix.

    The keyed objects are pinned so their ``id()`` cannot be recycled by the
    garbage collector while the entry is live, which would otherwise make
    identity-keyed lookups return the wrong matrix.
    """

    if len(_RESPONSE_CACHE) >= _RESPONSE_CACHE_MAX:
        clear_response_cache()
    _RESPONSE_CACHE[key] = value
    _RESPONSE_CACHE_PINS.append(pins)


def clear_response_cache() -> None:
    """Drop cached response matrices (call after mutating shared grids)."""

    _RESPONSE_CACHE.clear()
    _RESPONSE_CACHE_PINS.clear()


def juno_2025_response(nonlinearity: bool = True) -> DetectorResponse:
    """Detector response matching the JUNO 2025 first-measurement release.

    Uses the resolution fitted to the eight calibration peaks and, by default,
    the released positron non-linearity curve.  With ``nonlinearity=True`` the
    reconstructed energy this response produces is the **visible** energy, which
    is what the released spectrum is binned in -- the signal prediction extends
    below the 1.022 MeV kinematic minimum precisely because of it.
    """

    return DetectorResponse(
        resolution=EnergyResolution.from_juno_calibration(),
        nonlinearity=TabulatedNonLinearity.from_release("positron") if nonlinearity else None,
        use_ibd_recoil=True,
    )
