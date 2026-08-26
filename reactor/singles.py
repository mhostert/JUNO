"""Non-reactor single-event backgrounds in the JUNO liquid scintillator.

These are the backgrounds that limit any single-hit measurement (E$\\nu$ES here,
ALPs in notebook 7) and that the reactor-off running measures directly.  Two
families:

**(A) Natural radioactivity.**  JUNO's first data measure the singles rate in
the fiducial volume [arXiv:2607.17509]: **4.7 Hz above 0.7 MeV for R < 17.2 m**,
against a design budget of 7.2 Hz (LS 2.2 + materials 5.0).  The measured LS is
U/Th <~ 3.4e-17 g/g -- one to two orders below the 1e-15 design, i.e. the
"Ideal" radiopurity scenario of the solar programme -- so the internal
component is only ~0.2 Hz and **~95% of the fiducial singles are external
gammas** from the acrylic vessel, PMT glass and the steel truss.  That matters
for the spectrum: an external gamma deposits at most its own energy, so the
external component **stops at the 2.615 MeV 208Tl line**, whereas an internal
decay deposits the *full* Q (beta + the whole gamma cascade) and 208Tl
therefore reaches 5.0 MeV.

**(B) Cosmogenic isotopes.**  At 650 m depth the muon-induced isotopes are what
survive above the gamma wall.  11C (Q = 1.98 MeV) is why JUNO's own 8B analysis
cannot go below 2 MeV; 10C, 11Be and 12B reach higher.

Grade of the numbers
--------------------
The radioactivity *rate* is anchored to JUNO's measurement, but its *spectral
shape* here is a component model (lines + Compton continua + beta shapes), not
a Geant4 transport, and the *radial profile* is an effective exponential tuned
to their Fig. 6 rather than a gamma attenuation length -- their own
reconstruction systematic on the FV singles is 20-30%.  The cosmogenic rates
are depth-scaled estimates good to a factor of a few.  Everything therefore
carries a deliberately generous scale uncertainty, and the analysis that uses
it (reactor on/off, :mod:`reactor.near_sm`) is built so that the answer depends
on these rates only through their *statistical* fluctuation, not their assumed
level.
"""

from __future__ import annotations

import numpy as np

from .constants import SECONDS_PER_DAY

M_E = 0.51099895            # MeV
ALPHA = 1.0 / 137.035999

#: Measured JUNO fiducial singles, E > 0.7 MeV, R < 17.2 m [arXiv:2607.17509 Tab. 12].
JUNO_SINGLES_HZ = 4.7
JUNO_SINGLES_RADIUS_M = 17.2
JUNO_SINGLES_EMIN = 0.7

#: Measured LS radiopurity [arXiv:2607.17509 Tab. 3, ICP-MS, 95% CL upper
#: limits].  The design requirement was 1e-15 g/g, so the delivered scintillator
#: is ~30x purer and the internal component collapses to <~0.1 Hz -- their
#: Sec. 4 quotes the deficit as worth ~2 Hz of the 2.2 Hz LS budget.
LS_U238_GG = 3.4e-17
LS_TH232_GG = 3.3e-17
LS_MASS_G = 2.0e10                 # 20 kt

#: Bq per gram of the parent isotope.
SPECIFIC_ACTIVITY = {"U238": 1.244e4, "Th232": 4.057e3}

#: Effective attenuation length of the *reconstructed* external-gamma rate with
#: depth into the LS.  A 2.6 MeV gamma has a true attenuation length of ~30 cm,
#: but the measured radial profile (their Fig. 6) is much shallower because it
#: is dominated by vertex-reconstruction tails near the vessel -- the source of
#: their quoted 20-30% FV systematic.  We use the shallow, pessimistic value.
EXTERNAL_LAMBDA_M = 0.80

#: Composite external-gamma line list: (E_gamma [MeV], relative intensity).
#: 214Bi and 208Tl from the U/Th chains dominate, plus 40K and 60Co from the
#: steel.  Intensities are branching ratios weighted into a representative
#: material mix; only the *shape* matters, the total is normalised to data.
EXTERNAL_LINES = [
    (0.583, 0.30), (0.609, 0.46), (0.727, 0.07), (0.911, 0.26),
    (0.969, 0.16), (1.120, 0.15), (1.173, 0.12), (1.332, 0.12),
    (1.461, 0.35), (1.764, 0.15), (2.204, 0.05), (2.448, 0.016),
    (2.615, 0.36),
]

#: Fraction of external gammas reaching the fiducial volume that deposit their
#: full energy (the rest give a Compton continuum).
EXTERNAL_FULL_FRACTION = 0.40

#: Internal LS decays reaching above 1 MeV: name -> (Q_visible [MeV], parent
#: chain, branching within the chain).  Q_visible is the *total* energy released
#: to beta + the full gamma cascade: inside a 20 kt scintillator the cascade is
#: contained, which is why 208Tl reaches 5.0 MeV internally while an *external*
#: 208Tl gamma stops at its 2.615 MeV line.  This contrast is what sets the
#: background above 3 MeV.
INTERNAL_COMPONENTS = {
    "214Bi": (3.272, "U238", 1.00),
    "234Pam": (2.269, "U238", 1.00),
    "214Pb": (1.024, "U238", 1.00),
    "208Tl": (5.001, "Th232", 0.36),
    "212Bi": (2.254, "Th232", 0.64),
    "228Ac": (2.124, "Th232", 1.00),
}

#: 208Tl follows the 212Bi alpha within T_1/2 = 3.05 min, so it can be tagged by
#: the preceding alpha.  Borexino-style rejection; conservative default.
TL208_TAG_EFFICIENCY = 0.0

#: Cosmogenic isotopes at JUNO's 650 m depth: name -> (Q_visible [MeV],
#: lifetime [s], rate [per day in 20 kt], beta+ ?).
#:
#: ESTIMATE GRADE: depth-scaled from Borexino/KamLAND yields and JUNO
#: projections; individually good to a factor of a few.  ``rate_scale`` on
#: :class:`SinglesBackground` scales them all, and the notebook shows the
#: sensitivity is insensitive to it under reactor-off subtraction.
COSMOGENIC = {
    "11C":  (1.982, 1766.0, 1.9e4, True),
    "10C":  (3.648, 27.8, 5.0e2, True),
    "11Be": (11.51, 19.9, 5.0e1, False),
    "12B":  (13.37, 29.1e-3, 9.0e2, False),
    "6He":  (3.508, 1.16, 1.0e2, False),
    "8Li":  (16.00, 1.21, 4.0e1, False),
    "9Li":  (13.61, 0.257, 8.0e1, False),
}

#: Rejection efficiency of a realistic JUNO-like veto suite: a 5 ms full-detector
#: veto after every muon (their 95% live-time strategy), a ~1.2 s / 3 m
#: track-correlated veto for the short-lived species, and a threefold
#: (muon + neutron + decay) tag for 11C as used by Borexino.
VETO_EFFICIENCY = {
    "11C": 0.90, "10C": 0.90, "11Be": 0.50, "12B": 0.99,
    "6He": 0.95, "8Li": 0.95, "9Li": 0.98,
}


def _fermi(t_kin, z, beta_plus):
    """Non-relativistic Fermi function, adequate for spectral shapes."""

    e_tot = t_kin + M_E
    p = np.sqrt(np.maximum(e_tot**2 - M_E**2, 1e-12))
    eta = (-1.0 if beta_plus else 1.0) * z * ALPHA * e_tot / p
    x = 2.0 * np.pi * eta
    return np.where(np.abs(x) < 1e-6, 1.0, x / (1.0 - np.exp(-x)))


def beta_spectrum(t_kin, q_value, z_daughter=6, beta_plus=False):
    """Allowed beta spectrum dN/dT, normalised to unit area over [0, Q]."""

    t = np.atleast_1d(np.asarray(t_kin, dtype=float))
    out = np.zeros_like(t)
    m = (t > 0) & (t < q_value)
    if not m.any():
        return out
    e_tot = t[m] + M_E
    p = np.sqrt(np.maximum(e_tot**2 - M_E**2, 0.0))
    out[m] = (p * e_tot * (q_value - t[m]) ** 2
              * _fermi(t[m], z_daughter, beta_plus))
    area = np.trapezoid(out, t)
    return out / area if area > 0 else out


def compton_spectrum(t_kin, e_gamma):
    """Klein-Nishina electron-recoil spectrum, normalised to unit area."""

    t = np.atleast_1d(np.asarray(t_kin, dtype=float))
    a = e_gamma / M_E
    t_max = e_gamma * 2.0 * a / (1.0 + 2.0 * a)
    out = np.zeros_like(t)
    m = (t > 0) & (t < t_max)
    if not m.any():
        return out
    s = t[m] / e_gamma
    # KN in terms of the fractional energy transfer
    out[m] = (1.0 + (1.0 - s) ** 2
              - (1.0 - s) * (2.0 * s / (a * (1.0 - s)))
              * (2.0 + a * s / (1.0 - s) * (1.0 - s)))
    out[m] = np.maximum(out[m], 0.0)
    area = np.trapezoid(out, t)
    return out / area if area > 0 else out


class SinglesBackground:
    """Radioactivity + cosmogenic singles in a chosen fiducial volume.

    Parameters
    ----------
    fiducial_radius_m
        Analysis fiducial radius.  The default 16.5 m is the value used
        throughout notebooks 5-7; it sits 0.7 m inside JUNO's own 17.2 m
        baseline FV, which suppresses the external-gamma component that
        dominates the measured singles.
    rate_scale
        Overall multiplier on *everything* (default 1).  The rates are
        estimate-grade; the notebook scans this.
    resolution
        Callable sigma(E) [MeV] applied to every component, or None.
    """

    def __init__(
        self,
        fiducial_radius_m: float = 16.5,
        rate_scale: float = 1.0,
        cosmogenic_scale: float = 1.0,
        apply_veto: bool = True,
        resolution=None,
        lambda_m: float = EXTERNAL_LAMBDA_M,
        vessel_radius_m: float = 17.7,
        n_grid: int = 4000,
        e_max: float = 16.0,
    ):
        self.fiducial_radius_m = fiducial_radius_m
        self.rate_scale = rate_scale
        self.cosmogenic_scale = cosmogenic_scale
        self.apply_veto = apply_veto
        self.lambda_m = lambda_m
        self.vessel_radius_m = vessel_radius_m
        self._res = resolution
        self._e = np.linspace(1e-3, e_max, n_grid)

        # -- geometry.  External gammas are surface-peaked and attenuate into
        # the volume; internal decays and cosmogenics scale with mass.
        r0, r = JUNO_SINGLES_RADIUS_M, fiducial_radius_m
        self.external_scale = np.exp(-(r0 - r) / lambda_m)
        self.mass_fraction = (r / r0) ** 3

        # Internal component from the MEASURED concentrations, not from a split
        # of the total: each chain member decays at (mass x g/g x specific
        # activity), so 214Bi runs at ~730/day and 208Tl at ~84/day in 20 kt.
        self._int_shapes, int_hz = self._internal()

        # Whatever the measured 4.7 Hz is not accounted for internally is
        # external -- which is essentially all of it.
        ext_hz = (JUNO_SINGLES_HZ - int_hz / self.mass_fraction) * self.external_scale

        self._ext_shape = self._external_shape()
        self.external_hz = max(ext_hz, 0.0) * rate_scale
        self.internal_hz = int_hz * rate_scale
        self._cosmo_shapes, self.cosmogenic_hz = self._cosmogenic()

    # -- component shapes (per MeV, unit area above JUNO_SINGLES_EMIN) -------
    def _norm_above_emin(self, y):
        m = self._e >= JUNO_SINGLES_EMIN
        area = np.trapezoid(y[m], self._e[m])
        return y / area if area > 0 else y

    def _external_shape(self):
        y = np.zeros_like(self._e)
        width = 0.02                       # placeholder line width, re-smeared below
        for eg, w in EXTERNAL_LINES:
            y += w * (1.0 - EXTERNAL_FULL_FRACTION) * compton_spectrum(self._e, eg)
            y += (w * EXTERNAL_FULL_FRACTION
                  * np.exp(-0.5 * ((self._e - eg) / width) ** 2)
                  / (width * np.sqrt(2 * np.pi)))
        return self._norm_above_emin(self._smear(y))

    def _internal(self):
        """Per-isotope internal shapes and rates [Hz] from measured g/g."""

        conc = {"U238": LS_U238_GG, "Th232": LS_TH232_GG}
        shapes, total = {}, 0.0
        for name, (q, chain, br) in INTERNAL_COMPONENTS.items():
            act = LS_MASS_G * conc[chain] * SPECIFIC_ACTIVITY[chain]   # Bq, 20 kt
            hz = act * br * self.mass_fraction
            if name == "208Tl":
                hz *= (1.0 - TL208_TAG_EFFICIENCY)
            shapes[name] = (self._smear(beta_spectrum(self._e, q)), hz)
            total += hz
        return shapes, total

    def _cosmogenic(self):
        shapes, total = {}, 0.0
        for name, (q, _tau, rate_day, bplus) in COSMOGENIC.items():
            eff = VETO_EFFICIENCY.get(name, 0.0) if self.apply_veto else 0.0
            hz = (rate_day / SECONDS_PER_DAY * (1.0 - eff)
                  * self.mass_fraction * self.rate_scale * self.cosmogenic_scale)
            # beta+ deposits T + 2 m_e (the annihilation gammas)
            t = self._e - (2.0 * M_E if bplus else 0.0)
            y = beta_spectrum(t, q - (2.0 * M_E if bplus else 0.0), beta_plus=bplus)
            area = np.trapezoid(y, self._e)
            shapes[name] = (self._smear(y / area if area > 0 else y), hz)
            total += hz
        return shapes, total

    def _smear(self, y):
        if self._res is None:
            return y
        e = self._e
        sig = np.maximum(self._res(np.maximum(e, 0.1)), 1e-3)
        # Gaussian smearing on the (uniform) grid
        step = e[1] - e[0]
        out = np.zeros_like(y)
        nz = np.nonzero(y > 0)[0]
        for i in nz:
            s = sig[i]
            lo, hi = max(0, int(i - 5 * s / step)), min(len(e), int(i + 5 * s / step) + 1)
            k = np.exp(-0.5 * ((e[lo:hi] - e[i]) / s) ** 2)
            ks = k.sum()
            if ks > 0:
                out[lo:hi] += y[i] * k / ks
        return out

    # -- public API ---------------------------------------------------------
    def spectrum(self, edges) -> np.ndarray:
        """Total singles rate per bin [per second] on the given bin edges."""

        return sum(self.components(edges).values())

    def components(self, edges) -> dict:
        """Per-component singles rate per bin [per second]."""

        out = {"external gammas": self.external_hz * self._bin(self._ext_shape, edges)}
        for name, (shape, hz) in self._int_shapes.items():
            if hz > 0:
                out[f"internal {name}"] = hz * self.rate_scale * self._bin(shape, edges)
        for name, (shape, hz) in self._cosmo_shapes.items():
            if hz > 0:
                out[f"cosmogenic {name}"] = hz * self._bin(shape, edges)
        return out

    def _bin(self, shape, edges):
        """Integrate a per-MeV shape into bins."""

        e = self._e
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (shape[1:] + shape[:-1])
                                               * np.diff(e))])
        return np.diff(np.interp(edges, e, cum))

    def rate_per_day(self, e_min: float = 1.0, e_max: float = 6.5) -> float:
        edges = np.array([e_min, e_max])
        return float(self.spectrum(edges)[0]) * SECONDS_PER_DAY

    def summary(self, e_min: float = 1.0, e_max: float = 6.5) -> list:
        edges = np.array([e_min, e_max])
        rows = []
        for name, v in self.components(edges).items():
            rows.append([name, float(v[0]) * SECONDS_PER_DAY])
        rows.sort(key=lambda r: -r[1])
        return rows
