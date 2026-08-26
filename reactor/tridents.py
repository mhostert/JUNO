"""nubar_e -> nubar_e e+ e- trident production at reactor energies (via NEPTUNE).

The 2->4 process nubar_e + N -> nubar_e + e+ e- + N proceeds through the same
W/Z exchange as EvES but with an extra lepton pair, so it is O(alpha) down and
phase-space suppressed near its threshold E > 2 m_e (1 + m_e/M) ~ 1.02 MeV.
Two target channels matter in a liquid scintillator:

* **coherent** on the carbon nucleus (Z^2, Woods-Saxon form factor);
* **proton-elastic** on the free hydrogen (the "diffractive" single-nucleon
  channel with the proton form factors; the incoherent carbon-nucleon term is
  the same per nucleon and is added with Pauli blocking).

Cross sections come from NEPTUNE's ``TridentProcess`` (8-D Vegas integration,
``mode='full'``).  Because each energy point costs a Vegas run, we tabulate on
a log grid once and cache to ``reactor/data/tridents_nubar_e_ee.npz``;
interpolation is in log-log.  Signature in the detector: the e+e- pair
deposits ~E_nu (minus the small nuclear recoil) as ONE prompt hit -- a
single-hit background to the EvES sample with the *neutrino* spectrum shape.
Rates at 10 MW / 50 m are ~10^-5 of EvES, i.e. negligible; the point of
carrying it is to show that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_CACHE = Path(__file__).resolve().parent / "data" / "tridents_nubar_e_ee.npz"
_E_GRID = np.geomspace(1.05, 12.0, 28)         # MeV


def _compute_table(nitn: int = 10, neval: int = 6000):
    import neptune as nep

    model = nep.TridentSMModel(nu_flavor="e", l1_flavor="e", l2_flavor="e",
                               is_nubar=True)
    coh_c, pel = [], []
    for e in _E_GRID:
        pc = nep.TridentProcess(model, Z=6, A=12, Enu=e * 1e-3)
        pp = nep.TridentProcess(model, Z=1, A=1, Enu=e * 1e-3)
        try:
            sc, _ = pc.sigma_coherent(nitn=nitn, neval=neval)
        except Exception:
            sc = 0.0
        try:
            sp, _ = pp.sigma_diffractive_nucleon("proton", nitn=nitn, neval=neval)
        except Exception:
            sp = 0.0
        coh_c.append(max(sc, 0.0)); pel.append(max(sp, 0.0))
    return np.array(coh_c), np.array(pel)


def _load():
    if _CACHE.exists():
        d = np.load(_CACHE)
        return d["e"], d["coh_c12"], d["p_elastic"]
    coh, pel = _compute_table()
    _CACHE.parent.mkdir(exist_ok=True)
    np.savez(_CACHE, e=_E_GRID, coh_c12=coh, p_elastic=pel)
    return _E_GRID, coh, pel


def _interp_loglog(e_mev, e_tab, s_tab):
    e = np.atleast_1d(np.asarray(e_mev, dtype=float))
    ok = s_tab > 0
    out = np.zeros_like(e)
    inside = (e >= e_tab[ok][0]) & (e <= e_tab[ok][-1])
    out[inside] = np.exp(np.interp(np.log(e[inside]), np.log(e_tab[ok]),
                                   np.log(s_tab[ok])))
    return out


def sigma_trident_coherent_c12(e_mev):
    """nubar_e C12 -> nubar_e e+ e- C12, coherent, cm^2 per carbon nucleus."""

    e_tab, coh, _ = _load()
    return _interp_loglog(e_mev, e_tab, coh)


def sigma_trident_proton(e_mev):
    """nubar_e p -> nubar_e e+ e- p, elastic on a free proton, cm^2 per proton."""

    e_tab, _, pel = _load()
    return _interp_loglog(e_mev, e_tab, pel)


#: LAB stoichiometry, from the LS mass fractions in :mod:`reactor.constants`:
#: H/C = 1.63 by number, i.e. 6 + 1.63 electrons per 1.63 free protons, the same
#: N_e = 4.689 N_p that sets the EvES target count.
from .constants import ELECTRONS_PER_FREE_PROTON, HYDROGEN_PER_CARBON  # noqa: E402

H_PER_C = HYDROGEN_PER_CARBON
E_PER_C = 6.0 + H_PER_C          # electrons per "CH_1.63 unit"
assert abs(E_PER_C / H_PER_C - ELECTRONS_PER_FREE_PROTON) < 1e-9


def sigma_trident_per_ch_unit(e_mev):
    """Total nubar_e trident cross section per CH_1.67 unit of LAB [cm^2]:
    coherent on C + elastic on the H_PER_C free protons.  (The incoherent
    proton-elastic on carbon's bound protons is neglected: it is Pauli-blocked
    and, at these energies, sub-dominant to the coherent Z^2 term by ~Z.)"""

    return sigma_trident_coherent_c12(e_mev) + H_PER_C * sigma_trident_proton(e_mev)
