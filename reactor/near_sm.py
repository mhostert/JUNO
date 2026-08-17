"""A fixed near reactor at JUNO: Standard-Model precision with nu-e scattering.

A compact HALEU reactor (default 10 MW_th at 50 m) parked next to the detector.
The IBD channel is enormous there; the interesting one is **elastic
neutrino-electron scattering** (EvES), whose cross section carries the weak
mixing angle through the neutral-current couplings

    nubar_e + e   : Cv = 1/2 + 2 sin^2 theta_W          (CC + NC)
    nubar_mu/tau  : Cv = -1/2 + 2 sin^2 theta_W         (NC only)
    Ca            : +-1/2

Cross sections come from **NEPTUNE** (``neptune.nu_electron``); the parametric
(gV, gA) version implemented here is the identical tree-level formula with the
couplings freed, and is validated against NEPTUNE at the SM point to machine
precision (:meth:`FixedNearReactor.validate_against_neptune`).

The neutrino *flux* is the measured Daya Bay U235 IBD yield divided by the
Vogel-Beacom cross section -- i.e. the measured U235 antineutrino spectrum --
plus the small U238 (Huber-Mueller) component of the fresh HALEU core.  It is
therefore defined only above the IBD threshold, 1.806 MeV: recoil rates below
T ~ 1.5 MeV miss the (unmeasured) low-energy tail of the reactor spectrum and
are conservative.

Flavors: at the near baseline the survival probability is essentially 1
(1 - P_ee ~ 1e-5 at 50 m), but the nubar_mu/nubar_tau component from
oscillation is carried exactly, with its own (NC-only) cross section, so the
same object works at any baseline.

Backgrounds: the request covers the **IBD single-hit background** -- IBD events
whose delayed neutron tag is missed, promoting the positron (prompt) signal to
a single hit inside the EvES window.  This is modelled as the IBD prompt
spectrum times an untagged fraction (default 1%), with its own normalisation
prior (default 10%; it is measurable in situ from the tagged sample).
Radioactivity and cosmogenic singles are *not* modelled and at low recoil they
are known to be severe; quoted precisions are therefore reactor-limited ones.

Statistics: Asimov, C = diag(mu) + sum_k v_k v_k^T as in ``reactor.theta13``.
The prediction is affine in {CL^2, CR^2, CL*CR} per flavor, so three template
spectra per flavor make every coupling scan instantaneous.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .constants import (
    CM_PER_KM,
    JUNO2025_TARGET_PROTONS,
    OscillationParameters,
    SECONDS_PER_DAY,
)
from .cross_sections import vogel_beacom
from .dayabay_data import load_unfolded
from .detector import (
    DetectorResponse,
    EnergyResolution,
    TabulatedNonLinearity,
    gaussian_bin_response,
    integration_weights,
)
from .flux import fission_rate_per_second, haleu_fractions, spectrum_per_fission
from .oscillations import survival_probability_ee
from .theta13 import DEFAULT_TRUTH

#: Electrons per free (hydrogen) proton in the JUNO liquid scintillator.
#: LAB is close to CnH2n: mass fractions ~88% C / ~12% H give
#: (0.88 * 6/12 + 0.12) / 0.12 = 4.67 electrons per free proton.
ELECTRONS_PER_FREE_PROTON = 4.67
JUNO_ELECTRONS = ELECTRONS_PER_FREE_PROTON * JUNO2025_TARGET_PROTONS

#: Reference (gV, gA) point: the PDG 2024 world average of the neutrino-electron
#: NC couplings, gV = -0.040 +- 0.015, gA = -0.507 +- 0.014, which coincides with
#: the radiatively-corrected SM prediction (gV = -0.0395, gA = -0.5063; PDG
#: "Electroweak model" review).  In the tree-level parameterisation this is an
#: effective sin^2 theta_W = (gV + 1/2)/2 = 0.230, the low-energy value.
#: NEPTUNE's own default (tree level, sw2 = 0.2223, gA = -1/2) is used only in
#: the cross-section validation, where the parametric formula is compared to it
#: at the same couplings.
GV_SM = -0.040
GA_SM = -0.507
SW2_SM = (GV_SM + 0.5) / 2.0            # 0.230, effective
SW2_NEPTUNE = 0.2223                     # NEPTUNE's tree-level default

#: Electron binding energies [MeV] and electron counts for the LS "average"
#: electron: carbon 1s/2s/2p and hydrogen 1s.  The atomic (Kopeikin stepping)
#: correction: an electron participates in EvES only for T above its binding.
#: Exact null for T >~ 1 keV -- demonstrated, not assumed, in notebook 5.
_C_ELEC_FRACTION = (0.8799 * 6 / 12.011) / (0.8799 * 6 / 12.011 + 0.1201 / 1.008)
_BINDINGS_C = ((2.842e-4, 2), (1.94e-5, 2), (1.126e-5, 2))   # (B [MeV], count)
_BINDING_H = 1.36e-5


def atomic_stepping(t_mev) -> np.ndarray:
    """Active-electron fraction at true recoil energy T (stepping correction)."""

    t = np.asarray(t_mev, dtype=float)
    carbon = sum(n * (t > b) for b, n in _BINDINGS_C) / 6.0
    hydrogen = (t > _BINDING_H) * 1.0
    return _C_ELEC_FRACTION * carbon + (1.0 - _C_ELEC_FRACTION) * hydrogen


#: priors for the standard JUNO backgrounds under the IBD anchor spectrum
NEAR_IBD_BACKGROUND_PRIORS = {
    "far reactors": 0.02, "geoneutrino": 0.42, "9Li/8He": 0.33,
    "world reactors": 0.10, "214Bi-214Po": 0.56, "other": 1.00,
}

# Constants taken from NEPTUNE itself, so that "the cross sections come from
# NEPTUNE" is true to the last digit.  NB: NEPTUNE's GeV2_to_cm2 = 3.9204e-28
# sits 0.68% above (hbar c)^2 = 3.8938e-28 GeV^2 cm^2 -- flagged upstream; it
# rescales absolute EvES rates only, and is absorbed by the normalisation
# systematic in every sensitivity below.
try:
    from neptune import const as _nc
    M_E = _nc.m_e * 1e3                       # MeV
    GF = _nc.Gf * 1e-6                        # MeV^-2
    GEV2_TO_CM2 = _nc.GeV2_to_cm2
except ImportError:                            # pragma: no cover
    M_E = 0.511
    GF = 1.1663787e-5 * 1e-6
    GEV2_TO_CM2 = 3.893793721e-28
MEV2_TO_CM2 = GEV2_TO_CM2 * 1e6


def eves_dsigma_dT(e_nu_mev, t_mev, flavor: str = "e", nubar: bool = True,
                   gv: float = GV_SM, ga: float = GA_SM):
    """Tree-level d sigma/dT for nu + e -> nu + e, cm^2/MeV.

    ``(gv, ga)`` are the *neutral-current* electron couplings; the charged
    current adds (+1, +1) for the electron flavor.  Defaults reproduce
    NEPTUNE's SM exactly (same formula, couplings freed).
    """

    e = np.asarray(e_nu_mev, dtype=float)
    t = np.asarray(t_mev, dtype=float)
    cv = gv + (1.0 if flavor == "e" else 0.0)
    ca = ga + (1.0 if flavor == "e" else 0.0)
    if nubar:
        cl, cr = 0.5 * (cv - ca), 0.5 * (cv + ca)
    else:
        cl, cr = 0.5 * (cv + ca), 0.5 * (cv - ca)
    pref = 2.0 * M_E * GF**2 / np.pi * MEV2_TO_CM2
    ds = pref * (cl**2 + cr**2 * (1.0 - t / e) ** 2 - cl * cr * M_E * t / e**2)
    return np.where(t < 2.0 * e**2 / (M_E + 2.0 * e), ds, 0.0)


class FixedNearReactor:
    """IBD and EvES spectra, and coupling sensitivities, for a parked reactor."""

    def __init__(
        self,
        power_mwth: float = 10.0,
        baseline_m: float = 50.0,
        burnup: float = 0.0,
        truth: OscillationParameters = DEFAULT_TRUTH,
        efficiency: float = 0.8,
        duty_cycle: float = 0.9,
        # EvES analysis window and binning (reconstructed recoil energy, MeV)
        recoil_edges: np.ndarray | None = None,
        # IBD singles
        untagged_fraction: float = 0.01,
        # systematics priors
        sigma_norm: float = 0.02,
        sigma_u238: float = 0.15,
        sigma_ibd_singles: float = 0.10,
        use_flux_covariance: bool = True,
        # joint IBD+EvES fit: the IBD channel anchors the flux in situ
        ibd_edges: np.ndarray | None = None,
        sigma_ibd_xsec: float = 0.002,
        sigma_ibd_xsec_tilt: float = 0.001,
        sigma_channel_ratio: float = 0.005,
        joint_flux_prior: float = 0.10,
        # solar + standard JUNO backgrounds, and richer systematics
        include_backgrounds: bool = True,
        sigma_solar: float = 0.03,
        include_tridents: bool = False,   # nubar_e -> nubar_e e+e- tridents: 1e-5 of EvES, off
        sigma_trident: float = 0.10,
        sigma_scale: float = 0.005,
        sigma_bias: float = 0.005,
        sigma_res: float = 0.05,
        sigma_evolution: float = 0.30,
        resolution_a: float = 0.033,
        resolution_b: float = 0.010,
        n_e_grid: int = 1200,
    ):
        self.power_mwth = power_mwth
        self.baseline_m = baseline_m
        self.truth = truth
        self.efficiency = efficiency
        self.duty_cycle = duty_cycle
        self.untagged_fraction = untagged_fraction
        self.recoil_edges = (np.arange(1.0, 6.5 + 1e-9, 0.05)
                             if recoil_edges is None else recoil_edges)

        enu = np.linspace(1.806, 10.0, n_e_grid)
        self.e_nu_grid = enu
        w_e = integration_weights(enu)

        # -- the measured U235 flux (+ HALEU companions) ---------------------
        spectra, cov75, _ = load_unfolded()
        sigma_ibd = vogel_beacom(enu, order=1)
        self._flux_235 = spectra["U235"](enu) / np.maximum(sigma_ibd, 1e-99)
        self._flux_238 = spectrum_per_fission(enu, "U238")
        self._flux_239 = spectra["Pu239"](enu) / np.maximum(sigma_ibd, 1e-99)
        self._flux_241 = spectrum_per_fission(enu, "Pu241")
        f = haleu_fractions(burnup, evolve=True)
        self.fractions = f
        self._flux = (f["U235"] * self._flux_235 + f["U238"] * self._flux_238
                      + f["Pu239"] * self._flux_239 + f["Pu241"] * self._flux_241)
        self._yield_ibd = (f["U235"] * spectra["U235"](enu)
                           + f["Pu239"] * spectra["Pu239"](enu)
                           + (f["U238"] * self._flux_238
                              + f["Pu241"] * self._flux_241) * sigma_ibd)

        self.fission_rate = fission_rate_per_second(power_mwth / 1000.0, f)
        area = 4.0 * np.pi * (baseline_m / 1000.0 * CM_PER_KM) ** 2
        #: events per second of live time at full power, per unit "stuff" below
        self._geom = self.fission_rate / area

        # per MW.yr of delivered exposure (power already in fission_rate)
        self.seconds_per_mw_yr = 365.25 * SECONDS_PER_DAY / power_mwth

        # -- oscillated flavor decomposition ---------------------------------
        pee = survival_probability_ee(enu, baseline_m / 1000.0, truth)
        self.pee = pee
        self._flux_e = self._flux * pee
        self._flux_x = self._flux * (1.0 - pee)

        # -- recoil response (electron non-linearity + resolution) -----------
        t_grid = np.linspace(0.05, 9.5, 1200)
        self.t_grid = t_grid
        w_t = integration_weights(t_grid)
        nl_e = TabulatedNonLinearity.from_release("electron")
        res = EnergyResolution(a=resolution_a, b=resolution_b, c=0.0)
        t_vis = nl_e.visible_energy(t_grid)
        self._R_t = gaussian_bin_response(t_vis, self.recoil_edges, res.sigma(t_vis))
        step = atomic_stepping(t_grid)
        self._step_t = step

        def _resp(nl, src, edges, sc=0.0, bi=0.0, re_=0.0):
            ev = src * ((1.0 + sc) * nl.factor(src) + bi)
            return gaussian_bin_response(ev, edges, (1.0 + re_) * res.sigma(ev))

        hh = 1.0e-3
        self._dR_t = [(_resp(nl_e, t_grid, self.recoil_edges, **{k: hh})
                       - _resp(nl_e, t_grid, self.recoil_edges, **{k: -hh}))
                      / (2 * hh) for k in ("sc", "bi", "re_")]

        # -- EvES templates: spectrum is affine in CL^2, CR^2, CL*CR ----------
        # dR/dT = geom * N_e * eff * integral dE phi(E) * pref *
        #         [CL^2 + CR^2 (1-T/E)^2 - CL CR m_e T / E^2]
        pref = 2.0 * M_E * GF**2 / np.pi * MEV2_TO_CM2
        ee, tt = enu[None, :], t_grid[:, None]
        allowed = tt < 2.0 * ee**2 / (M_E + 2.0 * ee)
        k1 = np.where(allowed, 1.0, 0.0)
        k2 = np.where(allowed, (1.0 - tt / ee) ** 2, 0.0)
        k3 = np.where(allowed, -M_E * tt / ee**2, 0.0)
        scale = self._geom * JUNO_ELECTRONS * efficiency * pref

        def templates(flux):
            # (k @ f_w) integrates over E for every T; x w_t turns the recoil
            # density into per-grid-point events; x step applies the atomic
            # binding (stepping) correction -- exactly 1 in the MeV window.
            f_w = flux * w_e
            return [scale * (self._R_t @ ((k @ f_w) * w_t * step))
                    for k in (k1, k2, k3)]

        self._tmpl_e = templates(self._flux_e)    # per second, nubar_e
        self._tmpl_x = templates(self._flux_x)    # per second, nubar_mu/tau
        # unsmeared dR/dT templates on t_grid (per second), for plotting
        self._tmpl_e_true = [scale * (k @ (self._flux_e * w_e)) * step
                             for k in (k1, k2, k3)]
        self._tmpl_x_true = [scale * (k @ (self._flux_x * w_e)) * step
                             for k in (k1, k2, k3)]
        self._w_t = w_t

        # -- IBD prompt spectrum (positron response) --------------------------
        nl_p = TabulatedNonLinearity.from_release("positron")
        e_dep = DetectorResponse(use_ibd_recoil=True).deposited_energy(enu)
        e_vis = nl_p.visible_energy(e_dep)
        self._R_ibd = gaussian_bin_response(e_vis, self.recoil_edges, res.sigma(e_vis))
        self._dR_ibd = [(_resp(nl_p, e_dep, self.recoil_edges, **{k: hh})
                         - _resp(nl_p, e_dep, self.recoil_edges, **{k: -hh}))
                        / (2 * hh) for k in ("sc", "bi", "re_")]
        self._ibd_density = self._geom * JUNO2025_TARGET_PROTONS * efficiency \
            * self._yield_ibd * pee * w_e
        self.ibd_prompt = self._R_ibd @ self._ibd_density        # per second, binned
        self.ibd_rate_per_day = float(self._ibd_density.sum()) * SECONDS_PER_DAY

        # -- the IBD anchor spectrum (own, wider binning) ---------------------
        self.ibd_edges = (np.arange(1.0, 9.0 + 1e-9, 0.05)
                          if ibd_edges is None else ibd_edges)
        self._R_ibd_anchor = gaussian_bin_response(e_vis, self.ibd_edges,
                                                   res.sigma(e_vis))
        self.ibd_anchor = self._R_ibd_anchor @ self._ibd_density  # per second
        self._dR_anchor = [(_resp(nl_p, e_dep, self.ibd_edges, **{k: hh})
                            - _resp(nl_p, e_dep, self.ibd_edges, **{k: -hh}))
                           / (2 * hh) for k in ("sc", "bi", "re_")]

        # isotope shares in each channel (for the shared flux-shape modes)
        self._share_235_flux = f["U235"] * self._flux_235 / np.maximum(self._flux, 1e-99)
        self._share_235_yield = (f["U235"] * spectra["U235"](enu)
                                 / np.maximum(self._yield_ibd, 1e-99))
        self._share_238_flux = f["U238"] * self._flux_238 / np.maximum(self._flux, 1e-99)
        self._share_238_yield = (f["U238"] * self._flux_238 * sigma_ibd
                                 / np.maximum(self._yield_ibd, 1e-99))
        self.sigma_ibd_xsec = sigma_ibd_xsec
        self.sigma_ibd_xsec_tilt = sigma_ibd_xsec_tilt
        self.sigma_channel_ratio = sigma_channel_ratio
        self.joint_flux_prior = joint_flux_prior
        self.include_backgrounds = include_backgrounds
        self.sigma_solar = sigma_solar
        self.sigma_energy = (sigma_scale, sigma_bias, sigma_res)
        self.sigma_evolution = sigma_evolution

        # -- solar EvES background (LMA-weighted, calendar-time accrual) -----
        from .solar import PEP_E, PEP_FLUX, solar_flux_density, solar_pee

        es = np.linspace(1.0, 18.0, 600)
        ws = integration_weights(es)
        phi_s = solar_flux_density(es)
        pee_s = solar_pee(es, truth)
        k_se = eves_dsigma_dT(es[None, :], t_grid[:, None], "e", False,
                              GV_SM, GA_SM)
        k_sm = eves_dsigma_dT(es[None, :], t_grid[:, None], "mu", False,
                              GV_SM, GA_SM)
        dens_s = k_se @ (phi_s * pee_s * ws) + k_sm @ (phi_s * (1 - pee_s) * ws)
        p0 = float(solar_pee(np.array([PEP_E]), truth)[0])
        dens_s = dens_s + PEP_FLUX * (
            p0 * eves_dsigma_dT(PEP_E, t_grid, "e", False, GV_SM, GA_SM)
            + (1 - p0) * eves_dsigma_dT(PEP_E, t_grid, "mu", False,
                                        GV_SM, GA_SM))
        # per live-second equivalent (solar runs on calendar time -> /duty)
        self._solar_grid_vec = (dens_s * w_t * step * JUNO_ELECTRONS
                                * efficiency / duty_cycle)
        self.solar_eves_binned = self._R_t @ self._solar_grid_vec

        # -- nubar_e -> nubar_e e+e- tridents: a single-hit background ---------
        # The pair carries essentially E_nu (nuclear recoil is keV); its kinetic
        # energy plus the two annihilation gammas is deposited as ONE prompt hit,
        # so the visible energy is ~E_nu through the electron/positron response.
        # Coherent on carbon + elastic on the free protons; per fission-second.
        self.include_tridents = include_tridents
        self.sigma_trident = sigma_trident
        if include_tridents:
            from .tridents import sigma_trident_coherent_c12, sigma_trident_proton
            n_c = JUNO2025_TARGET_PROTONS / 1.667            # carbon nuclei ~ N_p / 1.67
            sig_tr = (n_c * sigma_trident_coherent_c12(enu)
                      + JUNO2025_TARGET_PROTONS * sigma_trident_proton(enu))
            dens_tr = self._geom * efficiency * self._flux * sig_tr * w_e   # per s, per E bin
            R_tr = gaussian_bin_response(nl_e.visible_energy(enu), self.recoil_edges,
                                         res.sigma(nl_e.visible_energy(enu)))
            self.trident_binned = R_tr @ dens_tr
            self.trident_rate_per_day = float(dens_tr.sum()) * SECONDS_PER_DAY
        else:
            self.trident_binned = np.zeros(len(self.recoil_edges) - 1)
            self.trident_rate_per_day = 0.0

        # -- standard JUNO backgrounds under the IBD anchor ------------------
        from .juno_data import load_spectrum

        rel = load_spectrum()
        cen_a = 0.5 * (self.ibd_edges[:-1] + self.ibd_edges[1:])
        wid_a = np.diff(self.ibd_edges)
        comps = {"far reactors": rel.pred_signal}
        comps.update(rel.backgrounds)
        self._anchor_bkg = {}
        for name, counts in comps.items():
            dens = counts / rel.widths / rel.livetime_days      # per MeV per day
            self._anchor_bkg[name] = (np.interp(cen_a, rel.centers, dens,
                                                left=0.0, right=0.0) * wid_a
                                      / SECONDS_PER_DAY / duty_cycle)

        # -- fuel-evolution (Pu ingrowth) distortion vectors -----------------
        evo_flux = (f["Pu239"] * (self._flux_239 - self._flux_235)
                    + f["Pu241"] * (self._flux_241 - self._flux_235))
        self._evo_rel_flux = evo_flux / np.maximum(self._flux, 1e-99)
        self._evo_rel_yield = (evo_flux * sigma_ibd
                               / np.maximum(self._yield_ibd, 1e-99))

        # -- systematics bookkeeping -----------------------------------------
        self.sigma_norm = sigma_norm
        self.sigma_u238 = sigma_u238
        self.sigma_ibd_singles = sigma_ibd_singles
        self.use_flux_covariance = use_flux_covariance
        if use_flux_covariance:
            vals = spectra["U235"].values
            cov = np.asarray(cov75)[:25, :25] * 1e-86
            rel = cov / np.outer(vals, vals)
            evals, evecs = np.linalg.eigh(rel)
            keep = evals > 1e-10 * evals.max()
            psi = (evecs[:, keep] * np.sqrt(evals[keep])).T
            cen = spectra["U235"].centers
            self._u235_modes = np.array([np.interp(enu, cen, m) for m in psi])
        else:
            self._u235_modes = np.zeros((0, enu.size))

    def _sm_combo(self, flavor: str):
        cv = GV_SM + (1.0 if flavor == "e" else 0.0)
        ca = GA_SM + (1.0 if flavor == "e" else 0.0)
        cl, cr = 0.5 * (cv - ca), 0.5 * (cv + ca)
        return (cl**2, cr**2, cl * cr)

    # -- spectra --------------------------------------------------------------
    def eves_spectrum_T(self, gv: float = GV_SM, ga: float = GA_SM,
                        smeared: bool = True):
        """EvES recoil spectrum.

        ``smeared=True`` (default): binned counts per second of live time in
        ``recoil_edges``.  ``smeared=False``: the true dR/dT density
        [events / MeV / s] on ``t_grid``.
        """

        def combo(tmpl, flavor):
            cv = gv + (1.0 if flavor == "e" else 0.0)
            ca = ga + (1.0 if flavor == "e" else 0.0)
            cl, cr = 0.5 * (cv - ca), 0.5 * (cv + ca)   # antineutrino
            return cl**2 * tmpl[0] + cr**2 * tmpl[1] + cl * cr * tmpl[2]

        if smeared:
            return combo(self._tmpl_e, "e") + combo(self._tmpl_x, "mu")
        return combo(self._tmpl_e_true, "e") + combo(self._tmpl_x_true, "mu")

    def eves_spectrum_Enu(self, t_min: float | None = None,
                          t_max: float | None = None,
                          gv: float = GV_SM, ga: float = GA_SM):
        """EvES dR/dEnu [events / MeV / s live], flavor-decomposed.

        Recoils are integrated over the analysis window (default: the recoil
        binning range).
        """

        t_min = self.recoil_edges[0] if t_min is None else t_min
        t_max = self.recoil_edges[-1] if t_max is None else t_max
        e = self.e_nu_grid
        out = {}
        for name, flux, flavor in (("nubar_e", self._flux_e, "e"),
                                   ("nubar_mu+tau", self._flux_x, "mu")):
            sig = np.array([
                np.trapezoid(eves_dsigma_dT(ee, self.t_grid, flavor, True, gv, ga)
                         * ((self.t_grid >= t_min) & (self.t_grid <= t_max)),
                         self.t_grid)
                for ee in e])
            out[name] = self._geom * JUNO_ELECTRONS * self.efficiency * flux * sig
        return out

    def ibd_singles(self):
        """IBD single-hit background in the recoil bins, per second live."""

        return self.untagged_fraction * self.ibd_prompt

    # -- statistics ------------------------------------------------------------
    def _exposure_seconds(self, mw_yr: float) -> float:
        return mw_yr * self.seconds_per_mw_yr

    def _build_covariance(self, mw_yr: float, joint: bool = False):
        t = self._exposure_seconds(mw_yr)
        sig = self.eves_spectrum_T() * t
        bkg = self.ibd_singles() * t

        sol = (self.solar_eves_binned * t if self.include_backgrounds
               else np.zeros_like(sig))
        tri = self.trident_binned * t
        # true-T per-grid event vector of the SM EvES signal (for energy pulls)
        # (the true templates already carry the stepping factor)
        sig_grid = ((sum(c * m for c, m in zip(self._sm_combo("e"),
                                               self._tmpl_e_true))
                     + sum(c * m for c, m in zip(self._sm_combo("mu"),
                                                 self._tmpl_x_true)))
                    * self._w_t)

        def energy_modes(anchor: bool):
            out = []
            for k, w in enumerate(self.sigma_energy):
                m_e = w * t * (self._dR_t[k]
                               @ (sig_grid + (self._solar_grid_vec
                                              if self.include_backgrounds
                                              else 0.0))
                               + self._dR_ibd[k]
                               @ (self.untagged_fraction * self._ibd_density))
                if anchor:
                    m_a = w * t * (self._dR_anchor[k] @ self._ibd_density)
                    out.append(np.concatenate([m_a, m_e]))
                else:
                    out.append(m_e)
            return out

        evo_eves = self.sigma_evolution * t * (
            self._flux_distortion(self._evo_rel_flux)
            + self.untagged_fraction
            * (self._R_ibd @ (self._ibd_density * self._evo_rel_yield)))

        if not joint:
            mu = sig + bkg + sol + tri
            modes = [self.sigma_norm * sig,
                     self.sigma_ibd_singles * bkg]
            if self.include_backgrounds:
                modes.append(self.sigma_solar * sol)
            if self.include_tridents:
                modes.append(self.sigma_trident * tri)
            modes += energy_modes(anchor=False)
            modes.append(evo_eves)
            modes.append(self.sigma_u238 * t
                         * self._flux_distortion(self._share_238_flux))
            for m in self._u235_modes:
                modes.append(t * self._flux_distortion(m * self._share_235_flux))
            v = np.array(modes)
            cov = np.diag(np.maximum(mu, 1e-9)) + v.T @ v
            return mu, sig, cho_factor(cov)

        # joint: [IBD anchor bins, EvES bins].  The flux normalisation is left
        # essentially free (loose prior) -- the IBD channel measures it, and
        # what limits the transfer to EvES is the IBD cross-section knowledge
        # and the channel ratio (N_e/N_p x relative efficiency).
        ibd = self.ibd_anchor * t
        n_i = ibd.size
        z_i, z_e = np.zeros(n_i), np.zeros(sig.size)
        anchor_bkg_tot = (sum(self._anchor_bkg.values()) * t
                          if self.include_backgrounds else z_i)
        mu = np.concatenate([ibd + anchor_bkg_tot, sig + bkg + sol + tri])
        stack = lambda a, b: np.concatenate([a, b])

        modes = [self.joint_flux_prior * stack(ibd, sig),        # flux norm, shared
                 self.sigma_ibd_xsec * stack(ibd, z_e),          # IBD xsec norm
                 self.sigma_channel_ratio * stack(z_i, sig),     # N_e/N_p x eff ratio
                 self.sigma_ibd_singles * stack(z_i, bkg)]
        if self.include_backgrounds:
            modes.append(self.sigma_solar * stack(z_i, sol))
        if self.include_tridents:
            modes.append(self.sigma_trident * stack(z_i, tri))
        if self.include_backgrounds:
            for name, shape in self._anchor_bkg.items():
                modes.append(NEAR_IBD_BACKGROUND_PRIORS[name]
                             * stack(shape * t, z_e))
        modes += energy_modes(anchor=True)
        modes.append(stack(self.sigma_evolution * t
                           * self._ibd_distortion(self._evo_rel_yield),
                           evo_eves))
        # IBD cross-section shape: a zero-mean linear tilt across the window
        e = self.e_nu_grid
        tilt = (e - e.mean()) / (0.5 * (e.max() - e.min()))
        modes.append(self.sigma_ibd_xsec_tilt * t * stack(self._ibd_distortion(tilt), z_e))
        # shared flux-shape modes, hitting both channels coherently
        modes.append(self.sigma_u238 * t * stack(
            self._ibd_distortion(self._share_238_yield),
            self._flux_distortion(self._share_238_flux)))
        for m in self._u235_modes:
            modes.append(t * stack(self._ibd_distortion(m * self._share_235_yield),
                                   self._flux_distortion(m * self._share_235_flux)))
        v = np.array(modes)
        cov = np.diag(np.maximum(mu, 1e-9)) + v.T @ v
        return mu, stack(ibd, sig), cho_factor(cov)

    def _ibd_distortion(self, rel):
        """IBD anchor-spectrum change for a relative distortion rel(E), per s."""

        return self._R_ibd_anchor @ (self._ibd_density * rel)

    def _flux_distortion(self, rel):
        """EvES spectrum change for a relative flux distortion rel(E), per s."""

        w_e = integration_weights(self.e_nu_grid)
        pref = 2.0 * M_E * GF**2 / np.pi * MEV2_TO_CM2
        scale = self._geom * JUNO_ELECTRONS * self.efficiency * pref
        ee, tt = self.e_nu_grid[None, :], self.t_grid[:, None]
        allowed = tt < 2.0 * ee**2 / (M_E + 2.0 * ee)
        out = np.zeros(len(self.recoil_edges) - 1)
        for flux, flavor in ((self._flux_e, "e"), (self._flux_x, "mu")):
            cv = GV_SM + (1.0 if flavor == "e" else 0.0)
            ca = GA_SM + (1.0 if flavor == "e" else 0.0)
            cl, cr = 0.5 * (cv - ca), 0.5 * (cv + ca)
            kern = np.where(allowed,
                            cl**2 + cr**2 * (1.0 - tt / ee) ** 2
                            - cl * cr * M_E * tt / ee**2, 0.0)
            w_t = integration_weights(self.t_grid)
            out = out + scale * (self._R_t
                                 @ ((kern @ (flux * rel * w_e)) * w_t
                                    * self._step_t))
        return out

    def _pad(self, d_eves, joint):
        if not joint:
            return d_eves
        return np.concatenate([np.zeros(self.ibd_anchor.size), d_eves])

    def chi2_gv_ga(self, gv, ga, mw_yr: float, joint: bool = False):
        """Asimov Delta chi^2 at (gV, gA) against the SM point."""

        mu, sig0, cho = self._cached(mw_yr, joint)
        t = self._exposure_seconds(mw_yr)
        # the prediction differs from the Asimov data only on the EvES side
        d = self._pad((self.eves_spectrum_T(gv, ga) - self.eves_spectrum_T()) * t,
                      joint)
        return float(d @ cho_solve(cho, d))

    def _cached(self, mw_yr, joint: bool = False):
        key = (round(float(mw_yr), 9), bool(joint))
        if not hasattr(self, "_cov_cache"):
            self._cov_cache = {}
        if key not in self._cov_cache:
            self._cov_cache[key] = self._build_covariance(mw_yr, joint)
        return self._cov_cache[key]

    def sigma_sw2(self, mw_yr: float, joint: bool = False) -> float:
        """Fisher error on sin^2 theta_W (gA fixed at its SM value)."""

        h = 1e-4
        t = self._exposure_seconds(mw_yr)
        _, _, cho = self._cached(mw_yr, joint)
        d = self._pad((self.eves_spectrum_T(gv=2 * (SW2_SM + h) - 0.5)
                       - self.eves_spectrum_T(gv=2 * (SW2_SM - h) - 0.5))
                      * t / (2 * h), joint)
        return 1.0 / np.sqrt(float(d @ cho_solve(cho, d)))

    def fisher_gv_ga(self, mw_yr: float, joint: bool = False):
        """2x2 Fisher covariance for (gV, gA) around the SM."""

        h = 1e-4
        t = self._exposure_seconds(mw_yr)
        _, _, cho = self._cached(mw_yr, joint)
        d = []
        for dgv, dga in ((h, 0.0), (0.0, h)):
            d.append(self._pad((self.eves_spectrum_T(GV_SM + dgv, GA_SM + dga)
                                - self.eves_spectrum_T(GV_SM - dgv, GA_SM - dga))
                               * t / (2 * h), joint))
        d = np.array(d)
        sol = np.array([cho_solve(cho, row) for row in d])
        return np.linalg.inv(d @ sol.T)

    # -- validation ------------------------------------------------------------
    def validate_against_neptune(self) -> float:
        """Max relative deviation of the parametric formula from NEPTUNE."""

        from neptune import nu_electron as ne

        e = np.array([2.5e-3, 4.0e-3, 6.0e-3])      # GeV
        t = np.array([0.5e-3, 1.5e-3, 3.0e-3])      # GeV
        worst = 0.0
        for flavor in ("e", "mu"):
            for ee in e:
                # NEPTUNE evaluates the SM at tree level (sw2 = 0.2223, gA = -1/2):
                # compare the parametric formula at exactly those couplings
                ours = eves_dsigma_dT(ee * 1e3, t * 1e3, flavor, True,
                                      gv=2 * SW2_NEPTUNE - 0.5, ga=-0.5) * 1e3  # cm2/GeV
                theirs = ne.dsigma_dTe(ee, t, nu_flavor=flavor, is_nubar=True)
                good = theirs > 0
                if np.any(good):
                    worst = max(worst, float(np.max(
                        np.abs(ours[good] / theirs[good] - 1.0))))
        return worst
