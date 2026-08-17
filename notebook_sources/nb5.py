OUT = "5_near_sm_precision.ipynb"

CELLS = [
("md", r"""
# 5 — A fixed near reactor: Standard-Model precision with $\nu$–$e$ scattering

A compact HALEU reactor ($10$ MW$_{\rm th}$) parked $50$ m from the JUNO detector. The IBD
channel is enormous at that distance; the physics target is **elastic neutrino–electron
scattering** (E$\nu$ES), whose cross section carries the weak mixing angle through the
neutral-current couplings — $C_V=\tfrac12+2\sin^2\theta_W$ for $\bar\nu_e$ (CC+NC),
$C_V=-\tfrac12+2\sin^2\theta_W$ for $\bar\nu_{\mu,\tau}$ (NC only).

**Ingredients** (`reactor/near_sm.py`):

* Cross sections from **NEPTUNE** (`neptune.nu_electron`); the parametric $(g_V,g_A)$
  version used for scans is the identical tree-level formula with NEPTUNE's own constants,
  validated at the SM point to machine precision below. (NEPTUNE's
  ${\rm GeV}^{-2}\to{\rm cm}^2$ constant sits $0.68\%$ above $(\hbar c)^2$ — flagged
  upstream; it rescales absolute rates only and is absorbed by the normalisation
  systematic.)
* Neutrino **flux** = the measured Daya Bay U235 yield $\div$ Vogel–Beacom, plus the $2\%$
  U238 Huber–Mueller component — defined above the IBD threshold $1.806$ MeV only, so
  recoil rates below $T\sim1.5$ MeV are conservative.
* Flavors from exact oscillations: $\bar\nu_e$ with $P_{ee}$ and $\bar\nu_{\mu}+\bar\nu_\tau$
  with $1-P_{ee}$ and the NC-only cross section. At $50$ m this is a $10^{-4}$-level
  component, but it is carried exactly and the same object works at any baseline.
* Detector: released electron non-linearity, $a=3.3\%/b=1\%$ resolution, efficiency $0.8$,
  duty cycle $0.9$; $N_e = 4.67\,N_p = 6.7\times10^{33}$ electrons.
* **Background modelled: IBD singles** — IBD events whose delayed-neutron tag is missed
  (default untagged fraction $1\%$), promoting the positron to a single hit. Radioactivity
  and cosmogenic singles are *not* modelled; results are reactor-limited precisions.
* Systematics: normalisation $2\%$, the 25-mode measured U235 flux-shape covariance,
  U238 $\pm15\%$, IBD-singles normalisation $\pm10\%$, **energy scale/bias/resolution
  pulls** ($0.5/0.5/5\%$, coherent across the electron and positron responses), and the
  **fuel-evolution** ($\pm30\%$ Pu-ingrowth) mode at the run-averaged burnup
  $\beta=0.5$. Asimov statistics with analytic profiling.
* **Backgrounds**: IBD singles, **solar E$\nu$ES** (B8/hep/CNO/pep, LMA-weighted, $3\%$
  prior, calendar-time accrual), and — under the IBD anchor of the joint fit — the
  **standard JUNO backgrounds** (far reactors, geoneutrinos, $^9$Li/$^8$He, world
  reactors, $^{214}$Bi–Po, other) at their release priors.
* **Atomic binding**: the Kopeikin stepping correction (an electron participates only
  above its binding energy; carbon K $284$ eV) is applied to every recoil spectrum — and
  demonstrated below to be *exactly* null above $1$ keV, four orders of magnitude below
  the analysis window.
"""),

("code", r"""
import numpy as np
import matplotlib.pyplot as plt

from reactor import plotting as pl
from reactor.detector import integration_weights
from reactor.near_sm import (
    FixedNearReactor, eves_dsigma_dT, GA_SM, GV_SM, SW2_SM,
)

pl.use_style()
DAY = 86400.0

# The standard configuration: run-averaged burnup on the Daya Bay trajectory.
STD = dict(burnup=0.5)
r = FixedNearReactor(**STD)
print(f"NEPTUNE validation: max relative deviation = {r.validate_against_neptune():.1e}")
print(f"source: {r.power_mwth} MW_th at {r.baseline_m} m, fresh HALEU "
      f"{ {k: round(v, 3) for k, v in r.fractions.items() if v > 0} }")
print(f"fission rate: {r.fission_rate:.3e} /s;  P_ee at 50 m: {r.pee.min():.5f}-{r.pee.max():.5f}")
print(f"SM couplings: sin^2(thetaW) = {SW2_SM}  ->  gV = {GV_SM:.4f}, gA = {GA_SM}")
"""),

("code", r"""
# The atomic (stepping) correction: account for it, then show it is null here.
from reactor.near_sm import atomic_stepping

t_dem = np.logspace(-6, 1, 800)
fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.semilogx(t_dem, atomic_stepping(t_dem), color=pl.BLUE, lw=1.7)
ax.axvspan(r.recoil_edges[0], r.recoil_edges[-1], color=pl.GREEN, alpha=0.15, lw=0)
ax.annotate("analysis window", xy=(1.3, 0.25), fontsize=8, color=pl.GREEN)
ax.set_xlabel(r"true recoil energy $T$ [MeV]")
ax.set_ylabel("active-electron fraction")
ax.set_title("Atomic binding (stepping) correction for LS electrons")
plt.tight_layout(); plt.show()

dev = abs(1 - atomic_stepping(np.linspace(1e-3, 10, 2000))).max()
print(f"maximum deviation from unity above 1 keV: {dev:.1e} -- the correction is applied")
print("to every spectrum in the module and is exactly null in the MeV window. (It would")
print("matter for a sub-keV threshold, e.g. a magnetic-moment search.)")
"""),

("md", r"""
## Cross sections per LAB molecule

LAB is C$_6$H$_5$C$_n$H$_{2n+1}$ with $n\approx12$, i.e. C$_{18}$H$_{30}$ or CH$_{1.67}$
per carbon: one carbon nucleus, $1.67$ free protons and $7.67$ electrons per unit — the
stoichiometry behind the repository's $N_e = 4.67\,N_p$. Below, every cross section is
quoted per CH$_{1.67}$ unit. (The $\bar\nu_e\to\bar\nu_e e^+e^-$ trident channel was
evaluated with NEPTUNE, `reactor/tridents.py`, and found to be $\sim10^{-5}$ of the E$\nu$ES
rate at reactor energies — $0.04$ events/day at 10 MW / 50 m — so it is not carried.)
"""),

("code", r"""
from reactor.tridents import H_PER_C, E_PER_C        # LAB stoichiometry constants
from reactor.cross_sections import vogel_beacom

enu = np.linspace(1.9, 10.0, 300)
tgrid = np.linspace(0.0, 10.0, 2000)
def sig_tot(flavor, nubar, tmin=0.0):
    return np.array([np.trapezoid(np.where(tgrid >= tmin, eves_dsigma_dT(e, tgrid, flavor, nubar), 0.0),
                                  tgrid) for e in enu]) * E_PER_C

fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.semilogy(enu, sig_tot("e", True), color=pl.BLUE, lw=1.8, label=r"$\bar\nu_e\,e$ (CC+NC)")
ax.semilogy(enu, sig_tot("e", False), color=pl.BLUE, ls="--", lw=1.4, label=r"$\nu_e\,e$ (CC+NC)")
ax.semilogy(enu, sig_tot("mu", True), color=pl.ORANGE, lw=1.8, label=r"$\bar\nu_{\mu,\tau}\,e$ (NC)")
ax.semilogy(enu, sig_tot("mu", False), color=pl.ORANGE, ls="--", lw=1.4, label=r"$\nu_{\mu,\tau}\,e$ (NC)")
ax.semilogy(enu, sig_tot("e", True, tmin=1.0), color=pl.BLUE, ls=":", lw=1.6,
            label=r"$\bar\nu_e\,e$, $T>1$ MeV (analysis window)")
ax.semilogy(enu, vogel_beacom(enu, order=1) * H_PER_C, color=pl.RED, lw=1.8, label=r"IBD $\bar\nu_e p$")
ax.set_ylim(1e-44, 2e-41)
ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel(r"$\sigma$ per CH$_{1.67}$ unit [cm$^2$]")
ax.set_title(r"Cross sections per LAB molecule (C$_{18}$H$_{30}$ / 18)")
ax.legend(fontsize=10, loc="lower right", ncol=2)
plt.tight_layout(); plt.show()
"""),

("md", r"""
## 1. The IBD spectrum

At $50$ m the parked reactor outshines the entire far-reactor complex by two orders of
magnitude. This sample is not the point of the study, but it is free flux calibration — and
its untagged tail is the background of section 3.
"""),

("code", r"""
fig, ax = plt.subplots(figsize=(6.8, 4.0))
centers = 0.5*(r.recoil_edges[:-1] + r.recoil_edges[1:])
widths = np.diff(r.recoil_edges)
ax.stairs(r.ibd_prompt * DAY / widths, r.recoil_edges, color=pl.BLUE, lw=1.6)
ax.set_xlabel(r"$E_{\rm prompt}$ [MeV]")
ax.set_ylabel("IBD events / MeV / day")
ax.set_title(f"IBD prompt spectrum, {r.power_mwth:.0f} MW at {r.baseline_m:.0f} m")
plt.tight_layout(); plt.show()

print(f"IBD rate: {r.ibd_rate_per_day:,.0f} / day  "
      f"({r.ibd_rate_per_day*365.25*r.duty_cycle:,.0f} / calendar year at duty {r.duty_cycle})")
print("For scale, the nine far reactors give JUNO ~35 IBD/day: the parked source is")
print(f"~{r.ibd_rate_per_day/35:,.0f}x the far-reactor rate.")
"""),

("md", r"""
## 2. The E$\nu$ES spectrum

In neutrino energy and in electron recoil energy, decomposed by flavor. The
$\bar\nu_{\mu}+\bar\nu_\tau$ component from oscillation is four orders of magnitude down at
$50$ m — but its cross section differs (NC only, no CC), and both pieces are carried exactly.
"""),

("code", r"""
# (a) in neutrino energy
sp = r.eves_spectrum_Enu()
w_e = integration_weights(r.e_nu_grid)
rate_e = float((sp["nubar_e"] * w_e).sum() * DAY)
rate_x = float((sp["nubar_mu+tau"] * w_e).sum() * DAY)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
ax.semilogy(r.e_nu_grid, sp["nubar_e"] * DAY, color=pl.BLUE, label=r"$\bar\nu_e$ (CC+NC)")
ax.semilogy(r.e_nu_grid, sp["nubar_mu+tau"] * DAY, color=pl.ORANGE,
            label=r"$\bar\nu_\mu+\bar\nu_\tau$ (NC), from oscillation")
ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel("events / MeV / day")
ax.set_xlim(1.8, 9.5); ax.set_ylim(1e-5, 3e3)
ax.set_title(r"E$\nu$ES in neutrino energy (recoils in the analysis window)")
ax.legend(fontsize=8.5)

# (b) in recoil energy
ax = axes[1]
true = r.eves_spectrum_T(smeared=False) * DAY
ax.semilogy(r.t_grid, true, color=pl.BLUE, lw=1.6, label="true $dR/dT$")
ax.stairs(r.eves_spectrum_T() * DAY / widths, r.recoil_edges, color=pl.ORANGE,
          label="reconstructed (NL + resolution), binned")
ax.set_xlabel(r"$T_{\rm recoil}$ [MeV]"); ax.set_ylabel("events / MeV / day")
ax.set_xlim(0, 8); ax.set_ylim(1e-2, 1e4)
ax.set_title(r"E$\nu$ES in electron recoil energy")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

print(f"EvES rates, analysis window [{r.recoil_edges[0]:.1f}, {r.recoil_edges[-1]:.1f}] MeV:")
print(f"   nubar_e          : {rate_e:8.1f} / day")
print(f"   nubar_mu+tau     : {rate_x:8.4f} / day   ({100*rate_x/(rate_e+rate_x):.3f}%)")
print(f"   binned, smeared  : {float(r.eves_spectrum_T().sum()*DAY):8.1f} / day")
print()
print("Below T ~ 1.5 MeV the true curve is conservative: the reactor flux below the IBD")
print("threshold (1.806 MeV) is not measured and is set to zero here.")
"""),

("md", r"""
## 3. The IBD single-hit background

An IBD whose delayed neutron capture goes untagged leaves only the positron — a single hit
with a spectrum starting at $1.02$ MeV, right inside the E$\nu$ES window. With an untagged
fraction of $1\%$, that is $\sim600$ events/day against $\sim3200$ E$\nu$ES events/day.
"""),

("code", r"""
sig = r.eves_spectrum_T() * DAY
bkg = r.ibd_singles() * DAY
sol = r.solar_eves_binned * DAY

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
ax.stairs(sig / widths, r.recoil_edges, color=pl.BLUE, lw=1.6, label=r"E$\nu$ES signal")
ax.stairs(bkg / widths, r.recoil_edges, color=pl.RED, lw=1.6,
          label=f"IBD singles ({100*r.untagged_fraction:.0f}\\% untagged)")
ax.stairs(sol / widths, r.recoil_edges, color=pl.GREEN, lw=1.4,
          label="solar neutrinos (LMA-weighted)")
ax.set_yscale("log")
ax.set_xlabel(r"reconstructed energy [MeV]"); ax.set_ylabel("events / MeV / day")
ax.legend(fontsize=8.5); ax.set_title("Signal and IBD-singles background")

ax = axes[1]
ratio = np.divide(sig, bkg, out=np.full_like(sig, np.inf), where=bkg > 0)
ax.stairs(ratio, r.recoil_edges, color=pl.GREEN, lw=1.6)
ax.set_yscale("log")
ax.set_xlabel(r"reconstructed energy [MeV]"); ax.set_ylabel("S / B per bin")
ax.set_title("Signal-to-background")
plt.tight_layout(); plt.show()

print(f"in-window totals per day: signal {sig.sum():,.1f}, IBD singles {bkg.sum():,.1f}, "
      f"solar {sol.sum():,.1f} -> S/B = {sig.sum()/(bkg.sum()+sol.sum()):.2f}")
print()
print("The IBD-singles shape is the prompt spectrum: it peaks at 3-4 MeV where the")
print("oscillation-free positron spectrum peaks, while EvES falls with T -- so S/B is")
print("best below ~2.5 MeV and worst near 4 MeV. Its normalisation is measurable in situ")
print("from the 99% tagged sample, motivating the 10% prior. Radioactivity and cosmogenic")
print("singles are NOT included; below ~3 MeV they would dominate both, so the results")
print("here are reactor-limited precisions.")
"""),

("md", r"""
## 4. Precision on $\sin^2\theta_W$

The recoil spectrum is fit for $\sin^2\theta_W$ (with $g_A$ at its SM value) under the full
systematics model, as a function of the delivered exposure in MW$\cdot$yr.

Two scales frame the result. The rate carries $|d\ln R/d\sin^2\theta_W|\approx4$, so a pure
normalisation uncertainty *alone* would floor the precision at $\sigma_{\rm norm}/4$ — but
the recoil *shape* also changes with the couplings, and the fit exploits it: at $2\%$
normalisation with no flux-shape uncertainty, $\sigma(\sin^2\theta_W)\approx0.002$, well
below the $0.005$ rate floor. What actually dominates is the **measured U235 spectrum-shape
covariance**: its $2$–$4\%$ correlated bin uncertainties can imitate the coupling-induced
shape change, and with them the precision saturates at $\approx0.0047$ almost independently
of exposure and of the normalisation prior. This is a **flux-shape-limited** measurement:
the lever is a better reactor spectrum (a TAO-like sub-percent measurement), not statistics
and not primarily the rate normalisation.
"""),

("code", r"""
import time as _t
t0 = _t.time()
exposures = np.array([1, 3, 10, 30, 100, 300], dtype=float)
variants = [("norm 2% (default)", dict(), pl.BLUE),
            ("norm 1%", dict(sigma_norm=0.01), pl.ORANGE),
            ("norm 0.5%", dict(sigma_norm=0.005), pl.GREEN),
            ("statistics only", dict(sigma_norm=1e-6, sigma_u238=1e-6,
                                     sigma_ibd_singles=1e-6, sigma_solar=1e-9,
                                     sigma_scale=1e-9, sigma_bias=1e-9,
                                     sigma_res=1e-9, sigma_evolution=1e-9,
                                     use_flux_covariance=False), pl.INK_SECONDARY)]
fig, ax = plt.subplots(figsize=(6.9, 4.4))
for label, kw, color in variants:
    rr = FixedNearReactor(**STD, **kw)
    vals = [rr.sigma_sw2(m) for m in exposures]
    ax.loglog(exposures, vals, color=color, lw=1.8, label=label)
    if label.startswith("norm 2"):
        table_vals = vals
ax.set_xlabel(r"delivered exposure [MW$\cdot$yr]")
ax.set_ylabel(r"$\sigma(\sin^2\theta_W)$")
ax.set_title(r"$10$ MW at $50$ m ($\approx9$ MW$\cdot$yr per calendar year)")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()
print(f"({_t.time()-t0:.0f} s)")

rows = [[f"{m:.0f}", v, 100*v/SW2_SM] for m, v in zip(exposures, table_vals)]
print(pl.table(rows, ["MW.yr", "sigma(sin^2 thetaW)", "relative [%]"], floatfmt="{:.4f}"))

# what limits it at 30 MW.yr
rows = []
for label, kw in [("all systematics", dict()),
                  ("no flux-shape covariance", dict(use_flux_covariance=False)),
                  ("no solar/JUNO backgrounds", dict(include_backgrounds=False)),
                  ("no energy scale/bias/res pulls",
                   dict(sigma_scale=1e-9, sigma_bias=1e-9, sigma_res=1e-9)),
                  ("no fuel-evolution mode", dict(sigma_evolution=1e-9)),
                  ("no IBD-singles uncertainty", dict(sigma_ibd_singles=1e-6)),
                  ("no U238 uncertainty", dict(sigma_u238=1e-6)),
                  ("statistics only", dict(sigma_norm=1e-6, sigma_u238=1e-6,
                                           sigma_ibd_singles=1e-6,
                                           sigma_solar=1e-9, sigma_scale=1e-9,
                                           sigma_bias=1e-9, sigma_res=1e-9,
                                           sigma_evolution=1e-9,
                                           use_flux_covariance=False))]:
    rows.append([label, FixedNearReactor(**STD, **kw).sigma_sw2(30.0)])
print()
print(pl.table(rows, ["variant (30 MW.yr)", "sigma(sin^2 thetaW)"], floatfmt="{:.5f}"))
print()
print(f"The decomposition: flux shape remains the dominant systematic ({rows[0][1]:.4f} ->")
print(f"{rows[1][1]:.4f} without it), and the upgrade's additions -- solar/JUNO backgrounds,")
print(f"energy pulls, fuel evolution -- cost a few percent each in the EvES-only fit")
print(f"(rows above). For reference, low-energy sin^2 thetaW determinations sit at sigma ~")
print("0.003-0.02 per channel: with today's U235 spectrum knowledge this measurement lands")
print("mid-pack (~2%); with a TAO-class flux shape it would approach the 0.001 level.")
"""),

("md", r"""
## 5. The $(g_V, g_A)$ plane

The same fit with both neutral-current electron couplings free. A single $\bar\nu_e$ beam
determines $(C_L, C_R)$ only up to the classic discrete degeneracies
($C_{L,R}\to-C_{L,R}$ and, at the rate level, partial $C_L\leftrightarrow C_R$ mirrors),
which appear as islands; the spectral shape and the tiny NC-only component break them only
weakly. Around the SM point, the Fisher ellipse tightens with exposure.
"""),

("code", r"""
import time as _t
t0 = _t.time()
gv_grid = np.linspace(-2.7, 1.7, 221)
ga_grid = np.linspace(-2.7, 1.7, 221)
chi2 = np.empty((ga_grid.size, gv_grid.size))
for j, ga in enumerate(ga_grid):
    for i, gv in enumerate(gv_grid):
        chi2[j, i] = r.chi2_gv_ga(gv, ga, 30.0)
print(f"({_t.time()-t0:.0f} s for {chi2.size} points)")

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
ax = axes[0]
pl.confidence_contours(ax, gv_grid, ga_grid, chi2, color=pl.BLUE,
                       levels=pl.DELTA_CHI2_2DOF, label="30 MW$\\cdot$yr")
ax.set_xlabel(r"$g_V$"); ax.set_ylabel(r"$g_A$")
ax.set_title(r"Global structure: the degenerate islands")
ax.legend(fontsize=8.5, loc="upper right")

ax = axes[1]
for mwyr, color in [(10, pl.BLUE), (30, pl.ORANGE), (100, pl.GREEN)]:
    C = r.fisher_gv_ga(float(mwyr))
    pl.error_ellipse(ax, (GV_SM, GA_SM), C, n_sigma=1.0, color=color,
                     label=f"{mwyr} MW$\\cdot$yr", lw=1.8)
ax.set_xlim(GV_SM - 0.06, GV_SM + 0.06); ax.set_ylim(GA_SM - 0.05, GA_SM + 0.05)
ax.set_xlabel(r"$g_V$"); ax.set_ylabel(r"$g_A$")
ax.set_title(r"Around the SM point ($1\sigma$ Fisher)")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

for mwyr in (10, 30, 100):
    C = r.fisher_gv_ga(float(mwyr))
    print(f"{mwyr:4.0f} MW.yr: sigma(gV) = {np.sqrt(C[0,0]):.4f}, "
          f"sigma(gA) = {np.sqrt(C[1,1]):.4f}, rho = {C[0,1]/np.sqrt(C[0,0]*C[1,1]):+.3f}")
"""),

("md", r"""
## 6. The IBD anchor: fitting both channels jointly

Sections 4–5 treated IBD only as a background. But the IBD sample is $20\times$ the
E$\nu$ES rate with the *same* flux, and its cross section is the best-known in low-energy
neutrino physics: the normalisation is set by $\tau_n$ (now $878.4\pm0.5$ s, $0.06\%$) and
radiative corrections — Strumia–Vissani quoted $0.4\%$ conservatively, the modern treatments
support $\sim0.1$–$0.2\%$ — with the shape known to well below $0.1\%$ across our window.

So the joint fit stacks both spectra, **frees the flux normalisation** (loose $10\%$ prior)
and lets the IBD channel measure it, with the transfer to E$\nu$ES limited by three
priors: the IBD cross-section normalisation (default $0.2\%$), a zero-mean linear
cross-section shape tilt ($0.1\%$), and the channel ratio $N_e/N_p\times$ relative
efficiency (default $0.5\%$). The shared U235 flux-shape modes now hit both channels
coherently — which means the IBD spectrum constrains them *in situ*. The anchor spectrum
now also carries the **standard JUNO backgrounds** (far reactors, geoneutrinos, spallation,
etc., $\sim45$/day against the $58$k/day near signal) at their release priors, and the
shared **energy-scale pulls** are constrained by the anchor's statistics — they turn out
to be an exact null in the joint fit.
"""),

("code", r"""
print("sigma(sin^2 thetaW): EvES-only vs joint IBD+EvES")
rows = []
for m in exposures:
    a, b = r.sigma_sw2(m), r.sigma_sw2(m, joint=True)
    rows.append([f"{m:.0f}", a, b, a/b])
print(pl.table(rows, ["MW.yr", "EvES only", "joint", "gain"], floatfmt="{:.5f}"))
print()
print("The gain grows with exposure (4x -> 8x) because the anchored fit keeps profiting")
print("from statistics while the EvES-only fit saturates on the flux systematics.")
print()

rows = [["defaults (xsec 0.2%, ratio 0.5%)",
         FixedNearReactor(**STD).sigma_sw2(30.0, joint=True)],
        ["IBD xsec -> 0.5% (Strumia-Vissani conservative)",
         FixedNearReactor(**STD, sigma_ibd_xsec=0.005).sigma_sw2(30.0, joint=True)],
        ["IBD xsec -> 0.1% (modern)",
         FixedNearReactor(**STD, sigma_ibd_xsec=0.001).sigma_sw2(30.0, joint=True)],
        ["channel ratio -> 0.2%",
         FixedNearReactor(**STD, sigma_channel_ratio=0.002).sigma_sw2(30.0, joint=True)],
        ["both at their best (0.1%, 0.2%)",
         FixedNearReactor(**STD, sigma_ibd_xsec=0.001,
                          sigma_channel_ratio=0.002).sigma_sw2(30.0, joint=True)],
        ["no U235 shape covariance",
         FixedNearReactor(**STD, use_flux_covariance=False).sigma_sw2(30.0, joint=True)],
        ["legacy systematics (pre-upgrade)",
         FixedNearReactor(**STD, include_backgrounds=False, sigma_scale=1e-9,
                          sigma_bias=1e-9, sigma_res=1e-9,
                          sigma_evolution=1e-9).sigma_sw2(30.0, joint=True)]]
print(pl.table(rows, ["joint fit at 30 MW.yr", "sigma(sin^2 thetaW)"], floatfmt="{:.5f}"))
print()
print("Structural results. The U235 shape covariance stays a NULL (the 7e7-event anchor")
print("measures the flux shape in situ), and so are the energy-scale pulls -- the anchor")
print("pins the shared response. The upgrade costs "
      f"{100*(rows[0][1]/rows[-1][1]-1):.0f}% against the legacy systematics, split")
print("roughly evenly between the fuel-evolution mode and the solar+JUNO backgrounds.")
print(f"The limiting transfer terms remain: channel ratio (0.5% -> 0.2% buys "
      f"{rows[0][1]:.5f} -> {rows[3][1]:.5f}) and the IBD cross-section normalisation;")
print(f"at their plausible best the measurement reaches {rows[4][1]:.5f} -- a "
      f"{100*rows[4][1]/0.2223:.2f}% determination of sin^2 thetaW at low energy.")
"""),

("code", r"""
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for mwyr, color in [(30, pl.ORANGE), (100, pl.GREEN)]:
    C0 = r.fisher_gv_ga(float(mwyr))
    C1 = r.fisher_gv_ga(float(mwyr), joint=True)
    pl.error_ellipse(ax, (GV_SM, GA_SM), C0, n_sigma=1.0, color=color, ls="--", lw=1.4,
                     label=f"{mwyr} MW$\\cdot$yr, E$\\nu$ES only")
    pl.error_ellipse(ax, (GV_SM, GA_SM), C1, n_sigma=1.0, color=color, lw=1.9,
                     label=f"{mwyr} MW$\\cdot$yr, joint")
ax.set_xlim(GV_SM - 0.05, GV_SM + 0.05); ax.set_ylim(GA_SM - 0.04, GA_SM + 0.04)
ax.set_xlabel(r"$g_V$"); ax.set_ylabel(r"$g_A$")
ax.set_title(r"$1\sigma$ regions, IBD-anchored vs E$\nu$ES-only")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

for mwyr in (30, 100):
    C = r.fisher_gv_ga(float(mwyr), joint=True)
    print(f"joint, {mwyr:4.0f} MW.yr: sigma(gV) = {np.sqrt(C[0,0]):.4f}, "
          f"sigma(gA) = {np.sqrt(C[1,1]):.4f}, rho = {C[0,1]/np.sqrt(C[0,0]*C[1,1]):+.3f}")
"""),

("md", r"""
## The $(g_V, g_A)$ plane against existing constraints

Convention: $g_V=-\tfrac12+2\sin^2\theta_W$, $g_A=-\tfrac12$ (PDG, CHARM II, TEXONO, LSND).
Electron-flavour experiments see the CC piece too, so their cross sections depend on
$(g_V+1,\,g_A+1)$ and constrain **bands** along the resulting degeneracies —
$\nu_e e$ (LSND): $(g_V+g_A+2)^2+(g_V-g_A)^2/3$; $\bar\nu_e e$ (TEXONO):
$(g_V-g_A)^2+(g_V+g_A+2)^2/3$ — not ellipses. Muon-flavour experiments (CHARM II, BNL E734)
are pure NC and give ellipses; the PDG world average is dominated by CHARM II. Neutrino
tridents (CCFR, CHARM II) constrain the $\nu_\mu$–*muon* couplings, not $\nu$–$e$, and are
not drawn. Our $\bar\nu_e$ measurement lands on TEXONO's band and shrinks it by two orders
of magnitude. (`reactor/gvga_limits.py` holds the numbers and the plotting function.)
"""),

("code", r"""
from reactor.gvga_limits import draw_gvga_limits

fig, ax = plt.subplots(figsize=(5.8, 5.4))
ax.set_xlim(-0.6, 0.5); ax.set_ylim(-1.0, 0.1)
draw_gvga_limits(ax, pl, zoom=False)
pl.error_ellipse(ax, (GV_SM, GA_SM), r.fisher_gv_ga(30.0, joint=True), n_sigma=1.0,
                 color=pl.RED, lw=2.0, label=r"this work, 30 MW$\cdot$yr joint")
ax.set_xlabel(r"$g_V$"); ax.set_ylabel(r"$g_A$")
ax.set_title(r"Existing constraints in the $(g_V, g_A)$ plane")
ax.legend(fontsize=10, loc="upper left")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(5.8, 5.4))
for mwyr, color in ((30, pl.ORANGE), (100, pl.GREEN)):
    pl.error_ellipse(ax, (GV_SM, GA_SM), r.fisher_gv_ga(float(mwyr)), n_sigma=1.0, color=color,
                     ls="--", lw=1.3, label=rf"{mwyr} MW$\cdot$yr, E$\nu$ES only")
    pl.error_ellipse(ax, (GV_SM, GA_SM), r.fisher_gv_ga(float(mwyr), joint=True), n_sigma=1.0,
                     color=color, lw=1.9, label=rf"{mwyr} MW$\cdot$yr, joint")
ax.set_xlim(GV_SM - 0.06, GV_SM + 0.06); ax.set_ylim(GA_SM - 0.05, GA_SM + 0.05)
draw_gvga_limits(ax, pl, zoom=True)
ax.set_xlabel(r"$g_V$"); ax.set_ylabel(r"$g_A$")
ax.set_title(r"$1\sigma$ regions, zoom on the SM point")
ax.legend(fontsize=10, loc="lower left")
plt.tight_layout(); plt.show()

print("Our joint 30 MW.yr ellipse (semi-axes 0.004 x 0.002) sits inside the CHARM II")
print("(0.017 x 0.017) and PDG world-average (0.015 x 0.014) circles: a factor ~5-8 tighter")
print("in each direction, and from a DIFFERENT flavour (nubar_e vs CHARM II's nu_mu). All")
print("contours are centred on the PDG world-average point (gV, gA) = (-0.040, -0.507),")
print("which is the module's reference (effective sin^2 thetaW = 0.230).")
"""),

("md", r"""
## Summary

For a $10$ MW$_{\rm th}$ HALEU reactor at $50$ m ($\approx9$ MW$\cdot$yr per calendar year):

| quantity | value |
|---|---|
| IBD rate (burnup $0.5$) | $\approx58{,}000$/day — $\sim1600\times$ the far reactors |
| E$\nu$ES rate, $T\in[1,6.5]$ MeV | $\approx3{,}100$/day, $99.99\%$ $\bar\nu_e$ |
| IBD-singles background ($1\%$ untagged) | $\approx580$/day |
| solar E$\nu$ES background | $\approx180$/day (live-equivalent) |
| $\sigma(\sin^2\theta_W)$, 30 MW·yr, E$\nu$ES only | $0.0047$ ($2.1\%$) |
| — **joint IBD+E$\nu$ES anchor** | $0.0011$ ($0.5\%$) |
| — joint, best-case transfer terms | $0.00054$ |
| — statistics only | $0.0001$ |
| $(g_V,g_A)$, 30 MW·yr | $\sigma(g_V)\approx0.014$, $\sigma(g_A)\approx0.013$, $\rho\approx+0.7$ |

E$\nu$ES alone is **flux-shape-limited**: the measured U235 covariance saturates it at
$\approx0.0047$ regardless of exposure. **The joint IBD+E$\nu$ES fit removes that wall**:
with the flux normalisation freed and measured by the IBD channel, the normalisation, the
flux shape *and the energy-scale pulls* all become nulls — the anchored fit is limited by
the transfer terms (IBD cross section $0.2\%$, channel ratio $0.5\%$) plus, after this
upgrade, the fuel-evolution mode and the solar/JUNO backgrounds ($\approx+6\%$ and
$+7\%$ respectively), giving $\sigma(\sin^2\theta_W)=0.0011$ at 30 MW·yr and $0.00054$
at the plausible best of the transfer inputs — still a $\sim0.25\%$ low-energy weak
mixing angle. The anchoring shrinks the $(g_V,g_A)$ ellipse to
$\sigma(g_V)\approx0.0026$, $\sigma(g_A)\approx0.0033$ at 30 MW·yr (the $g_A$ direction
absorbs most of the new shape systematics). The **atomic stepping correction is applied
and exactly null** above $1$ keV — it would matter only for a sub-keV-threshold analysis.

Caveats stated once more: no radioactivity or cosmogenic singles (severe below $\sim3$ MeV);
flux undefined below the IBD threshold (conservative at low recoil); tree-level cross
sections (radiative corrections shift effective couplings at the percent level and matter at
the precision frontier).
"""),
]
