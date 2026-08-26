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
  duty cycle $0.9$; $N_e = 4.689\,N_p = 6.76\times10^{33}$ electrons.
* **Backgrounds modelled: IBD singles** — IBD events whose delayed-neutron tag is missed
  (default untagged fraction $1\%$), promoting the positron to a single hit — and the
  **detector singles**, natural radioactivity and cosmogenic isotopes, anchored to JUNO's
  measured $4.7$ Hz and subtracted with reactor-off running (§3b).
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

The delivered scintillator has a hydrogen mass fraction of $12.01\%$, which is what
fixes $N_p = 1.44\times10^{33}$; the same number fixes the electron count, since
$\mathrm{H/C} = 1.63$ by number follows from it. One carbon then comes with $1.63$ free
protons and $7.63$ electrons, i.e. $N_e = 4.689\,N_p$, and both target counts are derived
from the one composition in `reactor.constants` rather than quoted separately. Below,
every cross section is quoted per CH$_{1.63}$ unit. (The $\bar\nu_e\to\bar\nu_e e^+e^-$ trident channel was
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
ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel(r"$\sigma$ per CH$_{1.63}$ unit [cm$^2$]")
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
## 3b. Detector singles, and the reactor-off measurement that removes them

The dominant single-hit background is not the reactor at all — it is JUNO's own
detector. Two recent measurements pin it down:

* **arXiv:2607.17509** (final radioactivity assessment) measures, in *first data*,
  a fiducial singles rate of $\mathbf{4.7}$ **Hz above 0.7 MeV for $R<17.2$ m** —
  better than the 7.2 Hz design budget. The delivered LS is U/Th $\lesssim3.4\times10^{-17}$ g/g,
  ~30× purer than the $10^{-15}$ requirement, so the internal component collapses to
  $\lesssim0.1$ Hz and **~95% of the fiducial singles are external $\gamma$** from the
  acrylic, PMT glass and steel truss.
* **PoS(ICRC2025)1041** (solar potential) shows the same backgrounds from the solar side:
  cosmogenic $^{11}$C is why JUNO's own $^8$B analysis cannot go below 2 MeV.

The spectral consequence is sharp and is what makes the measurement possible. An
*external* $\gamma$ can deposit at most its own line energy, so the external component
**stops dead at the 2.615 MeV $^{208}$Tl line**. An *internal* decay deposits its full
cascade, so internal $^{208}$Tl reaches 5.0 MeV — but at only ~84 decays/day in 20 kt at
the measured purity. Above 3 MeV the singles fall by three orders of magnitude.

Everything here is `reactor/singles.py`. The **rate** is anchored to JUNO's measurement;
the **shape** is a component model (lines + Compton continua + $\beta$ shapes) and the
**radial profile** an effective exponential tuned to their Fig. 6 rather than a $\gamma$
attenuation length — their own FV reconstruction systematic is 20–30%. Cosmogenic rates
are depth-scaled estimates good to a factor of a few. So the analysis below is built to
*not depend* on these numbers being right.
"""),

("code", r"""
from reactor.singles import SinglesBackground

cen = 0.5 * (r.recoil_edges[1:] + r.recoil_edges[:-1])
print(f"singles in the window at R = {r.fiducial_radius_m} m: "
      f"{r.singles_rate_per_day:,.0f}/day, against {float(r.eves_spectrum_T().sum())*DAY:,.0f}/day of EvES\n")
rows = []
for a_, b_ in ((1, 2), (2, 3), (3, 4), (4, 6.5)):
    e = np.array([a_, b_])
    tot = float(r.singles.spectrum(e)[0]) * DAY
    ext = float(r.singles.components(e)["external gammas"][0]) * DAY
    tl = float(r.singles.components(e)["internal 208Tl"][0]) * DAY
    sg = float(r.eves_spectrum_T()[(cen >= a_) & (cen < b_)].sum()) * DAY
    rows.append([f"{a_}-{b_}", tot, ext, tl, sg, sg / tot if tot else np.inf])
print(pl.table(rows, ["T [MeV]", "singles/day", "external g", "int 208Tl",
                      "EvES/day", "S/B"], floatfmt="{:.4g}"))

fig, ax = plt.subplots(figsize=(7.0, 4.6))
w = np.diff(r.recoil_edges)
ax.stairs(r.eves_spectrum_T() * DAY / w, r.recoil_edges, color=pl.BLUE, lw=2.0,
          label=r"E$\nu$ES signal")
comp = r.singles_components
for name, color, ls in (("external gammas", pl.RED, "-"),
                        ("internal 214Bi", pl.ORANGE, "--"),
                        ("internal 208Tl", pl.MAGENTA, "-."),
                        ("cosmogenic 11C", pl.GREEN, "--"),
                        ("cosmogenic 11Be", pl.INK_SECONDARY, ":")):
    if name in comp:
        ax.stairs(comp[name] * DAY / w, r.recoil_edges, color=color, lw=1.5, ls=ls,
                  label=name)
ax.axvline(2.615, color=pl.INK_MUTED, lw=1.0, ls=":")
ax.annotate(r"$^{208}$Tl 2.615 MeV" "\n" r"external $\gamma$ wall", xy=(2.68, 3e3),
            fontsize=9, color=pl.INK_SECONDARY)
ax.set_yscale("log"); ax.set_ylim(1e-2, 1e6)
ax.set_xlabel(r"$T_{\rm rec}$ [MeV]"); ax.set_ylabel("events / MeV / day")
ax.set_title("Detector singles against the reactor signal")
ax.legend(fontsize=10, ncol=2)
plt.tight_layout(); plt.show()
"""),

("md", r"""
### Neutrino interactions on $^{13}$C — background, and signal

Natural carbon is $1.1\%$ $^{13}$C, so the scintillator holds $\sim194$ t of it
($9.0\times10^{30}$ nuclei) — the target JUNO uses for its model-independent $^8$B
measurement. Both channels land in our window, and they behave very differently:

* **CC**, $\nu_e + {}^{13}$C $\to e^- + {}^{13}$N(g.s.), threshold $2.2$ MeV. This needs a
  *neutrino*: the antineutrino partner $\bar\nu_e + {}^{13}$C $\to e^+ + {}^{13}$B has a
  threshold of $14.5$ MeV, far above the reactor spectrum. So the reactor contributes
  **nothing** and this is a purely solar background, tagged besides by the $863$ s
  $^{13}$N $\beta^+$ coincidence.
* **NC**, $\nu_x + {}^{13}$C $\to \nu_x + {}^{13}$C$^*(3/2^-, 3.685$ MeV$)$ — flavour
  blind, so **the reactor antineutrinos excite it too**. The level sits below the $4.946$
  MeV neutron separation energy, so it is particle-bound and de-excites by a single
  $3.685$ MeV $\gamma$: a monoenergetic line, $73$ keV wide, landing **above the $2.615$
  MeV wall** in the cleanest part of the window.

#### Why the NC line measures the isovector axial current

At quark level $\mathcal{L}_{\rm NC}=-\frac{G_F}{\sqrt2}[\bar\nu\gamma^\mu(1-\gamma_5)\nu]
\sum_q\bar q\gamma_\mu(g_V^q-g_A^q\gamma_5)q$. Rearranged into isospin components this is
$$J_\mu^{Z}=J_\mu^{3}-2s_W^2 J_\mu^{\rm em},\qquad J_\mu^{3}=V_\mu^{3}-A_\mu^{3},$$
and since the electromagnetic current is purely vector, **the entire axial part of the NC
is $-A_\mu^3$, with coefficient exactly $1$ and no $\sin^2\theta_W$ at all.** Only the
vector part is mixed and $s_W^2$-suppressed. At the nucleon level, non-relativistically,
$$A^3_k \to \frac{g_A}{2}\sum_i\sigma_{k,i}\tau^3_i + \frac{\Delta s}{2}\sum_i\sigma_{k,i},$$
an isovector spin operator with $g_A=1.2754$ plus an isoscalar one carried by the strange
spin, $\Delta s\approx-0.08$, an order of magnitude down.

The transition does the rest. $^{13}$C(g.s.) is $1/2^-$ and the level is $3/2^-$: $\Delta
J=1$, no parity change. The leading *vector* operator is the Fermi charge $\sum_i\tau^3_i$,
which is $\Delta J=0$ and **cannot connect the two states**; the vector current enters only
via weak magnetism, suppressed by $q/2M_N\sim10^{-3}$. The leading *axial* operator is
exactly the $\Delta J=1$ spin operator. Hence
$$\sigma_{\rm NC}(E_\nu)=\frac{G_F^2}{\pi}(E_\nu-E_x)^2\frac{g_A^2}{4}B({\rm GT}^0),
\qquad B({\rm GT}^0)\equiv\frac{|\langle f\|\sum_i\sigma_i\tau^3_i\|i\rangle|^2}{2J_i+1},$$
with $E_x=3.685$ MeV and $(E_\nu-E_x)^2$ the outgoing-neutrino phase space. **The line
carries no leading $\sin^2\theta_W$ dependence**: it measures $g_A^2B({\rm GT}^0)$, so it is
not a second weak-mixing-angle measurement but a different observable — sensitive to the
hadronic axial current where E$\nu$ES is sensitive to the electron couplings, and to new
physics touching quark neutral currents without touching electrons. The same isovector spin
operator governs the $M1$ strength of the level, so $B({\rm GT}^0)$ is constrained by
nuclear data.

For CC, $\sigma_{\rm CC}=\frac{G_F^2\cos^2\theta_C}{\pi}p_eE_eF(Z{=}7,E_e)[B({\rm F})+
g_A^2B({\rm GT})]$ with $E_e=E_\nu-Q$. $^{13}$N(g.s.) is the isobaric analog of
$^{13}$C(g.s.), so $B({\rm F})=1$ and the bracket is fixed by the measured $ft$ of the
mirror $^{13}$N $\to{}^{13}$C $e^+\nu$ decay — **no nuclear model needed**.

Normalisation here (`reactor/carbon13.py`): I take the energy dependence above and fix the
two overall scales by JUNO's own $^8$B yields ($3032$ NC and $3929$ CC in 10 years) rather
than evaluating $B({\rm GT}^0)$ from a shell model. Carrying that law from the $^8$B
weighting ($3.7$–$15$ MeV) down to the reactor's ($3.7$–$9$ MeV) is a modest extrapolation,
so read the reactor rate as good to a factor of order unity.
"""),

("code", r"""
from reactor import carbon13 as c13

print(f"13C target: {c13.N_13C:.2e} nuclei ({c13.LS_MASS_G*c13.CARBON_MASS_FRACTION*c13.C13_ABUNDANCE/1e6:.0f} t)")
print(f"sigma_NC(8 MeV) = {c13.sigma_nc(8.0):.2e} cm^2   sigma_CC(8 MeV) = {c13.sigma_cc(8.0):.2e} cm^2")
print()
print(f"reactor 13C NC line  {r.c13_nc_rate_per_day:5.2f}/day   <- flavour-blind, so nubar works")
print(f"solar   13C NC + CC  {r.c13_solar_rate_per_day:5.2f}/day   <- cancels in the reactor-off subtraction")
print()
print("As a background to EvES it is negligible; as a measurement it is not:")
rows = [[f"{m:.0f}", r.c13_nc_rate_per_day * m / 9.0 * 365.25, r.c13_nc_significance(m)]
        for m in (10.0, 30.0, 100.0)]
print(pl.table(rows, ["MW.yr", "line events", "significance [sigma]"], floatfmt="{:.4g}"))
print()
print(f"cost to sin^2 thetaW of including all the 13C: "
      f"{FixedNearReactor(**STD, include_c13=False).sigma_sw2(30.0, joint=True):.5f}"
      f" -> {r.sigma_sw2(30.0, joint=True):.5f}")
print()
print("This is a neutral-current measurement on a nucleus at reactor energies: it probes the")
print("isovector axial current, not the electron couplings EvES measures, and comes free with")
print("the same data.  The line is the one place in the window where a sharp feature sits on")
print("a smooth continuum, so it is limited by statistics rather than by any shape systematic.")
"""),

("md", r"""
### Neutrino interactions on deuterium

Natural hydrogen carries $156$ ppm of deuterium, so JUNO's free protons come with
$2.2\times10^{29}$ deuterons — about $750$ kg of D, $2.3\%$ of the $^{13}$C count.
Deuterium is the target one would most like to have: the neutral-current breakup
$\nu d\to\nu np$ is a pure Gamow-Teller transition on the simplest of all nuclei, with a
matrix element known from pionless EFT to about $1\%$. Two channels are open,

$$\bar\nu_e + d \to \bar\nu_e + n + p, \qquad E_{\rm th} = B_d = 2.224~\mathrm{MeV},$$
$$\bar\nu_e + d \to e^+ + n + n, \qquad E_{\rm th} = B_d + (m_n-m_p) + m_e = 4.028~\mathrm{MeV},$$

and the gap between the thresholds is what decides everything: the NC samples $71\%$ of
the flux and the CC only $15\%$ — a factor of five that the cross sections then more than
undo, since the CC is the larger of the two per neutrino. As for $^{13}$C the *shapes*
are derived (allowed phase space against a two-body continuum) and the two overall
*scales* are external: the per-fission cross sections at $^{235}$U-dominated reactors,
$\sigma_{\rm CC} = 1.06\times10^{-44}$ and $\sigma_{\rm NC} = 5.6\times10^{-45}$
cm$^2$/fission, known to about $10\%$.
"""),

("code", r"""
d = r.deuterium()
print(f"deuterons in the fiducial target: {d['N_D']:.2e}  ({d['D mass [kg]']:.0f} kg of D)")
print(f"flux above the NC threshold: {100*d['flux fraction > NC threshold']:.1f}%   "
      f"above the CC threshold: {100*d['flux fraction > CC threshold']:.1f}%")
print()
rows = [["CC (e+ n n)", d["CC"], d["CC"] * 365.25 * 3, r.deuterium_cc_significance(30.0)],
        ["NC (n p)", d["NC"], d["NC"] * 365.25 * 3, r.deuterium_nc_significance(30.0)]]
print(pl.table(rows, ["channel", "events/day", "per 30 MW.yr", "significance [sigma]"],
               floatfmt="{:.4g}"))
print()
print(f"CC accidental (IBD + a second neutron within 1 m, 1 ms): "
      f"{d['CC accidental / day']:.1e}/day against {d['CC']:.2f}/day of signal")
print()
print("The two channels part company completely, and it is the SIGNATURE rather than the")
print("rate that decides.  The charged current gives a prompt positron followed by TWO")
print("neutron captures sharing a vertex -- a triple coincidence whose only serious")
print("accidental is an ordinary IBD picking up a second neutron from another IBD.  Both")
print("that and the signal scale with reactor power, so the signal-to-background is a")
print("property of the vertex cut and not of the exposure; the background that would need")
print("real work is cosmogenic fast neutrons, which genuinely make multi-neutron events.")
print("The neutral current, by contrast, produces a lone 2.22 MeV capture gamma with no")
print("prompt at all (the proton recoil is sub-MeV and heavily quenched), and even with")
print("equal reactor-off subtraction it is inaccessible.")
print()
print("The channel one wants is precisely the one the abundance denies:")
rows = [["JUNO as built (156 ppm D)", d["CC"], d["NC"]],
        ["scintillator doped to 1% D", d["doped 1% D"]["CC"], d["doped 1% D"]["NC"]],
        ["1 t D2O cell at 10 m", d["1 t D2O at 10 m"]["CC"], d["1 t D2O at 10 m"]["NC"]]]
print(pl.table(rows, ["configuration", "CC / day", "NC / day"], floatfmt="{:.3g}"))
print("Neither of the last two is a modification to JUNO; both are separate experiments.")
print("What the parked reactor does deliver for free is a respectable nubar_e d charged-")
print("current measurement riding on the same data as everything else.")
"""),

("md", r"""
### Turning the reactor off

A parked research reactor can be **switched off**, and the singles cannot. A reactor-off
run of $r\times$ the on-time measures the whole non-reactor background *in situ*, bin by
bin — radioactivity, cosmogenics and solar together — with no model at all. Profiling a
free per-bin background $B$ with a flat prior is analytic, and leaves only the statistical
cost of the subtraction,

$$\mathrm{Var} = S + B_{\rm reactor} + B_{\rm non-reactor}\left(1 + 1/r\right),$$

so **equal on and off exposure ($r=1$) gives the familiar factor 2**. Crucially the result
then depends on $B$ only through $\sqrt{B}$, never through its assumed level or shape —
which is exactly the protection we need given how provisional the rates are. The price is
calendar time: $r=1$ means half the wall-clock is spent with the reactor off, so a given
MW$\cdot$yr takes twice as long to deliver.
"""),

("code", r"""
import time as _t
t0 = _t.time()
STD = dict(burnup=0.5)

print("(a) off/on exposure ratio.  A short off-run is the worst of both worlds: the")
print("    subtraction inflates the variance by B/r without buying much sample.  The")
print("    last row is a different analysis, not the r -> 0 limit of this one -- there")
print("    the singles SHAPE comes from the model and only its normalisation floats.")
rows = [[f"{rr:g}", FixedNearReactor(**STD, reactor_off_ratio=rr).sigma_sw2(30.0) * 100,
         FixedNearReactor(**STD, reactor_off_ratio=rr).sigma_sw2(30.0, joint=True) * 100]
        for rr in (0.02, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)]
rows.append(["no off-run", FixedNearReactor(**STD, reactor_off_ratio=0.0).sigma_sw2(30.0) * 100,
             FixedNearReactor(**STD, reactor_off_ratio=0.0).sigma_sw2(30.0, joint=True) * 100])
print(pl.table(rows, ["r = t_off/t_on", "EvES-only (%)", "joint (%)"], floatfmt="{:.3f}"))

print("\n(b) robustness: scale every singles rate, keeping r = 1")
rows = []
for sc in (0.0, 0.1, 1.0, 3.0, 10.0, 30.0):
    x = FixedNearReactor(**STD, singles_scale=sc, include_singles=sc > 0)
    rows.append([f"x{sc:g}", x.singles_rate_per_day,
                 x.sigma_sw2(30.0) * 100, x.sigma_sw2(30.0, joint=True) * 100])
print(pl.table(rows, ["singles scale", "rate/day", "EvES-only (%)", "joint (%)"],
               floatfmt="{:.4g}"))

print("\n(c) fiducial radius -- a fixed, reasonable choice, not an optimisation")
rows = []
for R in (15.5, 16.0, 16.5, 17.0):
    x = FixedNearReactor(**STD, fiducial_radius_m=R)
    rows.append([R, x.singles_rate_per_day, (R / 16.5) ** 3,
                 x.sigma_sw2(30.0, joint=True) * 100])
print(pl.table(rows, ["R [m]", "singles/day", "target rel.", "joint (%)"],
               floatfmt="{:.4g}"))
print(f"({_t.time()-t0:.0f} s)")

print()
print("Three things fall out.  (a) Equal on/off running captures essentially the whole")
print("benefit, and it is also the optimum at fixed CALENDAR time: minimising")
print("(1+r)[S + R + B(1+1/r)] gives r_opt = sqrt(B/(S+R+B)) -> 1 for B >> S, which is")
print("this regime.  A 2%-length off-run costs ~50% on the joint fit -- worse than none.")
print("(b) The answer is remarkably insensitive to the singles level: 10x the background")
print("costs 18%, 30x costs 44%.  That is the point of measuring rather than modelling it.")
print("(c) The fiducial radius barely matters (0.00118 -> 0.00122 over 15.5-17.0 m), so")
print("we simply keep R = 16.5 m rather than tuning a cut against provisional rates.")
"""),

("md", r"""
## 4. Precision on $\sin^2\theta_W$

The recoil spectrum is fit for $\sin^2\theta_W$ (with $g_A$ at its SM value) under the full
systematics model, as a function of the delivered exposure in MW$\cdot$yr.

Two scales frame the result. The rate carries $|d\ln R/d\sin^2\theta_W|\approx4$, so a pure
normalisation uncertainty *alone* would floor the precision at $\sigma_{\rm norm}/4$ — but
the recoil *shape* also changes with the couplings, and the fit exploits it: at $2\%$
normalisation with no flux-shape uncertainty, $\sigma(\sin^2\theta_W)\approx0.0034$, still
below the $0.005$ rate floor. What actually dominates is the **measured U235 spectrum-shape
covariance**: its $2$–$4\%$ correlated bin uncertainties can imitate the coupling-induced
shape change, and with them the precision saturates at $\approx0.0050$ almost independently
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
| E$\nu$ES rate, $T\in[1,6.5]$ MeV | $\approx3{,}200$/day, $99.99\%$ $\bar\nu_e$ |
| IBD-singles background ($1\%$ untagged) | $\approx580$/day |
| solar E$\nu$ES background | $\approx180$/day (live-equivalent) |
| **detector singles** (radioactivity + cosmogenic) | $\approx1.2\times10^5$/day at $R=16.5$ m |
| **$^{13}$C NC line** ($3.685$ MeV, reactor-driven) | $\approx4.2$/day — $12\sigma$ at 30 MW·yr |
| $\sigma(\sin^2\theta_W)$, 30 MW·yr, E$\nu$ES only | $0.0050$ |
| — **joint IBD+E$\nu$ES anchor** | $0.0012$ ($0.5\%$) |
| — joint, best-case transfer terms | $0.00073$ |
| — statistics only | $0.0001$ |
| $(g_V,g_A)$, 30 MW·yr, joint | $\sigma(g_V)\approx0.0035$, $\sigma(g_A)\approx0.0067$ |

E$\nu$ES alone is **flux-shape-limited**: the measured U235 covariance saturates it at
$\approx0.0050$ regardless of exposure. **The joint IBD+E$\nu$ES fit removes that wall**:
with the flux normalisation freed and measured by the IBD channel, the normalisation, the
flux shape *and the energy-scale pulls* all become nulls, leaving the transfer terms (IBD
cross section $0.2\%$, channel ratio $0.5\%$), the detector singles ($+11\%$), and the
fuel-evolution and solar/JUNO backgrounds — giving $\sigma(\sin^2\theta_W)=0.0012$ at
30 MW·yr and $0.00073$ at the plausible best of the transfer inputs, still a $\sim0.5\%$
low-energy weak mixing angle.

**The detector singles are the largest background by two orders of magnitude and cost very
little.** JUNO's measured $4.7$ Hz is $\sim40\times$ our signal in the window, but it is
$\sim95\%$ external $\gamma$, which cannot deposit above the $2.615$ MeV $^{208}$Tl line:
across that wall the singles drop by three orders of magnitude and $S/B$ turns over from
$0.02$ to $9$. Running the reactor **off for as long as on** measures the whole non-reactor
background in situ, so the cost enters only as $\sqrt{B}$ — a background $30\times$ larger
than modelled would still leave $\sigma(\sin^2\theta_W)=0.0017$. That is what makes the
result trustworthy despite estimate-grade rates, and it is why we fix $R=16.5$ m rather
than tuning a fiducial cut.

**A neutral-current line comes free.** $^{13}$C is $1.1\%$ of the carbon ($194$ t), and
$\nu_x + {}^{13}$C $\to \nu_x + {}^{13}$C$^*(3.685$ MeV$)$ is flavour-blind, so the reactor
antineutrinos excite it: a monoenergetic $3.685$ MeV $\gamma$ at $4.2$/day, landing above
the $^{208}$Tl wall in the cleanest part of the window and reaching $12\sigma$ at 30 MW·yr.
It probes the isovector axial current rather than the electron couplings E$\nu$ES measures.
The CC channel is solar-only — its $\bar\nu_e$ partner has a $14.5$ MeV threshold.

The **atomic stepping correction is applied and exactly null** above $1$ keV — it would
matter only for a sub-keV-threshold analysis.

Caveats stated once more: the singles *shape* is a component model, not a transport
simulation, and the cosmogenic rates are depth-scaled estimates (the analysis is built not
to depend on either); flux undefined below the IBD threshold (conservative at low recoil);
tree-level cross sections (radiative corrections shift effective couplings at the percent
level and matter at the precision frontier).
"""),
]
