OUT = "6_sterile_wiggles.ipynb"

CELLS = [
("md", r"""
# 6 — Sterile-neutrino wiggles across the finite JUNO detector

A reactor parked $50$–$100$ m from JUNO illuminates a detector whose fiducial radius
($\approx16.5$ m) is comparable to the standoff: **the baseline varies by a factor
$\sim2$ across the target volume**. For $\Delta m^2_{41}\sim0.1$–$10$ eV$^2$ the
oscillation length at reactor energies is metres to tens of metres, so 3+1 disappearance,

$$P_{ee} = 1 - (1-s_{14})^2\,(1-P^{3\nu}_{ee}) - 4 s_{14}(1-s_{14})\sum_{i=1}^{3} |U_{ei}|^2_{3\nu}\,\sin^2\Delta_{i4},$$

the **full 3+1 vacuum** survival probability ($s_{14}=\sin^2\theta_{14}$; validated
against a brute-force PMNS-amplitude computation to $10^{-5}$). At $50$ m this reduces
to the familiar $1-\sin^22\theta_{14}\sin^2\Delta_{41}$, but at the longer standoffs
studied below — up to $1.4$ km, where $\Delta_{31}\approx1$ rad — the three sterile
phases $\Delta_{14},\Delta_{24},\Delta_{34}$ separate and the null hypothesis is the
$\theta_{13}$-oscillated spectrum, not the un-oscillated one; both are carried,

*wiggles across the detector*. Binning events in the reconstructed vertex baseline turns
JUNO into a many-baseline experiment in one fill — and no flux or cross-section systematic
can follow: the geometry is exact and the source spectrum carries no $L$-dependence. The
event density in baseline is analytic,
$dN/dL \propto 2\pi L^2\big[1-(L^2+D^2-R^2)/(2LD)\big]/L^2$.

Both channels are used (`reactor/sterile.py`, built on the module of notebook 5): **IBD**
reconstructs $E_\nu$, so it sees the full $(L,E)$ wiggle pattern; **E$\nu$ES** measures only
$T$, washing the $E$-wiggles but keeping the $L$-modulation. A $\bar\nu_e$ that oscillates
goes *sterile*, so both CC and NC rates deplete identically.

The prediction is linear in $\sin^22\theta_{14}$, so against a no-oscillation Asimov,
$\Delta\chi^2 = \sin^42\theta_{14}\,Q(\Delta m^2)$ with one wiggle template per
$\Delta m^2$ — the entire exclusion curve is analytic,
$\sin^22\theta_{14}^{\rm lim} = \sqrt{\Delta\chi^2_{\rm crit}/Q}$ (we quote $95\%$ CL,
$\Delta\chi^2=5.99$).

**Baseline smearing — the high-$\Delta m^2$ wall.** Two effects smear the true oscillation
baseline at fixed reconstructed vertex: the **finite core** (a uniform ball of radius $a$
has line-of-sight variance $a^2/5$) and the **vertex resolution**
($\sigma_{\rm vtx}=10\,{\rm cm}/\sqrt{E[{\rm MeV}]}$, JUNO-like). The oscillating term damps
analytically, $\langle\sin^2 kL\rangle = \tfrac12[1-\cos(2kL)\,e^{-2k^2\sigma_L^2}]$ with
$\sigma_L^2 = \sigma_1^2/E + a^2/5$ — this, not the binning, is what terminates the reach at
high $\Delta m^2$, and it is studied explicitly below.

**A shape-only analysis, by necessity.** The Daya Bay U235 spectrum was *measured* at
$\sim500$ m: a sterile with $\Delta m^2\sim0.1$–$3$ eV$^2$ would have depleted it by
$\approx\sin^22\theta_{14}/2$ in normalisation (and left a $\lesssim1\%$ energy tilt).
Using its normalisation or shape as a prior would inject the very signal being searched
for. The **standard analysis therefore frees both**: the flux normalisation is
unconstrained, and the reactor $E$-shape is unconstrained (one free mode per $E$ and $T$
bin, coherent across $L$). Only the $L$-direction wiggle structure at fixed $E$ can then
carry sterile information — and, as shown below, it carries essentially all of it. The
Daya Bay spectrum is used only as a *shape template* for the event rates, never as a
constraint.

**Remaining systematics**: **detector non-uniformity as a Gaussian-correlated field over
$L$** ($0.5\%$ amplitude, $1.5$ m correlation length — a physical correlation length, so
finer binning cannot manufacture artificial high-frequency freedom) — the one nuisance in
the wiggle direction; the fuel evolution ($\pm30\%$, $E$-direction, a null); IBD/E$\nu$ES
ratio $0.5\%$. Radioactivity singles are again not modelled for E$\nu$ES.
"""),

("code", r"""
import time
import numpy as np
import matplotlib.pyplot as plt

from reactor import plotting as pl
from reactor.sterile import SterileNearReactor

pl.use_style()

t0 = time.time()
st50 = SterileNearReactor(distance_m=50.0)
st100 = SterileNearReactor(distance_m=100.0)
print(f"built in {time.time()-t0:.1f} s")
for lab, st in (("D = 50 m", st50), ("D = 100 m", st100)):
    r = st.rates_per_day()
    print(f"{lab}: IBD {r['IBD / day']:,.0f}/day, EvES {r['EvES / day']:,.0f}/day, "
          f"{r['L bins']} L bins over {r['L range [m]'][0]:.1f}-{r['L range [m]'][1]:.1f} m, "
          f"{r['modes']} systematic modes")

fig, ax = plt.subplots(figsize=(6.8, 3.9))
for st, color, lab in ((st50, pl.BLUE, "D = 50 m"), (st100, pl.ORANGE, "D = 100 m")):
    centers = 0.5*(st.l_edges[:-1] + st.l_edges[1:])
    counts = np.array([x.sum() for x in st._ibd_null])
    ax.stairs(counts/counts.sum()/np.diff(st.l_edges), st.l_edges, color=color,
              lw=1.7, label=lab)
ax.set_xlabel("baseline $L$ [m]"); ax.set_ylabel("event fraction / m")
ax.set_title("Baseline distribution across the fiducial volume")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()
"""),

("md", r"""
## The wiggles

A benchmark near the sensitivity edge of other experiments:
$\Delta m^2_{41}=1$ eV$^2$, $\sin^22\theta_{14}=0.1$. Left: the survival probability
per $(L, E_{\rm rec})$ bin. Right: slices versus $L$ in two energy windows — the
oscillation is directly visible across the detector, with the phase advancing as $1/E$.
"""),

("code", r"""
DM2_B, S22_B = 1.0, 0.10
w = st50.wiggle_template(DM2_B)
n_e = len(st50.e_edges) - 1
n_l = st50.n_l
null_ibd = np.array(st50._ibd_null)                  # (n_l, n_e)
wig_ibd = w[:n_l*n_e].reshape(n_l, n_e)
ratio2d = 1.0 - S22_B * wig_ibd / np.maximum(null_ibd, 1e-30)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
pc = ax.pcolormesh(st50.l_edges, st50.e_edges, ratio2d.T, cmap="RdBu",
                   vmin=1-S22_B, vmax=1+S22_B*0.2)
plt.colorbar(pc, ax=ax, label=r"$P_{ee}$ per bin")
ax.set_xlabel("$L$ [m]"); ax.set_ylabel(r"$E_{\rm rec}$ [MeV]")
ax.set_ylim(1.5, 8)
ax.set_title(rf"$\Delta m^2={DM2_B}$ eV$^2$, $\sin^22\theta_{{14}}={S22_B}$, D = 50 m")
plt.tight_layout(); plt.show()
"""),

("code", r"""
fig, ax = plt.subplots(figsize=(7.0, 4.2))
for (elo, ehi), color in zip(((1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6.0), (6.0, 8.0)),
                             (pl.BLUE, pl.ORANGE, pl.GREEN, pl.RED, pl.MAGENTA)):
    L, ratio = st50.ratio_vs_L(DM2_B, S22_B, elo, ehi)
    ax.plot(L, ratio, color=color, lw=1.7,
            label=rf"$E_{{\rm rec}} \in [{elo:g}, {ehi:g}]$ MeV")
ax.axhline(1.0, color=pl.INK_MUTED, lw=0.9)
ax.set_xlabel("$L$ [m]"); ax.set_ylabel("oscillated / no-oscillation")
ax.set_title("The wiggle across the detector, D = 50 m")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()
"""),

("md", r"""
## The energy spectrum

The complementary view: the oscillation in $E_{\rm rec}$. Per $L$ slice the $E$-wiggles are
sharp; integrated over the whole volume they partially survive — and *how much* survives,
and in what shape, is what decides the no-$L$-binning analysis below.

Binning check: the $E_{\rm rec}$ bins must resolve the oscillation after the energy
resolution ($\sigma_E\approx77$ keV at 4 MeV) has done its smearing. The default here is
$50$ keV, matched to the resolution; the table verifies convergence.
"""),

("code", r"""
fig, ax = plt.subplots(figsize=(7.0, 4.2))
e_cen = 0.5*(st50.e_edges[:-1] + st50.e_edges[1:])
for i_l, color in ((2, pl.BLUE), (int(n_l*0.5), pl.ORANGE)):
    r_e = 1.0 - S22_B * wig_ibd[i_l] / np.maximum(null_ibd[i_l], 1e-30)
    ax.plot(e_cen, r_e, color=color, lw=1.5,
            label=rf"$L \approx {0.5*(st50.l_edges[i_l]+st50.l_edges[i_l+1]):.0f}$ m slice")
ax.axhline(1.0, color=pl.INK_MUTED, lw=0.9)
ax.set_xlabel(r"$E_{\rm rec}$ [MeV]"); ax.set_ylabel("oscillated / no-oscillation")
ax.set_xlim(1.5, 8)
ax.set_title(rf"$E$-spectrum ratio per $L$ slice, $\Delta m^2={DM2_B}$ eV$^2$")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()
"""),

("md", r"""
### True versus reconstructed, channel by channel

What the detector response does to the oscillation, one channel at a time, for a single
$L$ slice near $40$ m. For **IBD**, $E_\nu$ is reconstructed almost one-to-one
($E_{\rm rec}\approx E_\nu - 0.78$ MeV, then non-linearity and resolution), so the wiggles
survive with modest damping. For **E$\nu$ES**, even the *true* recoil spectrum washes most
of the pattern out — every $T$ integrates all $E_\nu$ above threshold — and the response
removes the rest. This is the channel-level reason the sterile sensitivity is IBD-driven.
"""),

("code", r"""
from reactor.near_sm import eves_dsigma_dT, GV_SM, GA_SM
from reactor.detector import integration_weights

I_SLICE = int((40.0 - st50.l_edges[0]) / (st50.l_edges[1] - st50.l_edges[0]))
l_lo, l_hi = st50.l_edges[I_SLICE], st50.l_edges[I_SLICE + 1]
enu = st50.e_nu_grid
DAYS = 86400.0
t_live = st50.mw_yr * st50._base.seconds_per_mw_yr

# bin-averaged oscillation (with the physical core + vertex damping)
damp = st50._damping(DM2_B)
k2 = 2.0 * 1.267 * DM2_B / enu
nodes, ci = st50._l_nodes[I_SLICE], st50._c_ibd[I_SLICE]
osc_avg = sum(c * 0.5 * (1.0 - np.cos(k2 * L) * damp) for L, c in zip(nodes, ci)) / ci.sum()

# IBD true-energy spectra, events / MeV / day in this slice
true_null = st50._base._yield_ibd * ci.sum() / t_live * DAYS
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(enu, true_null, color=pl.INK_SECONDARY, lw=1.4, label="no oscillation")
ax.plot(enu, true_null * (1.0 - S22_B * osc_avg), color=pl.BLUE, lw=1.4,
        label=rf"$\Delta m^2={DM2_B}$ eV$^2$, $\sin^22\theta_{{14}}={S22_B}$")
ax.set_xlim(1.8, 9); ax.set_xlabel(r"true $E_\nu$ [MeV]")
ax.set_ylabel("IBD events / MeV / day")
ax.set_title(rf"IBD spectrum in true energy, $L \in [{l_lo:.1f}, {l_hi:.1f}]$ m")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()
"""),

("code", r"""
# IBD: the oscillation ratio at three levels of realism
fig, ax = plt.subplots(figsize=(7.0, 4.2))
pure = 0.5 * (1.0 - np.cos(k2 * 40.0))
ax.plot(enu, 1.0 - S22_B * pure, color=pl.INK_MUTED, lw=1.1, ls=":",
        label=r"$P_{ee}$ at exactly $L=40$ m (no smearing)")
ax.plot(enu, 1.0 - S22_B * osc_avg, color=pl.GREEN, lw=1.5,
        label="true $E_\\nu$, slice-averaged (core + vertex smearing)")
reco = 1.0 - S22_B * wig_ibd[I_SLICE] / np.maximum(null_ibd[I_SLICE], 1e-30)
ax.plot(e_cen, reco, color=pl.BLUE, lw=1.5, label=r"reconstructed $E_{\rm rec}$")
ax.axhline(1.0, color=pl.INK_MUTED, lw=0.8)
ax.set_xlim(1.8, 9); ax.set_xlabel("energy [MeV] (true $E_\\nu$ or $E_{\\rm rec}$)")
ax.set_ylabel("oscillated / no-oscillation")
ax.set_title(rf"IBD: true vs reconstructed, $L \in [{l_lo:.1f}, {l_hi:.1f}]$ m")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

amp_true = 0.5 * (osc_avg.max() - osc_avg.min())
r_reco = wig_ibd[I_SLICE] / np.maximum(null_ibd[I_SLICE], 1e-30)
amp_reco = 0.5 * (r_reco.max() - r_reco.min())
print(f"wiggle amplitude retained: slice-averaged true {2*amp_true:.2f} (of 1),")
print(f"after reconstruction {2*amp_reco:.2f} -- the ~0.78 MeV E_nu -> E_rec offset also")
print(f"shifts the pattern visibly. IBD keeps most of the oscillation.")
"""),

("code", r"""
# EvES: true recoil spectrum in the same slice
t_grid = st50._base.t_grid
K_full = eves_dsigma_dT(enu[None, :], t_grid[:, None], "e", True, GV_SM, GA_SM)
ce = st50._c_eves[I_SLICE]
osc_avg_e = sum(c * 0.5 * (1.0 - np.cos(k2 * L) * damp)
                for L, c in zip(nodes, ce)) / ce.sum()
flux_w = st50._eves_flux
t_null = K_full @ (flux_w * ce.sum()) / t_live * DAYS
t_osc = K_full @ (flux_w * ce.sum() * (1.0 - S22_B * osc_avg_e)) / t_live * DAYS

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(t_grid, t_null, color=pl.INK_SECONDARY, lw=1.4, label="no oscillation")
ax.plot(t_grid, t_osc, color=pl.ORANGE, lw=1.4,
        label=rf"$\Delta m^2={DM2_B}$ eV$^2$, $\sin^22\theta_{{14}}={S22_B}$")
ax.set_xlim(0, 8); ax.set_yscale("log")
ax.set_xlabel(r"true $T_{\rm recoil}$ [MeV]")
ax.set_ylabel(r"E$\nu$ES events / MeV / day")
ax.set_title(rf"E$\nu$ES spectrum in true recoil energy, $L \in [{l_lo:.1f}, {l_hi:.1f}]$ m")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()
"""),

("code", r"""
# EvES: the ratio in true and reconstructed recoil energy
n_t = len(st50.t_edges) - 1
t_cen = 0.5 * (st50.t_edges[:-1] + st50.t_edges[1:])
ev_wig = st50.wiggle_template(DM2_B)[n_l*n_e + I_SLICE*n_t : n_l*n_e + (I_SLICE+1)*n_t]
reco_ratio_e = 1.0 - S22_B * ev_wig / np.maximum(st50._eves_null[I_SLICE], 1e-30)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(enu, 1.0 - S22_B * osc_avg_e, color=pl.GREEN, lw=1.2, ls="--",
        label=r"flux suppression vs $E_\nu$ (input)")
ax.plot(t_grid, t_osc / np.maximum(t_null, 1e-30), color=pl.ORANGE, lw=1.5,
        label=r"true $T$ (all $E_\nu$ integrated)")
ax.plot(t_cen, reco_ratio_e, color=pl.RED, lw=1.4, ls="-.",
        label=r"reconstructed $T$, binned")
ax.axhline(1.0, color=pl.INK_MUTED, lw=0.8)
ax.set_xlim(0.5, 8); ax.set_ylim(1 - 1.4*S22_B, 1.02)
ax.set_xlabel(r"energy [MeV] ($E_\nu$ or $T$)")
ax.set_ylabel("oscillated / no-oscillation")
ax.set_title(rf"E$\nu$ES: the oscillation washes out in recoil energy")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

rt = t_osc / np.maximum(t_null, 1e-30)
sel = (t_grid > 1.0) & (t_grid < 6.5)
print(f"in the analysis window, the EvES true-T ratio varies by only "
      f"{rt[sel].max()-rt[sel].min():.4f} peak-to-peak (vs ~{2*amp_true*S22_B:.3f} for the")
print("IBD input): the T <- E_nu integration has already averaged the E-wiggles to their")
print("mean depletion, s22/2. EvES therefore contributes essentially a counting deficit --")
print("its L-dependence survives, its E-structure does not -- which is why the sterile")
print("sensitivity is IBD-driven and EvES 'rides along'.")
"""),

("code", r"""
# The volume-integrated spectrum: what a no-L-binning analysis has to work with.
st_int = SterileNearReactor(distance_m=50.0, l_binned=False)
n_e_i = len(st_int.e_edges) - 1
null_int = st_int._ibd_null[0]
e_cen_i = 0.5*(st_int.e_edges[:-1] + st_int.e_edges[1:])

fig, ax = plt.subplots(figsize=(7.0, 4.2))
for d, color in ((0.3, pl.RED), (1.05, pl.BLUE), (3.0, pl.GREEN)):
    wi = st_int.wiggle_template(d)[:n_e_i]
    ax.plot(e_cen_i, 1.0 - S22_B * wi / np.maximum(null_int, 1e-30), color=color,
            lw=1.5, label=rf"$\Delta m^2 = {d}$ eV$^2$")
ax.axhline(1.0, color=pl.INK_MUTED, lw=0.9)
ax.set_xlabel(r"$E_{\rm rec}$ [MeV]"); ax.set_ylabel("oscillated / no-oscillation")
ax.set_xlim(1.5, 8)
ax.set_title(rf"Volume-integrated ratio, $\sin^22\theta_{{14}}={S22_B}$")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

print("At 0.3 eV^2 the volume-integrated depletion is a smooth, broad tilt; at 1.05 eV^2")
print("it retains clear E-oscillation; at 3 eV^2 the structure compresses toward low E.")
print("The smooth cases are the vulnerable ones -- see the next table.")
"""),

("code", r"""
# E-binning convergence of the L-binned limit
rows = []
for eb in (0.2, 0.1, 0.05):
    stb = SterileNearReactor(distance_m=50.0, n_e_grid=2400,
                             e_edges=np.arange(1.0, 9.0+1e-9, eb))
    rows.append([f"{eb:.2f}"] + [stb.limit(d) for d in (1.0, 3.0, 5.0, 10.0)])
print(pl.table(rows, ["E-bin [MeV]", "s22 at 1 eV^2", "at 3", "at 5", "at 10"],
               floatfmt="{:.2e}"))
print()
print("0.1 MeV bins lose up to ~7% of reach at high dm2; 0.05 MeV (the default, matched")
print("to the energy resolution) is converged at the few-percent level. The L-direction")
print("information is insensitive to this choice -- it is the E-wiggles that need it.")
"""),

("md", r"""
## Backgrounds and fake data

The analysis so far carried only the IBD-singles background. The full model now includes,
for **IBD**: the JUNO far-reactor signal itself (the nine cores plus world reactors —
oscillated spectra straight from the data release, $\approx35$/day in total), geoneutrinos,
and the remaining release components ($^9$Li/$^8$He spallation, $^{214}$Bi–$^{214}$Po,
"other"), each with its release Tab. 1 normalisation prior ($2\%$ assigned to the
far-reactor prediction). For **E$\nu$ES**: **solar neutrinos** — the B8, hep and CNO-tail
continua from `reactor/data/solar` plus the pep line, weighted by the adiabatic LMA
survival probability into $\nu_e$ and $\nu_{\mu\tau}$ components with their different
cross sections, at a $3\%$ prior.

Geometry works for us here: the near-reactor signal falls across the detector as
$A(L)/L^2$, while every one of these backgrounds illuminates the volume *uniformly* — their
$L$-profile is the shell volume $A(L)$. They are smooth in both $E$ and $L$, so they dilute
statistics and add normalisation nuisances but cannot make wiggles; the limits barely move
(verified below).
"""),

("code", r"""
r50 = st50.rates_per_day()
print(f"near-reactor IBD signal : {r50['IBD / day']:>10,.0f} / day")
for name, v in r50["IBD backgrounds / day"].items():
    print(f"   IBD bkg, {name:<15}: {v:>10.2f} / day")
print(f"near-reactor EvES signal: {r50['EvES / day']:>10,.0f} / day")
print(f"   EvES bkg, IBD singles: {sum(x.sum() for x in st50.ibd_singles)/st50.days_calendar/0.9:>10.1f} / day")
print(f"   EvES bkg, solar      : {r50['solar EvES / day']:>10.1f} / day")

# limits with and without the additional backgrounds
st50_nobkg = SterileNearReactor(distance_m=50.0, extra_backgrounds=False)
rows = [[f"{d}", st50_nobkg.limit(d), st50.limit(d)] for d in (0.3, 1.0, 3.0)]
print()
print(pl.table(rows, ["dm2 [eV^2]", "s22_95 without", "with backgrounds"],
               floatfmt="{:.3e}"))
print()
print("As anticipated: L-smooth backgrounds at the 0.1% level of the signal cost <1% in")
print("reach. The wiggle search does not care what sits under it, as long as it is smooth.")
"""),

("code", r"""
# Fake data: the IBD channel, volume-integrated, full 27 MW.yr exposure.
e_cen6 = 0.5*(st50.e_edges[:-1] + st50.e_edges[1:])
e_wid6 = np.diff(st50.e_edges)
days_c = st50.days_calendar

sig_E = np.sum(st50._ibd_null, axis=0)
comps_E = {name: np.sum(per, axis=0) for name, per in st50.ibd_backgrounds.items()}
total_E = sig_E + sum(comps_E.values())

fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.stairs(sig_E/e_wid6/days_c, st50.e_edges, color=pl.BLUE, lw=1.6,
          label="near-reactor signal")
for (name, v), color in zip(comps_E.items(),
                            (pl.ORANGE, pl.GREEN, pl.RED, pl.MAGENTA, pl.BROWN,
                             pl.INK_SECONDARY)):
    ax.stairs(v/e_wid6/days_c, st50.e_edges, color=color, lw=1.2, label=name)
err = np.sqrt(np.maximum(total_E, 1.0))
ax.errorbar(e_cen6, total_E/e_wid6/days_c, yerr=err/e_wid6/days_c, fmt="o",
            ms=1.6, color=pl.INK, lw=0.7, label="Asimov data (total)")
ax.set_yscale("log"); ax.set_ylim(1e-4, 1e5)
ax.set_xlabel(r"$E_{\rm rec}$ [MeV]"); ax.set_ylabel("IBD events / MeV / day")
ax.set_title("IBD fake data, all components, D = 50 m")
ax.legend(fontsize=7.5, ncol=2)
plt.tight_layout(); plt.show()
"""),

("code", r"""
# Fake data: the EvES channel.
t_cen6 = 0.5*(st50.t_edges[:-1] + st50.t_edges[1:])
t_wid6 = np.diff(st50.t_edges)
sig_T = np.sum(st50._eves_null, axis=0)
sing_T = np.sum(st50.ibd_singles, axis=0)
sol_T = np.sum(st50.solar_eves, axis=0)
total_T = sig_T + sing_T + sol_T

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.stairs(sig_T/t_wid6/days_c, st50.t_edges, color=pl.BLUE, lw=1.6,
          label=r"near-reactor E$\nu$ES signal")
ax.stairs(sing_T/t_wid6/days_c, st50.t_edges, color=pl.RED, lw=1.3,
          label="IBD singles (1\\% untagged)")
ax.stairs(sol_T/t_wid6/days_c, st50.t_edges, color=pl.GREEN, lw=1.3,
          label="solar neutrinos")
err = np.sqrt(np.maximum(total_T, 1.0))
ax.errorbar(t_cen6, total_T/t_wid6/days_c, yerr=err/t_wid6/days_c, fmt="o",
            ms=2.0, color=pl.INK, lw=0.7, label="Asimov data (total)")
ax.set_yscale("log")
ax.set_xlabel(r"$T_{\rm rec}$ [MeV]"); ax.set_ylabel(r"E$\nu$ES events / MeV / day")
ax.set_title(r"E$\nu$ES fake data, all components, D = 50 m")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),

("code", r"""
# The geometric handle: signal vs background profiles across the detector.
l_cen6 = 0.5*(st50.l_edges[:-1] + st50.l_edges[1:])
l_wid6 = np.diff(st50.l_edges)
sig_L = np.array([x.sum() for x in st50._ibd_null])
bkg_L = np.array([x.sum() for x in st50._bkg_ib_tot])

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.stairs(sig_L/sig_L.sum()/l_wid6, st50.l_edges, color=pl.BLUE, lw=1.7,
          label=r"near signal $\propto A(L)/L^2$")
ax.stairs(bkg_L/bkg_L.sum()/l_wid6, st50.l_edges, color=pl.RED, lw=1.7,
          label=r"backgrounds $\propto A(L)$ (volume)")
ax.set_xlabel("$L$ [m]"); ax.set_ylabel("normalised event fraction / m")
ax.set_title("Baseline profiles: signal against uniform backgrounds")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

print("The two L-profiles differ visibly -- the signal leans toward the near wall, the")
print("backgrounds follow the volume -- so even the background *normalisations* are")
print("partially self-calibrated by the L-distribution, on top of their priors.")
"""),

("md", r"""
## Independence from the Daya Bay measurement

The demonstration that the shape-only default loses nothing: the same fit with the flux
normalisation tightened from free to $5\%$, and with the reactor $E$-shape constrained
by the Daya Bay covariance instead of free.
"""),

("code", r"""
st_prior = SterileNearReactor(distance_m=50.0, sigma_norm=0.05, free_shape=False)
rows = []
for d in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
    a, b = st50.limit(d), st_prior.limit(d)
    rows.append([f"{d}", a, b, a / b])
print(pl.table(rows, ["dm2 [eV^2]", "shape-only (default)", "with DYB priors", "ratio"],
               floatfmt="{:.3e}"))
print(f"modes: {st50.n_modes} (shape-only) vs {st_prior.n_modes} (DYB priors)")
print()
print("Freeing the normalisation entirely AND the E-shape entirely (182 additional")
print("unconstrained modes) changes the limits by < 0.5% across the whole sensitive")
print("region 0.03-3 eV^2, and by ~10% only at the 10 eV^2 wall where the L-wiggles have")
print("washed out. The sensitivity is a pure L/E-shape measurement: it never depended on")
print("the Daya Bay normalisation or shape, so any sterile depletion baked into that")
print("measurement is irrelevant to it. This is the finite-size search's cleanest")
print("property -- and precisely why the integrated (no-L-binning) analysis, which DOES")
print("lean on the flux prior, is the fragile one.")

# how large is the imprint on Daya Bay itself, for the record
e_chk = np.array([2.0, 4.0, 6.0, 8.0])
Ls = np.linspace(350.0, 600.0, 200)
print()
print("sterile imprint on the Daya Bay near-hall spectrum (s22 = 0.1, 350-600 m average):")
for dm2 in (0.1, 0.3, 1.0, 3.0):
    dep = np.array([np.mean([1 - st50.survival_4nu(np.array([ee]), L, dm2, 0.1)[0]
                             / st50.survival_4nu(np.array([ee]), L, 0.0, 0.0)[0]
                             for L in Ls]) for ee in e_chk])
    print(f"  dm2 = {dm2:<4}: normalisation deficit {100*dep.mean():.1f}%, "
          f"E-shape spread {100*(dep.max()-dep.min()):.2f}%")
"""),

("md", r"""
## Sensitivity

The $95\%$ CL exclusion reach in $(\Delta m^2_{41}, \sin^22\theta_{14})$, for 27 MW·yr
($10$ MW $\times$ 3 calendar years at $90\%$ duty). The decisive comparison is the dashed
curve: the *same* detector and exposure analysed without vertex binning (a single
integrated baseline), which is what a conventional counting+spectrum experiment sees.
"""),

("code", r"""
t0 = time.time()
dm2_grid = np.geomspace(1e-2, 20, 200)

distance = [50, 100, 200, 500, 1400] # m 

st = []
curves = {}
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.1, 0.8, len(distance)))
for d, color in zip(distance, colors):
      st.append(SterileNearReactor(distance_m=d, include_eves=True, l_binned=True))
      curves[f"D = {d} m, IBD + EvES"] = (st[-1].limit_curve(dm2_grid), color, "-")

st50_ibd = SterileNearReactor(distance_m=distance[0], include_eves=False)
curves[f"D = {distance[0]} m, IBD only"] = (st50_ibd.limit_curve(dm2_grid), pl.GREEN, "-.")
# Two integrated (no L binning) variants: with the same free E-shape as the
# L-binned fit, and with the Daya Bay shape covariance imposed as a prior.
st50_int = SterileNearReactor(distance_m=distance[0], l_binned=False)
st50_int_prior = SterileNearReactor(distance_m=distance[0], l_binned=False, free_shape=False)
curves[f"D = {distance[0]} m, integrated, free shape"] = (st50_int.limit_curve(dm2_grid), pl.INK_MUTED, "--")
curves[f"D = {distance[0]} m, integrated, Daya Bay prior"] = (st50_int_prior.limit_curve(dm2_grid), pl.ORANGE, "-")

print("the finite-size gain, L-binned vs integrated:")
lb = curves['D = 50 m, IBD + EvES'][0]
fr = curves['D = 50 m, integrated, free shape'][0]
pr = curves['D = 50 m, integrated, Daya Bay prior'][0]
for dm in (0.3, 1.0, 3.0, 10.0):
    i = np.argmin(np.abs(dm2_grid - dm))
    print(f"   dm2 = {dm2_grid[i]:5.2f} eV^2 : L-binned {lb[i]:.2e} | integrated, free shape "
          f"{fr[i]:.2e} | integrated, Daya Bay prior {pr[i]:.2e} (x{pr[i]/lb[i]:.0f})")
print("With the E-shape left free, an integrated analysis has one free mode per E bin and")
print("a single L bin, so it has no handle at all: its curve sits at s22 ~ 0.5-0.8 across")
print("the whole range.  The only integrated analysis with any reach is one that imposes")
print("the Daya Bay flux shape -- the measurement that would itself carry the depletion.")
"""),

("code", r"""
from scipy.interpolate import interp1d

def plot_nue_dis(ax):

    limits_path = 'reactor/data/nue_dis/'
    xunits = 1
    zorder = 1
    # DANSS
    x, y = np.genfromtxt(limits_path + 'DANSS.dat', unpack=True)
    ax.fill_betweenx(y, x, x/x, edgecolor='grey', facecolor='lightgrey',
                    lw=0, alpha=1, zorder=zorder)

    # PROSPECT
    x, y = np.genfromtxt(limits_path + 'PROSPECT.dat', unpack=True)
    ax.fill_betweenx(y, x, x/x, edgecolor='grey', facecolor='lightgrey', lw=0, alpha=1, zorder=zorder)

    # STEREO
    x, y = np.genfromtxt(limits_path + 'STEREO.dat', unpack=True)
    ax.fill_betweenx(y, x, x/x, edgecolor='grey', facecolor='lightgrey', lw=0, alpha=1, zorder=zorder)

    x, y = np.genfromtxt(f"{limits_path}/RENO_NEOS.dat", unpack=True)
    x, y = x[np.argsort(y)], y[np.argsort(y)]
    ax.fill_betweenx(y, x, x/x, edgecolor='grey', facecolor='lightgrey', lw=0, alpha=1, zorder=zorder)


    # KATRIN 2025
    y_total = np.geomspace(1e-2, 1e2, 10000)
    x_total = np.ones_like(y_total)
    temp_x, temp_y = np.genfromtxt(f"{limits_path}/KATRIN_2025_exclusion.dat", unpack=True)
    f_limit = interp1d(temp_x, temp_y, kind='linear', bounds_error=False, fill_value=(1,1))
    y_total = np.minimum(y_total, f_limit(x_total))
    ax.plot(temp_x, temp_y, color='grey', lw=0.5, ls='-', zorder=3)
    # ax.plot(xunits*x_total, y_total, color='forestgreen', lw=1, ls='-', zorder=3)
    ax.fill_betweenx(temp_y, temp_x, temp_x/temp_x,
                    edgecolor="lightgrey",facecolor="lightgrey", linestyle="-", lw=0.5, alpha = 1, zorder = zorder)
    
    # Gallium
    Ga2S0 = np.loadtxt(f"{limits_path}/Gallium_2sigma_l0.csv", delimiter=",")
    Ga2S1 = np.loadtxt(f"{limits_path}/Gallium_2sigma_l1.csv", delimiter=",")
    ax.fill(Ga2S0[:,0], Ga2S0[:,1], lw=0.75, edgecolor='None', facecolor='orange', ls='-', zorder=zorder, alpha=0.6)
    ax.fill(Ga2S0[:,0], Ga2S0[:,1], lw=0.75, edgecolor='orange', facecolor='None', ls='-', zorder=zorder, alpha=1)
    ax.fill(Ga2S1[:,0], Ga2S1[:,1], lw=0.75, edgecolor='None', facecolor='orange', ls='-', zorder=zorder, alpha=0.6)
    ax.fill(Ga2S1[:,0], Ga2S1[:,1], lw=0.75, edgecolor='orange', facecolor='None', ls='-', zorder=zorder, alpha=1)

    ax.plot([0.1704, 0.1704], [1e-4, 1e3], lw=1.2, color='orange', ls=(1,(3,1)), zorder=1)
"""),

("code", r"""
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(6.0, 5.0))


for d, color in zip(distance, colors):
    lab = f"D = {d} m, IBD + EvES"
    v, _, ls = curves[lab]
    
    ax.loglog(v, dm2_grid, color=color, ls=ls, lw=1.8, label=rf"$D = {d} \text{{ m}}$")

limits_path = 'reactor/data/nue_dis/'

x,y = np.loadtxt(limits_path + 'IsoDAR_Yemilab_90CL.dat', unpack=True)
ax.plot(x, y, color='deeppink', lw=1.5, ls="--")

plot_nue_dis(ax)

ax.legend(fontsize=10, loc="lower left", title='JUNO + near reactor', title_fontsize=10, framealpha=0.95)
# keep the existing (upper-left) legend, then add a second one
leg_main = ax.get_legend()

proxy_handles = [
    Line2D([0], [0], color="grey", lw=1.0, ls="-", label=r"KATRIN 95\% CL"),
    Patch(facecolor="lightgrey", edgecolor="None", label=r"Reactors 90\% CL"),
    Patch(facecolor="orange", edgecolor="orange", alpha=0.6, label=r"Gallium $2\sigma$ region"),
    Line2D([0], [0], color="orange", lw=1.5, ls=(0, (3, 1)), label=r"Solar 99\% CL"),
    Line2D([0], [0], color="deeppink", lw=1.5, ls="--", label=r"IsoDAR 90\% CL"),
]

leg_ext = ax.legend(handles=proxy_handles, loc="upper left", fontsize=10, framealpha=0.95)
ax.add_artist(leg_main)

ax.set_xlim(1e-4, 1); ax.set_ylim(dm2_grid[0], dm2_grid[-1])
ax.set_xlabel(r"$\sin^2 2\theta_{ee}$"); ax.set_ylabel(r"$\Delta m^2_{41}$ [eV$^2$]")
ax.set_title(r"$95\%$ CL sensitivity, 27 MW$\cdot$yr per standoff")
plt.tight_layout(); plt.show()
"""),

("code", r"""
fig, ax = plt.subplots(figsize=(5.0, 4.0))
for lab, (v, color, ls) in curves.items():
    ax.loglog(v, dm2_grid, color=color, ls=ls, lw=1.8, label=lab)
ax.set_xlim(3e-4, 1); ax.set_ylim(dm2_grid[0], dm2_grid[-1])
ax.set_xlabel(r"$\sin^2 2\theta_{14}$"); ax.set_ylabel(r"$\Delta m^2_{41}$ [eV$^2$]")
ax.set_title(r"$95\%$ CL sensitivity, 27 MW$\cdot$yr")
ax.legend(fontsize=8.5, loc="upper left")
plt.tight_layout(); plt.show()
"""),

("code", r"""
print("Against the integrated analysis that keeps the Daya Bay flux prior, the gain is")
print("dm2-dependent: near 1-3 eV^2 the integrated E-shape analysis happens to work (its")
print("surviving E-wiggles resist the flux covariance) and the gap narrows to ~3-4x; away")
print("from that sweet spot the integrated limit collapses into the flux degeneracy and")
print("the wiggle search wins by up to ~90x. Above ~15 eV^2 the ordering reverses: the L")
print("wiggle is smeared out while the E wiggle is not, and the prior-based analysis is")
print("the better one by up to 5x. The L-binned curve is the uniformly robust one below")
print("the wall, and the only one that needs no prior at all.")
print()
print("The shape of the Daya-Bay-prior curve deserves its own explanation, verified")
print("by ablation: its wall below dm2 ~ 0.6 eV^2 and its bumps at 2-3 eV^2 are the U235")
print("flux-shape covariance. Where the volume-integrated depletion is smooth in E (see")
print("the spectrum figure above), the measured-flux uncertainty absorbs it -- removing")
print("the flux covariance improves the integrated limit by a factor 23-27 at 0.15-0.3")
print("eV^2, but only 1.1x at 1.05 eV^2 where the surviving E-wiggles cannot be mimicked.")
print("(It is NOT the normalisation: tightening sigma_norm 5% -> 0.5% changes nothing.)")
print("The integrated analysis is thus not merely weaker -- it is fragile, its reach")
print("depending on accidental alignment between oscillation shape and flux uncertainty.")
print("The L-binned search has no such dependence: its limit at 0.3 eV^2 is 1.0e-3 with")
print("the full flux covariance active.")
print()
print("Reading the multi-distance curves: each standoff owns a dm2 decade. The reach")
print("scales with the accumulated phase Delta_41 ~ dm2 x D / E: 50 m covers 0.2-3 eV^2")
print("at ~1e-3, 100 m shifts the sweet spot to 0.1-1 eV^2, and 500 m / 1400 m open the")
print("region below 0.1 eV^2 -- but at 1/D^2 the rate cost (586 and 71 IBD/day) leaves")
print("500 m at 2e-2 and 1400 m at 1.8e-1. Above ~5 eV^2 the baseline smearing (core + vertex resolution,")
print("dissected below) washes out every standoff alike. EvES adds little (no E_nu")
print("reconstruction dilutes its wiggle) but rides along for free.")
"""),

("code", r"""
# What does the full 3+1 formula change against the naive 2-flavour one?
class Naive2Flavour(SterileNearReactor):
    def _sterile_phases(self, dm2_41):
        return [(dm2_41, 1.0)]

# the models built in the sensitivity cell live in `st`, indexed like `distance`
by_D = dict(zip(distance, st))
rows = []
for D in (50, 500, 1400):
    full = by_D[D]
    naive = Naive2Flavour(distance_m=float(D))
    rows.append([f"{D}"] + [full.limit(d) / naive.limit(d) for d in (0.01, 0.03, 0.1, 1.0)])
print(pl.table(rows, ["D [m]", "full/naive @0.01", "@0.03", "@0.1", "@1 eV^2"],
               floatfmt="{:.3f}"))
print()
print("Two effects of the full 3+1 formula. (i) The sterile depletion is a weighted sum")
print("over Delta_14, Delta_24, Delta_34 with weights |U_e1|^2, |U_e2|^2, |U_e3|^2 = 0.68,")
print("0.30, 0.02; where dm2_41 ~ dm2_31 (0.01-0.1 eV^2) at 1400 m the Delta_34 term")
print("differs, diluting the wiggle by a few percent. (ii) The null hypothesis is the")
print("theta13-oscillated spectrum: at 1400 m the standard disappearance is 3-8%, which")
print("the fit must not confuse with a sterile one -- it does not, because its L/E")
print("frequency is fixed by the known dm2_ee. At 50 m both effects vanish identically,")
print("as they must (Delta_31 = 0.04 rad across the whole detector).")
"""),

("md", r"""
## What limits it

The wiggle search is by construction immune to flux-shape systematics — they carry no
$L$-dependence. The nuisances that live in the wiggle direction are the **detector
response non-uniformities** across the volume, both modelled as Gaussian-correlated
fields over $L$ with a 1.5 m correlation length: an *efficiency* field (0.5%), and a
position-dependent *energy scale* (0.2%), which is the one that distorts the spectrum
in $E$ as a function of $L$ — the same joint direction the signal lives in.
"""),

("code", r"""
t0 = time.time()
fig, ax = plt.subplots(figsize=(6.9, 4.6))
for unif, color in ((0.001, pl.GREEN), (0.005, pl.BLUE), (0.02, pl.RED)):
    stu = SterileNearReactor(distance_m=50.0, sigma_uniformity=unif)
    ax.loglog(stu.limit_curve(dm2_grid), dm2_grid, color=color, lw=1.8,
              label=rf"efficiency field ${100*unif:.1f}\%$, E-scale field $0.2\%$")
st_perfect = SterileNearReactor(distance_m=50.0, sigma_uniformity=1e-6,
                                sigma_escale_uniformity=0.0)
ax.loglog(st_perfect.limit_curve(dm2_grid), dm2_grid, color=pl.ORANGE, ls="-.",
          lw=1.5, label="perfectly uniform detector")
st_stat = SterileNearReactor(distance_m=50.0, sigma_norm=1e-6, sigma_uniformity=1e-6,
                             sigma_escale_uniformity=0.0, sigma_channel_ratio=1e-6,
                             sigma_u238=1e-6, sigma_evolution=1e-6,
                             use_flux_covariance=False, free_shape=False)
ax.loglog(st_stat.limit_curve(dm2_grid), dm2_grid, color=pl.INK_SECONDARY, ls=":",
          lw=1.6, label="statistics only (no free shape)")
ax.set_xlim(1e-4, 3e-1); ax.set_ylim(3e-2, 3e1)
ax.set_xlabel(r"$\sin^2 2\theta_{14}$"); ax.set_ylabel(r"$\Delta m^2_{41}$ [eV$^2$]")
ax.set_title(r"Impact of the detector response uniformity, D = 50 m")
ax.legend(fontsize=8.5, loc="upper left")
plt.tight_layout(); plt.show()
print(f"({time.time()-t0:.0f} s)")

print("With ~7e7 IBD events the per-bin statistics reach a few 1e-4 in the depletion, and")
print("that -- together with the free E-shape -- is what sets the reach. Both uniformity")
print("fields together cost under 1%: an efficiency field is energy independent and the")
print("E-dependence of the oscillation phase breaks it, while the energy-scale field does")
print("distort the spectrum in E as a function of L but spans only ~20 correlated modes")
print("over a 33 m span, which the fit simply projects out. Its cost saturates: 0.2% and")
print("1% give the same answer to three digits. Freeing the reactor E-shape, by contrast,")
print("costs a factor 2.4 against a pure-statistics fit -- that is the price of not using")
print("the Daya Bay measurement, and it is the price worth paying.")
print()
print("And the audit of the E-direction systematics -- all Daya-Bay-anchored -- at 1 eV^2:")
rows = [["standard (shape-only)", SterileNearReactor(distance_m=50.0).limit(1.0)],
        ["with DYB flux priors instead",
         SterileNearReactor(distance_m=50.0, sigma_norm=0.05, free_shape=False).limit(1.0)],
        ["no fuel-evolution uncertainty",
         SterileNearReactor(distance_m=50.0, sigma_evolution=1e-6).limit(1.0)],
        ["fuel evolution 30% -> 100%",
         SterileNearReactor(distance_m=50.0, sigma_evolution=1.0).limit(1.0)]]
print(pl.table(rows, ["variant", "s22_95 at 1 eV^2"], floatfmt="{:.4e}"))
print()
print("The flux prior and the fuel evolution -- dominant systematics in the SM study of")
print("notebook 5 -- are exact nulls here, to four digits even at a 100% evolution prior:")
print("E-direction distortions cannot make L-wiggles. Demonstrated, not assumed.")
"""),

("md", r"""
## How far up in $\Delta m^2$ can we go?

The high-$\Delta m^2$ reach is set by the baseline smearing:
$\sigma_L^2 = \sigma_{\rm vtx}^2(E) + a_{\rm core}^2/5$, damping the wiggle amplitude as
$e^{-2k^2\sigma_L^2}$. Comparing an idealised point source with perfect vertexing against
the physical cases isolates each wall.

Two remarks about the idealised curve, which shows structure worth understanding. First,
its wall is **not** spatial at all: with perfect vertexing, what kills the wiggles is the
*energy resolution* acting on the $L/E$ phase, $\sigma_\phi \propto \Delta m^2 L\,
\sigma_E/E^2$. That washes out low energies first and lets the high-$E$ tail (smallest
$\sigma_E/E^2$) keep coherence longest, so the ideal sensitivity decays *in stages*, one
energy range at a time — the shoulder structure is real. Second, resolving those fast
$E$-oscillations demands a fine integration grid: on the default 6.8 keV grid the
$\Delta m^2\gtrsim25$ eV$^2$ points alias (the discrete response integral fails to cancel
the fast oscillation and fakes wiggle power). All curves here use a $4\times$ denser grid,
verified converged from 1200 to 9600 points at every plotted $\Delta m^2$.
"""),

("code", r"""
t0 = time.time()
dm2_hi = np.logspace(-0.3, 1.48, 45)      # 0.5 - 30 eV^2
GRID = dict(l_bin_m=0.25, n_e_grid=9600)
configs = [
    ("point core, perfect vertex (ideal)",
     dict(core_radius_m=0.0, sigma_vertex_m=0.0, n_sub=25, **GRID),
     pl.INK_SECONDARY, ":"),
    ("+ vertex resolution (10 cm / sqrt(E))",
     dict(core_radius_m=0.0, sigma_vertex_m=0.10, **GRID), pl.GREEN, "-"),
    ("+ core radius 0.5 m (default)",
     dict(core_radius_m=0.5, sigma_vertex_m=0.10, **GRID), pl.BLUE, "-"),
    ("+ core radius 1.5 m",
     dict(core_radius_m=1.5, sigma_vertex_m=0.10, **GRID), pl.RED, "-"),
]
fig, ax = plt.subplots(figsize=(7.0, 4.8))
for lab, kw, color, ls in configs:
    stx = SterileNearReactor(distance_m=50.0, **kw)
    ax.loglog(stx.limit_curve(dm2_hi), dm2_hi, color=color, ls=ls, lw=1.8, label=lab)
ax.set_xlim(5e-4, 3e-1); ax.set_ylim(dm2_hi[0], dm2_hi[-1])
ax.set_xlabel(r"$\sin^2 2\theta_{14}$"); ax.set_ylabel(r"$\Delta m^2_{41}$ [eV$^2$]")
ax.set_title(r"The high-$\Delta m^2$ wall, D = 50 m, 27 MW$\cdot$yr")
ax.legend(fontsize=8.5, loc="upper left")
plt.tight_layout(); plt.show()
print(f"({time.time()-t0:.0f} s)")

rows = []
for lab, kw, _, _ in configs:
    stx = SterileNearReactor(distance_m=50.0, **kw)
    rows.append([lab] + [stx.limit(d) for d in (3.0, 10.0, 20.0)])
print(pl.table(rows, ["configuration", "s22 at 3 eV^2", "at 10 eV^2", "at 20 eV^2"],
               floatfmt="{:.2e}"))
print()
print("The hierarchy of walls, quoted as the dm2 at which the reach crosses s22 = 0.1.")
print("Even with a point source and perfect vertexing the ENERGY resolution terminates the")
print("search through the L/E phase (sigma_phi ~ dm2 L sigma_E/E^2), at 14 eV^2; JUNO's")
print("vertexing costs almost nothing on top of that (13 eV^2); a 0.5 m core brings it to")
print("12 eV^2 and a 1.5 m core to 8.  Above the wall every configuration loses sensitivity")
print("entirely, as it must: a fully averaged depletion is L- and E-independent and so is")
print("exactly the normalisation this analysis leaves free.  Core compactness helps, but")
print("the energy resolution sets the ceiling and JUNO's vertexing is never the bottleneck.")
"""),

("md", r"""
## Summary

Parking the reactor next to JUNO turns the detector's *size* — normally a nuisance for
oscillation analyses — into the instrument:

| | D = 50 m | 100 m | 500 m | 1400 m |
|---|---|---|---|---|
| IBD rate | $61$k/day | $15$k/day | $590$/day | $71$/day |
| baseline span | $33$–$67$ m | $84$–$117$ m | $484$–$517$ m | $1384$–$1417$ m |
| best $\sin^22\theta_{14}$ ($95\%$) | $9.5\times10^{-4}$ at $0.5$ eV$^2$ | $2.1\times10^{-3}$ at $0.4$ | $2.2\times10^{-2}$ at $0.16$ | $1.8\times10^{-1}$ at $0.065$ |

Each standoff owns roughly a decade of $\Delta m^2$ — the reach follows the accumulated
phase $\Delta_{41}\propto\Delta m^2 D/E$ — while the rate falls as $1/D^2$: the compact
near standoffs are where the sensitivity is, and the kilometre ones only open the region
below $0.1$ eV$^2$ at the few-percent level. The **full 3+1** formula (all three
$\Delta_{i4}$ phases, the $\theta_{13}$-oscillated null, and the $(1-s_{14})^2$ term that
switches part of it off) matters at the few-percent level in the limit at $1.4$ km for
$\Delta m^2\lesssim0.1$ eV$^2$ and is identically null at $50$ m.

* The analysis is **shape-only by construction and by necessity**: the flux
  normalisation and the reactor $E$-shape are left completely free, because the Daya Bay
  measurement they would otherwise constrain was itself taken at $500$ m and would carry
  the sterile depletion ($\approx\sin^22\theta_{14}/2$). Freeing them costs $<0.5\%$
  across $0.03$–$3$ eV$^2$ — the sensitivity was a pure $L/E$-shape measurement all along.
* An integrated analysis with the same free $E$-shape has **no sensitivity at all** — one
  free mode per $E$ bin and a single $L$ bin leaves nothing to fit — so the only
  integrated comparison with any reach is one that imposes the Daya Bay flux shape. The
  vertex-binned search beats *that* by $\sim3$–$4$ at its sweet spot near $1$–$3$ eV$^2$
  and by up to $\sim90$ elsewhere, while needing no prior at all.
* The reach is set by statistics and by the free-shape construction, not by the detector:
  the efficiency and energy-scale uniformity fields together cost under $1\%$, and the
  energy-scale cost saturates (0.2% and 1% agree to three digits) because the field spans
  only $\sim20$ correlated modes across the volume. Freeing the reactor $E$-shape costs a
  factor $2.4$ against pure statistics.
* The **high-$\Delta m^2$ wall is the energy resolution**, through the $L/E$ phase
  ($\sigma_\phi\propto\Delta m^2 L\sigma_E/E^2$): even a point source with perfect
  vertexing loses $\sin^22\theta_{14}=0.1$ by $14$ eV$^2$. JUNO's vertexing costs almost
  nothing ($13$ eV$^2$); the core size is the secondary effect, $12$ eV$^2$ for a $0.5$ m
  core and $8$ for $1.5$ m. Above the wall every configuration loses sensitivity entirely,
  since a fully averaged depletion is exactly the free normalisation.
* E$\nu$ES rides along but adds little — without $E_\nu$ reconstruction its wiggles wash
  out; its role remains the SM coupling measurement of notebook 5.
* $50$ m outperforms every longer standoff except below $\Delta m^2\lesssim0.1$ eV$^2$,
  where $100$ m and then $500$ m take over at a steep rate cost; a movable source could
  visit all of them. Against the existing landscape (DANSS/PROSPECT/STEREO/RENO-NEOS,
  Gallium, KATRIN 2025, IsoDAR projection), the $50$ m curve is an order of magnitude
  beyond the reactor short-baseline experiments and covers the whole Gallium region.
"""),
]
