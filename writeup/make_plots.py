"""Generate every manuscript figure as a 300-dpi PDF into writeup/plots/.

Run from the repository root:  python writeup/make_plots.py
Each figure is a single axes; every number underlying them comes from the
reactor.* modules exactly as used in notebooks 5-7.
"""
import sys, time, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = pathlib.Path(__file__).resolve().parent / "plots"
OUT.mkdir(exist_ok=True)

from reactor import plotting as pl
from reactor.near_sm import FixedNearReactor, atomic_stepping, GV_SM, GA_SM, SW2_SM
from reactor.sterile import SterileNearReactor
from reactor.alps import ALPSearchJUNO

pl.use_style()
DAY = 86400.0
SAVE = dict(dpi=300, bbox_inches="tight")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", **SAVE)
    plt.close(fig)
    print("wrote", name)


t_all = time.time()
# =============================================================== notebook 5
STD = dict(burnup=0.5)
r = FixedNearReactor(**STD)
widths = np.diff(r.recoil_edges)
cen = 0.5 * (r.recoil_edges[:-1] + r.recoil_edges[1:])

# --- 5a: fake data, EvES channel with all components
YEAR_CAL = 365.25 * DAY                      # one calendar year
YEAR_LIVE = YEAR_CAL * r.duty_cycle          # ... at 90% reactor duty
sig = r.eves_spectrum_T() * YEAR_LIVE
bkg = r.ibd_singles() * YEAR_LIVE
sol = r.solar_eves_binned * YEAR_CAL         # non-reactor: accrues on calendar time
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.stairs(sig / widths, r.recoil_edges, color=pl.BLUE, lw=1.6, label=r"reactor E$\nu$ES signal")
ax.stairs(bkg / widths, r.recoil_edges, color=pl.RED, lw=1.4,
          label=r"IBD singles (1\% untagged)")
ax.stairs(sol / widths, r.recoil_edges, color=pl.GREEN, lw=1.4, label="solar neutrinos")
sing = r.singles_binned * YEAR_CAL
ax.stairs(sing / widths, r.recoil_edges, color=pl.MAGENTA, lw=1.5,
          label="detector singles")
c13r = r.c13_nc_binned * YEAR_LIVE
c13s = r.c13_solar_binned * YEAR_CAL
ax.stairs(c13r / widths, r.recoil_edges, color=pl.BROWN, lw=1.6,
          label=r"$^{13}$C NC line (reactor)")
ax.stairs(c13s / widths, r.recoil_edges, color=pl.BROWN, lw=1.2, ls="--",
          label=r"$^{13}$C NC + CC (solar)")
tot = sig + bkg + sol + sing + c13r + c13s
ax.errorbar(cen, tot / widths, yerr=np.sqrt(tot) / widths, fmt="o", ms=1.8, color=pl.INK,
            lw=0.7, label="Asimov data (total)")
ax.set_yscale("log"); ax.set_ylim(1e2, 1e9)
ax.set_xlabel(r"$T_{\rm rec}$ [MeV]"); ax.set_ylabel(r"events / MeV / year")
ax.set_title(r"E$\nu$ES at JUNO from 10 MW at 50 m, one year at 90\% duty")
ax.legend(fontsize=10)
save(fig, "nb5_eves_fake_data")

# --- 5b: sigma(sw2) vs exposure, EvES-only vs joint
exposures = np.array([1, 3, 10, 30, 100, 300], dtype=float)
ev_only = [r.sigma_sw2(m) for m in exposures]
joint = [r.sigma_sw2(m, joint=True) for m in exposures]
best = FixedNearReactor(**STD, sigma_ibd_xsec=0.001, sigma_channel_ratio=0.002)
joint_best = [best.sigma_sw2(m, joint=True) for m in exposures]
stat = FixedNearReactor(**STD, sigma_norm=1e-6, sigma_u238=1e-6, sigma_ibd_singles=1e-6,
                        sigma_solar=1e-9, sigma_scale=1e-9, sigma_bias=1e-9, sigma_res=1e-9,
                        sigma_evolution=1e-9, use_flux_covariance=False)
stat_only = [stat.sigma_sw2(m) for m in exposures]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.loglog(exposures, ev_only, color=pl.RED, lw=1.8, label=r"E$\nu$ES only")
ax.loglog(exposures, joint, color=pl.BLUE, lw=1.8, label=r"joint IBD + E$\nu$ES")
ax.loglog(exposures, joint_best, color=pl.BLUE, ls="--", lw=1.4,
          label=r"joint, $\sigma_{\rm IBD}=0.1\%$, ratio $0.2\%$")
ax.loglog(exposures, stat_only, ":", color=pl.INK_SECONDARY, label="statistics only")
ax.set_xlabel(r"delivered exposure [MW$\cdot$yr]"); ax.set_ylabel(r"$\sigma(\sin^2\theta_W)$")
ax.set_title(r"Weak mixing angle from E$\nu$ES, 10 MW at 50 m")
ax.legend(fontsize=10)
save(fig, "nb5_sigma_sw2_vs_exposure")

# --- 5c: (gV, gA) ellipses -- zoom, with the CHARM II / PDG ellipses for scale
from reactor.gvga_limits import draw_gvga_limits
fig, ax = plt.subplots(figsize=(5.6, 5.2))
for mwyr, color in ((30, pl.ORANGE), (100, pl.GREEN)):
    C0 = r.fisher_gv_ga(float(mwyr)); C1 = r.fisher_gv_ga(float(mwyr), joint=True)
    pl.error_ellipse(ax, (GV_SM, GA_SM), C0, n_sigma=1.0, color=color, ls="--", lw=1.3,
                     label=rf"this work, {mwyr} MW$\cdot$yr, E$\nu$ES only")
    pl.error_ellipse(ax, (GV_SM, GA_SM), C1, n_sigma=1.0, color=color, lw=1.9,
                     label=rf"this work, {mwyr} MW$\cdot$yr, joint")
ax.set_xlim(GV_SM - 0.06, GV_SM + 0.06); ax.set_ylim(GA_SM - 0.05, GA_SM + 0.05)
draw_gvga_limits(ax, pl, zoom=True)
ax.set_xlabel(r"$g_V$"); ax.set_ylabel(r"$g_A$")
ax.set_title(r"$1\sigma$ regions in the $(g_V, g_A)$ plane (zoom)")
ax.legend(fontsize=10, loc="lower left")
save(fig, "nb5_gv_ga_ellipses")

# --- 5c': the wide view: existing nu_e / nubar_e bands and nu_mu ellipses
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.set_xlim(-0.6, 0.5); ax.set_ylim(-1.0, 0.1)
draw_gvga_limits(ax, pl, zoom=False)
C1 = r.fisher_gv_ga(30.0, joint=True)
pl.error_ellipse(ax, (GV_SM, GA_SM), C1, n_sigma=1.0, color=pl.RED, lw=2.0,
                 label=r"this work, 30 MW$\cdot$yr joint")
ax.set_xlabel(r"$g_V$"); ax.set_ylabel(r"$g_A$")
ax.set_title(r"Existing constraints in the $(g_V, g_A)$ plane")
ax.legend(fontsize=10, loc="upper left")
save(fig, "nb5_gv_ga_landscape")

# --- 5e: cross sections per LAB "molecule" (CH_1.63 unit: 1 C, 1.63 free protons,
#     7.63 electrons), including the nubar_e -> nubar_e e+e- tridents from NEPTUNE
from reactor.near_sm import eves_dsigma_dT
from reactor.cross_sections import vogel_beacom
from reactor.tridents import H_PER_C, E_PER_C     # LAB stoichiometry constants
enu = np.linspace(1.9, 10.0, 300)
tgrid = np.linspace(0.0, 10.0, 2000)
def sig_tot(flavor, nubar, tmin=0.0):
    out = []
    for e in enu:
        ds = eves_dsigma_dT(e, tgrid, flavor, nubar)
        out.append(np.trapezoid(np.where(tgrid >= tmin, ds, 0.0), tgrid))
    return np.array(out) * E_PER_C          # per molecule
fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.semilogy(enu, sig_tot("e", True), color=pl.BLUE, lw=1.8, label=r"$\bar\nu_e\,e$ (CC+NC)")
ax.semilogy(enu, sig_tot("e", False), color=pl.BLUE, ls="--", lw=1.4, label=r"$\nu_e\,e$ (CC+NC)")
ax.semilogy(enu, sig_tot("mu", True), color=pl.ORANGE, lw=1.8, label=r"$\bar\nu_{\mu,\tau}\,e$ (NC)")
ax.semilogy(enu, sig_tot("mu", False), color=pl.ORANGE, ls="--", lw=1.4, label=r"$\nu_{\mu,\tau}\,e$ (NC)")
ax.semilogy(enu, sig_tot("e", True, tmin=1.0), color=pl.BLUE, ls=":", lw=1.6,
            label=r"$\bar\nu_e\,e$, $T>1$ MeV (analysis window)")
ax.semilogy(enu, vogel_beacom(enu, order=1) * H_PER_C, color=pl.RED, lw=1.8,
            label=r"IBD $\bar\nu_e p$")
ax.set_ylim(1e-44, 2e-41)
ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel(r"$\sigma$ per CH$_{1.63}$ unit [cm$^2$]")
ax.set_title(r"Cross sections per carbon of LAB (CH$_{1.63}$)")
ax.legend(fontsize=10, loc="lower right", ncol=2)
save(fig, "nb5_cross_sections")

# --- 5f: differential recoil shapes at a fixed E_nu, the coupling structure
fig, ax = plt.subplots(figsize=(6.6, 4.2))
E0 = 4.0
tt = np.linspace(0.0, 2 * E0**2 / (0.511 + 2 * E0) * 0.999, 400)
ax.plot(tt, eves_dsigma_dT(E0, tt, "e", True) * 1e42, color=pl.BLUE, lw=1.8, label=r"$\bar\nu_e$")
ax.plot(tt, eves_dsigma_dT(E0, tt, "e", False) * 1e42, color=pl.BLUE, ls="--", lw=1.4, label=r"$\nu_e$")
ax.plot(tt, eves_dsigma_dT(E0, tt, "mu", True) * 1e42, color=pl.ORANGE, lw=1.8, label=r"$\bar\nu_{\mu,\tau}$")
ax.plot(tt, eves_dsigma_dT(E0, tt, "mu", False) * 1e42, color=pl.ORANGE, ls="--", lw=1.4, label=r"$\nu_{\mu,\tau}$")
ax.set_xlabel(r"$T$ [MeV]"); ax.set_ylabel(r"$d\sigma/dT$ [$10^{-42}$ cm$^2$/MeV]")
ax.set_title(rf"Recoil spectra at $E_\nu = {E0:.0f}$ MeV")
ax.legend(fontsize=10)
save(fig, "nb5_dsigma_dT")

# --- 5g: detector singles by component, showing the 208Tl external-gamma wall
fig, ax = plt.subplots(figsize=(6.8, 4.6))
w5 = np.diff(r.recoil_edges)
ax.stairs(r.eves_spectrum_T() * DAY / w5, r.recoil_edges, color=pl.BLUE, lw=2.0,
          label=r"E$\nu$ES signal")
for name, color, ls in (("external gammas", pl.RED, "-"),
                        ("internal 214Bi", pl.ORANGE, "--"),
                        ("internal 208Tl", pl.MAGENTA, "-."),
                        ("cosmogenic 11C", pl.GREEN, "--"),
                        ("cosmogenic 11Be", pl.INK_SECONDARY, ":")):
    if name in r.singles_components:
        ax.stairs(r.singles_components[name] * DAY / w5, r.recoil_edges,
                  color=color, lw=1.5, ls=ls, label=name)
ax.stairs(r.c13_nc_binned * DAY / w5, r.recoil_edges, color=pl.BROWN, lw=1.8,
          label=r"$^{13}$C NC line (reactor)")
ax.stairs(r.c13_solar_binned * DAY / w5, r.recoil_edges, color=pl.BROWN, lw=1.2,
          ls="--", label=r"$^{13}$C (solar)")
ax.axvline(2.615, color=pl.INK_MUTED, lw=1.0, ls=":")
ax.annotate(r"$^{208}$Tl 2.615 MeV" "\n" r"external $\gamma$ wall", xy=(2.70, 6e4),
            fontsize=9, color=pl.INK_SECONDARY)
ax.set_yscale("log"); ax.set_ylim(1e-2, 1e6); ax.set_xlim(1.0, 6.5)
ax.set_xlabel(r"$T_{\rm rec}$ [MeV]"); ax.set_ylabel("events / MeV / day")
ax.set_title("Detector singles against the reactor signal")
ax.legend(fontsize=10, ncol=2)
save(fig, "nb5_singles")

# --- 5h: sigma(sw2) vs the reactor-off exposure ratio, and vs the singles scale
r_grid = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0])
joint_r = [FixedNearReactor(burnup=0.5, reactor_off_ratio=rr).sigma_sw2(30.0, joint=True) * 100
           for rr in r_grid]
ev_r = [FixedNearReactor(burnup=0.5, reactor_off_ratio=rr).sigma_sw2(30.0) * 100
        for rr in r_grid]
# r = 0 is not the limit of the curve: it is the *other* analysis, in which the
# singles shape is taken from the model and only its normalisation is profiled.
trust = FixedNearReactor(burnup=0.5, reactor_off_ratio=0.0)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.semilogx(r_grid, joint_r, color=pl.BLUE, lw=1.9, label="joint IBD + E$\\nu$ES")
ax.semilogx(r_grid, ev_r, color=pl.RED, lw=1.9, ls="--", label="E$\\nu$ES only")
ax.axhline(trust.sigma_sw2(30.0, joint=True) * 100, color=pl.GREEN, lw=1.3, ls="-.",
           label="joint, no off-run (singles shape from the model)")
ax.axvline(1.0, color=pl.INK_MUTED, lw=1.0, ls=":")
ax.annotate("equal on/off", xy=(1.06, 0.30), fontsize=9, color=pl.INK_SECONDARY)
ax.set_xlabel(r"reactor-off exposure ratio $r = t_{\rm off}/t_{\rm on}$")
ax.set_ylabel(r"$\sigma(\sin^2\theta_W)$ [\%]")
ax.set_title(r"What the reactor-off run buys, 30 MW$\cdot$yr")
ax.legend(fontsize=10)
save(fig, "nb5_onoff")

# --- 5d: atomic stepping
t_dem = np.logspace(-6, 1, 800)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.semilogx(t_dem, atomic_stepping(t_dem), color=pl.BLUE, lw=1.7)
ax.axvspan(r.recoil_edges[0], r.recoil_edges[-1], color=pl.GREEN, alpha=0.15, lw=0)
ax.annotate("analysis window", xy=(1.3, 0.25), fontsize=10, color=pl.GREEN)
ax.set_xlabel(r"true recoil energy $T$ [MeV]"); ax.set_ylabel("active-electron fraction")
ax.set_title("Atomic binding (stepping) correction, LAB electrons")
save(fig, "nb5_atomic_stepping")

# --- App. A: the measured Daya Bay per-isotope yields as rates at JUNO, with error bands
from reactor.dayabay_data import load_unfolded
from reactor.flux import spectrum_per_fission, haleu_fractions, fission_rate_per_second, HM_EMAX
from reactor.constants import JUNO2025_TARGET_PROTONS, CM_PER_KM
spectra, cov75, _ = load_unfolded()
YR = 365.25 * DAY * 0.9

def per_iso_scale(iso):
    """events/yr at JUNO per (cm^2/fission) if the whole 10 MW core were this isotope."""
    Rf = fission_rate_per_second(0.010, {iso: 1.0})
    return Rf * JUNO2025_TARGET_PROTONS * 0.8 / (4 * np.pi * (0.05 * CM_PER_KM) ** 2) * YR

def step_xy(edges, vals):
    """(x, y) arrays tracing a step function, for fill_between."""
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(vals, 2)
    return x, y

enu_f = np.linspace(1.85, 9.5, 400)
xs = vogel_beacom(enu_f, order=1)
fig, ax = plt.subplots(figsize=(6.8, 4.6))
for iso, color, lab in (("U235", pl.BLUE, r"$^{235}$U (Daya Bay, measured, $\pm1\sigma$)"),
                        ("Pu239", pl.RED, r"$^{239}$Pu (Daya Bay, measured, $\pm1\sigma$)")):
    y = spectra[iso]; sc = per_iso_scale(iso)
    dens = y.values * sc / y.widths; derr = y.errors * sc / y.widths
    xx, yy = step_xy(y.edges, dens); _, ee = step_xy(y.edges, derr)
    ax.fill_between(xx, yy - ee, yy + ee, color=color, alpha=0.25, lw=0)
    ax.plot(xx, yy, color=color, lw=1.6, label=lab)
for iso, color, lab in (("U238", pl.GREEN, r"$^{238}$U (Huber--Mueller)"),
                        ("Pu241", pl.ORANGE, r"$^{241}$Pu (Huber--Mueller)")):
    ax.plot(enu_f, spectrum_per_fission(enu_f, iso) * xs * per_iso_scale(iso),
            color=color, lw=1.6, ls="--", label=lab)
ax.axvline(HM_EMAX, color=pl.INK_MUTED, lw=0.9, ls=":")
ax.annotate("HM fit range ends;\nartificial cutoff beyond", xy=(HM_EMAX + 0.05, 2.5e6),
            fontsize=8.5, color=pl.INK_SECONDARY)
ax.set_yscale("log"); ax.set_ylim(1e4, 5e7); ax.set_xlim(1.8, 9.5)
ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel(r"IBD events / MeV / year at JUNO")
ax.set_title(r"Per-isotope IBD rate at JUNO for a pure 10 MW core at 50 m")
ax.legend(fontsize=10)
save(fig, "appA_dayabay_isotopes")

# --- App. A companions: fuel-cycle evolution of the summed spectrum, 3:1 with a
#     ratio-to-start panel, for (i) the HALEU core and (ii) a Daya-Bay-style LEU core.
#     (Two-panel shared-x figures are the user's explicit request here.)
from reactor.dayabay_data import load_flux_evolution
import matplotlib.gridspec as gridspec

def core_spectrum(fracs):
    Rf = fission_rate_per_second(0.010, fracs)
    scale = Rf * JUNO2025_TARGET_PROTONS * 0.8 / (4 * np.pi * (0.05 * CM_PER_KM) ** 2) * YR
    yld = (fracs["U235"] * spectra["U235"](enu_f) + fracs["Pu239"] * spectra["Pu239"](enu_f)
           + (fracs["U238"] * spectrum_per_fission(enu_f, "U238")
              + fracs["Pu241"] * spectrum_per_fission(enu_f, "Pu241")) * xs)
    return yld * scale

def cycle_figure(specs, title, name, labels):
    """specs: dict beta -> spectrum, keyed 0.0/0.5/1.0. 3:1 spectrum + ratio panels."""
    fig = plt.figure(figsize=(6.8, 5.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
    sty = ((0.0, pl.BLUE, "-"), (0.5, pl.GREEN, "-."), (1.0, pl.RED, "--"))
    for b, color, ls in sty:
        ax.plot(enu_f, specs[b], color=color, ls=ls, lw=1.8, label=labels[b])
    ax.set_yscale("log"); ax.set_ylim(1e4, 5e7)
    ax.set_ylabel(r"IBD events / MeV / year at JUNO"); ax.set_title(title)
    ax.legend(fontsize=10); plt.setp(ax.get_xticklabels(), visible=False)
    for b, color, ls in sty[1:]:
        axr.plot(enu_f, specs[b] / specs[0.0], color=color, ls=ls, lw=1.8)
    axr.axhline(1.0, color=pl.INK_MUTED, lw=0.9)
    axr.set_ylim(0.85, 1.02); axr.set_xlim(1.8, 9.5)
    axr.set_xlabel(r"$E_\nu$ [MeV]"); axr.set_ylabel("ratio to start")
    save(fig, name)

# (i) HALEU
haleu = {b: core_spectrum(haleu_fractions(b, evolve=True)) for b in (0.0, 0.5, 1.0)}
cycle_figure(haleu, r"The 10 MW HALEU core at 50 m across its fuel cycle", "appA_haleu_cycle",
             {0.0: r"start of cycle ($\beta=0$)", 0.5: r"run-averaged ($\beta=0.5$)",
              1.0: r"end of cycle ($\beta=1$)"})

# (ii) a commercial LEU core: the Daya Bay measured fission fractions across their 20
#      burnup groups, same power and distance, so the two figures are directly comparable
ev = load_flux_evolution()
def dyb_fracs(k):
    return {"U235": ev["f235"][k], "U238": ev["f238"][k], "Pu239": ev["f239"][k], "Pu241": ev["f241"][k]}
kmid, kend = len(ev["group"]) // 2, len(ev["group"]) - 1
leu = {0.0: core_spectrum(dyb_fracs(0)), 0.5: core_spectrum(dyb_fracs(kmid)), 1.0: core_spectrum(dyb_fracs(kend))}
f0, fm, fe = dyb_fracs(0), dyb_fracs(kmid), dyb_fracs(kend)
cycle_figure(leu, r"A Daya-Bay-like LEU core (10 MW at 50 m) across its fuel cycle", "appA_leu_cycle",
             {0.0: rf"start of cycle ($F_{{235}}={f0['U235']:.2f}$, $F_{{239}}={f0['Pu239']:.2f}$)",
              0.5: rf"mid-cycle ($F_{{235}}={fm['U235']:.2f}$, $F_{{239}}={fm['Pu239']:.2f}$)",
              1.0: rf"end of cycle ($F_{{235}}={fe['U235']:.2f}$, $F_{{239}}={fe['Pu239']:.2f}$)"})
for b in (0.0, 0.5, 1.0):
    print(f"  HALEU beta={b}: {np.trapezoid(haleu[b], enu_f):.3e} IBD/yr | LEU: {np.trapezoid(leu[b], enu_f):.3e}")

# =============================================================== notebook 6
dm2_grid = np.logspace(-2, 2, 60)
st50 = SterileNearReactor(distance_m=50.0)
st50_nolbin = SterileNearReactor(distance_m=50.0, l_binned=False)

# --- 6a: the wiggle map
DM2_B, S22_B = 1.0, 0.10
w = st50.wiggle_template(DM2_B)
n_e = len(st50.e_edges) - 1; n_l = st50.n_l
null_ibd = np.array(st50._ibd_null); wig_ibd = w[:n_l * n_e].reshape(n_l, n_e)
ratio2d = 1.0 - S22_B * wig_ibd / np.maximum(null_ibd, 1e-30)
fig, ax = plt.subplots(figsize=(6.8, 4.4))
pc = ax.pcolormesh(st50.l_edges, st50.e_edges, ratio2d.T, cmap="RdBu",
                   vmin=1 - S22_B, vmax=1 + 0.2 * S22_B, rasterized=True)
plt.colorbar(pc, ax=ax, label=r"$P_{ee}$ per bin")
ax.set_xlabel(r"$L$ [m]"); ax.set_ylabel(r"$E_{\rm rec}$ [MeV]"); ax.set_ylim(1.5, 8)
ax.set_title(r"$\Delta m^2_{41}=1$ eV$^2$, $\sin^22\theta_{14}=0.1$, D = 50 m")
save(fig, "nb6_wiggle_map")

# --- 6b: slices vs L, five energy windows across the spectrum
fig, ax = plt.subplots(figsize=(6.6, 4.2))
slices = ((1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6.0), (6.0, 8.0))
cols = (pl.BLUE, pl.ORANGE, pl.GREEN, pl.RED, pl.MAGENTA)
for (elo, ehi), color in zip(slices, cols):
    L, ratio = st50.ratio_vs_L(DM2_B, S22_B, elo, ehi)
    ax.plot(L, ratio, color=color, lw=1.7,
            label=rf"$E_{{\rm rec}}\in[{elo:g},{ehi:g}]$ MeV")
ax.axhline(1.0, color=pl.INK_MUTED, lw=0.9)
ax.set_xlabel(r"$L$ [m]"); ax.set_ylabel("oscillated / no-oscillation")
ax.set_title("The wiggle across the detector, D = 50 m")
ax.legend(fontsize=10)
save(fig, "nb6_wiggle_slices")

# --- 6c: sensitivity landscape -- reproduces the user's notebook-6 figure exactly
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
dm2_land = np.geomspace(1e-2, 20, 200)
distance = [50, 100, 200, 500, 1400]
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0.1, 0.8, len(distance)))
land_curves = {}
for d, color in zip(distance, colors):
    land_curves[d] = (SterileNearReactor(distance_m=d, include_eves=True, l_binned=True)
                      .limit_curve(dm2_land), color)

def plot_nue_dis(ax):
    lp = ROOT / "reactor/data/nue_dis"; zorder = 1
    for f in ("DANSS.dat", "PROSPECT.dat", "STEREO.dat"):
        x, y = np.genfromtxt(lp / f, unpack=True)
        ax.fill_betweenx(y, x, x / x, edgecolor="grey", facecolor="lightgrey", lw=0, alpha=1, zorder=zorder)
    x, y = np.genfromtxt(lp / "RENO_NEOS.dat", unpack=True)
    o = np.argsort(y)
    ax.fill_betweenx(y[o], x[o], x[o] / x[o], edgecolor="grey", facecolor="lightgrey", lw=0, alpha=1, zorder=zorder)
    tx, ty = np.genfromtxt(lp / "KATRIN_2025_exclusion.dat", unpack=True)
    ax.plot(tx, ty, color="grey", lw=0.5, ls="-", zorder=3)
    ax.fill_betweenx(ty, tx, tx / tx, edgecolor="lightgrey", facecolor="lightgrey", lw=0.5, alpha=1, zorder=zorder)
    Ga0 = np.loadtxt(lp / "Gallium_2sigma_l0.csv", delimiter=",")
    Ga1 = np.loadtxt(lp / "Gallium_2sigma_l1.csv", delimiter=",")
    for G in (Ga0, Ga1):
        ax.fill(G[:, 0], G[:, 1], lw=0.75, edgecolor="None", facecolor="orange", zorder=zorder, alpha=0.6)
        ax.fill(G[:, 0], G[:, 1], lw=0.75, edgecolor="orange", facecolor="None", zorder=zorder, alpha=1)
    ax.plot([0.1704, 0.1704], [1e-4, 1e3], lw=1.2, color="orange", ls=(1, (3, 1)), zorder=1)

fig, ax = plt.subplots(figsize=(6.0, 5.0))
for d, color in zip(distance, colors):
    ax.loglog(land_curves[d][0], dm2_land, color=color, lw=1.8, label=rf"$D = {d}$ m")
x, y = np.loadtxt(ROOT / "reactor/data/nue_dis/IsoDAR_Yemilab_90CL.dat", unpack=True)
ax.plot(x, y, color="deeppink", lw=1.5, ls="--")
plot_nue_dis(ax)
ax.legend(fontsize=10, loc="lower left", title="JUNO + near reactor", title_fontsize=10, framealpha=0.95)
leg_main = ax.get_legend()
proxy = [Line2D([0], [0], color="grey", lw=1.0, label=r"KATRIN 95\% CL"),
         Patch(facecolor="lightgrey", edgecolor="None", label=r"Reactors 90\% CL"),
         Patch(facecolor="orange", edgecolor="orange", alpha=0.6, label=r"Gallium $2\sigma$ region"),
         Line2D([0], [0], color="orange", lw=1.5, ls=(0, (3, 1)), label=r"Solar 99\% CL"),
         Line2D([0], [0], color="deeppink", lw=1.5, ls="--", label=r"IsoDAR 90\% CL")]
ax.legend(handles=proxy, loc="upper left", fontsize=10, framealpha=0.95)
ax.add_artist(leg_main)
ax.set_xlim(1e-4, 1); ax.set_ylim(dm2_land[0], dm2_land[-1])
ax.set_xlabel(r"$\sin^2 2\theta_{14}$"); ax.set_ylabel(r"$\Delta m^2_{41}$ [eV$^2$]")
ax.set_title(r"$95\%$ CL sensitivity, 27 MW$\cdot$yr")
save(fig, "nb6_sensitivity_landscape")

# --- 6d: L-binned vs integrated, IBD-only, at 50 m.  Two integrated variants:
#     with the same free E-shape as the L-binned fit (which leaves it with no
#     handle at all), and with the Daya Bay shape covariance as a prior.
st50_nolbin_prior = SterileNearReactor(distance_m=50.0, l_binned=False, free_shape=False)
fig, ax = plt.subplots(figsize=(5.8, 4.6))
ax.loglog(land_curves[50][0], dm2_land, color=pl.BLUE, lw=1.9, label=r"$L$-binned, IBD + E$\nu$ES")
ax.loglog(SterileNearReactor(distance_m=50.0, include_eves=False).limit_curve(dm2_land), dm2_land,
          color=pl.GREEN, ls="-.", lw=1.5, label=r"$L$-binned, IBD only")
ax.loglog(st50_nolbin_prior.limit_curve(dm2_land), dm2_land, color=pl.ORANGE, ls="-", lw=1.6,
          label=r"integrated, Daya Bay shape prior")
ax.loglog(st50_nolbin.limit_curve(dm2_land), dm2_land, color=pl.INK_MUTED, ls="--", lw=1.6,
          label=r"integrated, free shape")
ax.set_xlim(3e-4, 1.2); ax.set_ylim(dm2_land[0], dm2_land[-1])
ax.set_xlabel(r"$\sin^2 2\theta_{14}$"); ax.set_ylabel(r"$\Delta m^2_{41}$ [eV$^2$]")
ax.set_title(r"What the finite size buys, D = 50 m")
ax.legend(fontsize=9, loc="upper left")
save(fig, "nb6_lbinned_vs_integrated")

# --- 6e: the high-dm2 wall
dm2_hi = np.logspace(-0.3, 1.78, 45)
# The oscillation length falls below a centimetre at the top of this range, so
# the E-grid has to resolve it: 9600 nodes, verified converged against 19200.
GRID = dict(l_bin_m=0.25, n_e_grid=9600)
configs = [
    ("point core, perfect vertex", dict(core_radius_m=0.0, sigma_vertex_m=0.0, n_sub=25, **GRID), pl.INK_SECONDARY, ":"),
    (r"+ vertex resolution $10\,{\rm cm}/\sqrt{E}$", dict(core_radius_m=0.0, sigma_vertex_m=0.10, **GRID), pl.GREEN, "-"),
    ("+ core radius 0.5 m (default)", dict(core_radius_m=0.5, sigma_vertex_m=0.10, **GRID), pl.BLUE, "-"),
    ("+ core radius 1.5 m", dict(core_radius_m=1.5, sigma_vertex_m=0.10, **GRID), pl.RED, "-"),
]
fig, ax = plt.subplots(figsize=(6.2, 4.6))
for lab, kw, color, ls in configs:
    stx = SterileNearReactor(distance_m=50.0, **kw)
    ax.loglog(stx.limit_curve(dm2_hi), dm2_hi, color=color, ls=ls, lw=1.8, label=lab)
ax.axvline(0.1, color=pl.INK_MUTED, lw=0.9, ls=":")
ax.set_xlim(5e-4, 1.0); ax.set_ylim(dm2_hi[0], dm2_hi[-1])
ax.set_xlabel(r"$\sin^2 2\theta_{14}$"); ax.set_ylabel(r"$\Delta m^2_{41}$ [eV$^2$]")
ax.set_title(r"The high-$\Delta m^2$ wall, D = 50 m")
ax.legend(fontsize=10, loc="upper left")
save(fig, "nb6_high_dm2_wall")

# =============================================================== notebook 7
alp = ALPSearchJUNO()
nb_full = alp.estimate_background()

for coupling, ma_grid, g_grid, name, ylab, tri in (
        ("gagg", np.logspace(np.log10(0.02), 1, 48), np.logspace(-7.5, -1.5, 120),
         "nb7_alp_photon", r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]",
         ([0.4, 6.0, 0.4, 0.4], [6e-6, 1.2e-4, 1.2e-4, 6e-6])),
        ("gaee", np.logspace(np.log10(0.02), 1, 48), np.logspace(-9, -4, 120),
         "nb7_alp_electron", r"$g_{aee}$", None)):
    chi2_b = alp.chi2_grid(coupling, ma_grid, g_grid, background=nb_full)
    chi2_f = alp.chi2_grid(coupling, ma_grid, g_grid, background=0.0)
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.contourf(ma_grid, g_grid, chi2_b.T, levels=[4.61, 1e9], colors=[pl.BLUE], alpha=0.25)
    ax.contour(ma_grid, g_grid, chi2_b.T, levels=[4.61], colors=[pl.BLUE], linewidths=1.8)
    ax.contour(ma_grid, g_grid, chi2_f.T, levels=[4.61], colors=[pl.BLUE], linewidths=1.4,
               linestyles="--")
    ax.plot([], [], color=pl.BLUE, lw=1.8, label=r"with background, 90\% CL")
    ax.plot([], [], color=pl.BLUE, lw=1.4, ls="--", label=r"background-free")
    if tri is not None:
        ax.plot(tri[0], tri[1], color=pl.RED, lw=1.4, ls=":", label="cosmological triangle (schematic)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$m_a$ [MeV]"); ax.set_ylabel(ylab)
    ax.set_title(r"ALP reach, 10 MW at 50 m, 27 MW$\cdot$yr")
    ax.legend(fontsize=10, loc="lower left")
    save(fig, name)

print(f"all figures written in {time.time()-t_all:.0f} s")
