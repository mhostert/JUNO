OUT = "0_validation.ipynb"

CELLS = [
("md", r"""
# 0 — Validation of the rate, oscillation, geometry and energy-response models

This notebook checks the ingredients of the movable-reactor sensitivity study one at a
time, against published numbers wherever they exist.

| Section | What is checked | Against |
|---|---|---|
| 1 | IBD cross section | Vogel–Beacom analytic vs the tabulated Strumia–Vissani values |
| 2 | Reactor flux | Huber–Mueller per-fission spectra, $\bar\nu$ and IBD yields per fission |
| 3 | JUNO event rate | 47.1 IBD/day (JUNO Yellow Book) |
| 4 | Geometric scaling | $1/L^2$, and the rate ratios quoted in the draft |
| 5 | Oscillations | dip depth/position, the $8.4\%$ claim, JUNO's $\lesssim2\%$ modulation |
| 6 | Energy resolution | $3\%/\sqrt{E}$ at 1 MeV, JUNO $a/b/c$ model |
| 7 | Non-linearity | Birks + Cherenkov model vs the published LSNL curve, and its impact |
"""),

("code", r"""
import numpy as np
import matplotlib.pyplot as plt

from reactor import plotting as pl
from reactor.constants import *
from reactor.cross_sections import (
    IBDCrossSection, load_ibd_cross_section, vogel_beacom, positron_energy,
)
from reactor.flux import *
from reactor.oscillations import *
from reactor.detector import *
from reactor.experiment import *
from reactor import juno_data as jd

pl.use_style()
np.set_printoptions(precision=4, suppress=True)

truth = NUFIT61_NO
print("NuFit 6.1 NO reference point")
print(f"  sin^2(th12) = {truth.sin2_theta12}, dm2_21 = {truth.dm2_21:.3e} eV^2")
print(f"  sin^2(th13) = {truth.sin2_theta13}, dm2_ee = {truth.dm2_ee:.4e} eV^2")
print(f"  -> theta13  = {truth.theta13_deg:.3f} deg,  sin^2(2 th13) = {truth.sin2_2theta13:.5f}")
print(f"  -> dm2_31   = {truth.dm2_31:.4e} eV^2 (NO)")
"""),

("md", r"""
## 1. IBD cross section

`reactor/data/TCS_CC_anue_p_1026_SV.txt` is the tabulated $\bar\nu_e p \to e^+ n$ cross
section (Strumia–Vissani).  We first confirm its units by comparing against the
analytic Vogel–Beacom expressions, and check the threshold and the mean positron
energy used to build the prompt-energy axis.
"""),

("code", r"""
xsec = load_ibd_cross_section()   # scale 1e-38 cm^2 assumed
e = np.linspace(1.81, 10.0, 600)

sig_tab = xsec(e)
sig_vb0 = vogel_beacom(e, order=0)
sig_vb1 = vogel_beacom(e, order=1)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
ax.plot(e, sig_tab * 1e42, color=pl.BLUE, label="Table (Strumia–Vissani)")
ax.plot(e, sig_vb1 * 1e42, color=pl.ORANGE, ls="--", label="Vogel–Beacom, $O(1/M)$")
ax.plot(e, sig_vb0 * 1e42, color=pl.AQUA, ls=":", label="Vogel–Beacom, naive")
ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel(r"$\sigma_{\rm IBD}$ [$10^{-42}$ cm$^2$]")
ax.set_title("IBD cross section"); ax.legend()

ax = axes[1]
ax.plot(e, sig_vb1 / np.where(sig_tab > 0, sig_tab, np.nan), color=pl.ORANGE, label="VB $O(1/M)$ / table")
ax.plot(e, sig_vb0 / np.where(sig_tab > 0, sig_tab, np.nan), color=pl.AQUA, ls=":", label="VB naive / table")
ax.axhline(1.0, color=pl.INK_MUTED, lw=1.0)
ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel("ratio to table")
ax.set_ylim(0.95, 1.08); ax.set_title("Agreement with the tabulated cross section"); ax.legend()
plt.show()

print(f"IBD threshold          : {IBD_THRESHOLD:.4f} MeV")
print(f"prompt offset E_nu-E_p : {PROMPT_ENERGY_OFFSET_MEV:.4f} MeV")
mask = (e > 2.0) & (e < 8.0)
print(f"max |VB(1) / table - 1| over 2-8 MeV : {np.max(np.abs(sig_vb1[mask]/sig_tab[mask]-1)):.3%}")
from reactor.cross_sections import SIGMA0_CM2_PER_MEV2
print(f"sigma_0 (1+3g^2)       : {SIGMA0_CM2_PER_MEV2*F_PLUS_G_SQUARED:.4e} cm^2/MeV^2 "
      f"(classic value 0.0952e-42 uses tau_n = 886 s; we use {NEUTRON_LIFETIME_S} s)")
"""),

("code", r"""
# Mean positron energy and the prompt-energy mapping.
e_e = positron_energy(e, order=1)
e_dep = e_e + M_E                      # positron total energy + annihilation gamma
naive = e - PROMPT_ENERGY_OFFSET_MEV

fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(e, e_dep - naive, color=pl.BLUE)
ax.set_xlabel(r"$E_\nu$ [MeV]")
ax.set_ylabel(r"$E_{\rm dep} - (E_\nu - 0.782)$ [MeV]")
ax.set_title("Recoil correction to the prompt-energy mapping")
plt.show()
print(f"recoil shift at 3 MeV: {np.interp(3.0, e, e_dep-naive)*1e3:+.1f} keV")
print(f"recoil shift at 8 MeV: {np.interp(8.0, e, e_dep-naive)*1e3:+.1f} keV")
"""),

("md", r"""
## 2. Reactor flux

Huber (${}^{235}$U, ${}^{239}$Pu, ${}^{241}$Pu) + Mueller (${}^{238}$U) polynomial
parameterisations.  Two quantities have well-known published values and make good
checks:

* the number of $\bar\nu_e$ per fission above the IBD threshold, and
* the **IBD yield per fission** $\sigma_f = \int dE\,S(E)\,\sigma_{\rm IBD}(E)$,
  which for Huber–Mueller is $6.69,\,10.10,\,4.40,\,6.03 \times 10^{-43}\,{\rm cm^2/fission}$
  for ${}^{235}$U, ${}^{238}$U, ${}^{239}$Pu, ${}^{241}$Pu.
"""),

("code", r"""
e_fine = np.linspace(1.806, 9.5, 4000)
w = integration_weights(e_fine)
sig = xsec(e_fine)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
for iso, color in zip(ISOTOPES, pl.PALETTE):
    ax.plot(e_fine, spectrum_per_fission(e_fine, iso), color=color, label=iso)
ax.set_yscale("log"); ax.set_ylim(1e-4, 3)
ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel(r"$dN/dE$ [$\bar\nu$/fission/MeV]")
ax.set_title("Huber–Mueller spectra"); ax.legend()

ax = axes[1]
juno_mix = juno_average_fractions()
s_mix = mixed_spectrum_per_fission(e_fine, juno_mix)
s_near = mixed_spectrum_per_fission(e_fine, haleu_fractions())
ax.plot(e_fine, s_mix * sig * 1e42, color=pl.BLUE, label="JUNO PWR mix")
ax.plot(e_fine, s_near * sig * 1e42, color=pl.ORANGE, label="HALEU microreactor (90/10)")
ax.set_xlabel(r"$E_\nu$ [MeV]")
ax.set_ylabel(r"$S(E)\,\sigma_{\rm IBD}$ [$10^{-42}$cm$^2$/fission/MeV]")
ax.set_title("Detected (IBD-weighted) spectrum"); ax.legend()
plt.show()

peak = e_fine[np.argmax(s_mix * sig)]
print(f"detected spectrum peaks at E_nu = {peak:.2f} MeV  (E_prompt = {peak-PROMPT_ENERGY_OFFSET_MEV:.2f} MeV)")
"""),

("code", r"""
rows = []
for iso in ISOTOPES:
    s = spectrum_per_fission(e_fine, iso)
    n_nu = float(np.sum(s * w))
    yield_1e43 = float(np.sum(s * sig * w)) * 1e43
    ref = HM_IBD_YIELD_REF_1E43[iso]
    rows.append([iso, n_nu, yield_1e43, ref, 100 * (yield_1e43 / ref - 1)])

print(pl.table(rows,
    ["isotope", "nu/fission (>1.8 MeV)", "sigma_f [1e-43 cm^2]", "HM reference", "diff [%]"],
    floatfmt="{:.3f}"))

s_mix_full = mixed_spectrum_per_fission(e_fine, juno_mix)
sf_mix = float(np.sum(s_mix_full * sig * w)) * 1e43
print(f"\nJUNO cycle-average mix (58/7/30/5): sigma_f = {sf_mix:.2f}e-43 cm^2/fission")
print(f"mean energy per fission            = {average_energy_per_fission(juno_mix):.2f} MeV")
sf_near = float(np.sum(mixed_spectrum_per_fission(e_fine, haleu_fractions()) * sig * w)) * 1e43
print(f"HALEU near core (90/10):            sigma_f = {sf_near:.2f}e-43 cm^2/fission "
      f"({sf_near/sf_mix:.3f} x the JUNO mix)")
"""),

("md", r"""
## 3. JUNO event rate

The reference reactor complex is Yangjiang ($6\times2.9$ GW$_{\rm th}$) + Taishan
($2\times4.6$ GW$_{\rm th}$) $=26.6$ GW$_{\rm th}$ at $\approx 52.5$ km.  The Yellow Book
quotes **47.1 IBD candidates/day** for 20 kt after selection.
"""),

("code", r"""
edges = juno_reco_edges()          # 20 keV bins, 0.94-9.0 MeV
far_full = juno_far_sample(365.25, duty_cycle=1.0)
far_duty = juno_far_sample(365.25)                 # 11/12 load factor

P_full = Predictor([far_full], reco_edges_mev=edges)
P_duty = Predictor([far_duty], reco_edges_mev=edges)

r_un = P_full.rate_per_day(oscillate=False)[0]
r_os = P_full.rate_per_day()[0]
print(f"target protons      : {JUNO_TARGET_PROTONS:.3e}   (20 kt, {JUNO_HYDROGEN_MASS_FRACTION:.1%} H by mass)")
print(f"  JUNO 2025 measured: {JUNO2025_TARGET_PROTONS:.3e} +- {JUNO2025_TARGET_PROTONS_ERR:.3e}"
      f"  -> ours is {100*(JUNO_TARGET_PROTONS/JUNO2025_TARGET_PROTONS-1):+.1f}%")
print(f"selection efficiency: {JUNO_IBD_EFFICIENCY:.3f}")
print(f"  JUNO 2025 measured: {JUNO2025_EFFICIENCY_TOTAL:.3f} +- "
      f"{JUNO2025_EFFICIENCY_TOTAL*JUNO2025_EFFICIENCY_TOTAL_ERR:.3f}  "
      f"(FV 80.6% x mu-veto 93.6% x mult. 97.4% x coinc. 95.1%)")
print(f"total thermal power : {sum(c.power_gwth for c in far_full.cores):.1f} GW")
print()
print(f"unoscillated rate   : {r_un:6.2f} IBD/day")
print(f"oscillated rate     : {r_os:6.2f} IBD/day     <P_ee> = {r_os/r_un:.4f}")
print(f"  with 11/12 duty   : {P_duty.rate_per_day()[0]:6.2f} IBD/day")
print(f"JUNO Yellow Book    :  47.1  IBD/day  -> we are {100*(r_os/47.1-1):+.1f}%")
print()
print("Against the 2025 measurement (arXiv:2511.14593, Table 1), which quotes")
print("efficiency-corrected rates for the actual 59.1-day run:")
print(f"  JUNO non-oscillated : {JUNO2025_NONOSC_CPD:6.1f} +- {JUNO2025_NONOSC_CPD_ERR} cpd")
print(f"  ours, nominal power : {r_un:6.1f} cpd   -> ratio {JUNO2025_NONOSC_CPD/r_un:.3f}")
print(f"  JUNO oscillated     : {JUNO2025_SIGNAL_CPD_CORRECTED:6.1f} +- {JUNO2025_SIGNAL_CPD_CORRECTED_ERR} cpd")
print(f"  ours                : {r_os:6.1f} cpd")
print(f"  <P_ee> measured     : {JUNO2025_SIGNAL_CPD_CORRECTED/JUNO2025_NONOSC_CPD:.4f}"
      f"     ours: {r_os/r_un:.4f}")
print()
print("The survival probabilities agree to better than 1%, but our absolute rate is ~19% high")
print("because we assume all eight cores at nominal 26.6 GW. During this run Yangjiang cores")
print("YJ1 and YJ4 were off for periods and Super Typhoon Ragasa forced a power reduction on")
print("24 September (their Fig. 10). Notebook 1 therefore anchors the normalisation to JUNO's")
print("quoted non-oscillated rate rather than to nominal power.")
"""),

("code", r"""
# Contribution of each core, and the effective single-core baseline.
rows = []
for core in far_full.cores:
    p = Predictor([juno_far_sample(365.25, duty_cycle=1.0)], reco_edges_mev=edges)
    single = Sample(core.name, (core,), 365.25, group="far")
    rate = Predictor([single], reco_edges_mev=edges).rate_per_day()[0]
    rows.append([core.name, core.power_gwth, core.baseline_km, rate, 100 * rate / r_os])
print(pl.table(rows, ["core", "P [GW]", "L [km]", "IBD/day", "share [%]"], floatfmt="{:.3f}"))

eq = equivalent_single_core(list(far_full.cores))
print(f"\nflux-weighted effective baseline: {eq.baseline_km:.3f} km at {eq.power_gwth:.1f} GW")
rate_eq = Predictor([Sample("eq", (eq,), 365.25)], reco_edges_mev=edges).rate_per_day()[0]
print(f"single-equivalent-core rate      : {rate_eq:.2f} IBD/day  ({100*(rate_eq/r_os-1):+.2f}% vs the 8-core sum)")
"""),

("md", r"""
## 4. Geometric scaling and the rate ratios quoted in the draft

The movable source is a compact core of $100\;{\rm MW_{th}}$.  Three ratios appear in the
draft; each is checked here explicitly.  The distinction that matters is between the
**unoscillated flux ratio** — which is what $(P_{\rm near}/P_{\rm far})(L_{\rm far}/L_{\rm near})^2$
gives — and the **event-rate ratio**, which additionally divides out JUNO's far-reactor
oscillation suppression $\langle P_{ee}\rangle \approx 0.31$ and accounts for the
different fuel mix.
"""),

("code", r"""
baselines = np.geomspace(0.1, 5.0, 60)
rate_near, rate_near_unosc = [], []
for L in baselines:
    s = near_stop_sample(L, 365.25, power_mwth=100.0, include_far=False)
    p = Predictor([s], reco_edges_mev=edges)
    rate_near.append(p.rate_per_day()[0])
    rate_near_unosc.append(p.rate_per_day(oscillate=False)[0])
rate_near = np.array(rate_near); rate_near_unosc = np.array(rate_near_unosc)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
ax.loglog(baselines, rate_near, color=pl.ORANGE, label="near source, oscillated")
ax.loglog(baselines, rate_near_unosc, color=pl.BLUE, ls="--", label=r"near source, $P_{ee}=1$")
ref = rate_near_unosc[0] * (baselines[0] / baselines) ** 2
ax.loglog(baselines, ref, color=pl.INK_MUTED, ls=":", lw=1.2, label=r"$\propto 1/L^2$")
pl.hline_reference(ax, r_os, "JUNO far-reactor rate")
ax.set_xlabel(r"$L$ [km]"); ax.set_ylabel("IBD / day")
ax.set_title(r"Movable 100 MW$_{\rm th}$ source at JUNO"); ax.legend(loc="lower left")

ax = axes[1]
ax.semilogx(baselines, rate_near / r_os, color=pl.ORANGE, label="event-rate ratio (oscillated)")
ax.semilogx(baselines, rate_near_unosc / r_un, color=pl.BLUE, ls="--",
            label="unoscillated flux ratio")
ax.set_yscale("log")
ax.set_xlabel(r"$L$ [km]"); ax.set_ylabel("ratio to JUNO far reactors")
ax.set_title("Near / far ratio"); ax.legend()
plt.show()

print(f"max |rate/(1/L^2) - 1| below 3 km : "
      f"{np.max(np.abs(rate_near_unosc*baselines**2/(rate_near_unosc[0]*baselines[0]**2) - 1)[baselines<3]):.2e}")
"""),

("code", r"""
def near_rate(L, power=100.0, oscillate=True):
    s = near_stop_sample(L, 365.25, power_mwth=power, include_far=False)
    return Predictor([s], reco_edges_mev=edges).rate_per_day(oscillate=oscillate)[0]

print("Draft claim 1: 'at the first minimum the IBD rate is about 6x the JUNO far-reactor setup'")
geom = (0.1 / 26.6) * (52.5 / 1.3) ** 2
print(f"   unoscillated flux ratio at 1.3 km, 100 MW : {geom:.2f}  <- matches the quoted 6x")
print(f"   actual event-rate ratio at 1.3 km         : {near_rate(1.3)/r_os:.1f}x")
print("   -> the 6x figure omits JUNO's own <P_ee> = 0.31 suppression and the fuel-mix difference\n")

print("Draft claim 2 (appendix): 'the rate at the dip is ~1800x the standard JUNO rate per unit mass'")
print(f"   (L_far/L_dip)^2 = (52.5/1.235)^2          : {(52.5/1.235)**2:.0f}   <- this is the quoted 1800")
print(f"   actual event-rate ratio at 1.235 km       : {near_rate(1.235)/r_os:.1f}x")
print("   -> 1800 is the bare geometric factor, not a rate ratio\n")

print("Draft claim 3: 'the 0.20 km anchor runs at ~190x JUNO's rate'")
print(f"   actual event-rate ratio at 0.20 km        : {near_rate(0.20)/r_os:.0f}x")
print(f"   unoscillated flux ratio, 26.6 GW          : {(0.1/26.6)*(52.5/0.2)**2:.0f}")
print(f"   unoscillated flux ratio, 36 GW (old CDR)  : {(0.1/36.0)*(52.5/0.2)**2:.0f}   <- reproduces 190")
"""),

("md", r"""
## 5. Oscillations

Three independent implementations (the $\Delta m^2_{ee}$ form, the textbook
$\Delta m^2_{ij}$ form, and numerical diagonalisation in matter with $\rho\to0$) agree to
machine precision, so the survival probability itself is not a source of error.
"""),

("code", r"""
E = np.linspace(1.81, 9.0, 3000)
for L in [0.20, 1.235, 52.5]:
    a = survival_probability_ee(E, L, truth)
    b = survival_probability_3nu_masses(E, L, truth)
    c = survival_probability_matter(E, L, truth, density_g_cm3=0.0)
    print(f"L = {L:6.3f} km : max|dm2ee-form - mass-form| = {np.max(np.abs(a-b)):.2e}, "
          f"max|dm2ee-form - matter(rho=0)| = {np.max(np.abs(a-c)):.2e}")
"""),

("code", r"""
loe = np.linspace(0.0, 1.2, 4000)     # km/MeV
E_ref = 4.0
p_exact = survival_probability_ee(E_ref, loe * E_ref, truth)
p_2nu   = survival_probability_2nu(E_ref, loe * E_ref, truth)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(loe, p_exact, color=pl.BLUE, label=r"exact 3$\nu$")
ax.plot(loe, p_2nu, color=pl.ORANGE, ls="--", label=r"2$\nu$: $1-\sin^2 2\theta_{13}\sin^2\Delta_{ee}$")
ax.axhline(1 - truth.sin2_2theta13, color=pl.INK_MUTED, ls=":", lw=1.2)
ax.set_xlabel(r"$L/E$ [km/MeV]"); ax.set_ylabel(r"$P_{ee}$")
ax.set_ylim(0.89, 1.005)
ax.set_title(r"Survival probability near the first atmospheric minimum")
ax.legend(loc="lower left")

i = int(np.argmin(p_exact))
ax.plot(loe[i], p_exact[i], "o", color=pl.BLUE, ms=6)
pl.annotate_right(ax, loe[i], p_exact[i] - 0.004,
                  f"min at {loe[i]:.3f} km/MeV\n$P_{{ee}}$ = {p_exact[i]:.4f}", color=pl.BLUE)
plt.show()

print(f"sin^2(2 theta13)                       = {truth.sin2_2theta13:.5f}  -> 2nu dip depth {100*truth.sin2_2theta13:.2f}%")
print(f"exact 3nu minimum of P_ee              = {p_exact[i]:.4f}  -> dip depth {100*(1-p_exact[i]):.2f}%")
print(f"analytic L/E of the first maximum      = {first_minimum_loe(truth):.4f} km/MeV")
print(f"  -> L at the detected-spectrum peak E = 3.65 MeV : {first_minimum_loe(truth)*3.65:.2f} km")
print(f"  -> L at E = 3.0 MeV (draft's choice)            : {first_minimum_loe(truth)*3.0:.2f} km")
print("  Note: the statistically optimal stop is NOT at this baseline. Sensitivity comes from")
print("  the gradient of P_ee across the whole spectrum weighted by the rate, which falls as")
print("  1/L^2, so the optimiser in notebook 2 settles nearer 1.2-1.5 km.")
print()
print("Draft states 'P_ee^min ~ 0.916, an 8.4% dip'.")
print(f"  At the NuFit 6.1 point the two-flavour value is 1 - sin^2(2th13) = {1-truth.sin2_2theta13:.4f} ({100*truth.sin2_2theta13:.2f}%),")
print(f"  and the exact three-flavour minimum is {p_exact[i]:.4f} ({100*(1-p_exact[i]):.2f}%) because the solar term adds a little depth.")
print("  0.916 / 8.4% corresponds to sin^2(2th13) = 0.084, i.e. sin^2(th13) = 0.0215.")
"""),

("code", r"""
# Bin-averaged dip after the detector response: what is actually observable.
resp_juno = DetectorResponse(resolution=EnergyResolution())          # JUNO a/b/c
L_dip = 1.235
sam = near_stop_sample(L_dip, 365.25, include_far=False)
edges_n = default_reco_edges()
p_on  = Predictor([sam], reco_edges_mev=edges_n, response=resp_juno).counts()[0]
p_off = Predictor([sam], reco_edges_mev=edges_n, response=resp_juno).counts(oscillate=False)[0]
centers = 0.5 * (edges_n[:-1] + edges_n[1:])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
ax.plot(centers, p_off / 1e3, color=pl.INK_SECONDARY, ls="--", label="no oscillation")
ax.plot(centers, p_on / 1e3, color=pl.ORANGE, label=r"oscillated, $L = 1.235$ km")
ax.set_xlabel(r"$E_{\rm prompt}$ [MeV]"); ax.set_ylabel(r"counts / bin [$10^3$, 1 yr]")
ax.set_title("Prompt spectrum at the oscillation dip"); ax.legend()

ax = axes[1]
ax.plot(centers, p_on / p_off, color=pl.ORANGE, label="observed ratio (smeared)")
e_nu_c = centers + PROMPT_ENERGY_OFFSET_MEV
ax.plot(centers, survival_probability_ee(e_nu_c, L_dip, truth), color=pl.BLUE, ls="--",
        label=r"$P_{ee}$, unsmeared")
ax.set_xlabel(r"$E_{\rm prompt}$ [MeV]"); ax.set_ylabel("ratio to unoscillated")
ax.set_title("Effect of energy resolution on the dip"); ax.legend()
plt.show()

print(f"deepest unsmeared point : {survival_probability_ee(e_nu_c, L_dip, truth).min():.4f}")
print(f"deepest smeared bin     : {(p_on/p_off).min():.4f}")
print(f"rate-averaged survival  : {p_on.sum()/p_off.sum():.4f}")
"""),

("code", r"""
# JUNO's own baseline: the theta13 wiggle on top of a fully developed solar dip.
# The observable piece is the *fast* wiggle, so compare against a spectrum in which
# the atmospheric oscillation has been averaged out but theta13 is otherwise present.
E = np.linspace(1.81, 9.0, 8000)
L_far = 52.5

# P_ee with cos(2 Delta_ee + Phi) -> 0, i.e. the fast wiggle averaged away.
def p_ee_atm_averaged(e_nu, L, p):
    d21 = KM_EV2_PER_GEV * p.dm2_21 * L / (e_nu / 1000.0)
    c13 = 1.0 - p.sin2_theta13
    return 1.0 - 0.5 * p.sin2_2theta13 - c13**2 * p.sin2_2theta12 * np.sin(d21) ** 2

p_full = survival_probability_ee(E, L_far, truth)
p_avg  = p_ee_atm_averaged(E, L_far, truth)
p_no13 = survival_probability_ee(E, L_far, truth.replace(sin2_theta13=0.0))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
ax.plot(E, p_full, color=pl.BLUE, lw=0.9, label=r"full $P_{ee}$")
ax.plot(E, p_avg, color=pl.ORANGE, lw=1.8, label=r"atmospheric term averaged")
ax.plot(E, p_no13, color=pl.AQUA, lw=1.4, ls="--", label=r"$\theta_{13}=0$")
ax.set_xlim(2, 8); ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel(r"$P_{ee}$")
ax.set_title(f"JUNO far reactors, L = {L_far} km"); ax.legend(loc="upper right", fontsize=8)

# Now the same thing at the level of observable, resolution-smeared counts.
sam6 = juno_far_sample(6 * 365.25)
pred6 = Predictor([sam6], reco_edges_mev=edges,
                  response=DetectorResponse(resolution=EnergyResolution()))
c_full = pred6.counts(truth)[0]
c_no13 = pred6.counts(truth.replace(sin2_theta13=0.0))[0]

c_avg  = pred6.counts_with_probability(lambda en, L: p_ee_atm_averaged(en, L, truth))[0]
cen6 = 0.5 * (edges[:-1] + edges[1:])

ax = axes[1]
ax.plot(cen6, c_full / np.where(c_avg > 0, c_avg, np.nan), color=pl.BLUE, lw=1.2,
        label="fast wiggle, smeared")
ax.axhline(1.0, color=pl.INK_MUTED, lw=1.0)
ax.set_xlim(1.0, 8.0)
ax.set_xlabel(r"$E_{\rm rec}$ [MeV]"); ax.set_ylabel(r"counts / counts(atm. averaged)")
ax.set_title("Observable $\\theta_{13}$ wiggle after energy smearing"); ax.legend()
plt.show()

wiggle = p_full - p_avg
print(f"mean suppression from theta13, <P>      : {1 - p_full.mean()/p_no13.mean():.4f}")
ratio = c_full / np.where(c_avg > 0, c_avg, np.nan)

# The wiggle period grows like E^2 while the resolution grows like sqrt(E), so the
# smearing suppression is strongly energy dependent. Quote a local envelope.
def local_envelope(x, y, half_width=0.35):
    return np.array([np.nanmax(np.abs(y[np.abs(x - xi) < half_width] - 1)) for xi in x])

env = local_envelope(cen6, ratio)
frac = np.cumsum(c_full) / c_full.sum()

print(f"unsmeared fast-wiggle amplitude in P_ee : +/- {np.abs(wiggle).max():.4f}"
      f"  (sin^2(2th13)/2 = {truth.sin2_2theta13/2:.4f})")
print()
print(" E_rec [MeV]   smeared wiggle envelope   cumulative event fraction")
for e0 in [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]:
    print(f"   {e0:4.1f}              +/- {100*np.interp(e0, cen6, env):5.2f}%"
          f"                  {np.interp(e0, cen6, frac):.3f}")
sel = cen6 < 4.0
w80 = np.nansum(env[sel] * c_full[sel]) / np.nansum(c_full[sel])
print(f"\nrate-weighted envelope below 4 MeV (~{100*np.interp(4.0, cen6, frac):.0f}% of the events): "
      f"+/- {100*w80:.2f}%")
print()
print("Draft: 'at 52.5 km the theta13-dependent signal is only a <~2% modulation'.")
print("  NOT reproduced. The wiggle period grows like E^2 while the resolution width grows")
print("  only like sqrt(E), so smearing suppresses the wiggle at low energy but barely touches")
print("  it above ~4 MeV. Weighted over the observed spectrum the surviving modulation is")
print(f"  +/-{100*np.nansum(env*c_full)/np.nansum(c_full):.1f}%, and it falls below 2% only under ~1.9 MeV, where few events sit.")
print("  So JUNO's theta13 imprint is not small in amplitude. The reason JUNO measures theta13")
print("  poorly is not the size of the signal but its degeneracy with the reactor flux shape")
print("  and with dm2_ee -- notebook 2 quantifies this and finds sigma(theta13) ~ 0.5 deg for")
print("  a far-reactor-only fit, in agreement with the draft's Table I.")
"""),

("code", r"""
# Matter effect at the JUNO baseline (used later for the mass-ordering fit).
E = np.linspace(1.81, 9.0, 1500)
p_vac = survival_probability_matter(E, 52.5, truth, density_g_cm3=0.0)
p_mat = survival_probability_matter(E, 52.5, truth, density_g_cm3=EARTH_CRUST_DENSITY_G_CM3)
p_io  = survival_probability_matter(E, 52.5, NUFIT61_IO, density_g_cm3=EARTH_CRUST_DENSITY_G_CM3)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
ax.plot(E, (p_mat - p_vac) * 1e3, color=pl.BLUE)
ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel(r"$(P^{\rm matter}-P^{\rm vac})\times 10^3$")
ax.set_title(r"Matter effect, $\rho = 2.6$ g/cm$^3$, $L=52.5$ km")
ax = axes[1]
ax.plot(E, p_mat, color=pl.BLUE, lw=1.0, label="normal ordering")
ax.plot(E, p_io, color=pl.ORANGE, lw=1.0, label="inverted ordering")
ax.set_xlim(3.0, 4.0); ax.set_xlabel(r"$E_\nu$ [MeV]"); ax.set_ylabel(r"$P_{ee}$")
ax.set_title("NO vs IO at fixed $|\\Delta m^2_{ee}|$"); ax.legend()
plt.show()
print(f"max matter-induced shift in P_ee : {np.max(np.abs(p_mat-p_vac)):.2e}")
print(f"max |P_NO - P_IO| at fixed |dm2_ee| : {np.max(np.abs(p_mat-p_io)):.3f}")
"""),

("md", r"""
## 6. Energy resolution -- as measured, not as designed

JUNO's design resolution is $\sigma_E/E = 3\%$ at 1 MeV, usually written
$\sqrt{(a/\sqrt{E})^2 + b^2 + (c/E)^2}$ with $(a,b,c)=(2.61,\,0.82,\,1.23)\%$. The 2025
release gives the **measured** resolution at eight calibration peaks, so we can fit $a,b,c$
to data rather than assume the design values.

The quoted errors are statistical only ($10^{-4}$ level) and much smaller than the
source-to-source spread the three-parameter form can absorb, so the fit adds a 2% relative
floor; otherwise it is driven by one or two peaks.
"""),

("code", r"""
E_cal, r_cal, r_err, labels = jd.load_energy_resolution()
res_meas = EnergyResolution.from_juno_calibration()
res_design = EnergyResolution()                       # a/b/c design values
res_draft = EnergyResolution.sqrt_only(0.03)          # 3% / sqrt(E)

print(f"fitted to the calibration peaks: a = {100*res_meas.a:.2f}%, b = {100*res_meas.b:.2f}%, "
      f"c = {100*res_meas.c:.2f}%")
print(f"design (Yellow Book)           : a = {100*res_design.a:.2f}%, b = {100*res_design.b:.2f}%, "
      f"c = {100*res_design.c:.2f}%")
print()
print(pl.table([[l, e, 100*m, 100*res_meas.relative(e), 100*(res_meas.relative(e)/m - 1)]
                for l, e, m in zip(labels, E_cal, r_cal)],
               ["source", "E_rec [MeV]", "measured [%]", "fit [%]", "diff [%]"],
               floatfmt="{:.3f}"))
print()
print(f"sigma_E/E at 1 MeV: measured-fit {100*res_meas.relative(1.0):.2f}%, "
      f"design {100*res_design.relative(1.0):.2f}%")
print(f"at the 68Ge peak (0.923 MeV): {100*res_meas.relative(0.923):.2f}% "
      f"(the paper quotes 'approximately 3.5%')")
print()
print("The detector is running ~12% worse than design at 1 MeV. The paper says as much:")
print("'Studies of the energy resolution achieved for gammas at the detector center show")
print("slightly worse performance than predicted by MC simulations.' This matters little")
print("for the solar parameters but a great deal for the mass ordering (notebook 1).")

Ev = np.linspace(0.5, 10.0, 400)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
ax.errorbar(E_cal, 100*r_cal, yerr=100*r_err, fmt="o", color=pl.INK, ms=5, capsize=2,
            label="JUNO 2025 calibration peaks", zorder=5)
ax.plot(Ev, 100*res_meas.relative(Ev), color=pl.ORANGE, label="fit to the measured peaks")
ax.plot(Ev, 100*res_design.relative(Ev), color=pl.BLUE, ls="--", label="design $a/b/c$")
ax.plot(Ev, 100*res_draft.relative(Ev), color=pl.AQUA, ls=":", label=r"draft: $3\%/\sqrt{E}$")
ax.set_xlabel(r"$E_{\rm rec}$ [MeV]"); ax.set_ylabel(r"$\sigma_E/E$ [%]")
ax.set_title("Energy resolution"); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(Ev, 100*res_meas.effective_a(Ev), color=pl.ORANGE, label="measured")
ax.plot(Ev, 100*res_design.effective_a(Ev), color=pl.BLUE, ls="--", label="design")
ax.axhline(3.0, color=pl.AQUA, ls=":", label="draft, 3% by construction")
ax.set_xlabel(r"$E_{\rm rec}$ [MeV]"); ax.set_ylabel(r"effective $a$ [%]")
ax.set_title(r"Equivalent $a/\sqrt{E}$ coefficient"); ax.legend(fontsize=8)
plt.show()
"""),

("code", r"""
# How much does the resolution choice matter for the near-reactor dip?
sam = near_stop_sample(1.235, 365.25, include_far=False)
ratios = {}
for name, res in [("measured (JUNO 2025)", res_meas), ("design $a/b/c$", res_design),
                  (r"draft $0.03\sqrt{E}$", res_draft),
                  ("perfect resolution", EnergyResolution(a=1e-6, b=0, c=0))]:
    p = Predictor([sam], reco_edges_mev=edges_n, response=DetectorResponse(resolution=res))
    ratios[name] = p.counts()[0] / p.counts(oscillate=False)[0]

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for (name, r), color in zip(ratios.items(), [pl.ORANGE, pl.BLUE, pl.AQUA, pl.INK_MUTED]):
    ax.plot(centers, r, color=color, label=name, lw=2.0 if "perfect" not in name else 1.2)
ax.set_xlabel(r"$E_{\rm prompt}$ [MeV]"); ax.set_ylabel("oscillated / unoscillated")
ax.set_title(r"Smearing of the dip at $L=1.235$ km"); ax.legend(fontsize=8)
plt.show()
for name, r in ratios.items():
    print(f"{name:24s}: minimum ratio {r.min():.4f}")
print()
print("All four are indistinguishable: at L ~ 1 km the atmospheric oscillation is slow in")
print("prompt energy (one cycle spans several MeV), so the dip is far wider than even the")
print("degraded resolution. Energy resolution is therefore NOT a limiting systematic for the")
print("near programme -- unlike at 52.5 km, where the wiggles have a ~0.2 MeV period and the")
print("0.35% resolution shortfall costs real mass-ordering sensitivity.")
"""),

("md", r"""
## 7. Liquid-scintillator non-linearity -- the measured curve

The release provides the collaboration's fitted non-linearity $E_{\rm vis}/E_{\rm true}$ for
positrons, electrons and gammas, with uncertainty bands. This replaces the guessed reference
curve used earlier in this repository.

Two things are worth separating:

* the **released positron curve** is what the analysis must use. It includes both
  scintillator non-linearity (Birks quenching + Cherenkov) and instrumental non-linearity
  from PMT charge reconstruction, and is normalised so that it crosses unity near 3.6 MeV.
* the **Birks + Cherenkov model** in `reactor.detector` describes only the scintillator part
  and normalises to unity at high energy. Comparing the two therefore only makes sense after
  matching the normalisation convention, and then it is a test of *shape*.
"""),

("code", r"""
E_nl, f_nl, lo_nl, hi_nl = jd.load_nonlinearity("positron")
nl_tab = TabulatedNonLinearity.from_release("positron")
nl_fit = juno_nonlinearity()      # Birks + Cherenkov fitted to the released shape

f_shape = f_nl / np.interp(12.0, E_nl, f_nl)   # released curve on the model's convention

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
ax = axes[0]
for kind, color in [("positron", pl.BLUE), ("electron", pl.ORANGE), ("gamma", pl.AQUA)]:
    e_k, f_k, lo_k, hi_k = jd.load_nonlinearity(kind)
    ax.plot(e_k, f_k, color=color, label=kind)
    ax.fill_between(e_k, f_k + lo_k, f_k + hi_k, color=color, alpha=0.25, lw=0)
ax.axhline(1.0, color=pl.INK_MUTED, lw=1.0)
ax.set_xlim(0, 10); ax.set_xlabel(r"$E_{\rm true}$ [MeV]")
ax.set_ylabel(r"$E_{\rm vis}/E_{\rm true}$")
ax.set_title("Released non-linearity curves"); ax.legend(fontsize=8)

ax = axes[1]
sel = (E_nl >= 1.022) & (E_nl <= 11.0)
ax.plot(E_nl[sel], f_shape[sel], color=pl.BLUE, lw=2.2, label="released positron curve (shape)")
ax.plot(E_nl[sel], nl_fit.factor(E_nl[sel]), color=pl.ORANGE, ls="--",
        label="Birks + Cherenkov model")
ax.set_xlabel(r"$E_{\rm true}$ [MeV]"); ax.set_ylabel(r"$E_{\rm vis}/E_{\rm true}$ (norm. at 12 MeV)")
ax.set_title("Physics model vs the measured shape"); ax.legend(fontsize=8)
plt.show()

dev = nl_fit.factor(E_nl[sel]) - f_shape[sel]
print(f"released positron curve: f(1.022) = {f_nl[0]:.4f}, crosses 1 at "
      f"{np.interp(1.0, f_nl, E_nl):.2f} MeV, f(12) = {np.interp(12.0, E_nl, f_nl):.4f}")
print(f"published band at 1.022 MeV: +-{100*(hi_nl[0]-lo_nl[0])/2:.2f}%  "
      f"(the paper quotes 1% overall for positrons)")
print()
print(f"Birks + Cherenkov fit to the shape: kB = {nl_fit.kb:.4f} cm/MeV, "
      f"Cherenkov weight = {nl_fit.fc:.3f}")
print(f"  max deviation {100*np.max(np.abs(dev)):.2f}%, rms {100*np.sqrt(np.mean(dev**2)):.3f}%")
print()
print("The shape is reproduced to ~0.1% rms, but the fit drives the Cherenkov weight to zero")
print("and kB to ~0.05 cm/MeV, well above the 0.006-0.015 typical of organic scintillators.")
print("That is the expected outcome: the released curve is the FULL positron non-linearity,")
print("so a scintillator-only model absorbs the instrumental component into an effective")
print("quenching parameter. The number should not be read as a physical Birks constant.")
"""),

("code", r"""
# Which energy is the released spectrum binned in: visible, or non-linearity corrected?
# The answer is decided by the bins below the 1.022 MeV kinematic minimum.
spec = jd.load_spectrum()
xsec_v = load_ibd_cross_section()
enu_v = np.linspace(1.806, 13.0, 2600)
w_v = integration_weights(enu_v)
dens_v = np.zeros_like(enu_v)
truth_j = truth.replace(sin2_theta12=JUNO2025_SIN2_THETA12, dm2_21=JUNO2025_DM2_21)
for core in default_juno_cores(duty_cycle=1.0):
    dens_v += (core.flux_at_detector(enu_v) * xsec_v(enu_v) * w_v
               * survival_probability_ee(enu_v, core.baseline_km, truth_j))

rows = []
for label, use_nl in [("E_rec = visible energy (NL applied)", True),
                      ("E_rec = deposited energy (NL corrected)", False)]:
    resp_v = DetectorResponse(resolution=res_meas,
                              nonlinearity=nl_tab if use_nl else None)
    ev = resp_v.visible_energy(enu_v)
    pred = gaussian_bin_response(ev, spec.edges, res_meas.sigma(ev)) @ dens_v
    pred = pred / pred.sum() * spec.pred_signal.sum()
    m = spec.pred_signal > 1.0
    rows.append([label, np.sum((pred[m]-spec.pred_signal[m])**2/spec.pred_signal[m]),
                 pred[0], pred[1]])
rows.append(["released Npred_signal", float("nan"), spec.pred_signal[0], spec.pred_signal[1]])
print(pl.table(rows, ["interpretation", "shape chi2 (66 bins)",
                      "counts 0.7-1.0 MeV", "counts 1.0-1.1 MeV"], floatfmt="{:.2f}"))
print()
print("The released signal prediction puts 12.4 events below 1.022 MeV, the minimum possible")
print("prompt deposit (2 m_e). Only the visible-energy interpretation produces them: the")
print("non-linearity pulls the 1.022 MeV edge down to 0.91 MeV, from where the ~4% resolution")
print("tail reaches 0.85. So the release is binned in VISIBLE energy, and the analysis in")
print("notebook 1 applies the non-linearity rather than correcting for it.")
"""),

("md", r"""
## Summary

| Quantity | This work | Reference | Status |
|---|---|---|---|
| IBD cross-section table units | $10^{-38}$ cm$^2$ | Vogel–Beacom $O(1/M)$ | agree to $<2\%$ over 2–8 MeV |
| $\sigma_f$ per isotope | computed | Huber–Mueller | $\le0.8\%$ on all four |
| JUNO IBD rate | 46.1/day | 47.1/day (Yellow Book) | $-2.2\%$ |
| $1/L^2$ scaling | exact | — | $<10^{-15}$ |
| Resolution at 1 MeV | $3.00\%$ | JUNO design $3\%$ | exact |
| LSNL curve | Birks + Cherenkov fit | published curve | $0.31\%$ max deviation |
| Oscillation code | 3 implementations | each other | machine precision |

### Statements in the draft that do not reproduce

1. **"$P_{ee}^{\rm min}\sim0.916$, an $8.4\%$ dip"** (§III and appendix). At the NuFit 6.1
   point $\sin^2 2\theta_{13}=0.0866$, so the two-flavour minimum is $0.9134$ ($8.66\%$) and
   the exact three-flavour minimum is $0.9116$ ($8.84\%$). The quoted $0.916/8.4\%$ corresponds
   to $\sin^2\theta_{13}=0.0215$, not the $0.0222$ stated in the same section. The draft is
   also internally inconsistent: the discussion says "an $8.6\%$ dip".
2. **"the IBD event rate is about $6\times$ larger than the standard far-reactor JUNO setup"**
   (§III). $6.1$ is the *unoscillated flux* ratio. The event-rate ratio is $\approx21\times$,
   because JUNO's far reactors are suppressed by $\langle P_{ee}\rangle=0.31$ and the HALEU
   core has a $13\%$ higher IBD yield per fission.
3. **"the event rate at the dip is about ${\sim}1800\times$ the standard JUNO rate per unit
   mass"** (appendix). $1807=(52.5/1.235)^2$ is the bare geometric factor with the power ratio
   omitted; the rate ratio is $\approx24\times$.
4. **"the $L\sim0.20$ km anchor, at ${\sim}190\times$ JUNO's rate"** (§VI). $190$ follows from
   a $36\;{\rm GW}$ reactor complex (the old CDR value); with the $26.6\;{\rm GW}$ used
   elsewhere in the draft the unoscillated ratio is $259$ and the event-rate ratio is
   $\approx950\times$.
5. **"the $\theta_{13}$-dependent signal is only a $\lesssim2\%$ modulation"** at 52.5 km
   (§I). The observable, resolution-smeared modulation is $\pm4\%$ averaged over the spectrum
   and drops below $2\%$ only below $\approx1.9$ MeV. JUNO's poor $\theta_{13}$ reach comes from
   the degeneracy with the flux shape and $\Delta m^2_{ee}$, not from a small signal — see
   notebook 2.

None of these affect the sensitivity results, which are computed from the full spectra rather
than from these ratios; they are wording/arithmetic issues in the text.
"""),
]
