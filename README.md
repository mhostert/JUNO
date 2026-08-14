# JUNO sensitivity with a movable near reactor

Tools for estimating JUNO's sensitivity to the reactor oscillation parameters, with and
without a movable compact near reactor at $L \lesssim 3$ km.

## Notebooks

| Notebook | Content |
|---|---|
| `0_validation.ipynb` | IBD cross section, Huber–Mueller flux, event rates, geometric scaling, oscillation probabilities, and the detector response — the last two now checked against JUNO's *measured* resolution and released non-linearity curves rather than against design values. |
| `1_sensitivity_solar.ipynb` | The fit to the actual 2379-candidate JUNO spectrum for $\sin^2\theta_{12}$ and $\Delta m^2_{21}$ using the **standard method** (`reactor.nufit.standard_juno_fit`), compared with the official $\Delta\chi^2$ surface, with the independent `JUNO2025Model` as cross-check; then the six-year projection and the mass ordering. |
| `2_sensitivity_atmospheric.ipynb` | $\theta_{13}$ and $\Delta m^2_{ee}$: the draft's analysis reproduced, then the **standard-method near+far analysis** — a HALEU microreactor (98/2, Daya-Bay-shaped fuel evolution) with the measured U235/Pu239 spectra and the full $75\times75$ flux covariance shared between near and far, detector systematics correlated across the datasets, and the burnup-matching scheduling lesson. |
| `4_nufit_comparison.ipynb` | NuFit's JUNO analysis (arXiv:2601.09791v2) reproduced: their Tab. 2 rates core by core, their Tab. 1 configurations, their Fig. 4 $\Delta m^2_{ee}$ profile — and an ablation showing the one ingredient that matters is the bin-per-bin rescaling to JUNO's own un-oscillated spectrum. |
| `3_oscillation_validation.ipynb` | The survival probability on its own: exact analytic limits, the $\Phi$ branch and the $\pm\Phi$ ordering sign, matter effects, JUNO's Fig. 5, grid convergence — and the comparison against the $P_{ee}$ JUNO's own model implies, which is what identified the missing Daya Bay contribution described below; its fit comparisons now use the standard method. |

The notebooks are generated from plain-Python cell lists in [`notebook_sources/`](notebook_sources/);
see the README there. Editing the `.ipynb` files directly works too.

## Package layout

| Module | Content |
|---|---|
| `reactor.dayabay_data` | Loaders for the Daya Bay 2025 flux and spectrum release (`reactor/data/DayaBay_release_2025`): the unfolded IBD-yield spectra with their covariance, the prompt spectra, and the fuel-evolution groups. `DayaBayYield` is a drop-in reactor spectrum carrying the measured flux deficit and the 5 MeV bump. |
| `reactor.juno_data` | Loaders for the JUNO 2025 data release (`reactor/data/JUNO_data_release_2025`): the measured 66-bin spectrum and its backgrounds, the official $\Delta\chi^2$ surface, the measured energy resolution, and the fitted non-linearity curves. |
| `reactor.nufit` | NuFit's analysis (arXiv:2601.09791v2), implemented literally: their Appendix-A flux (Huber $\times$ cardinal cubics, bin-average-conserving, 25 covariance pulls), the bin-per-bin rescaling to JUNO's un-oscillated spectrum that defines their cnf 1, Vogel--Beacom, and the full Tab. 1 nuisance set. Reproduces their best fit to $0.1\sigma$ and their Fig. 4 $\Delta m^2_{ee}$ profile; `standard_juno_fit()` is the repository's **standard method** for fitting the 2025 release: their prescription with JUNO's CNP statistic and the full documented $2.4\%$ rate budget (Tab. 2 rates $\oplus$ selection efficiency, the note-added correction). |
| `reactor.juno_fit` | `JUNO2025Model`: prediction of, and fit to, the measured spectrum. Its docstring lists exactly what is taken from the release and what is assumed. |
| `reactor.constants` | Units, IBD kinematics, JUNO detector parameters, NuFit 6.1 / Daya Bay / JUNO 2025 measured values, `OscillationParameters`. |
| `reactor.flux` | Huber–Mueller per-fission spectra, fission fractions and burnup models, `ReactorCore`, the JUNO core list, and the movable source. |
| `reactor.cross_sections` | Tabulated Strumia–Vissani IBD cross section plus the analytic Vogel–Beacom expressions. |
| `reactor.oscillations` | Two-flavour, exact three-flavour vacuum (in the $\Delta m^2_{ee}$ and $\Delta m^2_{ij}$ forms), and exact constant-density matter survival probabilities. |
| `reactor.detector` | Energy resolution ($a/b/c$), Birks + Cherenkov non-linearity, and $E_\nu \to E_{\rm rec}$ response matrices. |
| `reactor.experiment` | `Sample` / `Predictor`: stacked spectra over data-taking blocks, evaluated fast by pre-computing everything independent of the oscillation parameters. |
| `reactor.backgrounds` | Geoneutrinos, accidentals, ${}^9$Li/${}^8$He, fast neutrons, $(\alpha,n)$, world reactors. |
| `reactor.statistics` | Covariance blocks, `Analysis` (Asimov $\chi^2$, Fisher, profiling, grids), priors, mass-ordering $\Delta\chi^2$. |
| `reactor.theta13` | `NearFarTheta13`: the joint standard-method far + HALEU-microreactor Asimov analysis — measured DYB per-isotope fluxes with the joint covariance, DYB-shaped fuel evolution, shared detector nuisances, analytic profiling. |
| `reactor.optimize` | `ProgramSpec`, greedy Fisher stop placement, L-BFGS-B dwell refinement, exposure scaling. |
| `reactor.plotting` | Shared matplotlib style and contour helpers. |

## Conventions

* Energies in MeV, baselines in km, mass splittings in eV$^2$.
* The atmospheric splitting is carried as $\Delta m^2_{ee}$; `OscillationParameters.ordering`
  is $+1$ (normal) or $-1$ (inverted) and enters through the sign of $\Phi$ in the vacuum
  probability. It is the entire vacuum mass-ordering signal and vanishes at short baselines.
* JUNO's signal is not only the eight Yangjiang/Taishan cores at $52.1$–$52.8$ km: it also
  includes the distant complexes in `JUNO_DISTANT_CORE_TABLE`. Averaged over the neutrino
  energies that feed the solar dip,

  | complex | $L$ | $\Delta_{21}$ | $\langle P_{ee}\rangle$ |
  |---|---|---|---|
  | Yangjiang / Taishan | 52.5 km | $0.50\pi$ | 0.15 (solar minimum) |
  | Daya Bay | 215 km | $2.05\pi$ | 0.86 (solar maximum) |
  | Taipingling | 265 km | $2.51\pi$ | 0.28 |
  | Fangchenggang | 411.7 km | $3.90\pi$ | 0.65 |

  Daya Bay carries only $\sim4\%$ of the unoscillated rate but lands almost exactly on a
  solar *maximum* for those energies, so its flux arrives six times less oscillated than the
  near cores and fills the solar dip in. Omitting it biases $\sin^2\theta_{12}$ low by
  $1.4\sigma$ and costs $7$ units of $\chi^2$; see notebook 3, §9.

  Only Daya Bay is carried as signal, which is JUNO's own definition — their signal sum runs
  over the nine reactors of Tab. 2 of JUNO:2022mxj, "eight reactors at a distance of about 53
  km and a single effective reactor from the Daya Bay complex at 215 km" (quoted in NuFit,
  arXiv:2601.09791v2). Everything further away is in the release's `world reactors`
  background. Moving Fangchenggang into the signal instead — NuFit's v2 choice — is a null
  effect provided its events are removed from that background: $\sin^2\theta_{12}$ shifts by
  $<0.001$. Doing one without the other double-counts it and shifts the fit by $+0.002$.
  Taipingling is at zero power because it was still starting up during the 2025 dataset —
  restore its design $17.4$ GW for future projections. Pass `include_distant=False` to
  recover the near-cores-only model.
* Samples carry a `group` tag. Correlated systematics act within a group only, so the near
  programme and the JUNO far-reactor dataset share no nuisance parameters.
* Within the near programme the shape covariance extends off-diagonally across every stop —
  a single source illuminates the detector at all positions — which is what lets the 200 m
  reference stop calibrate the oscillation-maximum stop.

## Install

```bash
pip install -e .
```

Requires numpy, scipy and matplotlib.
