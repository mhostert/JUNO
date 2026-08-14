"""JUNO movable-reactor sensitivity toolkit."""

from .backgrounds import JUNO_BACKGROUND_RATES, total_background_counts
from .constants import (
    DEFAULT_OSCILLATION_PARAMS,
    NUFIT61_IO,
    NUFIT61_NO,
    NUFIT61_NO_ERRORS,
    OscillationParameters,
)
from .cross_sections import AnalyticIBD, load_ibd_cross_section, vogel_beacom
from .detector import (
    DetectorResponse,
    EnergyResolution,
    NonLinearity,
    TabulatedNonLinearity,
    juno_2025_response,
    juno_nonlinearity,
)
from . import dayabay_data, juno_data
from .dayabay_data import DayaBayYield, dayabay_yield_model, load_unfolded
from .juno_data import load_chi2_map, load_nonlinearity, load_spectrum
from .juno_fit import JUNO2025Model
from .experiment import (
    Detector,
    Predictor,
    Sample,
    default_e_nu_grid,
    default_reco_edges,
    juno_far_sample,
    juno_reco_edges,
    near_program,
    near_stop_sample,
)
from .flux import ReactorCore, default_juno_cores, movable_reactor
from .oscillations import (
    survival_probability_2nu,
    survival_probability_ee,
    survival_probability_matter,
)
from .statistics import (
    Analysis,
    GaussianPrior,
    Systematics,
    build_covariance,
    chi2,
    chi2_cnp,
    cnp_variance,
)

__all__ = [
    "Analysis",
    "AnalyticIBD",
    "DEFAULT_OSCILLATION_PARAMS",
    "DayaBayYield",
    "Detector",
    "DetectorResponse",
    "EnergyResolution",
    "GaussianPrior",
    "JUNO2025Model",
    "JUNO_BACKGROUND_RATES",
    "NUFIT61_IO",
    "NUFIT61_NO",
    "NUFIT61_NO_ERRORS",
    "NonLinearity",
    "OscillationParameters",
    "Predictor",
    "ReactorCore",
    "Sample",
    "Systematics",
    "TabulatedNonLinearity",
    "build_covariance",
    "chi2",
    "chi2_cnp",
    "cnp_variance",
    "dayabay_data",
    "dayabay_yield_model",
    "default_e_nu_grid",
    "default_juno_cores",
    "default_reco_edges",
    "juno_2025_response",
    "juno_data",
    "juno_far_sample",
    "juno_nonlinearity",
    "juno_reco_edges",
    "load_chi2_map",
    "load_ibd_cross_section",
    "load_nonlinearity",
    "load_spectrum",
    "load_unfolded",
    "movable_reactor",
    "near_program",
    "near_stop_sample",
    "survival_probability_2nu",
    "survival_probability_ee",
    "survival_probability_matter",
    "total_background_counts",
    "vogel_beacom",
]
