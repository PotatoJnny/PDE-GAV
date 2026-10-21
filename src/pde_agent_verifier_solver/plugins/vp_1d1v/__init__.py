"""Canonical Artemis / forced 1D1V Vlasov--Poisson product plugin."""

from .generator import ArtemisBackend, ArtemisCheckpoint, ArtemisGenerator
from .guidance import ArtemisRewardGuidance, ArtemisRewardState
from .macro_matching import VP1DMacroMatcher
from .physical import ARTEMIS_PHYSICAL_REVISION, ArtemisPhysicalPackage
from .problem import VP1DProblem, VP1D_MACRO_FIELDS, causal_horizon_case
from .product import build_artemis_plugin, load_artemis_backend, preflight_artemis_backend
from .public_problem import vp1d_public_problem_bundle
from .task import ArtemisEvaluation, ArtemisObservableTask, LegacyMacroCandidateAdapter

__all__ = [
    "ARTEMIS_PHYSICAL_REVISION", "ArtemisBackend", "ArtemisCheckpoint",
    "ArtemisEvaluation", "ArtemisGenerator", "ArtemisObservableTask",
    "ArtemisRewardGuidance", "ArtemisRewardState",
    "ArtemisPhysicalPackage", "LegacyMacroCandidateAdapter", "VP1DMacroMatcher",
    "VP1DProblem", "VP1D_MACRO_FIELDS", "build_artemis_plugin",
    "causal_horizon_case", "load_artemis_backend", "preflight_artemis_backend",
    "vp1d_public_problem_bundle",
]
