from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol, Tuple

Array = Any


@dataclass(frozen=True)
class GridSpec:
    shape: Tuple[int, int, int] = (32, 32, 32)
    domain: Tuple[float, float, float] = (2.0 * 3.141592653589793,) * 3
    dtype: str = "float64"
    dealias: bool = True


@dataclass(frozen=True)
class TimeSpec:
    t0: float
    t1: float
    nt: int

    @property
    def dt(self) -> float:
        if self.nt < 2:
            raise ValueError("TimeSpec.nt must be at least 2")
        return (self.t1 - self.t0) / (self.nt - 1)


@dataclass(frozen=True)
class PDEParams:
    nu: float
    forcing_kmax: int
    equation: str = "periodic_incompressible_ns3d"


@dataclass(frozen=True)
class PublicCaseSpec:
    case_id: str
    grid: GridSpec
    time: TimeSpec
    params: PDEParams
    source_fn: Callable[[Array, float, Dict[str, Any]], Array]
    initial_condition_fn: Callable[[Array, Dict[str, Any]], Array]
    public_params: Dict[str, Any]


@dataclass
class HiddenTruthCase:
    public: PublicCaseSpec
    u_true: Array
    f_grid: Array
    metadata: Dict[str, Any]
    validity: Dict[str, float]


@dataclass
class VerificationResult:
    case_id: str
    agent_id: str
    valid_prediction: bool
    scores: Dict[str, float]
    failure_modes: List[str]
    recommended_next_actions: List[str]
    raw_metrics: Dict[str, Any]


@dataclass
class ExperimentManifest:
    run_id: str
    run_dir: str
    suite: str
    seed: int
    config: Dict[str, Any]
    git_head: str
    command: str


class AgentProtocol(Protocol):
    def solve(self, case: PublicCaseSpec) -> Dict[str, Any]:
        """Return at least {"u": array shaped (nt,3,nx,ny,nz)}."""
