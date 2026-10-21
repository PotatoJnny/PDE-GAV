"""Tasks offered by the VPFP plugin."""

from .observable import VPFPObservableTask
from .reduced_dynamics import VPFPReducedDynamicsTask
from .statistics import VPFPStatisticsTask

__all__ = [
    "VPFPObservableTask",
    "VPFPReducedDynamicsTask",
    "VPFPStatisticsTask",
]
