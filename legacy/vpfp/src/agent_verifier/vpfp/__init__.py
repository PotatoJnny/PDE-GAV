"""Factorized electrostatic VPFP agent-verifier track.

The MVP keeps this adapter independent from the legacy Navier-Stokes and
triplane code paths.  Cases are answer-first and store the hidden kinetic
answer as rho(t, x) and g(t, v | x), with f = rho * g treated as a derived
quantity.
"""

from agent_verifier.vpfp.autodiff_source import (
    AUTODIFF_SOURCE_ORIGIN,
    VPFPAutodiffSourceResult,
    compute_autodiff_source_from_triplanes,
)
from agent_verifier.vpfp.continuous_diffusion import (
    VPFPContinuousDenoiser,
    VPFPContinuousDiffusionConfig,
    sample_continuous_latents,
)
from agent_verifier.vpfp.continuous_guidance import (
    VPFPContinuousRewardConfig,
    VPFPContinuousRewardModel,
    VPFPContinuousRewardModelConfig,
    hold_initial_solver_reward,
    make_reward_guidance,
)
from agent_verifier.vpfp.continuous_oracle import (
    VPFPContinuousPrivateCase,
    VPFPContinuousPublicOracle,
    VPFPOracleQueryLimits,
)
from agent_verifier.vpfp.continuous_training import (
    VPFPContinuousEncoder,
    VPFPContinuousEncoderConfig,
    VPFPContinuousLossConfig,
    continuous_autoencoder_loss,
    continuous_feature_grid,
)
from agent_verifier.vpfp.continuous_v2 import (
    CONTINUOUS_VPFP_REPRESENTATION,
    STANDARD_VPFP_EQUATION_VERSION,
    VELOCITY_DENSITY_DICTIONARY,
    VELOCITY_SQUARE_HERMITE,
    VPFPContinuousField,
    VPFPContinuousFieldConfig,
    VPFPStandardEquationConfig,
    manufacture_standard_vpfp_source,
)
from agent_verifier.vpfp.baselines import hold_initial_solution, truth_solution
from agent_verifier.vpfp.basis import VPFPBasisConfig, decode_basis_coeffs, project_g_to_basis, velocity_basis
from agent_verifier.vpfp.d0 import audit_numerical_d0, build_numerical_d0
from agent_verifier.vpfp.generator import CPUMMSGenerator, VPFPGeneratorProtocol
from agent_verifier.vpfp.numerical import VPFPNumericalConfig, make_numerical_d0_case, solve_public_equation
from agent_verifier.vpfp.io import VPFPCase, assert_public_bundle_isolated, load_case, write_case
from agent_verifier.vpfp.operators import (
    VPFPGrid,
    VPFPTime,
    compute_source,
    f_from_rho_g,
    poisson_electric_field,
    residual,
)
from agent_verifier.vpfp.verifier import evaluate_prediction, verify_case_dir

__all__ = [
    "AUTODIFF_SOURCE_ORIGIN",
    "CONTINUOUS_VPFP_REPRESENTATION",
    "CPUMMSGenerator",
    "STANDARD_VPFP_EQUATION_VERSION",
    "VELOCITY_DENSITY_DICTIONARY",
    "VELOCITY_SQUARE_HERMITE",
    "VPFPBasisConfig",
    "VPFPCase",
    "VPFPAutodiffSourceResult",
    "VPFPContinuousDenoiser",
    "VPFPContinuousDiffusionConfig",
    "VPFPContinuousEncoder",
    "VPFPContinuousEncoderConfig",
    "VPFPContinuousField",
    "VPFPContinuousFieldConfig",
    "VPFPContinuousLossConfig",
    "VPFPContinuousPrivateCase",
    "VPFPContinuousPublicOracle",
    "VPFPContinuousRewardConfig",
    "VPFPContinuousRewardModel",
    "VPFPContinuousRewardModelConfig",
    "VPFPGeneratorProtocol",
    "VPFPGrid",
    "VPFPOracleQueryLimits",
    "VPFPStandardEquationConfig",
    "VPFPTime",
    "VPFPNumericalConfig",
    "audit_numerical_d0",
    "build_numerical_d0",
    "assert_public_bundle_isolated",
    "compute_autodiff_source_from_triplanes",
    "compute_source",
    "continuous_autoencoder_loss",
    "continuous_feature_grid",
    "decode_basis_coeffs",
    "evaluate_prediction",
    "f_from_rho_g",
    "hold_initial_solution",
    "hold_initial_solver_reward",
    "load_case",
    "make_reward_guidance",
    "manufacture_standard_vpfp_source",
    "make_numerical_d0_case",
    "poisson_electric_field",
    "project_g_to_basis",
    "residual",
    "sample_continuous_latents",
    "solve_public_equation",
    "truth_solution",
    "velocity_basis",
    "verify_case_dir",
    "write_case",
]
