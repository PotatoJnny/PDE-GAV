"""Bind existing Artemis assets to the single canonical CampaignController."""

from __future__ import annotations

from pathlib import Path

from pde_agent_verifier_solver.plugins.registry import PDEPlugin
from pde_agent_verifier_solver.sdk import DatasetInput, DatasetKind
from .generator import ArtemisBackend, ArtemisGenerator
from .macro_matching import VP1DMacroMatcher
from .physical import ARTEMIS_PHYSICAL_REVISION, ArtemisPhysicalPackage
from .problem import VP1DProblem, VP1D_EXACT_COORDINATE_ORACLE_MODE
from .public_problem import vp1d_public_problem_bundle
from .task import ArtemisObservableTask
from ..plasma_transport_settings import configured_matcher, transport_calibration


def build_artemis_plugin(backend: ArtemisBackend | None = None, *, metric_profile=None,
                         metric_calibration=None) -> PDEPlugin:
    matcher = configured_matcher({"metric_profile": metric_profile,
        "metric_calibration": metric_calibration}, VP1DMacroMatcher(), "vp_1d1v")
    generator = ArtemisGenerator(backend)
    return PDEPlugin(
        plugin_id="vp_1d1v", title="Artemis / 1D1V Vlasov--Poisson",
        description="Find a macro-only model from admitted continuous kinetic answers.",
        aliases=("artemis", "vp-1d1v", "vlasov_poisson_1d1v"),
        problem=VP1DProblem(), generator=generator,
        tasks={"observable_model": ArtemisObservableTask(matcher)},
        default_task="observable_model", macro_matcher=matcher,
        public_problem_bundle=vp1d_public_problem_bundle(
            oracle_mode=backend.public_oracle_mode if backend else VP1D_EXACT_COORDINATE_ORACLE_MODE,
            guidance_supported=generator.capabilities.guidance,
        ),
    )


def preflight_artemis_backend(
    *, physical_checkout: str | Path, admission_record_path: str | Path,
    expected_admission_record_sha256: str, science_config_path: str | Path,
    expected_science_config_sha256: str, expected_physical_revision: str | None = None,
):
    """Verify existing source/artifact bindings without drawing or training cases."""
    package = ArtemisPhysicalPackage(physical_checkout, expected_revision=expected_physical_revision)
    startup = package.module("artemis_compact_v5_startup").preflight_artemis_compact_v5_startup(
        admission_record_path=admission_record_path,
        expected_admission_record_sha256=expected_admission_record_sha256,
        science_config_path=science_config_path,
        expected_science_config_sha256=expected_science_config_sha256,
    )
    return package, startup


def load_artemis_backend(
    *, physical_checkout: str | Path, admission_record_path: str | Path,
    expected_admission_record_sha256: str, science_config_path: str | Path,
    expected_science_config_sha256: str, device: str = "cpu",
    runtime_settings=None, guidance_adapter_factory=None, reward_guidance_settings=None,
    oracle_settings=None, expected_physical_revision: str | None = None,
    metric_profile=None, metric_calibration=None,
):
    """Return BackendRunBinding for the already admitted compact-v5 assets.

    Loading does not run a campaign or train the physical Generator. The
    optional guidance factory must implement canonical fit_guidance/generate
    and report its own qualification; it never changes the admitted base.
    Omitting oracle_settings retains the physical baseline execution path.
    An offline experiment may supply expected_physical_revision to select
    another exact committed physical package; no source verification is skipped.
    Explicit optimized settings have isolated paired-pilot evidence, not a
    general numerical-equivalence or scientific-admission certificate.
    """
    from pde_agent_verifier_solver.app.backend_run import BackendRunBinding

    package, startup = preflight_artemis_backend(
        physical_checkout=physical_checkout,
        admission_record_path=admission_record_path,
        expected_admission_record_sha256=expected_admission_record_sha256,
        science_config_path=science_config_path,
        expected_science_config_sha256=expected_science_config_sha256,
        expected_physical_revision=expected_physical_revision,
    )
    loaded = package.module("artemis_compact_v5_startup").load_artemis_compact_v5_startup_artifacts(startup, device=device)
    runtime_module = package.module("artemis_compact_v5_production_runtime")
    runtime_kwargs = dict(
        config=runtime_module.ArtemisCompactV5ProductionRuntimeConfig(**dict(runtime_settings or {})),
        admission=startup.admission, artifact=loaded.artifact,
        sensor_candidate=loaded.sensor_candidate,
    )
    if oracle_settings is not None:
        runtime_kwargs["oracle_options"] = runtime_module.ArtemisCompactV5OracleOptions(
            **dict(oracle_settings)
        )
    runtime = runtime_module.VP1DArtemisCompactV5ProductionRuntime(**runtime_kwargs)
    source = package.module("manufactured_backend").VP1DManufacturedCaseSource(
        runtime, generator_id=ArtemisGenerator.info.program_id,
        maximum_equation_relative_residual=0.0,
        maximum_poisson_relative_residual=0.0,
        maximum_source_cross_backend_relative_l2=0.0,
        expected_offline_admission_sha256=startup.admission.record_sha256,
    )
    if guidance_adapter_factory is not None and reward_guidance_settings is not None:
        raise ValueError("choose one Artemis guidance adapter")
    guide = None if guidance_adapter_factory is None else guidance_adapter_factory(source)
    if reward_guidance_settings is not None:
        from .guidance import ArtemisRewardGuidance
        guide = ArtemisRewardGuidance(package, source, settings=reward_guidance_settings, device=device,
            metric_profile=metric_profile, metric_calibration=metric_calibration)
    physical_revision = ARTEMIS_PHYSICAL_REVISION if expected_physical_revision is None else expected_physical_revision
    backend = ArtemisBackend(source, physical_revision=physical_revision,
                             admission_sha256=startup.admission.record_sha256,
                             guidance_adapter=guide)
    backend.score_calibration = transport_calibration(metric_profile, metric_calibration)
    if guide is not None and backend.score_calibration is not None:
        if (getattr(guide, "score_calibration", None) is None or
                guide.score_calibration.objective_hash != backend.score_calibration.objective_hash):
            raise ValueError("Artemis Guidance must bind the same calibrated task objective")
    plugin = build_artemis_plugin(backend, metric_profile=metric_profile, metric_calibration=metric_calibration)
    dataset = plugin.problem.prepare_dataset(DatasetInput(
        "artemis-admitted-online", DatasetKind.NUMERICAL, settings={"online_only": True},
    ))
    return BackendRunBinding(plugin, backend.checkpoint(), dataset, {
        "physics": "vp_1d1v", "physical_revision": physical_revision,
        "admission_sha256": startup.admission.record_sha256,
        "checkpoint_sha256": startup.admission.checkpoint_sha256,
        "oracle_mode": source.public_oracle_mode,
        "guidance_enabled": backend.supports_guidance,
        "guidance_qualification": "experimental_insufficient_perturbation_evidence" if reward_guidance_settings is not None else "not_requested",
        "perturbations_enabled": backend.supports_perturbations,
        "model_training_performed": False,
        **({"scoring_objective": backend.score_calibration.objective_identity}
           if backend.score_calibration is not None else {}),
        "oracle_optimization_settings": dict(oracle_settings or {}),
        # Broader qualification remains pending even for the measured pilot arm.
        "oracle_optimization_qualification": (
            "experimental_equivalence_pending"
            if oracle_settings is not None else "not_requested"
        ),
    })
