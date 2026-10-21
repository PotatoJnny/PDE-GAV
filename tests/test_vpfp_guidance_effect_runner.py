"""Tiny synthetic runner contracts, not trained Guidance effectiveness evidence."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_PATH = Path(__file__).resolve().parents[1] / "scripts/run_vpfp_strong_guidance_effect.py"
_SPEC = importlib.util.spec_from_file_location("vpfp_guidance_effect_runner_test", _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def _asset(path):
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def effect_config(tmp_path, monkeypatch):
    # Only external model/codec validation is replaced. SHA checks and actual
    # registered-root/stratum derivation remain production implementations.
    validated = []

    def validate(config):
        validated.append(deepcopy(config))
        return config

    monkeypatch.setattr(runner, "validate_config", validate)
    round_cfg = {
        "public_access_mode": "demand_legacy_arrays_v1",
        "counts": {"fit": 64, "validation": 16, "selection": 16, "confirmation": 32},
        "legacy_solver_array_grid": {"nx": 33, "ny": 33, "nt": 13, "nvx": 9, "nvy": 9,
                                     "velocity_bounds": [-5, 5, -5, 5]},
        "condition_strata": {"collision_nu_edges": [10, 15, 20, 25, 30],
                             "temperature_edges": [.8, 1., 1.2]},
        "base_weights_trainable": False,
        "solver_weights_or_code_trainable": False,
        "failed_solver_outputs_are_high_rewards": False,
        "old_scores_used_as_training_labels": False,
        "base_checkpoint_sha256": "a" * 64,
        "solver_sha256": "b" * 64,
        "portable_codec": {"synthetic_fixture": True},
        "sampling": {"selection_strengths": [0., .5, 2.], "heun_steps": 2},
        "seeds": {"fit": 10, "validation": 20, "selection": 30, "confirmation": 40},
    }
    old_plan = {"config": deepcopy(round_cfg), "roots": runner.registered_roots(round_cfg)}
    paths = {role: tmp_path / (role + ".json")
             for role in ("guide", "training_plan", "training_labels", "training_report")}
    paths["guide"].write_bytes(b"synthetic, not a Torch checkpoint")
    _write_json(paths["training_labels"], {"synthetic_fixture": True})
    _write_json(paths["training_plan"], old_plan)
    training = {"final_step": 2000, "target_step": 2000,
                "label_manifest_sha256": _asset(paths["training_labels"])["sha256"]}
    _write_json(paths["training_report"], training)
    round_path = tmp_path / "round.json"
    _write_json(round_path, round_cfg)
    cfg = {"schema": runner.SCHEMA, "expected_pairs": 32, "strength": 2.,
           "bootstrap_replicates": 10000, "bootstrap_seed": 20260908,
           "round_config": str(round_path),
           "fresh_split_seeds": {"fit": 101, "validation": 202,
                                 "selection": 303, "confirmation": 404},
           **{role: _asset(path) for role, path in paths.items()}}
    config_path = tmp_path / "effect.json"
    _write_json(config_path, cfg)
    return SimpleNamespace(path=config_path, cfg=cfg, round_cfg=round_cfg,
                           old_plan=old_plan, training=training, paths=paths,
                           validated=validated)


def test_effect_loads_32_fresh_roots_balanced_across_eight_strata(effect_config):
    fixture = effect_config
    original = fixture.path.read_bytes()
    cfg, new_round, roots, training = runner.load_effect_config(fixture.path)
    assert cfg["strength"] == 2. and training["final_step"] == 2000
    assert len(roots) == len({row["root_seed"] for row in roots}) == 32
    assert Counter(row["stratum"] for row in roots) == {s: 4 for s in range(8)}
    assert all(row["split"] == "confirmation" for row in roots)
    old_roots = {row["root_seed"] for rows in fixture.old_plan["roots"].values() for row in rows}
    new_roots = {row["root_seed"] for rows in runner.registered_roots(new_round).values() for row in rows}
    assert len(new_roots) == 128 and old_roots.isdisjoint(new_roots)
    assert new_round["seeds"] == cfg["fresh_split_seeds"]
    assert fixture.validated == [fixture.round_cfg]
    assert fixture.path.read_bytes() == original
    assert runner.load_effect_config(fixture.path)[2] == roots


@pytest.mark.parametrize("split", ["fit", "validation", "selection", "confirmation"])
def test_old_root_overlap_in_any_split_is_rejected(effect_config, split):
    fixture = effect_config
    fresh = deepcopy(fixture.round_cfg)
    fresh["seeds"] = fixture.cfg["fresh_split_seeds"]
    fixture.old_plan["roots"][split].append(runner.registered_roots(fresh)[split][0])
    _write_json(fixture.paths["training_plan"], fixture.old_plan)
    fixture.cfg["training_plan"] = _asset(fixture.paths["training_plan"])
    _write_json(fixture.path, fixture.cfg)
    with pytest.raises(ValueError, match="overlap"):
        runner.load_effect_config(fixture.path)


@pytest.mark.parametrize("strength", [0., .5, 3., True])
def test_strength_cannot_be_retuned_from_fixed_original_maximum(effect_config, strength):
    effect_config.cfg["strength"] = strength
    _write_json(effect_config.path, effect_config.cfg)
    with pytest.raises(ValueError):
        runner.load_effect_config(effect_config.path)


def test_incomplete_training_is_rejected_even_when_report_hash_matches(effect_config):
    fixture = effect_config
    fixture.training["final_step"] = 1999
    _write_json(fixture.paths["training_report"], fixture.training)
    fixture.cfg["training_report"] = _asset(fixture.paths["training_report"])
    _write_json(fixture.path, fixture.cfg)
    with pytest.raises(ValueError, match="training has not completed"):
        runner.load_effect_config(fixture.path)


@pytest.mark.parametrize("role", ["guide", "training_plan", "training_labels", "training_report"])
def test_changed_frozen_evidence_fails_sha_check(effect_config, role):
    effect_config.paths[role].write_bytes(b"modified synthetic evidence")
    with pytest.raises(ValueError, match="frozen " + role + " changed"):
        runner.load_effect_config(effect_config.path)


def _bind_prior_effect(fixture, pairs):
    path = fixture.path.parent / "prior-effect.json"
    _write_json(path, {"schema": runner.SCHEMA, "pairs": pairs})
    fixture.cfg["prior_effect"] = _asset(path)
    _write_json(fixture.path, fixture.cfg)
    return path


def test_optional_prior_effect_is_bound_and_copied_without_changing_fresh_roots(effect_config, tmp_path):
    fixture = effect_config
    expected = runner.load_effect_config(fixture.path)[2]
    # A historical failed pair still belongs to the excluded root inventory.
    source = _bind_prior_effect(fixture, [{"root_seed": 0, "base": {"status": "failed"},
                                          "guided": {"status": "pending"}}])
    cfg, round_cfg, roots, _ = runner.load_effect_config(fixture.path)
    assert roots == expected
    assert runner.fresh_confirmation_roots(round_cfg, fixture.old_plan) == roots
    output = tmp_path / "frozen-copy"
    (output / "frozen-assets").mkdir(parents=True)
    copied = runner.freeze_evidence(cfg, output)
    assert set(copied) == {"guide", "training_plan", "training_labels", "training_report", "prior_effect"}
    item = copied["prior_effect"]
    assert Path(item["path"]).read_bytes() == source.read_bytes()
    assert item["source_path"] == str(source)
    # This is the same complete evidence inventory used by run's final audit.
    assert all(runner.file_sha(entry["path"]) == runner.file_sha(entry["source_path"]) == entry["sha256"]
               for entry in copied.values())


@pytest.mark.parametrize("split", ["fit", "validation", "selection", "confirmation"])
def test_prior_effect_overlap_checks_all_128_new_registered_roots_including_failures(effect_config, split):
    fixture = effect_config
    fresh = deepcopy(fixture.round_cfg)
    fresh["seeds"] = fixture.cfg["fresh_split_seeds"]
    seed = runner.registered_roots(fresh)[split][0]["root_seed"]
    _bind_prior_effect(fixture, [{"root_seed": seed, "base": {"valid": False, "status": "failed"},
                                "guided": {"valid": False, "status": "pending"}}])
    with pytest.raises(ValueError, match="overlap"):
        runner.load_effect_config(fixture.path)
    with pytest.raises(ValueError, match="overlap"):
        runner.fresh_confirmation_roots(fresh, fixture.old_plan, additional_roots=(seed,))


def test_prior_effect_sha_change_is_rejected_before_using_roots(effect_config):
    path = _bind_prior_effect(effect_config, [{"root_seed": 0}])
    _write_json(path, {"schema": runner.SCHEMA, "pairs": [{"root_seed": 1}]})
    with pytest.raises(ValueError, match="frozen prior_effect changed"):
        runner.load_effect_config(effect_config.path)


@pytest.mark.parametrize("pairs", [[], [{}], [{"root_seed": True}], [{"root_seed": 1.5}],
                                   [{"root_seed": -1}], [{"root_seed": 2**63}],
                                   [{"root_seed": 0}, {"root_seed": 0}]])
def test_prior_effect_malformed_inventory_is_not_silently_ignored(effect_config, pairs):
    _bind_prior_effect(effect_config, pairs)
    with pytest.raises(ValueError, match="prior_effect"):
        runner.load_effect_config(effect_config.path)


def _draw():
    return {"root_seeds": [101], "known_prior_seeds": [202], "future_noise_seeds": [303],
            "initial_context": torch.tensor([[.1, .2]], dtype=torch.float64),
            "equation_conditions": torch.tensor([[20., 1.]], dtype=torch.float64),
            "full": torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64),
            "future_white": torch.tensor([[.5, -.5]], dtype=torch.float64)}


def test_paired_diagnostics_preserves_exact_shared_inputs_and_detects_tiny_change():
    base = _draw()
    guided = deepcopy(base)
    unchanged = runner.paired_draw_diagnostics(base, guided)
    assert unchanged["full"]["bitwise_equal"]
    assert unchanged["future_white"]["max_abs_difference"] == 0.
    epsilon = 2. ** -35  # Exactly representable and much smaller than an effect margin.
    guided["full"][0, 0] += epsilon
    changed = runner.paired_draw_diagnostics(base, guided)
    assert changed["shared_initial_context_exact"]
    assert changed["shared_conditions_exact"]
    assert changed["shared_prior_and_future_noise_seeds"]
    assert not changed["full"]["bitwise_equal"]
    assert changed["full"]["max_abs_difference"] == epsilon
    assert changed["full"]["rms_difference"] == epsilon / 2
    assert changed["full"]["l2_difference"] == epsilon
    assert changed["future_white"]["bitwise_equal"]
    assert torch.equal(base["full"], _draw()["full"])
    json.dumps(changed, allow_nan=False)


@pytest.mark.parametrize("key", ["root_seeds", "known_prior_seeds", "future_noise_seeds",
                                 "initial_context", "equation_conditions"])
def test_pair_rejects_changed_seeds_initial_state_or_conditions(key):
    base = _draw()
    guided = deepcopy(base)
    if isinstance(guided[key], list):
        guided[key][0] += 1
    else:
        guided[key][0, 0] += 2. ** -35
    with pytest.raises(ValueError, match="pair changed"):
        runner.paired_draw_diagnostics(base, guided)


@pytest.mark.parametrize("key", ["full", "future_white"])
@pytest.mark.parametrize("failure", ["shape", "nonfinite"])
def test_pair_rejects_nonfinite_or_incompatible_generated_state(key, failure):
    base = _draw()
    guided = deepcopy(base)
    if failure == "shape":
        guided[key] = guided[key][:, :-1]
    else:
        guided[key][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite and shape-compatible"):
        runner.paired_draw_diagnostics(base, guided)


def test_head_hook_does_not_change_output_or_input_gradient():
    model = torch.nn.Identity()
    original = torch.tensor([[.2, -.4], [1.2, .3]], dtype=torch.float64, requires_grad=True)
    expected_output = model(original)
    expected_gradient = torch.autograd.grad(expected_output.square().sum(), original)[0]
    observed_input = original.detach().clone().requires_grad_(True)
    diagnostic = runner.HeadDiagnostic()
    hook = model.register_forward_hook(diagnostic)
    try:
        observed_output = model(observed_input)
        observed_gradient = torch.autograd.grad(observed_output.square().sum(), observed_input)[0]
    finally:
        hook.remove()
    assert torch.equal(observed_output, expected_output)
    assert torch.equal(observed_gradient, expected_gradient)
    assert diagnostic.summary()["observed_calls"] == 1
    assert all(type(value) in (float, bool) for value in diagnostic.rows[0].values())


def test_head_hook_reports_both_saturation_directions_and_skips_nonfinite():
    diagnostic = runner.HeadDiagnostic()
    assert diagnostic.summary() == {"observed_calls": 0}
    model = torch.nn.Identity()
    hook = model.register_forward_hook(diagnostic)
    try:
        for sign in (-1., 1.):
            values = torch.tensor([[sign * 1000., -2.], [sign * 1001., 3.]], dtype=torch.float32)
            assert model(values) is values
        nonfinite = torch.tensor([[float("nan"), 0.]])
        assert model(nonfinite) is nonfinite
    finally:
        hook.remove()
    assert diagnostic.summary() == {
        "observed_calls": 2, "all_reward_exact_zero_calls": 1,
        "all_reward_exact_one_calls": 1, "reward_logit_min": -1001.,
        "reward_logit_max": 1001., "validity_logit_min": -2.,
        "validity_logit_max": 3., "predicted_reward_min": 0., "predicted_reward_max": 1.}


@pytest.mark.parametrize("job_id,scratch", [(None, None), ("123", None), ("not-a-job", "/tmp")])
def test_run_refuses_no_compute_allocation_before_loading_or_writing(tmp_path, monkeypatch, job_id, scratch):
    for key, value in (("SLURM_JOB_ID", job_id), ("SLURM_TMPDIR", scratch)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    def forbidden(*args, **kwargs):
        pytest.fail("no-allocation gate must precede config loading or subprocess work")

    monkeypatch.setattr(runner, "load_effect_config", forbidden)
    monkeypatch.setattr(runner.subprocess, "check_output", forbidden)
    output = tmp_path / "must-not-exist"
    args = SimpleNamespace(config=tmp_path / "missing.json", output=output, wall_seconds=300.)
    with pytest.raises(ValueError, match="Slurm compute allocation required"):
        runner.run(args)
    assert not output.exists()


def _exponential_config(fixture):
    reference = fixture.path.parent / "reference.pt"
    reference.write_bytes(b"synthetic frozen old guidance")
    report = fixture.path.parent / "reference-training.json"
    _write_json(report, fixture.training)
    fixture.cfg.update(guidance_objective=runner.EXPONENTIAL_OBJECTIVE, beta=2., strength=1.,
        reference_strength=2., reference_guide=_asset(reference), reference_training_report=_asset(report))
    _write_json(fixture.path, fixture.cfg)
    return reference, report


def test_exponential_config_binds_beta_in_target_and_unit_score_multiplier(effect_config, tmp_path):
    _exponential_config(effect_config)
    cfg, _, roots, _ = runner.load_effect_config(effect_config.path)
    assert cfg["beta"] == 2. and cfg["strength"] == 1. and cfg["reference_strength"] == 2.
    assert len(roots) == 32
    output = tmp_path / "exponential-copy"
    (output / "frozen-assets").mkdir(parents=True)
    copied = runner.freeze_evidence(cfg, output)
    assert {"reference_guide", "reference_training_report"} <= set(copied)
    assert all(runner.file_sha(item["path"]) == runner.file_sha(item["source_path"]) == item["sha256"]
               for item in copied.values())


@pytest.mark.parametrize("key,value", [("strength", 2.), ("beta", 1.), ("beta", True),
                                      ("reference_strength", 1.), ("reference_strength", True),
                                      ("guidance_objective", "unregistered-objective")])
def test_exponential_config_rejects_double_beta_or_unregistered_design(effect_config, key, value):
    _exponential_config(effect_config)
    effect_config.cfg[key] = value
    _write_json(effect_config.path, effect_config.cfg)
    with pytest.raises(ValueError):
        runner.load_effect_config(effect_config.path)


@pytest.mark.parametrize("key", ["reference_guide", "reference_training_report"])
def test_exponential_config_requires_both_reference_assets(effect_config, key):
    _exponential_config(effect_config)
    del effect_config.cfg[key]
    _write_json(effect_config.path, effect_config.cfg)
    with pytest.raises(ValueError, match="requires the frozen original reference"):
        runner.load_effect_config(effect_config.path)


@pytest.mark.parametrize("role", ["reference_guide", "reference_training_report"])
def test_exponential_reference_assets_are_sha_bound(effect_config, role):
    _exponential_config(effect_config)
    Path(effect_config.cfg[role]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen " + role + " changed"):
        runner.load_effect_config(effect_config.path)


@pytest.mark.parametrize("change,match", [({"final_step": 1999}, "training has not completed"),
                                        ({"label_manifest_sha256": "0" * 64}, "training labels changed")])
def test_exponential_reference_report_binds_completed_training_and_labels(effect_config, change, match):
    _, path = _exponential_config(effect_config)
    report = dict(effect_config.training, **change)
    _write_json(path, report)
    effect_config.cfg["reference_training_report"] = _asset(path)
    _write_json(effect_config.path, effect_config.cfg)
    with pytest.raises(ValueError, match=match):
        runner.load_effect_config(effect_config.path)


def _checkpoint_metadata(fixture):
    return {"label_manifest_sha256": fixture.cfg["training_labels"]["sha256"],
        "training": {"step": 2000, "label_source": "actual_public_solver_macro_error"},
        "root_partition": {key: [row["slot_id"] for row in fixture.old_plan["roots"][split]]
            for key, split in (("train_root_ids", "fit"), ("validation_root_ids", "validation"))}}


@pytest.mark.parametrize("role", ["guide", "reference guide"])
@pytest.mark.parametrize("corruption", ["step", "label", "source", "train_roots", "validation_roots", "duplicate"])
def test_both_checkpoint_metadata_bind_step_real_labels_and_exact_roots(effect_config, role, corruption):
    raw = _checkpoint_metadata(effect_config)
    assert runner.validate_checkpoint_training(raw, effect_config.training,
        effect_config.cfg["training_labels"]["sha256"], effect_config.old_plan, role=role) == 2000
    if corruption == "step":
        raw["training"]["step"] = 2001
    elif corruption == "label":
        raw["label_manifest_sha256"] = "0" * 64
    elif corruption == "source":
        raw["training"]["label_source"] = "synthetic"
    elif corruption == "duplicate":
        raw["root_partition"]["train_root_ids"].append(raw["root_partition"]["train_root_ids"][0])
    else:
        key = "train_root_ids" if corruption == "train_roots" else "validation_root_ids"
        raw["root_partition"][key][0] = "unregistered-root"
    with pytest.raises(ValueError, match=role):
        runner.validate_checkpoint_training(raw, effect_config.training,
            effect_config.cfg["training_labels"]["sha256"], effect_config.old_plan, role=role)


def test_loaded_exponential_and_old_reference_bind_objective_and_config():
    config = SimpleNamespace(to_dict=lambda: {"same_frozen_base_and_architecture": True})
    guide = SimpleNamespace(beta=2., expected_sampling_strength=1., score_potential=lambda: None, config=config)
    reference = SimpleNamespace(config=config)
    cfg = {"guidance_objective": runner.EXPONENTIAL_OBJECTIVE, "beta": 2., "strength": 1.}
    runner.validate_loaded_guidance(cfg, guide, reference)
    guide.beta = 1.
    with pytest.raises(ValueError, match="bind beta"):
        runner.validate_loaded_guidance(cfg, guide, reference)
    guide.beta = 2.
    reference.config = SimpleNamespace(to_dict=lambda: {"different": True})
    with pytest.raises(ValueError, match="same frozen guide/base config"):
        runner.validate_loaded_guidance(cfg, guide, reference)


def test_multiple_prior_reports_are_copied_and_final_auditable(effect_config, tmp_path):
    _bind_prior_effect(effect_config, [{"root_seed": 0}])
    extra = tmp_path / "second-prior.json"
    _write_json(extra, {"schema": runner.SCHEMA, "pairs": [{"root_seed": 1}]})
    effect_config.cfg["prior_effects"] = [_asset(extra)]
    _write_json(effect_config.path, effect_config.cfg)
    cfg, _, roots, _ = runner.load_effect_config(effect_config.path)
    assert len(roots) == 32
    output = tmp_path / "multiple-priors-copy"
    (output / "frozen-assets").mkdir(parents=True)
    copied = runner.freeze_evidence(cfg, output)
    assert {"prior_effect", "prior_effects_0000"} <= set(copied)
    for item in copied.values():
        assert runner.file_sha(item["path"]) == runner.file_sha(item["source_path"]) == item["sha256"]
    extra.write_bytes(b"changed second historical report")
    with pytest.raises(ValueError, match="frozen prior_effects_0000 changed"):
        runner.load_effect_config(effect_config.path)


@pytest.mark.parametrize("split", ["fit", "validation", "selection", "confirmation"])
def test_each_additional_prior_excludes_all_128_registered_roots(effect_config, tmp_path, split):
    _bind_prior_effect(effect_config, [{"root_seed": 0}])
    fresh = dict(effect_config.round_cfg, seeds=effect_config.cfg["fresh_split_seeds"])
    seed = runner.registered_roots(fresh)[split][0]["root_seed"]
    path = tmp_path / "second-prior-overlap.json"
    _write_json(path, {"schema": runner.SCHEMA, "pairs": [{"root_seed": seed, "guided": {"status": "failed"}}]})
    effect_config.cfg["prior_effects"] = [_asset(path)]
    _write_json(effect_config.path, effect_config.cfg)
    with pytest.raises(ValueError, match="overlap"):
        runner.load_effect_config(effect_config.path)


def test_three_arm_order_rotates_once_per_root_and_old_order_is_unchanged():
    assert [runner.effect_arm_order(i, with_reference=True) for i in range(3)] == [
        ("base", "control", "guided"), ("control", "guided", "base"), ("guided", "base", "control")]
    orders = [runner.effect_arm_order(i, with_reference=True) for i in range(32)]
    assert Counter(arm for order in orders for arm in order) == {"base": 32, "control": 32, "guided": 32}
    assert runner.effect_arm_order(0) == ("base", "guided")
    assert runner.effect_arm_order(1) == ("guided", "base")


def test_three_arm_summary_mapping_retains_32_roots_and_failures_without_imputation():
    from vpfp_guidance_effect_summary import summarize_pairs
    pairs = [{"root_seed": index, "stratum": index // 4,
        **{arm: {"status": "completed", "valid": True, "error": error, "attempt_count": 1}
           for arm, error in (("base", 1.), ("control", 2.), ("guided", 3.))}}
        for index in range(32)]
    original = deepcopy(pairs)
    reference = runner.remap_comparison_pairs(pairs, baseline_arm="base", guided_arm="control")
    new = runner.remap_comparison_pairs(pairs, baseline_arm="control", guided_arm="guided")
    assert len(reference) == len(new) == 32
    for rows, baseline, guided in ((reference, 1., 2.), (new, 2., 3.)):
        summary = summarize_pairs(rows, bootstrap_seed=15, bootstrap_replicates=100)
        assert summary["paired"]["base_mean_error"] == baseline
        assert summary["paired"]["guided_mean_error"] == guided
    reference[0]["guided"]["valid"] = False
    assert pairs == original
    pairs[0]["control"] = {"status": "failed", "valid": False, "error": None, "attempt_count": 1}
    failed = runner.remap_comparison_pairs(pairs, baseline_arm="control", guided_arm="guided")
    assert failed[0]["base"]["error"] is None and failed[0]["base"]["status"] == "failed"
    summary = summarize_pairs(failed, bootstrap_seed=15, bootstrap_replicates=100)
    assert summary["paired"]["paired_valid_roots"] == 31


def test_log_desirability_hook_observes_forward_dictionary_without_sigmoid_or_gradient_change():
    class Guide(torch.nn.Module):
        def forward(self, value):
            return {"log_desirability": value[:, 0], "desirability": value[:, 0].exp(),
                    "valid_probability": value[:, 1].sigmoid()}
    model = Guide()
    values = torch.tensor([[2., .2], [-3., -.4]], dtype=torch.float64, requires_grad=True)
    expected = torch.autograd.grad(model(values)["log_desirability"].sum(), values)[0]
    diagnostic = runner.HeadDiagnostic(kind="log_desirability")
    hook = model.register_forward_hook(diagnostic)
    try:
        observed = model(values)
        actual = torch.autograd.grad(observed["log_desirability"].sum(), values)[0]
    finally:
        hook.remove()
    assert torch.equal(actual, expected)
    summary = diagnostic.summary()
    assert summary["kind"] == "log_desirability" and summary["observed_calls"] == 1
    assert summary["log_desirability_min"] == -3. and summary["log_desirability_max"] == 2.
    assert summary["desirability_max"] == pytest.approx(torch.exp(torch.tensor(2., dtype=torch.float64)).item())
    assert summary["log_desirability_mean_over_calls"] == -.5
    assert not any("reward" in key for key in summary)
