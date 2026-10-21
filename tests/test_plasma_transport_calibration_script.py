"""Synthetic CLI/packing tests only; these are not real-data calibration evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import calibrate_plasma_transport as script


def test_collection_refuses_login_node_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute allocation"):
        script.collect("weak", tmp_path/"absent.json", tmp_path/"result")
    assert not (tmp_path/"result").exists()


def test_fixed_budget_seed_namespace_and_native_eight_strata():
    assert script.COUNTS == {"strong": 64, "weak": 64, "artemis": 32}
    assert script.POINT_COUNT == 128
    assert script.MASTER_SEED == 202610031701
    seeds = {script._seed(family, i, kind) for family, count in script.COUNTS.items()
             for i in range(count) for kind in ("draw", "points")}
    assert len(seeds) == 320
    ranges = {"collision_nu_range": (.01, .05), "temperature_range": (.8, 1.2)}
    bins = {script._stratum((.01+(i+.5)*.01, .8+(j+.5)*.2), ranges)
            for i in range(4) for j in range(2)}
    assert bins == script._expected_strata("weak")
    assert script._stratum((.05, 1.2), ranges) == "nu3_T1"
    with pytest.raises(ValueError, match="leaves"):
        script._stratum((.5, 1.), ranges)


def test_ranges_must_come_from_real_decoder_and_agree_with_sampler():
    cfg = SimpleNamespace(collision_nu_range=(10., 30.), temperature_range=(.8, 1.2))
    runtime = SimpleNamespace(expanded_model=SimpleNamespace(config=cfg))
    backend = SimpleNamespace(capture=SimpleNamespace(base=SimpleNamespace(config=cfg)))
    assert script._condition_ranges(runtime, backend)["collision_nu_range"] == (10., 30.)
    with pytest.raises(ValueError, match="cannot locate"):
        script._condition_ranges(SimpleNamespace(), backend)
    backend.capture.base.config = SimpleNamespace(collision_nu_range=(.01, .05), temperature_range=(.8, 1.2))
    with pytest.raises(ValueError, match="differ"):
        script._condition_ranges(runtime, backend)


def test_uniform_points_and_packed_tensor_preserve_correct_physics():
    case = SimpleNamespace(public_data={"question": {"time_domain": (2., 5.), "grid": {"x_domain": (3., 7.)}}})
    p = script._uniform_points(case, "weak", 112)
    np.testing.assert_array_equal(p, script._uniform_points(case, "weak", 112))
    assert p.shape == (128, 3) and np.all(p[:, 0] >= 2.) and np.all(p < [5., 3., 7.])
    values = {"rho": np.ones(128), "electric": np.ones((128, 2)), "momentum": np.ones((128, 2)),
              "pressure": np.tile([1., .3, 2.], (128, 1)), "heat_flux": np.ones((128, 2))}
    canonical = script._canonical_vpfp(values)
    np.testing.assert_allclose(np.sum(canonical["pressure"]**2, axis=1), 1.+2*.3**2+4.)


def _synthetic_collections(root):
    directories = []
    for family, count in script.COUNTS.items():
        directory = root/family
        directory.mkdir()
        directories.append(directory)
        strata = sorted(script._expected_strata(family))
        ranges = {} if family == "artemis" else {"collision_nu_range": [10., 30.], "temperature_range": [.8, 1.2]}
        records = []
        for index in range(count):
            stratum = strata[index % len(strata)]
            d = 1 if family == "artemis" else 2
            values = {"density": np.full((128, 1), 1.1), "electric_field": np.full((128, d), .2),
                      "particle_flux": np.full((128, d), .3), "pressure": np.full((128, d*d), .4),
                      "heat_flux": np.full((128, d), .05)}
            condition = [] if family == "artemis" else [10.+(int(stratum[2])+.5)*5., .8+(int(stratum[-1])+.5)*.2]
            asset = directory/f"reference-{index:03d}.npz"
            np.savez(asset, points=np.ones((128, d+1)), condition=condition, **values)
            records.append({"index": index, "status": "passed", "released": True,
                "request_seed": script._seed(family, index, "draw"), "point_seed": script._seed(family, index, "points"),
                "root_id": f"{family}:{index}", "native_root_seed": index, "stratum": stratum,
                "sample_count": 128, "condition": condition, "asset": asset.name, "asset_sha256": script._sha(asset)})
        payload = {"schema": script.SCHEMA, "family": family, "status": "passed", "master_seed": script.MASTER_SEED,
            "source_role": "independent_reference_calibration", "points_per_case": 128, "units": script.UNITS,
            "expected_case_count": count, "candidate_data_used": False, "failed_draw_replacement": False,
            "condition_ranges": ranges, "config_sha256": "synthetic-test-only", "collection_script_sha256": "synthetic-test-only",
            "records": records}
        (directory/"collection.json").write_text(json.dumps(payload))
    return directories


def test_fit_requires_complete_hash_checked_fixed_sample_budget(tmp_path):
    directories = _synthetic_collections(tmp_path)
    target = tmp_path/"calibration.json"
    result = script.fit(directories, target)
    assert result.to_dict()["floors"]["density"] == pytest.approx(1.1)
    assert result.to_dict()["calibration"]["families"]["strong"]["case_count"] == 64
    assert json.loads(target.read_text())["objective_hash"] == result.objective_hash
    with pytest.raises(FileExistsError):
        script.fit(directories, target)
    manifest = directories[0]/"collection.json"
    value = json.loads(manifest.read_text())
    value["records"][0]["status"] = "failed"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="failed"):
        script.fit(directories, tmp_path/"must-not-exist.json")
    assert not (tmp_path/"must-not-exist.json").exists()


def test_collection_keeps_failed_attempt_without_replacement_and_releases_all(tmp_path, monkeypatch):
    from scripts import plasma_gpu_campaign
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda: "synthetic-test-only", synchronize=lambda: None))
    monkeypatch.setattr(script, "_require_compute", lambda: fake_torch)
    cfg = SimpleNamespace(collision_nu_range=(10., 30.), temperature_range=(.8, 1.2))
    runtime = SimpleNamespace(expanded_model=SimpleNamespace(config=cfg))
    backend = SimpleNamespace(runtime=runtime, _draws={}, capture=SimpleNamespace(base=SimpleNamespace(config=cfg)))
    released, requests = [], []
    def sample(state, request):
        i = len(requests)
        requests.append(request)
        case = SimpleNamespace(case_id=f"test-{i}", public_data={"question": {
            "time_domain": (0., 1.), "grid": {"x_domain": (1., 1.)},
            "collision_nu": 10.+(i % 4+.5)*5., "temperature": .8+(i//4 % 2+.5)*.2}})
        backend._draws[case.case_id] = {"root_seed": i}
        return (SimpleNamespace(case=case),)
    def values(backend, runtime, record, family, points):
        if record.case.case_id == "test-3":
            raise FloatingPointError("synthetic injected failure")
        return {"density": np.ones((128, 1)), "electric_field": np.ones((128, 2)),
                "particle_flux": np.ones((128, 2)), "pressure": np.ones((128, 4)), "heat_flux": np.ones((128, 2))}
    generator = SimpleNamespace(_backend=lambda: backend, load=lambda *args: "state", sample=sample,
                                release=lambda state, ids: released.extend(ids))
    binding = SimpleNamespace(report={"test_only": True}, checkpoint="fake", plugin=SimpleNamespace(generator=generator))
    monkeypatch.setattr(plasma_gpu_campaign, "_binding", lambda *args: (binding, {}))
    monkeypatch.setattr(script, "_reference_values", values)
    config = tmp_path/"config.json"
    config.write_text(json.dumps({"schema": "pde_gav_plasma_gpu_v1", "equation": "strong", "backend": {"device": "cuda"}}))
    output = tmp_path/"collection"
    with pytest.raises(RuntimeError, match="incomplete"):
        script.collect("strong", config, output)
    report = json.loads((output/"collection.json").read_text())
    assert len(requests) == len(released) == len(report["records"]) == 64
    assert report["records"][3]["status"] == "failed"
    assert report["records"][3]["error"] == "synthetic injected failure"
    assert report["records"][3]["released"] is True
    assert report["successful_case_count"] == 63 and report["status"] == "failed"
