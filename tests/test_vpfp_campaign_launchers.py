"""Static and mocked shell checks; never run numerical work or a scheduler."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = "${PDE_GAV_WORKTREES_ROOT}/vpfp-moving-frame-campaign"
SCRATCH = "${PDE_GAV_RUN_ROOT}"
PREFLIGHT_SHA = "5f76d5781691aa40fdf8c6b1a43b3b5c221821cab09fa459ec7caef9be09110e"
MECHANICS_SHA = "0800081e703fef70bbb5eb4078854a39dca9b390e9b4c6591f43703fe8a31399"
SELECTION_SHA = "93e125134a9b044c140ae44e12eb4bb0d429b45a40d4228be2c033d9b8033b30"
WRAPPERS = {"strong": "run_vpfp_strong_compare.slurm", "audit": "run_vpfp_regime_audit_screen.slurm", "weak": "run_vpfp_weak_compare.slurm"}
LABELS = {"strong": "vpfp-strong-regime-coherent", "audit": "vpfp-regime-audit-screen", "weak": "vpfp-weak-regime-coherent"}
MOVING_TESTS = [f"tests/test_vpfp_moving_frame{name}.py" for name in ("", "_audit", "_backend", "_binding", "_smoke")]

MOCK = r'''
import hashlib,json,os,sys,subprocess
from pathlib import Path
name=Path(sys.argv[0]).name
args=sys.argv[1:]
with open(os.environ["MOCK_EVENTS"],"a") as log:
    log.write(json.dumps({"command":name,"args":args})+"\n")
def code(key): return int(os.environ.get(key,"0"))
if name=="git":
    if args[0]=="rev-parse":
        print("f"*40)
        sys.exit(code("MOCK_COMMIT_EXIT"))
    if args[0]=="ls-files": sys.exit(code("MOCK_TRACKED_EXIT"))
    if args[0]=="status":
        print(os.environ.get("MOCK_DIRTY",""),end="")
        sys.exit(code("MOCK_STATUS_EXIT"))
    sys.exit(91)
if name=="module": sys.exit(1)
if name=="sha256sum":
    if code("MOCK_HASH_EXIT"): sys.exit(code("MOCK_HASH_EXIT"))
    sys.exit(subprocess.call(["/usr/bin/sha256sum",*args]))
if name=="rg":
    assert args==["--files","tests","-g","test_vpfp_*moving_frame*.py"]
    print("\n".join(json.loads(os.environ["MOCK_MOVING_TESTS"])))
    sys.exit(0)
if name=="timeout":
    assert args[:3]==["--signal=TERM","--kill-after=30s","900"]
    sys.exit(subprocess.call(args[3:]))
if name=="python":
    if args[:2]==["-m","pytest"]: sys.exit(code("MOCK_TEST_EXIT"))
    if args and args[0]=="-c":
        if "torch.cuda" in args[1]: sys.exit(code("MOCK_GPU_EXIT"))
        # Exercise the real stdlib-only frozen receipt validator, not Torch.
        sys.exit(subprocess.call([sys.executable,*args]))
    output=Path(args[args.index("--output")+1])
    output.mkdir(parents=True)
    (output/"mock-retained-artifact.txt").write_text("synthetic only")
    if args[0]=="scripts/run_vpfp_strong_regime_coherent.py": sys.exit(code("MOCK_STRONG_EXIT"))
    if args[0]=="scripts/run_vpfp_weak_coherent.py":
        if code("MOCK_WEAK_TRAIN_EXIT"): sys.exit(code("MOCK_WEAK_TRAIN_EXIT"))
        entries=[]
        for i in (0,1):
            directory=output/f"seed-{i}"
            directory.mkdir()
            manifest=directory/"native-moving-frame-model.json"
            manifest.write_text(json.dumps({"synthetic_seed":i}))
            checkpoint=directory/"W0/checkpoints/fixed-step-020000.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("synthetic checkpoint")
            entries.append({"native_A_asset":{"manifest":str(manifest),"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()},
                "W0":{"checkpoint":str(checkpoint),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}})
        (output/"result.json").write_text(json.dumps({"execution_passed":True,"training":entries}))
        if code("MOCK_TAMPER_A0"):
            (output/"seed-0/native-moving-frame-model.json").write_text("changed")
        sys.exit(0)
    if args[0]=="scripts/run_vpfp_moving_frame_smoke.py":
        sys.exit(code("MOCK_A_SMOKE_0_EXIT" if "seed-0" in args[args.index("--manifest")+1] else "MOCK_A_SMOKE_1_EXIT"))
    if args[0]=="scripts/run_plasma_canonical_smoke.py":
        sys.exit(code("MOCK_W0_SMOKE_0_EXIT" if "seed-0" in args[args.index("--diffusion-checkpoint")+1] else "MOCK_W0_SMOKE_1_EXIT"))
    assert args[0]=="scripts/run_vpfp_regime_fidelity_coverage.py"
    source=args[args.index("--source")+1]
    sys.exit(code("MOCK_STRONG_EXIT" if "strong-main" in source else "MOCK_WEAK_EXIT"))
raise RuntimeError("unexpected executable: "+name)
'''


@pytest.fixture
def launch(tmp_path):
    checkout, scratch, commands = (tmp_path/name for name in ("checkout", "scratch", "commands"))
    for directory in (checkout/"scripts", checkout/"configs", checkout/"tests", scratch, commands, tmp_path/"local"):
        directory.mkdir(parents=True)
    for name in ("run_vpfp_strong_regime_coherent.py", "run_vpfp_regime_fidelity_coverage.py",
            "run_vpfp_weak_coherent.py", "run_vpfp_moving_frame_smoke.py", "run_plasma_canonical_smoke.py"):
        (checkout/"scripts"/name).write_text("# synthetic presence only")
    (checkout/"configs/vpfp_unified_live_strong_collision.json").write_text("{}")
    (checkout/"configs/vpfp_unified_live_weak_collision.json").write_text("{}")
    for root in ("vpfp-strong-main-5244223", "vpfp-scaleup-weak-5241246"):
        (scratch/root/"corpus").mkdir(parents=True)
        for name in ("corpus/manifest.json", "result.json", "frozen-reference-features.json"):
            (scratch/root/name).write_text("{}")
    for root, arms, tail in (("vpfp-strong-initial-5259887", ("cov_base", "initial_conditioned"), "evaluation"),
            ("vpfp-hermite-joint-recovery-5258664", ("hermite_joint", "hermite_permuted_control"), "evaluation/best")):
        for replica in (0, 1):
            for arm in arms:
                directory = scratch/root/f"seed-{replica}"/arm/tail
                directory.mkdir(parents=True)
                for name in ("generated-coefficients.npy", "generated-conditions.npy", "generated-provenance.json"):
                    (directory/name).write_text("synthetic placeholder, never loaded")
        for name in ("result.json", "frozen-plan.json", "frozen-checkpoints.json"):
            (scratch/root/name).write_text("{}")
    preflight = scratch/"vpfp-regime-coherent-preflight-5267643/strong-preflight/result.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(json.dumps({"execution_passed": True, "preflight_only": True, "training": {}, "audits": {},
        "preflight": {"initial_conditioned_rho_cov": {"execution_passed": True, "normalization_gate": {"passed": True}}}}))
    weak_preflight = tmp_path/"completed-weak-preflight.json"
    weak_preflight.write_text(json.dumps({"numerical_mechanics_passed": True, "source_manifest_sha256": "c"*64}))
    selection = checkout/"configs/vpfp_weak_coherent_representation_selection_5267643.json"
    selection.write_text(json.dumps({"schema": "vpfp_moving_frame_representation_selection_v1", "mechanics_passed": True,
        "selected_mode": 6, "selected_fit_grid": 33, "preflight_result": str(weak_preflight),
        "preflight_result_sha256": sha256(weak_preflight.read_bytes()).hexdigest(), "corpus_manifest_sha256": "c"*64}))
    mechanics_log = tmp_path/"completed-mechanics.log"
    mechanics_log.write_text("synthetic completed mechanics only")
    mechanics = tmp_path/"root-issued-frozen-mechanics.json"
    mechanics.write_text(json.dumps({"schema": "vpfp_completed_mechanics_receipt_v1", "stage": "mechanics_tests", "exit_code": 0,
        "command": ["python", "-m", "pytest", "tests/test_vpfp_regime_fidelity_coverage.py", "tests/test_vpfp_regime_fidelity_coverage_cli.py"],
        "log": str(mechanics_log), "log_sha256": sha256(mechanics_log.read_bytes()).hexdigest(),
        "overall_preflight_execution_passed_at_capture": False}))
    (checkout/"configs/vpfp_regime_mechanics_receipt_5267643.json").write_text(mechanics.read_text())
    files = ["test_vpfp_coherent_diffusion.py", "test_vpfp_training_resume.py", "test_vpfp_training_snapshots.py",
        "test_vpfp_conditional_rho_covariance_gate.py", "test_vpfp_numerical_roundtrip.py", "test_vpfp_regime_fidelity_coverage.py",
        "test_vpfp_regime_fidelity_coverage_cli.py", "test_vpfp_weak_coherent_driver.py", "test_vpfp_strong_regime_coherent.py",
        "test_vpfp_campaign_launchers.py"]
    for name in [*("tests/"+name for name in files), *MOVING_TESTS]:
        (checkout/name).write_text("# mock pytest must not execute any test")
    runtime = commands/"mock-runtime"
    runtime.write_text(f"#!{sys.executable}\n"+MOCK)
    runtime.chmod(0o700)
    for command in ("python", "git", "module", "sha256sum", "timeout", "rg", "sbatch", "srun", "scancel", "scontrol"):
        (commands/command).symlink_to(runtime)
    for name in WRAPPERS.values():
        text = (ROOT/"scripts"/name).read_text()
        for old, new in ((CHECKOUT, str(checkout)), (SCRATCH, str(scratch)),
                ("${PDE_GAV_PYTHON}", str(commands/"python")),
                (PREFLIGHT_SHA, sha256(preflight.read_bytes()).hexdigest()),
                (MECHANICS_SHA, sha256(mechanics.read_bytes()).hexdigest()),
                (SELECTION_SHA, sha256(selection.read_bytes()).hexdigest())):
            text = text.replace(old, new)
        (checkout/"scripts"/name).write_text(text)
    events_path = tmp_path/"events.jsonl"
    env = {key: value for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "MOCK_", "BASH_FUNC_", "VPFP_AUDIT_")) and key != "BASH_ENV"}
    env.update(PATH=f"{commands}:/usr/bin:/bin", SLURM_JOB_ID="123", SLURM_TMPDIR=str(tmp_path/"local"),
        MOCK_EVENTS=str(events_path), MOCK_MOVING_TESTS=json.dumps(MOVING_TESTS),
        VPFP_AUDIT_MECHANICS_RESULT=str(mechanics), VPFP_AUDIT_MECHANICS_SHA256=sha256(mechanics.read_bytes()).hexdigest())
    def events():
        return [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    def run(kind, **overrides):
        selected = {key: value for key, value in {**env, **overrides}.items() if value is not None}
        result = subprocess.run(["bash", str(checkout/"scripts"/WRAPPERS[kind])], env=selected,
            capture_output=True, text=True, timeout=30, check=False)
        assert not any(row["command"] in ("sbatch", "srun", "scancel", "scontrol") for row in events())
        return result
    def receipt(kind):
        return dict(line.split("\t") for line in (scratch/f"{LABELS[kind]}-123.launch-exit-codes.tsv").read_text().splitlines()[1:])
    return SimpleNamespace(run=run, events=events, receipt=receipt, scratch=scratch, checkout=checkout,
        mechanics=mechanics, mechanics_log=mechanics_log, preflight=preflight)


@pytest.mark.parametrize("kind,hours", [("audit", "07"), ("strong", "08"), ("weak", "20")])
def test_static_single_gpu_fixed_budget_and_no_child_submission(kind, hours):
    path = ROOT/"scripts"/WRAPPERS[kind]
    text = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)
    for value in ("--partition=gpubase_l40s_b3", "--gres=gpu:l40s:1", "--cpus-per-task=8", "--mem=64G", f"--time={hours}:00:00", CHECKOUT):
        assert value in text
    assert not re.search(r"^\s*(?:sbatch|srun|scancel|scontrol)\b", text, re.M)
    assert "timeout --signal=TERM --kill-after=30s 900" in text
    assert "git ls-files --error-unmatch" in text and "HEAD^{commit}" in text
    if kind == "strong":
        assert PREFLIGHT_SHA in text
        assert "5267643/strong-preflight/result.json" in text
        assert "5267643/result.json" not in text
    elif kind == "audit":
        assert MECHANICS_SHA in text
        assert "configs/vpfp_regime_mechanics_receipt_5267643.json" in text
        assert "5267643/result.json" not in text
    else:
        assert SELECTION_SHA in text
        assert "configs/vpfp_weak_coherent_representation_selection_5267643.json" in text


def python_calls(launch, prefix):
    return [row["args"] for row in launch.events() if row["command"] == "python" and row["args"][0] == prefix]


def test_audit_exactly_two_calls_four_explicit_targets_each_and_tests_first(launch):
    outcome = launch.run("audit")
    assert outcome.returncode == 0, outcome.stderr
    audits = python_calls(launch, "scripts/run_vpfp_regime_fidelity_coverage.py")
    assert len(audits) == 2
    for call, token in zip(audits, ("strong-initial-5259887", "hermite-joint-recovery-5258664")):
        assert call.count("--generated-dir") == 4
        targets = [call[index+1] for index, value in enumerate(call) if value == "--generated-dir"]
        assert len(set(targets)) == 4 and all(token in target for target in targets)
        assert call[call.index("--sample-count")+1] == "128"
    tests, = python_calls(launch, "-m")
    assert all(name in tests for name in MOVING_TESTS)
    sequence = [row["args"][0] for row in launch.events() if row["command"] == "python"]
    assert sequence.index("-m") < sequence.index("scripts/run_vpfp_regime_fidelity_coverage.py")
    assert "overall preflight status at capture = False" in outcome.stdout
    assert launch.receipt("audit")["launch"] == "0"


def test_strong_binds_only_passed_substage_and_runs_four_tests_before_six_run_driver(launch):
    outcome = launch.run("strong")
    assert outcome.returncode == 0, outcome.stderr
    calls = python_calls(launch, "scripts/run_vpfp_strong_regime_coherent.py")
    assert len(calls) == 1 and "--preflight-only" not in calls[0]
    assert calls[0][calls[0].index("--pilot")+1].endswith("strong-initial-5259887")
    tests, = python_calls(launch, "-m")
    assert sum(value.startswith("tests/") for value in tests) == 4
    assert launch.receipt("strong")["mechanics_tests"] == "0"


@pytest.mark.parametrize("kind", ["strong", "audit", "weak"])
@pytest.mark.parametrize("override,code", [({"MOCK_GPU_EXIT": "17"}, 17), ({"MOCK_TEST_EXIT": "23"}, 23),
    ({"MOCK_TEST_EXIT": "124"}, 124), ({"MOCK_HASH_EXIT": "13"}, 13), ({"MOCK_DIRTY": " M src/changed.py\n"}, 2)])
def test_failures_preserve_exit_and_never_reach_numeric_driver(launch, kind, override, code):
    outcome = launch.run(kind, **override)
    assert outcome.returncode == code
    assert not python_calls(launch, "scripts/run_vpfp_strong_regime_coherent.py")
    assert not python_calls(launch, "scripts/run_vpfp_regime_fidelity_coverage.py")
    assert not python_calls(launch, "scripts/run_vpfp_weak_coherent.py")
    assert launch.receipt(kind)["launch"] == str(code)


def test_audit_keeps_independent_second_regime_and_first_failure_reason(launch):
    outcome = launch.run("audit", MOCK_STRONG_EXIT="29")
    assert outcome.returncode == 29
    assert len(python_calls(launch, "scripts/run_vpfp_regime_fidelity_coverage.py")) == 2
    receipt = launch.receipt("audit")
    assert receipt["strong_audit"] == "29" and receipt["weak_audit"] == "0"
    assert (launch.scratch/"vpfp-regime-audit-screen-123/strong/mock-retained-artifact.txt").exists()


def test_changed_bound_mechanics_log_is_rejected(launch):
    launch.mechanics_log.write_text("changed")
    outcome = launch.run("audit")
    assert outcome.returncode != 0
    assert not python_calls(launch, "-m")
    assert launch.receipt("audit")["mechanics_binding"] != "0"


def test_default_binding_is_exact_frozen_receipt_not_overall_preflight(launch):
    outcome = launch.run("audit", VPFP_AUDIT_MECHANICS_RESULT=None, VPFP_AUDIT_MECHANICS_SHA256=None)
    assert outcome.returncode == 0, outcome.stderr
    validators = [call for call in python_calls(launch, "-c") if "vpfp_completed_mechanics_receipt_v1" in call[1]]
    assert len(validators) == 1
    assert validators[0][-1] == str(launch.checkout/"configs/vpfp_regime_mechanics_receipt_5267643.json")


def test_invalid_explicit_binding_never_auto_selects_overall_preflight(launch):
    assert launch.run("audit", VPFP_AUDIT_MECHANICS_SHA256="bad").returncode == 2
    assert not launch.events()


@pytest.mark.parametrize("kind", ["strong", "audit", "weak"])
def test_no_allocation_blocks_before_external_commands(launch, kind):
    assert launch.run(kind, SLURM_JOB_ID=None).returncode == 2
    assert not launch.events()


@pytest.mark.parametrize("kind", ["strong", "audit", "weak"])
def test_existing_artifacts_are_never_overwritten(launch, kind):
    path = launch.scratch/f"{LABELS[kind]}-123.mechanics-tests.log"
    path.write_text("retain this previous log")
    assert launch.run(kind).returncode == 2
    assert path.read_text() == "retain this previous log" and not launch.events()


def test_weak_registered_training_then_one_four_directory_audit_then_four_independent_smokes(launch):
    outcome = launch.run("weak")
    assert outcome.returncode == 0, outcome.stderr
    train, = python_calls(launch,"scripts/run_vpfp_weak_coherent.py")
    for flag,value in (("--replicas","2"),("--steps","20000"),("--batch-size","4"),
            ("--generated-answers","128"),("--sample-steps","64"),("--train-seconds","7200")):
        assert train[train.index(flag)+1] == value
    audits, = python_calls(launch,"scripts/run_vpfp_regime_fidelity_coverage.py")
    targets = [audits[i+1] for i,arg in enumerate(audits) if arg == "--generated-dir"]
    root = launch.scratch/"vpfp-weak-regime-coherent-123"
    assert targets == [str(root/f"seed-{i}"/arm) for i in (0,1) for arm in ("W0","paired_AB")]
    assert audits[audits.index("--output")+1] == str(root)+"-audit"
    assert audits[audits.index("--sample-count")+1] == "128"
    a_smokes = python_calls(launch,"scripts/run_vpfp_moving_frame_smoke.py")
    w0_smokes = python_calls(launch,"scripts/run_plasma_canonical_smoke.py")
    assert len(a_smokes) == len(w0_smokes) == 2
    for i,(a,w0) in enumerate(zip(a_smokes,w0_smokes)):
        manifest = root/f"seed-{i}/native-moving-frame-model.json"
        checkpoint = root/f"seed-{i}/W0/checkpoints/fixed-step-020000.pt"
        assert a[a.index("--manifest")+1] == str(manifest)
        assert a[a.index("--manifest-sha256")+1] == sha256(manifest.read_bytes()).hexdigest()
        assert w0[w0.index("--physics")+1] == "vpfp"
        assert w0[w0.index("--diffusion-checkpoint")+1] == str(checkpoint)
        assert w0[w0.index("--diffusion-checkpoint-sha256")+1] == sha256(checkpoint.read_bytes()).hexdigest()
    sequence = [row["args"][0] for row in launch.events() if row["command"] == "python"]
    assert sequence.index("-m") < sequence.index(train[0]) < sequence.index(audits[0]) < sequence.index(a_smokes[0][0])
    tests, = python_calls(launch,"-m")
    assert all(name in tests for name in MOVING_TESTS)
    receipt = launch.receipt("weak")
    assert all(receipt[key] == "0" for key in ("launch","native_A_seed_0","native_A_seed_1","native_W0_seed_0","native_W0_seed_1"))


@pytest.mark.parametrize("override,code", [({"MOCK_WEAK_EXIT":"29"},29),({"MOCK_A_SMOKE_0_EXIT":"31"},31),
    ({"MOCK_W0_SMOKE_0_EXIT":"37"},37)])
def test_weak_audit_or_smoke_failure_keeps_all_independent_checks(launch,override,code):
    outcome = launch.run("weak",**override)
    assert outcome.returncode == code
    assert len(python_calls(launch,"scripts/run_vpfp_moving_frame_smoke.py")) == 2
    assert len(python_calls(launch,"scripts/run_plasma_canonical_smoke.py")) == 2
    assert launch.receipt("weak")["launch"] == str(code)


def test_weak_training_failure_does_not_audit_unfrozen_assets(launch):
    assert launch.run("weak",MOCK_WEAK_TRAIN_EXIT="41").returncode == 41
    assert not python_calls(launch,"scripts/run_vpfp_regime_fidelity_coverage.py")
    assert not python_calls(launch,"scripts/run_vpfp_moving_frame_smoke.py")
    assert launch.receipt("weak")["weak_comparisons"] == "41"


def test_weak_changed_seed0_manifest_does_not_replace_it_or_skip_other_smokes(launch):
    assert launch.run("weak",MOCK_TAMPER_A0="1").returncode != 0
    assert len(python_calls(launch,"scripts/run_vpfp_moving_frame_smoke.py")) == 1
    assert len(python_calls(launch,"scripts/run_plasma_canonical_smoke.py")) == 2
    receipt = launch.receipt("weak")
    assert receipt["native_A_seed_0"] != "0" and receipt["native_A_seed_1"] == "0"
