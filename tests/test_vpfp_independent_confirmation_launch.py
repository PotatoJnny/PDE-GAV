"""Only mocked shell processes and synthetic files; no scheduler/numeric work."""
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
CHECKOUT = "${PDE_GAV_WORKTREES_ROOT}/vpfp-independent-regime-confirmation"
SCRATCH = "${PDE_GAV_RUN_ROOT}"
SCRIPT = "scripts/run_vpfp_independent_confirmation.slurm"
MOVING = [f"tests/test_vpfp_moving_frame{suffix}.py" for suffix in ("","_audit","_backend","_binding","_smoke")]
TESTS = ["test_vpfp_independent_physics.py","test_vpfp_physical_controls.py","test_vpfp_diffusion_data.py",
    "test_vpfp_reference_refinement.py","test_vpfp_independent_physics_cli.py",
    "test_vpfp_independent_confirmation.py","test_vpfp_regime_fidelity_coverage.py",
    "test_vpfp_regime_fidelity_coverage_cli.py","test_vpfp_independent_confirmation_launch.py"]
MOCK = r'''
import json,os,sys,subprocess
from pathlib import Path
name=Path(sys.argv[0]).name; args=sys.argv[1:]
with open(os.environ["MOCK_EVENTS"],"a") as log:
    log.write(json.dumps({"command":name,"args":args})+"\n")
def code(key): return int(os.environ.get(key,"0"))
if name=="git":
    if args[0]=="rev-parse": print("e"*40); sys.exit(0)
    if args[0]=="ls-files": sys.exit(code("MOCK_TRACKED_EXIT"))
    if args[0]=="status": print(os.environ.get("MOCK_DIRTY",""),end=""); sys.exit(0)
if name=="module": sys.exit(1)
if name=="sha256sum":
    if code("MOCK_HASH_EXIT"): sys.exit(code("MOCK_HASH_EXIT"))
    sys.exit(subprocess.call(["/usr/bin/sha256sum",*args]))
if name=="rg":
    assert args==["--files","tests","-g","test_vpfp_*moving_frame*.py"]
    print("\n".join(json.loads(os.environ["MOCK_MOVING"]))); sys.exit(0)
if name=="timeout":
    assert args[:2]==["--signal=TERM","--kill-after=15s"]
    assert 0<int(args[2])<=39600
    sys.exit(subprocess.call(args[3:]))
if name=="python":
    if args[:2]==["-m","pytest"]: sys.exit(code("MOCK_TEST_EXIT"))
    if args and args[0]=="-c":
        if "torch.cuda" in args[1]: sys.exit(code("MOCK_GPU_EXIT"))
        sys.exit(subprocess.call([sys.executable,*args]))
    assert args[0]=="scripts/run_vpfp_independent_confirmation.py"
    if "--help" in args: print("synthetic entrypoint --selection --selection-sha256 --wall-seconds"); sys.exit(code("MOCK_HELP_EXIT"))
    output=Path(args[args.index("--output")+1]); output.mkdir(parents=True)
    (output/"partial-model-generation.json").write_text("synthetic retained partial only")
    sys.exit(code("MOCK_CONFIRM_EXIT"))
raise RuntimeError("unexpected executable: "+name)
'''


@pytest.fixture
def launch(tmp_path):
    checkout,scratch,commands,local = [tmp_path/name for name in ("checkout","scratch","commands","local")]
    for directory in (checkout/"scripts",checkout/"tests",scratch,commands,local): directory.mkdir(parents=True)
    for name in [*("tests/"+name for name in TESTS),*MOVING]: (checkout/name).write_text("# synthetic presence only")
    runtime = commands/"mock-runtime"
    runtime.write_text(f"#!{sys.executable}\n"+MOCK); runtime.chmod(0o700)
    for name in ("python","git","module","sha256sum","rg","timeout","sbatch","srun","scancel","scontrol"):
        (commands/name).symlink_to(runtime)
    text = (ROOT/SCRIPT).read_text().replace(CHECKOUT,str(checkout)).replace(SCRATCH,str(scratch)).replace(
        "${PDE_GAV_PYTHON}",str(commands/"python"))
    (checkout/SCRIPT).write_text(text)
    selection = tmp_path/"reviewed-selection.json"
    selection.write_text(json.dumps({"schema":"vpfp_independent_confirmation_selection_v1","scientific_admission":False,
        "review":"Synthetic explicit launcher fixture only","candidates":[{"id":"synthetic"}]}))
    selected = {"VPFP_CONFIRMATION_SELECTION":str(selection),"VPFP_CONFIRMATION_SELECTION_SHA256":sha256(selection.read_bytes()).hexdigest(),
        "VPFP_CONFIRMATION_SEED":"20260908"}
    events = tmp_path/"events.jsonl"
    environment = {key:value for key,value in os.environ.items()
        if not key.startswith(("SLURM_","VPFP_CONFIRMATION_","MOCK_","BASH_FUNC_")) and key!="BASH_ENV"}
    environment.update(PATH=f"{commands}:/usr/bin:/bin",SLURM_JOB_ID="123",SLURM_TMPDIR=str(local),
        MOCK_EVENTS=str(events),MOCK_MOVING=json.dumps(MOVING))
    output = scratch/"vpfp-independent-confirmation-123"
    def read_events(): return [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    def run(confirm=False,**overrides):
        env = {**environment,**(selected if confirm else {}),**overrides}
        result = subprocess.run(["bash",str(checkout/SCRIPT)],env={k:v for k,v in env.items() if v is not None},
            capture_output=True,text=True,timeout=30,check=False)
        assert not any(row["command"] in ("sbatch","srun","scancel","scontrol") for row in read_events())
        return result
    def receipt(): return dict(line.split("\t") for line in (output/"launch-receipt.tsv").read_text().splitlines()[1:])
    def confirms(): return [row["args"] for row in read_events() if row["command"]=="python" and "--selection" in row["args"]]
    return SimpleNamespace(run=run,receipt=receipt,events=read_events,confirms=confirms,output=output,checkout=checkout,selection=selection)


def test_static_one_gpu12h_manual_only_and_separate_phase_caps():
    text = (ROOT/SCRIPT).read_text()
    subprocess.run(["bash","-n",str(ROOT/SCRIPT)],check=True)
    for field in ("--gres=gpu:l40s:1","--cpus-per-task=8","--mem=64G","--time=12:00:00",CHECKOUT,
            "3600-(SECONDS-VPFP_STARTED_SECONDS)","VPFP_CONFIRM_SECONDS=39600","HEAD^{commit}","git ls-files --error-unmatch"):
        assert field in text
    assert not re.search(r"^\s*(sbatch|srun|scancel|scontrol)\b",text,re.M)
    assert 'mkdir -- "${VPFP_OUTPUT}"' in text and "launch-receipt.tsv.partial" in text
    assert "preflight_only_not_confirmation" in text and "--wall-seconds" in text


def test_missing_selection_is_preflight_only_and_runs_all_new_test_families(launch):
    result = launch.run()
    assert result.returncode==0,result.stderr
    assert not launch.confirms() and "PREFLIGHT ONLY completed" in result.stdout
    calls = [row["args"] for row in launch.events() if row["command"]=="python" and row["args"][:2]==["-m","pytest"]]
    assert len(calls)==1 and all("tests/"+name in calls[0] for name in TESTS)
    assert all(name in calls[0] for name in MOVING)
    receipt = launch.receipt()
    assert receipt["mode"]=="preflight_only_not_confirmation" and receipt["confirmation_handoff"]=="not_requested"
    assert receipt["scientific_admission"]==receipt["refined_reference_confirmation"]=="false"
    assert not (launch.output/"confirmation").exists()


def test_explicit_selection_after_gate_gets_at_most11h_and_preserves_exact_sha(launch):
    result = launch.run(confirm=True)
    assert result.returncode==0,result.stderr
    command, = launch.confirms()
    assert command[command.index("--selection")+1]==str(launch.selection)
    assert command[command.index("--selection-sha256")+1]==sha256(launch.selection.read_bytes()).hexdigest()
    assert command[command.index("--seed")+1]=="20260908"
    assert 30<float(command[command.index("--wall-seconds")+1])<=39600
    assert command[command.index("--output")+1]==str(launch.output/"confirmation")
    python_calls = [row["args"] for row in launch.events() if row["command"]=="python"]
    assert next(i for i,row in enumerate(python_calls) if row[:2]==["-m","pytest"])<python_calls.index(command)
    assert launch.receipt()["confirmation_handoff"]=="0"


@pytest.mark.parametrize("changes",[{"VPFP_CONFIRMATION_SELECTION_SHA256":None},
    {"VPFP_CONFIRMATION_SELECTION":None},{"VPFP_CONFIRMATION_SEED":None},{"VPFP_CONFIRMATION_SEED":"-1"},
    {"VPFP_CONFIRMATION_SELECTION_SHA256":"bad"}])
def test_half_binding_or_missing_seed_cannot_fall_back_to_preflight(launch,changes):
    assert launch.run(confirm=True,**changes).returncode==2
    assert not launch.events() and not launch.output.exists()


@pytest.mark.parametrize("changes,code",[({"MOCK_GPU_EXIT":"17"},17),({"MOCK_HELP_EXIT":"19"},19),
    ({"MOCK_TEST_EXIT":"23"},23),({"MOCK_TEST_EXIT":"124"},124),({"MOCK_HASH_EXIT":"13"},13),
    ({"MOCK_DIRTY":" M src/changed.py"},2)])
def test_gate_failures_preserve_exact_exit_and_never_start_confirmation(launch,changes,code):
    assert launch.run(confirm=True,**changes).returncode==code
    assert launch.receipt()["launch"]==str(code) and not launch.confirms()


def test_missing_control_test_is_not_silently_skipped(launch):
    (launch.checkout/"tests/test_vpfp_physical_controls.py").unlink()
    assert launch.run(confirm=True).returncode==2
    assert not launch.confirms() and launch.receipt()["mechanics_tests"]=="not_run"


def test_confirmation_timeout_retains_partial_artifact_and_receipt(launch):
    assert launch.run(confirm=True,MOCK_CONFIRM_EXIT="124").returncode==124
    assert (launch.output/"confirmation/partial-model-generation.json").exists()
    assert launch.receipt()["confirmation_handoff"]=="124"


def test_existing_output_is_never_reused_or_overwritten(launch):
    launch.output.mkdir()
    keep = launch.output/"retained.txt"; keep.write_text("keep")
    assert launch.run().returncode==2 and keep.read_text()=="keep" and not launch.events()


def test_no_allocation_cannot_create_or_run_anything(launch):
    assert launch.run(SLURM_JOB_ID=None).returncode==2
    assert not launch.output.exists() and not launch.events()
