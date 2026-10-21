"""NS callbacks around the unmodified official ERA FUTS search implementation."""
from hashlib import sha256
import math
from pathlib import Path

from ._vendor import era_futs
from .search import _write_json

ERA_REVISION = "eb56676ce0a9b22b0c357c218de09ac61e94fc4b"


def run_official_futs(config, *, initial_key, propose, score, history_for):
    """Keep official node selection/ranking/backpropagation; adapt only callbacks.

    Solution.program carries a frozen bundle digest instead of inline Python.
    Generation uses the project's CommandAgentProgram and developer feedback;
    execution is negative fixed-validation error. These are declared changes
    from the paper's model, prompt and empirical benchmark.
    """
    source_sha256 = sha256(Path(era_futs.__file__).read_bytes()).hexdigest()
    if source_sha256 != "5cc827f3d609105c1bd7b1103e4c33e393043c6db882bd1f44b9e4ec4301b814":
        raise RuntimeError("vendored official ERA FUTS differs from the pinned source")
    root = Path(config.output_directory).expanduser().resolve()
    _write_json(root / "era_adaptation.json", {
        "schema": "era_futs_ns_adaptation_v1",
        "upstream_repository": "https://github.com/google-research/era",
        "upstream_revision": ERA_REVISION,
        "upstream_file": "implementation/futs.py",
        "local_source_sha256": sha256(Path(era_futs.__file__).read_bytes()).hexdigest(),
        "search_modified": False,
        "solution_representation": "frozen_python_bundle_tree_sha256",
        "generation": "same_project_agent_with_fixed_development_evaluator",
        "execution_score": "negative_fixed_hidden_validation_mean_primary_error",
        "paper_end_to_end_reproduction": False,
    })
    remaining = config.iterations
    if initial_key is None:
        initial_key = propose(None, (), "era_initial")
        remaining -= 1
    # A failed node is a real spent generation, with worst reward; FUTS may
    # select it again and the adapter then generates from a cold start.
    missing = "__failed_baseline_proposal__"
    initial = era_futs.Solution(program=initial_key or missing)
    failed_counter = 0

    def generate(problem, parent_solution, parent_score):
        nonlocal failed_counter
        del problem, parent_score
        parent = parent_solution.program
        if parent.startswith(missing):
            parent = None
        key = propose(parent, history_for(parent), "era_expand")
        if key is None:
            failed_counter += 1
            key = f"{missing}{failed_counter}"
        return era_futs.Solution(program=key)

    def execute(problem, solution):
        del problem
        return -math.inf if solution.program.startswith(missing) else -score(solution.program)

    solution, reward = era_futs.search(
        era_futs.Problem(description=config.prompt),
        initial, execute(None, initial), generate, execute,
        num_iterations=remaining, c_puct=config.era_c_puct,
    )
    return None if not math.isfinite(reward) else solution.program
