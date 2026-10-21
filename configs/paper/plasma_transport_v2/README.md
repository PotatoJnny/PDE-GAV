# Vlasov–Poisson and VPFP experiments: transport v2

This release lets a researcher inspect the completed six-round experiments,
retrieve an Agent-generated solver, and trace it to its actual selection and
evaluation records. It covers ARTEMIS 1D1V Vlasov–Poisson, strong-collision
2D2V VPFP, and weak-collision 2D2V VPFP, each with Sol and Astra.

The layout follows the repository's NS experiment convention: a versioned
directory under `configs/paper/`, recorded JSON configurations, and solver
directories with `candidate_bundle.json`, `submission.py`, and every declared
supporting file. Publication date: **2026-10-21**. Campaign archive: 2026-10-19;
independent validation includes the completed 2026-10-20/21 follow-ups.

## Start here

| Content | Entry point |
| --- | --- |
| Equations, public inputs, budgets, selection and scoring | [PROTOCOL.md](PROTOCOL.md) |
| Effective protocol in JSON | [protocol.json](protocol.json) |
| Six original resolved configurations | [recorded_configs/](recorded_configs/) |
| Original Agent instructions, shared by the two model arms | [prompts/](prompts/) |
| All 77 program bundles | [solutions/INDEX.csv](solutions/INDEX.csv), [JSON index](solutions/INDEX.json) |
| All 36 retained round selections | [results/selection.csv](results/selection.csv) |
| Original terminal results, including failure and eligibility fields | [results/native_round_scores.csv](results/native_round_scores.csv), [per-case scores](results/native_case_scores.csv) |
| Independent validation, original failures and supplements | [results/README.md](results/README.md) |
| Source locations and checksums | [provenance.json](provenance.json), [release_manifest.json](release_manifest.json) |
| Large data, weights and saved numerical predictions | [data/README.md](data/README.md) |

## Final programs

The deliverable is the program retained after round 6 (S6). Terminal and
independent errors do not choose a different winner retrospectively.

| Equation | Model / effort | Final bundle |
| --- | --- | --- |
| ARTEMIS VP | `gpt-5.6-sol` / `max` | [candidate-r6-0011](solutions/artemis-sol/candidate-r6-0011/) |
| ARTEMIS VP | `gpt-6-astra` / `ultra` | [candidate-r6-0012](solutions/artemis-astra/candidate-r6-0012/) |
| Strong VPFP | `gpt-5.6-sol` / `max` | [candidate-r6-0011](solutions/strong-sol/candidate-r6-0011/) |
| Strong VPFP | `gpt-6-astra` / `ultra` | [candidate-r6-0012](solutions/strong-astra/candidate-r6-0012/) |
| Weak VPFP | `gpt-5.6-sol` / `max` | [candidate-r6-0012](solutions/weak-sol/candidate-r6-0012/) |
| Weak VPFP | `gpt-6-astra` / `ultra` | [candidate-r6-0012](solutions/weak-astra/candidate-r6-0012/) |

The 77 bundles comprise six arm-specific S0 entries and 71 submitted Agent
programs, including candidates that were not selected. Their 443 Python files
are unchanged from the archive. There are 35 distinct selected arm/program
pairs across 36 rounds: Strong-Astra R4 retains its R3 program. Keep an entire
bundle when copying a solver; `submission.py` alone is often insufficient.

## Verify and load a bundle

From the repository root, with the package's NumPy dependency installed:

```bash
python configs/paper/plasma_transport_v2/verify.py
```

This checks the release files, computes each bundle's SDK tree identity,
parses the Python sources without executing them, checks calibration and
prompt identities, and reconciles selection and result tables. It does not
load model weights, run solvers, or submit cluster jobs.

Use the same candidate loader as NS:

```python
import json
from pathlib import Path
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateProgram

release = Path("configs/paper/plasma_transport_v2")
index = json.loads((release / "solutions/INDEX.json").read_text())
record = next(p for p in index["programs"]
              if p["arm"] == "artemis-sol" and p["is_final_program"])
program = PythonCandidateProgram(
    release / record["bundle"], expected_tree_sha256=record["tree_sha256"]
)
```

Execution requires a compatible public Case, scoped Oracle session, and the
scientific dependencies used by that bundle (NumPy, SciPy, and, for some
programs, PyTorch/CUDA). The program prepares a solution before receiving
scoring coordinates. It is not a standalone script that recreates its own
benchmark inputs.

The resolved configurations are exact historical records with original asset
locators and larger sealed-bank counts. **They are not portable launch
templates for current `main`.** The executed subset rules and two historical
first-round exceptions are documented in the protocol. Full replay uses the
matching frozen framework and external assets; the release identifies both.
No new Agent or GPU experiment was run to publish this package.

Publication checks on 2026-10-21 passed for all 77 bundle identities, 443 Python
sources, 36 selections, 2,240 native terminal records, and the independent
result tables. The existing candidate-bundle and transport-scoring test suites
also passed (41 tests). Original line endings and whitespace are preserved by
the package's Git attributes so that checkout does not change archived hashes.
